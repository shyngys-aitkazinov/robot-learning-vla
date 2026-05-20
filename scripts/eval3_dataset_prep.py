"""Dataset preprocessing wrappers for Eval 3 training.

This module provides two preprocessing layers applied to each per-celebrity
LeRobotDataset BEFORE it gets concatenated and fed to lerobot-train:

1. ``Eval3PrepDataset`` — a thin proxy around a LeRobotDataset that:
   * Truncates each episode to its first ``max_frames_per_episode`` frames so
     the model only ever sees 20-s-fittable trajectories (Eval 3 has a 20-s
     wall-clock budget; 17/18 Obama episodes were recorded longer than that).
   * Optionally rewrites ``row["task"]`` on the fly using ``task_aug_fn`` so the
     SmolVLA tokenizer sees varied prompt strings (the canonical demo prompts
     drop the ``"the"`` prefix that the recordings always include).
   * Optionally repairs dataset_v2 gripper-open labels before training. The v2
     truncated datasets have open-gripper q90/q99 values below the deployment
     target; lifting only already-open commands preserves close/grasp labels
     while teaching the policy a wider pre-grasp/release aperture.
   * Optionally applies a small centered moving average to arm-action labels
     (not the gripper by default) to reduce learned high-frequency shake.

2. ``make_task_augmenter`` — returns a random task-string mutator. Variations
   are weighted toward the canonical demo wording (drop ``"the"``) since that
   is the dominant wording-drift problem identified by the v2 audit.

Why a wrapper instead of a custom processor step:
   * SmolVLA's preprocessor pipeline starts with ``RenameObservationsProcessorStep``
     followed by ``AddBatchDimension`` and only THEN the tokenizer. To inject a
     task-string change we'd need a custom processor inserted at the right index
     — fragile to lerobot upstream changes. Mutating ``row["task"]`` at
     ``__getitem__`` time is the simplest, most stable injection point: the
     tokenizer reads ``complementary_data["task"]`` (see
     ``.venv/.../processor/tokenizer_processor.py:132``).

The wrapper exposes the trainer-required attributes (``meta``, ``num_frames``,
``num_episodes``, ``episodes``, ``features``, ``repo_id``) and proxies everything
else to the underlying dataset via ``__getattr__``, mirroring the pattern in
``scripts/eval3_concat_patch.py:ConcatLeRobotDataset``.
"""
from __future__ import annotations

import glob
import hashlib
import logging
import os
import pickle
import random
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch.utils.data import Dataset


# --- Eval3PrepDataset.__init__ cache (opt-in) -----------------------------
#
# Computing the per-dataset prep state (valid_indices, truncation log, action
# overrides, recomputed action/state stats) iterates 100k+ frames from parquet
# and takes ~80 s per dataset on a v3 synth corpus. With 9 datasets that's
# ~12 min of stats-merge on every train launch.
#
# The result is fully determined by (repo_id, dataset content, aug config), so
# we OPTIONALLY cache it to <cache_dir>/<key>.pkl. Subsequent launches with
# the same config skip the slow path entirely (~15 s for all 9 datasets).
#
# **Disabled by default** — no log lines, no filesystem writes, behavior is
# identical to the original implementation unless the user opts in.
#
# Env-var contract:
#   EVAL3_PREP_CACHE       — "0" (default) disables both reads and writes.
#                            Set to "1"/"true"/"yes"/"on" (case-insensitive)
#                            to enable the cache.
#   EVAL3_PREP_CACHE_DIR   — override the cache directory. Default
#                            ~/.cache/eval3_concat_patch.
#
# Invalidate by removing the cache dir or bumping _PREP_CACHE_VERSION below.
_PREP_CACHE_DEFAULT_DIR = Path.home() / ".cache" / "eval3_concat_patch"
_PREP_CACHE_VERSION = 1


def _prep_cache_enabled() -> bool:
    raw = os.environ.get("EVAL3_PREP_CACHE", "0").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _prep_cache_dir() -> Path:
    override = os.environ.get("EVAL3_PREP_CACHE_DIR", "").strip()
    return Path(override).expanduser() if override else _PREP_CACHE_DEFAULT_DIR


def _prep_cache_key(
    repo_id: str,
    total_frames: int,
    total_episodes: int,
    max_frames_per_episode,
    episode_filter,
    truncate_at_placement: bool,
    truncate_placement_mode: str,
    truncate_allow_over_cap_episodes,
    truncate_over_cap_extra_frames: int,
    truncate_grip_threshold: float,
    truncate_lift_threshold: float,
    truncate_buffer_frames: int,
    repair_gripper_open: bool,
    gripper_open_target: float,
    gripper_open_threshold: float,
    action_smooth_window: int,
    action_smooth_gripper: bool,
) -> str:
    over_cap = tuple(sorted(int(e) for e in truncate_allow_over_cap_episodes)) if truncate_allow_over_cap_episodes else ()
    ep_filter_key = tuple(sorted(int(e) for e in episode_filter)) if episode_filter else None
    payload = (
        _PREP_CACHE_VERSION,
        repo_id,
        int(total_frames),
        int(total_episodes),
        int(max_frames_per_episode) if max_frames_per_episode is not None else None,
        ep_filter_key,
        bool(truncate_at_placement),
        str(truncate_placement_mode),
        over_cap,
        int(truncate_over_cap_extra_frames),
        float(truncate_grip_threshold),
        float(truncate_lift_threshold),
        int(truncate_buffer_frames),
        bool(repair_gripper_open),
        float(gripper_open_target),
        float(gripper_open_threshold),
        int(action_smooth_window or 0),
        bool(action_smooth_gripper),
    )
    return hashlib.sha256(repr(payload).encode()).hexdigest()[:16]


def _load_prep_cache(key: str):
    if not _prep_cache_enabled():
        return None
    path = _prep_cache_dir() / f"{key}.pkl"
    if not path.exists():
        return None
    try:
        with path.open("rb") as f:
            return pickle.load(f)
    except Exception as e:
        logging.warning("eval3_prep_cache: failed to load %s: %s", path, e)
        return None


def _save_prep_cache(key: str, state: dict) -> None:
    if not _prep_cache_enabled():
        return
    cache_dir = _prep_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{key}.pkl"
    try:
        with path.open("wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        logging.warning("eval3_prep_cache: failed to save %s: %s", path, e)


# Substring-match list. If the recorded ``task`` string contains one of these names,
# we know which celebrity to augment around. Any task string without a known
# celebrity name is passed through unchanged (defensive).
KNOWN_CELEBRITIES = ("Taylor Swift", "Yann LeCun", "Barack Obama")
SLOT_NAMES = ("left", "middle", "right")

# Canonical HOME pose averaged across the first 5 frames of every episode in
# dataset_v3_charuco_{left,middle,right}_2 (30 episodes × 5 frames = 150 samples).
# Values are in degrees. Joint order: shoulder_pan, shoulder_lift, elbow_flex,
# wrist_flex, wrist_roll, gripper. See `tools/eval3_compute_canonical_home.py`
# for the derivation (script committed alongside this constant).
#
# Why a GLOBAL average and not per-position: the per-position wrist_roll-at-HOME
# carries directional info (left: +9.3°, middle: +0.2°, right: +12.3°) that the
# model could exploit to predict trajectory direction from state alone. The
# global average erases that cue so HOME-masked frames look truly identical
# (modulo jitter) regardless of the eventual target.
CANONICAL_HOME_STATE = (1.3574, -102.8120, 96.3487, -99.8464, 7.2586, 0.6771)


class _CurriculumStepCounter:
    """Shared training-step counter for noise curriculum.

    Workers spawned by DataLoader read the counter via the shared
    multiprocessing.Value created at module import time. The train loop is
    expected to call ``set_step(n)`` once per step (see the patch wiring in
    ``eval3_smolvla_aux_head.apply()``).
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            import multiprocessing

            cls._instance = super().__new__(cls)
            # 'i' = signed int; initial value 0; lock=True for safe writes
            cls._instance._step = multiprocessing.Value("i", 0)
        return cls._instance

    def set_step(self, step: int) -> None:
        self._step.value = int(step)

    def get_step(self) -> int:
        return int(self._step.value)


# Module-level singleton — instantiated lazily at first import so the
# multiprocessing.Value is created BEFORE any DataLoader workers fork from it.
def get_curriculum_step_counter() -> "_CurriculumStepCounter":
    return _CurriculumStepCounter()


class StateAugmenter:
    """Picklable state-augmentation callable for breaking the state shortcut.

    Three orthogonal pressures, each opt-in via env vars / kwargs:

    1. **Gaussian noise** with optional cosine-decay curriculum on ``sigma``.
       At training progress ``p`` (= step / curriculum_steps clamped to [0, 1]),
       sigma transitions from ``sigma_max`` (high) to ``sigma_min`` (low):

           sigma(p) = sigma_min + 0.5 * (sigma_max - sigma_min) * (1 + cos(pi * p))

       So at p=0: sigma = sigma_max. At p=1: sigma = sigma_min. Encourages the
       policy to lean on visual + language features early when state is
       unreliable, while still letting it learn from clean state late.

       **Sigma is in NORMALIZED STDDEV UNITS, not raw degrees.** That is,
       sigma=0.3 means "perturb each joint by 0.3 of its per-joint training
       stddev". This gives uniform perturbation magnitude across joints in
       the space the model actually sees (after the lerobot normalizer's
       (x - mean) / std step). With per-joint raw-degree noise, joints with
       small stddev (shoulder_pan) get 3.7× more noise than joints with
       large stddev (shoulder_lift) — not what we want.

       If ``state_std`` is not passed (e.g., constructed before stats are
       available), sigma is interpreted as raw degrees as a fallback — but
       the augmenter logs a warning since this gives uneven per-joint effect.

    2. **State replacement** with probability ``replace_prob`` per frame. Of
       the replaced frames, ``mode_weights`` decides between two modes:
       - ``home``: replace with ``CANONICAL_HOME_STATE`` + jitter (raw degrees)
       - ``zero``: replace with all zeros (raw degrees)
       The HOME mode breaks the shortcut by making the same start-state map
       to multiple actions (only language distinguishes). The zero mode is
       more aggressive — forces image + language to predict from no state.
       Note: after the lerobot normalizer, HOME normalizes to ~3.91 stddev
       away from training mean (trajectory-centered stats put HOME in the
       OOD tail — this is intended behavior because HOME IS the deploy
       starting condition).

       The HOME-jitter (1° default) and replacement are applied in RAW
       degree space. The Gaussian noise (which IS in normalized units) is
       applied AFTER replacement, in raw-degree space scaled per-joint by
       std.

    3. **Gripper protection**: noise on the gripper is reduced by
       ``gripper_noise_scale`` (default 0.1) because gripper is near-binary
       and large noise destroys grasp/release labels.

    The augmenter operates on a 6-dim float tensor (the ``observation.state``
    AFTER it's been pulled from ``row["observation.state"]``). It returns a
    new tensor with the same shape and dtype.
    """

    def __init__(
        self,
        sigma_max: float = 0.0,
        sigma_min: float = 0.0,
        curriculum_steps: int = 0,
        replace_prob: float = 0.0,
        mode_home_weight: float = 0.7,
        mode_zero_weight: float = 0.3,
        home_jitter_sigma: float = 1.0,
        gripper_noise_scale: float = 0.1,
        state_std: tuple | list | None = None,
        seed: int = 1337,
        postgrasp_grip_thresh: float | None = None,
        postgrasp_sigma_mult: float = 1.0,
        postgrasp_replace_prob: float | None = None,
    ):
        self._sigma_max = float(sigma_max)
        self._sigma_min = float(sigma_min)
        self._curriculum_steps = int(curriculum_steps)
        self._replace_prob = float(replace_prob)
        self._mode_home_weight = float(mode_home_weight)
        self._mode_zero_weight = float(mode_zero_weight)
        self._home_jitter_sigma = float(home_jitter_sigma)
        self._gripper_noise_scale = float(gripper_noise_scale)
        # Post-grasp regime: when the raw gripper aperture is below this
        # threshold the arm is at-rest or carrying the can — the phase where
        # the target direction is chosen. Amplify state noise there so the
        # policy cannot ride proprio-continuation and must read the slot token.
        # None = feature disabled.
        self._postgrasp_grip_thresh = (
            float(postgrasp_grip_thresh) if postgrasp_grip_thresh is not None else None
        )
        self._postgrasp_sigma_mult = float(postgrasp_sigma_mult)
        self._postgrasp_replace_prob = (
            float(postgrasp_replace_prob) if postgrasp_replace_prob is not None else None
        )
        # state_std (6,) — per-joint training-set stddev. sigma is interpreted
        # in normalized stddev units; raw-degree noise = sigma * state_std.
        # None → fallback to raw-degree interpretation (uneven per joint).
        self._state_std: tuple | None = tuple(state_std) if state_std is not None else None
        self._seed = int(seed)
        self._rng = random.Random(seed)
        # mp.Value singleton (created at module import, shared across workers).
        self._step_counter = get_curriculum_step_counter()
        # Pre-compute the HOME / state_std tensors once (lazy — don't know device yet).
        self._home_tensor: torch.Tensor | None = None
        self._std_tensor: torch.Tensor | None = None
        if self._state_std is None and self._sigma_max > 0.0:
            import logging
            logging.warning(
                "[StateAugmenter] state_std not provided; sigma=%.3f will be "
                "interpreted as RAW DEGREES, giving uneven per-joint noise after "
                "normalization (shoulder_pan std≈14° gets 3.7× more effective "
                "perturbation than shoulder_lift std≈50°). Pass state_std=dataset.meta.stats[\"observation.state\"][\"std\"] for uniform perturbation.",
                self._sigma_max,
            )

    def _current_sigma(self) -> float:
        """Cosine-decay sigma. Constant sigma_max if curriculum is disabled."""
        if self._curriculum_steps <= 0:
            return self._sigma_max
        step = self._step_counter.get_step()
        # Clamp progress to [0, 1].
        progress = max(0.0, min(1.0, step / self._curriculum_steps))
        import math
        return self._sigma_min + 0.5 * (self._sigma_max - self._sigma_min) * (
            1.0 + math.cos(math.pi * progress)
        )

    def _get_home(self, ref: torch.Tensor) -> torch.Tensor:
        """Return the HOME tensor on the same device/dtype as ``ref``."""
        if self._home_tensor is None or self._home_tensor.device != ref.device or self._home_tensor.dtype != ref.dtype:
            self._home_tensor = torch.tensor(
                CANONICAL_HOME_STATE, dtype=ref.dtype, device=ref.device
            )
        return self._home_tensor

    def _get_std(self, ref: torch.Tensor) -> torch.Tensor | None:
        """Return per-joint stddev tensor on ref's device/dtype, or None for raw mode."""
        if self._state_std is None:
            return None
        if self._std_tensor is None or self._std_tensor.device != ref.device or self._std_tensor.dtype != ref.dtype:
            self._std_tensor = torch.tensor(
                self._state_std, dtype=ref.dtype, device=ref.device
            )
        return self._std_tensor

    def __call__(self, state: torch.Tensor) -> torch.Tensor:
        """Augment a single-sample state vector (shape: (6,) — float tensor)."""
        # Post-grasp regime check — read the ORIGINAL gripper (last joint)
        # before any replacement/noise. Below the threshold => at-rest or
        # carrying the can => amplify replacement probability and noise sigma.
        replace_prob = self._replace_prob
        sigma_mult = 1.0
        if self._postgrasp_grip_thresh is not None:
            try:
                grip = float(state[-1])
            except Exception:
                grip = None
            if grip is not None and grip < self._postgrasp_grip_thresh:
                if self._postgrasp_replace_prob is not None:
                    replace_prob = self._postgrasp_replace_prob
                sigma_mult = self._postgrasp_sigma_mult
        # First decide on replacement (B + D unified). The two modes are
        # mutually exclusive per call; weights pick which one runs (if any).
        if replace_prob > 0.0 and self._rng.random() < replace_prob:
            total = self._mode_home_weight + self._mode_zero_weight
            if total <= 0:
                mode_roll = -1.0  # both weights zero → skip replacement
            else:
                mode_roll = self._rng.random() * total
            if mode_roll < self._mode_home_weight:
                # HOME mode: canonical home + jitter (raw degrees).
                home = self._get_home(state)
                jitter = torch.randn_like(state) * self._home_jitter_sigma
                jitter[-1] = jitter[-1] * self._gripper_noise_scale
                state = home + jitter
            elif mode_roll < total:
                # Zero mode (raw degrees).
                state = torch.zeros_like(state)

        # Gaussian noise (A) with cosine-decay sigma. sigma is in NORMALIZED
        # stddev units, so raw-degree noise per joint = sigma * state_std[j].
        # This gives uniform perturbation across joints in the space the
        # model sees (after lerobot normalizer's (x - mean) / std).
        sigma = self._current_sigma() * sigma_mult
        if sigma > 0.0:
            std_t = self._get_std(state)
            if std_t is not None:
                # Normalized-units mode: scale per-joint by std.
                noise = torch.randn_like(state) * sigma * std_t
            else:
                # Fallback: raw degrees (warned at __init__).
                noise = torch.randn_like(state) * sigma
            noise[-1] = noise[-1] * self._gripper_noise_scale
            state = state + noise

        return state

    def __reduce__(self):
        # Reconstructable in worker processes (DataLoader forks). state_std is
        # included so forked workers reconstruct an augmenter with the same
        # per-joint scaling.
        return (
            self.__class__,
            (
                self._sigma_max,
                self._sigma_min,
                self._curriculum_steps,
                self._replace_prob,
                self._mode_home_weight,
                self._mode_zero_weight,
                self._home_jitter_sigma,
                self._gripper_noise_scale,
                self._state_std,
                self._seed,
                self._postgrasp_grip_thresh,
                self._postgrasp_sigma_mult,
                self._postgrasp_replace_prob,
            ),
        )


def make_state_augmenter(state_std: tuple | list | None = None) -> "StateAugmenter | None":
    """Build a StateAugmenter from env vars. Returns None if no augmentation is enabled.

    Parameters
    ----------
    state_std : tuple | list | None
        Per-joint training-set stddev (6,) for ``observation.state``. Pulled from
        ``dataset.meta.stats["observation.state"]["std"]`` by the caller. If passed,
        ``EVAL3_STATE_NOISE_SIGMA_MAX`` and ``_MIN`` are interpreted as NORMALIZED
        STDDEV UNITS (raw-degree noise per joint = sigma * state_std[j]).
        If None, sigma is interpreted as raw degrees with a warning.
    """
    def _ef(name: str, default: float) -> float:
        raw = os.environ.get(name, "").strip()
        if not raw:
            return default
        try:
            return float(raw)
        except ValueError:
            return default

    def _ei(name: str, default: int) -> int:
        raw = os.environ.get(name, "").strip()
        if not raw:
            return default
        try:
            return int(raw)
        except ValueError:
            return default

    # Default sigma_max = 0.3 (= 30% of one normalized stddev per joint) when
    # state_std is available — pre-tuned for the v6_synth corpus. Raw-degree
    # equivalent at a typical std=30° would be 9° of effective noise.
    sigma_max = _ef("EVAL3_STATE_NOISE_SIGMA_MAX", 0.0)
    sigma_min = _ef("EVAL3_STATE_NOISE_SIGMA_MIN", 0.0)
    curriculum_steps = _ei("EVAL3_STATE_NOISE_CURRICULUM_STEPS", 0)
    replace_prob = _ef("EVAL3_STATE_REPLACE_PROB", 0.0)
    home_jitter_sigma = _ef("EVAL3_STATE_HOME_JITTER_SIGMA", 1.0)
    gripper_noise_scale = _ef("EVAL3_STATE_GRIPPER_NOISE_SCALE", 0.1)

    # Parse mode weights from EVAL3_STATE_REPLACE_MODES="home:0.7,zero:0.3"
    modes_str = os.environ.get("EVAL3_STATE_REPLACE_MODES", "home:0.7,zero:0.3").strip()
    home_w, zero_w = 0.7, 0.3
    try:
        parts = [p.strip() for p in modes_str.split(",") if p.strip()]
        for p in parts:
            k, v = p.split(":")
            k = k.strip().lower()
            v = float(v.strip())
            if k == "home":
                home_w = v
            elif k == "zero":
                zero_w = v
    except Exception:
        pass  # fall back to defaults on parse failure

    # Post-grasp heavy-noise knobs (see StateAugmenter docstring).
    pg_thresh_raw = os.environ.get("EVAL3_STATE_POSTGRASP_GRIP_THRESH", "").strip()
    postgrasp_grip_thresh: float | None = None
    if pg_thresh_raw:
        try:
            postgrasp_grip_thresh = float(pg_thresh_raw)
        except ValueError:
            postgrasp_grip_thresh = None
    postgrasp_sigma_mult = _ef("EVAL3_STATE_POSTGRASP_SIGMA_MULT", 1.0)
    pg_replace_raw = os.environ.get("EVAL3_STATE_POSTGRASP_REPLACE_PROB", "").strip()
    postgrasp_replace_prob: float | None = None
    if pg_replace_raw:
        try:
            postgrasp_replace_prob = float(pg_replace_raw)
        except ValueError:
            postgrasp_replace_prob = None

    # If nothing is enabled, return None — the caller will skip wiring.
    if sigma_max <= 0.0 and replace_prob <= 0.0:
        return None
    return StateAugmenter(
        sigma_max=sigma_max,
        sigma_min=sigma_min,
        curriculum_steps=curriculum_steps,
        replace_prob=replace_prob,
        mode_home_weight=home_w,
        mode_zero_weight=zero_w,
        home_jitter_sigma=home_jitter_sigma,
        gripper_noise_scale=gripper_noise_scale,
        state_std=state_std,
        postgrasp_grip_thresh=postgrasp_grip_thresh,
        postgrasp_sigma_mult=postgrasp_sigma_mult,
        postgrasp_replace_prob=postgrasp_replace_prob,
    )


# Task-string wording templates. Each is a `.format(celeb=...)` template.
# The model should see all of these during training so it doesn't overfit to
# one specific phrasing (which would fail at deploy if the demo prompt varies).
# The first two are the canonical/recorded demo prompts — kept at higher
# baseline weight via EVAL3_TASK_AUG_CANONICAL_P.
#
# The varied pool emphasizes "image of" / "photo of" / "picture of" / "print"
# phrasings — these match what the robot literally sees (a printed photo
# of a celebrity, not the actual person) and help the text encoder learn
# image-grounded language.
TASK_TEMPLATES = (
    # Canonical demo prompts (indices 0..1) — used when the canonical roll wins.
    "Place the coke on {celeb}",                       # canonical demo prompt
    "Place the coke on the {celeb}",                   # original recorded wording
    # Varied pool (indices 2..) — uniformly sampled when the varied roll wins.
    "Put the coke on {celeb}",                         # synonym verb
    "Put the coke on the {celeb}",
    "Place the coke can on {celeb}",                   # "coke can" instead of "coke"
    "Drop the coke on {celeb}",                        # different verb
    "Move the coke to {celeb}",                        # different verb structure
    # Image-grounded phrasings — what the robot literally sees:
    "Place the coke on the image of {celeb}",
    "Place the coke on the photo of {celeb}",
    "Place the coke on the picture of {celeb}",
    "Put the coke on the image of {celeb}",
    "Put the coke on the photo of {celeb}",
    "Put the coke on the picture of {celeb}",
    "Drop the coke on the image of {celeb}",
    "Drop the coke on the photo of {celeb}",
    "Place the can on the image of {celeb}",
    "Place the can on the photo of {celeb}",
    "Place the can on the {celeb} print",
    "Place the coke on {celeb}'s image",
    "Place the coke on {celeb}'s photo",
    "Move the coke to the image of {celeb}",
)


class TaskAugmenter:
    """Picklable callable that randomly rewrites Eval 3 task strings.

    Implemented as a class (not a closure) because PyTorch DataLoader with
    ``num_workers > 0`` pickles the dataset object — including any callable
    attributes — to ship to worker processes. Local closures from
    ``make_task_augmenter.<locals>.aug`` are not picklable.

    Two-tier wording strategy:

      1. With probability ``canonical_p`` (default 0.8) emit one of the
         CANONICAL demo prompts:
            * "Place the coke on <Celeb>"             (demo, weight 0.7)
            * "Place the coke on the <Celeb>"         (recorded, weight 0.3)
      2. With probability ``1 - canonical_p`` emit a uniformly-random
         non-canonical wording from the remaining ``TASK_TEMPLATES`` to
         build language robustness against demo-day prompt drift.

    The celebrity name is always preserved exactly so the language→celebrity
    mapping the aux head trains on stays intact. Unknown tasks are passed
    through unchanged (defensive).
    """

    def __init__(self, seed: int = 42, canonical_p: float = 0.8):
        self._seed = seed
        self._canonical_p = float(canonical_p)
        # Use a Python random.Random with the supplied seed. Each forked worker
        # will get its own copy after pickling — that's fine for augmentation.
        self._rng = random.Random(seed)
        # Pre-split canonical vs non-canonical pools (templates[0..1] vs [2..])
        self._canonical_templates = TASK_TEMPLATES[:2]
        self._varied_templates = TASK_TEMPLATES[2:]

    def __call__(self, task: str) -> str:
        if not isinstance(task, str):
            return task
        for celeb in KNOWN_CELEBRITIES:
            if celeb in task:
                # First decide canonical vs varied.
                if self._rng.random() < self._canonical_p:
                    # Canonical pool: 70/30 split between the two demo wordings
                    # (preserves the original behavior of this augmenter).
                    if self._rng.random() < 0.7:
                        return self._canonical_templates[0].format(celeb=celeb)
                    else:
                        return self._canonical_templates[1].format(celeb=celeb)
                else:
                    # Varied pool: uniform pick from non-canonical templates.
                    template = self._rng.choice(self._varied_templates)
                    return template.format(celeb=celeb)
        return task  # unknown celebrity — leave untouched

    def __reduce__(self):
        # When DataLoader forks a worker, this returns a constructor + args
        # so a fresh TaskAugmenter is built in the child process.
        return (self.__class__, (self._seed, self._canonical_p))


def make_task_augmenter(seed: int = 42, canonical_p: float | None = None) -> "TaskAugmenter":
    """Backward-compatible factory returning a TaskAugmenter instance."""
    if canonical_p is None:
        raw = os.environ.get("EVAL3_TASK_AUG_CANONICAL_P", "0.8").strip()
        try:
            canonical_p = float(raw)
        except ValueError:
            canonical_p = 0.8
    canonical_p = min(max(float(canonical_p), 0.0), 1.0)
    return TaskAugmenter(seed=seed, canonical_p=canonical_p)


class SlotTaskAugmenter:
    """Picklable task mutator for slot-conditioned policy runs.

    This is the policy side of the target-selector design: an external
    selector/classifier decides which physical print slot matches the celebrity
    prompt, then the motor policy receives a spatial instruction such as
    ``"Place the coke on the left print"``. ``slot_p`` lets a retrain mix slot
    prompts with the original celebrity prompts instead of committing fully.
    """

    def __init__(
        self,
        slot: str,
        base_aug: Callable[[str], str] | None = None,
        slot_p: float = 0.5,
        seed: int = 42,
    ):
        slot = str(slot).strip().lower()
        if slot not in SLOT_NAMES:
            raise ValueError(f"slot must be one of {SLOT_NAMES}, got {slot!r}")
        self._slot = slot
        self._base_aug = base_aug
        self._slot_p = min(max(float(slot_p), 0.0), 1.0)
        self._seed = int(seed)
        self._rng = random.Random(seed)

    def __call__(self, task: str) -> str:
        if self._rng.random() < self._slot_p:
            variants = (
                f"Place the coke on the {self._slot} print",
                f"Place the coke on the {self._slot} target",
                f"Place the coke on the {self._slot}",
            )
            return variants[self._rng.randrange(len(variants))]
        if self._base_aug is not None:
            return self._base_aug(task)
        return task

    def __reduce__(self):
        return (self.__class__, (self._slot, self._base_aug, self._slot_p, self._seed))


def make_slot_task_augmenter(
    slot: str,
    base_aug: Callable[[str], str] | None = None,
    slot_p: float = 0.5,
    seed: int = 42,
) -> "SlotTaskAugmenter":
    return SlotTaskAugmenter(slot=slot, base_aug=base_aug, slot_p=slot_p, seed=seed)


def _stat_tensor(
    values: torch.Tensor,
    source_stat: dict | None = None,
) -> dict[str, torch.Tensor]:
    """Compute LeRobot-style stats for a filtered tensor column.

    The filtered Eval 3 wrapper changes which frames are visible to training,
    so action/state normalization must be recomputed from the same frame set.
    """
    x = values.detach().float()
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim == 0:
        x = x.reshape(1, 1)

    out = {
        "min": x.amin(dim=0),
        "max": x.amax(dim=0),
        "mean": x.mean(dim=0),
        "std": x.std(dim=0, unbiased=False),
        "q01": x.quantile(0.01, dim=0),
        "q10": x.quantile(0.10, dim=0),
        "q50": x.quantile(0.50, dim=0),
        "q90": x.quantile(0.90, dim=0),
        "q99": x.quantile(0.99, dim=0),
    }

    if source_stat and isinstance(source_stat.get("count"), torch.Tensor):
        src_count = source_stat["count"]
        out["count"] = torch.full_like(src_count, float(x.shape[0]))
    else:
        out["count"] = torch.tensor(float(x.shape[0]), dtype=torch.float32)

    if source_stat:
        for key, src in source_stat.items():
            if key in out and isinstance(src, torch.Tensor):
                out[key] = out[key].to(dtype=src.dtype)
    return out


def _column_values_to_tensor(values: list[Any]) -> torch.Tensor:
    """Convert a HF Dataset column slice to a float tensor without decoding videos."""
    if values and any(isinstance(v, torch.Tensor) for v in values):
        return torch.stack([torch.as_tensor(v).detach().float() for v in values])
    return torch.as_tensor(values, dtype=torch.float32)


def _compute_filtered_stats(
    dataset,
    valid_indices: list[int],
    action_overrides: dict[int, torch.Tensor] | None = None,
) -> dict[str, dict[str, torch.Tensor]]:
    """Recompute non-visual metadata stats for the filtered frame subset."""
    hf = dataset.hf_dataset
    stats = deepcopy(dataset.meta.stats)
    column_names = set(getattr(hf, "column_names", []))

    # These are the fields that affect normalization or are saved in processor
    # state. Images are intentionally left alone; training uses ImageNet visual
    # stats when requested and VISUAL normalization is IDENTITY for SmolVLA.
    feature_keys = (
        "action",
        "observation.state",
        "episode_index",
        "frame_index",
        "index",
        "timestamp",
        "task_index",
    )
    for key in feature_keys:
        if key not in column_names or key not in stats:
            continue
        col = hf[key]
        if key == "action" and action_overrides:
            values = [action_overrides.get(int(i), col[i]) for i in valid_indices]
        else:
            values = [col[i] for i in valid_indices]
        if not values:
            continue
        stats[key] = _stat_tensor(_column_values_to_tensor(values), stats.get(key))
    return stats


def _feature_names(dataset, key: str) -> list[str]:
    feature = dataset.meta.features.get(key) or {}
    names = feature.get("names") if isinstance(feature, dict) else getattr(feature, "names", None)
    return list(names or [])


def _joint_index(dataset, key: str, joint_name: str, fallback: int) -> int:
    for idx, name in enumerate(_feature_names(dataset, key)):
        normalized = str(name).lower().removesuffix(".pos")
        if normalized == joint_name.lower():
            return idx
    return int(fallback)


def _value_to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy().astype(np.float32)
    return np.asarray(value, dtype=np.float32)


def _centered_moving_average(values: np.ndarray, window: int, dims: list[int]) -> np.ndarray:
    window = int(window)
    if window <= 1 or not dims or values.shape[0] <= 1:
        return values
    if window % 2 == 0:
        window += 1
    window = min(window, values.shape[0] if values.shape[0] % 2 == 1 else max(1, values.shape[0] - 1))
    if window <= 1:
        return values

    out = values.copy()
    left = window // 2
    right = window - 1 - left
    kernel = np.ones(window, dtype=np.float32) / float(window)
    for dim in dims:
        padded = np.pad(values[:, dim], (left, right), mode="edge")
        out[:, dim] = np.convolve(padded, kernel, mode="valid")
    return out


def _build_action_overrides(
    dataset,
    truncation_log: list[dict],
    *,
    repair_gripper_open: bool,
    gripper_open_target: float,
    gripper_open_threshold: float,
    action_smooth_window: int,
    action_smooth_gripper: bool,
) -> tuple[dict[int, torch.Tensor], dict[str, Any]]:
    """Precompute per-frame repaired action labels and a compact summary."""
    action_smooth_window = int(action_smooth_window or 0)
    if not repair_gripper_open and action_smooth_window <= 1:
        return {}, {
            "gripper_repair_enabled": False,
            "gripper_repair_changed_frames": 0,
            "action_smooth_window": 0,
            "action_smooth_changed_frames": 0,
            "action_override_frames": 0,
        }

    actions_col = dataset.hf_dataset["action"]
    sample = _value_to_numpy(actions_col[0]).reshape(-1)
    action_dim = int(sample.shape[0])
    gripper_idx = _joint_index(dataset, "action", "gripper", action_dim - 1)
    smooth_dims = list(range(action_dim))
    if not action_smooth_gripper and gripper_idx in smooth_dims:
        smooth_dims.remove(gripper_idx)

    overrides: dict[int, torch.Tensor] = {}
    repair_changed = 0
    smooth_changed = 0
    total_frames = 0

    for record in truncation_log:
        f0 = int(record["f0"])
        kept_to = int(record["kept_to"])
        if kept_to <= f0:
            continue
        indices = list(range(f0, kept_to))
        original = np.stack([_value_to_numpy(actions_col[i]).reshape(-1) for i in indices]).astype(np.float32)
        repaired = original.copy()
        total_frames += len(indices)

        if repair_gripper_open:
            before = repaired[:, gripper_idx].copy()
            open_mask = before >= float(gripper_open_threshold)
            repaired[open_mask, gripper_idx] = np.maximum(
                repaired[open_mask, gripper_idx],
                float(gripper_open_target),
            )
            repair_changed += int(np.count_nonzero(np.abs(repaired[:, gripper_idx] - before) > 1e-6))

        if action_smooth_window > 1 and smooth_dims:
            before = repaired.copy()
            repaired = _centered_moving_average(repaired, action_smooth_window, smooth_dims)
            smooth_changed += int(np.count_nonzero(np.linalg.norm(repaired - before, axis=1) > 1e-6))

        changed = np.linalg.norm(repaired - original, axis=1) > 1e-6
        for idx, action, did_change in zip(indices, repaired, changed):
            if did_change:
                overrides[int(idx)] = torch.as_tensor(action, dtype=torch.float32)

    return overrides, {
        "gripper_repair_enabled": bool(repair_gripper_open),
        "gripper_repair_open_target": float(gripper_open_target),
        "gripper_repair_open_threshold": float(gripper_open_threshold),
        "gripper_repair_changed_frames": int(repair_changed),
        "gripper_repair_changed_fraction": (float(repair_changed) / float(total_frames)) if total_frames else 0.0,
        "action_smooth_window": int(action_smooth_window if action_smooth_window > 1 else 0),
        "action_smooth_gripper": bool(action_smooth_gripper),
        "action_smooth_changed_frames": int(smooth_changed),
        "action_smooth_changed_fraction": (float(smooth_changed) / float(total_frames)) if total_frames else 0.0,
        "action_override_frames": int(len(overrides)),
        "action_override_fraction": (float(len(overrides)) / float(total_frames)) if total_frames else 0.0,
    }


# ----------------------------------------------------------------------------
# Image augmenters injected at Eval3PrepDataset.__getitem__ time.
#
# They run AFTER lerobot's image_transforms (those happen inside DatasetReader.
# get_item at .venv/.../lerobot/datasets/dataset_reader.py:~277). The input is
# the CHW float-in-[0,1] tensor that the trainer expects.
#
# Both classes are pickled to DataLoader workers (num_workers > 0 forks). We
# follow TaskAugmenter's pattern: store only str/int paths in instance state,
# expose __reduce__ that re-runs __init__ in each worker, lazy-load mask/bg
# bytes on first __call__.
# ----------------------------------------------------------------------------


def _load_mask(mask_path: str) -> np.ndarray:
    """Load a HxW boolean mask saved via np.save."""
    arr = np.load(mask_path)
    if arr.dtype != bool:
        arr = arr.astype(bool)
    return arr


def _bbox_of_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    """Tight bbox (x1, y1, x2, y2) of True pixels in a HxW boolean mask."""
    ys, xs = np.where(mask)
    if ys.size == 0:
        return 0, 0, 0, 0
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


class BackgroundReplaceAugmenter:
    """Replace background pixels with a random image from a pool.

    Behavior at __call__(img_chw):
      * With probability ``p``, load a random ``.png`` from ``bg_dir``, resize
        to the input H,W, then write its pixels into ``img_chw`` everywhere
        ``bg_mask`` is True. Pixels where ``bg_mask`` is False (= table top +
        prints + arm + can) are preserved.
      * With probability 1-p, return the input unchanged.

    Implemented for picklability: stores only string paths + seed in state.
    Mask + bg-path list are lazy-loaded on first __call__ in each worker so the
    pickled instance stays tiny.
    """

    def __init__(self, mask_path: str, bg_dir: str, p: float = 0.3, seed: int = 0):
        self._mask_path = str(mask_path)
        self._bg_dir = str(bg_dir)
        self._p = float(p)
        self._seed = int(seed)
        self._rng = random.Random(seed)
        # Lazy state (populated per-worker on first call)
        self._mask: np.ndarray | None = None
        self._bg_paths: list[str] | None = None

    def _ensure_loaded(self) -> None:
        if self._mask is None:
            self._mask = _load_mask(self._mask_path)
        if self._bg_paths is None:
            self._bg_paths = sorted(glob.glob(os.path.join(self._bg_dir, "*.png")))
            if not self._bg_paths:
                raise FileNotFoundError(f"no .png backgrounds under {self._bg_dir}")

    def __call__(self, img_chw: torch.Tensor) -> torch.Tensor:
        if self._rng.random() >= self._p:
            return img_chw
        self._ensure_loaded()
        # img_chw: (3, H, W) float in [0,1]. mask: (H, W) bool.
        c, h, w = img_chw.shape
        bg_path = self._bg_paths[self._rng.randrange(len(self._bg_paths))]
        from PIL import Image
        bg_pil = Image.open(bg_path).convert("RGB").resize((w, h), Image.BILINEAR)
        bg_arr = np.array(bg_pil, dtype=np.float32) / 255.0  # (H, W, 3)
        bg_t = torch.from_numpy(bg_arr).permute(2, 0, 1).to(img_chw.dtype).to(img_chw.device)
        # Resize mask to (H, W) if it doesn't match (the polygons were authored
        # at the native frame resolution, so this should match — but be safe).
        m = self._mask
        if m.shape != (h, w):
            m_pil = Image.fromarray(m.astype(np.uint8) * 255).resize((w, h), Image.NEAREST)
            m = np.array(m_pil) > 127
        mask_t = torch.from_numpy(m).to(img_chw.device)
        # Replace bg pixels in place. Broadcast (3, H, W)[:, mask_t] = bg_t[:, mask_t]
        out = img_chw.clone()
        for ch in range(c):
            out[ch][mask_t] = bg_t[ch][mask_t]
        return out

    def __reduce__(self):
        return (self.__class__, (self._mask_path, self._bg_dir, self._p, self._seed))


class PrintShuffleAugmenter:
    """Swap the contents of two non-target print regions.

    The TARGET print position is recorded per-dataset and never moves -- this
    preserves action-image alignment (the arm is reaching toward the recorded
    target pixel position). The two non-target prints (``other1_mask``,
    ``other2_mask``) get their pixel content swapped with probability ``p``.

    Bounding boxes of the two non-target masks may differ in size; we
    rectangular-crop each, resize to the other's bbox dimensions, and paste
    back. The masks themselves are bounding-rectangle polygons (not tight
    silhouettes), so a rectangular paste is correct.
    """

    def __init__(self, other1_path: str, other2_path: str, p: float = 0.5, seed: int = 0):
        self._other1_path = str(other1_path)
        self._other2_path = str(other2_path)
        self._p = float(p)
        self._seed = int(seed)
        self._rng = random.Random(seed)
        self._bbox1: tuple[int, int, int, int] | None = None
        self._bbox2: tuple[int, int, int, int] | None = None

    def _ensure_loaded(self) -> None:
        if self._bbox1 is None:
            self._bbox1 = _bbox_of_mask(_load_mask(self._other1_path))
        if self._bbox2 is None:
            self._bbox2 = _bbox_of_mask(_load_mask(self._other2_path))

    def __call__(self, img_chw: torch.Tensor) -> torch.Tensor:
        if self._rng.random() >= self._p:
            return img_chw
        self._ensure_loaded()
        x1a, y1a, x2a, y2a = self._bbox1
        x1b, y1b, x2b, y2b = self._bbox2
        if x2a <= x1a or y2a <= y1a or x2b <= x1b or y2b <= y1b:
            return img_chw  # empty bbox - skip
        # Crop each region; CHW slicing.
        patch_a = img_chw[:, y1a:y2a, x1a:x2a].clone()
        patch_b = img_chw[:, y1b:y2b, x1b:x2b].clone()
        # Resize each to fit the OTHER's bbox via torchvision functional resize.
        import torchvision.transforms.v2.functional as F  # noqa: N812
        size_b = (y2b - y1b, x2b - x1b)
        size_a = (y2a - y1a, x2a - x1a)
        a_to_b = F.resize(patch_a, size_b, antialias=True)
        b_to_a = F.resize(patch_b, size_a, antialias=True)
        out = img_chw.clone()
        out[:, y1a:y2a, x1a:x2a] = b_to_a
        out[:, y1b:y2b, x1b:x2b] = a_to_b
        return out

    def __reduce__(self):
        return (self.__class__, (self._other1_path, self._other2_path, self._p, self._seed))


class Eval3PrepDataset(Dataset):
    """Proxy wrapping a LeRobotDataset with episode truncation + task aug.

    Parameters
    ----------
    dataset : LeRobotDataset
        Underlying dataset (already constructed with delta_timestamps, transforms, etc.)
    max_frames_per_episode : int | None
        If set, keep only the first ``max_frames_per_episode`` frames of each
        episode. ``None`` disables truncation (equivalent to passing a huge number).
    task_aug_fn : Callable[[str], str] | None
        If provided, called on each row's ``task`` string before return.
    repair_gripper_open : bool
        If true, every action whose original gripper command is already above
        ``gripper_open_threshold`` is lifted to at least
        ``gripper_open_target``. Closed/grasp commands below the threshold are
        left unchanged.
    action_smooth_window : int
        If >1, centered moving-average window for action labels. The gripper is
        excluded unless ``action_smooth_gripper=True``.
    """

    def __init__(
        self,
        dataset,
        max_frames_per_episode: int | None = 600,
        task_aug_fn: Callable[[str], str] | None = None,
        bg_aug_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
        print_aug_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
        state_aug_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
        image_key: str = "observation.images.front",
        episode_filter: list[int] | None = None,
        truncate_at_placement: bool = False,
        truncate_placement_mode: str = "first",
        truncate_allow_over_cap_episodes: list[int] | set[int] | None = None,
        truncate_over_cap_extra_frames: int = 0,
        truncate_grip_threshold: float = 20.0,
        truncate_lift_threshold: float = -30.0,
        truncate_buffer_frames: int = 60,
        repair_gripper_open: bool = False,
        gripper_open_target: float = 55.0,
        gripper_open_threshold: float = 20.0,
        action_smooth_window: int = 0,
        action_smooth_gripper: bool = False,
        target_position_idx: int | None = None,
    ):
        self._ds = dataset
        self._task_aug_fn = task_aug_fn
        self._bg_aug_fn = bg_aug_fn
        self._print_aug_fn = print_aug_fn
        self._state_aug_fn = state_aug_fn
        self._image_key = image_key
        # v16 slot-bottleneck caches (populated at the end of __init__ when
        # EVAL3_SLOT_FRAME0=1; stay empty otherwise and on prep-cache hits).
        self._frame0_by_ep: dict[int, Any] = {}
        self._grasp_offset_by_ep: dict[int, int] = {}
        # Auxiliary-head supervision label (0=left, 1=middle, 2=right, None=unknown
        # -> emit -100 = CrossEntropy ignore_index so old single-celeb datasets
        # like `taylor_swift_1` don't contribute aux gradient).
        self._target_position_idx = (
            int(target_position_idx) if target_position_idx is not None else -100
        )

        # Optional per-dataset prep cache (opt-in via EVAL3_PREP_CACHE=1).
        # When the env var is unset/0, this block is a no-op and the rest of
        # __init__ runs exactly as in the original implementation.
        self._cache_key: str | None = None
        if _prep_cache_enabled():
            self._cache_key = _prep_cache_key(
                repo_id=dataset.repo_id,
                total_frames=int(dataset.meta.info.get("total_frames", 0)),
                total_episodes=int(dataset.meta.info.get("total_episodes", 0)),
                max_frames_per_episode=max_frames_per_episode,
                episode_filter=episode_filter,
                truncate_at_placement=truncate_at_placement,
                truncate_placement_mode=truncate_placement_mode,
                truncate_allow_over_cap_episodes=truncate_allow_over_cap_episodes,
                truncate_over_cap_extra_frames=truncate_over_cap_extra_frames,
                truncate_grip_threshold=truncate_grip_threshold,
                truncate_lift_threshold=truncate_lift_threshold,
                truncate_buffer_frames=truncate_buffer_frames,
                repair_gripper_open=repair_gripper_open,
                gripper_open_target=gripper_open_target,
                gripper_open_threshold=gripper_open_threshold,
                action_smooth_window=int(action_smooth_window or 0),
                action_smooth_gripper=action_smooth_gripper,
            )
            cached = _load_prep_cache(self._cache_key)
            if cached is not None:
                logging.info(
                    "eval3_prep_cache: HIT  %s  key=%s  (skipping ~80s slow path)",
                    dataset.repo_id, self._cache_key,
                )
                self._episode_from_idxs = cached["episode_from_idxs"]
                self._episode_to_idxs = cached["episode_to_idxs"]
                self._episode_index_by_frame = cached["episode_index_by_frame"]
                self._action_delta_indices = cached["action_delta_indices"]
                self._valid_indices = cached["valid_indices"]
                self._max_frames_per_episode = max_frames_per_episode
                self._original_num_frames = cached["original_num_frames"]
                self._episode_filter = cached["episode_filter"]
                self._truncate_at_placement = cached["truncate_at_placement"]
                self._truncate_placement_mode = cached["truncate_placement_mode"]
                self._truncation_log = cached["truncation_log"]
                self._kept_episode_indices = cached["kept_episode_indices"]
                self._action_overrides = cached["action_overrides"]
                self._action_repair_summary = cached["action_repair_summary"]
                self._meta = deepcopy(dataset.meta)
                self._meta.stats = cached["meta_stats"]
                return
            logging.info(
                "eval3_prep_cache: MISS %s  key=%s  (computing + saving to %s)",
                dataset.repo_id, self._cache_key, _prep_cache_dir(),
            )

        # Pull episode boundaries from the underlying meta. LeRobotDatasetMetadata.episodes
        # is a pandas DataFrame with columns ``dataset_from_index`` and ``dataset_to_index``.
        ep_df = dataset.meta.episodes
        from_idxs = list(ep_df["dataset_from_index"])
        to_idxs = list(ep_df["dataset_to_index"])
        self._episode_from_idxs = [int(x) for x in from_idxs]
        self._episode_to_idxs = [int(x) for x in to_idxs]
        self._episode_index_by_frame = [int(x) for x in dataset.hf_dataset["episode_index"]]
        self._action_delta_indices = self._get_action_delta_indices(dataset)

        # Pre-pull action column once if we need placement detection. This is
        # column-oriented access on the HF dataset (one fetch, not per-row).
        actions_col = None
        needs_actions = bool(truncate_at_placement or repair_gripper_open or int(action_smooth_window or 0) > 1)
        if needs_actions:
            actions_col = dataset.hf_dataset["action"]

        placement_mode = str(truncate_placement_mode or "first").strip().lower()
        if placement_mode not in {"first", "last"}:
            placement_mode = "first"
        over_cap_eps = (
            {int(e) for e in truncate_allow_over_cap_episodes}
            if truncate_allow_over_cap_episodes is not None
            else set()
        )
        over_cap_extra_frames = max(0, int(truncate_over_cap_extra_frames))

        # Build the per-episode kept-range.
        #   * truncate_at_placement + mode=first: find FIRST frame with
        #     gripper>=grip_thresh AND shoulder_lift>=lift_thresh, set cap =
        #     match_frame + buffer. This was the original v6 rule, but it can
        #     catch the pre-grasp gripper-open event in dataset_v2_*.
        #   * truncate_at_placement + mode=last: find the LAST matching frame
        #     before the final home-return tail, set cap to that frame inclusive,
        #     and still respect max_frames_per_episode. This keeps the actual
        #     visual placement while dropping the home tail.
        #   * otherwise: cap = f0 + max_frames_per_episode (legacy behaviour).
        # If episode_filter is provided, only include those episode indices.
        valid: list[int] = []
        original_total = 0
        keep_eps = set(int(e) for e in episode_filter) if episode_filter is not None else None
        truncation_log: list[dict] = []
        for ep_idx, (f0, f1) in enumerate(zip(from_idxs, to_idxs)):
            f0i, f1i = int(f0), int(f1)
            original_total += f1i - f0i
            if keep_eps is not None and ep_idx not in keep_eps:
                continue
            cap = None
            match_offset = None
            if truncate_at_placement and actions_col is not None:
                # Scan for placement-like open/lift frames.
                matched_offsets = []
                for off in range(f1i - f0i):
                    a = actions_col[f0i + off]
                    # Tensors may be torch.Tensor or python list depending on HF backend.
                    grip = float(a[5])
                    lift = float(a[1])
                    if grip >= truncate_grip_threshold and lift >= truncate_lift_threshold:
                        matched_offsets.append(off)
                        if placement_mode == "first":
                            break
                if matched_offsets:
                    match_offset = matched_offsets[0] if placement_mode == "first" else matched_offsets[-1]
                    if placement_mode == "first":
                        cap = f0i + match_offset + int(truncate_buffer_frames)
                    else:
                        cap = f0i + match_offset + 1
                    if max_frames_per_episode is not None:
                        extra = over_cap_extra_frames if ep_idx in over_cap_eps else 0
                        cap = min(cap, f0i + int(max_frames_per_episode) + extra)
                    cap = min(cap, f1i)
            if cap is None:
                if max_frames_per_episode is None:
                    cap = f1i
                else:
                    cap = min(f0i + int(max_frames_per_episode), f1i)
            valid.extend(range(f0i, cap))
            truncation_log.append({
                "ep_idx": ep_idx, "f0": f0i, "f1": f1i,
                "kept_to": cap, "match_offset": match_offset,
                "kept_frames": cap - f0i,
            })
        self._valid_indices = valid
        self._max_frames_per_episode = max_frames_per_episode
        self._original_num_frames = original_total
        self._episode_filter = list(keep_eps) if keep_eps is not None else None
        self._truncate_at_placement = truncate_at_placement
        self._truncate_placement_mode = placement_mode
        self._truncation_log = truncation_log
        self._kept_episode_indices = sorted(
            set(int(dataset.hf_dataset["episode_index"][i]) for i in self._valid_indices)
        )
        self._action_overrides, self._action_repair_summary = _build_action_overrides(
            dataset,
            self._truncation_log,
            repair_gripper_open=bool(repair_gripper_open),
            gripper_open_target=float(gripper_open_target),
            gripper_open_threshold=float(gripper_open_threshold),
            action_smooth_window=int(action_smooth_window or 0),
            action_smooth_gripper=bool(action_smooth_gripper),
        )
        self._meta = deepcopy(dataset.meta)
        self._meta.stats = _compute_filtered_stats(
            dataset,
            self._valid_indices,
            action_overrides=self._action_overrides,
        )

        # Persist for next launch — only if cache is enabled (self._cache_key
        # is None when EVAL3_PREP_CACHE is unset/0, so this is a no-op for the
        # default code path).
        if self._cache_key is not None:
            _save_prep_cache(self._cache_key, {
                "episode_from_idxs": self._episode_from_idxs,
                "episode_to_idxs": self._episode_to_idxs,
                "episode_index_by_frame": self._episode_index_by_frame,
                "action_delta_indices": self._action_delta_indices,
                "valid_indices": self._valid_indices,
                "original_num_frames": self._original_num_frames,
                "episode_filter": self._episode_filter,
                "truncate_at_placement": self._truncate_at_placement,
                "truncate_placement_mode": self._truncate_placement_mode,
                "truncation_log": self._truncation_log,
                "kept_episode_indices": self._kept_episode_indices,
                "action_overrides": self._action_overrides,
                "action_repair_summary": self._action_repair_summary,
                "meta_stats": self._meta.stats,
            })

        # ----- v16 slot-bottleneck: frame-0 image + grasp detection --------
        # Gated on EVAL3_SLOT_FRAME0=1. The slot classifier reads the
        # episode's frame-0 (static, pre-motion) scene via the camera2 slot;
        # the slot CE loss is counted only on pre-grasp frames.
        if os.environ.get("EVAL3_SLOT_FRAME0", "0").strip() == "1":
            try:
                grasp_delta = float(os.environ.get("EVAL3_GRASP_GRIP_DELTA", "25"))
            except ValueError:
                grasp_delta = 25.0
            state_col = dataset.hf_dataset["observation.state"]
            for ep_idx, (f0, f1) in enumerate(zip(self._episode_from_idxs, self._episode_to_idxs)):
                f0i, f1i = int(f0), int(f1)
                ep_len = max(1, f1i - f0i)
                try:
                    self._frame0_by_ep[ep_idx] = self._ds[f0i][self._image_key]
                except Exception as exc:
                    logging.warning("eval3_prep: frame-0 cache miss ep %d (%s)", ep_idx, exc)
                # grasp-complete = first frame the gripper has moved markedly
                # from its home value (open->closed). Detecting the transition
                # rather than an absolute level is convention-agnostic. Clamp
                # the result to a sane pre-grasp window [15%, 50%] of the
                # episode so a noisy / missing signal can't break it.
                g0 = float(state_col[f0i][5])
                grasp_off = None
                for off in range(ep_len):
                    if abs(float(state_col[f0i + off][5]) - g0) > grasp_delta:
                        grasp_off = off
                        break
                if grasp_off is None:
                    grasp_off = int(0.35 * ep_len)
                grasp_off = max(int(0.15 * ep_len), min(grasp_off, int(0.50 * ep_len)))
                self._grasp_offset_by_ep[ep_idx] = int(grasp_off)
            _g = list(self._grasp_offset_by_ep.values())
            logging.info(
                "eval3_prep: v16 frame-0 mode — cached %d frame-0 images, "
                "grasp offset median=%d (%s)",
                len(self._frame0_by_ep),
                int(np.median(_g)) if _g else -1, self._ds.repo_id,
            )

    # ----- Trainer-required attributes ------------------------------------

    @property
    def meta(self):
        return self._meta

    @property
    def features(self):
        return self._ds.features

    @property
    def repo_id(self) -> str:
        return self._ds.repo_id

    @property
    def num_frames(self) -> int:
        return len(self._valid_indices)

    @property
    def num_episodes(self) -> int:
        return len(self._kept_episode_indices)

    @property
    def episodes(self):
        # Episode-index filter (subset of episodes loaded). We don't filter episodes,
        # only their tails, so this passes through unchanged.
        return self._ds.episodes

    # ----- Dataset protocol -----------------------------------------------

    def __len__(self) -> int:
        return len(self._valid_indices)

    def _get_action_delta_indices(self, dataset) -> list[int] | None:
        delta_timestamps = getattr(dataset, "delta_timestamps", None)
        if not isinstance(delta_timestamps, dict) or "action" not in delta_timestamps:
            return None
        fps = float(getattr(dataset.meta, "fps", 30) or 30)
        return [int(round(float(ts) * fps)) for ts in delta_timestamps["action"]]

    def _apply_action_overrides(self, original_idx: int, current_action):
        if not self._action_overrides or current_action is None:
            return current_action

        was_tensor = isinstance(current_action, torch.Tensor)
        action_t = current_action if was_tensor else torch.as_tensor(current_action)
        if action_t.ndim == 0:
            return current_action

        def _cast_override(override: torch.Tensor) -> torch.Tensor:
            return override.to(dtype=action_t.dtype, device=action_t.device)

        if action_t.ndim == 1:
            override = self._action_overrides.get(int(original_idx))
            if override is None:
                return current_action
            repaired = _cast_override(override)
            return repaired if was_tensor else repaired.detach().cpu().tolist()

        repaired = action_t.clone()
        deltas = self._action_delta_indices
        if deltas is None or len(deltas) != repaired.shape[0]:
            deltas = list(range(int(repaired.shape[0])))

        ep_idx = self._episode_index_by_frame[int(original_idx)]
        ep_start = self._episode_from_idxs[ep_idx]
        ep_end = self._episode_to_idxs[ep_idx]
        for step_idx, delta in enumerate(deltas[: repaired.shape[0]]):
            query_idx = max(ep_start, min(ep_end - 1, int(original_idx) + int(delta)))
            override = self._action_overrides.get(int(query_idx))
            if override is not None:
                repaired[step_idx, : override.shape[-1]] = _cast_override(override)

        return repaired if was_tensor else repaired.detach().cpu().tolist()

    def __getitem__(self, idx) -> Any:
        original_idx = self._valid_indices[int(idx)]
        row = self._ds[original_idx]
        mutated = False
        if self._action_overrides and "action" in row:
            if not mutated:
                row = dict(row)
                mutated = True
            row["action"] = self._apply_action_overrides(int(original_idx), row["action"])
        if self._task_aug_fn is not None and "task" in row:
            if not mutated:
                row = dict(row)
                mutated = True
            row["task"] = self._task_aug_fn(row["task"])
        # Apply image augs: order matters. bg-replace first (writes random
        # pixels everywhere outside the table), then print-shuffle (swaps two
        # rectangles within the table) -- so we never paste random pixels over
        # a freshly shuffled print.
        if (self._bg_aug_fn is not None or self._print_aug_fn is not None) and self._image_key in row:
            if not mutated:
                row = dict(row)
                mutated = True
            img = row[self._image_key]
            if self._bg_aug_fn is not None:
                img = self._bg_aug_fn(img)
            if self._print_aug_fn is not None:
                img = self._print_aug_fn(img)
            row[self._image_key] = img
        # Apply state augmentation (Gaussian noise with optional curriculum,
        # HOME/zero replacement). See `StateAugmenter` for the design rationale.
        if self._state_aug_fn is not None and "observation.state" in row:
            if not mutated:
                row = dict(row)
                mutated = True
            row["observation.state"] = self._state_aug_fn(row["observation.state"])
        # Emit the auxiliary-head supervisory label. Always emit so the
        # default collate stacks a long tensor — value is -100 for datasets
        # without a known board slot (CrossEntropy ignore_index).
        if not mutated:
            row = dict(row)
            mutated = True
        row["target_position"] = torch.as_tensor(self._target_position_idx, dtype=torch.long)
        # v16: attach the slot-classifier image (-> camera2 via rename_map) +
        # the pre-grasp flag. camera2 = the CURRENT frame when this sample is
        # pre-grasp (so CE supervises every pre-grasp frame as the dataloader
        # sweeps the episode), else the episode's frame-0 (a valid pre-grasp
        # scene — keeps h_slot defined for the action expert; not CE'd because
        # is_pregrasp=0). At deploy camera2 is always frame-0, which is in this
        # CE-training distribution (t=0 is itself a pre-grasp frame).
        if self._frame0_by_ep or self._grasp_offset_by_ep:
            ep = int(self._episode_index_by_frame[int(original_idx)])
            grasp_off = self._grasp_offset_by_ep.get(ep)
            if grasp_off is None:
                is_pre = 1
            else:
                ep_start = self._episode_from_idxs[ep]
                is_pre = 1 if (int(original_idx) - ep_start) <= grasp_off else 0
            row["is_pregrasp"] = torch.as_tensor(int(is_pre), dtype=torch.long)
            if is_pre and self._image_key in row:
                row["observation.images.front_frame0"] = row[self._image_key]
            else:
                f0img = self._frame0_by_ep.get(ep)
                if f0img is not None:
                    row["observation.images.front_frame0"] = f0img
        return row

    # ----- Catch-all proxy --------------------------------------------------

    def __getattr__(self, name):
        # Called only when the attribute isn't found on this instance. Falls through
        # to the wrapped dataset. Note: properties defined above (meta, features, etc.)
        # are *not* dispatched through here.
        if "_ds" in self.__dict__:
            return getattr(self._ds, name)
        raise AttributeError(name)

    # ----- Debug helpers ---------------------------------------------------

    def truncation_summary(self) -> dict:
        # Placement-truncation summary (only meaningful when truncate_at_placement=True)
        place_matched = sum(1 for r in self._truncation_log if r.get("match_offset") is not None)
        place_total = len(self._truncation_log)
        place_kept_frames = sum(r["kept_frames"] for r in self._truncation_log)
        return {
            "repo_id": self.repo_id,
            "max_frames_per_episode": self._max_frames_per_episode,
            "truncate_at_placement": self._truncate_at_placement,
            "truncate_placement_mode": self._truncate_placement_mode,
            "placement_match_rate": (place_matched / place_total) if place_total else 0.0,
            "placement_avg_kept_frames": (place_kept_frames / place_total) if place_total else 0,
            "original_num_frames": int(self._original_num_frames),
            "kept_num_frames": int(self.num_frames),
            "dropped_num_frames": int(self._original_num_frames - self.num_frames),
            "kept_fraction": float(self.num_frames) / float(self._original_num_frames)
            if self._original_num_frames else 0.0,
            "original_num_episodes": int(self._ds.num_episodes),
            "kept_num_episodes": int(self.num_episodes),
            "kept_episode_indices": list(self._kept_episode_indices),
            **self._action_repair_summary,
            "action_stat_count": float(torch.as_tensor(self.meta.stats["action"]["count"]).flatten()[0])
            if "action" in self.meta.stats and "count" in self.meta.stats["action"]
            else None,
        }
