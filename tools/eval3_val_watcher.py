#!/usr/bin/env python3
"""Held-out validation watcher for v16/v17 SmolVLA training.

Polls a training output dir for new checkpoints; for each one, loads the
policy in this subprocess and scores it on a user-configurable list of
LeRobot-format datasets, then emits one JSONL line per checkpoint with
overall + per-slot + per-repo breakdown.

Four "right path" metrics (see docs/eval3/v17_playbook.md §4.1):

  slot_acc                  argmax(model._last_slot_logits) == target_position
                            (derived from repo name via _slot_from_repo)
  action_mae                |predicted_action - recorded_action|, joint-wise mean
  action_mae_per_joint      same, broken out per of the 6 SO-101 joints
  prompt_nearest_accuracy   correct-prompt prediction is closest L2 to GT
                            among predictions under {correct, other1, other2}
  cross_prompt_delta        mean pairwise L2 across the three prompts;
                            a sanity gate for prompt-sensitivity

Outputs:

  EVAL3_VAL_OUT  (default <train_out>/val_metrics.jsonl)   — always written
  stdout                                                    — always written
  wandb sidecar run                                         — opt-in via EVAL3_VAL_WANDB=1

Env-var contract: see plan / playbook. CLI flags override env vars.

Usage:

  # Continuous (default; the launcher uses this mode)
  python tools/eval3_val_watcher.py --train-out outputs/train/myjob

  # Single-shot: process the latest checkpoint then exit
  python tools/eval3_val_watcher.py --train-out outputs/train/myjob --once

  # Score a specific checkpoint without polling
  python tools/eval3_val_watcher.py --policy-path <ckpt-dir-or-hf-repo>
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ----- env defaults BEFORE we import any eval3 modules ---------------------
# The slot patch and shim must be loaded before any lerobot.policies import.
_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "scripts"))
sys.path.insert(0, str(_REPO / "tools"))

# Trigger slot-patch installation so model._last_slot_logits is populated.
os.environ.setdefault("EVAL3_SLOT_LOSS_WEIGHT", "0.5")

import eval3_lerobot_shim  # noqa: E402

eval3_lerobot_shim.apply()

import eval3_smolvla_slot_bottleneck as _SB  # noqa: E402

_SB.apply()

import numpy as np  # noqa: E402
import torch  # noqa: E402

from lerobot.configs.types import PolicyFeature  # noqa: E402  (unused but keeps import order)
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata  # noqa: E402
from lerobot.policies.factory import make_policy, make_pre_post_processors  # noqa: E402
from lerobot.utils.device_utils import get_safe_torch_device  # noqa: E402
from lerobot.configs.policies import PreTrainedConfig  # noqa: E402
from lerobot.processor.rename_processor import rename_stats  # noqa: E402
from lerobot.scripts.lerobot_record import predict_action  # noqa: E402

# Slot derivation regex shared with training (eval3_concat_patch._slot_from_repo)
JOINT_NAMES = (
    "shoulder_pan", "shoulder_lift", "elbow_flex",
    "wrist_flex", "wrist_roll", "gripper",
)
SLOT_NAMES = ("left", "middle", "right")
SLOT_TO_IDX = {s: i for i, s in enumerate(SLOT_NAMES)}

# Default identity → prompt mapping. Override with EVAL3_VAL_PROMPTS (JSON).
DEFAULT_PROMPTS = {
    "swift": "Place the coke on Taylor Swift",
    "lecun": "Place the coke on Yann LeCun",
    "obama": "Place the coke on Barack Obama",
}
IMAGE_KEY = "observation.images.front"
FRAME0_KEY = "observation.images.front_frame0"
STATE_KEY = "observation.state"


# ============================================================
# Config / env-var loading
# ============================================================

@dataclass
class Cfg:
    train_out: Path | None
    policy_path: str | None
    val_repos: list[str]
    val_local_repos: set[str]
    episodes_per_repo: int
    frames_per_episode: int
    prompts: dict[str, str]
    device: str
    poll_sec: int
    idle_sec: int
    out_path: Path | None
    seed: int
    final_step: int | None
    once: bool
    # wandb sidecar
    wandb_enable: bool
    wandb_project: str | None
    wandb_name: str | None


def _ev_str(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _ev_int(name: str, default: int) -> int:
    raw = _ev_str(name)
    try:
        return int(raw) if raw else default
    except ValueError:
        return default


def _ev_bool(name: str, default: bool = False) -> bool:
    raw = _ev_str(name).lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def _parse_csv(name: str) -> list[str]:
    raw = _ev_str(name)
    return [s.strip() for s in raw.split(",") if s.strip()]


def _parse_prompts() -> dict[str, str]:
    raw = _ev_str("EVAL3_VAL_PROMPTS")
    if not raw:
        return dict(DEFAULT_PROMPTS)
    try:
        return {str(k): str(v) for k, v in json.loads(raw).items()}
    except Exception as e:
        logging.warning("EVAL3_VAL_PROMPTS not valid JSON (%s) — using defaults", e)
        return dict(DEFAULT_PROMPTS)


def load_cfg(args: argparse.Namespace) -> Cfg:
    train_out = Path(args.train_out) if args.train_out else None
    out_default = train_out / "val_metrics.jsonl" if train_out else None
    out_env = _ev_str("EVAL3_VAL_OUT")
    out_path = Path(out_env) if out_env else out_default

    val_repos = args.val_repos or _parse_csv("EVAL3_VAL_REPOS")
    val_local = args.val_local_repos or _parse_csv("EVAL3_VAL_LOCAL_REPOS")

    device = args.device or _ev_str("EVAL3_VAL_DEVICE") or _ev_str("EVAL3_POLICY_DEVICE") or "cpu"

    return Cfg(
        train_out=train_out,
        policy_path=args.policy_path,
        val_repos=val_repos,
        val_local_repos=set(val_local),
        episodes_per_repo=args.episodes_per_repo or _ev_int("EVAL3_VAL_EPISODES_PER_REPO", 3),
        frames_per_episode=args.frames_per_episode or _ev_int("EVAL3_VAL_FRAMES_PER_EPISODE", 30),
        prompts=_parse_prompts(),
        device=device,
        poll_sec=args.poll_sec or _ev_int("EVAL3_VAL_POLL_SEC", 60),
        idle_sec=args.idle_sec or _ev_int("EVAL3_VAL_IDLE_SEC", 600),
        out_path=out_path,
        seed=args.seed or _ev_int("EVAL3_VAL_SEED", 0),
        final_step=args.final_step,
        once=args.once,
        wandb_enable=args.wandb or _ev_bool("EVAL3_VAL_WANDB", False),
        wandb_project=args.wandb_project or _ev_str("EVAL3_VAL_WANDB_PROJECT") or _ev_str("EVAL3_WANDB_PROJECT") or None,
        wandb_name=args.wandb_name or _ev_str("EVAL3_VAL_WANDB_NAME") or None,
    )


# ============================================================
# Repo / slot helpers
# ============================================================

def slot_from_repo(repo_id: str) -> str | None:
    rl = repo_id.lower()
    for slot in SLOT_NAMES:
        if f"_{slot}_" in rl or rl.endswith(f"_{slot}") or rl.endswith(f"_{slot}_full"):
            return slot
    return None


def identity_from_repo(repo_id: str, known_idents: set[str] | None = None) -> str | None:
    """Derive an identity slug from a repo name.

    First checks the 3 hardcoded TOY identities (Swift / LeCun / Obama). Then,
    if ``known_idents`` is supplied (typically the keys of
    ``EVAL3_VAL_PROMPTS``), tries matching each one as a substring of the
    repo basename. This lets the prompt JSON drive identity recognition for
    held-out celebrities (e.g. ``andrea_vedaldi``, ``hugh_jackman``, …)
    without code edits to this regex table.
    """
    rl = repo_id.lower()
    if "taylor_swift" in rl or "_taylor_" in rl:
        return "swift"
    if "yann_lecun" in rl or "_yann_" in rl:
        return "lecun"
    if "barack_obama" in rl or "_barack_" in rl:
        return "obama"
    if known_idents:
        # Try the prompt-dict slugs as substrings, longest first so a more
        # specific slug wins over a shorter prefix.
        for slug in sorted(known_idents, key=len, reverse=True):
            if slug and slug.lower() in rl:
                return slug
    return None


def local_root_for(repo_id: str, local_set: set[str]) -> str | None:
    if repo_id not in local_set:
        return None
    name = repo_id.rsplit("/", 1)[-1]
    for cand in (_REPO / "datasets" / name, Path.cwd() / "datasets" / name):
        if (cand / "meta").is_dir():
            return str(cand.resolve())
    raise FileNotFoundError(
        f"eval3_val_watcher: {repo_id} marked local but datasets/{name}/meta not found"
    )


# ============================================================
# Policy bundle loading (v17-aware rename_map)
# ============================================================

def detect_rename_map(policy_path: str) -> dict[str, str]:
    """Read rename_map from the checkpoint's train_config.json.
    Falls back to the single-camera v16/legacy default if unavailable.
    """
    tc: Path | None = None
    local = Path(policy_path) / "train_config.json"
    if local.is_file():
        tc = local
    elif "/" in policy_path and not Path(policy_path).exists():
        try:
            from huggingface_hub import hf_hub_download
            tc = Path(hf_hub_download(policy_path, "train_config.json"))
        except Exception:
            tc = None
    if tc is not None and tc.is_file():
        try:
            data = json.loads(tc.read_text(encoding="utf-8"))
            rmap = data.get("rename_map") or {}
            if isinstance(rmap, dict) and rmap:
                return {str(k): str(v) for k, v in rmap.items()}
        except Exception as e:
            logging.warning("could not parse train_config.json (%s): %s", tc, e)
    # default single-cam
    return {"observation.images.front": "observation.images.camera1"}


def is_v17_ckpt(rename_map: dict[str, str]) -> bool:
    return any("_frame0" in k for k in rename_map.keys())


def load_bundle(policy_path: str, device: str, meta_repo_id: str, rename_map: dict[str, str],
                meta_local_root: str | None = None):
    """Return (cfg, policy, preprocessor, postprocessor, torch_device).

    Mirrors scripts/eval3_smolvla_checkpoint_sweep.py:_load_policy_bundle but
    fetches dataset stats from the first val repo so normalization matches the
    eval distribution.

    ``meta_local_root`` lets the caller pin stats loading to a local on-disk
    copy (parallel to how ``LeRobotDataset(root=...)`` is used in this file);
    when omitted, ``LeRobotDatasetMetadata`` will fall back to the Hub.
    """
    if meta_local_root is not None:
        ds_meta = LeRobotDatasetMetadata(meta_repo_id, root=meta_local_root)
    else:
        ds_meta = LeRobotDatasetMetadata(meta_repo_id)
    cfg = PreTrainedConfig.from_pretrained(str(policy_path))
    cfg.device = device
    cfg.pretrained_path = str(policy_path)
    policy = make_policy(cfg, ds_meta=ds_meta, rename_map=rename_map)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path=str(policy_path),
        dataset_stats=rename_stats(ds_meta.stats, rename_map),
        preprocessor_overrides={
            "device_processor": {"device": cfg.device},
            "rename_observations_processor": {"rename_map": rename_map},
        },
    )
    return cfg, policy, preprocessor, postprocessor, get_safe_torch_device(cfg.device)


# ============================================================
# Sampling
# ============================================================

def _episode_bounds(ds: LeRobotDataset, ep_idx: int) -> tuple[int, int]:
    ep_df = ds.meta.episodes
    return int(ep_df["dataset_from_index"][ep_idx]), int(ep_df["dataset_to_index"][ep_idx])


def sample_frames(ds: LeRobotDataset, episodes_per_repo: int, frames_per_episode: int,
                  seed: int) -> list[tuple[int, int, int]]:
    """Return list of (episode_idx, original_frame_idx, episode_local_idx).

    Episode indices: evenly spread across the dataset's episode range.
    Frame indices: uniform stride within each episode.
    """
    n_ep = int(ds.num_episodes)
    if n_ep == 0:
        return []
    ep_step = max(1, n_ep // max(1, episodes_per_repo))
    ep_idxs = [min(n_ep - 1, ep_step * i) for i in range(episodes_per_repo)]
    # dedupe (small datasets may collapse multiple picks to the same episode)
    ep_idxs = sorted(set(ep_idxs))

    out: list[tuple[int, int, int]] = []
    for ep_idx in ep_idxs:
        f0, f1 = _episode_bounds(ds, ep_idx)
        ep_len = max(1, f1 - f0)
        n_pick = min(frames_per_episode, ep_len)
        if n_pick <= 1:
            picks_local = [0]
        else:
            picks_local = [int(i * (ep_len - 1) / (n_pick - 1)) for i in range(n_pick)]
        for li in picks_local:
            out.append((ep_idx, f0 + li, li))
    return out


# ============================================================
# Per-frame inference
# ============================================================

def _image_to_uint8(image: Any) -> np.ndarray:
    if isinstance(image, torch.Tensor):
        img_t = image.detach().cpu()
        if img_t.ndim == 3 and img_t.shape[0] in (1, 3):
            img_t = img_t.permute(1, 2, 0)
        if img_t.dtype.is_floating_point:
            img_t = (img_t.clamp(0, 1) * 255).to(torch.uint8)
        return img_t.numpy()
    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def _state_to_np(state: Any) -> np.ndarray:
    if isinstance(state, torch.Tensor):
        return state.detach().cpu().numpy().astype(np.float32)
    return np.asarray(state, dtype=np.float32)


def _action_to_np(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        arr = value.detach().cpu().numpy().astype(np.float32)
    else:
        arr = np.asarray(value, dtype=np.float32)
    return arr.reshape(-1)[:6]


def build_observation(row: dict, *, frame0_img: np.ndarray | None) -> dict[str, np.ndarray]:
    obs = {
        IMAGE_KEY: _image_to_uint8(row[IMAGE_KEY]),
        STATE_KEY: _state_to_np(row[STATE_KEY]),
    }
    if frame0_img is not None:
        obs[FRAME0_KEY] = frame0_img.copy()
    return obs


def predict_action_chunk(obs: dict[str, np.ndarray], *, task: str, bundle) -> np.ndarray:
    cfg, policy, preproc, postproc, device = bundle
    policy.reset()
    preproc.reset()
    postproc.reset()
    obs_copy = {k: np.array(v, copy=True) for k, v in obs.items()}
    with torch.no_grad():
        action = predict_action(
            observation=obs_copy,
            policy=policy,
            device=device,
            preprocessor=preproc,
            postprocessor=postproc,
            use_amp=cfg.use_amp,
            task=task,
        )
    return _action_to_np(action)


def slot_logits_for(bundle) -> np.ndarray | None:
    """Read the stashed slot logits from the policy (set by the patch)."""
    _, policy, _, _, _ = bundle
    logits = getattr(policy.model, "_last_slot_logits", None)
    if logits is None:
        return None
    return logits.detach().float().cpu().numpy().reshape(-1, 3)[0]


# ============================================================
# Per-checkpoint evaluation
# ============================================================

def eval_checkpoint(ckpt: str, cfg: Cfg) -> dict[str, Any]:
    """Score one checkpoint on all val repos; return a single result dict."""
    if not cfg.val_repos:
        raise RuntimeError("EVAL3_VAL_REPOS / --val-repos is empty; nothing to evaluate.")

    rename_map = detect_rename_map(ckpt)
    is_v17 = is_v17_ckpt(rename_map)
    logging.info("ckpt=%s  rename_map=%s  v17_frame0=%s", ckpt, rename_map, is_v17)

    # Load datasets; skip ones that fail rather than aborting the whole pass.
    datasets: dict[str, LeRobotDataset] = {}
    failed_repos: list[tuple[str, str]] = []
    for repo in cfg.val_repos:
        try:
            root = local_root_for(repo, cfg.val_local_repos)
            datasets[repo] = LeRobotDataset(repo, root=root, video_backend="pyav")
        except Exception as exc:
            logging.warning("val repo %s skipped (%s: %s)",
                            repo, type(exc).__name__, exc)
            failed_repos.append((repo, f"{type(exc).__name__}: {exc}"))
    if not datasets:
        raise RuntimeError(
            f"All val repos failed to load: {failed_repos}"
        )
    stats_repo = next(iter(datasets))
    stats_local_root = local_root_for(stats_repo, cfg.val_local_repos)
    bundle = load_bundle(ckpt, cfg.device, stats_repo, rename_map,
                         meta_local_root=stats_local_root)

    per_repo_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    t_start = time.time()

    # Per-repo distractor sampling: when cfg.prompts has more than 3 entries
    # (e.g. EVAL3_VAL_PROMPTS supplies the whole holdout celebrity pool), we
    # still only run 3 inferences per frame — the correct identity's prompt
    # + 2 distractors chosen deterministically per (repo, identity). This
    # keeps wall time O(3 inferences × N frames) instead of O(|prompts| × N).

    for repo, ds in datasets.items():
        slot_label = slot_from_repo(repo)
        ident_label = identity_from_repo(repo, known_idents=set(cfg.prompts.keys()))
        target_idx = SLOT_TO_IDX.get(slot_label) if slot_label else None
        if target_idx is None:
            logging.warning("repo %s: cannot derive slot from name; slot_acc will be NaN", repo)

        # Build the 3-prompt subset for this repo: correct identity + 2 random
        # distractors from cfg.prompts. If cfg.prompts has ≤3 entries, use all.
        prompts_for_repo: dict[str, str]
        if len(cfg.prompts) <= 3:
            prompts_for_repo = dict(cfg.prompts)
        else:
            others = [s for s in cfg.prompts if s != ident_label]
            # Deterministic per (repo, identity) — same seed -> same distractors.
            rng = random.Random(f"{cfg.seed}/{repo}/{ident_label}")
            distractors = rng.sample(others, min(2, len(others)))
            prompts_for_repo = {}
            if ident_label and ident_label in cfg.prompts:
                prompts_for_repo[ident_label] = cfg.prompts[ident_label]
            for d in distractors:
                prompts_for_repo[d] = cfg.prompts[d]

        # Cache frame-0 per episode if v17
        frame0_cache: dict[int, np.ndarray] = {}
        samples = sample_frames(ds, cfg.episodes_per_repo, cfg.frames_per_episode, cfg.seed)
        if not samples:
            continue
        if is_v17:
            for ep_idx, _, _ in samples:
                if ep_idx in frame0_cache:
                    continue
                f0, _ = _episode_bounds(ds, ep_idx)
                frame0_cache[ep_idx] = _image_to_uint8(ds[f0][IMAGE_KEY])

        for ep_idx, oi, local_idx in samples:
            row = ds[oi]
            recorded = _action_to_np(row["action"])
            f0_img = frame0_cache.get(ep_idx)
            obs = build_observation(row, frame0_img=f0_img)

            # Predict under three prompts; the correct one drives slot_acc.
            preds: dict[str, np.ndarray] = {}
            correct_logits: np.ndarray | None = None
            for slug, prompt in prompts_for_repo.items():
                preds[slug] = predict_action_chunk(obs, task=prompt, bundle=bundle)
                if slug == ident_label:
                    correct_logits = slot_logits_for(bundle)

            # Metric: action_mae and per-joint MAE (correct prompt vs recorded)
            correct_pred = preds.get(ident_label)
            if correct_pred is None:
                # identity not in prompt set — skip this frame's row-level metrics
                continue
            per_joint_abs_err = np.abs(correct_pred - recorded)
            action_mae = float(per_joint_abs_err.mean())

            # Metric: prompt_nearest_accuracy (closest L2 to recorded action)
            l2_to_gt = {slug: float(np.linalg.norm(p - recorded)) for slug, p in preds.items()}
            nearest = min(l2_to_gt, key=lambda k: l2_to_gt[k])
            prompt_nearest_correct = int(nearest == ident_label)

            # Metric: cross_prompt_delta — mean pairwise L2 between predictions
            keys = sorted(preds)
            pairwise = []
            for i, a in enumerate(keys):
                for b in keys[i + 1:]:
                    pairwise.append(float(np.linalg.norm(preds[a] - preds[b])))
            cross_prompt_delta = float(np.mean(pairwise)) if pairwise else float("nan")

            # Metric: slot_acc — only when target_idx is known and logits present.
            slot_correct: int | None = None
            slot_pred: int | None = None
            if correct_logits is not None and target_idx is not None:
                slot_pred = int(np.argmax(correct_logits))
                slot_correct = int(slot_pred == target_idx)

            per_repo_rows[repo].append({
                "ep_idx": ep_idx,
                "frame_idx": oi,
                "local_idx": local_idx,
                "identity": ident_label,
                "slot": slot_label,
                "target_idx": target_idx,
                "slot_pred": slot_pred,
                "slot_correct": slot_correct,
                "action_mae": action_mae,
                "per_joint_abs_err": per_joint_abs_err.tolist(),
                "prompt_nearest": nearest,
                "prompt_nearest_correct": prompt_nearest_correct,
                "cross_prompt_delta": cross_prompt_delta,
            })

    # Free memory before next checkpoint
    del bundle
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return _summarize(per_repo_rows, t_start)


def _summarize(per_repo_rows: dict[str, list[dict[str, Any]]],
               t_start: float) -> dict[str, Any]:
    all_rows = [r for rows in per_repo_rows.values() for r in rows]
    n_total = len(all_rows)

    def _agg(rows: list[dict[str, Any]]) -> dict[str, Any]:
        if not rows:
            return {
                "slot_acc": None, "action_mae": None,
                "action_mae_per_joint": {j: None for j in JOINT_NAMES},
                "prompt_nearest_accuracy": None,
                "cross_prompt_delta": None,
                "n_frames": 0,
            }
        slot_vals = [r["slot_correct"] for r in rows if r["slot_correct"] is not None]
        per_joint = np.stack([np.asarray(r["per_joint_abs_err"]) for r in rows], axis=0)
        return {
            "slot_acc": float(np.mean(slot_vals)) if slot_vals else None,
            "action_mae": float(np.mean([r["action_mae"] for r in rows])),
            "action_mae_per_joint": {
                j: float(per_joint[:, i].mean()) for i, j in enumerate(JOINT_NAMES)
            },
            "prompt_nearest_accuracy": float(np.mean([r["prompt_nearest_correct"] for r in rows])),
            "cross_prompt_delta": float(np.mean([r["cross_prompt_delta"] for r in rows])),
            "n_frames": len(rows),
        }

    by_slot: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in all_rows:
        if r["slot"]:
            by_slot[r["slot"]].append(r)
    per_slot_agg = {slot: _agg(by_slot.get(slot, [])) for slot in SLOT_NAMES}
    per_repo_agg = [
        {"repo": repo, **_agg(rows)} for repo, rows in per_repo_rows.items()
    ]

    return {
        "wall_time_s": float(time.time() - t_start),
        "n_frames_evaluated": n_total,
        "overall": _agg(all_rows),
        "per_slot": per_slot_agg,
        "per_repo": per_repo_agg,
    }


# ============================================================
# Output writers (JSONL + stdout + wandb sidecar)
# ============================================================

def _jsonl_header(cfg: Cfg) -> dict[str, Any]:
    return {
        "schema": "eval3_val_watcher/v1",
        "started_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "train_out": str(cfg.train_out) if cfg.train_out else None,
        "policy_path": cfg.policy_path,
        "val_repos": cfg.val_repos,
        "val_local_repos": sorted(cfg.val_local_repos),
        "episodes_per_repo": cfg.episodes_per_repo,
        "frames_per_episode": cfg.frames_per_episode,
        "prompts": cfg.prompts,
        "device": cfg.device,
        "seed": cfg.seed,
    }


def _append_jsonl(cfg: Cfg, out_path: Path, record: dict[str, Any]) -> None:
    """Self-bootstrapping append: writes the header on first call (lazy),
    creating the parent dir only at that point. This avoids racing with
    lerobot's train.validate() which rejects a pre-existing train_out dir.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not out_path.exists()
    with out_path.open("a") as f:
        if new_file:
            f.write(json.dumps(_jsonl_header(cfg)) + "\n")
        f.write(json.dumps(record) + "\n")


def _print_summary(step: int | None, result: dict[str, Any]) -> None:
    o = result["overall"]
    step_str = f"step={step}" if step is not None else "step=?"
    print(
        f"[val] {step_str}  n_frames={result['n_frames_evaluated']}  "
        f"wall={result['wall_time_s']:.1f}s  "
        f"slot_acc={o['slot_acc']}  action_mae={o['action_mae']}  "
        f"prompt_nearest_acc={o['prompt_nearest_accuracy']}  "
        f"cross_prompt_delta={o['cross_prompt_delta']}",
        flush=True,
    )
    for slot, s in result["per_slot"].items():
        if s["n_frames"] == 0:
            continue
        print(
            f"    [{slot:6s}] n={s['n_frames']:3d}  "
            f"slot_acc={s['slot_acc']}  action_mae={s['action_mae']}  "
            f"prompt_nearest_acc={s['prompt_nearest_accuracy']}  "
            f"cross_prompt_delta={s['cross_prompt_delta']}",
            flush=True,
        )


def _wandb_log(run, step: int | None, result: dict[str, Any]) -> None:
    if run is None:
        return
    o = result["overall"]
    payload: dict[str, Any] = {
        "val/slot_acc": o["slot_acc"],
        "val/action_mae": o["action_mae"],
        "val/prompt_nearest_accuracy": o["prompt_nearest_accuracy"],
        "val/cross_prompt_delta": o["cross_prompt_delta"],
        "val/n_frames_evaluated": result["n_frames_evaluated"],
        "val/wall_time_s": result["wall_time_s"],
    }
    for j, v in o["action_mae_per_joint"].items():
        payload[f"val/action_mae_per_joint/{j}"] = v
    for slot, s in result["per_slot"].items():
        payload[f"val/per_slot/{slot}/slot_acc"] = s["slot_acc"]
        payload[f"val/per_slot/{slot}/action_mae"] = s["action_mae"]
        payload[f"val/per_slot/{slot}/prompt_nearest_accuracy"] = s["prompt_nearest_accuracy"]
        payload[f"val/per_slot/{slot}/cross_prompt_delta"] = s["cross_prompt_delta"]
        payload[f"val/per_slot/{slot}/n_frames"] = s["n_frames"]
    run.log(payload, step=step or 0)


def _maybe_init_wandb(cfg: Cfg):
    if not cfg.wandb_enable:
        return None
    try:
        import wandb
    except ImportError:
        logging.warning("EVAL3_VAL_WANDB=1 but wandb not installed; skipping sidecar.")
        return None
    name = cfg.wandb_name
    if name is None and cfg.train_out is not None:
        name = f"{cfg.train_out.name}_val"
    elif name is None:
        name = "eval3_val"
    return wandb.init(
        project=cfg.wandb_project,
        name=name,
        job_type="val",
        reinit=True,
        config={
            "val_repos": cfg.val_repos,
            "val_local_repos": sorted(cfg.val_local_repos),
            "episodes_per_repo": cfg.episodes_per_repo,
            "frames_per_episode": cfg.frames_per_episode,
            "prompts": cfg.prompts,
        },
    )


# ============================================================
# Watch loop
# ============================================================

def discover_checkpoints(train_out: Path) -> list[tuple[int, Path]]:
    ckpt_root = train_out / "checkpoints"
    if not ckpt_root.is_dir():
        return []
    out: list[tuple[int, Path]] = []
    for d in sorted(ckpt_root.iterdir()):
        if d.name.isdigit() and (d / "pretrained_model").is_dir():
            out.append((int(d.name), d / "pretrained_model"))
    return out


def watch_loop(cfg: Cfg) -> int:
    if cfg.train_out is None:
        raise RuntimeError("--train-out is required for watch mode")
    out_path = cfg.out_path
    assert out_path is not None
    wandb_run = _maybe_init_wandb(cfg)

    seen: set[int] = set()
    last_new = time.time()

    print(f"[val] watching {cfg.train_out}/checkpoints  out={out_path}", flush=True)
    while True:
        steps = discover_checkpoints(cfg.train_out)
        for step, ckpt_dir in steps:
            if step in seen:
                continue
            seen.add(step)
            last_new = time.time()
            print(f"\n=== val on checkpoint step {step} ({ckpt_dir}) ===", flush=True)
            try:
                result = eval_checkpoint(str(ckpt_dir), cfg)
            except Exception as exc:  # keep watching even if one eval fails
                logging.exception("eval error on step %d: %s", step, exc)
                continue
            record = {
                "step": step,
                "checkpoint": str(ckpt_dir),
                **result,
            }
            _append_jsonl(cfg, out_path, record)
            _print_summary(step, result)
            _wandb_log(wandb_run, step, result)
            if cfg.once:
                print("[val] --once mode; exiting after first checkpoint.", flush=True)
                if wandb_run is not None:
                    wandb_run.finish()
                return 0
            if cfg.final_step is not None and step >= cfg.final_step:
                print(f"[val] final_step={cfg.final_step} reached — exiting.", flush=True)
                if wandb_run is not None:
                    wandb_run.finish()
                return 0
        if (time.time() - last_new) > cfg.idle_sec:
            print(f"[val] no new checkpoint for {cfg.idle_sec}s — exiting.", flush=True)
            if wandb_run is not None:
                wandb_run.finish()
            return 0
        time.sleep(cfg.poll_sec)


def one_shot(cfg: Cfg) -> int:
    if cfg.policy_path is None and cfg.train_out is not None:
        steps = discover_checkpoints(cfg.train_out)
        if not steps:
            print("[val] no checkpoints found.", flush=True)
            return 2
        step, ckpt_dir = steps[-1]
        policy_path = str(ckpt_dir)
    else:
        step = None
        policy_path = cfg.policy_path
    if policy_path is None:
        raise RuntimeError("Neither --policy-path nor --train-out (with checkpoints) provided.")
    wandb_run = _maybe_init_wandb(cfg)
    result = eval_checkpoint(policy_path, cfg)
    record = {"step": step, "checkpoint": policy_path, **result}
    if cfg.out_path is not None:
        _append_jsonl(cfg, cfg.out_path, record)
    _print_summary(step, result)
    _wandb_log(wandb_run, step, result)
    if wandb_run is not None:
        wandb_run.finish()
    return 0


# ============================================================
# CLI
# ============================================================

def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train-out", default=None, help="training output dir to watch")
    ap.add_argument("--policy-path", default=None,
                    help="score this specific checkpoint dir (or HF repo); skips polling")
    ap.add_argument("--once", action="store_true",
                    help="process the most-recent checkpoint and exit")
    ap.add_argument("--val-repos", nargs="*", default=None,
                    help="override EVAL3_VAL_REPOS")
    ap.add_argument("--val-local-repos", nargs="*", default=None,
                    help="override EVAL3_VAL_LOCAL_REPOS")
    ap.add_argument("--episodes-per-repo", type=int, default=None)
    ap.add_argument("--frames-per-episode", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--poll-sec", type=int, default=None)
    ap.add_argument("--idle-sec", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--final-step", type=int, default=None,
                    help="exit after this step is processed (matches training step count)")
    ap.add_argument("--wandb", action="store_true",
                    help="enable wandb sidecar run (overrides EVAL3_VAL_WANDB)")
    ap.add_argument("--wandb-project", default=None)
    ap.add_argument("--wandb-name", default=None)
    args = ap.parse_args()

    cfg = load_cfg(args)
    if not cfg.val_repos:
        print("ERROR: provide --val-repos or EVAL3_VAL_REPOS (comma-separated).",
              file=sys.stderr, flush=True)
        return 2

    if cfg.policy_path is not None or cfg.once:
        return one_shot(cfg)
    return watch_loop(cfg)


if __name__ == "__main__":
    raise SystemExit(main())
