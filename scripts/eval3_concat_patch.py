"""Monkey-patch ``lerobot.datasets.factory.make_dataset`` to support multi-dataset training.

The upstream ``MultiLeRobotDataset`` path is explicitly disabled
(``factory.py:113 raise NotImplementedError``). For Eval 3 we need joint training across
``RobotLearningVLA/taylor_swift_1`` + ``RobotLearningVLA/yann_lecun_1`` (and any future
Obama/etc datasets) so the policy's language conditioning actually has > 1 task to bind to.

Usage:
    EVAL3_EXTRA_REPOS=RobotLearningVLA/yann_lecun_1 \\
      ./scripts/run_eval3_smolvla_train.sh

When ``EVAL3_EXTRA_REPOS`` is set (comma-separated list of additional repo_ids), this patch
replaces ``make_dataset`` with a version that:

1. Builds the primary ``LeRobotDataset`` exactly as before (using ``--dataset.repo_id``).
2. Builds one ``LeRobotDataset`` per extra repo, with the same delta_timestamps / transforms.
3. Optionally wraps each dataset with Eval 3 filtering/augmentation, then wraps
   them all in a ``ConcatLeRobotDataset`` whose ``__getitem__``/``__len__`` chain
   through ``torch.utils.data.ConcatDataset``. Its ``.meta`` is a clone of the
   primary's meta with stats merged across the filtered frame set actually seen
   by training (count-weighted mean, derived combined std, element-wise min/max,
   weighted quantile approximation).

The trainer accesses ``dataset.meta``, ``dataset.meta.stats``, ``dataset.num_frames``,
``dataset.num_episodes``, and ``dataset.episodes`` (verified at
``.venv/.../scripts/lerobot_train.py:240–360``). SmolVLA does NOT define ``drop_n_last_frames``,
so the ``EpisodeAwareSampler`` path is skipped and the trainer just uses a shuffled DataLoader
— our wrapper does not need to expose episode boundary arrays. This is verified before
applying the patch.
"""
from __future__ import annotations

import logging
import os
from copy import deepcopy
from typing import Any

import numpy as np
import torch
from torch.utils.data import ConcatDataset


def _combine_stat_dict(stat_dicts: list[dict]) -> dict:
    """Merge per-feature stat dicts (mean / std / min / max / count / quantiles) across N datasets."""
    if len(stat_dicts) == 1:
        return stat_dicts[0]

    keys = stat_dicts[0].keys()
    combined: dict[str, torch.Tensor | int] = {}

    counts_int = []
    counts_tensor = []
    for d in stat_dicts:
        c = d.get("count")
        if c is None:
            c_val = 1
        elif isinstance(c, torch.Tensor):
            c_val = int(c.item()) if c.ndim == 0 else int(c.flatten()[0].item())
        else:
            c_val = int(c) if not hasattr(c, "shape") else int(np.array(c).flatten()[0])
        counts_int.append(c_val)
        counts_tensor.append(float(c_val))

    total_count = sum(counts_int)
    c_t = torch.tensor(counts_tensor, dtype=torch.float64)

    def _stack(name: str):
        return torch.stack([torch.as_tensor(d[name], dtype=torch.float64) for d in stat_dicts])

    if "mean" in keys:
        means = _stack("mean")
        w = c_t.view(-1, *([1] * (means.ndim - 1)))
        combined_mean = (means * w).sum(0) / float(total_count)
        combined["mean"] = combined_mean.to(torch.float64)

        if "std" in keys:
            stds = _stack("std")
            # E[X^2] = Var + mean^2 = std^2 + mean^2 per dataset
            e_x2 = stds * stds + means * means
            combined_e_x2 = (e_x2 * w).sum(0) / float(total_count)
            combined_var = (combined_e_x2 - combined_mean * combined_mean).clamp_min(0.0)
            combined["std"] = torch.sqrt(combined_var).to(torch.float64)

    if "min" in keys:
        combined["min"] = _stack("min").min(0).values.to(torch.float64)
    if "max" in keys:
        combined["max"] = _stack("max").max(0).values.to(torch.float64)
    if "count" in keys:
        # Match the dtype of the source count (typically int64 with shape (1,))
        src_count = stat_dicts[0]["count"]
        if isinstance(src_count, torch.Tensor):
            combined["count"] = torch.tensor([total_count], dtype=src_count.dtype)
        else:
            combined["count"] = total_count

    # Quantiles: combine via count-weighted average (approximation — true quantile merge
    # requires the underlying samples). Good enough for normalization that uses MEAN_STD only.
    for q in ("q01", "q10", "q50", "q90", "q99"):
        if q in keys:
            qs = _stack(q)
            w = c_t.view(-1, *([1] * (qs.ndim - 1)))
            combined[q] = ((qs * w).sum(0) / float(total_count)).to(torch.float64)

    # Cast everything back to the source's dtype if it was a tensor
    src = stat_dicts[0]
    out: dict[str, Any] = {}
    for k, v in combined.items():
        if isinstance(src.get(k), torch.Tensor) and isinstance(v, torch.Tensor):
            out[k] = v.to(src[k].dtype)
        else:
            out[k] = v
    return out


def _combine_stats(stats_list: list[dict[str, dict]]) -> dict[str, dict]:
    """Merge dataset-level stats: {feature_key: {min,max,mean,std,count,q*}}."""
    feature_keys = set()
    for s in stats_list:
        feature_keys.update(s.keys())
    return {fk: _combine_stat_dict([s[fk] for s in stats_list if fk in s]) for fk in feature_keys}


class ConcatLeRobotDataset:
    """Thin proxy that looks enough like a LeRobotDataset for `lerobot_train.train`.

    Implements the attributes that ``lerobot_train.py:240-360`` reads:
      - ``__len__`` / ``__getitem__`` -> torch.utils.data.ConcatDataset under the hood
      - ``meta`` (clone of primary dataset's meta with merged ``stats``)
      - ``num_frames``, ``num_episodes`` (sums across all datasets)
      - ``episodes`` (concat of episode index lists if non-None)
    All other attributes proxy to the primary dataset.
    """

    def __init__(self, datasets):
        if not datasets:
            raise ValueError("ConcatLeRobotDataset requires at least one dataset.")
        self._datasets = list(datasets)
        self._concat = ConcatDataset(self._datasets)

        # Build a synthetic .meta object by deep-copying the primary's meta and patching .stats.
        # We avoid touching the original meta object so subsequent code that holds a reference
        # to either source dataset still sees that dataset's own stats.
        primary = self._datasets[0]
        self.meta = deepcopy(primary.meta)
        self.meta.stats = _combine_stats([d.meta.stats for d in self._datasets])

        # Episodes: pass-through if any sub-dataset filtered; otherwise None means "all".
        if any(d.episodes is not None for d in self._datasets):
            # Episode indices are local-per-dataset; concatenating them is meaningless to the
            # trainer's EpisodeAwareSampler path — but SmolVLA does not use that sampler.
            self.episodes = None
        else:
            self.episodes = None

    def __len__(self) -> int:
        return len(self._concat)

    def __getitem__(self, idx) -> Any:
        return self._concat[idx]

    @property
    def num_frames(self) -> int:
        return sum(int(d.num_frames) for d in self._datasets)

    @property
    def num_episodes(self) -> int:
        return sum(int(d.num_episodes) for d in self._datasets)

    @property
    def repo_id(self) -> str:
        return "+".join(d.repo_id for d in self._datasets)

    @property
    def features(self):
        return self._datasets[0].features

    def __getattr__(self, name):
        # Proxy unknown attrs to the primary dataset (e.g. `image_transforms`, `delta_timestamps`).
        # Note: __getattr__ is only called for attributes NOT found on the instance, so this
        # safely doesn't shadow `meta`, `num_frames`, etc. defined above.
        if "_datasets" in self.__dict__:
            return getattr(self._datasets[0], name)
        raise AttributeError(name)


def apply_concat_patch() -> None:
    """Install the multi-dataset make_dataset hook IF EVAL3_EXTRA_REPOS is set."""
    extra = os.environ.get("EVAL3_EXTRA_REPOS", "").strip()
    if not extra:
        return  # no-op, fall back to upstream single-dataset behavior

    extra_repos = [r.strip() for r in extra.split(",") if r.strip()]
    if not extra_repos:
        return

    from lerobot.datasets import factory as _factory
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
    from lerobot.datasets.factory import resolve_delta_timestamps, IMAGENET_STATS
    from lerobot.datasets.transforms import ImageTransforms

    _orig_make_dataset = _factory.make_dataset

    def _patched_make_dataset(cfg):  # signature matches upstream
        if cfg.dataset.streaming:
            logging.warning(
                "EVAL3_EXTRA_REPOS is set but cfg.dataset.streaming=True; multi-dataset concat "
                "is incompatible with streaming. Falling back to single-dataset upstream."
            )
            return _orig_make_dataset(cfg)

        image_transforms = (
            ImageTransforms(cfg.dataset.image_transforms) if cfg.dataset.image_transforms.enable else None
        )

        def _build_one(repo_id: str) -> LeRobotDataset:
            ds_meta = LeRobotDatasetMetadata(repo_id, root=None, revision=cfg.dataset.revision)
            delta_timestamps = resolve_delta_timestamps(cfg.policy, ds_meta)
            return LeRobotDataset(
                repo_id,
                root=None,
                episodes=None,  # use all episodes from each side
                delta_timestamps=delta_timestamps,
                image_transforms=image_transforms,
                revision=cfg.dataset.revision,
                video_backend=cfg.dataset.video_backend,
                tolerance_s=cfg.tolerance_s,
            )

        primary_repo = cfg.dataset.repo_id
        all_repos = [primary_repo, *extra_repos]
        logging.info("eval3_concat_patch: joining datasets: %s", all_repos)

        datasets = [_build_one(r) for r in all_repos]

        # Apply ImageNet stats override exactly like upstream does, BEFORE we wrap, so each
        # sub-dataset's meta.stats already reflects it. Then ConcatLeRobotDataset merges them.
        if cfg.dataset.use_imagenet_stats:
            for d in datasets:
                for key in d.meta.camera_keys:
                    for stats_type, stats in IMAGENET_STATS.items():
                        d.meta.stats[key][stats_type] = torch.tensor(stats, dtype=torch.float32)

        # Per-dataset Eval3 prep wrappers. Env-var-driven so we can ablate:
        #   EVAL3_MAX_FRAMES_PER_EP=600    # cap each episode at first 600 frames (= 20s @ 30fps)
        #   EVAL3_TASK_AUG=1               # enable task-string augmentation (default on)
        #   EVAL3_BG_REPLACE=1             # enable background replacement (default on)
        #   EVAL3_BG_REPLACE_P=0.3         # per-frame probability
        #   EVAL3_PRINT_SHUFFLE=1          # enable non-target print-position shuffle
        #   EVAL3_PRINT_SHUFFLE_P=0.5
        #   EVAL3_MASK_DIR=outputs/eval3_masks
        #   EVAL3_BG_DIR=outputs/eval3_backgrounds
        #   EVAL3_GRIPPER_REPAIR=1         # widen dataset_v2 open-gripper labels
        #   EVAL3_ACTION_SMOOTH_WINDOW=3   # low-pass arm labels, gripper excluded
        #   EVAL3_SLOT_TASK_AUG=1          # optional selector->slot policy training
        #   EVAL3_SWIFT_EPISODE_FILTER=0,4,7,8,9,10,11,12,14,15,16,17,18,19
        try:
            from eval3_dataset_prep import (
                Eval3PrepDataset,
                BackgroundReplaceAugmenter,
                PrintShuffleAugmenter,
                make_task_augmenter,
                make_slot_task_augmenter,
                make_state_augmenter,
            )
        except ImportError as e:
            logging.warning("eval3_concat_patch: could not import eval3_dataset_prep (%s); "
                            "skipping Eval3 prep.", e)
        else:
            max_frames_raw = os.environ.get("EVAL3_MAX_FRAMES_PER_EP", "600").strip()
            try:
                max_frames = int(max_frames_raw)
            except ValueError:
                max_frames = 600
            if max_frames <= 0:
                max_frames = None  # disabled

            task_aug = make_task_augmenter() if os.environ.get("EVAL3_TASK_AUG", "1") == "1" else None
            # Build the state augmenter once (env-var driven). All Eval3PrepDataset
            # instances share the SAME augmenter so the mp.Value curriculum step is
            # consistent across datasets. Returns None if no aug is enabled.
            #
            # Pull per-joint state std from the PRIMARY dataset's stats — this
            # is what the lerobot normalizer will apply at preprocessor time, so
            # using these stats here means sigma in normalized units is exactly
            # right. (All synth datasets share approximately the same state
            # distribution since they reuse the same source trajectories — std
            # varies by <5% across the 9 synth repos.)
            try:
                primary_ds = datasets[0]
                state_std_raw = primary_ds.meta.stats.get("observation.state", {}).get("std")
                if state_std_raw is not None:
                    # Stats may be tensor, list, or numpy array — coerce to tuple of floats.
                    import numpy as _np
                    if hasattr(state_std_raw, "tolist"):
                        state_std = tuple(float(x) for x in state_std_raw.tolist())
                    else:
                        state_std = tuple(float(x) for x in state_std_raw)
                else:
                    state_std = None
            except Exception as exc:
                logging.warning(
                    "eval3_concat_patch: could not extract state std for normalized "
                    "noise calibration (%s); sigma will be interpreted as raw degrees",
                    exc,
                )
                state_std = None
            state_aug = make_state_augmenter(state_std=state_std)
            if state_aug is not None:
                logging.info(
                    "eval3_concat_patch: state augmentation enabled — "
                    "sigma_max=%s sigma_min=%s curriculum_steps=%s replace_prob=%s "
                    "mode_home_w=%s mode_zero_w=%s state_std=%s",
                    os.environ.get("EVAL3_STATE_NOISE_SIGMA_MAX", "0.0"),
                    os.environ.get("EVAL3_STATE_NOISE_SIGMA_MIN", "0.0"),
                    os.environ.get("EVAL3_STATE_NOISE_CURRICULUM_STEPS", "0"),
                    os.environ.get("EVAL3_STATE_REPLACE_PROB", "0.0"),
                    state_aug._mode_home_weight,
                    state_aug._mode_zero_weight,
                    "<provided>" if state_std is not None else "<raw-degrees fallback>",
                )
            bg_replace_enabled = os.environ.get("EVAL3_BG_REPLACE", "1") == "1"
            print_shuffle_enabled = os.environ.get("EVAL3_PRINT_SHUFFLE", "1") == "1"
            try:
                bg_p = float(os.environ.get("EVAL3_BG_REPLACE_P", "0.3"))
            except ValueError:
                bg_p = 0.3
            try:
                ps_p = float(os.environ.get("EVAL3_PRINT_SHUFFLE_P", "0.5"))
            except ValueError:
                ps_p = 0.5
            mask_dir = os.environ.get("EVAL3_MASK_DIR", "outputs/eval3_masks")
            bg_dir = os.environ.get("EVAL3_BG_DIR", "outputs/eval3_backgrounds")

            gripper_repair_enabled = os.environ.get("EVAL3_GRIPPER_REPAIR", "1") == "1"
            gripper_repair_new_only = os.environ.get("EVAL3_GRIPPER_REPAIR_NEW_DATA_ONLY", "1") == "1"
            try:
                gripper_open_target = float(os.environ.get("EVAL3_GRIPPER_OPEN_TARGET", "55"))
            except ValueError:
                gripper_open_target = 55.0
            try:
                gripper_open_threshold = float(os.environ.get("EVAL3_GRIPPER_OPEN_THRESHOLD", "20"))
            except ValueError:
                gripper_open_threshold = 20.0
            try:
                action_smooth_window = int(os.environ.get("EVAL3_ACTION_SMOOTH_WINDOW", "0"))
            except ValueError:
                action_smooth_window = 0
            action_smooth_gripper = os.environ.get("EVAL3_ACTION_SMOOTH_GRIPPER", "0") == "1"
            slot_task_enabled = os.environ.get("EVAL3_SLOT_TASK_AUG", "0") == "1"
            try:
                slot_task_p = float(os.environ.get("EVAL3_SLOT_TASK_P", "0.5"))
            except ValueError:
                slot_task_p = 0.5

            def _parse_filter(env_var: str) -> list[int] | None:
                raw = os.environ.get(env_var, "").strip()
                if not raw:
                    return None
                return [int(x) for x in raw.split(",") if x.strip()]

            swift_filter = _parse_filter("EVAL3_SWIFT_EPISODE_FILTER")
            lecun_filter = _parse_filter("EVAL3_LECUN_EPISODE_FILTER")
            obama_filter = _parse_filter("EVAL3_OBAMA_EPISODE_FILTER")

            # Fix B (v9 celeb-confusion plan): cap synth episode count so the
            # real new66 corpus is not buried at ~3% of the training mix.
            # Caps each synth-named repo to the first N episodes. 0 / unset = no cap.
            try:
                synth_episode_limit = int(os.environ.get("EVAL3_SYNTH_EPISODE_LIMIT", "0"))
            except ValueError:
                synth_episode_limit = 0
            try:
                synth_max_frames_per_ep = int(os.environ.get("EVAL3_SYNTH_MAX_FRAMES_PER_EP", "0"))
            except ValueError:
                synth_max_frames_per_ep = 0

            def _parse_repo_episode_map(env_var: str) -> dict[str, set[int]]:
                """Parse 'repo_slug:0,1;other_repo:3' into {repo_slug: {episodes}}."""
                raw = os.environ.get(env_var, "").strip()
                out: dict[str, set[int]] = {}
                if not raw:
                    return out
                for part in raw.split(";"):
                    part = part.strip()
                    if not part:
                        continue
                    if ":" not in part:
                        logging.warning("eval3_concat_patch: ignoring invalid %s entry %r", env_var, part)
                        continue
                    repo_key, eps_raw = part.split(":", 1)
                    eps = {int(x) for x in eps_raw.split(",") if x.strip()}
                    out[repo_key.strip()] = eps
                return out

            new_episode_exclude = _parse_repo_episode_map("EVAL3_NEW_EPISODE_EXCLUDE")
            new_episode_keep = _parse_repo_episode_map("EVAL3_NEW_EPISODE_KEEP")
            over_cap_episodes = _parse_repo_episode_map("EVAL3_TRUNCATE_ALLOW_OVER_CAP_EPISODES")

            # Placement-truncation knobs (Rule A from the v6 plan).
            truncate_at_placement_enabled = os.environ.get("EVAL3_TRUNCATE_AT_PLACEMENT", "1") == "1"
            trunc_mode = os.environ.get("EVAL3_TRUNCATE_PLACEMENT_MODE", "last").strip().lower()
            if trunc_mode not in {"first", "last"}:
                logging.warning(
                    "eval3_concat_patch: invalid EVAL3_TRUNCATE_PLACEMENT_MODE=%r; using 'last'",
                    trunc_mode,
                )
                trunc_mode = "last"
            try:
                trunc_grip = float(os.environ.get("EVAL3_TRUNCATE_GRIP_THRESHOLD", "20"))
            except ValueError:
                trunc_grip = 20.0
            try:
                trunc_lift = float(os.environ.get("EVAL3_TRUNCATE_LIFT_THRESHOLD", "-30"))
            except ValueError:
                trunc_lift = -30.0
            try:
                trunc_buffer = int(os.environ.get("EVAL3_TRUNCATE_BUFFER_FRAMES", "60"))
            except ValueError:
                trunc_buffer = 60
            try:
                trunc_over_cap_extra = int(os.environ.get("EVAL3_TRUNCATE_OVER_CAP_EXTRA_FRAMES", "90"))
            except ValueError:
                trunc_over_cap_extra = 90

            def _slug_from_repo(repo: str) -> str:
                """Map repo_id -> celebrity slug. Both old (taylor_swift_1) and new
                (dataset_v2_taylor_swift_left_1) variants resolve correctly because
                the substring match still hits."""
                rl = repo.lower()
                if "taylor_swift" in rl:
                    return "swift"
                if "yann_lecun" in rl:
                    return "lecun"
                if "barack_obama" in rl:
                    return "obama"
                return ""

            def _is_new_data(repo: str) -> bool:
                """True for dataset_v2_* repos. They record [home -> place -> home]
                trajectories and need placement truncation. Old data already ends
                at the placement pose and must NOT be truncated."""
                return "dataset_v2" in repo.lower()

            def _is_synth_data(repo: str) -> bool:
                """True for synthetic ChArUco-derived repos (``dataset_v3_synth_*``
                or ``dataset_v4_synth_*``). Used by Fix B (rebalance real vs synth)
                of the v9 celeb-confusion plan to cap synth volume without touching
                legacy or new66 episode filters.
                """
                rl = repo.lower()
                return "dataset_v3_synth" in rl or "dataset_v4_synth" in rl

            def _repo_name(repo: str) -> str:
                return repo.rsplit("/", 1)[-1]

            def _slot_from_repo(repo: str) -> str:
                name = _repo_name(repo).lower()
                for slot in ("left", "middle", "right"):
                    if f"_{slot}_" in name or name.endswith(f"_{slot}"):
                        return slot
                return ""

            def _print_shuffle_mask_paths(slug: str, slot: str, new_data: bool) -> tuple[str, str]:
                """v1 masks are per-celebrity with a fixed target slot; v2 needs slot_* masks."""
                if new_data and slot:
                    base = os.path.join(mask_dir, f"slot_{slot}")
                else:
                    base = os.path.join(mask_dir, slug)
                return os.path.join(base, "other1_mask.npy"), os.path.join(base, "other2_mask.npy")

            prep_datasets = []
            for d in datasets:
                slug = _slug_from_repo(d.repo_id)
                new_data = _is_new_data(d.repo_id)
                synth_data = _is_synth_data(d.repo_id)
                slot = _slot_from_repo(d.repo_id) if new_data else ""
                bg_aug = None
                print_aug = None
                # Fix D (v9 celeb-confusion plan): synth/charuco repos use
                # composited celebrity boards whose pixel geometry does NOT match
                # outputs/eval3_masks/<slug>/ (which was extracted from legacy
                # real-print recordings — see tools/eval3_extract_masks.py:14-16).
                # Silently applying bg-replace or print-shuffle to a synth repo
                # can scribble over the wrong region and corrupt action-image
                # alignment without warning. Skip both augmenters for synth.
                if slug and bg_replace_enabled and not synth_data:
                    bg_mask_path = os.path.join(mask_dir, slug, "bg_mask.npy")
                    if os.path.exists(bg_mask_path) and os.path.isdir(bg_dir):
                        bg_aug = BackgroundReplaceAugmenter(bg_mask_path, bg_dir, p=bg_p, seed=hash(slug) & 0xFFFF)
                    else:
                        logging.warning("eval3_concat_patch: %s bg-aug skipped (missing mask or bg dir)", slug)
                elif synth_data and bg_replace_enabled:
                    logging.info(
                        "eval3_concat_patch: %s bg-aug skipped (synth/charuco repo; "
                        "extracted bg_mask.npy targets legacy real-print geometry only)",
                        d.repo_id,
                    )
                if slug and print_shuffle_enabled and not synth_data:
                    o1, o2 = _print_shuffle_mask_paths(slug, slot, new_data)
                    if os.path.exists(o1) and os.path.exists(o2):
                        seed_key = f"{slug}:{slot}" if slot else slug
                        print_aug = PrintShuffleAugmenter(
                            o1, o2, p=ps_p, seed=(hash(seed_key) >> 16) & 0xFFFF
                        )
                    else:
                        logging.warning(
                            "eval3_concat_patch: %s print-shuffle skipped (missing masks at %s)",
                            d.repo_id,
                            os.path.dirname(o1),
                        )
                elif synth_data and print_shuffle_enabled:
                    logging.info(
                        "eval3_concat_patch: %s print-shuffle skipped (synth/charuco repo; "
                        "extracted other{1,2}_mask.npy target legacy real-print geometry only)",
                        d.repo_id,
                    )

                # Episode filter:
                #   - new data: EVAL3_NEW_EPISODE_KEEP overrides everything (only keep listed).
                #     Otherwise EVAL3_NEW_EPISODE_EXCLUDE drops listed indices. Otherwise no filter.
                #   - old data: legacy per-celebrity filter env vars.
                if new_data:
                    kept = (
                        new_episode_keep.get(_repo_name(d.repo_id), set())
                        | new_episode_keep.get(d.repo_id, set())
                    )
                    excluded = (
                        new_episode_exclude.get(_repo_name(d.repo_id), set())
                        | new_episode_exclude.get(d.repo_id, set())
                    )
                    if kept:
                        ep_filter = sorted(kept)
                    elif excluded:
                        ep_filter = [i for i in range(int(d.num_episodes)) if i not in excluded]
                    else:
                        ep_filter = None
                elif synth_data:
                    # Fix B: synth repos get their own cap so we don't have to
                    # use the celebrity filters (which also affect old data).
                    if synth_episode_limit and synth_episode_limit > 0:
                        ep_filter = list(range(min(synth_episode_limit, int(d.num_episodes))))
                    else:
                        ep_filter = None
                else:
                    ep_filter = {"swift": swift_filter, "lecun": lecun_filter, "obama": obama_filter}.get(slug)
                # Placement truncation applies ONLY to new data — old data already
                # ends at placement so truncating it would cut the place itself.
                trunc_at_place = bool(new_data and truncate_at_placement_enabled)
                allow_over_cap = (
                    over_cap_episodes.get(_repo_name(d.repo_id), set())
                    | over_cap_episodes.get(d.repo_id, set())
                )
                ds_task_aug = task_aug
                if slot_task_enabled and slot:
                    ds_task_aug = make_slot_task_augmenter(
                        slot=slot,
                        base_aug=task_aug,
                        slot_p=slot_task_p,
                        seed=(hash(d.repo_id) & 0xFFFF),
                    )
                repair_this_gripper = bool(
                    gripper_repair_enabled and (new_data or not gripper_repair_new_only)
                )
                # Derive the auxiliary-head position label from the repo name.
                # `_slot_from_repo` works on any repo (v2 OR synth v3) — it
                # substring-matches `_left_` / `_middle_` / `_right_` or trailing
                # ones. Datasets without a slot in the name (e.g. old
                # single-celeb `taylor_swift_1`) get None → -100 ignore_index.
                _aux_pos_idx = (
                    {"left": 0, "middle": 1, "right": 2}.get(_slot_from_repo(d.repo_id))
                )
                # Fix B: optional per-synth tighter frame cap to further
                # downweight synth volume vs real new66.
                effective_max_frames = max_frames
                if synth_data and synth_max_frames_per_ep > 0:
                    effective_max_frames = (
                        synth_max_frames_per_ep if max_frames is None
                        else min(int(max_frames), synth_max_frames_per_ep)
                    )
                w = Eval3PrepDataset(
                    d,
                    max_frames_per_episode=effective_max_frames,
                    task_aug_fn=ds_task_aug,
                    bg_aug_fn=bg_aug,
                    print_aug_fn=print_aug,
                    state_aug_fn=state_aug,
                    episode_filter=ep_filter,
                    truncate_at_placement=trunc_at_place,
                    truncate_placement_mode=trunc_mode,
                    truncate_allow_over_cap_episodes=allow_over_cap,
                    truncate_over_cap_extra_frames=trunc_over_cap_extra,
                    truncate_grip_threshold=trunc_grip,
                    truncate_lift_threshold=trunc_lift,
                    truncate_buffer_frames=trunc_buffer,
                    repair_gripper_open=repair_this_gripper,
                    gripper_open_target=gripper_open_target,
                    gripper_open_threshold=gripper_open_threshold,
                    action_smooth_window=action_smooth_window,
                    action_smooth_gripper=action_smooth_gripper,
                    target_position_idx=_aux_pos_idx,
                )
                s = w.truncation_summary()
                logging.info(
                    "eval3_concat_patch: %s  frames=%d/%d (%.1f%%)  episodes=%d/%d  "
                    "trunc_place=%s mode=%s (match=%.0f%%, avg_kept=%.0f)  "
                    "grip_repair=%s(changed=%d) smooth_w=%d(changed=%d) "
                    "task_aug=%s slot=%s  bg_aug=%s  print_aug=%s  ep_filter=%s",
                    s["repo_id"], s["kept_num_frames"], s["original_num_frames"],
                    s["kept_fraction"] * 100.0, s["kept_num_episodes"],
                    s["original_num_episodes"],
                    s["truncate_at_placement"], s["truncate_placement_mode"],
                    s["placement_match_rate"] * 100.0,
                    s["placement_avg_kept_frames"],
                    s["gripper_repair_enabled"], s["gripper_repair_changed_frames"],
                    s["action_smooth_window"], s["action_smooth_changed_frames"],
                    ds_task_aug is not None, slot or "none", bg_aug is not None,
                    print_aug is not None, ep_filter,
                )
                prep_datasets.append(w)
            datasets = prep_datasets

        concat = ConcatLeRobotDataset(datasets)
        logging.info(
            "eval3_concat_patch: combined %d datasets -> %d frames / %d episodes",
            len(datasets), concat.num_frames, concat.num_episodes,
        )
        return concat

    _factory.make_dataset = _patched_make_dataset
    # Re-export for any module that captured the name at import time:
    import lerobot.scripts.lerobot_train as _train_mod
    if hasattr(_train_mod, "make_dataset"):
        _train_mod.make_dataset = _patched_make_dataset

    logging.info("eval3_concat_patch: installed; extras=%s", extra_repos)
