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
3. Wraps them all in a ``ConcatLeRobotDataset`` whose ``__getitem__``/``__len__`` chain through
   ``torch.utils.data.ConcatDataset``, and whose ``.meta`` is a clone of the primary's meta
   with stats merged across ALL datasets (count-weighted mean, derived combined std, element-
   wise min/max, weighted quantile approximation).

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

        # Layer 1+2: per-dataset Eval3 prep (episode truncation + task-string augmentation).
        # Both are env-var-driven so we can ablate independently:
        #   EVAL3_MAX_FRAMES_PER_EP=600    # cap each episode at first 600 frames (= 20s @ 30fps)
        #                                  # Set to "0" or a huge number to disable truncation.
        #   EVAL3_TASK_AUG=1               # enable task-string augmentation (default on)
        try:
            from eval3_dataset_prep import Eval3PrepDataset, make_task_augmenter
        except ImportError as e:
            logging.warning("eval3_concat_patch: could not import eval3_dataset_prep (%s); "
                            "skipping Layer 1+2 prep.", e)
        else:
            max_frames_raw = os.environ.get("EVAL3_MAX_FRAMES_PER_EP", "600").strip()
            try:
                max_frames = int(max_frames_raw)
            except ValueError:
                max_frames = 600
            if max_frames <= 0:
                max_frames = None  # disabled
            task_aug = make_task_augmenter() if os.environ.get("EVAL3_TASK_AUG", "1") == "1" else None
            prep_datasets = []
            for d in datasets:
                w = Eval3PrepDataset(d, max_frames_per_episode=max_frames, task_aug_fn=task_aug)
                s = w.truncation_summary()
                logging.info(
                    "eval3_concat_patch: %s  before=%d  after=%d  kept=%.1f%%  task_aug=%s",
                    s["repo_id"], s["original_num_frames"], s["kept_num_frames"],
                    s["kept_fraction"] * 100.0, task_aug is not None,
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
