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
import os
import random
from copy import deepcopy
from typing import Any, Callable

import numpy as np
import torch
from torch.utils.data import Dataset


# Substring-match list. If the recorded ``task`` string contains one of these names,
# we know which celebrity to augment around. Any task string without a known
# celebrity name is passed through unchanged (defensive).
KNOWN_CELEBRITIES = ("Taylor Swift", "Yann LeCun", "Barack Obama")


class TaskAugmenter:
    """Picklable callable that randomly rewrites Eval 3 task strings.

    Implemented as a class (not a closure) because PyTorch DataLoader with
    ``num_workers > 0`` pickles the dataset object — including any callable
    attributes — to ship to worker processes. Local closures from
    ``make_task_augmenter.<locals>.aug`` are not picklable.

    Variant probabilities intentionally stay close to demo-day wording:
      80 %  "Place the coke on <Celeb>"           (canonical demo prompt)
      20 %  "Place the coke on the <Celeb>"       (original recorded wording)
    """

    def __init__(self, seed: int = 42, canonical_p: float = 0.8):
        self._seed = seed
        self._canonical_p = float(canonical_p)
        # Use a Python random.Random with the supplied seed. Each forked worker
        # will get its own copy after pickling — that's fine for augmentation.
        self._rng = random.Random(seed)

    def __call__(self, task: str) -> str:
        if not isinstance(task, str):
            return task
        for celeb in KNOWN_CELEBRITIES:
            if celeb in task:
                roll = self._rng.random()
                if roll < self._canonical_p:
                    return f"Place the coke on {celeb}"
                else:
                    return f"Place the coke on the {celeb}"
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
    if values and isinstance(values[0], torch.Tensor):
        return torch.stack([v.detach().float() for v in values])
    return torch.as_tensor(values, dtype=torch.float32)


def _compute_filtered_stats(dataset, valid_indices: list[int]) -> dict[str, dict[str, torch.Tensor]]:
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
        values = [col[i] for i in valid_indices]
        if not values:
            continue
        stats[key] = _stat_tensor(_column_values_to_tensor(values), stats.get(key))
    return stats


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
    """

    def __init__(
        self,
        dataset,
        max_frames_per_episode: int | None = 600,
        task_aug_fn: Callable[[str], str] | None = None,
        bg_aug_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
        print_aug_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
        image_key: str = "observation.images.front",
        episode_filter: list[int] | None = None,
    ):
        self._ds = dataset
        self._task_aug_fn = task_aug_fn
        self._bg_aug_fn = bg_aug_fn
        self._print_aug_fn = print_aug_fn
        self._image_key = image_key

        # Pull episode boundaries from the underlying meta. LeRobotDatasetMetadata.episodes
        # is a pandas DataFrame with columns ``dataset_from_index`` and ``dataset_to_index``.
        ep_df = dataset.meta.episodes
        from_idxs = list(ep_df["dataset_from_index"])
        to_idxs = list(ep_df["dataset_to_index"])

        # Build the per-episode kept-range. If max_frames_per_episode is None or larger
        # than the longest episode, this is a no-op (kept range == full episode).
        # If episode_filter is provided, only include those episode indices.
        valid: list[int] = []
        original_total = 0
        keep_eps = set(int(e) for e in episode_filter) if episode_filter is not None else None
        for ep_idx, (f0, f1) in enumerate(zip(from_idxs, to_idxs)):
            f0i, f1i = int(f0), int(f1)
            original_total += f1i - f0i
            if keep_eps is not None and ep_idx not in keep_eps:
                continue
            if max_frames_per_episode is None:
                cap = f1i
            else:
                cap = min(f0i + int(max_frames_per_episode), f1i)
            valid.extend(range(f0i, cap))
        self._valid_indices = valid
        self._max_frames_per_episode = max_frames_per_episode
        self._original_num_frames = original_total
        self._episode_filter = list(keep_eps) if keep_eps is not None else None
        self._kept_episode_indices = sorted(
            set(int(dataset.hf_dataset["episode_index"][i]) for i in self._valid_indices)
        )
        self._meta = deepcopy(dataset.meta)
        self._meta.stats = _compute_filtered_stats(dataset, self._valid_indices)

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

    def __getitem__(self, idx) -> Any:
        original_idx = self._valid_indices[int(idx)]
        row = self._ds[original_idx]
        mutated = False
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
        return {
            "repo_id": self.repo_id,
            "max_frames_per_episode": self._max_frames_per_episode,
            "original_num_frames": int(self._original_num_frames),
            "kept_num_frames": int(self.num_frames),
            "dropped_num_frames": int(self._original_num_frames - self.num_frames),
            "kept_fraction": float(self.num_frames) / float(self._original_num_frames)
            if self._original_num_frames else 0.0,
            "original_num_episodes": int(self._ds.num_episodes),
            "kept_num_episodes": int(self.num_episodes),
            "kept_episode_indices": list(self._kept_episode_indices),
            "action_stat_count": float(torch.as_tensor(self.meta.stats["action"]["count"]).flatten()[0])
            if "action" in self.meta.stats and "count" in self.meta.stats["action"]
            else None,
        }
