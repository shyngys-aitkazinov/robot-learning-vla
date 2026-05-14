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

    Variant probabilities (must keep canonical-demo wording dominant since that's
    the wording the TA will type):
      40 %  "Place the coke on <Celeb>"           (drop "the" — canonical demo)
      15 %  "Place the coke on the <Celeb>"       (original recorded wording)
      15 %  "Put the coke on <Celeb>"             (verb swap)
      10 %  "Place the can on <Celeb>"            (noun swap)
      10 %  "Put the can on top of <Celeb>"
      10 %  "Place the can on top of <Celeb>"
    """

    def __init__(self, seed: int = 42):
        self._seed = seed
        # Use a Python random.Random with the supplied seed. Each forked worker
        # will get its own copy after pickling — that's fine for augmentation.
        self._rng = random.Random(seed)

    def __call__(self, task: str) -> str:
        if not isinstance(task, str):
            return task
        for celeb in KNOWN_CELEBRITIES:
            if celeb in task:
                roll = self._rng.random()
                if roll < 0.40:
                    return f"Place the coke on {celeb}"
                elif roll < 0.55:
                    return f"Place the coke on the {celeb}"
                elif roll < 0.70:
                    return f"Put the coke on {celeb}"
                elif roll < 0.80:
                    return f"Place the can on {celeb}"
                elif roll < 0.90:
                    return f"Put the can on top of {celeb}"
                else:
                    return f"Place the can on top of {celeb}"
        return task  # unknown celebrity — leave untouched

    def __reduce__(self):
        # When DataLoader forks a worker, this returns a constructor + args
        # so a fresh TaskAugmenter is built in the child process.
        return (self.__class__, (self._seed,))


def make_task_augmenter(seed: int = 42) -> "TaskAugmenter":
    """Backward-compatible factory returning a TaskAugmenter instance."""
    return TaskAugmenter(seed=seed)


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

    # ----- Trainer-required attributes ------------------------------------

    @property
    def meta(self):
        return self._ds.meta

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
        return self._ds.num_episodes

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
        }
