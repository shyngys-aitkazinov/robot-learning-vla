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

import random
from typing import Any, Callable

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
    ):
        self._ds = dataset
        self._task_aug_fn = task_aug_fn

        # Pull episode boundaries from the underlying meta. LeRobotDatasetMetadata.episodes
        # is a pandas DataFrame with columns ``dataset_from_index`` and ``dataset_to_index``.
        ep_df = dataset.meta.episodes
        from_idxs = list(ep_df["dataset_from_index"])
        to_idxs = list(ep_df["dataset_to_index"])

        # Build the per-episode kept-range. If max_frames_per_episode is None or larger
        # than the longest episode, this is a no-op (kept range == full episode).
        valid: list[int] = []
        original_total = 0
        for f0, f1 in zip(from_idxs, to_idxs):
            f0i, f1i = int(f0), int(f1)
            original_total += f1i - f0i
            if max_frames_per_episode is None:
                cap = f1i
            else:
                cap = min(f0i + int(max_frames_per_episode), f1i)
            valid.extend(range(f0i, cap))
        self._valid_indices = valid
        self._max_frames_per_episode = max_frames_per_episode
        self._original_num_frames = original_total

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
        if self._task_aug_fn is not None and "task" in row:
            row = dict(row)  # don't mutate the underlying dataset's cached row
            row["task"] = self._task_aug_fn(row["task"])
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
