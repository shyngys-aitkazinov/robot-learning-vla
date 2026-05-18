#!/usr/bin/env python3
"""Shared exact-data adapter for Eval 3 external VLA runs.

This module is intentionally separate from the SmolVLA training patch. Runs 7
and 8 need the exact new66 / new+old88 labels and frame inclusion:

* no v8 gripper-open repair
* no arm-label smoothing
* no extra 600-frame cap

The adapter reads LeRobot Hub datasets, returns one front camera frame, padded
7-D proprio/action tensors, and action chunks suitable for FlowerVLA or
OpenVLA-style trainers.
"""

from __future__ import annotations

import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

try:
    from eval3_lerobot_shim import apply as _eval3_shim_apply
except ImportError:  # pragma: no cover - supports package-style imports
    from scripts.eval3_lerobot_shim import apply as _eval3_shim_apply


IMAGE_KEY = "observation.images.front"
STATE_KEY = "observation.state"
ACTION_KEY = "action"
PADDED_DIM = 7
DEFAULT_CHUNK_SIZE = 10
ACTION_NAMES_6 = (
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
)
ACTION_NAMES_7 = (*ACTION_NAMES_6, "padding.zero")


NEW66_REPOS = (
    "RobotLearningVLA/dataset_v2_taylor_swift_left_1_v6_truncated",
    "RobotLearningVLA/dataset_v2_taylor_swift_middle_1_v6_truncated",
    "RobotLearningVLA/dataset_v2_taylor_swift_right_1_v6_truncated",
    "RobotLearningVLA/dataset_v2_yann_lecun_left_1_v6_truncated",
    "RobotLearningVLA/dataset_v2_yann_lecun_middle_1_v6_truncated",
    "RobotLearningVLA/dataset_v2_yann_lecun_right_1_v6_truncated",
    "RobotLearningVLA/dataset_v2_barack_obama_left_1_v6_truncated",
    "RobotLearningVLA/dataset_v2_barack_obama_middle_1_v6_truncated",
    "RobotLearningVLA/dataset_v2_barack_obama_right_1_v6_truncated",
)

OLD_FILTERS = {
    "RobotLearningVLA/taylor_swift_1": (0, 4, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19),
    "RobotLearningVLA/yann_lecun_1": (3, 7, 9, 13),
    "RobotLearningVLA/barack_obama_1": (5, 6, 8, 17),
}


@dataclass(frozen=True)
class RecipeSource:
    repo_id: str
    episodes: tuple[int, ...] | None = None
    split: Literal["new66", "old_filtered"] = "new66"


@dataclass(frozen=True)
class ExternalVLARecipe:
    name: str
    job_ids: dict[str, str]
    sources: tuple[RecipeSource, ...]
    expected_frames: int
    expected_episodes: int
    unnorm_key: str


RECIPES = {
    "new66": ExternalVLARecipe(
        name="new66",
        job_ids={
            "flower": "eval3-flower-new66-50k",
            "openvla": "eval3-openvla-lora-new66-50k",
        },
        sources=tuple(RecipeSource(repo_id=r) for r in NEW66_REPOS),
        expected_frames=36_865,
        expected_episodes=66,
        unnorm_key="eval3_so101_new66",
    ),
    "new_old88": ExternalVLARecipe(
        name="new_old88",
        job_ids={
            "flower": "eval3-flower-new-old88-50k",
            "openvla": "eval3-openvla-lora-new-old88-50k",
        },
        sources=tuple(RecipeSource(repo_id=r) for r in NEW66_REPOS)
        + tuple(RecipeSource(repo_id=r, episodes=eps, split="old_filtered") for r, eps in OLD_FILTERS.items()),
        expected_frames=52_546,
        expected_episodes=88,
        unnorm_key="eval3_so101_new_old88",
    ),
}


def get_recipe(name: str) -> ExternalVLARecipe:
    key = str(name).strip()
    if key not in RECIPES:
        raise KeyError(f"Unknown Eval3 external VLA recipe {name!r}. Options: {sorted(RECIPES)}")
    return RECIPES[key]


def canonicalize_task(task: str) -> str:
    """Normalize recorded Eval3 wording to the hardware prompt wording."""
    text = str(task).strip()
    replacements = {
        "Place the coke on the Taylor Swift": "Place the coke on Taylor Swift",
        "Place the coke on the Yann LeCun": "Place the coke on Yann LeCun",
        "Place the coke on the Barack Obama": "Place the coke on Barack Obama",
    }
    return replacements.get(text, text)


class Eval3TaskText:
    """Picklable Eval3 task string mutator.

    ``mixed`` mirrors the new66 SmolVLA path: mostly canonical hardware wording,
    sometimes original recorded wording. It changes only text conditioning, not
    action labels or frame inclusion.
    """

    def __init__(self, mode: Literal["raw", "canonical", "mixed"] = "mixed", canonical_p: float = 0.8, seed: int = 42):
        self.mode = mode
        self.canonical_p = min(max(float(canonical_p), 0.0), 1.0)
        self.rng = random.Random(int(seed))

    def __call__(self, task: str) -> str:
        if self.mode == "raw":
            return str(task)
        if self.mode == "canonical":
            return canonicalize_task(task)
        if self.mode == "mixed":
            return canonicalize_task(task) if self.rng.random() < self.canonical_p else str(task)
        raise ValueError(f"Unsupported task mode {self.mode!r}")

    def __reduce__(self):
        return (self.__class__, (self.mode, self.canonical_p))


def _load_lerobot_dataset(source: RecipeSource, *, revision: str | None, video_backend: str, download_videos: bool):
    _eval3_shim_apply()
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    return LeRobotDataset(
        source.repo_id,
        episodes=list(source.episodes) if source.episodes is not None else None,
        revision=revision,
        video_backend=video_backend,
        download_videos=download_videos,
    )


def _episode_end_indices(episode_indices: Iterable[Any]) -> list[int]:
    eps = [int(torch.as_tensor(x).item()) if hasattr(x, "__len__") and not isinstance(x, (str, bytes)) else int(x) for x in episode_indices]
    out = [0] * len(eps)
    start = 0
    while start < len(eps):
        end = start + 1
        while end < len(eps) and eps[end] == eps[start]:
            end += 1
        for i in range(start, end):
            out[i] = end
        start = end
    return out


def _pad_last_dim(x: torch.Tensor, *, target_dim: int = PADDED_DIM) -> torch.Tensor:
    x = torch.as_tensor(x, dtype=torch.float32)
    if x.shape[-1] > target_dim:
        raise ValueError(f"Cannot pad tensor with last dim {x.shape[-1]} down to {target_dim}")
    if x.shape[-1] == target_dim:
        return x
    pad_shape = (*x.shape[:-1], target_dim - x.shape[-1])
    return torch.cat([x, torch.zeros(pad_shape, dtype=x.dtype, device=x.device)], dim=-1)


def _column_to_tensor(values: list[Any]) -> torch.Tensor:
    if values and any(isinstance(v, torch.Tensor) for v in values):
        return torch.stack([torch.as_tensor(v, dtype=torch.float32) for v in values], dim=0)
    return torch.tensor(values, dtype=torch.float32)


def _resize_chw(image: torch.Tensor, image_size: int | None) -> torch.Tensor:
    x = torch.as_tensor(image)
    if x.ndim != 3:
        raise ValueError(f"Expected CHW image tensor, got shape {tuple(x.shape)}")
    if x.dtype != torch.float32:
        x = x.float()
    if float(x.max()) > 2.0:
        x = x / 255.0
    if image_size and (x.shape[-2] != image_size or x.shape[-1] != image_size):
        x = F.interpolate(x.unsqueeze(0), size=(int(image_size), int(image_size)), mode="bilinear", align_corners=False)[0]
    return x.clamp(0.0, 1.0)


def tensor_chw_to_pil(image: torch.Tensor):
    from PIL import Image

    x = _resize_chw(image, image_size=None).detach().cpu()
    arr = (x.permute(1, 2, 0).numpy() * 255.0).round().clip(0, 255).astype("uint8")
    return Image.fromarray(arr)


class Eval3ExternalVLADataset(Dataset):
    """Map-style dataset for exact Eval3 external VLA training."""

    def __init__(
        self,
        recipe: str | ExternalVLARecipe = "new66",
        *,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        image_size: int | None = 224,
        task_mode: Literal["raw", "canonical", "mixed"] = "mixed",
        task_canonical_p: float = 0.8,
        seed: int = 42,
        revision: str | None = None,
        video_backend: str = "pyav",
        download_videos: bool = True,
    ):
        self.recipe = get_recipe(recipe) if isinstance(recipe, str) else recipe
        self.chunk_size = int(chunk_size)
        self.image_size = image_size
        self.task_text = Eval3TaskText(task_mode, task_canonical_p, seed)
        self.revision = revision
        self.video_backend = video_backend
        self.download_videos = bool(download_videos)

        self.datasets = [
            _load_lerobot_dataset(src, revision=revision, video_backend=video_backend, download_videos=download_videos)
            for src in self.recipe.sources
        ]
        self.actions = [_column_to_tensor(ds.hf_dataset[ACTION_KEY]) for ds in self.datasets]
        self.states = [_column_to_tensor(ds.hf_dataset[STATE_KEY]) for ds in self.datasets]
        self.episode_ends = [_episode_end_indices(ds.hf_dataset["episode_index"]) for ds in self.datasets]

        self.records: list[tuple[int, int]] = []
        for ds_idx, ds in enumerate(self.datasets):
            self.records.extend((ds_idx, local_idx) for local_idx in range(len(ds)))

        self._validate_schema()

    def _validate_schema(self) -> None:
        for ds in self.datasets:
            features = getattr(ds.meta, "features", {})
            if IMAGE_KEY not in features:
                raise ValueError(f"{ds.repo_id} is missing {IMAGE_KEY}")
            for key in (STATE_KEY, ACTION_KEY):
                if key not in features:
                    raise ValueError(f"{ds.repo_id} is missing {key}")
                shape = tuple(features[key].get("shape", ()))
                if shape != (6,):
                    raise ValueError(f"{ds.repo_id} {key} shape must be (6,), got {shape}")
            names = tuple(features[ACTION_KEY].get("names") or ())
            if names and names != ACTION_NAMES_6:
                raise ValueError(f"{ds.repo_id} action names changed: {names}")

    @property
    def num_frames(self) -> int:
        return len(self.records)

    @property
    def num_episodes(self) -> int:
        return sum(int(ds.num_episodes) for ds in self.datasets)

    def __len__(self) -> int:
        return len(self.records)

    def _action_chunk(self, ds_idx: int, local_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        actions = self.actions[ds_idx]
        episode_end = self.episode_ends[ds_idx][local_idx]
        chunk = []
        valid_mask = []
        for offset in range(self.chunk_size):
            src_idx = local_idx + offset
            valid = src_idx < episode_end
            if not valid:
                src_idx = episode_end - 1
            chunk.append(_pad_last_dim(actions[src_idx]))
            valid_mask.append(valid)
        return torch.stack(chunk, dim=0), torch.tensor(valid_mask, dtype=torch.bool)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        ds_idx, local_idx = self.records[int(idx)]
        ds = self.datasets[ds_idx]
        row = ds[local_idx]
        actions, action_valid = self._action_chunk(ds_idx, local_idx)
        state = _pad_last_dim(torch.as_tensor(row[STATE_KEY], dtype=torch.float32))
        image = _resize_chw(row[IMAGE_KEY], self.image_size)
        task = self.task_text(row.get("task", "Place the coke on Taylor Swift"))

        return {
            "image": image,
            "state": state,
            "actions": actions,
            "action_valid": action_valid,
            "task": task,
            "repo_id": ds.repo_id,
            "recipe": self.recipe.name,
            "episode_index": int(torch.as_tensor(row["episode_index"]).item()),
            "frame_index": int(torch.as_tensor(row["frame_index"]).item()),
            "index": int(torch.as_tensor(row["index"]).item()),
            "local_index": int(local_idx),
            "source_index": int(ds_idx),
        }


def collate_external_vla(items: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "images": torch.stack([x["image"] for x in items], dim=0),
        "states": torch.stack([x["state"] for x in items], dim=0),
        "actions": torch.stack([x["actions"] for x in items], dim=0),
        "action_valid": torch.stack([x["action_valid"] for x in items], dim=0),
        "tasks": [x["task"] for x in items],
        "repo_ids": [x["repo_id"] for x in items],
        "recipe": items[0]["recipe"] if items else "",
        "episode_indices": torch.tensor([x["episode_index"] for x in items], dtype=torch.long),
        "frame_indices": torch.tensor([x["frame_index"] for x in items], dtype=torch.long),
        "indices": torch.tensor([x["index"] for x in items], dtype=torch.long),
    }


def _stats(values: torch.Tensor) -> dict[str, list[float]]:
    x = torch.as_tensor(values, dtype=torch.float32)
    return {
        "mean": x.mean(dim=0).tolist(),
        "std": x.std(dim=0, unbiased=False).tolist(),
        "max": x.amax(dim=0).tolist(),
        "min": x.amin(dim=0).tolist(),
        "q01": x.quantile(0.01, dim=0).tolist(),
        "q99": x.quantile(0.99, dim=0).tolist(),
    }


def compute_recipe_statistics(
    recipe: str | ExternalVLARecipe,
    *,
    revision: str | None = None,
    video_backend: str = "pyav",
) -> dict[str, Any]:
    """Compute OpenVLA-style action/proprio stats from exact raw labels."""
    recipe_obj = get_recipe(recipe) if isinstance(recipe, str) else recipe
    actions = []
    states = []
    num_trajectories = 0
    per_source = []
    for source in recipe_obj.sources:
        ds = _load_lerobot_dataset(source, revision=revision, video_backend=video_backend, download_videos=False)
        a = _pad_last_dim(_column_to_tensor(ds.hf_dataset[ACTION_KEY]))
        s = _pad_last_dim(_column_to_tensor(ds.hf_dataset[STATE_KEY]))
        actions.append(a)
        states.append(s)
        num_trajectories += int(ds.num_episodes)
        per_source.append({
            "repo_id": source.repo_id,
            "split": source.split,
            "episodes": list(source.episodes) if source.episodes is not None else None,
            "num_frames": int(ds.num_frames),
            "num_episodes": int(ds.num_episodes),
        })
    action_t = torch.cat(actions, dim=0)
    state_t = torch.cat(states, dim=0)
    action_stats = _stats(action_t)
    proprio_stats = _stats(state_t)
    action_stats["mask"] = [True, True, True, True, True, True, False]
    proprio_stats["mask"] = [True, True, True, True, True, True, False]
    return {
        "recipe": recipe_obj.name,
        "unnorm_key": recipe_obj.unnorm_key,
        "action": action_stats,
        "proprio": proprio_stats,
        "num_transitions": int(action_t.shape[0]),
        "num_trajectories": int(num_trajectories),
        "action_names": list(ACTION_NAMES_7),
        "sources": per_source,
    }


def compute_openvla_chunk_statistics(
    recipe: str | ExternalVLARecipe,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    revision: str | None = None,
    video_backend: str = "pyav",
) -> dict[str, Any]:
    """Compute OpenVLA stats for flattened action chunks.

    Upstream OpenVLA chooses generation length from
    ``len(norm_stats[unnorm_key]["action"]["q01"])``. For chunked Eval3 runs we
    therefore store 10x7 actions as a flattened 70-D vector. Inference reshapes
    back to ``[chunk_size, 7]`` and strips the seventh padded channel before
    sending commands to SO-101.
    """
    recipe_obj = get_recipe(recipe) if isinstance(recipe, str) else recipe
    ds = Eval3ExternalVLADataset(
        recipe_obj,
        chunk_size=chunk_size,
        image_size=None,
        task_mode="canonical",
        revision=revision,
        video_backend=video_backend,
        download_videos=False,
    )
    chunks = []
    for ds_idx, local_idx in ds.records:
        chunk, _ = ds._action_chunk(ds_idx, local_idx)
        chunks.append(chunk.reshape(-1))
    action_t = torch.stack(chunks, dim=0)
    base_stats = compute_recipe_statistics(recipe_obj, revision=revision, video_backend=video_backend)
    action_stats = _stats(action_t)
    action_stats["mask"] = [(i % PADDED_DIM) != (PADDED_DIM - 1) for i in range(chunk_size * PADDED_DIM)]
    return {
        "recipe": recipe_obj.name,
        "unnorm_key": recipe_obj.unnorm_key,
        "action": action_stats,
        "proprio": base_stats["proprio"],
        "num_transitions": int(action_t.shape[0]),
        "num_trajectories": int(base_stats["num_trajectories"]),
        "action_names": [f"t{t}.{name}" for t in range(chunk_size) for name in ACTION_NAMES_7],
        "chunk_size": int(chunk_size),
        "action_step_dim": PADDED_DIM,
        "action_shape": [int(chunk_size), PADDED_DIM],
        "sources": base_stats["sources"],
    }


def normalize_q01_q99(actions: torch.Tensor, stats: dict[str, Any], *, clip: bool = True) -> torch.Tensor:
    """Normalize raw degree actions to OpenVLA's expected [-1, 1] token range."""
    x = torch.as_tensor(actions, dtype=torch.float32)
    q01 = torch.tensor(stats["action"]["q01"], dtype=x.dtype, device=x.device)
    q99 = torch.tensor(stats["action"]["q99"], dtype=x.dtype, device=x.device)
    mask = torch.tensor(stats["action"].get("mask", [True] * x.shape[-1]), dtype=torch.bool, device=x.device)
    denom = (q99 - q01).clamp_min(1e-6)
    out = 2.0 * (x - q01) / denom - 1.0
    out = torch.where(mask, out, torch.zeros_like(out))
    return out.clamp(-1.0, 1.0) if clip else out


def denormalize_q01_q99(actions: torch.Tensor, stats: dict[str, Any]) -> torch.Tensor:
    """Invert ``normalize_q01_q99`` and return raw SO-101 degree-space actions."""
    x = torch.as_tensor(actions, dtype=torch.float32)
    q01 = torch.tensor(stats["action"]["q01"], dtype=x.dtype, device=x.device)
    q99 = torch.tensor(stats["action"]["q99"], dtype=x.dtype, device=x.device)
    mask = torch.tensor(stats["action"].get("mask", [True] * x.shape[-1]), dtype=torch.bool, device=x.device)
    out = (x + 1.0) * 0.5 * (q99 - q01) + q01
    return torch.where(mask, out, torch.zeros_like(out))


def write_openvla_dataset_statistics(
    recipes: Iterable[str | ExternalVLARecipe],
    out_path: str | Path,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    revision: str | None = None,
    video_backend: str = "pyav",
) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for recipe in recipes:
        stats = compute_openvla_chunk_statistics(
            recipe,
            chunk_size=chunk_size,
            revision=revision,
            video_backend=video_backend,
        )
        key = stats.pop("unnorm_key")
        payload[key] = stats
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def recipe_manifest(recipe: str | ExternalVLARecipe, *, revision: str | None = None, video_backend: str = "pyav") -> dict[str, Any]:
    stats = compute_recipe_statistics(recipe, revision=revision, video_backend=video_backend)
    expected = get_recipe(recipe).expected_frames if isinstance(recipe, str) else recipe.expected_frames
    return {
        "recipe": stats["recipe"],
        "unnorm_key": stats["unnorm_key"],
        "expected_frames": expected,
        "num_frames": stats["num_transitions"],
        "expected_episodes": get_recipe(stats["recipe"]).expected_episodes,
        "num_episodes": stats["num_trajectories"],
        "action_names": stats["action_names"],
        "sources": stats["sources"],
        "action_q01": stats["action"]["q01"],
        "action_q99": stats["action"]["q99"],
    }
