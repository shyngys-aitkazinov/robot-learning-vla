#!/usr/bin/env python3
"""Materialize Eval 3 v6 fixed dataset_v2 repos and optionally upload them.

The training wrapper can truncate raw dataset_v2 episodes on the fly. This tool
turns that same rule into real LeRobot datasets so they can be inspected,
loaded directly, and pushed to Hugging Face without modifying the raw repos.
"""
from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from huggingface_hub import HfApi

_REPO = Path(__file__).resolve().parent.parent
_SCRIPTS = _REPO / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval3_lerobot_shim import apply as _shim_apply  # noqa: E402

_shim_apply()

from eval3_export_truncated_videos import (  # noqa: E402
    DEFAULT_NEW_REPOS,
    DEFAULT_OVER_CAP_EPISODES,
    GRIPPER_IDX,
    LIFT_IDX,
    WRIST_ROLL_IDX,
    _action_at,
    _find_placement_match,
    _parse_repo_episode_map,
    _repo_slug,
    _tensor_image_to_pil,
)
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from lerobot.datasets.utils import DEFAULT_FEATURES  # noqa: E402


@dataclass
class EpisodeMaterializeInfo:
    source_repo: str
    dest_repo: str
    episode: int
    raw_frames: int
    kept_frames: int
    match_frame: int | None
    end_shoulder_lift: float
    end_gripper: float
    end_wrist_roll: float


@dataclass
class RepoMaterializeInfo:
    source_repo: str
    dest_repo: str
    root: str
    episodes: int
    frames: int
    uploaded: bool
    url: str
    episode_rows: list[EpisodeMaterializeInfo]


def _dest_repo_for(source_repo: str, dest_org: str, dest_suffix: str) -> str:
    return f"{dest_org}/{_repo_slug(source_repo)}{dest_suffix}"


def _feature_payload(features: dict[str, dict]) -> dict[str, dict]:
    return copy.deepcopy({k: v for k, v in features.items() if k not in DEFAULT_FEATURES})


def _to_numpy(value: Any, dtype: str | None = None) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        arr = value.detach().cpu().numpy()
    else:
        arr = np.asarray(value)
    if dtype is not None and arr.dtype != np.dtype(dtype):
        arr = arr.astype(np.dtype(dtype))
    return arr


def _frame_from_row(row: dict[str, Any], features: dict[str, dict]) -> dict[str, Any]:
    frame: dict[str, Any] = {}
    for key, ft in features.items():
        dtype = ft["dtype"]
        if dtype in ("image", "video"):
            frame[key] = _tensor_image_to_pil(row[key])
        else:
            frame[key] = _to_numpy(row[key], dtype=dtype)
    frame["task"] = row["task"]
    return frame


def _kept_frames_for_episode(
    actions: list[Any],
    f0: int,
    f1: int,
    episode_index: int,
    over_cap_eps: set[int],
    args: argparse.Namespace,
) -> tuple[int, int | None]:
    match_frame = _find_placement_match(
        actions,
        f0,
        f1,
        grip_threshold=args.grip_threshold,
        lift_threshold=args.lift_threshold,
        placement_mode=args.placement_mode,
    )
    raw_frames = f1 - f0
    if match_frame is None:
        kept_frames = min(args.max_frames_per_episode, raw_frames)
    elif args.placement_mode == "first":
        kept_frames = min(match_frame + args.buffer_frames, raw_frames)
    else:
        kept_frames = min(match_frame + 1, raw_frames)

    extra = args.over_cap_extra_frames if episode_index in over_cap_eps else 0
    kept_frames = min(kept_frames, args.max_frames_per_episode + extra)
    return kept_frames, match_frame


def _remove_local_root(root: Path, overwrite: bool) -> None:
    if not root.exists():
        return
    if not overwrite:
        raise FileExistsError(f"{root} already exists. Pass --overwrite-local to regenerate it.")
    shutil.rmtree(root)


def _assert_hub_destination_ok(dest_repo: str, allow_existing_hub: bool) -> None:
    if allow_existing_hub:
        return
    api = HfApi()
    if api.repo_exists(repo_id=dest_repo, repo_type="dataset"):
        raise RuntimeError(
            f"{dest_repo} already exists on the Hub. Pass --allow-existing-hub if you want to update it."
        )


def materialize_repo(args: argparse.Namespace, source_repo: str) -> RepoMaterializeInfo:
    dest_repo = _dest_repo_for(source_repo, args.dest_org, args.dest_suffix)
    if args.upload:
        _assert_hub_destination_ok(dest_repo, args.allow_existing_hub)

    source_slug = _repo_slug(source_repo)
    root = Path(args.out_dir) / _repo_slug(dest_repo)
    _remove_local_root(root, overwrite=args.overwrite_local)

    print(f"loading {source_repo}")
    src = LeRobotDataset(source_repo, video_backend=args.video_backend)
    features = _feature_payload(src.meta.features)
    over_cap_map = _parse_repo_episode_map(args.allow_over_cap_episodes)
    over_cap_eps = over_cap_map.get(source_slug, set()) | over_cap_map.get(source_repo, set())
    actions = src.hf_dataset["action"]
    ep_df = src.meta.episodes
    boundaries = [
        (int(f0), int(f1))
        for f0, f1 in zip(ep_df["dataset_from_index"], ep_df["dataset_to_index"])
    ]
    if args.limit_episodes is not None:
        boundaries = boundaries[: args.limit_episodes]

    print(f"creating {dest_repo} at {root}")
    dst = LeRobotDataset.create(
        repo_id=dest_repo,
        fps=int(src.meta.fps),
        features=features,
        root=root,
        robot_type=src.meta.info.get("robot_type"),
        use_videos=True,
        video_backend=args.video_backend,
        image_writer_threads=args.image_writer_threads,
        image_writer_processes=args.image_writer_processes,
        batch_encoding_size=args.batch_encoding_size,
        vcodec=args.vcodec,
        encoder_threads=args.encoder_threads,
    )

    episode_rows: list[EpisodeMaterializeInfo] = []
    total_frames = 0
    try:
        for ep_idx, (f0, f1) in enumerate(boundaries):
            kept_frames, match_frame = _kept_frames_for_episode(
                actions, f0, f1, ep_idx, over_cap_eps, args
            )
            print(f"  ep {ep_idx:03d}: raw={f1 - f0} kept={kept_frames} match={match_frame}")
            for frame_i in range(kept_frames):
                row = src[f0 + frame_i]
                dst.add_frame(_frame_from_row(row, features))
            dst.save_episode(parallel_encoding=args.parallel_encoding)

            end_action = _action_at(actions, f0 + kept_frames - 1)
            episode_rows.append(
                EpisodeMaterializeInfo(
                    source_repo=source_repo,
                    dest_repo=dest_repo,
                    episode=ep_idx,
                    raw_frames=f1 - f0,
                    kept_frames=kept_frames,
                    match_frame=match_frame,
                    end_shoulder_lift=end_action[LIFT_IDX],
                    end_gripper=end_action[GRIPPER_IDX],
                    end_wrist_roll=end_action[WRIST_ROLL_IDX],
                )
            )
            total_frames += kept_frames
    finally:
        dst.finalize()

    verify = LeRobotDataset(dest_repo, root=root, video_backend=args.video_backend)
    if verify.num_episodes != len(boundaries) or verify.num_frames != total_frames:
        raise RuntimeError(
            f"verification mismatch for {dest_repo}: "
            f"expected {len(boundaries)} eps/{total_frames} frames, "
            f"loaded {verify.num_episodes} eps/{verify.num_frames} frames"
        )

    uploaded = False
    if args.upload:
        print(f"uploading {dest_repo}")
        dst.push_to_hub(
            private=args.private,
            tag_version=True,
            upload_large_folder=args.upload_large_folder,
            tags=["lerobot", "eval3", "v6", "truncated"],
        )
        uploaded = True

    return RepoMaterializeInfo(
        source_repo=source_repo,
        dest_repo=dest_repo,
        root=str(root),
        episodes=len(boundaries),
        frames=total_frames,
        uploaded=uploaded,
        url=f"https://huggingface.co/datasets/{dest_repo}",
        episode_rows=episode_rows,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repos", default=",".join(DEFAULT_NEW_REPOS))
    parser.add_argument("--dest-org", default="RobotLearningVLA")
    parser.add_argument("--dest-suffix", default="_v6_truncated")
    parser.add_argument("--out-dir", default="outputs/eval3_fixed_datasets")
    parser.add_argument("--video-backend", default="pyav")
    parser.add_argument("--vcodec", default="h264")
    parser.add_argument("--image-writer-threads", type=int, default=4)
    parser.add_argument("--image-writer-processes", type=int, default=0)
    parser.add_argument("--encoder-threads", type=int, default=None)
    parser.add_argument("--batch-encoding-size", type=int, default=1)
    parser.add_argument("--parallel-encoding", action="store_true")
    parser.add_argument("--grip-threshold", type=float, default=20.0)
    parser.add_argument("--lift-threshold", type=float, default=-30.0)
    parser.add_argument("--placement-mode", choices=["first", "last"], default="last")
    parser.add_argument("--allow-over-cap-episodes", default=DEFAULT_OVER_CAP_EPISODES)
    parser.add_argument("--over-cap-extra-frames", type=int, default=90)
    parser.add_argument("--buffer-frames", type=int, default=60)
    parser.add_argument("--max-frames-per-episode", type=int, default=600)
    parser.add_argument("--limit-repos", type=int, default=None)
    parser.add_argument("--limit-episodes", type=int, default=None)
    parser.add_argument("--overwrite-local", action="store_true")
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--allow-existing-hub", action="store_true")
    parser.add_argument("--upload-large-folder", action="store_true")
    parser.add_argument("--manifest", default="outputs/eval3_fixed_datasets/manifest.json")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    repos = [repo.strip() for repo in args.repos.split(",") if repo.strip()]
    if args.limit_repos is not None:
        repos = repos[: args.limit_repos]

    results: list[RepoMaterializeInfo] = []
    for repo in repos:
        result = materialize_repo(args, repo)
        results.append(result)
        print(
            f"done {result.dest_repo}: {result.episodes} eps, "
            f"{result.frames} frames, uploaded={result.uploaded}"
        )

    manifest = {
        "rule": {
            "grip_threshold": args.grip_threshold,
            "lift_threshold": args.lift_threshold,
            "placement_mode": args.placement_mode,
            "max_frames_per_episode": args.max_frames_per_episode,
            "allow_over_cap_episodes": args.allow_over_cap_episodes,
            "over_cap_extra_frames": args.over_cap_extra_frames,
            "dest_suffix": args.dest_suffix,
        },
        "repos": [
            asdict(result)
            for result in results
        ],
        "total_episodes": sum(result.episodes for result in results),
        "total_frames": sum(result.frames for result in results),
    }
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"wrote {manifest_path}")
    print(
        f"materialized {manifest['total_episodes']} episodes / "
        f"{manifest['total_frames']} frames"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
