#!/usr/bin/env python3
"""Inspect LeRobot v3 Hub/local datasets (Eval 3 QA gate).

Usage:
  python tools/inspect_lerobot_dataset.py --repo-id RobotLearningVLA/taylor_swift_1
  python tools/inspect_lerobot_dataset.py --repo-id RobotLearningVLA/banana_green_bowl_eval1 --episodes 0 1
"""

from __future__ import annotations

import argparse
import json
from collections import Counter

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect LeRobotDataset metadata and sample tensors.")
    parser.add_argument(
        "--repo-id",
        default="RobotLearningVLA/taylor_swift_1",
        help="HF dataset repo id",
    )
    parser.add_argument(
        "--episodes",
        nargs="*",
        type=int,
        default=None,
        help="Optional episode indices to restrict (subset loading)",
    )
    parser.add_argument("--sample-index", type=int, default=0, help="Frame index for tensor shape preview")
    parser.add_argument("--no-videos", action="store_true", help="Skip downloading/decoding videos if supported")
    parser.add_argument(
        "--video-backend",
        default="pyav",
        help="Video decoder for frames (use pyav on macOS if torchcodec/FFmpeg fails)",
    )
    args = parser.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(
        args.repo_id,
        episodes=args.episodes if args.episodes else None,
        download_videos=not args.no_videos,
        video_backend=args.video_backend,
    )

    print("=== LeRobotDataset ===")
    print(f"repo_id:          {ds.repo_id}")
    print(f"revision:         {ds.revision}")
    print(f"num_episodes:     {ds.num_episodes}")
    print(f"num_frames:       {ds.num_frames}")
    print(f"fps:              {ds.meta.fps}")

    feats = ds.features
    print("\n=== features ===")
    for name, ft in sorted(feats.items()):
        shape = getattr(ft, "shape", None)
        print(f"  {name}: shape={shape}")

    tasks = ds.meta.tasks
    print("\n=== tasks (meta.tasks) ===")
    try:
        if hasattr(tasks, "to_string"):
            print(tasks.to_string())
        else:
            print(repr(tasks))
    except Exception as e:
        print(f"(could not print tasks: {e})")

    # Task histogram from hf_dataset if task column exists
    print("\n=== task string histogram (from episodes / parquet) ===")
    try:
        hf = ds.hf_dataset
        col = "task" if "task" in hf.column_names else None
        if col:
            counts = Counter(hf[col])
            for k, v in counts.most_common(20):
                print(f"  {v:5d}  {k!r}")
            if len(counts) > 20:
                print(f"  ... ({len(counts)} unique)")
        else:
            print("  (no 'task' column in hf_dataset)")
    except Exception as e:
        print(f"  (skipped: {e})")

    idx = max(0, min(args.sample_index, len(ds) - 1))
    print(f"\n=== sample[{idx}] tensor shapes ===")
    row = ds[idx]
    for k, v in sorted(row.items()):
        if isinstance(v, torch.Tensor):
            print(f"  {k}: Tensor dtype={v.dtype} shape={tuple(v.shape)}")
        else:
            preview = repr(v)
            if len(preview) > 120:
                preview = preview[:117] + "..."
            print(f"  {k}: {type(v).__name__} {preview}")

    info_path = ds.root / "meta" / "info.json"
    if info_path.is_file():
        print("\n=== meta/info.json codebase_version ===")
        info = json.loads(info_path.read_text())
        print(f"  codebase_version: {info.get('codebase_version')}")


if __name__ == "__main__":
    main()
