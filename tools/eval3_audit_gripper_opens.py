#!/usr/bin/env python3
"""Audit gripper joint values in the approach phase of Eval3 LeRobot datasets.

Looks at the first ``--approach-frac`` of each episode (default 40%, before grasp).
If demos systematically keep the gripper mid-closed, policies may learn to approach
without a full pre-grasp open — a data issue, not fixable by deploy flags alone.

Example::

    uv run python tools/eval3_audit_gripper_opens.py \\
      --repo-ids RobotLearningVLA/dataset_v2_taylor_swift_left_1,RobotLearningVLA/dataset_v2_barack_obama_left_1 \\
      --revision v3.0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset


ACTION_KEY = "action"


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--repo-ids",
        default=(
            "RobotLearningVLA/dataset_v2_barack_obama_left_1,"
            "RobotLearningVLA/dataset_v2_yann_lecun_left_1,"
            "RobotLearningVLA/dataset_v2_taylor_swift_left_1"
        ),
        help="Comma-separated dataset repo ids.",
    )
    ap.add_argument("--revision", default="v3.0")
    ap.add_argument("--video-backend", default="pyav")
    ap.add_argument(
        "--approach-frac",
        type=float,
        default=0.40,
        help="Fraction of each episode length (from start) treated as approach segment.",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/eval3_analysis/gripper_approach_audit.json"),
        help="Write numeric summary as JSON.",
    )
    return ap.parse_args()


def _to_numpy_action(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy().astype(np.float32).reshape(-1)
    return np.asarray(value, dtype=np.float32).reshape(-1)


def _gripper_index(names: list[str]) -> int:
    for i, n in enumerate(names):
        if "gripper" in n.lower():
            return i
    raise ValueError(f"No gripper channel in action names: {names}")


def _audit_repo(repo_id: str, *, revision: str, video_backend: str, approach_frac: float) -> dict:
    ds = LeRobotDataset(repo_id, revision=revision, video_backend=video_backend)
    names = ds.meta.features[ACTION_KEY].get("names") or []
    if not names:
        raise RuntimeError(f"{repo_id}: missing action names in metadata")
    gi = _gripper_index(names)
    episodes = ds.meta.episodes
    all_vals: list[float] = []
    ep_mins: list[float] = []
    ep_maxs: list[float] = []

    ep_idx_col = episodes["episode_index"]
    from_col = episodes["dataset_from_index"]
    to_col = episodes["dataset_to_index"]

    for i in range(len(episodes)):
        start = int(from_col[i])
        stop = int(to_col[i])
        length = max(stop - start, 1)
        n = max(int(np.floor(length * approach_frac)), 1)
        end = start + n
        ep_sample: list[float] = []
        for global_idx in range(start, end):
            row = ds[global_idx]
            act = _to_numpy_action(row[ACTION_KEY])
            g = float(act[gi])
            ep_sample.append(g)
            all_vals.append(g)
        if ep_sample:
            ep_mins.append(min(ep_sample))
            ep_maxs.append(max(ep_sample))

    arr = np.asarray(all_vals, dtype=np.float64)
    return {
        "repo_id": repo_id,
        "gripper_action_name": names[gi],
        "episodes_audited": len(episodes),
        "approach_frac": approach_frac,
        "n_approach_frames": int(arr.size),
        "gripper_min": float(arr.min()) if arr.size else None,
        "gripper_max": float(arr.max()) if arr.size else None,
        "gripper_mean": float(arr.mean()) if arr.size else None,
        "gripper_p10": float(np.percentile(arr, 10)) if arr.size else None,
        "gripper_p50": float(np.percentile(arr, 50)) if arr.size else None,
        "gripper_p90": float(np.percentile(arr, 90)) if arr.size else None,
        "per_episode_approach_gripper_min_mean": float(np.mean(ep_mins)) if ep_mins else None,
        "per_episode_approach_gripper_max_mean": float(np.mean(ep_maxs)) if ep_maxs else None,
    }


def main() -> int:
    args = _parse_args()
    repos = [x.strip() for x in args.repo_ids.split(",") if x.strip()]
    summaries = []
    print("# Gripper approach audit (normalized action space)", flush=True)
    print(f"- approach_frac={args.approach_frac}", flush=True)
    print("| repo | episodes | frames | mean | p10 | p50 | p90 | min | max |", flush=True)
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|", flush=True)
    for rid in repos:
        s = _audit_repo(rid, revision=args.revision, video_backend=args.video_backend, approach_frac=args.approach_frac)
        summaries.append(s)
        print(
            f"| `{s['repo_id']}` | {s['episodes_audited']} | {s['n_approach_frames']} | "
            f"{s['gripper_mean']:.4f} | {s['gripper_p10']:.4f} | {s['gripper_p50']:.4f} | {s['gripper_p90']:.4f} | "
            f"{s['gripper_min']:.4f} | {s['gripper_max']:.4f} |",
            flush=True,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(f"\nWrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
