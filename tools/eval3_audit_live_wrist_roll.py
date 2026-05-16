#!/usr/bin/env python3
"""Pre-flight audit 1: independently re-verify final-1s wrist_roll for each
live-eval HF dataset and cross-check against the stored summary.

The stored summary at outputs/eval3_live_analysis/live_rollout_summary.json
was produced earlier in the conversation by an ad-hoc script. Before betting
a $1+ warm-restart on that diagnosis, we re-pull the action streams directly
from each eval HF repo, compute the final-1s mean of action[:, 4] (wrist_roll)
independently, and report any discrepancy > 1 deg.

action channel layout (from tools/eval3_render_overlay.py:55):
  0=shoulder_pan, 1=shoulder_lift, 2=elbow_flex, 3=wrist_flex, 4=wrist_roll, 5=gripper
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

EVAL_REPOS = [
    ("RobotLearningVLA/eval_taylor_swift_smolvla_v1", "swift",  -80.0),
    ("RobotLearningVLA/eval_taylor_swift_smolvla_v2", "swift",  -80.0),
    ("RobotLearningVLA/eval_taylor_swift_smolvla_v3", "swift",  -80.0),
    ("RobotLearningVLA/eval_yann_lecun_smolvla_v2",   "lecun",  +60.0),
    ("RobotLearningVLA/eval_barack_obama_smolvla_v1", "obama",  +54.0),
    ("RobotLearningVLA/eval_barack_obama_smolvla_v4", "obama",  +54.0),
]
WRIST_ROLL_IDX = 4
SHOULDER_LIFT_IDX = 1
FINAL_WINDOW_FRAMES = 30  # 1 s at 30 fps
DISCREPANCY_THRESHOLD_DEG = 1.0

# Training-data per-celebrity shoulder_lift means (from audit 2). The
# wrist_roll targets in EVAL_REPOS are the means of bimodal distributions,
# so they are not reliable single-value targets. shoulder_lift has unimodal
# distributions and provides cleaner ground-truth.
TRAIN_SHOULDER_LIFT_MEAN = {"swift": -23.68, "lecun": -35.96, "obama": -81.67}


def compute_final_1s(actions: torch.Tensor, axis: int = WRIST_ROLL_IDX) -> float:
    """Mean of action[:, axis] over the final FINAL_WINDOW_FRAMES rows."""
    n = actions.shape[0]
    start = max(0, n - FINAL_WINDOW_FRAMES)
    return actions[start:n, axis].float().mean().item()


def load_actions(repo_id: str) -> torch.Tensor:
    """Pull action column straight from hf_dataset (no video decode)."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    ds = LeRobotDataset(repo_id, video_backend="pyav")
    actions = torch.stack([torch.as_tensor(r["action"]) for r in ds.hf_dataset])
    return actions


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--stored-summary",
        default="outputs/eval3_live_analysis/live_rollout_summary.json",
        help="JSON to compare against (set --no-compare to skip)",
    )
    ap.add_argument("--out-dir", default="outputs/eval3_pre_flight")
    ap.add_argument("--no-compare", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stored = {}
    if not args.no_compare:
        with open(args.stored_summary) as f:
            for entry in json.load(f):
                stored[entry["repo"]] = entry["wr_final_1s"]

    rows = []
    max_disc = 0.0
    print(f"{'repo':<55} {'celeb':<6} {'n':>5} {'wr_final':>10} {'stored':>10} {'delta':>8} {'sh_lift':>10} {'sh_lift_train':>13}")
    print("-" * 130)
    for repo, celeb, expected in EVAL_REPOS:
        actions = load_actions(repo)
        wr = compute_final_1s(actions, WRIST_ROLL_IDX)
        sl = compute_final_1s(actions, SHOULDER_LIFT_IDX)
        train_sl = TRAIN_SHOULDER_LIFT_MEAN[celeb]
        stored_wr = stored.get(repo)
        delta = wr - stored_wr if stored_wr is not None else None
        max_disc = max(max_disc, abs(delta) if delta is not None else 0.0)
        rows.append({
            "repo": repo,
            "celeb": celeb,
            "n_frames": int(actions.shape[0]),
            "wr_final_1s_recomputed": wr,
            "wr_final_1s_stored": stored_wr,
            "delta_vs_stored": delta,
            "sh_lift_final_1s": sl,
            "sh_lift_train_mean": train_sl,
            "sh_lift_delta": sl - train_sl,
            "expected_wr": expected,
        })
        delta_str = f"{delta:+.3f}" if delta is not None else "  --"
        stored_str = f"{stored_wr:.3f}" if stored_wr is not None else "  --"
        print(f"{repo:<55} {celeb:<6} {actions.shape[0]:>5} {wr:>+10.2f} {stored_str:>10} {delta_str:>8} {sl:>+10.2f} {train_sl:>+13.2f}")

    # Save
    with open(out_dir / "audit1_live_wrist_roll.json", "w") as f:
        json.dump({"max_discrepancy": max_disc, "rows": rows}, f, indent=2)
    print()
    print(f"max |delta vs stored| : {max_disc:.4f}  (threshold {DISCREPANCY_THRESHOLD_DEG})")

    if max_disc > DISCREPANCY_THRESHOLD_DEG:
        print("FAIL — discrepancy exceeds threshold; halt plan.")
        return 2
    print("PASS — stored summary verified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
