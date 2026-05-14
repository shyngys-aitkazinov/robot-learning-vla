#!/usr/bin/env python3
"""Pre-flight audit 3: Swift training place-completion quality.

Sign convention verified empirically: shoulder_lift starts at ~-103 deg (arm
folded at home) and INCREASES as the arm extends forward toward the table.
Swift training typically extends ~150 deg; LeCun ~93; Obama only ~17 (the
Obama print is closest to the base, Swift the farthest).

Each Swift training episode should end with:
  (a) gripper FULLY OPEN (delta from min to last-1s mean >= GRIPPER_OPEN_DELTA)
  (b) wrist_roll within WRIST_ROLL_TOL of the canonical -86.2 deg mean
  (c) shoulder_lift EXTENDS by at least EXTEND_MIN_DEG from start to end

The audit is **diagnostic, not gating** — it tells us whether the v1/v2/v3
placement misses (the user's flagged worry #3) are inherited from noisy
training labels or are a model artifact.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
SHOULDER_LIFT_IDX = 1
WRIST_ROLL_IDX = 4
GRIPPER_IDX = 5

FINAL_WINDOW = 30  # 1 s at 30 fps
START_WINDOW = 30  # 1 s of the episode start
GRIPPER_OPEN_DELTA = 10.0  # mean(last 1s) - min(whole episode) > this -> gripper opened
WRIST_ROLL_TOL = 15.0
SWIFT_WRIST_ROLL_MEAN = -86.21
EXTEND_MIN_DEG = 50.0  # end_mean - start_mean must be at least +50 deg (extended forward)


def audit_repo(repo_id: str, celeb_label: str):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    ds = LeRobotDataset(repo_id, video_backend="pyav")
    actions = torch.stack([torch.as_tensor(r["action"]) for r in ds.hf_dataset]).float().numpy()
    ep_df = ds.meta.episodes
    rows = []
    for ep_idx, (f0, f1) in enumerate(zip(ep_df["dataset_from_index"], ep_df["dataset_to_index"])):
        f0i, f1i = int(f0), int(f1)
        ep_actions = actions[f0i:f1i]
        n = ep_actions.shape[0]
        start = ep_actions[:min(n, START_WINDOW)]
        end = ep_actions[max(0, n - FINAL_WINDOW):]

        sl_start = float(start[:, SHOULDER_LIFT_IDX].mean())
        sl_end = float(end[:, SHOULDER_LIFT_IDX].mean())
        sl_extend = sl_end - sl_start  # positive = arm extends forward toward table

        wr_end = float(end[:, WRIST_ROLL_IDX].mean())
        wr_err = abs(wr_end - SWIFT_WRIST_ROLL_MEAN)

        gr_min = float(ep_actions[:, GRIPPER_IDX].min())
        gr_end = float(end[:, GRIPPER_IDX].mean())
        gr_delta = gr_end - gr_min

        check_a = gr_delta >= GRIPPER_OPEN_DELTA
        check_b = wr_err <= WRIST_ROLL_TOL
        check_c = sl_extend >= EXTEND_MIN_DEG
        passed = check_a and check_b and check_c

        rows.append({
            "ep_idx": ep_idx, "n_frames": int(n),
            "shoulder_lift_start": sl_start, "shoulder_lift_end": sl_end, "shoulder_lift_extend": sl_extend,
            "wrist_roll_end": wr_end, "wrist_roll_err": wr_err,
            "gripper_min": gr_min, "gripper_end": gr_end, "gripper_delta": gr_delta,
            "check_a_gripper_opens": check_a,
            "check_b_wr_canonical": check_b,
            "check_c_arm_extends": check_c,
            "passed": passed,
        })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="outputs/eval3_pre_flight")
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== AUDIT 3: Swift training place-completion ===\n")
    rows = audit_repo("RobotLearningVLA/taylor_swift_1", "swift")

    print(f"{'ep':>3} {'n_fr':>5} {'sl_extend':>9} {'wr_err':>7} {'gr_delta':>9} {'A':>2} {'B':>2} {'C':>2} {'PASS':>5}")
    print("-" * 60)
    n_pass = 0
    for r in rows:
        print(f"{r['ep_idx']:>3} {r['n_frames']:>5} {r['shoulder_lift_extend']:>+9.2f} "
              f"{r['wrist_roll_err']:>7.2f} {r['gripper_delta']:>+9.2f} "
              f"{'Y' if r['check_a_gripper_opens'] else 'n':>2} "
              f"{'Y' if r['check_b_wr_canonical'] else 'n':>2} "
              f"{'Y' if r['check_c_arm_extends'] else 'n':>2} "
              f"{'PASS' if r['passed'] else 'fail':>5}")
        if r["passed"]:
            n_pass += 1

    n_total = len(rows)
    pass_rate = n_pass / n_total if n_total else 0.0
    print(f"\nSwift place-completion rate: {n_pass}/{n_total} = {pass_rate*100:.1f}%")
    print("Per-check failure counts:")
    for chk in ["check_a_gripper_opens", "check_b_wr_canonical", "check_c_arm_extends"]:
        miss = sum(1 for r in rows if not r[chk])
        print(f"  {chk}: {miss}/{n_total} fail")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].hist([r["gripper_delta"] for r in rows], bins=15, color="#16a34a", alpha=0.7)
    axes[0].axvline(GRIPPER_OPEN_DELTA, color="red", linestyle="--", label=f"threshold {GRIPPER_OPEN_DELTA}")
    axes[0].set_xlabel("gripper delta (end_mean - min)")
    axes[0].set_title("Gripper opens?")
    axes[0].legend()

    axes[1].hist([r["wrist_roll_err"] for r in rows], bins=15, color="#3b82f6", alpha=0.7)
    axes[1].axvline(WRIST_ROLL_TOL, color="red", linestyle="--", label=f"threshold {WRIST_ROLL_TOL}")
    axes[1].set_xlabel("|wrist_roll_end - -86.2|")
    axes[1].set_title("Canonical wrist_roll?")
    axes[1].legend()

    axes[2].hist([r["shoulder_lift_extend"] for r in rows], bins=15, color="#ef4444", alpha=0.7)
    axes[2].axvline(EXTEND_MIN_DEG, color="red", linestyle="--", label=f"threshold {EXTEND_MIN_DEG}")
    axes[2].set_xlabel("shoulder_lift_end - start (deg)")
    axes[2].set_title("Arm extends forward?")
    axes[2].legend()
    fig.suptitle("Swift training place-completion per-episode checks")
    fig.tight_layout()
    fig.savefig(out_dir / "audit3_swift_place_checks.png", dpi=120)

    verdict = "GOOD" if pass_rate >= 0.80 else "REVIEW"
    out_json = {
        "verdict": verdict,
        "n_total": n_total, "n_pass": n_pass, "pass_rate": pass_rate,
        "failing_episodes": [r["ep_idx"] for r in rows if not r["passed"]],
        "rows": rows,
    }
    with open(out_dir / "audit3_swift_place.json", "w") as f:
        json.dump(out_json, f, indent=2)

    print(f"\nAUDIT 3 : {verdict} (>= 80%% pass = GOOD; else REVIEW the failing episodes before warm-restart)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
