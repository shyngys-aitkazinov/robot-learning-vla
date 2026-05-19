#!/usr/bin/env python3
"""Cross-prompt sensitivity test for the aux-head experiment.

For a given checkpoint path (local pretrained_model dir, or HF repo), feeds the
two training frames `LeCun_left_ep0` and `Swift_right_ep1` through the policy
with all three celebrity prompts and reports:

  - GT shoulder_pan (the supervised action at that frame)
  - Predicted pan with each prompt
  - Δ_max(prompt) — the spread across the 3 prompts

A healthy language-conditioned model would show Δ_max ≈ 50° on these frames
(LeCun's training frame goes pan≈+29°, Swift's training frame goes pan≈−20°,
so different prompts SHOULD produce different actions). A collapsed model
shows Δ_max ≈ 0°.

This is the canonical test we used to diagnose v6_synth_15k. Re-running it
on the aux-head-trained checkpoint tells us whether the patch helped.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "scripts"))
sys.path.insert(0, str(_REPO / "tools"))

import eval3_lerobot_shim
eval3_lerobot_shim.apply()
# Apply the aux-head patch so the policy class has `position_clf_head` when
# we load a state_dict that contains those keys (trained-with-aux runs).
# Without this, load_policy_bundle would error on "unexpected keys". The
# head sits unused during inference (sample_actions doesn't touch it).
import eval3_smolvla_aux_head  # noqa: E402
eval3_smolvla_aux_head.apply()

import torch  # noqa: E402

from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from eval3_abcd_benchmark import load_policy_bundle, predict  # noqa: E402


NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
PAN_I = NAMES.index("shoulder_pan")

PROMPTS = [
    "Place the coke on Yann LeCun",
    "Place the coke on Taylor Swift",
    "Place the coke on Barack Obama",
]

# Two specific training frames used in the diagnosis:
#  - LeCun_left_ep0: scene [LeCun-L, Obama-M, Swift-R], action goes LEFT (+29°)
#  - Swift_right_ep1: scene [LeCun-L, Obama-M, Swift-R] (SAME visual layout!),
#    action goes RIGHT (-20°). This is the cleanest language-disambiguation pair.
FRAMES = [
    ("LeCun_left_ep0", "RobotLearningVLA/dataset_v3_synth_yann_lecun_left_2", 0, [100, 200, 300, 400]),
    ("Swift_right_ep1", "RobotLearningVLA/dataset_v3_synth_taylor_swift_right_2", 1, [100, 200, 300, 400]),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True, help="HF repo id or local pretrained_model path")
    ap.add_argument("--device", default="mps")
    ap.add_argument(
        "--stats-repo",
        default="RobotLearningVLA/dataset_v3_synth_yann_lecun_left_2",
        help="Dataset to pull stats from (must match training schema)",
    )
    args = ap.parse_args()

    print(f"\n[aux-head x-prompt test] checkpoint = {args.checkpoint}")
    print(f"[aux-head x-prompt test] device = {args.device}\n")

    bundle = load_policy_bundle(args.checkpoint, args.device, args.stats_repo)

    print(f"{'episode':>20s}  {'frame':>5s}  {'GT_pan':>7s}  ", end="")
    for p in PROMPTS:
        print(f"{p.split(' on ')[1][:11]:>12s}", end="  ")
    print(f"{'Δ_max':>7s}  {'verdict':>10s}")
    print("-" * 100)

    all_deltas: list[float] = []
    for label, repo, ep, frame_picks in FRAMES:
        ds = LeRobotDataset(repo, episodes=[ep], revision="v3.0", video_backend="pyav")
        for fi in frame_picks:
            if fi >= len(ds):
                continue
            row = ds[fi]
            gt_pan = float(row["action"][PAN_I])
            preds = []
            for prompt in PROMPTS:
                p = predict(bundle, row, prompt)
                preds.append(p[PAN_I])
            spread = max(preds) - min(preds)
            all_deltas.append(spread)
            verdict = "✓ uses lang" if spread > 10 else ("?? weak" if spread > 2 else "✗ ignores")
            print(
                f"{label:>20s}  {fi:>5d}  {gt_pan:>+7.2f}  ",
                end="",
            )
            for pp in preds:
                print(f"{pp:>+12.2f}", end="  ")
            print(f"{spread:>7.2f}  {verdict:>10s}")

    print("-" * 100)
    mean_delta = sum(all_deltas) / len(all_deltas) if all_deltas else 0.0
    max_delta = max(all_deltas) if all_deltas else 0.0
    print(f"\nSummary: mean Δ_max(prompt) = {mean_delta:.2f}°,  max Δ_max(prompt) = {max_delta:.2f}°")
    print(f"  Reference (v6_synth_15k before fix): mean ≈ 0.6°, max ≈ 1.95°")
    print(f"  Healthy language-conditioned model: Δ_max ≥ 20° on these frames")
    return 0


if __name__ == "__main__":
    sys.exit(main())
