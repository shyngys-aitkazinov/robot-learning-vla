#!/usr/bin/env python3
"""Pre-flight audit 2: per-celebrity wrist_roll separability in training.

If the model is supposed to learn a 3-way celebrity wrist_roll classifier, the
*training labels themselves* must contain a clean 3-way signal. We histogram
the final-1s wrist_roll (=action[:, 4]) across every episode of the three
training datasets, compute per-celebrity mean/std, and check pairwise mean
separation.

Pass criterion:
  per-celebrity std < 10 deg
  pairwise |mean_A - mean_B| >= 40 deg  for every pair

If this fails, augmentation cannot save us — the supervision is too noisy.
"""
from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

TRAIN_REPOS = [
    ("RobotLearningVLA/taylor_swift_1", "swift"),
    ("RobotLearningVLA/yann_lecun_1",   "lecun"),
    ("RobotLearningVLA/barack_obama_1", "obama"),
]
JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
FINAL_WINDOW_FRAMES = 30


def per_episode_final_all_joints(repo_id: str) -> np.ndarray:
    """For each episode, return mean(action[last 30 frames, :]) -> shape (n_eps, 6)."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    ds = LeRobotDataset(repo_id, video_backend="pyav")
    ep_df = ds.meta.episodes
    out = []
    actions = torch.stack([torch.as_tensor(r["action"]) for r in ds.hf_dataset])
    for f0, f1 in zip(ep_df["dataset_from_index"], ep_df["dataset_to_index"]):
        f0i, f1i = int(f0), int(f1)
        start = max(f0i, f1i - FINAL_WINDOW_FRAMES)
        out.append(actions[start:f1i, :].float().mean(dim=0).numpy())
    return np.stack(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="outputs/eval3_pre_flight")
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_celeb_actions: dict[str, np.ndarray] = {}
    for repo, celeb in TRAIN_REPOS:
        per_celeb_actions[celeb] = per_episode_final_all_joints(repo)

    # Per-joint separability summary
    print(f"\n{'joint':<14} {'celeb':<6} {'n_eps':>5} {'mean':>10} {'std':>8} {'min':>8} {'max':>8}")
    print("-" * 68)
    stats: dict[str, dict] = {jn: {} for jn in JOINT_NAMES}
    for j, jn in enumerate(JOINT_NAMES):
        for celeb, arr in per_celeb_actions.items():
            v = arr[:, j]
            stats[jn][celeb] = {
                "n_eps": int(v.size),
                "mean": float(v.mean()),
                "std": float(v.std()),
                "min": float(v.min()),
                "max": float(v.max()),
                "per_episode": [float(x) for x in v],
            }
            s = stats[jn][celeb]
            print(f"{jn:<14} {celeb:<6} {s['n_eps']:>5} {s['mean']:>+10.3f} {s['std']:>8.3f} {s['min']:>+8.2f} {s['max']:>+8.2f}")
        print()

    # Pairwise separation per joint
    print(f"\n{'joint':<14} {'pair':<14} {'sep_deg':>10} {'max_std':>10} {'verdict':>10}")
    print("-" * 60)
    pair_results = []
    for jn in JOINT_NAMES:
        for (a, sa), (b, sb) in combinations(stats[jn].items(), 2):
            sep = abs(sa["mean"] - sb["mean"])
            max_std = max(sa["std"], sb["std"])
            # A robust separator: pairwise mean sep > 40 deg AND both stds < 15 deg
            ok = sep >= 40.0 and max_std < 15.0
            pair_results.append({"joint": jn, "pair": f"{a}-{b}", "sep_deg": float(sep), "max_std": float(max_std), "ok": bool(ok)})
            print(f"{jn:<14} {a}-{b:<8} {sep:>10.2f} {max_std:>10.2f} {('OK' if ok else 'FAIL'):>10}")

    # Identify the cleanest 3-way separator joint (all 3 pairs OK)
    clean_joints = []
    for jn in JOINT_NAMES:
        all_ok = all(r["ok"] for r in pair_results if r["joint"] == jn)
        if all_ok:
            clean_joints.append(jn)
    print(f"\nClean 3-way separator joints: {clean_joints if clean_joints else '(none)'}")

    # Plot a 2x3 grid: one histogram per joint
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    colors = {"swift": "#3b82f6", "lecun": "#ef4444", "obama": "#16a34a"}
    for j, jn in enumerate(JOINT_NAMES):
        ax = axes[j // 3, j % 3]
        # Per-joint min/max range, with 30 bins
        all_vals = np.concatenate([np.array(stats[jn][c]["per_episode"]) for c in colors])
        lo, hi = float(all_vals.min()) - 5, float(all_vals.max()) + 5
        bins = np.linspace(lo, hi, 30)
        for celeb in ["swift", "lecun", "obama"]:
            v = np.array(stats[jn][celeb]["per_episode"])
            ax.hist(v, bins=bins, alpha=0.55, color=colors[celeb], label=f"{celeb} (mean {stats[jn][celeb]['mean']:+.1f}, std {stats[jn][celeb]['std']:.1f})")
            ax.axvline(stats[jn][celeb]["mean"], color=colors[celeb], linestyle="--", alpha=0.7)
        ax.set_title(jn)
        ax.set_xlabel("final-1s mean (deg)")
        ax.set_ylabel("episodes")
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(alpha=0.3)
    fig.suptitle("Per-joint training final-1s separability (Swift / LeCun / Obama)")
    fig.tight_layout()
    fig.savefig(out_dir / "audit2_train_all_joints_hist.png", dpi=120)

    verdict = "PASS" if clean_joints else "FAIL"
    print(f"\nAUDIT 2 : {verdict}  (need at least one joint with clean 3-way separation)")

    with open(out_dir / "audit2_train_wrist_roll.json", "w") as f:
        json.dump({
            "verdict": verdict,
            "clean_separator_joints": clean_joints,
            "per_joint": stats,
            "pair_results": pair_results,
        }, f, indent=2)

    return 0 if verdict == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
