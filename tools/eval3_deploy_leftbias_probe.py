#!/usr/bin/env python3
"""Investigate the closed-loop 'always places on the LEFT' deploy bias.

Two offline tests on a checkpoint:

TEST A  — image-vs-state ablation. Take a committed-LEFT frame and a
committed-RIGHT frame, cross their image and state, predict shoulder_pan.
Tells us which input the policy actually follows — and that it is doing
trajectory *continuation*, not target *selection*.

TEST B  — cold start. Early/home-ish frames (the deploy starts from home),
every dataset x every prompt. If predicted pan is ~constant and one-signed
regardless of scene or prompt, that constant IS the deploy bias.

Convention: +shoulder_pan = LEFT, -shoulder_pan = RIGHT.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "scripts"))
sys.path.insert(0, str(_REPO / "tools"))

import eval3_lerobot_shim  # noqa: E402

eval3_lerobot_shim.apply()
import eval3_smolvla_aux_head  # noqa: E402

eval3_smolvla_aux_head.apply()

import numpy as np  # noqa: E402
import torch  # noqa: E402

from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from eval3_abcd_benchmark import load_policy_bundle, predict  # noqa: E402

PROMPTS = {
    "Taylor Swift": "Place the coke on Taylor Swift",
    "Yann LeCun": "Place the coke on Yann LeCun",
    "Barack Obama": "Place the coke on Barack Obama",
}
STATS_REPO = "RobotLearningVLA/dataset_v3_synth_pinned_idood_taylor_swift_left_2"
DATASETS = {
    "left": "dataset_v3_synth_pinned_taylor_swift_left_1",
    "middle": "dataset_v3_synth_pinned_taylor_swift_middle_1",
    "right": "dataset_v3_synth_pinned_taylor_swift_right_1",
}


def gpan(row) -> float:
    a = row["action"]
    a = a.detach().cpu().numpy() if isinstance(a, torch.Tensor) else np.asarray(a)
    return float(a.reshape(-1)[0])


def committed(ds, sign: int, n: int = 1) -> list[int]:
    """Indices of the n frames whose GT shoulder_pan is most extreme in `sign`."""
    step = max(1, len(ds) // 80)
    cand = [(i, gpan(ds[i])) for i in range(0, len(ds), step)]
    cand.sort(key=lambda t: sign * t[1], reverse=True)
    return [i for i, _ in cand[:n]]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--checkpoint",
        default="outputs/train/eval3_3way_20k_b128_v6_pinned_idood_aux_fresh_final/pretrained_model",
    )
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--datasets-root", default=str(_REPO / "datasets"))
    args = ap.parse_args()

    print(f"\n[leftbias] checkpoint = {args.checkpoint}")
    print(f"[leftbias] device     = {args.device}   (+pan = LEFT, -pan = RIGHT)\n")
    bundle = load_policy_bundle(args.checkpoint, args.device, STATS_REPO)

    root = Path(args.datasets_root)
    ds = {
        pos: LeRobotDataset(
            f"RobotLearningVLA/{name}", root=str(root / name), episodes=[0], video_backend="pyav"
        )
        for pos, name in DATASETS.items()
    }

    # ===================== TEST A — image vs state ablation =================
    L, R = ds["left"], ds["right"]
    li, ri = committed(L, +1)[0], committed(R, -1)[0]
    lrow, rrow = L[li], R[ri]
    print("=" * 84)
    print(f"TEST A  —  image-vs-state ablation")
    print(f"  LEFT-committed frame  : {DATASETS['left']}  idx {li}  GT pan {gpan(lrow):+.1f}")
    print(f"  RIGHT-committed frame : {DATASETS['right']} idx {ri}  GT pan {gpan(rrow):+.1f}")
    print(f"  prompt fixed = 'Place the coke on Taylor Swift'")
    print("=" * 84)
    combos = [
        ("img=LEFT   state=LEFT  ", lrow["observation.images.front"], lrow["observation.state"]),
        ("img=LEFT   state=RIGHT ", lrow["observation.images.front"], rrow["observation.state"]),
        ("img=RIGHT  state=LEFT  ", rrow["observation.images.front"], lrow["observation.state"]),
        ("img=RIGHT  state=RIGHT ", rrow["observation.images.front"], rrow["observation.state"]),
    ]
    res = {}
    for label, img, st in combos:
        hyb = {"observation.images.front": img, "observation.state": st}
        pan = float(predict(bundle, hyb, PROMPTS["Taylor Swift"])[0])
        res[label.strip()] = pan
        print(f"  {label}  ->  pred shoulder_pan {pan:+7.1f}")
    # which input flips the sign?
    img_effect = abs(res["img=LEFT   state=LEFT"] - res["img=RIGHT  state=LEFT"])
    state_effect = abs(res["img=LEFT   state=LEFT"] - res["img=LEFT   state=RIGHT"])
    print("-" * 84)
    print(f"  swapping IMAGE  (state held) moves pan by ~{img_effect:5.1f} deg")
    print(f"  swapping STATE  (image held) moves pan by ~{state_effect:5.1f} deg")
    driver = "STATE (proprio)" if state_effect > img_effect else "IMAGE"
    print(f"  => dominant driver of direction: {driver}")

    # ===================== TEST B — cold start ==============================
    print()
    print("=" * 84)
    print("TEST B  —  COLD START (arm near home; deploy always begins from home)")
    print("  predicted shoulder_pan for early frames, every dataset x every prompt")
    print("=" * 84)
    hdr = f"{'dataset':>9s} {'frame':>6s} {'GT pan':>8s} | " + " ".join(f"{p:>13s}" for p in PROMPTS)
    print(hdr)
    print("-" * len(hdr))
    cold_preds = []
    for pos in ("left", "middle", "right"):
        d = ds[pos]
        early = sorted(set(int(x) for x in np.linspace(0, max(1, int(len(d) * 0.05)), 4)))
        for fi in early:
            row = d[fi]
            preds = {p: float(predict(bundle, row, PROMPTS[p])[0]) for p in PROMPTS}
            cold_preds.extend(preds.values())
            print(
                f"{pos:>9s} {fi:>6d} {gpan(row):>+8.1f} | "
                + " ".join(f"{preds[p]:>+13.1f}" for p in PROMPTS)
            )
    cp = np.array(cold_preds)
    print("-" * len(hdr))
    print(
        f"  cold-start predicted pan:  mean {cp.mean():+.1f}   std {cp.std():.1f}   "
        f"range [{cp.min():+.1f}, {cp.max():+.1f}]"
    )
    side = "LEFT" if cp.mean() > 1 else ("RIGHT" if cp.mean() < -1 else "≈neutral")
    print(
        f"  => from a home start the policy commits {side} with std {cp.std():.1f} deg "
        f"across all scenes & prompts."
    )
    print("\n[leftbias] done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
