#!/usr/bin/env python3
"""Aux-head sensitivity to initial state and image.

Runs SmolVLA's aux position head while independently varying the
`observation.state` and `observation.images.front` it is fed, to see what
actually drives the slot prediction at a cold (home) start.

  TEST A — natural scenes: left / middle / right frame-0 (image + that
           scene's own home state).
  TEST B — state sweep: middle frame-0 image held fixed; state swapped for
           {middle / left / right home, left / right committed-pose, zeros}.
  TEST C — image x state 3x3 grid: does the aux head follow the IMAGE
           (scene) or the proprio STATE?

Also prints predicted shoulder_pan per cell (+pan = LEFT, -pan = RIGHT).

NOTE: uses the SYNTHETIC dataset images (roughly in-distribution). The live
deploy additionally suffers the synthetic->real camera domain gap, which
this offline probe cannot reproduce without the real firstframe.png.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("EVAL3_AUX_POS_LOSS_WEIGHT", "0.5")  # makes the aux head run in forward

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

PROMPT = "Place the coke on Taylor Swift"
POS = ["left", "middle", "right"]
STATS_REPO = "RobotLearningVLA/dataset_v3_synth_pinned_idood_taylor_swift_left_2"
DSETS = {
    "left": "dataset_v3_synth_pinned_taylor_swift_left_1",
    "middle": "dataset_v3_synth_pinned_taylor_swift_middle_1",
    "right": "dataset_v3_synth_pinned_taylor_swift_right_1",
}


def gpan(row) -> float:
    a = row["action"]
    a = a.detach().cpu().numpy() if isinstance(a, torch.Tensor) else np.asarray(a)
    return float(a[0, 0]) if a.ndim == 2 else float(a.reshape(-1)[0])


def aux(policy, preprocessor, capture, row, n=4):
    """Mean softmax [p_left, p_mid, p_right] from the aux position head."""
    batch = {k: (v.unsqueeze(0) if isinstance(v, torch.Tensor) else [v]) for k, v in row.items()}
    batch["task"] = [PROMPT]
    base = preprocessor(batch)
    probs = []
    policy.eval()
    with torch.no_grad():
        for _ in range(n):
            base["target_position"] = torch.tensor([1], dtype=torch.long)  # any valid label; makes head run
            capture.pop("logits", None)
            policy.forward(base)
            lg = capture.get("logits")
            if lg is not None:
                probs.append(torch.softmax(lg.float(), -1)[0].cpu().numpy())
    return np.mean(probs, axis=0) if probs else None


def make_row(template, image=None, state=None):
    r = dict(template)
    if image is not None:
        r["observation.images.front"] = image
    if state is not None:
        r["observation.state"] = state
    return r


def show(label, p, pan):
    arg = POS[int(np.argmax(p))]
    print(f"  {label:<34s}  L {p[0]:.2f}  M {p[1]:.2f}  R {p[2]:.2f}   -> aux={arg:<6s}  pred_pan={pan:+6.1f}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--checkpoint",
        default="outputs/train/eval3_3way_20k_b128_v6_pinned_idood_aux_fresh_final/pretrained_model",
    )
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--datasets-root", default=str(_REPO / "datasets"))
    args = ap.parse_args()

    print(f"\n[state-image probe] checkpoint = {args.checkpoint}")
    print(f"[state-image probe] device     = {args.device}   (+pan = LEFT, -pan = RIGHT)\n")
    bundle = load_policy_bundle(args.checkpoint, args.device, STATS_REPO)
    policy, preprocessor, postprocessor, device = bundle
    chunk = int(policy.model.config.chunk_size)
    capture: dict = {}
    policy.model.position_clf_head.register_forward_hook(
        lambda m, i, o: capture.__setitem__("logits", o.detach())
    )

    root = Path(args.datasets_root)
    ds = {
        pos: LeRobotDataset(
            f"RobotLearningVLA/{DSETS[pos]}",
            root=str(root / DSETS[pos]),
            episodes=[0],
            delta_timestamps={"action": [i / 30 for i in range(chunk)]},
            video_backend="pyav",
        )
        for pos in POS
    }
    f0 = {pos: ds[pos][0] for pos in POS}  # frame-0 rows (home-ish state)

    # committed-pose states (arm already swung left / right, mid-trajectory)
    def committed(d, sign):
        step = max(1, len(d) // 60)
        cand = [(i, gpan(d[i])) for i in range(0, len(d), step)]
        cand.sort(key=lambda t: sign * t[1], reverse=True)
        return d[cand[0][0]]["observation.state"], cand[0][0]
    l_state, l_idx = committed(ds["left"], +1)
    r_state, r_idx = committed(ds["right"], -1)
    zero_state = torch.zeros_like(f0["middle"]["observation.state"])

    # ===================== TEST A — natural scenes ==========================
    print("=" * 84)
    print("TEST A  —  natural frame-0 scenes (image + that scene's own home state)")
    print("=" * 84)
    for pos in POS:
        p = aux(policy, preprocessor, capture, f0[pos])
        pan = float(predict(bundle, f0[pos], PROMPT)[0])
        show(f"{pos:<6s} target  (img={pos}, state={pos})", p, pan)

    # ===================== TEST B — state sweep =============================
    print()
    print("=" * 84)
    print("TEST B  —  IMAGE held = middle frame-0;  STATE swapped")
    print(f"  (left-committed state from frame {l_idx}, right-committed from frame {r_idx})")
    print("=" * 84)
    states = [
        ("state = middle home", f0["middle"]["observation.state"]),
        ("state = left home", f0["left"]["observation.state"]),
        ("state = right home", f0["right"]["observation.state"]),
        ("state = LEFT committed", l_state),
        ("state = RIGHT committed", r_state),
        ("state = zeros", zero_state),
    ]
    for label, st in states:
        row = make_row(f0["middle"], state=st)
        p = aux(policy, preprocessor, capture, row)
        pan = float(predict(bundle, row, PROMPT)[0])
        show(label, p, pan)

    # ===================== TEST C — image x state grid =====================
    print()
    print("=" * 84)
    print("TEST C  —  image x state 3x3 grid  (does the aux head follow IMAGE or STATE?)")
    print("=" * 84)
    for img_pos in POS:
        for st_pos in POS:
            row = make_row(
                f0["middle"],
                image=f0[img_pos]["observation.images.front"],
                state=f0[st_pos]["observation.state"],
            )
            p = aux(policy, preprocessor, capture, row)
            pan = float(predict(bundle, row, PROMPT)[0])
            show(f"img={img_pos:<6s} state={st_pos:<6s}", p, pan)

    print("\n[state-image probe] done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
