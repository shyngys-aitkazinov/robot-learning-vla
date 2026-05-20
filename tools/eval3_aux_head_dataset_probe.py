#!/usr/bin/env python3
"""Probe a SmolVLA aux-head checkpoint on real dataset episodes.

For each probed dataset it reports:
  1. aux position-head softmax (left/middle/right) vs the known print position,
     averaged over several episode-0 frames;
  2. cross-prompt aux sensitivity — same scene, all three celebrity prompts
     (if the prediction is frozen across prompts the head ignores language);
  3. open-loop action prediction vs ground truth across episode 0 — per-joint
     MAE + correlation, with the grasp + lift phase called out (gripper close,
     shoulder_lift / elbow rise).

`predict()` resets the policy each frame, so the trajectory is an open-loop
"if the arm were at this observation, what would it do next" comparison.

Usage:
  python tools/eval3_aux_head_dataset_probe.py \
      --checkpoint /ephemeral/outputs/train/eval3_3way_20k_b128_v6_pinned_idood_aux_fresh/checkpoints/005000/pretrained_model \
      --device cuda
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# The aux head only runs inside the training forward when its loss weight > 0.
os.environ.setdefault("EVAL3_AUX_POS_LOSS_WEIGHT", "0.5")

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

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
POS_NAMES = ["left", "middle", "right"]
PROMPTS = {
    "taylor_swift": "Place the coke on Taylor Swift",
    "yann_lecun": "Place the coke on Yann LeCun",
    "barack_obama": "Place the coke on Barack Obama",
}
# (local dir name, celebrity key, known print-position label: left=0 middle=1 right=2)
DEFAULT_PROBES = [
    ("dataset_v3_synth_pinned_taylor_swift_left_1", "taylor_swift", 0),
    ("dataset_v3_synth_pinned_taylor_swift_middle_1", "taylor_swift", 1),
    ("dataset_v3_synth_pinned_taylor_swift_right_1", "taylor_swift", 2),
]
STATS_REPO = "RobotLearningVLA/dataset_v3_synth_pinned_idood_taylor_swift_left_2"


def load_ep(name: str, root_dir: str, chunk_size: int, ep: int = 0) -> LeRobotDataset:
    return LeRobotDataset(
        f"RobotLearningVLA/{name}",
        root=str(Path(root_dir) / name),
        episodes=[ep],
        delta_timestamps={"action": [i / 30 for i in range(chunk_size)]},
        video_backend="pyav",
    )


def gt_action(row) -> np.ndarray:
    a = row["action"]
    a = a.detach().cpu().numpy() if isinstance(a, torch.Tensor) else np.asarray(a)
    return (a[0] if a.ndim == 2 else a).astype(np.float32)[:6]  # delta-0 = action at this frame


def aux_softmax(policy, preprocessor, row, prompt, label, capture, n=3):
    """Mean softmax over [left, middle, right] from the aux position head."""
    batch = {k: (v.unsqueeze(0) if isinstance(v, torch.Tensor) else [v]) for k, v in row.items()}
    batch["task"] = [prompt]
    base = preprocessor(batch)
    probs = []
    policy.eval()
    with torch.no_grad():
        for _ in range(n):
            base["target_position"] = torch.tensor([label], dtype=torch.long)
            capture.pop("logits", None)
            policy.forward(base)  # patched training forward -> runs aux head -> hook fires
            lg = capture.get("logits")
            if lg is not None:
                probs.append(torch.softmax(lg.float(), dim=-1)[0].cpu().numpy())
    return np.mean(probs, axis=0) if probs else None


def trajectory(bundle, ds, prompt, n_frames=14):
    n = len(ds)
    idxs = sorted(set(int(round(x)) for x in np.linspace(0, n - 1, min(n_frames, n))))
    gts, preds = [], []
    for i in idxs:
        row = ds[i]
        gts.append(gt_action(row))
        preds.append(predict(bundle, row, prompt))
    return idxs, np.stack(gts), np.stack(preds)


def corr(a: np.ndarray, b: np.ndarray) -> str:
    if a.std() < 0.5:
        return "~const"
    if b.std() < 1e-6:
        return "pred~flat"
    return f"{np.corrcoef(a, b)[0, 1]:+.2f}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--checkpoint",
        default="/ephemeral/outputs/train/eval3_3way_20k_b128_v6_pinned_idood_aux_fresh/checkpoints/005000/pretrained_model",
    )
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--datasets-root", default=str(_REPO / "datasets"))
    ap.add_argument("--aux-frames", type=int, default=5)
    ap.add_argument("--traj-frames", type=int, default=14)
    args = ap.parse_args()

    print(f"\n[probe] checkpoint = {args.checkpoint}")
    print(f"[probe] device     = {args.device}\n")
    bundle = load_policy_bundle(args.checkpoint, args.device, STATS_REPO)
    policy, preprocessor, postprocessor, device = bundle
    chunk = int(policy.model.config.chunk_size)

    capture: dict = {}
    policy.model.position_clf_head.register_forward_hook(
        lambda m, i, o: capture.__setitem__("logits", o.detach())
    )

    # ---- 1. aux-head position prediction (correct prompt) -------------------
    print("=" * 92)
    print("1. AUX POSITION HEAD  —  correct prompt, mean softmax over episode-0 frames")
    print("=" * 92)
    print(f"{'dataset':>46s} {'p_left':>8s} {'p_mid':>8s} {'p_right':>8s}  {'pred':>7s}  {'expect':>7s}  ok")
    print("-" * 92)
    episodes: dict[str, LeRobotDataset] = {}
    for name, celeb, label in DEFAULT_PROBES:
        ds = load_ep(name, args.datasets_root, chunk)
        episodes[name] = ds
        idxs = sorted(set(int(round(x)) for x in np.linspace(0, len(ds) - 1, args.aux_frames)))
        ps = [aux_softmax(policy, preprocessor, ds[i], PROMPTS[celeb], label, capture) for i in idxs]
        ps = [p for p in ps if p is not None]
        mean_p = np.mean(ps, axis=0)
        pred = int(np.argmax(mean_p))
        ok = "OK" if pred == label else "XX"
        print(
            f"{name:>46s} {mean_p[0]:>8.3f} {mean_p[1]:>8.3f} {mean_p[2]:>8.3f}  "
            f"{POS_NAMES[pred]:>7s}  {POS_NAMES[label]:>7s}  {ok}"
        )

    # ---- 2. cross-prompt aux sensitivity ------------------------------------
    xp_name, xp_celeb, xp_label = DEFAULT_PROBES[1]  # the *_middle_1 dataset
    ds_xp = episodes[xp_name]
    xp_idxs = sorted(set(int(round(x)) for x in np.linspace(0, len(ds_xp) - 1, args.aux_frames)))
    print()
    print("=" * 92)
    print(f"2. CROSS-PROMPT AUX  —  scene fixed ({xp_name}, target at {POS_NAMES[xp_label]});")
    print("   swap only the celebrity prompt. Frozen rows => head ignores language.")
    print("=" * 92)
    print(f"{'prompt':>28s} {'p_left':>8s} {'p_mid':>8s} {'p_right':>8s}  {'argmax':>7s}")
    print("-" * 92)
    for celeb, prompt in PROMPTS.items():
        ps = [aux_softmax(policy, preprocessor, ds_xp[i], prompt, xp_label, capture) for i in xp_idxs]
        ps = [p for p in ps if p is not None]
        mp = np.mean(ps, axis=0)
        print(f"{celeb:>28s} {mp[0]:>8.3f} {mp[1]:>8.3f} {mp[2]:>8.3f}  {POS_NAMES[int(np.argmax(mp))]:>7s}")

    # ---- 3. open-loop action trajectory vs ground truth ---------------------
    for name, celeb, label in DEFAULT_PROBES:
        ds = episodes[name]
        idxs, gt, pred = trajectory(bundle, ds, PROMPTS[celeb], args.traj_frames)
        mae = np.abs(gt - pred).mean(axis=0)
        print()
        print("=" * 92)
        print(f"3. TRAJECTORY  {name}   (episode 0, {len(ds)} frames, prompt='{PROMPTS[celeb]}')")
        print("=" * 92)
        print(f"{'joint':>14s}  {'GT range':>18s}  {'pred range':>18s}  {'MAE deg':>8s}  {'corr':>9s}")
        print("-" * 92)
        for j, jn in enumerate(JOINTS):
            print(
                f"{jn:>14s}  {gt[:,j].min():>+8.1f}..{gt[:,j].max():>+7.1f}  "
                f"{pred[:,j].min():>+8.1f}..{pred[:,j].max():>+7.1f}  "
                f"{mae[j]:>8.2f}  {corr(gt[:,j], pred[:,j]):>9s}"
            )
        # grasp/lift phase: GT gripper minimum = deepest grasp
        gi = int(np.argmin(gt[:, 5]))
        liftslice = slice(gi, len(idxs))
        print("-" * 92)
        print(
            f"  grasp (GT gripper min) at sampled frame idx {idxs[gi]} "
            f"(gripper {gt[gi,5]:+.1f} deg);  lift/place phase = last {len(idxs)-gi} sampled frames"
        )
        if len(idxs) - gi >= 2:
            mae_lift = np.abs(gt[liftslice] - pred[liftslice]).mean(axis=0)
            print(
                "  lift-phase MAE  "
                + "  ".join(f"{jn}={mae_lift[j]:.1f}" for j, jn in enumerate(JOINTS))
            )
        # per-frame pan + lift + elbow + gripper table
        print(f"\n  {'frame':>6s} | {'pan GT/pred':>16s} | {'lift GT/pred':>16s} | "
              f"{'elbow GT/pred':>16s} | {'grip GT/pred':>16s}")
        for k, fi in enumerate(idxs):
            tag = "  <-grasp" if k == gi else ""
            print(
                f"  {fi:>6d} | {gt[k,0]:>+7.1f}/{pred[k,0]:>+7.1f} | "
                f"{gt[k,1]:>+7.1f}/{pred[k,1]:>+7.1f} | "
                f"{gt[k,2]:>+7.1f}/{pred[k,2]:>+7.1f} | "
                f"{gt[k,5]:>+7.1f}/{pred[k,5]:>+7.1f}{tag}"
            )

    print("\n[probe] done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
