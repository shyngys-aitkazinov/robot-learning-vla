#!/usr/bin/env python3
"""Temporal-sweep mean-seeking probe for a trained SmolVLA policy.

Proves (or refutes) the hypothesis that the policy ignores the language prompt
and collapses to the *mean* of the three left / middle / right trajectory
families.

Method
------
For one celebrity, load its three position datasets (left / middle / right) —
the exact synthetic data the policy trained on. For episode 0 of each, sweep
frames across the whole episode (t=0 .. end). At each frame:

  * run the policy under ALL THREE celebrity prompts -> 3 predicted actions.
    Δ_xprompt = max-min of predicted shoulder_pan. ~0 => the prompt is ignored.
  * compare the policy's prediction under the dataset's OWN correct prompt to
    (a) that dataset's GT pan and (b) the mean of the L/M/R GT pans at the same
    point in the episode. pred ≈ mean and pred ≠ own-GT  =>  mean-seeking.

Why sweep instead of testing only t=0
-------------------------------------
At t=0 the arm is at home and the first action is the shared "go grasp the
can" motion — identical for every target — so a near-zero Δ at t=0 proves
nothing (a perfect language-conditioned model looks the same there). The
bifurcation, where the L/M/R trajectories diverge, is mid-episode during the
carry. The sweep locates it and reports Δ THERE.

Why training data
-----------------
This is an optimisation (underfitting) diagnosis, not a generalisation one. If
the policy predicts the mean on the exact frames it was optimised on for
thousands of steps, the mean-seeking shortcut is proven — it is not a
held-out generalisation gap.

The reported number is predicted action step 0 (the first step of the action
chunk), post-processed, in degrees. The mean-seeking conclusion is identical
for the full chunk: the chunk is a smooth rollout committed from this step.

Usage
-----
  python tools/eval3_t0_mean_seeking_probe.py \
      --checkpoint RobotLearningVLA/eval3-smolvla-3way-5k-b128-slot-bottleneck-step3k \
      --celeb taylor_swift --device cpu
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "scripts"))
sys.path.insert(0, str(_REPO / "tools"))

import eval3_lerobot_shim  # noqa: E402

eval3_lerobot_shim.apply()

# Trained slot checkpoints carry model.slot_clf.* weights. Force the slot patch
# on (apply() is a no-op unless the weight is > 0) so the SmolVLA module gains
# the slot_clf submodule and the checkpoint's state_dict loads cleanly. At
# inference the slot CE loss is skipped (no target_position); the head just
# contributes the h_slot prefix token exactly as it did at train time.
os.environ.setdefault("EVAL3_SLOT_LOSS_WEIGHT", "0.5")
import eval3_smolvla_slot_bottleneck  # noqa: E402

eval3_smolvla_slot_bottleneck.apply()

import numpy as np  # noqa: E402

from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from eval3_abcd_benchmark import load_policy_bundle, predict  # noqa: E402


PAN = 0  # shoulder_pan index; +pan = LEFT, -pan = RIGHT
PROMPTS = {
    "taylor_swift": "Place the coke on Taylor Swift",
    "yann_lecun": "Place the coke on Yann LeCun",
    "barack_obama": "Place the coke on Barack Obama",
}
POSITIONS = ("left", "middle", "right")
T_FRACS = [0.0, 0.15, 0.30, 0.45, 0.60, 0.75, 0.90]


def repo_for(celeb: str, position: str) -> str:
    return f"RobotLearningVLA/dataset_v3_synth_pinned_idood_{celeb}_{position}_2"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True, help="HF repo id or local pretrained_model path")
    ap.add_argument("--celeb", default="taylor_swift",
                    choices=list(PROMPTS.keys()))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--episode", type=int, default=0)
    args = ap.parse_args()

    own_prompt = PROMPTS[args.celeb]
    print(f"\n[mean-seeking probe] checkpoint = {args.checkpoint}")
    print(f"[mean-seeking probe] celebrity  = {args.celeb}   device = {args.device}")
    print(f"[mean-seeking probe] reported value = predicted action step 0 (degrees)\n")

    stats_repo = repo_for(args.celeb, "left")
    bundle = load_policy_bundle(args.checkpoint, args.device, stats_repo)

    # results[position] = list of dicts {t_frac, frame, gt_pan, preds:{prompt_key:pan}}
    results: dict[str, list[dict]] = {}
    for pos in POSITIONS:
        repo = repo_for(args.celeb, pos)
        ds = LeRobotDataset(repo, episodes=[args.episode], revision="v3.0", video_backend="pyav")
        n = len(ds)
        rows = []
        for tf in T_FRACS:
            fi = int(round(tf * (n - 1)))
            row = ds[fi]
            gt_pan = float(np.asarray(row["action"]).reshape(-1)[PAN])
            preds = {}
            for ck, prompt in PROMPTS.items():
                preds[ck] = float(predict(bundle, row, prompt)[PAN])
            rows.append({"t_frac": tf, "frame": fi, "gt_pan": gt_pan, "preds": preds})
        results[pos] = rows
        print(f"  loaded {repo}  episode {args.episode}: {n} frames")

    # ---- per-position tables -------------------------------------------------
    for pos in POSITIONS:
        print(f"\n=== {args.celeb}_{pos}  (episode {args.episode}) ===")
        print(f"{'t_frac':>7s} {'frame':>6s} {'GT_pan':>8s} "
              f"{'p[swift]':>9s} {'p[lecun]':>9s} {'p[obama]':>9s} "
              f"{'Δ_xprompt':>10s} {'pred_own':>9s} {'own-GT':>8s}")
        for r in results[pos]:
            p = r["preds"]
            vals = [p["taylor_swift"], p["yann_lecun"], p["barack_obama"]]
            dxp = max(vals) - min(vals)
            own = p[args.celeb]
            print(f"{r['t_frac']:>7.2f} {r['frame']:>6d} {r['gt_pan']:>+8.2f} "
                  f"{p['taylor_swift']:>+9.2f} {p['yann_lecun']:>+9.2f} {p['barack_obama']:>+9.2f} "
                  f"{dxp:>10.2f} {own:>+9.2f} {own - r['gt_pan']:>+8.2f}")

    # ---- cross-position summary: does model spread match GT spread? ----------
    print(f"\n=== cross-position summary (does the policy diverge L/M/R like the GT does?) ===")
    print(f"{'t_frac':>7s} {'GT_L':>7s} {'GT_M':>7s} {'GT_R':>7s} {'GT_spread':>10s}  "
          f"{'mdl_L':>7s} {'mdl_M':>7s} {'mdl_R':>7s} {'mdl_spread':>11s} {'GT_mean':>8s}")
    summary = []
    for i, tf in enumerate(T_FRACS):
        gt = {pos: results[pos][i]["gt_pan"] for pos in POSITIONS}
        # model prediction under each dataset's OWN correct prompt
        mdl = {pos: results[pos][i]["preds"][args.celeb] for pos in POSITIONS}
        gt_spread = max(gt.values()) - min(gt.values())
        mdl_spread = max(mdl.values()) - min(mdl.values())
        gt_mean = sum(gt.values()) / 3.0
        summary.append({"t_frac": tf, "gt": gt, "mdl": mdl,
                        "gt_spread": gt_spread, "mdl_spread": mdl_spread, "gt_mean": gt_mean})
        print(f"{tf:>7.2f} {gt['left']:>+7.1f} {gt['middle']:>+7.1f} {gt['right']:>+7.1f} "
              f"{gt_spread:>10.2f}  {mdl['left']:>+7.1f} {mdl['middle']:>+7.1f} {mdl['right']:>+7.1f} "
              f"{mdl_spread:>11.2f} {gt_mean:>+8.2f}")

    # ---- verdict -------------------------------------------------------------
    bif = max(summary, key=lambda s: s["gt_spread"])
    # cross-prompt Δ at the bifurcation, averaged over the 3 position datasets
    bi = T_FRACS.index(bif["t_frac"])
    xprompt_deltas = []
    meanseek_errs = []
    for pos in POSITIONS:
        r = results[pos][bi]
        vals = list(r["preds"].values())
        xprompt_deltas.append(max(vals) - min(vals))
        own = r["preds"][args.celeb]
        # mean-seeking error: |pred - own_GT| vs |pred - workspace_mean|
        meanseek_errs.append((pos, own, r["gt_pan"], bif["gt_mean"]))

    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    print(f"  bifurcation point: t_frac={bif['t_frac']:.2f}  "
          f"(GT spreads {bif['gt_spread']:.1f}° across L/M/R)")
    print(f"  policy cross-prompt Δ there (avg over L/M/R datasets): "
          f"{sum(xprompt_deltas) / 3.0:.2f}°   [healthy ≥ 20°]")
    print(f"  policy output spread across L/M/R there (own prompt): "
          f"{bif['mdl_spread']:.2f}°   [should match GT spread {bif['gt_spread']:.1f}°]")
    print(f"  workspace mean of the 3 GT trajectories at bifurcation: {bif['gt_mean']:+.2f}°")
    print(f"\n  mean-seeking check at the bifurcation frame:")
    print(f"    {'dataset':>16s} {'pred_own':>9s} {'own_GT':>8s} {'mean_GT':>8s} "
          f"{'|pred-GT|':>10s} {'|pred-mean|':>12s}  verdict")
    for pos, pred, own_gt, mean_gt in meanseek_errs:
        d_gt = abs(pred - own_gt)
        d_mean = abs(pred - mean_gt)
        verdict = "MEAN-SEEKING" if d_mean < d_gt else "tracks GT"
        print(f"    {args.celeb + '_' + pos:>16s} {pred:>+9.2f} {own_gt:>+8.2f} {mean_gt:>+8.2f} "
              f"{d_gt:>10.2f} {d_mean:>12.2f}  {verdict}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
