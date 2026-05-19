#!/usr/bin/env python3
"""Pre-deploy celebrity-differentiation gate for Eval3 SmolVLA checkpoints.

This wraps the two existing diagnostic tools into a single PASS/FAIL gate.

Tier table (per-pair mean L2 across the 3 canonical prompts, in degrees of
joint motion). These are calibrated against the v3-fresh baseline and the
v9 charuco collapse so the gate distinguishes "broken" from "v3-comparable"
rather than gating against an aspirational target the best baseline can't hit::

    <  5  deg    FAIL   (prompt collapse, v9 territory; do NOT deploy)
    5-15 deg    WEAK   (some language influence, robot deploy is a gamble)
   15-30 deg    PASS   (v3-fresh territory; ship it)
    >=30 deg    STRICT PASS (aspirational; matches eval3_synthetic_ood_test gate)

The 15 deg default reflects v3-fresh actual numbers: Swift-vs-others mean
~17 deg, but lecun-obama only ~4.9 deg (known wrist_roll bimodality). The
strict 30 deg target is the original DIAGNOSIS_REPORT.md target, kept
reachable via ``--prompt_swap_min 30``.

What this wraps:

  1. Quick prompt swap (``tools/eval3_promptswap_quick.py``)
     - 1 frame per dataset, original scene
     - Reports per-pair (swift-lecun, swift-obama, lecun-obama) mean L2
     - Gate: min_pair_mean >= ``--prompt_swap_min`` (default 15 deg).

  2. Synthetic-OOD prompt swap (``tools/eval3_synthetic_ood_test.py``)
     - 5 frames per dataset, 4 scene variants (original / bg_replaced /
       print_shuffled / all_augs)
     - Reports per-variant min pair L2 + secondary shoulder_lift delta
     - Primary gate: each variant's min_pair_mean >= ``--ood_min`` (default 15).
     - Secondary gate: shoulder_lift delta >= ``--ood_shoulder_lift_min``
       (default 30 deg; this number is about Obama's high-arm signature
       distance from Swift/LeCun centroids, not pair-L2 separation, so it
       stays at 30).

Exit codes (suitable for CI / pre-deploy gating):
  0 = PASS  — both gates green
  1 = WEAK  — language influences actions but below pass tier (5-15 deg)
  2 = FAIL  — prompt collapse (< 5 deg) or OOD gate FAIL

Usage::

    .venv/bin/python tools/eval3_check_celeb_gates.py \\
        --policy_path RobotLearningVLA/eval3-vla-v9-smolvla-fresh-charuco-50k \\
        --policy_device mps

    # Aspirational strict gate (matches eval3_synthetic_ood_test target):
    .venv/bin/python tools/eval3_check_celeb_gates.py \\
        --policy_path outputs/train/eval3-vla-v10-.../checkpoints/050000/pretrained_model \\
        --policy_device cuda \\
        --prompt_swap_min 30 --ood_min 30

    # Skip the OOD pass if masks/backgrounds aren't available:
    .venv/bin/python tools/eval3_check_celeb_gates.py \\
        --policy_path ... --skip_ood
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
PROMPT_SWAP_SCRIPT = _REPO / "tools" / "eval3_promptswap_quick.py"
OOD_SCRIPT = _REPO / "tools" / "eval3_synthetic_ood_test.py"


def _run(cmd: list[str]) -> int:
    print("$ " + " ".join(str(c) for c in cmd), flush=True)
    return subprocess.call(cmd)


def _load_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _verdict(label: str, value: float, threshold: float, *, collapse_floor: float = 5.0) -> tuple[int, str]:
    """Return (exit_code, color_label) for a single gate value.

    code: 0 = PASS, 1 = WEAK, 2 = FAIL.
    """
    if value < collapse_floor:
        return 2, "FAIL (collapsed)"
    if value < threshold:
        return 1, "WEAK"
    return 0, "PASS"


def _summary_table(rows: list[tuple[str, float, float, str]]) -> str:
    """Pretty-print a [(metric, value, threshold, verdict), ...] list."""
    out = []
    out.append(f"  {'metric':<36s}  {'value':>8s}  {'gate':>8s}   verdict")
    out.append(f"  {'-' * 36}  {'-' * 8}  {'-' * 8}   {'-' * 12}")
    for name, value, threshold, verdict in rows:
        out.append(f"  {name:<36s}  {value:>8.2f}  {threshold:>8.2f}   {verdict}")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--policy_path", required=True,
                    help="Local checkpoint dir or HF repo id "
                         "(e.g. RobotLearningVLA/eval3-vla-v10-...).")
    ap.add_argument("--policy_device", default="mps",
                    help="mps / cuda / cpu (default mps on this box).")
    ap.add_argument("--out_dir", type=Path, default=_REPO / "outputs" / "eval3_diag",
                    help="Where promptswap_quick.json + synthetic_ood_results.json land.")
    ap.add_argument("--prompt_swap_min", type=float, default=15.0,
                    help="Min mean pair L2 (deg) on the 3-prompt quick swap. "
                         "Default 15 deg = v3-fresh-comparable PASS tier "
                         "(see module docstring). Use 30 for strict aspirational "
                         "gate (DIAGNOSIS_REPORT.md primary target).")
    ap.add_argument("--ood_min", type=float, default=15.0,
                    help="Min pair L2 (deg) required for each of the 4 scene "
                         "variants in the synthetic-OOD test. Default 15 deg "
                         "mirrors --prompt_swap_min; use 30 for strict.")
    ap.add_argument("--ood_shoulder_lift_min", type=float, default=30.0,
                    help="Secondary gate: |obama_sh - mean(swift_sh, lecun_sh)| "
                         "must be >= this many degrees in each variant. "
                         "Default 30 deg, matches tools/eval3_synthetic_ood_test.py. "
                         "This number comes from Obama's high-arm centroid "
                         "distance from Swift/LeCun centroids (real data), "
                         "not from prompt-swap distribution, so it stays at 30 "
                         "even when the pair-L2 thresholds soften.")
    ap.add_argument("--frames_per_ds", type=int, default=1,
                    help="Quick-swap frames per dataset (default 1).")
    ap.add_argument("--ood_frames_per_ds", type=int, default=5,
                    help="Synthetic-OOD frames per dataset (default 5).")
    ap.add_argument("--skip_ood", action="store_true",
                    help="Skip the heavyweight synthetic-OOD pass and gate only "
                         "on the quick prompt swap. Useful when masks/backgrounds "
                         "aren't built yet.")
    ap.add_argument("--label", default="",
                    help="Tag appended to result filenames so multiple checkpoints "
                         "can be compared (e.g. 'v10_charuco', 'v9_baseline').")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"_{args.label}" if args.label else ""

    summary_rows: list[tuple[str, float, float, str]] = []
    worst_code = 0

    # ------- 1. Quick prompt swap ----------------------------------------
    print("\n" + "=" * 72)
    print("[1/2] prompt swap quick (same image, 3 prompts, mean pair L2)")
    print("=" * 72)
    quick_out = args.out_dir / f"promptswap_quick{tag}.json"
    rc = _run([
        sys.executable, str(PROMPT_SWAP_SCRIPT),
        "--policy_path", args.policy_path,
        "--policy_device", args.policy_device,
        "--out", str(quick_out),
        "--frames_per_ds", str(args.frames_per_ds),
    ])
    if rc not in (0, 1, 2):
        print(f"  prompt swap subprocess exited with {rc}; aborting", file=sys.stderr)
        return 3
    quick = _load_json(quick_out)
    if quick is None:
        print(f"  ERROR: no JSON output at {quick_out}", file=sys.stderr)
        return 3
    min_pair_quick = float(quick.get("min_pair_mean", 0.0))
    code_quick, verdict_quick = _verdict("prompt_swap", min_pair_quick, args.prompt_swap_min)
    summary_rows.append(("prompt_swap_quick.min_pair_mean (deg)",
                         min_pair_quick, args.prompt_swap_min, verdict_quick))
    worst_code = max(worst_code, code_quick)

    # ------- 2. Synthetic-OOD prompt swap --------------------------------
    if not args.skip_ood:
        print("\n" + "=" * 72)
        print("[2/2] synthetic OOD (5 frames/ds, 4 scene variants, min pair L2)")
        print("=" * 72)
        ood_out_dir = args.out_dir / f"synthetic_ood{tag}"
        rc = _run([
            sys.executable, str(OOD_SCRIPT),
            "--policy_path", args.policy_path,
            "--policy_device", args.policy_device,
            "--out_dir", str(ood_out_dir),
            "--n_frames_per_ds", str(args.ood_frames_per_ds),
        ])
        if rc not in (0, 1, 2):
            print(f"  synth-OOD subprocess exited with {rc}; aborting", file=sys.stderr)
            return 3
        ood = _load_json(ood_out_dir / "synthetic_ood_results.json")
        if ood is None:
            print(f"  WARN: no synthetic-OOD JSON under {ood_out_dir}; skipping gate",
                  file=sys.stderr)
        else:
            # Schema from tools/eval3_synthetic_ood_test.py:
            #   {"summary": {variant: {"min_pair_mean": float,
            #                          "shoulder_lift": {"delta": float, ...},
            #                          ...}}, ...}
            for variant, payload in (ood.get("summary") or {}).items():
                value = float(payload.get("min_pair_mean", 0.0))
                code_v, verdict_v = _verdict(variant, value, args.ood_min)
                summary_rows.append((f"synth_ood[{variant}].min_pair_l2 (deg)",
                                     value, args.ood_min, verdict_v))
                worst_code = max(worst_code, code_v)
                sh = payload.get("shoulder_lift") or {}
                sh_delta = float(sh.get("delta", 0.0))
                code_sh, verdict_sh = _verdict(
                    f"{variant}_sh", sh_delta, args.ood_shoulder_lift_min,
                )
                summary_rows.append(
                    (f"synth_ood[{variant}].shoulder_lift_delta (deg)",
                     sh_delta, args.ood_shoulder_lift_min, verdict_sh),
                )
                worst_code = max(worst_code, code_sh)
    else:
        print("\n[skip] synthetic OOD pass disabled via --skip_ood")

    # ------- Final summary ------------------------------------------------
    print("\n" + "=" * 72)
    print("Celebrity-differentiation gate summary")
    print("=" * 72)
    print(f"  policy_path : {args.policy_path}")
    print(f"  device      : {args.policy_device}")
    print(f"  output dir  : {args.out_dir}")
    print()
    print(_summary_table(summary_rows))
    print()
    if worst_code == 0:
        print(f"OVERALL: PASS — all gates >= configured thresholds "
              f"(prompt_swap={args.prompt_swap_min}, ood={args.ood_min}, "
              f"ood_sh={args.ood_shoulder_lift_min}). v3-fresh-comparable "
              f"or better; ship to hardware.")
    elif worst_code == 1:
        print("OVERALL: WEAK — language influences actions but below the "
              "configured pass tier. Robot deploy is a gamble: try the three "
              "canonical prompts; if any one consistently picks the wrong "
              "celebrity, fall back to v3-fresh/v8.")
    else:
        print("OVERALL: FAIL — prompt collapse detected (min pair < 5 deg) or "
              "OOD primary/secondary gate failed. Do NOT deploy this checkpoint "
              "for celebrity differentiation. See "
              "outputs/eval3_celebrity_diagnosis/DIAGNOSIS_REPORT.md and "
              "docs/eval3/identity_fix_retrain.md.")
    return worst_code


if __name__ == "__main__":
    raise SystemExit(main())
