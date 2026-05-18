#!/usr/bin/env python3
"""One-off helper: upload outputs/train checkpoints to RobotLearningVLA model repos.

Run from repo root:  uv run python tools/hf_upload_train_checkpoints.py
Uses .venv/bin/hf (not pyenv) for correct LFS handling.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HF = ROOT / ".venv" / "bin" / "hf"

# (job_folder, step_dir, hub_repo_name_without_org)
REPO_RULES: list[tuple[str, str, str]] = [
    # v6 cloud 50k (dataset_v2 / Taylor naming experiments)
    ("v6-smolvla-fresh-combined88", "050000", "eval3-vla-v6-smolvla-fresh-combined88-50k"),
    ("v6-smolvla-fresh-new66", "050000", "eval3-vla-v6-smolvla-fresh-new66-50k"),
    # 3-way 50k published naming
    ("eval3_3way_50k_aug", "050000", "eval3-smolvla-3way-50k-aug-v1"),
    ("eval3_3way_50k_v3_fresh", "050000", "eval3-smolvla-3way-50k-v3-fresh"),
    ("eval3_3way_50k_v5_newdata_balanced", "050000", "eval3-smolvla-3way-v5-newdata-balanced"),
    ("eval3_3way_aug_smoke", "000200", "eval3-smolvla-3way-aug-smoke-200"),
    # small sanity / smoke
    ("eval3_smoke_fresh", "000020", "eval3-vla-sanity-smoke-fresh-20"),
    ("eval3_smoke_fresh2", "000020", "eval3-vla-sanity-smoke-fresh2-20"),
    ("eval3_smoke_test", "000020", "eval3-vla-sanity-smoke-test-20"),
    ("eval3_smolvla_500", "000500", "eval3-vla-sanity-smolvla-500"),
    ("eval3_smolvla_smoke", "000020", "eval3-vla-sanity-smolvla-smoke-20"),
    ("eval3_swift_lecun_8k", "008000", "eval3-vla-swift-lecun-8k"),
    ("eval3_v6_smoke_fresh_20260517", "000020", "eval3-vla-v6-smoke-fresh-20260517-20"),
    ("eval3_v6_smoke_warm_20260517", "000020", "eval3-vla-v6-smoke-warm-20260517-20"),
    ("v6_smoke", "000010", "eval3-vla-v6-smoke-10"),
    ("v6_smoke2", "000005", "eval3-vla-v6-smoke2-5"),
    # v7 Train A
    ("eval3_v7_A_smolvla_new", "010000", "eval3-vla-v7-A-smolvla-new-10k"),
    ("eval3_v7_A_smolvla_new", "020000", "eval3-vla-v7-A-smolvla-new-20k"),
    ("eval3_v7_A_smolvla_new", "030000", "eval3-vla-v7-A-smolvla-new-30k"),
    ("eval3_v7_A_smolvla_new", "040000", "eval3-vla-v7-A-smolvla-new-40k"),
    ("eval3_v7_A_smolvla_new", "050000", "eval3-vla-v7-A-smolvla-new-50k"),
    # v7 Train B
    ("eval3_v7_B_smolvla_new_old", "010000", "eval3-vla-v7-B-smolvla-new-old-10k"),
    ("eval3_v7_B_smolvla_new_old", "020000", "eval3-vla-v7-B-smolvla-new-old-20k"),
    ("eval3_v7_B_smolvla_new_old", "030000", "eval3-vla-v7-B-smolvla-new-old-30k"),
    ("eval3_v7_B_smolvla_new_old", "040000", "eval3-vla-v7-B-smolvla-new-old-40k"),
    ("eval3_v7_B_smolvla_new_old", "050000", "eval3-vla-v7-B-smolvla-new-old-50k"),
    # v7 Train C
    ("eval3_v7_C_warm_v3_new", "003000", "eval3-vla-v7-C-warm-v3-3k"),
    ("eval3_v7_C_warm_v3_new", "006000", "eval3-vla-v7-C-warm-v3-6k"),
    ("eval3_v7_C_warm_v3_new", "009000", "eval3-vla-v7-C-warm-v3-9k"),
    ("eval3_v7_C_warm_v3_new", "012000", "eval3-vla-v7-C-warm-v3-12k"),
    # v7 Train D (Obama-only) — step digits match folder name
    ("eval3_v7_D_obama_only", "002500", "eval3-vla-v7-D-obama-only-2500"),
    ("eval3_v7_D_obama_only", "005000", "eval3-vla-v7-D-obama-only-5000"),
    ("eval3_v7_D_obama_only", "007500", "eval3-vla-v7-D-obama-only-7500"),
    ("eval3_v7_D_obama_only", "010000", "eval3-vla-v7-D-obama-only-10000"),
    ("eval3_v7_D_obama_only", "012500", "eval3-vla-v7-D-obama-only-12500"),
]


def main() -> int:
    if not HF.is_file():
        print("Missing", HF, file=sys.stderr)
        return 1
    fails: list[str] = []
    for job, step, name in REPO_RULES:
        folder = ROOT / "outputs/train" / job / "checkpoints" / step / "pretrained_model"
        rid = f"RobotLearningVLA/{name}"
        if not folder.is_dir() or not (folder / "model.safetensors").is_file():
            if folder.parent.is_dir():
                print("SKIP missing weights:", folder)
            else:
                print("SKIP no checkpoint dir:", folder.parent)
            continue
        subprocess.run(
            [str(HF), "repo", "create", rid, "--private", "--exist-ok", "--repo-type", "model"],
            check=False,
        )
        msg = f"Upload {job} step {step} ({name})"
        print("UPLOAD", rid, "←", folder, flush=True)
        p = subprocess.run(
            [
                str(HF),
                "upload",
                rid,
                str(folder),
                ".",
                "--repo-type",
                "model",
                "--commit-message",
                msg,
            ]
        )
        if p.returncode != 0:
            fails.append(rid)
            print("FAILED", rid, file=sys.stderr)
    if fails:
        print("Failures:", len(fails), file=sys.stderr)
        for r in fails:
            print(" ", r, file=sys.stderr)
        return 1
    print("All uploads attempted successfully (check Hub for skipped-identical commits).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
