#!/usr/bin/env python3
"""Compare dataset observation keys to ``lerobot/smolvla_base`` and print train/deploy flags.

SmolVLA base expects three RGB streams named ``observation.images.camera{1,2,3}``.
Single-camera SO101 datasets often use ``observation.images.front``. At runtime the policy
fills missing camera keys with padded placeholders when ``policy.empty_cameras`` allows it.

Usage:
  python tools/eval3_smolvla_compat.py
  python tools/eval3_smolvla_compat.py --repo-id RobotLearningVLA/taylor_swift_1
"""

from __future__ import annotations

import argparse
import json


def _visual_keys_from_dataset_meta(repo_id: str, root: str | None) -> set[str]:
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata

    meta = LeRobotDatasetMetadata(repo_id, root=root)
    return {name for name in meta.features if name.startswith("observation.images.")}


def _visual_keys_smolvla_base() -> set[str]:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download("lerobot/smolvla_base", "config.json")
    with open(path) as f:
        cfg = json.load(f)
    feats = cfg.get("input_features") or {}
    return {k for k in feats if k.startswith("observation.images.")}


def main() -> None:
    ap = argparse.ArgumentParser(description="SmolVLA base vs dataset observation compatibility")
    ap.add_argument("--repo-id", default="RobotLearningVLA/taylor_swift_1")
    ap.add_argument("--dataset-root", default=None, help="Optional local root override")
    args = ap.parse_args()

    base_keys = _visual_keys_smolvla_base()
    ds_keys = _visual_keys_from_dataset_meta(args.repo_id, args.dataset_root)

    print("=== lerobot/smolvla_base image keys ===")
    for k in sorted(base_keys):
        print(f"  {k}")

    print(f"\n=== {args.repo_id} image keys ===")
    if not ds_keys:
        print("  (none — unexpected for Eval3)")
    for k in sorted(ds_keys):
        print(f"  {k}")

    print("\n=== diff ===")
    print(f"  base_minus_dataset: {sorted(base_keys - ds_keys)}")
    print(f"  dataset_minus_base: {sorted(ds_keys - base_keys)}")

    rename: dict[str, str] = {}
    if "observation.images.front" in ds_keys and "observation.images.camera1" in base_keys:
        rename["observation.images.front"] = "observation.images.camera1"

    provided_after_rename = {rename.get(k, k) for k in ds_keys}
    missing_after = sorted(base_keys - provided_after_rename)
    n_pad = len(missing_after)

    print("\n=== suggested training flags ===")
    if rename:
        rm = json.dumps(rename, separators=(",", ":"))
        print(f"  --rename_map='{rm}'")
    else:
        print("  (no front→camera1 rename applied automatically)")
    if n_pad > 0:
        print(f"  --policy.empty_cameras={n_pad}")
    elif base_keys != ds_keys and not rename:
        print("  (inspect manually — empty_cameras may still be required)")

    print("\n=== deploy (eval3_vla_deploy.py) ===")
    print("  Use the same rename_map so deploy matches train.")
    print('  Example: --rename_map \'{"observation.images.front":"observation.images.camera1"}\'')


if __name__ == "__main__":
    main()
