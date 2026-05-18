#!/usr/bin/env python3
"""Print eval3_vla_deploy.py flags derived from a pretrained_model/train_config.json.

Usage::

    uv run python tools/eval3_deploy_flags_from_checkpoint.py \\
      outputs/train/eval3_v7_B_smolvla_new_old/checkpoints/040000/pretrained_model

Also accepts a Hugging Face model repo id if it exposes ``train_config.json`` at repo root.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "pretrained_model",
        type=str,
        help="Path to pretrained_model directory or Hugging Face model repo id.",
    )
    return ap.parse_args()


def _load_train_config(pretrained_model: str) -> dict:
    p = Path(pretrained_model)
    if p.is_dir() and (p / "train_config.json").is_file():
        return json.loads((p / "train_config.json").read_text(encoding="utf-8"))
    try:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(pretrained_model, "train_config.json")
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as exc:
        raise FileNotFoundError(
            f"Could not load train_config.json from {pretrained_model!r}: {exc}"
        ) from exc


def main() -> int:
    args = _parse_args()
    tc = _load_train_config(args.pretrained_model)

    ds_block = tc.get("dataset") or {}
    repo_id = ds_block.get("repo_id") if isinstance(ds_block, dict) else None
    rename_map = tc.get("rename_map") or {}
    policy = tc.get("policy") or {}
    empty_cameras = policy.get("empty_cameras")
    train_n_action = policy.get("n_action_steps")

    print("# Eval3 deploy flags (merge into eval3_vla_deploy.py CLI)")
    print("# Source: train_config dataset.repo_id + rename_map")
    if repo_id:
        print(f"--dataset_repo_id={repo_id}")
    else:
        print("# WARNING: dataset.repo_id missing in train_config")

    rename_json = json.dumps(rename_map, separators=(",", ":"))
    print(f"--rename_map='{rename_json}'")

    if empty_cameras is not None:
        print(f"--policy.empty_cameras={empty_cameras}")

    print("# Smoothing (recommended; train_config often has n_action_steps=50)")
    print("--policy.num_steps=20")
    print("--policy.n_action_steps=25")
    print("--interpolation_multiplier=2")
    print("--action_smoothing_alpha=0.25")
    print("--max_action_delta_deg=6")
    print("--gripper_open_bias_deg=5")
    print("--gripper_open_bias_threshold_deg=20")
    print("--fps=30 --episode_time_s=20")

    print("# Policy weights")
    print(f"--policy.path={args.pretrained_model}")

    print(f"# Training n_action_steps (reference): {train_n_action}")
    print(f"# Training num_steps / denoising steps (reference): {policy.get('num_steps')}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
