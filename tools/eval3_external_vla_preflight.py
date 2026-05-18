#!/usr/bin/env python3
"""Preflight gate for Eval 3 FlowerVLA/OpenVLA runs 7 and 8."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from eval3_external_vla_data import (  # noqa: E402
    ACTION_NAMES_7,
    RECIPES,
    Eval3ExternalVLADataset,
    collate_external_vla,
    get_recipe,
    recipe_manifest,
    write_openvla_dataset_statistics,
)


def _check_recipe(
    recipe_name: str,
    *,
    chunk_size: int,
    image_size: int,
    batch_size: int,
    skip_batch: bool,
    revision: str | None,
    video_backend: str,
) -> dict[str, Any]:
    recipe = get_recipe(recipe_name)
    manifest = recipe_manifest(recipe, revision=revision, video_backend=video_backend)
    errors: list[str] = []

    if manifest["num_frames"] != recipe.expected_frames:
        errors.append(f"frame_count {manifest['num_frames']} != expected {recipe.expected_frames}")
    if manifest["num_episodes"] != recipe.expected_episodes:
        errors.append(f"episode_count {manifest['num_episodes']} != expected {recipe.expected_episodes}")
    if tuple(manifest["action_names"]) != ACTION_NAMES_7:
        errors.append(f"action_names changed: {manifest['action_names']}")

    summary: dict[str, Any] = {
        **manifest,
        "chunk_size": int(chunk_size),
        "image_size": int(image_size),
        "no_gripper_repair": True,
        "no_action_smoothing": True,
        "no_extra_frame_cap": True,
        "batch": None,
        "errors": errors,
        "ok": not errors,
    }

    if skip_batch:
        return summary

    ds = Eval3ExternalVLADataset(
        recipe,
        chunk_size=chunk_size,
        image_size=image_size,
        task_mode="mixed",
        revision=revision,
        video_backend=video_backend,
        download_videos=True,
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_external_vla)
    batch = next(iter(loader))
    images = batch["images"]
    states = batch["states"]
    actions = batch["actions"]
    seventh_abs_max = float(actions[..., 6].abs().max().item())
    state_seventh_abs_max = float(states[..., 6].abs().max().item())

    batch_errors = []
    if tuple(images.shape[1:]) != (3, image_size, image_size):
        batch_errors.append(f"image_shape {tuple(images.shape)} does not end in (3,{image_size},{image_size})")
    if tuple(states.shape[1:]) != (7,):
        batch_errors.append(f"state_shape {tuple(states.shape)} does not end in (7,)")
    if tuple(actions.shape[1:]) != (chunk_size, 7):
        batch_errors.append(f"action_shape {tuple(actions.shape)} does not end in ({chunk_size},7)")
    if seventh_abs_max > 1e-6:
        batch_errors.append(f"padded action dim is nonzero: max_abs={seventh_abs_max:.6g}")
    if state_seventh_abs_max > 1e-6:
        batch_errors.append(f"padded state dim is nonzero: max_abs={state_seventh_abs_max:.6g}")

    summary["batch"] = {
        "image_shape": list(images.shape),
        "state_shape": list(states.shape),
        "action_shape": list(actions.shape),
        "tasks": batch["tasks"],
        "action_7th_abs_max": seventh_abs_max,
        "state_7th_abs_max": state_seventh_abs_max,
        "action_chunk_first_sample": actions[0].detach().cpu().tolist(),
        "action_valid_first_sample": batch["action_valid"][0].detach().cpu().tolist(),
    }
    summary["errors"].extend(batch_errors)
    summary["ok"] = not summary["errors"]
    return summary


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--recipe", choices=[*RECIPES.keys(), "all"], default="all")
    ap.add_argument("--chunk-size", type=int, default=10)
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--skip-batch", action="store_true", help="Only verify metadata/stats; do not decode video.")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--video-backend", default="pyav")
    ap.add_argument("--out", default="outputs/eval3_external_vla/preflight_summary.json")
    ap.add_argument(
        "--write-openvla-stats",
        default="",
        help="Optional path for OpenVLA dataset_statistics.json with eval3_so101_* keys.",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    recipe_names = list(RECIPES) if args.recipe == "all" else [args.recipe]
    summaries = [
        _check_recipe(
            name,
            chunk_size=args.chunk_size,
            image_size=args.image_size,
            batch_size=args.batch_size,
            skip_batch=args.skip_batch,
            revision=args.revision,
            video_backend=args.video_backend,
        )
        for name in recipe_names
    ]
    payload = {
        "recipes": summaries,
        "ok": all(x["ok"] for x in summaries),
    }

    if args.write_openvla_stats:
        payload["openvla_dataset_statistics_path"] = str(args.write_openvla_stats)
        write_openvla_dataset_statistics(
            recipe_names,
            args.write_openvla_stats,
            chunk_size=args.chunk_size,
            revision=args.revision,
            video_backend=args.video_backend,
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")

    print(json.dumps(payload, indent=2))
    if not payload["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    main()
