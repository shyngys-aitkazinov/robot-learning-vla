#!/usr/bin/env python3
"""Offline probe: measure OpenVLA action spread across prompts for many images.

Does **not** import LeRobot — provide PNG paths via ``--manifest`` (JSON list of objects with
``image`` + ``instruction`` fields) or ``--images-glob`` paired with ``--prompts`` (comma-separated).

Writes Markdown summary under ``integrations/openvla/outputs/`` (created automatically).
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from openvla_eval3.inference import (
    build_inputs,
    load_openvla,
    pick_device,
    pick_dtype,
    predict_action_numpy,
)
from openvla_eval3.prompts import wrap_openvla_prompt


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--manifest", type=Path, help="JSON list of {image, instruction}")
    g.add_argument("--images-glob", type=str, help="Glob of images (same prompt set applied to each)")
    p.add_argument(
        "--prompts",
        type=str,
        default="",
        help="Comma-separated prompts used with --images-glob (same prompts evaluated per image)",
    )
    p.add_argument("--unnorm-key", type=str, required=True)
    p.add_argument("--model-id", type=str, default="openvla/openvla-7b")
    p.add_argument("--device", type=str, default="")
    p.add_argument("--dtype", type=str, default="auto", choices=("auto", "bfloat16", "float16", "float32"))
    p.add_argument("--output-md", type=Path, default=None)
    return p.parse_args()


def _load_manifest(path: Path) -> list[tuple[Path, str]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for row in rows:
        out.append((Path(row["image"]), str(row["instruction"])))
    return out


def _load_glob_pairs(pattern: str, prompts_csv: str) -> list[tuple[Path, str]]:
    prompts = [x.strip() for x in prompts_csv.split(",") if x.strip()]
    if not prompts:
        raise ValueError("--prompts required with --images-glob")
    paths = sorted(Path(p) for p in glob.glob(pattern))
    pairs: list[tuple[Path, str]] = []
    for img in paths:
        for pr in prompts:
            pairs.append((img, pr))
    return pairs


def main() -> int:
    args = _parse_args()
    if args.manifest:
        pairs = _load_manifest(args.manifest)
    else:
        pairs = _load_glob_pairs(args.images_glob, args.prompts)

    device = pick_device(args.device)
    torch_dtype = pick_dtype(device, args.dtype)
    processor, vla = load_openvla(args.model_id, device=device, torch_dtype=torch_dtype)

    # Group by image path for variance stats
    by_image: dict[str, dict[str, np.ndarray]] = {}
    for img_path, instruction in pairs:
        key = str(img_path.resolve())
        prompt = wrap_openvla_prompt(instruction)
        model_inputs = build_inputs(processor, prompt=prompt, image_path=img_path, device=device, torch_dtype=torch_dtype)
        arr = predict_action_numpy(vla, model_inputs, unnorm_key=args.unnorm_key, do_sample=False)
        by_image.setdefault(key, {})[instruction] = arr

    lines = [
        "# OpenVLA offline probe",
        "",
        f"- model_id: `{args.model_id}`",
        f"- unnorm_key: `{args.unnorm_key}`",
        "",
        "## Per-image prompt spread (L2 between first two prompts if available)",
        "",
        "| image | n_prompts | pairwise_mean_l2 |",
        "|---|---:|---:|",
    ]

    for img_key, pmap in sorted(by_image.items()):
        acts = list(pmap.values())
        if len(acts) < 2:
            l2 = float("nan")
        else:
            deltas = []
            for i in range(len(acts)):
                for j in range(i + 1, len(acts)):
                    deltas.append(float(np.linalg.norm(acts[i] - acts[j])))
            l2 = float(np.mean(deltas)) if deltas else float("nan")
        lines.append(f"| `{Path(img_key).name}` | {len(pmap)} | {l2:.4f} |")

    out_md = args.output_md or (ROOT / "outputs" / "openvla_offline_probe.md")
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
