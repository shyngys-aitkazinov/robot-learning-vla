#!/usr/bin/env python3
"""Convenience wrapper: Eval3-style defaults around scripts/predict.py logic."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

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
    p.add_argument("--image", type=Path, required=True, help="PNG/JPEG frame (e.g. from LeRobot camera capture)")
    p.add_argument(
        "--instruction",
        type=str,
        default="Place the coke on Taylor Swift",
        help="Eval 3 instruction text before OpenVLA wrapping.",
    )
    p.add_argument("--unnorm-key", type=str, required=True)
    p.add_argument("--model-id", type=str, default="openvla/openvla-7b")
    p.add_argument("--device", type=str, default="")
    p.add_argument("--dtype", type=str, default="auto", choices=("auto", "bfloat16", "float16", "float32"))
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    device = pick_device(args.device)
    torch_dtype = pick_dtype(device, args.dtype)
    processor, vla = load_openvla(args.model_id, device=device, torch_dtype=torch_dtype)
    prompt = wrap_openvla_prompt(args.instruction)
    model_inputs = build_inputs(processor, prompt=prompt, image_path=args.image, device=device, torch_dtype=torch_dtype)
    arr = predict_action_numpy(vla, model_inputs, unnorm_key=args.unnorm_key, do_sample=False)
    print(json.dumps({"instruction_wrapped": prompt, "result": arr.tolist()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
