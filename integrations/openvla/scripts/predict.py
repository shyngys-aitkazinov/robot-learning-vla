#!/usr/bin/env python3
"""Load OpenVLA from Hugging Face and run predict_action on one image."""

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
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--instruction", type=str, required=True)
    p.add_argument(
        "--unnorm-key",
        type=str,
        required=True,
        help="Must match a dataset stats key bundled with the checkpoint (e.g. bridge_orig).",
    )
    p.add_argument("--model-id", type=str, default="openvla/openvla-7b")
    p.add_argument("--device", type=str, default="", help="cuda:0 | mps | cpu (empty = auto)")
    p.add_argument("--dtype", type=str, default="auto", choices=("auto", "bfloat16", "float16", "float32"))
    p.add_argument("--do-sample", action="store_true")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    device = pick_device(args.device)
    torch_dtype = pick_dtype(device, args.dtype)

    processor, vla = load_openvla(args.model_id, device=device, torch_dtype=torch_dtype)
    prompt = wrap_openvla_prompt(args.instruction)
    model_inputs = build_inputs(processor, prompt=prompt, image_path=args.image, device=device, torch_dtype=torch_dtype)
    arr = predict_action_numpy(vla, model_inputs, unnorm_key=args.unnorm_key, do_sample=args.do_sample)
    print(json.dumps({"dtype": str(arr.dtype), "shape": list(arr.shape), "action": arr.tolist()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
