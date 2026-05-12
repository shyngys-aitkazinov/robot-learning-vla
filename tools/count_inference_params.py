#!/usr/bin/env python3
"""Count trainable parameters for bonus tracking / reporting.

Supports pickles saved by scripts/train_eval3_bc_overfit.py (dict with model_state).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True, help=".pt from BC script")
    parser.add_argument("--meta", type=Path, default=None, help="eval3_bc_meta.json")
    args = parser.parse_args()

    import torch

    blob = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if isinstance(blob, dict) and "model_state" in blob:
        state = blob["model_state"]
    elif isinstance(blob, dict) and "state_dict" in blob:
        state = blob["state_dict"]
    elif isinstance(blob, dict):
        state = blob
    else:
        raise ValueError("Unrecognized checkpoint format")

    total = sum(v.numel() for v in state.values() if hasattr(v, "numel"))
    print(f"checkpoint: {args.checkpoint}")
    print(f"tensor_entries: {len(state)}")
    print(f"total_param_elements: {total:,}")

    meta_path = args.meta
    if meta_path is None:
        cand = args.checkpoint.parent / "eval3_bc_meta.json"
        if cand.is_file():
            meta_path = cand
    if meta_path is not None and meta_path.is_file():
        meta = json.loads(meta_path.read_text())
        print("meta excerpt:")
        print(json.dumps(meta, indent=2)[:4000])


if __name__ == "__main__":
    main()
