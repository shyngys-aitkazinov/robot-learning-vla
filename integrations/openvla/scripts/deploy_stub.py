#!/usr/bin/env python3
"""Timing-only closed loop over offline PNG frames — logs JSONL, never touches motors."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
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
    p.add_argument("--frames-dir", type=Path, required=True, help="Directory of PNG/JPEG frames (sorted by name)")
    p.add_argument("--instruction", type=str, required=True)
    p.add_argument("--unnorm-key", type=str, required=True)
    p.add_argument("--model-id", type=str, default="openvla/openvla-7b")
    p.add_argument("--fps", type=float, default=5.0, help="Target loop rate (may miss if inference slower)")
    p.add_argument("--device", type=str, default="")
    p.add_argument("--dtype", type=str, default="auto", choices=("auto", "bfloat16", "float16", "float32"))
    p.add_argument(
        "--output-jsonl",
        type=Path,
        default=None,
        help="Default: ../../outputs/eval3_rollouts/openvla_stub_<UTC>.jsonl relative to repo root",
    )
    return p.parse_args()


def _frame_paths(d: Path) -> list[Path]:
    exts = {".png", ".jpg", ".jpeg", ".webp"}
    files = sorted([p for p in d.iterdir() if p.suffix.lower() in exts])
    if not files:
        raise FileNotFoundError(f"No images found under {d}")
    return files


def main() -> int:
    args = _parse_args()
    repo_root = ROOT.parent.parent
    out_dir = repo_root / "outputs" / "eval3_rollouts"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = args.output_jsonl or (out_dir / f"openvla_stub_{ts}.jsonl")

    device = pick_device(args.device)
    torch_dtype = pick_dtype(device, args.dtype)
    processor, vla = load_openvla(args.model_id, device=device, torch_dtype=torch_dtype)
    prompt = wrap_openvla_prompt(args.instruction)

    frames = _frame_paths(args.frames_dir)
    header = {
        "kind": "openvla_deploy_stub",
        "model_id": args.model_id,
        "unnorm_key": args.unnorm_key,
        "instruction": args.instruction,
        "prompt_wrapped": prompt,
        "frames_dir": str(args.frames_dir.resolve()),
        "n_frames": len(frames),
        "target_fps": args.fps,
    }

    interval = 1.0 / max(args.fps, 1e-3)
    with out_path.open("w", encoding="utf-8") as log:
        log.write(json.dumps(header) + "\n")
        ep_start = time.perf_counter()
        for step, fp in enumerate(frames):
            t0 = time.perf_counter()
            model_inputs = build_inputs(processor, prompt=prompt, image_path=fp, device=device, torch_dtype=torch_dtype)
            arr = predict_action_numpy(vla, model_inputs, unnorm_key=args.unnorm_key, do_sample=False)
            dt = time.perf_counter() - t0
            row = {
                "step": step,
                "frame": str(fp.name),
                "dt_s": dt,
                "loop_hz": 1.0 / dt if dt > 0 else None,
                "action": arr.tolist(),
            }
            log.write(json.dumps(row) + "\n")
            log.flush()
            sleep_s = interval - dt
            if sleep_s > 0:
                time.sleep(sleep_s)

        total = time.perf_counter() - ep_start
        log.write(json.dumps({"summary": {"wall_s": total, "avg_hz": len(frames) / total if total > 0 else None}}) + "\n")

    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
