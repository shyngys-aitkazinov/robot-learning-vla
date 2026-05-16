#!/usr/bin/env python3
"""Eval 3 demo CLI: stdin instruction + Enter, then 20s timed policy loop.

Modes:
  --mock-dataset: reuse frame(s) from LeRobotDataset as a stationary observation (software test).
  --dry-run: no policy; only timing + logging.

Robot bridge is team-owned — wire live observations into this loop when ready (docs/eval3/rollout.md).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torchvision.transforms.functional import resize as tv_resize

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from eval3_device import resolve_eval3_device  # noqa: E402
from eval3_models import Eval3TinyBC  # noqa: E402


def load_mock_frame(
    repo_id: str,
    frame_index: int,
    image_key: str,
    state_key: str | None,
    image_size: int,
    video_backend: str = "pyav",
):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(repo_id, video_backend=video_backend)
    idx = max(0, min(frame_index, len(ds) - 1))
    row = ds[idx]
    img = row[image_key]
    if img.dim() == 3 and img.shape[-1] == 3:
        img = img.permute(2, 0, 1)
    img = img.float()
    if img.max() > 1.5:
        img = img / 255.0
    img = tv_resize(img, [image_size, image_size]).unsqueeze(0)
    if state_key and state_key in row:
        state = row[state_key].float().flatten().unsqueeze(0)
    else:
        state = torch.zeros(1, 1)
    return img, state


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration-s", type=float, default=20.0)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument(
        "--device",
        default="auto",
        help="auto (mps on Apple Silicon when available), or cpu | mps | cuda",
    )
    ap.add_argument("--policy-path", type=Path, default=None, help="best.pt from train_eval3_bc_overfit")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--mock-dataset-repo", type=str, default="RobotLearningVLA/taylor_swift_1")
    ap.add_argument("--mock-frame-index", type=int, default=0)
    ap.add_argument("--video-backend", default="pyav", help="Passed to LeRobotDataset for mock frames")
    args = ap.parse_args()
    device = resolve_eval3_device(args.device)

    if sys.stdin.isatty():
        print("Enter instruction (single line), then press Enter:", file=sys.stderr)

    instruction = sys.stdin.readline()
    if not instruction:
        print("EOF with no instruction", file=sys.stderr)
        sys.exit(2)
    instruction = instruction.strip()
    if not instruction:
        print("Empty instruction", file=sys.stderr)
        sys.exit(2)

    out_dir = Path("outputs/eval3_rollouts")
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    log_path = out_dir / f"rollout_{stamp}.jsonl"

    policy = None
    meta: dict = {}
    mock_img = mock_state = None
    resolved_mock_repo: str | None = None

    if args.policy_path and args.policy_path.is_file() and not args.dry_run:
        blob = torch.load(args.policy_path, map_location="cpu", weights_only=False)
        meta = blob.get("meta", {})
        if not meta:
            raise RuntimeError("Checkpoint missing 'meta' dict; use outputs from train_eval3_bc_overfit.py")
        policy = Eval3TinyBC(
            action_dim=int(meta["action_dim"]),
            state_dim=int(meta["state_dim"]),
            image_size=int(meta["image_size"]),
        )
        policy.load_state_dict(blob["model_state"])
        policy.eval()
        policy.to(device)

        mock_repo = args.mock_dataset_repo or meta.get("repo_id")
        resolved_mock_repo = mock_repo
        if mock_repo:
            mock_img, mock_state = load_mock_frame(
                mock_repo,
                args.mock_frame_index,
                meta["image_key"],
                meta.get("state_key"),
                int(meta["image_size"]),
                video_backend=args.video_backend,
            )
            mock_img = mock_img.to(device)
            mock_state = mock_state.to(device)

    interval = 1.0 / max(1e-3, args.fps)
    deadline = time.monotonic() + args.duration_s
    steps = 0

    header = {
        "instruction": instruction,
        "duration_s": args.duration_s,
        "fps": args.fps,
        "dry_run": args.dry_run,
        "policy_path": str(args.policy_path) if args.policy_path else None,
        "device": str(device),
        "mock_repo": resolved_mock_repo or args.mock_dataset_repo,
    }
    log_path.write_text(json.dumps(header) + "\n", encoding="utf-8")

    while time.monotonic() < deadline:
        t0 = time.monotonic()
        rec = {"step": steps, "t_monotonic": t0}
        if args.dry_run or policy is None:
            rec["action"] = None
        elif mock_img is not None:
            with torch.no_grad():
                act = policy(mock_img, mock_state).squeeze(0).cpu().tolist()
            rec["action"] = act
        else:
            rec["action"] = None
            rec["note"] = "Provide --mock-dataset-repo or wire live robot observations"

        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

        steps += 1
        elapsed = time.monotonic() - t0
        sleep_s = max(0.0, interval - elapsed)
        time.sleep(sleep_s)

    print(f"logged {steps} steps to {log_path}")


if __name__ == "__main__":
    main()
