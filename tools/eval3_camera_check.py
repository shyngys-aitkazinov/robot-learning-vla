#!/usr/bin/env python3
"""Visual camera-match diagnostic: save 3 dataset frames and 3 live frames side-by-side.

Use this BEFORE running scripts/eval3_vla_deploy.py to verify your bench camera sees
the same view the training dataset was recorded with. Adjust the camera physically
until the live shots match the dataset shots in framing, distance, and angle.

Usage:
  python tools/eval3_camera_check.py                       # default: index 1, taylor_swift_1
  python tools/eval3_camera_check.py --camera-index 0      # try the other camera
  python tools/eval3_camera_check.py --episode 5           # use frames from episode 5
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

_HERE = Path(__file__).resolve().parent.parent / "scripts"
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from eval3_lerobot_shim import apply as _shim  # noqa: E402

_shim()


def save(img_hwc_uint8: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img_hwc_uint8).save(path)
    print(f"  wrote {path}  shape={img_hwc_uint8.shape}")


def grab_dataset_frames(repo_id: str, episode: int, n: int, out_dir: Path) -> list[Path]:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(repo_id, video_backend="pyav", episodes=[episode])
    print(f"Dataset {repo_id} episode={episode}: {len(ds)} frames")
    rng = np.random.default_rng(seed=0)
    indices = sorted(rng.choice(len(ds), size=n, replace=False).tolist())
    paths = []
    for i, idx in enumerate(indices):
        row = ds[int(idx)]
        # observation.images.front is CHW float32 in [0, 1]
        img_chw = row["observation.images.front"].numpy()
        img_hwc = (img_chw.transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
        out = out_dir / f"dataset_ep{episode}_frame{idx:05d}.png"
        save(img_hwc, out)
        paths.append(out)
    return paths


def grab_live_frames(camera_index: int, width: int, height: int, fps: int, n: int, out_dir: Path) -> list[Path]:
    from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

    cfg = OpenCVCameraConfig(index_or_path=camera_index, width=width, height=height, fps=fps)
    cam = OpenCVCamera(cfg)
    cam.connect()
    print(f"Camera index {camera_index} connected at {width}x{height}@{fps}fps")
    # Warm-up a few reads — first frames can be black on some macOS cameras
    for _ in range(5):
        try:
            _ = cam.async_read(timeout_ms=2000)
        except Exception:
            pass
        time.sleep(0.05)
    paths = []
    for i in range(n):
        img = cam.async_read(timeout_ms=2000)
        out = out_dir / f"live_cam{camera_index}_{i:02d}.png"
        save(img, out)
        paths.append(out)
        time.sleep(0.3)
    cam.disconnect()
    return paths


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", default="RobotLearningVLA/taylor_swift_1")
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--camera-index", type=int, default=1)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--n", type=int, default=3, help="frames to grab from each source")
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/eval3_camera_check"))
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print("== dataset frames ==")
    ds_paths = grab_dataset_frames(args.repo_id, args.episode, args.n, args.out_dir)
    print("\n== live camera frames ==")
    live_paths = grab_live_frames(args.camera_index, args.width, args.height, args.fps, args.n, args.out_dir)

    print(f"\nWrote {len(ds_paths)} dataset frames and {len(live_paths)} live frames to {args.out_dir.resolve()}")
    print("Open the folder and visually compare. Adjust the camera physically until the live")
    print("frames match the dataset frames in framing, distance, angle, and lighting.")
    print(f"\nQuick viewer:  open {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
