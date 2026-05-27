#!/usr/bin/env python3
"""Create the Eval 3 sample frames used by eval3_extract_masks.py.

This downloads/uses only the first front-camera video from each Hugging Face
dataset, decodes selected frames with OpenCV, and writes the PNGs expected by
``tools/eval3_extract_masks.py``:

  outputs/eval3_deep_analysis_v2/sample_frames/swift/ep00_approach_frame76.png
  outputs/eval3_deep_analysis_v2/sample_frames/lecun/ep00_approach_frame92.png
  outputs/eval3_deep_analysis_v2/sample_frames/obama/ep00_approach_frame76.png

Usage:
  python tools/eval3_extract_sample_frames.py
  python tools/eval3_extract_sample_frames.py --local-files-only
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image


VIDEO_KEY = "observation.images.front"

SAMPLES = {
    "swift": {
        "repo_id": "RobotLearningVLA/taylor_swift_1",
        "episode": 0,
        "frame": 76,
        "filename": "ep00_approach_frame76.png",
    },
    "lecun": {
        "repo_id": "RobotLearningVLA/yann_lecun_1",
        "episode": 0,
        "frame": 92,
        "filename": "ep00_approach_frame92.png",
    },
    "obama": {
        "repo_id": "RobotLearningVLA/barack_obama_1",
        "episode": 0,
        "frame": 76,
        "filename": "ep00_approach_frame76.png",
    },
}


def snapshot_root(repo_id: str, local_files_only: bool) -> Path:
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            allow_patterns=[
                "meta/info.json",
                f"videos/{VIDEO_KEY}/chunk-000/file-000.mp4",
            ],
            local_files_only=local_files_only,
        )
    )


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text()) if path.is_file() else {}


def first_video_path(root: Path) -> Path:
    info = read_json(root / "meta" / "info.json")
    template = info.get(
        "video_path",
        "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
    )
    rel = template.format(video_key=VIDEO_KEY, chunk_index=0, file_index=0)
    return root / rel


def decode_frame(video_path: Path, frame_idx: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame_bgr = cap.read()
    cap.release()
    if not ok or frame_bgr is None:
        raise RuntimeError(f"OpenCV could not read frame {frame_idx} from {video_path}")
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="outputs/eval3_deep_analysis_v2/sample_frames")
    ap.add_argument("--local-files-only", action="store_true", help="Use only cached Hugging Face files")
    args = ap.parse_args()

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    for slug, cfg in SAMPLES.items():
        root = snapshot_root(cfg["repo_id"], local_files_only=args.local_files_only)
        video_path = first_video_path(root)
        frame_idx = int(cfg["frame"])
        frame = decode_frame(video_path, frame_idx)

        out_dir = out_root / slug
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / cfg["filename"]
        Image.fromarray(frame).save(out_path)
        print(f"[{slug}] {cfg['repo_id']} frame {frame_idx} from {video_path} -> {out_path}")

    print("\nDone. Now run: python tools/eval3_extract_masks.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
