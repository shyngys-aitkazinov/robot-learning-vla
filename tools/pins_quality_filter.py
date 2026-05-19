#!/usr/bin/env python3
"""Pre-filter Pins celebrity photos by quality + rank them best-first.

For each celeb in an input pool JSON (Pins top-30 / top-50 / full),
runs a Haar face detector on every photo and computes a quality score:

  score =   (face_area / image_area)      * 100      # bigger close-up = better
          + (10 if single_face else 0)               # single face strongly preferred
          + (5  if portrait_orientation else 0)      # h > w
          + (5  if long_edge >= 600 else 0)          # decent resolution
          + (10 if face_centered (within 30% from center horizontally) else 0)

Photos must pass ALL hard filters to be kept:
  * exactly 1 face detected
  * face_area >= 8% of image area (the face is clearly the subject)
  * long_edge >= 400 px
  * portrait or near-square (h >= w * 0.85)

Output JSON: same shape as the input pool JSON, but each celeb gets a
``quality_photos`` field listing the kept photo PATHS sorted best-first
(by score descending). ``n_images`` is updated to the filtered count.

Usage::

    python tools/pins_quality_filter.py \\
        --pool-json datasets/pins-face-recognition-top30.json \\
        --out-json  datasets/pins-face-recognition-top30-quality.json

    # Smoke (1 celeb, prints scores):
    python tools/pins_quality_filter.py \\
        --pool-json datasets/pins-face-recognition-top30.json \\
        --out-json  /tmp/pins30-quality.json \\
        --debug-celeb emma_stone

Then point the synth generator at the new file::

    python tools/eval3_synth_pins_dataset_gen.py \\
        --pool-json datasets/pins-face-recognition-top30-quality.json ...
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Face detector (Haar)
# ---------------------------------------------------------------------------

_CASCADE_FRONTAL = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
_CASCADE_PROFILE = cv2.data.haarcascades + "haarcascade_profileface.xml"


def _make_detectors():
    """Build per-worker face detectors (cv2.CascadeClassifier is not picklable
    across spawn workers, so we lazy-create them inside the worker)."""
    return (
        cv2.CascadeClassifier(_CASCADE_FRONTAL),
        cv2.CascadeClassifier(_CASCADE_PROFILE),
    )


def _detect_faces(gray: np.ndarray, frontal, profile) -> list[tuple[int, int, int, int]]:
    """Return list of (x, y, w, h) face boxes; profile detector adds non-frontal poses."""
    boxes: list[tuple[int, int, int, int]] = []
    for det in (frontal, profile):
        rects = det.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(60, 60))
        for r in rects:
            boxes.append(tuple(int(v) for v in r))
    # Merge overlapping boxes (frontal + profile often double-detect)
    merged: list[tuple[int, int, int, int]] = []
    for b in boxes:
        x1, y1, w1, h1 = b
        keep = True
        for i, (x2, y2, w2, h2) in enumerate(merged):
            # IoU-ish overlap check
            ix1, iy1 = max(x1, x2), max(y1, y2)
            ix2, iy2 = min(x1+w1, x2+w2), min(y1+h1, y2+h2)
            if ix1 < ix2 and iy1 < iy2:
                inter = (ix2-ix1) * (iy2-iy1)
                a1, a2 = w1*h1, w2*h2
                if inter / min(a1, a2) > 0.5:
                    # keep the larger one
                    if a1 > a2:
                        merged[i] = b
                    keep = False
                    break
        if keep:
            merged.append(b)
    return merged


# ---------------------------------------------------------------------------
# Per-photo scoring
# ---------------------------------------------------------------------------

@dataclass
class PhotoScore:
    path: str
    score: float
    n_faces: int
    face_area_frac: float
    long_edge: int
    aspect_h_over_w: float
    keep: bool
    reason: str   # 'OK' or 'FAIL_<criterion>'


HARD_FACE_AREA_MIN = 0.08      # face must occupy >= 8% of image
HARD_LONG_EDGE_MIN = 180       # at least 180 px on long edge (Pins median ~155, this drops the worst tail)
HARD_ASPECT_H_OVER_W_MIN = 0.85  # portrait-or-near-square
HARD_DOMINANT_FACE_FRAC = 0.5  # if multiple faces, largest must own >= 50% of total face area
                                # (handles Haar's frequent double-detection on frontal+profile)


def _score_one_photo(args: tuple[str, str]) -> PhotoScore:
    """Worker function — must accept a single picklable arg."""
    path_str, _ = args
    p = Path(path_str)
    frontal, profile = _make_detectors()
    img = cv2.imread(str(p))
    if img is None:
        return PhotoScore(path=str(p), score=-1.0, n_faces=0, face_area_frac=0.0,
                          long_edge=0, aspect_h_over_w=0.0, keep=False, reason="FAIL_unreadable")
    h, w = img.shape[:2]
    long_edge = max(h, w)
    aspect_hw = h / w
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    boxes = _detect_faces(gray, frontal, profile)
    if len(boxes) == 0:
        return PhotoScore(path=str(p), score=-1.0, n_faces=0, face_area_frac=0.0,
                          long_edge=long_edge, aspect_h_over_w=aspect_hw,
                          keep=False, reason="FAIL_no_face")
    n_faces = len(boxes)
    # Largest face for area + centering checks
    largest = max(boxes, key=lambda b: b[2] * b[3])
    fx, fy, fw, fh = largest
    face_area = fw * fh
    img_area = h * w
    face_area_frac = face_area / img_area
    face_cx = fx + fw / 2.0
    img_cx = w / 2.0
    centered = abs(face_cx - img_cx) <= 0.30 * w
    # Dominance ratio: largest face vs sum of all face areas. Helps survive
    # Haar's frequent frontal+profile double-detection on the same face.
    total_face_area = sum(b[2] * b[3] for b in boxes)
    dominance = face_area / max(total_face_area, 1)

    # Hard filters
    if n_faces > 1 and dominance < HARD_DOMINANT_FACE_FRAC:
        return PhotoScore(path=str(p), score=-1.0, n_faces=n_faces, face_area_frac=face_area_frac,
                          long_edge=long_edge, aspect_h_over_w=aspect_hw,
                          keep=False, reason=f"FAIL_multi_face_{n_faces}")
    if face_area_frac < HARD_FACE_AREA_MIN:
        return PhotoScore(path=str(p), score=-1.0, n_faces=n_faces, face_area_frac=face_area_frac,
                          long_edge=long_edge, aspect_h_over_w=aspect_hw,
                          keep=False, reason="FAIL_face_too_small")
    if long_edge < HARD_LONG_EDGE_MIN:
        return PhotoScore(path=str(p), score=-1.0, n_faces=n_faces, face_area_frac=face_area_frac,
                          long_edge=long_edge, aspect_h_over_w=aspect_hw,
                          keep=False, reason="FAIL_low_res")
    if aspect_hw < HARD_ASPECT_H_OVER_W_MIN:
        return PhotoScore(path=str(p), score=-1.0, n_faces=n_faces, face_area_frac=face_area_frac,
                          long_edge=long_edge, aspect_h_over_w=aspect_hw,
                          keep=False, reason="FAIL_too_wide")

    # Soft score (continuous, higher = better)
    score = face_area_frac * 100.0
    score += 10.0  # single-face bonus (already hard-required, just bake into score)
    score += 5.0 if aspect_hw >= 1.0 else 0.0          # portrait > square
    score += 5.0 if long_edge >= 600 else 0.0
    score += 10.0 if centered else 0.0

    return PhotoScore(path=str(p), score=round(score, 2), n_faces=n_faces,
                      face_area_frac=round(face_area_frac, 4), long_edge=long_edge,
                      aspect_h_over_w=round(aspect_hw, 3), keep=True, reason="OK")


# ---------------------------------------------------------------------------
# Pool filtering
# ---------------------------------------------------------------------------

def filter_one_celeb(celeb_dict: dict, debug: bool = False,
                     n_workers: int | None = None) -> tuple[list[PhotoScore], list[PhotoScore]]:
    """Score all photos for one celeb, return (kept_sorted_best_first, dropped)."""
    jpgs = sorted(Path(celeb_dict["dir"]).glob("*.jpg"))
    args = [(str(p), celeb_dict["slug"]) for p in jpgs]
    if n_workers and n_workers > 1 and len(args) > 8:
        with mp.Pool(processes=n_workers) as pool:
            scores = pool.map(_score_one_photo, args)
    else:
        scores = [_score_one_photo(a) for a in args]
    kept = [s for s in scores if s.keep]
    dropped = [s for s in scores if not s.keep]
    kept.sort(key=lambda s: s.score, reverse=True)
    if debug:
        print(f"\n  {celeb_dict['slug']}: {len(kept)}/{len(scores)} kept "
              f"(dropped reasons: { _summarize_drop_reasons(dropped) })")
        print("    TOP 5 kept:")
        for s in kept[:5]:
            print(f"      score={s.score:6.2f}  face={s.face_area_frac:.2%}  "
                  f"long_edge={s.long_edge}  aspect={s.aspect_h_over_w:.2f}  "
                  f"{Path(s.path).name}")
        print("    BOTTOM 5 dropped:")
        for s in dropped[-5:]:
            print(f"      reason={s.reason:<25s}  {Path(s.path).name}")
    return kept, dropped


def _summarize_drop_reasons(dropped: list[PhotoScore]) -> dict[str, int]:
    out: dict[str, int] = {}
    for s in dropped:
        out[s.reason] = out.get(s.reason, 0) + 1
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--pool-json", type=Path, required=True,
                    help="Input pool JSON (Pins-style, with celebrities[].dir).")
    ap.add_argument("--out-json", type=Path, required=True,
                    help="Output filtered JSON path.")
    ap.add_argument("--n-workers", type=int, default=0,
                    help="Per-celeb scoring workers (0 = auto = os.cpu_count()).")
    ap.add_argument("--debug-celeb", default=None,
                    help="If set, only filter this one celeb and print detail.")
    args = ap.parse_args()

    if not args.pool_json.is_file():
        sys.exit(f"--pool-json not found: {args.pool_json}")
    meta = json.loads(args.pool_json.read_text(encoding="utf-8"))
    celebs = meta["celebrities"]
    print(f"Filtering {len(celebs)} celebs from {args.pool_json}")

    import os
    n_workers = args.n_workers if args.n_workers > 0 else (os.cpu_count() or 1)
    print(f"Using {n_workers} workers per celeb")

    t0 = time.time()
    new_celebs = []
    total_kept = total_dropped = 0
    for ci, c in enumerate(celebs):
        if args.debug_celeb and c["slug"] != args.debug_celeb:
            continue
        kept, dropped = filter_one_celeb(c, debug=bool(args.debug_celeb), n_workers=n_workers)
        total_kept += len(kept)
        total_dropped += len(dropped)
        # Write a new celeb entry that:
        # - keeps all original fields
        # - replaces n_images with the kept count
        # - adds quality_photos (sorted best-first paths)
        # - adds quality_meta with the score thresholds used
        new_c = dict(c)
        new_c["n_images"] = len(kept)
        new_c["n_images_pre_filter"] = c.get("n_images", len(kept) + len(dropped))
        new_c["quality_photos"] = [s.path for s in kept]
        new_c["quality_drop_reasons"] = _summarize_drop_reasons(dropped)
        new_c["quality_top10_scores"] = [s.score for s in kept[:10]]
        new_celebs.append(new_c)
        elapsed = time.time() - t0
        rate = (ci + 1) / max(elapsed, 1e-6)
        eta = (len(celebs) - ci - 1) / max(rate, 1e-6)
        print(f"  [{ci+1:>3d}/{len(celebs)}] {c['slug']:<25s} "
              f"{len(kept):>3d}/{len(kept)+len(dropped):>3d} kept "
              f"({len(kept)/(len(kept)+len(dropped))*100:>5.1f}%)  "
              f"elapsed {elapsed/60:.1f} min  eta {eta/60:.1f} min", flush=True)

    if args.debug_celeb and not new_celebs:
        sys.exit(f"--debug-celeb={args.debug_celeb!r} not found in pool")

    # Update top-level totals
    out_meta = dict(meta)
    out_meta["dataset"] = (meta.get("dataset") or "") + " — quality-filtered"
    out_meta["source"] = ((meta.get("source") or "") +
                          f"  | quality-filtered with tools/pins_quality_filter.py "
                          f"(hard filters: dominant face >= {HARD_DOMINANT_FACE_FRAC:.0%}, "
                          f"face_area>={HARD_FACE_AREA_MIN:.0%}, "
                          f"long_edge>={HARD_LONG_EDGE_MIN}px, aspect_hw>={HARD_ASPECT_H_OVER_W_MIN})")
    out_meta["total_images"] = total_kept
    out_meta["total_images_pre_filter"] = sum(c.get("n_images_pre_filter", 0) for c in new_celebs)
    out_meta["quality_filter"] = {
        "n_faces_required": 1,
        "min_face_area_frac": HARD_FACE_AREA_MIN,
        "min_long_edge_px": HARD_LONG_EDGE_MIN,
        "min_aspect_h_over_w": HARD_ASPECT_H_OVER_W_MIN,
    }
    out_meta["celebrities"] = new_celebs
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(out_meta, indent=2, ensure_ascii=False) + "\n",
                              encoding="utf-8")
    elapsed = time.time() - t0
    print()
    print(f"Wrote {args.out_json}")
    print(f"Total: {total_kept} kept / {total_kept + total_dropped} = "
          f"{total_kept / max(total_kept + total_dropped, 1) * 100:.1f}%")
    print(f"Elapsed: {elapsed / 60:.1f} min")


if __name__ == "__main__":
    main()
