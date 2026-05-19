#!/usr/bin/env python3
"""Per-dataset static masks for background-replacement + print-shuffle augmentation.

Why per-dataset masks (not per-frame): the camera is FIXED within each
training dataset (intra-dataset table variance < 5%), so a single mask bundle
per dataset suffices. See outputs/eval3_deep_analysis_v2/sample_frames/.

Why a "target + 2 others" split (not "swift / lecun / obama"): identifying
which celebrity is in each print position across 3 datasets is fragile from
low-res frames. What MATTERS for print-shuffle is preserving the print the
operator was placing on (so action labels stay valid) and randomly permuting
the other two. We identify the target position per dataset by looking at the
final frame of ep0 (or another late episode if ep0 isn't placed):
  - swift dataset target  = MIDDLE  (can placed on middle print)
  - lecun dataset target  = RIGHT   (can placed on right print, verified ep5)
  - obama dataset target  = RIGHT   (can placed on right print, verified ep0)

What we produce per dataset under outputs/eval3_masks/{slug}/:
  bg_mask.npy      bool (H, W); True = background -> replaceable
  target_mask.npy  bool (H, W); True = target print region -> KEEP
  other1_mask.npy  bool (H, W); True = non-target print position 1 -> shuffleable
  other2_mask.npy  bool (H, W); True = non-target print position 2 -> shuffleable
  meta.json        layout details + target_position string
  preview.png      visual sanity overlay
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


# Each dataset has 3 print rectangles in LEFT/MIDDLE/RIGHT visual positions
# in the image plane. `target` names which rectangle the operator was placing on.
PRESETS = {
    "swift": {
        "sample_frame": "outputs/eval3_deep_analysis_v2/sample_frames/swift/ep00_approach_frame76.png",
        "table_polygon": [(0, 105), (640, 105), (640, 480), (0, 480)],
        "prints": {
            "left":   [(135, 95),  (295, 95),  (295, 200), (135, 200)],
            "middle": [(270, 125), (420, 125), (420, 245), (270, 245)],
            "right":  [(410, 95),  (545, 95),  (545, 185), (410, 185)],
        },
        "target": "middle",
    },
    "lecun": {
        "sample_frame": "outputs/eval3_deep_analysis_v2/sample_frames/lecun/ep00_approach_frame92.png",
        "table_polygon": [(0, 130), (640, 130), (640, 480), (0, 480)],
        "prints": {
            "left":   [(35,  130), (220, 130), (220, 250), (35,  250)],
            "middle": [(225, 175), (395, 175), (395, 355), (225, 355)],
            "right":  [(405, 115), (590, 115), (590, 290), (405, 290)],
        },
        "target": "right",
    },
    "obama": {
        "sample_frame": "outputs/eval3_deep_analysis_v2/sample_frames/obama/ep00_approach_frame76.png",
        "table_polygon": [(0, 130), (640, 130), (640, 480), (0, 480)],
        "prints": {
            "left":   [(20,  205), (280, 205), (280, 380), (20,  380)],
            "middle": [(220, 220), (450, 220), (450, 430), (220, 430)],
            "right":  [(395, 175), (615, 175), (615, 360), (395, 360)],
        },
        "target": "right",
    },
}


def polygon_to_mask(poly: list[tuple[int, int]], shape_hw: tuple[int, int]) -> np.ndarray:
    h, w = shape_hw
    img = Image.new("L", (w, h), 0)
    ImageDraw.Draw(img).polygon(poly, fill=255)
    return np.array(img) > 0


def render_preview(frame: np.ndarray, masks: dict[str, np.ndarray], out_path: Path, header: str) -> None:
    """Tint each mask region a distinct color and save a preview PNG."""
    overlay = frame.copy().astype(np.float32)
    tints = {
        "background": (255, 0, 255),   # magenta = will be replaced
        "table":      (255, 255, 0),   # yellow  = kept (non-print table)
        "target":     (0, 191, 255),   # cyan    = target print (KEEP)
        "other1":     (255, 64, 64),   # red     = shuffleable
        "other2":     (64, 255, 64),   # green   = shuffleable
    }
    for name, mask in masks.items():
        if mask is None:
            continue
        tint = np.array(tints[name], dtype=np.float32)
        overlay[mask] = overlay[mask] * 0.6 + tint * 0.4
    out = Image.fromarray(overlay.clip(0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(out)
    legend = "magenta=bg(replace)  yellow=table  cyan=TARGET  red/green=shuffleable"
    draw.text((6, 6), header, fill=(0, 0, 0))
    draw.text((6, 22), legend, fill=(0, 0, 0))
    out.save(out_path)


def export_masks_for_target(
    *,
    slug: str,
    cfg: dict,
    out_root: Path,
    target_pos: str,
    header_suffix: str = "",
) -> None:
    """Write bg/target/other masks for one target print position."""
    frame = np.array(Image.open(cfg["sample_frame"]).convert("RGB"))
    h, w = frame.shape[:2]
    prints_masks = {pos: polygon_to_mask(poly, (h, w)) for pos, poly in cfg["prints"].items()}
    table_full = polygon_to_mask(cfg["table_polygon"], (h, w))
    any_print = np.zeros_like(table_full)
    for m in prints_masks.values():
        any_print |= m
    table_only = table_full & ~any_print
    bg_mask = ~table_full

    target_mask = prints_masks[target_pos]
    other_positions = [p for p in ("left", "middle", "right") if p != target_pos]
    other1_mask = prints_masks[other_positions[0]]
    other2_mask = prints_masks[other_positions[1]]

    ds_dir = out_root / slug
    ds_dir.mkdir(parents=True, exist_ok=True)
    np.save(ds_dir / "bg_mask.npy", bg_mask)
    np.save(ds_dir / "target_mask.npy", target_mask)
    np.save(ds_dir / "other1_mask.npy", other1_mask)
    np.save(ds_dir / "other2_mask.npy", other2_mask)

    render_preview(
        frame,
        {
            "background": bg_mask,
            "table": table_only,
            "target": target_mask,
            "other1": other1_mask,
            "other2": other2_mask,
        },
        ds_dir / "preview.png",
        header=f"{slug}  target={target_pos}{header_suffix}",
    )

    meta = {
        "slug": slug,
        "sample_frame": cfg["sample_frame"],
        "image_shape_hw": [h, w],
        "table_polygon": cfg["table_polygon"],
        "prints": cfg["prints"],
        "target_position": target_pos,
        "other_positions": other_positions,
        "bg_mask_pixels": int(bg_mask.sum()),
        "target_mask_pixels": int(target_mask.sum()),
        "other1_mask_pixels": int(other1_mask.sum()),
        "other2_mask_pixels": int(other2_mask.sum()),
    }
    with open(ds_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n[{slug}]  target={target_pos}  -> {ds_dir}")
    print(f"  other1 ({other_positions[0]:>6}): {other1_mask.sum():>7d} px")
    print(f"  other2 ({other_positions[1]:>6}): {other2_mask.sum():>7d} px")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="outputs/eval3_masks")
    ap.add_argument(
        "--slot-masks",
        action="store_true",
        help="Also emit slot_left/slot_middle/slot_right masks for dataset_v2_* "
        "(shared print geometry from the swift preset; target slot varies).",
    )
    args = ap.parse_args()
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    for slug, cfg in PRESETS.items():
        target_pos = cfg["target"]
        print(f"\n[{slug}]  sample {cfg['sample_frame']}  target={target_pos}")
        export_masks_for_target(slug=slug, cfg=cfg, out_root=out_root, target_pos=target_pos)

    if args.slot_masks:
        # dataset_v2_* uses the same camera rig; only the target board slot changes.
        v2_geometry = PRESETS["swift"]
        for slot in ("left", "middle", "right"):
            export_masks_for_target(
                slug=f"slot_{slot}",
                cfg=v2_geometry,
                out_root=out_root,
                target_pos=slot,
                header_suffix="  (v2 shared geometry)",
            )

    print("\nDone. Inspect each preview.png and adjust PRESETS in this file if a polygon is wrong.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
