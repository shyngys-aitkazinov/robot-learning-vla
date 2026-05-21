#!/usr/bin/env python3
"""Build a print-ready A4 PDF of OOD PINS celebrity photos for Eval 3.

Picks the top-N celebrities from ``datasets/pins-face-recognition-top30-quality.json``
(the curated global-recognizability ranking), selects the highest-resolution
photo within each celebrity's top-10 quality-ranked PINS images, and lays them
out two per A4 page (each photo ~A5, sized to be cut out) in the visual style of
the course-provided ``in-distribution-eval-3.pdf``.

PINS images are low resolution (<= ~564x802 px); printed at A5 size they will
look soft. That is an accepted tradeoff -- PINS is the requested photo source.

Outputs:
  * out-distribution-eval-3-pins.pdf            -- the printable file
  * datasets/out-distribution-eval-3-pins.json  -- manifest (mirrors the
                                                   out-distribution-eval-3.json schema)
  * datasets/out-distribution-eval-3-pins/<slug>/<slug>_01.jpg -- chosen sources

Usage:
  python tools/eval3_build_ood_pins_pdf.py
  python tools/eval3_build_ood_pins_pdf.py --n 10 --out out-distribution-eval-3-pins.pdf
"""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

DPI = 300
PAGE_W = round(8.2677 * DPI)   # A4 width  -> 2480 px
PAGE_H = round(11.6929 * DPI)  # A4 height -> 3508 px
CELL_H = PAGE_H // 2           # two photos stacked per page (each ~A5)

SIDE_MARGIN = 135              # white border L/R of every photo cell
V_MARGIN = 95                  # white border top/bottom of a normal photo cell
HEADER_INSET = 470             # extra top inset on page 1, cell 0, for the title

FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/Library/Fonts/Arial.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
]
FONT_BOLD_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/Library/Fonts/Arial Bold.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
]


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    for path in (FONT_BOLD_CANDIDATES if bold else FONT_CANDIDATES):
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return ImageFont.load_default()


def resolve_pins_path(json_path: str) -> Path:
    """JSON paths are rooted at datasets/pins-face-recognition/...; the dataset
    actually lives directly under datasets/."""
    return Path(json_path.replace("datasets/pins-face-recognition/", "datasets/"))


def pick_best_photo(celeb: dict, top_k: int = 10) -> Path:
    """Largest-pixel-area image among the celebrity's top-K quality photos."""
    best: tuple[int, Path] | None = None
    for jp in celeb["quality_photos"][:top_k]:
        p = resolve_pins_path(jp)
        if not p.exists():
            continue
        with Image.open(p) as im:
            area = im.size[0] * im.size[1]
        if best is None or area > best[0]:
            best = (area, p)
    if best is None:
        raise SystemExit(f"no readable PINS photo found for {celeb['name']}")
    return best[1]


def draw_centered(draw: ImageDraw.ImageDraw, y: int, text: str,
                   font: ImageFont.FreeTypeFont, fill=(20, 20, 20)) -> int:
    box = draw.textbbox((0, 0), text, font=font)
    w, h = box[2] - box[0], box[3] - box[1]
    draw.text(((PAGE_W - w) // 2, y), text, font=font, fill=fill)
    return y + h


def paste_into_cell(page: Image.Image, photo: Image.Image,
                    cell_top: int, top_inset: int) -> None:
    box_w = PAGE_W - 2 * SIDE_MARGIN
    box_h = CELL_H - top_inset - V_MARGIN
    scale = min(box_w / photo.width, box_h / photo.height)
    new_w, new_h = round(photo.width * scale), round(photo.height * scale)
    resized = photo.resize((new_w, new_h), Image.LANCZOS)
    x = (PAGE_W - new_w) // 2
    y = cell_top + top_inset + (box_h - new_h) // 2
    page.paste(resized, (x, y))


def build_pdf(celebs: list[dict], photos: list[Path], out_pdf: Path) -> None:
    title_font = load_font(58, bold=True)
    body_font = load_font(34)

    pages: list[Image.Image] = []
    for i in range(0, len(photos), 2):
        page = Image.new("RGB", (PAGE_W, PAGE_H), "white")
        draw = ImageDraw.Draw(page)

        # Header band on page 1 only.
        if i == 0:
            y = 120
            y = draw_centered(draw, y, "Out-of-Distribution Celebrity Images "
                              "— Eval 3", title_font) + 46
            y = draw_centered(draw, y, "Print this document once in color on A4.",
                              body_font, fill=(70, 70, 70)) + 14
            draw_centered(draw, y, "Cut out all 10 celebrity images and remove "
                          "the white borders.", body_font, fill=(70, 70, 70))

        for cell, photo_path in enumerate(photos[i:i + 2]):
            cell_top = cell * CELL_H
            top_inset = HEADER_INSET if (i == 0 and cell == 0) else V_MARGIN
            with Image.open(photo_path) as im:
                paste_into_cell(page, im.convert("RGB"), cell_top, top_inset)

        pages.append(page)

    pages[0].save(out_pdf, "PDF", save_all=True, append_images=pages[1:],
                  resolution=float(DPI))


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path,
                    default=Path("datasets/pins-face-recognition-top30-quality.json"))
    ap.add_argument("--n", type=int, default=10, help="number of celebrities")
    ap.add_argument("--out", type=Path, default=Path("out-distribution-eval-3-pins.pdf"))
    ap.add_argument("--img-root", type=Path,
                    default=Path("datasets/out-distribution-eval-3-pins"))
    ap.add_argument("--manifest", type=Path,
                    default=Path("datasets/out-distribution-eval-3-pins.json"))
    args = ap.parse_args()

    if not args.src.exists():
        raise SystemExit(f"--src {args.src} not found "
                          "(run tools/build_pins_metadata.py + build_pins_top30.py)")

    src = json.loads(args.src.read_text(encoding="utf-8"))
    celebs = src["celebrities"][:args.n]
    if len(celebs) < args.n:
        raise SystemExit(f"only {len(celebs)} celebrities in {args.src}, need {args.n}")

    photos: list[Path] = []
    manifest_celebs: list[dict] = []
    args.img_root.mkdir(parents=True, exist_ok=True)

    for c in celebs:
        chosen = pick_best_photo(c)
        with Image.open(chosen) as im:
            w, h = im.size
        slug = c["slug"]
        dst_dir = args.img_root / slug
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / f"{slug}_01.jpg"
        shutil.copy2(chosen, dst)
        photos.append(dst)
        manifest_celebs.append({
            "name": c["name"],
            "slug": slug,
            "rank": c.get("rank"),
            "category": c.get("category"),
            "held_out": c.get("held_out", False),
            "source_pins_image": str(chosen),
            "source_resolution_px": [w, h],
            "pdf_image": str(dst),
        })
        print(f"  {c.get('rank'):>2}. {c['name']:24s} {w}x{h}px  <- {chosen.name}")

    build_pdf(celebs, photos, args.out)

    manifest = {
        "dataset": "Eval 3 Out-of-Distribution (OOD) — PINS celebrity photos",
        "source": ("Top-N celebrities from the curated PINS global-recognizability "
                   "ranking (build_pins_top30.py). Photo per celebrity = the "
                   "largest-pixel-area image among that celebrity's top-10 "
                   "quality-ranked PINS images. Images used as-is: no re-crop, no "
                   "upscale. PINS images are low resolution (<= ~564x802 px) so "
                   "they print soft at A5 size — an accepted tradeoff."),
        "source_url": src.get("source_url"),
        "source_json": str(args.src),
        "root": str(args.img_root),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "pdf": str(args.out),
        "pdf_layout": "A4 portrait, 300 DPI, 2 photos per page (each ~A5), "
                      "styled after in-distribution-eval-3.pdf",
        "total_celebrities": len(manifest_celebs),
        "total_images": len(manifest_celebs),
        "held_out_identities_target": ["Barack Obama", "Taylor Swift", "Yann LeCun"],
        "selection_rationale": ("OOD identities for Eval 3 — none overlap the "
                                "three TOY identities. Picked for global "
                                "recognizability so the SmolVLM backbone is most "
                                "likely to recognise them zero-shot."),
        "celebrities": manifest_celebs,
    }
    args.manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
                             encoding="utf-8")

    size_mb = args.out.stat().st_size / 1024 / 1024
    print(f"\nWrote {args.out}  ({size_mb:.1f} MB, "
          f"{(len(photos) + 1) // 2} pages, {len(photos)} photos)")
    print(f"Wrote {args.manifest}")
    print(f"Copied {len(photos)} source images under {args.img_root}/")


if __name__ == "__main__":
    main()
