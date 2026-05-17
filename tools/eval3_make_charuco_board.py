#!/usr/bin/env python3
"""Generate a printable ChArUco board with a central chroma-key target.

By default the *board* is A5 sized (matching the eval celebrity prints) but the
*PDF page* is A4 — the A5 content is centred on the A4 page with crop marks at
the four corners so you can print on standard office A4 paper and cut to size.
Pass ``--print-on A5`` to skip the crop marks and emit a single-page A5 PDF.

Intended workflow for synthetic-on-real Eval 3 data:
  1. Print the PDF at 100% scale (disable 'Fit to page') on matte A4 paper.
     Verify with a ruler against the top-edge tick marks (10 mm apart).
  2. Cut along the four crop marks to obtain an A5 sheet matching the eval
     print footprint.
  3. Place the coke can on the chroma square; record teleop episodes with
     `lerobot-record` as usual.
  4. Post-process each frame: detect ChArUco corners, solve the plane
     homography, and warp arbitrary celebrity images over the entire A5 board
     region. Mask the can with HSV chroma keying (cheap) or a segmentation
     model (robust to the red Coke label that breaks pure HSV).

Dependencies (one-time):
  uv pip install opencv-contrib-python pillow

Quick start:
  python tools/eval3_make_charuco_board.py --chroma-mm 80          # default A5-on-A4
  python tools/eval3_make_charuco_board.py --print-on A5           # native A5 page, no crop marks
  python tools/eval3_make_charuco_board.py --no-chroma             # plain calibration board
  python tools/eval3_make_charuco_board.py --content-mm 130x180    # smaller than A5

Multi-board recording (3 prints in semicircle): the simplest robust recipe is
three IDENTICAL boards (same dict, same size). The post-processor detects all
markers, clusters them spatially into 3 groups (left/centre/right by image x),
and computes one plane homography per cluster. ID collisions are harmless
because spatial position disambiguates. Generate three copies:

  for pos in left centre right; do
    python tools/eval3_make_charuco_board.py --content-mm 130x180 \\
        --squares-x 5 --squares-y 7 --chroma-mm 60 \\
        --out outputs/eval3_charuco/board_${pos}
  done

Stick to DICT_4X4_50 (default) — 5x5 / 6x6 markers shrink badly at typical
camera distances. Keep --chroma-mm well under the board's shorter dimension so
enough markers survive around the perimeter for the homography solve.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

try:
    import cv2
    if not hasattr(cv2, "aruco"):
        raise ImportError("cv2.aruco missing")
    import cv2.aruco as aruco
except ImportError as e:
    sys.exit(
        f"opencv-contrib-python with aruco is required ({e}).\n"
        "  uv pip install 'opencv-contrib-python>=4.7'"
    )

from PIL import Image


ARUCO_DICTS = {
    "4X4_50":   aruco.DICT_4X4_50,
    "4X4_100":  aruco.DICT_4X4_100,
    "4X4_250":  aruco.DICT_4X4_250,   # use this when generating 3+ boards from one dict
    "5X5_50":   aruco.DICT_5X5_50,
    "5X5_100":  aruco.DICT_5X5_100,
    "5X5_250":  aruco.DICT_5X5_250,
    "6X6_50":   aruco.DICT_6X6_50,
    "6X6_250":  aruco.DICT_6X6_250,
}

PAPER_SIZES_MM = {
    "A6":     (105.0, 148.0),
    "A5":     (148.0, 210.0),
    "A4":     (210.0, 297.0),
    "A3":     (297.0, 420.0),
    "Letter": (215.9, 279.4),
}

# Approximate broadcast chroma keys (sRGB). Lime is the easiest to threshold
# but tends to print off-hue on inkjets; standard chroma green/blue are friendlier
# to office printers and downstream HSV thresholds.
CHROMA_PRESETS_RGB = {
    "green":   (0, 177, 64),
    "blue":    (0, 71, 187),
    "lime":    (0, 255, 0),
    "magenta": (255, 0, 255),
}


def mm_to_px(mm: float, dpi: float) -> int:
    return int(round(mm * dpi / 25.4))


def draw_crop_marks(
    page: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    length_px: int, gap_px: int,
    thickness: int = 1,
    color: tuple[int, int, int] = (0, 0, 0),
) -> None:
    """L-shaped printer crop marks just outside a rectangle's four corners."""
    g, L = gap_px, length_px
    # Top-left
    cv2.line(page, (x1 - g - L, y1), (x1 - g, y1), color, thickness)
    cv2.line(page, (x1, y1 - g - L), (x1, y1 - g), color, thickness)
    # Top-right
    cv2.line(page, (x2 + g, y1), (x2 + g + L, y1), color, thickness)
    cv2.line(page, (x2, y1 - g - L), (x2, y1 - g), color, thickness)
    # Bottom-left
    cv2.line(page, (x1 - g - L, y2), (x1 - g, y2), color, thickness)
    cv2.line(page, (x1, y2 + g), (x1, y2 + g + L), color, thickness)
    # Bottom-right
    cv2.line(page, (x2 + g, y2), (x2 + g + L, y2), color, thickness)
    cv2.line(page, (x2, y2 + g), (x2, y2 + g + L), color, thickness)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--paper", choices=list(PAPER_SIZES_MM), default="A5",
                    help="Size of the ChArUco board content (default: A5).")
    ap.add_argument("--content-mm", type=str, default=None,
                    help="Override --paper with a custom content size '<W>x<H>' in mm "
                         "(e.g. '130x180'). Useful when matching non-standard print sizes.")
    ap.add_argument("--print-on", dest="print_on", choices=list(PAPER_SIZES_MM), default="A4",
                    help="PDF page size; the board content is centred inside with crop marks "
                         "if larger than --paper (default: A4).")
    ap.add_argument("--landscape", action="store_true",
                    help="Rotate both content and page to landscape orientation.")
    ap.add_argument("--dpi", type=float, default=300.0,
                    help="Print resolution in dots per inch (default: 300).")
    ap.add_argument("--margin-mm", type=float, default=8.0,
                    help="White margin inside the content rectangle (covers printer unprintable).")
    ap.add_argument("--squares-x", type=int, default=7,
                    help="ChArUco grid: columns of chessboard squares.")
    ap.add_argument("--squares-y", type=int, default=10,
                    help="ChArUco grid: rows of chessboard squares.")
    ap.add_argument("--square-mm", type=float, default=None,
                    help="Chessboard square side length (mm). Default: maximise within the content.")
    ap.add_argument("--marker-ratio", type=float, default=0.7,
                    help="ArUco marker side / chessboard square side. Must be in (0, 1).")
    ap.add_argument("--dict", dest="dict_name", choices=list(ARUCO_DICTS), default="4X4_50",
                    help="Dictionary. Small dicts (DICT_4X4_50) maximise inter-marker Hamming distance.")
    ap.add_argument("--chroma", choices=list(CHROMA_PRESETS_RGB), default="green",
                    help="Colour of the central chroma-key square.")
    ap.add_argument("--chroma-mm", type=float, default=None,
                    help="Side length of the central chroma square (mm). "
                         "Default: 55%% of the shorter board dimension.")
    ap.add_argument("--no-chroma", action="store_true",
                    help="Skip the chroma overlay — produces a vanilla ChArUco calibration board.")
    ap.add_argument("--crop-mark-mm", type=float, default=5.0,
                    help="Length of each crop-mark stroke (default: 5 mm).")
    ap.add_argument("--out", type=Path,
                    default=Path("outputs/eval3_charuco/charuco_a5"),
                    help="Output basename (no extension). Writes <out>.png and <out>.pdf.")
    args = ap.parse_args()

    if not 0.0 < args.marker_ratio < 1.0:
        sys.exit(f"--marker-ratio must be in (0, 1), got {args.marker_ratio}")

    # Content rectangle (the cuttable region).
    if args.content_mm is not None:
        try:
            w_str, h_str = args.content_mm.lower().replace("mm", "").split("x")
            content_w_mm, content_h_mm = float(w_str), float(h_str)
        except (ValueError, AttributeError):
            sys.exit(f"--content-mm must be '<W>x<H>' in mm, got {args.content_mm!r}")
        content_paper_label = f"{content_w_mm:.0f}x{content_h_mm:.0f}mm"
    else:
        content_w_mm, content_h_mm = PAPER_SIZES_MM[args.paper]
        content_paper_label = args.paper
    # PDF page (what the printer sees).
    page_w_mm, page_h_mm = PAPER_SIZES_MM[args.print_on]
    if args.landscape:
        content_w_mm, content_h_mm = content_h_mm, content_w_mm
        page_w_mm, page_h_mm = page_h_mm, page_w_mm

    if page_w_mm + 1e-6 < content_w_mm or page_h_mm + 1e-6 < content_h_mm:
        sys.exit(
            f"--print-on {args.print_on} ({page_w_mm:.0f}x{page_h_mm:.0f}mm) is smaller "
            f"than --paper {args.paper} ({content_w_mm:.0f}x{content_h_mm:.0f}mm); cannot fit."
        )

    inner_w_mm = content_w_mm - 2 * args.margin_mm
    inner_h_mm = content_h_mm - 2 * args.margin_mm
    if inner_w_mm <= 0 or inner_h_mm <= 0:
        sys.exit(f"margin {args.margin_mm}mm leaves no room on {args.paper}")

    if args.square_mm is None:
        square_mm = min(inner_w_mm / args.squares_x, inner_h_mm / args.squares_y)
    else:
        square_mm = args.square_mm
    marker_mm = square_mm * args.marker_ratio
    board_w_mm = args.squares_x * square_mm
    board_h_mm = args.squares_y * square_mm
    if board_w_mm > inner_w_mm + 1e-6 or board_h_mm > inner_h_mm + 1e-6:
        sys.exit(
            f"board {board_w_mm:.1f}x{board_h_mm:.1f}mm does not fit inside printable "
            f"{inner_w_mm:.1f}x{inner_h_mm:.1f}mm — reduce --squares-* or --square-mm."
        )

    page_w_px = mm_to_px(page_w_mm, args.dpi)
    page_h_px = mm_to_px(page_h_mm, args.dpi)
    content_w_px = mm_to_px(content_w_mm, args.dpi)
    content_h_px = mm_to_px(content_h_mm, args.dpi)
    # Derive board pixels from one integer square_px so generateImage's internal
    # chessboard-cell rasteriser doesn't trip on fractional per-square sizes.
    square_px = mm_to_px(square_mm, args.dpi)
    if square_px <= 0:
        sys.exit(f"square {square_mm}mm rounds to 0 px at {args.dpi} DPI")
    board_w_px = args.squares_x * square_px
    board_h_px = args.squares_y * square_px

    # Render the ChArUco board. Side lengths are passed in metres for downstream
    # `aruco.estimatePoseCharucoBoard` compatibility.
    dictionary = aruco.getPredefinedDictionary(ARUCO_DICTS[args.dict_name])
    board = aruco.CharucoBoard(
        (args.squares_x, args.squares_y),
        squareLength=square_mm / 1000.0,
        markerLength=marker_mm / 1000.0,
        dictionary=dictionary,
    )
    n_markers = len(board.getIds())
    board_gray = board.generateImage((board_w_px, board_h_px), marginSize=0, borderBits=1)
    board_bgr = cv2.cvtColor(board_gray, cv2.COLOR_GRAY2BGR)

    chroma_mm: float | None = None
    if not args.no_chroma:
        chroma_mm = args.chroma_mm if args.chroma_mm is not None else 0.55 * min(board_w_mm, board_h_mm)
        if chroma_mm <= 0 or chroma_mm > min(board_w_mm, board_h_mm):
            sys.exit(f"--chroma-mm {chroma_mm} out of range for board {board_w_mm}x{board_h_mm}mm")
        chroma_px = mm_to_px(chroma_mm, args.dpi)
        cx, cy = board_bgr.shape[1] // 2, board_bgr.shape[0] // 2
        x1, y1 = cx - chroma_px // 2, cy - chroma_px // 2
        x2, y2 = x1 + chroma_px, y1 + chroma_px
        r, g, b = CHROMA_PRESETS_RGB[args.chroma]
        cv2.rectangle(board_bgr, (x1, y1), (x2, y2), (b, g, r), thickness=cv2.FILLED)

    # Build the page canvas; centre the content inside it.
    page = np.full((page_h_px, page_w_px, 3), 255, dtype=np.uint8)
    content_x0 = (page_w_px - content_w_px) // 2
    content_y0 = (page_h_px - content_h_px) // 2

    # Centre the board within the content rectangle.
    board_x0 = content_x0 + (content_w_px - board_w_px) // 2
    board_y0 = content_y0 + (content_h_px - board_h_px) // 2
    page[board_y0:board_y0 + board_h_px, board_x0:board_x0 + board_w_px] = board_bgr

    # Print-scale ruler in the top margin of the *content* (survives the cut).
    ruler_y = content_y0 + max(mm_to_px(args.margin_mm * 0.4, args.dpi), 6)
    tick_long = ruler_y + mm_to_px(3.0, args.dpi)
    tick_short = ruler_y + mm_to_px(1.5, args.dpi)
    for i in range(0, int(board_w_mm) // 10 + 1):
        x = board_x0 + mm_to_px(10 * i, args.dpi)
        if x >= content_x0 + content_w_px - 4:
            break
        cv2.line(page, (x, ruler_y), (x, tick_long if i % 5 == 0 else tick_short),
                 (0, 0, 0), 1)

    # Label across the bottom margin of the content (also survives the cut).
    label = (f"{content_paper_label}{' L' if args.landscape else ''}  "
             f"square={square_mm:.2f}mm  marker={marker_mm:.2f}mm  "
             f"dict={args.dict_name}  n_markers={n_markers}  dpi={int(args.dpi)}")
    if not args.no_chroma:
        label += f"  chroma={chroma_mm:.1f}mm/{args.chroma}"
    label_y = content_y0 + content_h_px - max(mm_to_px(args.margin_mm * 0.4, args.dpi), 12)
    cv2.putText(page, label, (board_x0, label_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)

    # Crop marks at the four content corners (only if content is smaller than the page).
    draw_crop = (content_w_px < page_w_px) or (content_h_px < page_h_px)
    if draw_crop:
        draw_crop_marks(
            page,
            content_x0, content_y0,
            content_x0 + content_w_px - 1, content_y0 + content_h_px - 1,
            length_px=mm_to_px(args.crop_mark_mm, args.dpi),
            gap_px=mm_to_px(2.0, args.dpi),
            thickness=1,
        )
        cut_label = f"Cut to {content_paper_label}{' (landscape)' if args.landscape else ''}"
        cv2.putText(page, cut_label,
                    (content_x0, content_y0 - mm_to_px(2.0, args.dpi)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    png_path = args.out.with_suffix(".png")
    pdf_path = args.out.with_suffix(".pdf")
    cv2.imwrite(str(png_path), page)
    Image.fromarray(cv2.cvtColor(page, cv2.COLOR_BGR2RGB)).save(
        str(pdf_path), format="PDF", resolution=args.dpi
    )

    print(f"Wrote {png_path}  ({page_w_px}x{page_h_px} px @ {int(args.dpi)} DPI)")
    print(f"Wrote {pdf_path}  (page {page_w_mm:.1f}x{page_h_mm:.1f} mm)")
    print()
    print(f"  PDF page      : {args.print_on}{' landscape' if args.landscape else ''}")
    print(f"  Content (cut) : {content_paper_label}{' landscape' if args.landscape else ''}  "
          f"({content_w_mm:.1f} x {content_h_mm:.1f} mm)")
    print(f"  Board grid    : {args.squares_x} x {args.squares_y} squares  ({n_markers} markers)")
    print(f"  Square length : {square_mm:.2f} mm  (marker {marker_mm:.2f} mm)")
    print(f"  Dictionary    : {args.dict_name}")
    if not args.no_chroma:
        print(f"  Chroma centre : {chroma_mm:.1f} x {chroma_mm:.1f} mm  ({args.chroma})")
    if draw_crop:
        print(f"  Crop marks    : {args.crop_mark_mm:.1f} mm at the four content corners")
    print()
    print("PRINT AT 100% SCALE. In the printer dialog disable 'Fit to page' / 'Scale to fit'.")
    print("After printing, verify the top-edge tick marks are exactly 10 mm apart with a ruler.")
    if draw_crop:
        print(f"Then cut along the crop marks to obtain a clean {args.paper} sheet.")


if __name__ == "__main__":
    main()
