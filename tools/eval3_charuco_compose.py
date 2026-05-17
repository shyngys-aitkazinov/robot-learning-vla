#!/usr/bin/env python3
"""Live ChArUco pipeline: warp a target image onto the board, keep the can on top.

Locks the plane homography on the FIRST frame where the board detects, then
keeps it frozen (the prints don't move within an episode, per the recording
plan). Press 'r' to re-acquire if you bump the camera.

Per frame after lock:
  1. warpPerspective(target_image, H_target, frame_size) -> warped target
  2. board_mask  = projected board outline (full 130x180mm by default)
  3. chroma_mask = projected chroma rectangle (60mm centred by default)
  4. can_mask    = chroma_mask AND NOT (green pixels)
  5. result = frame.copy(); result[board_mask] = warped[board_mask];
              result[can_mask] = frame[can_mask]

Defaults match `tools/eval3_make_charuco_board.py --content-mm 130x180
--squares-x 5 --squares-y 7 --chroma-mm 60`.

Controls (live mode):
  q   quit
  s   save the current display to outputs/eval3_charuco_compose/
  r   re-detect on the next good frame and re-lock the homography
  t   toggle display: COMPOSITE / RAW (raw camera with no warp, for debugging)
  m   toggle display of intermediate masks (board / chroma / can) as overlays

Usage:
  python tools/eval3_charuco_compose.py                              # synth target
  python tools/eval3_charuco_compose.py --target-image celeb.jpg
  python tools/eval3_charuco_compose.py --camera-index 1 --hsv-lo 35 50 40
  python tools/eval3_charuco_compose.py --snapshot --n 5             # batch mode
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

try:
    import cv2
    if not hasattr(cv2, "aruco"):
        raise ImportError("cv2.aruco missing")
    import cv2.aruco as aruco
except ImportError as e:
    sys.exit(f"opencv-contrib-python required: {e}")

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
from eval3_charuco_check import build_board, open_camera  # reuse


def load_target(path: str | None, w_px: int, h_px: int) -> np.ndarray:
    """Load target image or generate an orientation-test pattern at (w x h) px."""
    if path is not None:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            sys.exit(f"cannot read --target-image {path}")
        return cv2.resize(img, (w_px, h_px), interpolation=cv2.INTER_AREA)
    # Checkered fallback with coloured corner tabs so warp orientation is obvious.
    img = np.zeros((h_px, w_px, 3), dtype=np.uint8)
    tile = max(8, min(h_px, w_px) // 12)
    for y in range(0, h_px, tile):
        for x in range(0, w_px, tile):
            if ((x // tile) + (y // tile)) % 2 == 0:
                img[y:y + tile, x:x + tile] = (210, 210, 210)
            else:
                img[y:y + tile, x:x + tile] = (60, 60, 60)
    s = tile * 2
    cv2.rectangle(img, (0, 0), (s, s), (0, 0, 200), cv2.FILLED)             # TL red
    cv2.rectangle(img, (w_px - s, 0), (w_px, s), (0, 200, 0), cv2.FILLED)   # TR green
    cv2.rectangle(img, (0, h_px - s), (s, h_px), (200, 0, 0), cv2.FILLED)   # BL blue
    cv2.rectangle(img, (w_px - s, h_px - s), (w_px, h_px), (0, 200, 200), cv2.FILLED)  # BR yellow
    cv2.putText(img, "TARGET", (w_px // 8, h_px // 2),
                cv2.FONT_HERSHEY_SIMPLEX, max(0.6, h_px / 500.0),
                (40, 40, 40), 2, cv2.LINE_AA)
    return img


def lock_homography(
    frame_bgr: np.ndarray,
    detector: aruco.ArucoDetector,
    char_detector: aruco.CharucoDetector,
    board: aruco.CharucoBoard,
):
    """Detect markers + chessboard corners; return (H, n_markers, n_chess, rms)."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    marker_corners, marker_ids, _ = detector.detectMarkers(gray)
    nm = 0 if marker_ids is None else len(marker_ids)
    if marker_ids is None or nm < 4:
        return None, nm, 0, None
    try:
        chess_corners, chess_ids, _, _ = char_detector.detectBoard(
            gray, markerCorners=marker_corners, markerIds=marker_ids
        )
    except Exception:
        chess_corners, chess_ids = None, None
    nc = 0 if chess_ids is None else len(chess_ids)
    if chess_corners is None or chess_ids is None or nc < 4:
        return None, nm, nc, None
    obj_pts, img_pts = board.matchImagePoints(chess_corners, chess_ids)
    if obj_pts is None or len(obj_pts) < 4:
        return None, nm, nc, None
    src_mm = (obj_pts.reshape(-1, 3)[:, :2] * 1000.0).astype(np.float32)
    dst_px = img_pts.reshape(-1, 2).astype(np.float32)
    H, mask_h = cv2.findHomography(src_mm, dst_px, cv2.RANSAC, 2.0)
    if H is None:
        return None, nm, nc, None
    inl = mask_h.ravel().astype(bool)
    proj = cv2.perspectiveTransform(src_mm[inl].reshape(-1, 1, 2), H).reshape(-1, 2)
    rms = float(np.sqrt(np.mean(np.sum((proj - dst_px[inl]) ** 2, axis=1))))
    return H, nm, nc, rms


def compose(
    frame_bgr: np.ndarray,
    target_bgr: np.ndarray,
    H_board: np.ndarray,
    board_w_mm: float,
    board_h_mm: float,
    chroma_mm: float,
    hsv_lo: tuple[int, int, int],
    hsv_hi: tuple[int, int, int],
):
    h_img, w_img = frame_bgr.shape[:2]

    # 1) Project the four board corners into image coordinates via H.
    board_corners_mm = np.float32([
        [0, 0], [board_w_mm, 0],
        [board_w_mm, board_h_mm], [0, board_h_mm],
    ]).reshape(-1, 1, 2)
    board_corners_img = cv2.perspectiveTransform(board_corners_mm, H_board).reshape(-1, 2).astype(np.float32)

    # 2) Build a per-frame homography that maps the target image corners onto the projected board.
    th, tw = target_bgr.shape[:2]
    target_src = np.float32([[0, 0], [tw - 1, 0], [tw - 1, th - 1], [0, th - 1]])
    H_target = cv2.getPerspectiveTransform(target_src, board_corners_img)
    warped = cv2.warpPerspective(target_bgr, H_target, (w_img, h_img))

    # 3) Masks: full board area, chroma centre.
    board_mask = np.zeros((h_img, w_img), dtype=np.uint8)
    cv2.fillConvexPoly(board_mask, board_corners_img.astype(np.int32), 255)

    cx, cy = board_w_mm / 2.0, board_h_mm / 2.0
    ch = chroma_mm / 2.0
    chroma_corners_mm = np.float32([
        [cx - ch, cy - ch], [cx + ch, cy - ch],
        [cx + ch, cy + ch], [cx - ch, cy + ch],
    ]).reshape(-1, 1, 2)
    chroma_corners_img = cv2.perspectiveTransform(chroma_corners_mm, H_board).reshape(-1, 2).astype(np.int32)
    chroma_mask = np.zeros((h_img, w_img), dtype=np.uint8)
    cv2.fillConvexPoly(chroma_mask, chroma_corners_img, 255)

    # 4) Can mask: pixels INSIDE the chroma area that are NOT green.
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    green_mask = cv2.inRange(hsv, np.array(hsv_lo, dtype=np.uint8), np.array(hsv_hi, dtype=np.uint8))
    can_mask = cv2.bitwise_and(chroma_mask, cv2.bitwise_not(green_mask))
    # Close small holes; erode 1 px to trim chroma bleed off the can's outline.
    can_mask = cv2.morphologyEx(can_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    can_mask = cv2.erode(can_mask, np.ones((3, 3), np.uint8))

    # 5) Layer: original -> warped (full board) -> can (restored from original).
    result = frame_bgr.copy()
    result[board_mask > 0] = warped[board_mask > 0]
    result[can_mask > 0] = frame_bgr[can_mask > 0]

    return result, {
        "board_mask": board_mask,
        "chroma_mask": chroma_mask,
        "can_mask": can_mask,
        "board_corners_img": board_corners_img.astype(np.int32),
    }


def overlay_masks(display: np.ndarray, masks: dict) -> np.ndarray:
    """Tint the three masks so the user can see what's going where."""
    out = display.copy()
    overlay = np.zeros_like(out)
    overlay[masks["board_mask"] > 0]  = (0, 80, 0)     # board area = dark green tint
    overlay[masks["chroma_mask"] > 0] = (80, 80, 0)    # chroma = teal tint
    overlay[masks["can_mask"] > 0]    = (0, 0, 160)    # can = red tint
    out = cv2.addWeighted(out, 0.7, overlay, 0.3, 0)
    cv2.polylines(out, [masks["board_corners_img"]], True, (0, 255, 0), 2)
    return out


def draw_hud(frame, lines):
    bg_h = 22 * len(lines) + 8
    cv2.rectangle(frame, (0, 0), (480, bg_h), (0, 0, 0), -1)
    for i, (txt, col) in enumerate(lines):
        cv2.putText(frame, txt, (8, 22 + 22 * i),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # camera
    ap.add_argument("--camera-index", type=int, default=0)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30)
    # board geometry
    ap.add_argument("--squares-x", type=int, default=5)
    ap.add_argument("--squares-y", type=int, default=7)
    ap.add_argument("--square-mm", type=float, default=22.8)
    ap.add_argument("--marker-ratio", type=float, default=0.7)
    ap.add_argument("--chroma-mm", type=float, default=60.0)
    ap.add_argument("--dict", dest="dict_name", default="4X4_50")
    # target
    ap.add_argument("--target-image", type=str, default=None,
                    help="Celebrity image path. If omitted, a checkered test pattern is used.")
    # HSV green key
    ap.add_argument("--hsv-lo", type=int, nargs=3, default=(35, 60, 40), metavar=("H", "S", "V"),
                    help="Lower HSV bound for the chroma green key (default 35 60 40).")
    ap.add_argument("--hsv-hi", type=int, nargs=3, default=(85, 255, 255), metavar=("H", "S", "V"),
                    help="Upper HSV bound for the chroma green key (default 85 255 255).")
    # mode
    ap.add_argument("--snapshot", action="store_true",
                    help="Don't open a window — grab N composited frames and exit.")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/eval3_charuco_compose"))
    args = ap.parse_args()

    board, dictionary = build_board(
        args.squares_x, args.squares_y, args.square_mm, args.marker_ratio, args.dict_name,
    )
    board_w_mm = args.squares_x * args.square_mm
    board_h_mm = args.squares_y * args.square_mm
    detector = aruco.ArucoDetector(dictionary, aruco.DetectorParameters())
    char_detector = aruco.CharucoDetector(board)

    # Render the target at ~10 px/mm internal resolution so warpPerspective downsamples cleanly.
    target_w_px = int(round(board_w_mm * 10))
    target_h_px = int(round(board_h_mm * 10))
    target = load_target(args.target_image, target_w_px, target_h_px)
    print(f"Target: {args.target_image or '(synthetic test pattern)'} "
          f"-> {target.shape[1]}x{target.shape[0]} px")
    print(f"Board : {args.squares_x}x{args.squares_y} squares of {args.square_mm}mm "
          f"= {board_w_mm:.1f}x{board_h_mm:.1f} mm  chroma {args.chroma_mm}mm  dict {args.dict_name}")
    print(f"HSV   : green key range [{tuple(args.hsv_lo)}, {tuple(args.hsv_hi)}]")

    read_bgr, close = open_camera(args.camera_index, args.width, args.height, args.fps)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Acquire homography from the first usable frame.
    print("Acquiring homography... (point camera at the board)")
    H_locked = None
    last_lock_info = (0, 0, None)
    t0 = time.time()
    while H_locked is None and time.time() - t0 < 15.0:
        bgr = read_bgr()
        H_locked, nm, nc, rms = lock_homography(bgr, detector, char_detector, board)
        last_lock_info = (nm, nc, rms)
        if H_locked is not None:
            print(f"  locked: markers={nm}  chess_corners={nc}  rms={rms:.2f} px")
        else:
            time.sleep(0.05)
    if H_locked is None:
        print(f"  WARNING: no lock in 15s "
              f"(last try: markers={last_lock_info[0]}, chess={last_lock_info[1]}); "
              f"will keep retrying live — press 'r' to force.")

    show_raw = False
    show_masks = False
    saved = 0
    fps_smooth = 0.0
    last_t = time.time()

    try:
        if args.snapshot:
            for i in range(args.n):
                bgr = read_bgr()
                if H_locked is None:
                    H_locked, *rest = lock_homography(bgr, detector, char_detector, board)
                if H_locked is None:
                    out_path = args.out_dir / f"compose_{i:02d}_RAW.png"
                    cv2.imwrite(str(out_path), bgr)
                    print(f"  {out_path} (no homography)")
                    continue
                result, _ = compose(bgr, target, H_locked,
                                    board_w_mm, board_h_mm, args.chroma_mm,
                                    tuple(args.hsv_lo), tuple(args.hsv_hi))
                out_path = args.out_dir / f"compose_{i:02d}.png"
                cv2.imwrite(str(out_path), result)
                print(f"  {out_path}")
                time.sleep(0.3)
            return

        print("Live: q=quit  s=save  r=relock  t=toggle raw  m=toggle masks")
        while True:
            bgr = read_bgr()
            now = time.time()
            inst = 1.0 / max(now - last_t, 1e-6)
            last_t = now
            fps_smooth = 0.9 * fps_smooth + 0.1 * inst if fps_smooth > 0 else inst

            if H_locked is None:
                H_try, nm, nc, rms = lock_homography(bgr, detector, char_detector, board)
                if H_try is not None:
                    H_locked = H_try
                    last_lock_info = (nm, nc, rms)
                    print(f"  acquired: markers={nm} chess={nc} rms={rms:.2f} px")

            if show_raw or H_locked is None:
                display = bgr.copy()
            else:
                composited, masks = compose(
                    bgr, target, H_locked,
                    board_w_mm, board_h_mm, args.chroma_mm,
                    tuple(args.hsv_lo), tuple(args.hsv_hi),
                )
                display = overlay_masks(composited, masks) if show_masks else composited
                cv2.polylines(display, [masks["board_corners_img"]], True, (0, 255, 0), 2)

            rms_s = (f"{last_lock_info[2]:.2f}"
                     if last_lock_info[2] is not None else "?")
            status = (("OK  rms=" + rms_s + " px")
                      if H_locked is not None else "NO LOCK")
            status_col = (0, 255, 0) if H_locked is not None else (0, 0, 255)
            mode = ("RAW" if show_raw else ("COMPOSITE+MASKS" if show_masks else "COMPOSITE"))
            draw_hud(display, [
                (f"FPS: {fps_smooth:.1f}", (0, 255, 0)),
                (f"Lock: markers={last_lock_info[0]} chess={last_lock_info[1]} {status}", status_col),
                (f"Mode: {mode}   (t=raw  m=masks  r=relock)", (200, 200, 0)),
            ])
            cv2.imshow("ChArUco compose (q s r t m)", display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('s'):
                out_path = args.out_dir / f"compose_{saved:02d}.png"
                cv2.imwrite(str(out_path), display)
                print(f"  saved {out_path}")
                saved += 1
            if key == ord('r'):
                H_locked = None
                print("  relocking on next good frame ...")
            if key == ord('t'):
                show_raw = not show_raw
            if key == ord('m'):
                show_masks = not show_masks
    finally:
        cv2.destroyAllWindows()
        close()


if __name__ == "__main__":
    main()
