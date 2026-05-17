#!/usr/bin/env python3
"""Live ChArUco detection check against a printed board.

Opens a camera, detects DICT_4X4_50 markers on every frame, draws them and the
projected board outline + chroma rectangle. Use this BEFORE recording teleop
episodes to confirm:
  - markers detect at your real camera distance / lighting,
  - the homography is stable frame-to-frame,
  - the chroma region projects where the can will actually sit.

Defaults assume the board printed by ``tools/eval3_make_charuco_board.py`` with
``--content-mm 130x180 --squares-x 5 --squares-y 7 --chroma-mm 60``.

Live controls:
  q   quit
  s   save the current annotated frame to outputs/eval3_charuco_check/

Usage:
  python tools/eval3_charuco_check.py                                # live, camera 0
  python tools/eval3_charuco_check.py --camera-index 1
  python tools/eval3_charuco_check.py --snapshot --n 5               # batch mode
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
    sys.exit(f"opencv-contrib-python required: {e}\n  uv pip install 'opencv-contrib-python>=4.7'")


HUD_GREEN  = (0, 255, 0)
HUD_AMBER  = (0, 200, 255)
HUD_RED    = (0, 0, 255)
BOARD_RGB  = (0, 255, 0)     # board outline (BGR drawn)
CHROMA_RGB = (255, 80, 0)    # chroma rectangle outline (BGR drawn)


def build_board(squares_x: int, squares_y: int, square_mm: float,
                marker_ratio: float, dict_name: str):
    dict_id = getattr(aruco, f"DICT_{dict_name}", None)
    if dict_id is None:
        sys.exit(f"unknown dictionary DICT_{dict_name}")
    dictionary = aruco.getPredefinedDictionary(dict_id)
    board = aruco.CharucoBoard(
        (squares_x, squares_y),
        squareLength=square_mm / 1000.0,
        markerLength=square_mm * marker_ratio / 1000.0,
        dictionary=dictionary,
    )
    return board, dictionary


def annotate(
    frame_bgr: np.ndarray,
    detector: aruco.ArucoDetector,
    charuco_detector: "aruco.CharucoDetector | None",
    board: aruco.CharucoBoard,
    board_w_mm: float,
    board_h_mm: float,
    chroma_mm: float,
    expected_markers: int,
) -> tuple[np.ndarray, dict]:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    marker_corners, marker_ids, _ = detector.detectMarkers(gray)
    info = {
        "n_markers": 0 if marker_ids is None else len(marker_ids),
        "expected_markers": expected_markers,
        "n_chess_corners": 0,
        "homography_ok": False,
        "reproj_rms_px": None,
    }
    out = frame_bgr.copy()
    if marker_ids is not None and len(marker_ids) > 0:
        aruco.drawDetectedMarkers(out, marker_corners, marker_ids)

    # Try ChArUco chessboard-corner refinement for a robust homography.
    chess_corners, chess_ids = None, None
    if charuco_detector is not None and marker_ids is not None and len(marker_ids) >= 1:
        try:
            chess_corners, chess_ids, _, _ = charuco_detector.detectBoard(
                gray, markerCorners=marker_corners, markerIds=marker_ids
            )
        except Exception:
            chess_corners, chess_ids = None, None
    if chess_corners is not None and chess_ids is not None and len(chess_ids) >= 4:
        info["n_chess_corners"] = int(len(chess_ids))
        try:
            obj_pts, img_pts = board.matchImagePoints(chess_corners, chess_ids)
        except Exception:
            obj_pts, img_pts = None, None
        if obj_pts is not None and len(obj_pts) >= 4:
            src_mm = obj_pts.reshape(-1, 3)[:, :2] * 1000.0  # m → mm
            dst_px = img_pts.reshape(-1, 2)
            H, mask_h = cv2.findHomography(
                src_mm.astype(np.float32), dst_px.astype(np.float32),
                method=cv2.RANSAC, ransacReprojThreshold=2.0,
            )
            if H is not None:
                info["homography_ok"] = True
                # Reprojection RMS in pixels (inliers only).
                inliers = mask_h.ravel().astype(bool)
                src_in = src_mm[inliers]
                dst_in = dst_px[inliers]
                proj = cv2.perspectiveTransform(src_in.reshape(-1, 1, 2).astype(np.float32), H).reshape(-1, 2)
                info["reproj_rms_px"] = float(np.sqrt(np.mean(np.sum((proj - dst_in) ** 2, axis=1))))
                # Draw the full board outline (0..board_w_mm, 0..board_h_mm).
                board_corners_mm = np.float32([
                    [0, 0], [board_w_mm, 0],
                    [board_w_mm, board_h_mm], [0, board_h_mm],
                ]).reshape(-1, 1, 2)
                board_proj = cv2.perspectiveTransform(board_corners_mm, H).reshape(-1, 2)
                cv2.polylines(out, [board_proj.astype(np.int32)], True, BOARD_RGB, 2)
                # Draw the chroma rectangle.
                cx, cy = board_w_mm / 2.0, board_h_mm / 2.0
                ch = chroma_mm / 2.0
                chroma_mm_arr = np.float32([
                    [cx - ch, cy - ch], [cx + ch, cy - ch],
                    [cx + ch, cy + ch], [cx - ch, cy + ch],
                ]).reshape(-1, 1, 2)
                chroma_proj = cv2.perspectiveTransform(chroma_mm_arr, H).reshape(-1, 2)
                cv2.polylines(out, [chroma_proj.astype(np.int32)], True, CHROMA_RGB, 2)
    elif info["n_markers"] >= 4:
        # Fallback: draw min-area-rect of detected markers so the user has *some*
        # visual cue even when chessboard corners failed to refine.
        all_pts = np.concatenate([c[0] for c in marker_corners]).astype(np.float32)
        rect = cv2.minAreaRect(all_pts)
        box = cv2.boxPoints(rect).astype(np.int32)
        cv2.polylines(out, [box], True, HUD_AMBER, 1)
    return out, info


def draw_hud(frame: np.ndarray, info: dict, fps: float) -> None:
    n_m, exp = info["n_markers"], info["expected_markers"]
    n_c = info["n_chess_corners"]
    if info["homography_ok"]:
        status_color = HUD_GREEN
    elif n_m >= 4:
        status_color = HUD_AMBER
    else:
        status_color = HUD_RED
    lines = [
        (f"FPS: {fps:.1f}",                                       HUD_GREEN),
        (f"Markers: {n_m} / {exp}",                                status_color),
        (f"Chess corners: {n_c}",                                  status_color),
        ((f"Homography OK  RMS={info['reproj_rms_px']:.2f} px"
          if info["homography_ok"]
          else ("Bounding-rect fallback (no chess corners)"
                if n_m >= 4 else "Insufficient markers")),         status_color),
    ]
    bg_h = 22 * len(lines) + 8
    cv2.rectangle(frame, (0, 0), (380, bg_h), (0, 0, 0), thickness=cv2.FILLED)
    for i, (line, col) in enumerate(lines):
        cv2.putText(frame, line, (8, 22 + 22 * i),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)


def open_camera(camera_index: int, width: int, height: int, fps: int):
    """Try the lerobot OpenCV wrapper first (matches other tools/scripts), fall
    back to plain cv2.VideoCapture if lerobot isn't importable in this env."""
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
        from eval3_lerobot_shim import apply as _shim
        _shim()
        from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
        from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
        cfg = OpenCVCameraConfig(index_or_path=camera_index, width=width, height=height, fps=fps)
        cam = OpenCVCamera(cfg)
        cam.connect()
        def read_bgr():
            rgb = cam.async_read(timeout_ms=2000)
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        def close():
            try: cam.disconnect()
            except Exception: pass
        # Warm-up — first frames are often black on macOS.
        for _ in range(5):
            try: _ = cam.async_read(timeout_ms=2000)
            except Exception: pass
            time.sleep(0.05)
        return read_bgr, close
    except Exception as e:
        print(f"lerobot camera unavailable ({e}); falling back to cv2.VideoCapture")
        cap = cv2.VideoCapture(camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)
        if not cap.isOpened():
            sys.exit(f"cannot open camera index {camera_index}")
        def read_bgr():
            ok, bgr = cap.read()
            if not ok:
                raise RuntimeError("camera read failed")
            return bgr
        def close():
            cap.release()
        return read_bgr, close


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera-index", type=int, default=0)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30)
    # Board geometry — keep aligned with eval3_make_charuco_board.py defaults.
    ap.add_argument("--squares-x", type=int, default=5)
    ap.add_argument("--squares-y", type=int, default=7)
    ap.add_argument("--square-mm", type=float, default=22.8)
    ap.add_argument("--marker-ratio", type=float, default=0.7)
    ap.add_argument("--chroma-mm", type=float, default=60.0)
    ap.add_argument("--dict", dest="dict_name", default="4X4_50")
    # Mode
    ap.add_argument("--snapshot", action="store_true",
                    help="Don't open a window — grab N annotated frames and exit.")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/eval3_charuco_check"))
    args = ap.parse_args()

    board, dictionary = build_board(
        args.squares_x, args.squares_y, args.square_mm, args.marker_ratio, args.dict_name,
    )
    board_w_mm = args.squares_x * args.square_mm
    board_h_mm = args.squares_y * args.square_mm
    expected_markers = int(len(board.getIds()))
    detector = aruco.ArucoDetector(dictionary, aruco.DetectorParameters())
    try:
        charuco_detector = aruco.CharucoDetector(board)
    except Exception:
        charuco_detector = None

    print(f"Board   : {args.squares_x}x{args.squares_y} squares of {args.square_mm}mm "
          f"= {board_w_mm:.1f}x{board_h_mm:.1f}mm   chroma {args.chroma_mm}mm   dict {args.dict_name}")
    print(f"Expected {expected_markers} markers per board.\n")

    read_bgr, close = open_camera(args.camera_index, args.width, args.height, args.fps)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    try:
        if args.snapshot:
            for i in range(args.n):
                bgr = read_bgr()
                annotated, info = annotate(
                    bgr, detector, charuco_detector, board,
                    board_w_mm, board_h_mm, args.chroma_mm, expected_markers,
                )
                draw_hud(annotated, info, fps=float(args.fps))
                out_path = args.out_dir / f"check_{i:02d}.png"
                cv2.imwrite(str(out_path), annotated)
                print(f"  {out_path}  markers={info['n_markers']}/{expected_markers}  "
                      f"chess={info['n_chess_corners']}  hom={info['homography_ok']}  "
                      f"rms={info['reproj_rms_px']}")
                time.sleep(0.3)
        else:
            print("Live preview. q=quit  s=save current frame.")
            saved = 0
            t0 = time.time()
            fps_smooth = 0.0
            last = t0
            while True:
                bgr = read_bgr()
                now = time.time()
                inst = 1.0 / max(now - last, 1e-6)
                last = now
                fps_smooth = 0.9 * fps_smooth + 0.1 * inst if fps_smooth > 0 else inst
                annotated, info = annotate(
                    bgr, detector, charuco_detector, board,
                    board_w_mm, board_h_mm, args.chroma_mm, expected_markers,
                )
                draw_hud(annotated, info, fps=fps_smooth)
                cv2.imshow("ChArUco check (q=quit, s=save)", annotated)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                if key == ord('s'):
                    out_path = args.out_dir / f"snap_{saved:02d}.png"
                    cv2.imwrite(str(out_path), annotated)
                    print(f"  saved {out_path}  markers={info['n_markers']}/{expected_markers}  "
                          f"chess={info['n_chess_corners']}  hom={info['homography_ok']}")
                    saved += 1
    finally:
        cv2.destroyAllWindows()
        close()


if __name__ == "__main__":
    main()
