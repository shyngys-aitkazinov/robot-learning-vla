#!/usr/bin/env python3
"""Multi-board ChArUco celebrity inpainting demo on one recorded episode.

Smoke-test for the ChArUco synthetic-on-real pipeline (see
``docs/eval3/charuco_pipeline.md`` §5). Pulls one episode out of a recorded
LeRobot v3.0 dataset, picks three random celebrities, locks one homography
per visible board on frame 0, and walks every frame warping the three
celebrity images onto the three boards while preserving the Coke can on top
via HSV chroma keying.

Outputs (default ``outputs/eval3_charuco_demo/``):
    ep{N}_raw.mp4                — episode stream-copied out of the chunked dataset MP4
    ep{N}_composed.mp4           — same length, multi-celebrity composited
    ep{N}_composed_first_frame.png  — first composited frame for one-glance sanity
    ep{N}_composed.json          — sidecar metadata (which celeb went where, etc.)

Defaults match the rest of the ChArUco tooling: 5x7 grid, 22.8mm squares,
60mm chroma, DICT_4X4_50, HSV green [(35,60,40), (85,255,255)].

Usage:
    python tools/eval3_charuco_episode_demo.py
    python tools/eval3_charuco_episode_demo.py --dataset datasets/dataset_v3_charuco_middle_1 --episode 3
    python tools/eval3_charuco_episode_demo.py --seed 7
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
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

import pyarrow.parquet as pq

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from eval3_charuco_check import build_board


# ---------------------------------------------------------------------------
# Multi-board homography lock
# ---------------------------------------------------------------------------

def _cluster_markers_by_x(
    marker_corners: list[np.ndarray],
    marker_ids: np.ndarray,
    k: int,
) -> list[tuple[list[np.ndarray], np.ndarray]]:
    """Split detected markers into ``k`` groups by image-x of each marker centroid.

    Sort centroid x ascending, then quantile-cut into k equal-sized buckets.
    Sufficient for a well-separated semicircle layout (the boards are physically
    spread across the camera frame). Returns a list of (corners_subset, ids_subset)
    in left-to-right order.
    """
    centroids_x = np.array([c[0].mean(axis=0)[0] for c in marker_corners])
    order = np.argsort(centroids_x)
    # Quantile split: cut points at k-1 evenly spaced positions.
    n = len(order)
    cuts = [int(round(i * n / k)) for i in range(k + 1)]
    groups = []
    for i in range(k):
        idxs = order[cuts[i]:cuts[i + 1]]
        if len(idxs) == 0:
            groups.append(([], np.empty((0, 1), dtype=marker_ids.dtype)))
            continue
        sub_corners = [marker_corners[j] for j in idxs]
        sub_ids = marker_ids[idxs]
        groups.append((sub_corners, sub_ids))
    return groups


def _homography_from_markers_only(
    marker_corners: list[np.ndarray],
    marker_ids: np.ndarray,
    board: aruco.CharucoBoard,
) -> tuple[np.ndarray | None, dict]:
    """Fallback: compute the plane homography directly from marker corners.

    Used when ``CharucoDetector.detectBoard`` can't refine chessboard interior
    corners (steep angle, partial occlusion). Each detected ArUco marker
    contributes its 4 corner points; with ID collisions across boards (one
    DICT_4X4_50 shared by all 3 boards), an ID may match multiple object-point
    candidates — we just take the first match per ID, which is fine because
    the spatial cluster already restricts us to one board's worth of markers.
    """
    board_ids = board.getIds().flatten()
    obj_points_all = board.getObjPoints()  # list of (4,1,3) per marker
    src_mm, dst_px = [], []
    for mid, mc in zip(marker_ids.flatten(), marker_corners):
        hits = np.where(board_ids == int(mid))[0]
        if len(hits) == 0:
            continue
        obj_corners = obj_points_all[hits[0]].reshape(-1, 3) * 1000.0  # m -> mm
        src_mm.append(obj_corners[:, :2])
        dst_px.append(mc[0])  # mc shape (1,4,2)
    if not src_mm:
        return None, {"used_fallback": True, "n_marker_pts": 0}
    src_mm_a = np.concatenate(src_mm).astype(np.float32)
    dst_px_a = np.concatenate(dst_px).astype(np.float32)
    if len(src_mm_a) < 4:
        return None, {"used_fallback": True, "n_marker_pts": int(len(src_mm_a))}
    H, mask_h = cv2.findHomography(src_mm_a, dst_px_a, cv2.RANSAC, 3.0)
    if H is None:
        return None, {"used_fallback": True, "n_marker_pts": int(len(src_mm_a))}
    inl = mask_h.ravel().astype(bool)
    proj = cv2.perspectiveTransform(src_mm_a[inl].reshape(-1, 1, 2), H).reshape(-1, 2)
    rms = float(np.sqrt(np.mean(np.sum((proj - dst_px_a[inl]) ** 2, axis=1))))
    return H, {"used_fallback": True, "n_marker_pts": int(len(src_mm_a)),
               "n_inliers": int(inl.sum()), "rms": rms}


def _homography_from_group(
    gray: np.ndarray,
    marker_corners: list[np.ndarray],
    marker_ids: np.ndarray,
    char_detector: aruco.CharucoDetector,
    board: aruco.CharucoBoard,
) -> tuple[np.ndarray | None, dict]:
    """Run chessboard-corner refinement + plane homography for one board's markers.

    Falls back to marker-corner-only homography if chessboard refinement
    yields fewer than 4 usable corners (steep angle / partial occlusion).
    Mirrors the logic in ``eval3_charuco_compose.lock_homography`` but
    operates on a marker subset (one board's worth) rather than the whole frame.
    """
    info = {"n_markers": int(len(marker_ids)), "n_chess": 0, "rms": None,
            "used_fallback": False}
    if len(marker_ids) < 4:
        return None, info
    # --- preferred path: chessboard-corner refinement -----------------
    try:
        chess_corners, chess_ids, _, _ = char_detector.detectBoard(
            gray, markerCorners=marker_corners, markerIds=marker_ids,
        )
    except Exception:
        chess_corners, chess_ids = None, None
    if chess_corners is not None and chess_ids is not None and len(chess_ids) >= 4:
        info["n_chess"] = int(len(chess_ids))
        obj_pts, img_pts = board.matchImagePoints(chess_corners, chess_ids)
        if obj_pts is not None and len(obj_pts) >= 4:
            src_mm = (obj_pts.reshape(-1, 3)[:, :2] * 1000.0).astype(np.float32)
            dst_px = img_pts.reshape(-1, 2).astype(np.float32)
            H, mask_h = cv2.findHomography(src_mm, dst_px, cv2.RANSAC, 2.0)
            if H is not None:
                inl = mask_h.ravel().astype(bool)
                proj = cv2.perspectiveTransform(src_mm[inl].reshape(-1, 1, 2), H).reshape(-1, 2)
                info["rms"] = float(np.sqrt(np.mean(np.sum((proj - dst_px[inl]) ** 2, axis=1))))
                return H, info
    # --- fallback: marker-corner-only solve ---------------------------
    H_fb, info_fb = _homography_from_markers_only(marker_corners, marker_ids, board)
    info.update(info_fb)
    return H_fb, info


def lock_homographies_multi(
    frame_bgr: np.ndarray,
    detector: aruco.ArucoDetector,
    char_detector: aruco.CharucoDetector,
    board: aruco.CharucoBoard,
    n_boards: int = 3,
) -> tuple[list[np.ndarray | None], list[dict]]:
    """Lock one homography per spatially-clustered board cluster.

    Returns (list of H matrices, list of info dicts) ordered left-to-right by
    cluster centroid x. Any cluster that fails to produce a homography yields
    None at its slot; the caller decides how to handle partial failure.
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    marker_corners, marker_ids, _ = detector.detectMarkers(gray)
    if marker_ids is None or len(marker_ids) == 0:
        return [None] * n_boards, [{"n_markers": 0, "n_chess": 0, "rms": None}] * n_boards

    groups = _cluster_markers_by_x(marker_corners, marker_ids, n_boards)
    Hs: list[np.ndarray | None] = []
    infos: list[dict] = []
    for g_corners, g_ids in groups:
        H, info = _homography_from_group(gray, g_corners, g_ids, char_detector, board)
        Hs.append(H)
        infos.append(info)
    return Hs, infos


# ---------------------------------------------------------------------------
# Multi-board composite
# ---------------------------------------------------------------------------

def _pick_upright_orientation(
    board_corners_img: np.ndarray,
    source_aspect_h_over_w: float,
) -> int:
    """Choose the 90° rotation that keeps source-aspect AND faces the robot.

    Two-level scoring:

    1. **Aspect-preserving**: prefer rotations where source-right maps to the
       destination's SHORT edge and source-down maps to the LONG edge — so a
       portrait source isn't squashed into a landscape destination by the warp.
       For our 5x7 board (114x160mm portrait), source-aspect h/w ≈ 1.4 — we
       want ``down_len > right_len`` in image space.
    2. **Robot-facing**: at eval time the celebrity prints lie on the table
       with the face's TOP (forehead) closer to the robot (= closer to the
       bottom of the camera frame). From the camera POV the face therefore
       appears UPSIDE DOWN — head at image-bottom, chin at image-top. So the
       picked rotation should put source-down (chin direction in the source
       JPG) at the SMALLEST image-y (chin at top of image, forehead at
       bottom). Implemented as ``MAX(-down_y)`` = MIN(down_y).
    """
    want_portrait_dst = source_aspect_h_over_w > 1.0
    best_aspect, best_orient, best_k = -1, -float("inf"), 0
    for k in range(4):
        rolled = np.roll(board_corners_img, -k, axis=0)
        right_vec = rolled[1] - rolled[0]
        down_vec = rolled[3] - rolled[0]
        right_len = float(np.linalg.norm(right_vec))
        down_len = float(np.linalg.norm(down_vec))
        if want_portrait_dst:
            aspect_ok = 1 if down_len > right_len else 0
        else:
            aspect_ok = 1 if right_len > down_len else 0
        # Face the robot: source-down (chin) should point toward image-up.
        # We want minimum down_y, equivalently maximum -down_y.
        robot_facing_score = -float(down_vec[1])
        if (aspect_ok, robot_facing_score) > (best_aspect, best_orient):
            best_aspect, best_orient, best_k = aspect_ok, robot_facing_score, k
    return best_k


def _center_crop_to_aspect(img: np.ndarray, target_aspect_h_over_w: float) -> np.ndarray:
    """Centre-crop an image so its h/w ratio matches the target.

    Preferred over letterboxing for this demo: the celebrity must fill the
    entire board area (no neutral bars), so we crop a centred sub-rectangle
    of the source whose aspect matches the board's. Small amount of the
    source's longest dimension gets cut (typically a strip of background /
    hair / chin), which is preferable to dark bars at eval time.
    """
    h, w = img.shape[:2]
    cur = h / w
    if abs(cur - target_aspect_h_over_w) < 1e-3:
        return img
    if cur > target_aspect_h_over_w:
        # too tall -> crop top + bottom
        new_h = int(round(w * target_aspect_h_over_w))
        crop = (h - new_h) // 2
        return img[crop:crop + new_h, :]
    # too wide -> crop left + right
    new_w = int(round(h / target_aspect_h_over_w))
    crop = (w - new_w) // 2
    return img[:, crop:crop + new_w]


# ---------------------------------------------------------------------------
# Can-mask strategies (plug-in interface for compose_multi)
# ---------------------------------------------------------------------------

# Hue bands for the Coke can's red label. HSV hue wraps around 0/180.
_COKE_RED_LO_1 = np.array([  0, 130,  80], dtype=np.uint8)
_COKE_RED_HI_1 = np.array([ 15, 255, 255], dtype=np.uint8)
_COKE_RED_LO_2 = np.array([165, 130,  80], dtype=np.uint8)
_COKE_RED_HI_2 = np.array([179, 255, 255], dtype=np.uint8)


def _detect_coke_red(frame_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    return cv2.bitwise_or(
        cv2.inRange(hsv, _COKE_RED_LO_1, _COKE_RED_HI_1),
        cv2.inRange(hsv, _COKE_RED_LO_2, _COKE_RED_HI_2),
    )


def _largest_red_bbox(frame_bgr: np.ndarray, min_area: int = 50) -> tuple[int, int, int, int] | None:
    """Bounding box of the largest connected red component (= can label)."""
    red = _detect_coke_red(frame_bgr)
    n, _, stats, _ = cv2.connectedComponentsWithStats(red, connectivity=8)
    if n < 2:
        return None
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = areas >= min_area
    if not keep.any():
        return None
    i = 1 + int(np.argmax(np.where(keep, areas, -1)))
    x = stats[i, cv2.CC_STAT_LEFT]; y = stats[i, cv2.CC_STAT_TOP]
    w = stats[i, cv2.CC_STAT_WIDTH]; h = stats[i, cv2.CC_STAT_HEIGHT]
    return int(x), int(y), int(w), int(h)


class _BaseMasker:
    """Stateful per-strategy mask builder. Subclasses override ``compute``."""
    def __init__(self, hsv_lo: tuple[int, int, int], hsv_hi: tuple[int, int, int]):
        self.hsv_lo = np.array(hsv_lo, dtype=np.uint8)
        self.hsv_hi = np.array(hsv_hi, dtype=np.uint8)

    def init(self, frame_bgr: np.ndarray, chroma_union: np.ndarray) -> None:
        pass

    def compute(self, frame_bgr: np.ndarray, chroma_union: np.ndarray,
                board_quads_img: list[np.ndarray]) -> np.ndarray:
        raise NotImplementedError

    def _green_mask(self, frame_bgr: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        return cv2.inRange(hsv, self.hsv_lo, self.hsv_hi)

    @staticmethod
    def _cleanup(mask: np.ndarray) -> np.ndarray:
        m = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        return cv2.erode(m, np.ones((3, 3), np.uint8))

    @staticmethod
    def _board_area_mask(shape, board_quads_img) -> np.ndarray:
        """Union of all projected board quads — the area painted with celebs.
        Most extended detection paths want to know 'am I inside the board area
        where the chessboard markers live?' so they can SKIP marker pixels rather
        than the whole board area (which would also skip the can sitting on the
        chroma).
        """
        h, w = shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        for q in board_quads_img:
            cv2.fillConvexPoly(mask, q.astype(np.int32), 255)
        return mask

    @staticmethod
    def _marker_mask(
        frame_bgr: np.ndarray,
        dark_v: int = 60,
        white_v: int = 175,
        low_s: int = 55,
    ) -> np.ndarray:
        """Pixels that look like ChArUco chessboard markers — either nearly-black
        OR nearly-white-with-low-saturation. Subtracted from extended can
        detection so the markers don't leak through the celebrity face.

        Empirically the captured chessboard 'black' averages ~40 in v with
        s~30, and 'white' averages ~193 with s~10 (measured against the
        digital rendering's 0 and 255). Defaults are a bit wider than that to
        catch lighting variation across the board.

        Side-effect: the Coke can's silver top (low saturation, v 120-180) may
        partially overlap the dark side of this band — fine in practice since
        the red label is still preserved and the white logo's pure-white pixels
        are already lost to the white-v cutoff regardless.
        """
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        _, s, v = cv2.split(hsv)
        dark = (v < dark_v) & (s < low_s)
        light = (v > white_v) & (s < low_s)
        return (dark | light).astype(np.uint8) * 255


class ChromaOnlyMasker(_BaseMasker):
    """Baseline (current behavior): chroma rectangles, non-green pixels = can.

    Fails when the can extends BEYOND the chroma rectangle — the can body and
    gripper get painted over by the warped celebrity.
    """
    def compute(self, frame_bgr, chroma_union, board_quads_img):
        green = self._green_mask(frame_bgr)
        return self._cleanup(cv2.bitwise_and(chroma_union, cv2.bitwise_not(green)))


class RedHSVMasker(_BaseMasker):
    """Option 1: HSV red detection (anywhere in frame) + dilation for the white
    Coca-Cola logo + silver top + base. Also includes the gripper via a dark-pixel
    detection in a vertical column above each board centroid. Restricted to a
    circular buffer around board centroids to avoid spurious detections from the
    background or the warped celebrity image (which only exists in the composited
    frame, not in the original passed here, but skin tones could match red).

    Cheapest of the four: ~1-2 ms per frame. No new dependencies.
    """
    def __init__(self, hsv_lo, hsv_hi, buffer_px: int = 180, dilate_px: int = 10,
                 gripper_dark_thresh: int = 55, gripper_half_w_px: int = 70):
        super().__init__(hsv_lo, hsv_hi)
        self.buffer_px = buffer_px
        self.dilate_px = dilate_px
        self.gripper_dark_thresh = gripper_dark_thresh
        self.gripper_half_w_px = gripper_half_w_px

    def compute(self, frame_bgr, chroma_union, board_quads_img):
        h, w = frame_bgr.shape[:2]
        green = self._green_mask(frame_bgr)
        marker = self._marker_mask(frame_bgr)
        not_garbage = cv2.bitwise_not(cv2.bitwise_or(green, marker))

        red = _detect_coke_red(frame_bgr)
        centroids = [q.mean(axis=0) for q in board_quads_img]

        # Restrict red detection to a buffer zone around each board centroid.
        # (Filters out spurious red elsewhere in the scene — e.g. the robot's
        # base/peripherals if they happen to contain red.)
        buf = np.zeros((h, w), dtype=np.uint8)
        for cx, cy in centroids:
            cv2.circle(buf, (int(cx), int(cy)), self.buffer_px, 255, -1)
        red_buf = cv2.bitwise_and(red, buf)

        # Dilate red to include adjacent silver/white-logo pixels. We use a small
        # kernel (10 px ~= 1.5 % of the frame width) so dilation doesn't reach
        # nearby chessboard squares; the marker-mask subtraction catches any
        # squares that do get hit. No more outside-board restriction — the
        # marker filter lets us also preserve the can portion sitting inside
        # the board's projected area but above the chroma rectangle.
        k = max(3, self.dilate_px | 1)
        red_dilated = cv2.dilate(red_buf, np.ones((k, k), np.uint8))
        red_can = cv2.bitwise_and(red_dilated, not_garbage)

        # Gripper: dark pixels in a vertical rectangle ABOVE each board's topmost
        # projected point. Restricting to above the board (not just NOT-board)
        # tightens the gripper search and avoids false positives from the robot
        # base further down in the frame.
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        dark = cv2.inRange(gray, 0, self.gripper_dark_thresh)
        gripper_zone = np.zeros((h, w), dtype=np.uint8)
        for q in board_quads_img:
            cx = float(q.mean(axis=0)[0])
            top_y = max(0, int(q[:, 1].min()))
            x0 = max(0, int(cx) - self.gripper_half_w_px)
            x1 = min(w, int(cx) + self.gripper_half_w_px)
            cv2.rectangle(gripper_zone, (x0, 0), (x1, top_y), 255, -1)
        gripper = cv2.bitwise_and(dark, gripper_zone)

        # Safety net: on-chroma silver base, anything not-green inside chroma.
        chroma_can = cv2.bitwise_and(chroma_union, cv2.bitwise_not(green))

        combined = cv2.bitwise_or(chroma_can, red_can)
        combined = cv2.bitwise_or(combined, gripper)
        combined = cv2.bitwise_and(combined, cv2.bitwise_not(green))
        return self._cleanup(combined)


class ChromaDilateUpMasker(_BaseMasker):
    """Option 2: Dilate the chroma rectangles UPWARD (asymmetric kernel) and keep
    non-green pixels in the extended area. Color-agnostic — works for any object
    that isn't green. Also rejects very-bright table pixels so the table doesn't
    leak through above the board.
    """
    def __init__(self, hsv_lo, hsv_hi, dilate_up_px: int = 130,
                 dilate_h_px: int = 14, table_bright_thresh: int = 215):
        super().__init__(hsv_lo, hsv_hi)
        self.dilate_up_px = dilate_up_px
        self.dilate_h_px = dilate_h_px
        self.table_bright_thresh = table_bright_thresh

    def compute(self, frame_bgr, chroma_union, board_quads_img):
        h_img, w_img = frame_bgr.shape[:2]
        green = self._green_mask(frame_bgr)
        marker = self._marker_mask(frame_bgr)
        not_garbage = cv2.bitwise_not(cv2.bitwise_or(green, marker))

        # Build an upward-extended "can column" centered on each chroma's
        # centroid. The column starts at the chroma top and extends UP by
        # `dilate_up_px` (the can's expected height in pixels at this camera
        # scale). Used everywhere — marker filter handles the chessboard, and
        # the table-bright filter handles the table edges above the board.
        column = np.zeros((h_img, w_img), dtype=np.uint8)
        for q in board_quads_img:
            cx = float(q.mean(axis=0)[0])
            cy_top = max(0, int(q[:, 1].min()))
            cv2.rectangle(
                column,
                (max(0, int(cx) - self.dilate_h_px), max(0, cy_top - self.dilate_up_px)),
                (min(w_img, int(cx) + self.dilate_h_px), int(q[:, 1].max())),
                255, -1,
            )

        b, g, r = cv2.split(frame_bgr)
        table_mask = (
            (b > self.table_bright_thresh)
            & (g > self.table_bright_thresh)
            & (r > self.table_bright_thresh)
        )
        column_keep = cv2.bitwise_and(column, not_garbage)
        column_keep[table_mask] = 0

        chroma_can = cv2.bitwise_and(chroma_union, cv2.bitwise_not(green))
        return self._cleanup(cv2.bitwise_or(chroma_can, column_keep))


class GrabCutMasker(_BaseMasker):
    """Option 3: SAM-like refinement using OpenCV GrabCut, prompted by the red bbox.

    GrabCut converges to a pixel-accurate foreground mask from a coarse rectangle
    prompt. Fallback to chroma-only when no red is detected (can off-screen).
    Slowest of the four — ~80-150 ms per frame.
    """
    def __init__(self, hsv_lo, hsv_hi, expand_lr: int = 35, expand_up: int = 90, iters: int = 3):
        super().__init__(hsv_lo, hsv_hi)
        self.expand_lr = expand_lr
        self.expand_up = expand_up
        self.iters = iters

    def compute(self, frame_bgr, chroma_union, board_quads_img):
        h_img, w_img = frame_bgr.shape[:2]
        bbox = _largest_red_bbox(frame_bgr)
        if bbox is None:
            green = self._green_mask(frame_bgr)
            return self._cleanup(cv2.bitwise_and(chroma_union, cv2.bitwise_not(green)))
        x, y, w, h = bbox
        # Expand bbox to surround the full can (red is the LABEL only; can extends
        # above + slightly sideways).
        x0 = max(0, x - self.expand_lr)
        y0 = max(0, y - self.expand_up)
        x1 = min(w_img - 1, x + w + self.expand_lr)
        y1 = min(h_img - 1, y + h + self.expand_lr)
        rect = (x0, y0, x1 - x0, y1 - y0)
        if rect[2] <= 1 or rect[3] <= 1:
            green = self._green_mask(frame_bgr)
            return self._cleanup(cv2.bitwise_and(chroma_union, cv2.bitwise_not(green)))
        gc_mask = np.zeros(frame_bgr.shape[:2], dtype=np.uint8)
        bgd = np.zeros((1, 65), dtype=np.float64)
        fgd = np.zeros((1, 65), dtype=np.float64)
        try:
            cv2.grabCut(frame_bgr, gc_mask, rect, bgd, fgd, self.iters, cv2.GC_INIT_WITH_RECT)
        except cv2.error:
            green = self._green_mask(frame_bgr)
            return self._cleanup(cv2.bitwise_and(chroma_union, cv2.bitwise_not(green)))
        can = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
        # Exclude chessboard markers from the GrabCut output so they don't leak
        # through the celebrity face.
        marker = self._marker_mask(frame_bgr)
        can = cv2.bitwise_and(can, cv2.bitwise_not(marker))
        # Union with chroma-non-green as a safety net (the on-chroma silver base).
        green = self._green_mask(frame_bgr)
        chroma_can = cv2.bitwise_and(chroma_union, cv2.bitwise_not(green))
        return self._cleanup(cv2.bitwise_or(can, chroma_can))


class TrackerMasker(_BaseMasker):
    """Option 4: Initialise an OpenCV tracker (MIL — the only one available in
    this build) on frame 0 with the red-bbox-derived can rectangle. Tracks the
    can per-frame. Inside the tracked box: non-green AND not-bright = can pixels.
    Combined with chroma-non-green as a safety net.

    Trade-offs vs the others: handles arbitrary objects, robust to occlusion by
    the gripper, but the tracker can drift (MIL is the most basic OpenCV tracker
    — CSRT/KCF aren't compiled into this build).
    """
    def __init__(self, hsv_lo, hsv_hi, init_expand_lr: int = 25, init_expand_up: int = 90,
                 table_bright_thresh: int = 215):
        super().__init__(hsv_lo, hsv_hi)
        self.init_expand_lr = init_expand_lr
        self.init_expand_up = init_expand_up
        self.table_bright_thresh = table_bright_thresh
        self.tracker = None
        self.initialized = False

    def init(self, frame_bgr, chroma_union):
        bbox = _largest_red_bbox(frame_bgr)
        if bbox is None:
            return
        x, y, w, h = bbox
        h_img, w_img = frame_bgr.shape[:2]
        x0 = max(0, x - self.init_expand_lr)
        y0 = max(0, y - self.init_expand_up)
        x1 = min(w_img - 1, x + w + self.init_expand_lr)
        y1 = min(h_img - 1, y + h + self.init_expand_lr)
        try:
            self.tracker = cv2.TrackerMIL.create()
            self.tracker.init(frame_bgr, (x0, y0, x1 - x0, y1 - y0))
            self.initialized = True
        except Exception:
            self.initialized = False

    def compute(self, frame_bgr, chroma_union, board_quads_img):
        green = self._green_mask(frame_bgr)
        chroma_can = cv2.bitwise_and(chroma_union, cv2.bitwise_not(green))
        if not self.initialized:
            return self._cleanup(chroma_can)
        ok, bbox = self.tracker.update(frame_bgr)
        if not ok:
            return self._cleanup(chroma_can)
        x, y, w_b, h_b = (int(v) for v in bbox)
        h_img, w_img = frame_bgr.shape[:2]
        x0 = max(0, x); y0 = max(0, y)
        x1 = min(w_img, x + w_b); y1 = min(h_img, y + h_b)
        bbox_mask = np.zeros((h_img, w_img), dtype=np.uint8)
        if x1 > x0 and y1 > y0:
            bbox_mask[y0:y1, x0:x1] = 255
        # Inside the tracked bbox: keep non-green non-marker non-bright-table pixels.
        marker = self._marker_mask(frame_bgr)
        b, g, r = cv2.split(frame_bgr)
        table = (b > self.table_bright_thresh) & (g > self.table_bright_thresh) & (r > self.table_bright_thresh)
        inside = cv2.bitwise_and(bbox_mask, cv2.bitwise_not(green))
        inside = cv2.bitwise_and(inside, cv2.bitwise_not(marker))
        inside[table] = 0
        return self._cleanup(cv2.bitwise_or(inside, chroma_can))


_MASKER_REGISTRY: dict[str, type] = {
    "chroma_only":       ChromaOnlyMasker,
    "red_hsv":           RedHSVMasker,
    "chroma_dilate_up":  ChromaDilateUpMasker,
    "grabcut":           GrabCutMasker,
    "tracker":           TrackerMasker,
}


def _apply_table_blend(
    img: np.ndarray,
    contrast: float = 0.6,
    brightness_offset: int = 40,
    saturation: float = 0.75,
    blur_kernel: int = 3,
    noise_sigma: float = 5.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Apply printer + camera distortions so the warped celeb blends with the table.

    Measured from this rig's recordings of the printed ChArUco boards:
      - The contrast compresses from [0, 255] digital to ~[40, 193] captured
        (linear ~0.6x scale with +40 offset on each channel).
      - Saturation drops ~25-30% on the printed-on-paper green chroma.
      - Ambient sensor noise + lighting gradient yield pixel std ~5-8 in
        nominally-uniform regions.
      - Slight focus blur (~1 px) from camera depth-of-field.

    Defaults reproduce that distortion profile. All knobs are exposed as
    --blend-* CLI flags so the user can iterate.
    """
    out = img.astype(np.float32)

    # 1) Brightness + contrast: compress dynamic range to match the captured rig.
    out = out * contrast + brightness_offset

    # 2) Saturation: convert to HSV, scale S, convert back.
    if saturation != 1.0:
        out_u8 = np.clip(out, 0, 255).astype(np.uint8)
        hsv = cv2.cvtColor(out_u8, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 1] *= saturation
        out_u8 = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
        out = out_u8.astype(np.float32)

    # 3) Slight blur (focus + sensor PSF).
    if blur_kernel >= 3 and blur_kernel % 2 == 1:
        out = cv2.GaussianBlur(out, (blur_kernel, blur_kernel), 0)

    # 4) Sensor noise — Gaussian per pixel per channel.
    if noise_sigma > 0:
        if rng is None:
            rng = np.random.default_rng(0)
        noise = rng.normal(0.0, noise_sigma, size=out.shape).astype(np.float32)
        out = out + noise

    return np.clip(out, 0, 255).astype(np.uint8)


def compose_multi(
    frame_bgr: np.ndarray,
    targets: list[np.ndarray],
    homographies: list[np.ndarray | None],
    board_w_mm: float,
    board_h_mm: float,
    chroma_mm: float,
    hsv_lo: tuple[int, int, int],
    hsv_hi: tuple[int, int, int],
    masker: _BaseMasker | None = None,
) -> np.ndarray:
    """Warp N targets onto N boards in one frame and preserve the can on top.

    Uses a single combined can mask (union of all chroma regions, AND NOT
    green) so the can is restored from the ORIGINAL frame regardless of which
    board it sits on or whether it straddles an edge.
    """
    h_img, w_img = frame_bgr.shape[:2]
    board_corners_mm = np.float32([
        [0, 0], [board_w_mm, 0],
        [board_w_mm, board_h_mm], [0, board_h_mm],
    ]).reshape(-1, 1, 2)
    cx, cy = board_w_mm / 2.0, board_h_mm / 2.0
    ch = chroma_mm / 2.0
    chroma_corners_mm = np.float32([
        [cx - ch, cy - ch], [cx + ch, cy - ch],
        [cx + ch, cy + ch], [cx - ch, cy + ch],
    ]).reshape(-1, 1, 2)

    result = frame_bgr.copy()
    chroma_union = np.zeros((h_img, w_img), dtype=np.uint8)
    board_quads_img: list[np.ndarray] = []

    for target, H in zip(targets, homographies):
        if H is None:
            continue
        # Per-board outline in image px.
        board_img = cv2.perspectiveTransform(board_corners_mm, H).reshape(-1, 2).astype(np.float32)
        board_quads_img.append(board_img)
        # Pick the rotation that preserves source aspect AND puts the face
        # right-side up. Source aspect comes from the target tensor we'll warp.
        th, tw = target.shape[:2]
        k = _pick_upright_orientation(board_img, source_aspect_h_over_w=th / tw)
        dst = np.roll(board_img, -k, axis=0)
        # Per-frame target homography (target px -> rotated destination quad).
        target_src = np.float32([[0, 0], [tw - 1, 0], [tw - 1, th - 1], [0, th - 1]])
        H_target = cv2.getPerspectiveTransform(target_src, dst)
        warped = cv2.warpPerspective(target, H_target, (w_img, h_img))
        board_mask = np.zeros((h_img, w_img), dtype=np.uint8)
        cv2.fillConvexPoly(board_mask, board_img.astype(np.int32), 255)
        result[board_mask > 0] = warped[board_mask > 0]

        # Accumulate the chroma rectangle for the combined can mask.
        chroma_img = cv2.perspectiveTransform(chroma_corners_mm, H).reshape(-1, 2).astype(np.int32)
        cv2.fillConvexPoly(chroma_union, chroma_img, 255)

    if not chroma_union.any():
        return result

    if masker is None:
        masker = ChromaOnlyMasker(hsv_lo, hsv_hi)
    can_mask = masker.compute(frame_bgr, chroma_union, board_quads_img)
    result[can_mask > 0] = frame_bgr[can_mask > 0]
    return result


# ---------------------------------------------------------------------------
# Episode metadata + extraction
# ---------------------------------------------------------------------------

def load_episode_meta(dataset_root: Path, episode_idx: int, camera_key: str) -> dict:
    """Read the meta/episodes parquet shards and pick out one episode's row."""
    files = sorted(dataset_root.glob("meta/episodes/chunk-*/file-*.parquet"))
    if not files:
        sys.exit(f"no meta/episodes parquet files under {dataset_root}")
    cols = [
        "episode_index", "length",
        "dataset_from_index", "dataset_to_index",
        f"videos/{camera_key}/chunk_index",
        f"videos/{camera_key}/file_index",
        f"videos/{camera_key}/from_timestamp",
        f"videos/{camera_key}/to_timestamp",
    ]
    for f in files:
        df = pq.read_table(f, columns=cols).to_pandas()
        match = df[df["episode_index"] == episode_idx]
        if len(match) > 0:
            r = match.iloc[0]
            return {
                "episode_index": int(r["episode_index"]),
                "length": int(r["length"]),
                "from_index": int(r["dataset_from_index"]),
                "to_index": int(r["dataset_to_index"]),
                "video_chunk": int(r[f"videos/{camera_key}/chunk_index"]),
                "video_file": int(r[f"videos/{camera_key}/file_index"]),
                "from_timestamp": float(r[f"videos/{camera_key}/from_timestamp"]),
                "to_timestamp": float(r[f"videos/{camera_key}/to_timestamp"]),
            }
    sys.exit(f"episode {episode_idx} not found in {dataset_root}/meta/episodes/")


def transcode_to_h264(src_mp4: Path, dst_mp4: Path) -> None:
    """Re-encode an mp4v / mpeg4 file to H.264 for QuickLook / Safari previewability.

    cv2.VideoWriter with fourcc 'mp4v' produces MPEG-4 Part 2, which macOS
    Quick Look and Safari refuse to preview inline. Hardware-accelerated
    h264_videotoolbox on Apple Silicon does the transcode in ~1-2s for a 30s
    clip. Falls back to libx264 if h264_videotoolbox isn't available.
    """
    common = ["-loglevel", "error", "-y", "-i", str(src_mp4),
              "-pix_fmt", "yuv420p", "-movflags", "+faststart"]
    # libx264 first — slightly slower (CPU-only) but no I-frame warmup
    # artifacts. h264_videotoolbox is the hardware fallback for environments
    # where libx264 isn't compiled in.
    for codec_args in (["-c:v", "libx264", "-preset", "fast", "-crf", "20"],
                       ["-c:v", "h264_videotoolbox", "-b:v", "4M"]):
        cmd = ["ffmpeg", *common, *codec_args, str(dst_mp4)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode == 0:
            return
    raise RuntimeError(f"ffmpeg failed to transcode {src_mp4} to H.264")


def extract_episode_mp4(dataset_root: Path, ep: dict, camera_key: str, out_mp4: Path) -> None:
    """Stream-copy the episode's range out of the chunked dataset MP4."""
    src = (dataset_root / "videos" / camera_key /
           f"chunk-{ep['video_chunk']:03d}" / f"file-{ep['video_file']:03d}.mp4")
    if not src.exists():
        sys.exit(f"source video {src} missing")
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    if out_mp4.exists():
        out_mp4.unlink()
    cmd = [
        "ffmpeg", "-loglevel", "error",
        "-ss", f"{ep['from_timestamp']:.3f}",
        "-to", f"{ep['to_timestamp']:.3f}",
        "-i", str(src), "-c", "copy", str(out_mp4),
    ]
    subprocess.run(cmd, check=True)


# ---------------------------------------------------------------------------
# Celebrity sampling
# ---------------------------------------------------------------------------

def pick_celebrities(
    top30_json: Path,
    rng: random.Random,
    min_portrait_frac: float = 0.80,
    min_images: int = 30,
    n: int = 3,
) -> list[dict]:
    """Pick ``n`` celebrities + one random portrait JPG each, deterministically."""
    meta = json.loads(top30_json.read_text(encoding="utf-8"))
    pool = [
        c for c in meta["celebrities"]
        if not c.get("held_out")
        and c.get("aspect_portrait_frac", 0.0) >= min_portrait_frac
        and c.get("n_images", 0) >= min_images
    ]
    if len(pool) < n:
        sys.exit(f"only {len(pool)} celebrities pass the filter; need {n}")
    chosen = rng.sample(pool, n)
    picks = []
    for c in chosen:
        jpgs = sorted(Path(c["dir"]).glob("*.jpg"))
        if not jpgs:
            sys.exit(f"no JPGs found under {c['dir']}")
        img_path = rng.choice(jpgs)
        picks.append({"celeb": c, "image_path": str(img_path)})
    return picks


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--dataset", type=Path,
                    default=Path("datasets/dataset_v3_charuco_left_1"))
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--camera-key", default="observation.images.front")
    ap.add_argument("--top30-json", type=Path,
                    default=Path("datasets/pins-face-recognition-top30.json"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", type=Path,
                    default=Path("outputs/eval3_charuco_demo"))
    # Board geometry — must match the printed/recorded board.
    ap.add_argument("--squares-x", type=int, default=5)
    ap.add_argument("--squares-y", type=int, default=7)
    ap.add_argument("--square-mm", type=float, default=22.8)
    ap.add_argument("--marker-ratio", type=float, default=0.7)
    ap.add_argument("--chroma-mm", type=float, default=60.0)
    ap.add_argument("--dict", dest="dict_name", default="4X4_50")
    # HSV green for the can-preservation chroma key.
    ap.add_argument("--hsv-lo", type=int, nargs=3, default=(35, 60, 40),
                    metavar=("H", "S", "V"))
    ap.add_argument("--hsv-hi", type=int, nargs=3, default=(85, 255, 255),
                    metavar=("H", "S", "V"))
    ap.add_argument("--n-boards", type=int, default=3,
                    help="Number of ChArUco boards visible in the scene (default 3 = semicircle layout).")
    # Table-blend distortion knobs. Defaults measured from the captured ChArUco rig.
    ap.add_argument("--blend", action=argparse.BooleanOptionalAction, default=True,
                    help="Apply brightness/contrast/saturation/blur/noise to celeb images "
                         "so they match the captured-on-table look (default: on; --no-blend to disable).")
    ap.add_argument("--blend-contrast", type=float, default=0.6,
                    help="Linear contrast multiplier (default 0.6 — compresses [0,255]->[~40,~193]).")
    ap.add_argument("--blend-brightness-offset", type=int, default=40,
                    help="Additive offset after contrast scaling (default 40 = measured black-lift).")
    ap.add_argument("--blend-saturation", type=float, default=0.75,
                    help="HSV S-channel multiplier (default 0.75 = ~25%% desaturation).")
    ap.add_argument("--blend-blur-kernel", type=int, default=3,
                    help="Gaussian blur kernel size, must be odd >=3 (default 3, set 1 to disable).")
    ap.add_argument("--blend-noise-sigma", type=float, default=5.0,
                    help="Per-pixel Gaussian noise stddev added after blur (default 5.0).")
    ap.add_argument("--can-mask-strategies", type=str,
                    default="red_hsv,chroma_only,chroma_dilate_up,grabcut,tracker",
                    help="Comma-separated list of can-mask strategies to run. One MP4 per "
                         "strategy is written. Options: chroma_only (baseline), red_hsv, "
                         "chroma_dilate_up, grabcut, tracker. First in the list becomes the "
                         "canonical ep{N}_composed.mp4 alias for back-compat.")
    args = ap.parse_args()
    args.can_mask_strategies = [s.strip() for s in args.can_mask_strategies.split(",") if s.strip()]
    if not args.can_mask_strategies:
        sys.exit("--can-mask-strategies must list at least one strategy")

    if shutil.which("ffmpeg") is None:
        sys.exit("ffmpeg not on PATH — install it before running the demo")
    if not args.dataset.is_dir():
        sys.exit(f"--dataset {args.dataset} does not exist")
    if not args.top30_json.is_file():
        sys.exit(f"--top30-json {args.top30_json} missing — "
                 f"run tools/build_pins_top30.py first")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw_mp4 = args.out_dir / f"ep{args.episode}_raw.mp4"
    composed_mp4 = args.out_dir / f"ep{args.episode}_composed.mp4"
    first_frame_png = args.out_dir / f"ep{args.episode}_composed_first_frame.png"
    sidecar_json = args.out_dir / f"ep{args.episode}_composed.json"

    rng = random.Random(args.seed)

    # --- 1. Episode metadata + raw video extract ------------------------
    print(f"[1/5] reading episode metadata from {args.dataset}")
    ep = load_episode_meta(args.dataset, args.episode, args.camera_key)
    print(f"      ep{ep['episode_index']}: length={ep['length']} frames  "
          f"t=[{ep['from_timestamp']:.2f}..{ep['to_timestamp']:.2f}] s  "
          f"video chunk{ep['video_chunk']}/file{ep['video_file']}")

    print(f"[2/5] stream-copying episode to {raw_mp4}")
    extract_episode_mp4(args.dataset, ep, args.camera_key, raw_mp4)
    print(f"      wrote {raw_mp4}  ({raw_mp4.stat().st_size / 1024 / 1024:.1f} MB)")

    # --- 2. Pick celebrities --------------------------------------------
    print(f"[3/5] picking {args.n_boards} celebrities (seed={args.seed})")
    picks = pick_celebrities(args.top30_json, rng, n=args.n_boards)
    BOARD_LABELS = ["left", "middle", "right"][:args.n_boards]
    celebrities_by_board: dict[str, dict] = {}
    for label, p in zip(BOARD_LABELS, picks):
        celebrities_by_board[label] = {
            "name": p["celeb"]["name"],
            "rank": p["celeb"].get("rank"),
            "category": p["celeb"].get("category"),
            "n_images": p["celeb"]["n_images"],
            "image_path": p["image_path"],
        }
        print(f"      {label:6s} -> {p['celeb']['name']:25s}  image={Path(p['image_path']).name}")

    # --- 3. Build board + load target images at internal warp resolution -
    board, dictionary = build_board(args.squares_x, args.squares_y, args.square_mm,
                                     args.marker_ratio, args.dict_name)
    board_w_mm = args.squares_x * args.square_mm
    board_h_mm = args.squares_y * args.square_mm
    target_w_px = int(round(board_w_mm * 10))
    target_h_px = int(round(board_h_mm * 10))
    # Load each celebrity JPG; centre-crop to the board's portrait aspect
    # (no letterbox — face must fill the entire board area) and then resize
    # to the internal warp resolution. Apply table-blend distortions
    # (brightness / contrast / saturation / blur / noise) measured from the
    # captured ChArUco frames so the warped face matches the lighting +
    # camera response of the recorded video.
    board_aspect = board_h_mm / board_w_mm
    blend_rng = np.random.default_rng(args.seed)
    targets = []
    for p in picks:
        raw = cv2.imread(p["image_path"], cv2.IMREAD_COLOR)
        if raw is None:
            sys.exit(f"cannot read {p['image_path']}")
        cropped = _center_crop_to_aspect(raw, board_aspect)
        resized = cv2.resize(cropped, (target_w_px, target_h_px),
                             interpolation=cv2.INTER_AREA)
        if args.blend:
            resized = _apply_table_blend(
                resized,
                contrast=args.blend_contrast,
                brightness_offset=args.blend_brightness_offset,
                saturation=args.blend_saturation,
                blur_kernel=args.blend_blur_kernel,
                noise_sigma=args.blend_noise_sigma,
                rng=blend_rng,
            )
        targets.append(resized)
    detector = aruco.ArucoDetector(dictionary, aruco.DetectorParameters())
    char_detector = aruco.CharucoDetector(board)

    # --- 4. Open the raw MP4, lock homographies on frame 0 --------------
    print(f"[4/5] opening {raw_mp4} and locking homographies on frame 0")
    cap = cv2.VideoCapture(str(raw_mp4))
    if not cap.isOpened():
        sys.exit(f"cv2.VideoCapture failed on {raw_mp4}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames_meta = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"      raw video: {width}x{height} @ {fps:.2f} fps  ({n_frames_meta} frames)")

    ok, frame0 = cap.read()
    if not ok:
        sys.exit("could not read frame 0 from raw video")
    Hs, infos = lock_homographies_multi(frame0, detector, char_detector, board, args.n_boards)
    for label, H, info in zip(BOARD_LABELS, Hs, infos):
        status = "OK" if H is not None else "FAILED"
        rms_s = f"rms={info['rms']:.2f}" if info["rms"] is not None else "rms=?"
        print(f"      lock {label:6s}: {status}  markers={info['n_markers']:2d}  "
              f"chess={info['n_chess']:2d}  {rms_s}")
    if any(H is None for H in Hs):
        print("      WARNING: some boards failed to lock; composite will skip those regions.")

    cap.release()

    # --- 5. Per-frame composite + write the new MP4 (one run per strategy) ---
    strategies_to_run = args.can_mask_strategies
    print(f"[5/5] compositing with strategies: {strategies_to_run}")
    n_written, elapsed = 0, 0.0  # last-run stats for the sidecar
    composed_mp4_by_strategy: dict[str, Path] = {}
    first_frame_by_strategy: dict[str, Path] = {}
    for strategy in strategies_to_run:
        if strategy not in _MASKER_REGISTRY:
            sys.exit(f"unknown can-mask strategy {strategy!r}; "
                     f"available: {list(_MASKER_REGISTRY)}")
        out_mp4 = args.out_dir / f"ep{args.episode}_composed_{strategy}.mp4"
        out_png = args.out_dir / f"ep{args.episode}_composed_first_frame_{strategy}.png"
        composed_mp4_by_strategy[strategy] = out_mp4
        first_frame_by_strategy[strategy] = out_png

        masker = _MASKER_REGISTRY[strategy](tuple(args.hsv_lo), tuple(args.hsv_hi))
        cap = cv2.VideoCapture(str(raw_mp4))
        if not cap.isOpened():
            sys.exit(f"cv2.VideoCapture failed on {raw_mp4}")
        ok0, frame0_local = cap.read()
        if not ok0:
            sys.exit("could not read frame 0 from raw video (strategy init)")
        masker.init(frame0_local, np.zeros(frame0_local.shape[:2], dtype=np.uint8))
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        tmp_mp4v = out_mp4.with_suffix(".mp4v.tmp.mp4")
        writer = cv2.VideoWriter(str(tmp_mp4v), fourcc, fps, (width, height))
        if not writer.isOpened():
            sys.exit(f"cv2.VideoWriter failed for {tmp_mp4v}")

        print(f"  -> {strategy}")
        t0 = time.time()
        n_written = 0
        saved_first = False
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            composed = compose_multi(
                frame, targets, Hs, board_w_mm, board_h_mm, args.chroma_mm,
                tuple(args.hsv_lo), tuple(args.hsv_hi),
                masker=masker,
            )
            writer.write(composed)
            if not saved_first:
                cv2.imwrite(str(out_png), composed)
                saved_first = True
            n_written += 1
            if n_written % 200 == 0:
                rate = n_written / max(time.time() - t0, 1e-6)
                print(f"     {n_written}/{n_frames_meta} frames ({rate:.1f} fps)")
        writer.release()
        cap.release()
        elapsed = time.time() - t0
        print(f"     wrote {n_written} frames in {elapsed:.1f}s "
              f"({n_written / elapsed:.1f} fps)")
        # Transcode to H.264 for QuickLook previewability.
        try:
            transcode_to_h264(tmp_mp4v, out_mp4)
        finally:
            tmp_mp4v.unlink(missing_ok=True)
        print(f"     {out_mp4} ({out_mp4.stat().st_size / 1024 / 1024:.1f} MB)")

    # Keep the first listed strategy as the canonical default for back-compat
    # with downstream consumers that look at ep{N}_composed.mp4 / .png.
    canonical = strategies_to_run[0]
    if composed_mp4.exists() or composed_mp4.is_symlink():
        composed_mp4.unlink()
    if first_frame_png.exists() or first_frame_png.is_symlink():
        first_frame_png.unlink()
    shutil.copy2(composed_mp4_by_strategy[canonical], composed_mp4)
    shutil.copy2(first_frame_by_strategy[canonical], first_frame_png)

    # --- 6. Sidecar JSON ------------------------------------------------
    sidecar = {
        "source_dataset": str(args.dataset),
        "source_episode": ep,
        "seed": args.seed,
        "n_boards": args.n_boards,
        "celebrities_by_board": celebrities_by_board,
        "board_geometry": {
            "squares_x": args.squares_x,
            "squares_y": args.squares_y,
            "square_mm": args.square_mm,
            "marker_ratio": args.marker_ratio,
            "chroma_mm": args.chroma_mm,
            "dict": args.dict_name,
        },
        "hsv_green_range": {"lo": list(args.hsv_lo), "hi": list(args.hsv_hi)},
        "homography_locks": [
            {"label": label,
             "ok": H is not None,
             "n_markers": info["n_markers"],
             "n_chess": info["n_chess"],
             "rms": info["rms"]}
            for label, H, info in zip(BOARD_LABELS, Hs, infos)
        ],
        "n_frames_written": n_written,
        "raw_mp4": str(raw_mp4),
        "composed_mp4": str(composed_mp4),
        "first_frame_png": str(first_frame_png),
        "processing_time_s": round(elapsed, 2),
        "can_mask_strategies": args.can_mask_strategies,
        "composed_mp4_by_strategy": {k: str(v) for k, v in composed_mp4_by_strategy.items()},
        "first_frame_by_strategy": {k: str(v) for k, v in first_frame_by_strategy.items()},
    }
    sidecar_json.write_text(json.dumps(sidecar, indent=2, ensure_ascii=False) + "\n",
                            encoding="utf-8")
    print(f"\nwrote sidecar -> {sidecar_json}")
    print(f"\ndone. quick view:")
    print(f"  open {first_frame_png}")
    print(f"  open {composed_mp4}")


if __name__ == "__main__":
    main()
