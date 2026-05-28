#!/usr/bin/env python3
"""Materialize `dataset_v5_charuko_<slot>_full` by cross-producting approach × placement episodes.

Source corpus (already pulled to ./datasets/):
  * dataset_v5_charuko_approach        — 5 episodes, "pick up the coke" trajectories
  * dataset_v5_charuko_{left,middle,right}_1 — 2 episodes each, "place coke on slot" trajectories

Per slot, this script materializes a full LeRobot dataset where every episode is
the concatenation of one approach trajectory + a short linear-interpolated bridge
+ one placement trajectory. All-pairs cross product → 5 × 2 = 10 synthetic
episodes per slot.

Output tree (one per slot):
    datasets/dataset_v5_charuko_<slot>_full/
        README.md
        meta/info.json
        meta/tasks.parquet
        meta/stats.json
        meta/episodes/chunk-000/file-000.parquet
        data/chunk-000/file-000.parquet
        videos/observation.images.front/chunk-000/file-000.mp4

The task string ``"Place the coke on <placeholder>"`` is preserved unchanged —
substitution to a celebrity name is a separate train-time concern.

Usage:
    # Smoke: report what would be built, no writes
    python tools/eval3_v5_concat_pairs.py --slots left --dry-run

    # Build one slot, materialize to ./datasets/<...>_full/
    python tools/eval3_v5_concat_pairs.py --slots left --bridge-frames 20

    # Build all three and push to Hub with v3.0 tag
    python tools/eval3_v5_concat_pairs.py --slots left,middle,right \
        --push-to-hub --tag v3.0
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("eval3_v5_concat_pairs")

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASETS_ROOT = REPO_ROOT / "datasets"
TASK_STRING = "Place the coke on <placeholder>"
FPS = 30
IMAGE_KEY = "observation.images.front"
CODEBASE_VERSION = "v3.0"


# --------------------------------------------------------------------------- #
# Source loaders                                                              #
# --------------------------------------------------------------------------- #

class SourceDataset:
    """Lightweight wrapper around a materialized LeRobot dataset on disk."""

    def __init__(self, name: str):
        self.name = name
        self.root = DATASETS_ROOT / name
        if not (self.root / "meta" / "info.json").is_file():
            raise FileNotFoundError(
                f"source dataset not pulled: {self.root}. "
                f"Run `python -c \"from huggingface_hub import snapshot_download; "
                f"snapshot_download('RobotLearningVLA/{name}', repo_type='dataset', "
                f"local_dir='{self.root}')\"`"
            )
        self.info = json.loads((self.root / "meta" / "info.json").read_text())
        self.data_path = self.root / "data" / "chunk-000" / "file-000.parquet"
        self.episodes_path = self.root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
        self.video_path = self.root / "videos" / IMAGE_KEY / "chunk-000" / "file-000.mp4"
        self.stats_path = self.root / "meta" / "stats.json"
        self.tasks_path = self.root / "meta" / "tasks.parquet"

        self._data_table = pq.read_table(self.data_path)
        self._data_df = self._data_table.to_pandas()
        self._episodes_table = pq.read_table(self.episodes_path)
        self._episodes_df = self._episodes_table.to_pandas()

    @property
    def num_episodes(self) -> int:
        return int(self.info["total_episodes"])

    @property
    def num_frames(self) -> int:
        return int(self.info["total_frames"])

    def episode_slice(self, ep_idx: int) -> pd.DataFrame:
        row = self._episodes_df[self._episodes_df["episode_index"] == ep_idx].iloc[0]
        f0, f1 = int(row["dataset_from_index"]), int(row["dataset_to_index"])
        return self._data_df.iloc[f0:f1].reset_index(drop=True).copy()

    def episode_video_window(self, ep_idx: int) -> tuple[float, float]:
        row = self._episodes_df[self._episodes_df["episode_index"] == ep_idx].iloc[0]
        return (
            float(row["videos/observation.images.front/from_timestamp"]),
            float(row["videos/observation.images.front/to_timestamp"]),
        )

    def episode_image_stats(self, ep_idx: int) -> dict[str, list]:
        """Return the (per-episode) image stats columns to copy into the new dataset."""
        row = self._episodes_df[self._episodes_df["episode_index"] == ep_idx].iloc[0]
        out: dict[str, list] = {}
        for stat in ("min", "max", "mean", "std", "count", "q01", "q10", "q50", "q90", "q99"):
            col = f"stats/{IMAGE_KEY}/{stat}"
            v = row[col]
            # pandas returns numpy arrays for list cells; convert to plain python lists
            out[stat] = v.tolist() if isinstance(v, np.ndarray) else v
        return out

    def global_image_stats(self) -> dict:
        stats = json.loads(self.stats_path.read_text())
        return stats[IMAGE_KEY]


# --------------------------------------------------------------------------- #
# Bridge construction                                                         #
# --------------------------------------------------------------------------- #

def build_bridge_frames(
    approach_last: pd.Series,
    place_first: pd.Series,
    n_frames: int,
    mode: str,
) -> list[dict]:
    """Return ``n_frames`` interpolated rows between two boundary frames.

    Only ``action`` and ``observation.state`` are interpolated (everything else
    is rewritten downstream when episode-local indices are assigned).
    """
    if n_frames <= 0:
        return []
    a_act = np.asarray(approach_last["action"], dtype=np.float32)
    p_act = np.asarray(place_first["action"], dtype=np.float32)
    a_st = np.asarray(approach_last["observation.state"], dtype=np.float32)
    p_st = np.asarray(place_first["observation.state"], dtype=np.float32)

    rows: list[dict] = []
    for i in range(n_frames):
        if mode == "linear":
            # i+1 / N+1 so we never repeat either endpoint (clean transition)
            t = (i + 1) / (n_frames + 1)
        elif mode == "hold":
            t = 0.0
        else:
            raise ValueError(f"unknown bridge mode: {mode}")
        action = (1.0 - t) * a_act + t * p_act
        state = (1.0 - t) * a_st + t * p_st
        rows.append({
            "action": action.astype(np.float32).tolist(),
            "observation.state": state.astype(np.float32).tolist(),
            "timestamp": 0.0,  # rewritten below
            "frame_index": 0,
            "episode_index": 0,
            "index": 0,
            "task_index": 0,
        })
    return rows


def build_synthetic_episode(
    approach_df: pd.DataFrame,
    place_df: pd.DataFrame,
    bridge_frames: int,
    bridge_mode: str,
) -> pd.DataFrame:
    """Glue [approach | bridge | place] into one synthetic episode DataFrame.

    Frame metadata (frame_index/episode_index/index/timestamp/task_index) is
    NOT set here — callers re-number after concat.
    """
    cols = ["action", "observation.state", "timestamp", "frame_index",
            "episode_index", "index", "task_index"]

    a_rows = approach_df[cols].copy()
    # Lists came through as numpy arrays; convert to python lists so
    # downstream pyarrow construction sees a uniform type.
    a_rows["action"] = a_rows["action"].apply(lambda v: np.asarray(v, dtype=np.float32).tolist())
    a_rows["observation.state"] = a_rows["observation.state"].apply(
        lambda v: np.asarray(v, dtype=np.float32).tolist()
    )
    p_rows = place_df[cols].copy()
    p_rows["action"] = p_rows["action"].apply(lambda v: np.asarray(v, dtype=np.float32).tolist())
    p_rows["observation.state"] = p_rows["observation.state"].apply(
        lambda v: np.asarray(v, dtype=np.float32).tolist()
    )

    bridge = build_bridge_frames(
        approach_df.iloc[-1],
        place_df.iloc[0],
        bridge_frames,
        bridge_mode,
    )
    bridge_df = pd.DataFrame(bridge, columns=cols)

    ep_df = pd.concat([a_rows, bridge_df, p_rows], ignore_index=True)
    return ep_df


def seam_distance(approach_last: pd.Series, place_first: pd.Series) -> float:
    """Joint-Euclidean distance between two boundary frames, raw degrees."""
    a = np.asarray(approach_last["observation.state"], dtype=np.float32)
    p = np.asarray(place_first["observation.state"], dtype=np.float32)
    return float(np.linalg.norm(a - p))


# --------------------------------------------------------------------------- #
# Stats helpers                                                               #
# --------------------------------------------------------------------------- #

def per_episode_numeric_stats(values: np.ndarray, count_dtype: str = "int64") -> dict:
    """Compute the LeRobot per-episode stat dict for a 1- or 2-D numeric column.

    Returns python primitives / lists matching the source episodes parquet's
    column shapes (`stats/<feat>/<stat>` is a `list<...>` even for scalars).
    """
    x = np.asarray(values, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    out: dict[str, list] = {}
    out["min"] = x.min(axis=0).tolist()
    out["max"] = x.max(axis=0).tolist()
    out["mean"] = x.mean(axis=0).tolist()
    out["std"] = x.std(axis=0).tolist()
    out["count"] = [int(x.shape[0])]
    for q, name in [(0.01, "q01"), (0.10, "q10"), (0.50, "q50"),
                    (0.90, "q90"), (0.99, "q99")]:
        out[name] = np.quantile(x, q, axis=0).tolist()
    return out


def global_numeric_stats(values: np.ndarray) -> dict:
    """Compute global stats.json shape (lists of per-dim arrays)."""
    return per_episode_numeric_stats(values)


# --------------------------------------------------------------------------- #
# Video stitching                                                             #
# --------------------------------------------------------------------------- #

def run_ffmpeg(args: list[str], log_label: str = "ffmpeg") -> None:
    proc = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        log.error("%s failed: %s", log_label, " ".join(args))
        log.error("stderr: %s", proc.stderr.decode("utf-8", "replace")[-2000:])
        raise RuntimeError(f"{log_label} failed (returncode={proc.returncode})")


def extract_clip(src_mp4: Path, t_from: float, t_to: float, out_mp4: Path) -> None:
    """Extract a clip from src_mp4 between two timestamps, re-encoded to a clean h264."""
    # Place -ss after -i for accurate (frame-precise) seek even though slower.
    duration = max(0.001, t_to - t_from)
    run_ffmpeg([
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(src_mp4),
        "-ss", f"{t_from:.6f}",
        "-t", f"{duration:.6f}",
        "-an", "-vsync", "cfr", "-r", str(FPS),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-preset", "fast", "-crf", "18",
        str(out_mp4),
    ], log_label=f"extract_clip {out_mp4.name}")


def extract_last_frame_png(src_mp4: Path, n_frames_total: int, out_png: Path) -> None:
    """Save the (n_frames_total-1)th frame of src_mp4 as a PNG."""
    run_ffmpeg([
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(src_mp4),
        "-vf", f"select=eq(n\\,{max(0, n_frames_total - 1)})",
        "-vframes", "1",
        str(out_png),
    ], log_label=f"extract_last_frame {out_png.name}")


def extract_first_frame_png(src_mp4: Path, out_png: Path) -> None:
    """Save the first frame of src_mp4 as a PNG."""
    run_ffmpeg([
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(src_mp4),
        "-vf", "select=eq(n\\,0)",
        "-vframes", "1",
        str(out_png),
    ], log_label=f"extract_first_frame {out_png.name}")


def build_bridge_freeze_mp4(start_png: Path, end_png: Path, n_frames: int, out_mp4: Path) -> None:
    """Frozen-frame bridge: hold start_png for n_frames at FPS. end_png is ignored.

    Kept for backward compatibility with the original freeze-on-approach-last behaviour.
    """
    duration = n_frames / FPS
    run_ffmpeg([
        "ffmpeg", "-y", "-loglevel", "error",
        "-loop", "1", "-framerate", str(FPS),
        "-i", str(start_png),
        "-t", f"{duration:.6f}",
        "-an", "-r", str(FPS),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-preset", "fast", "-crf", "18",
        str(out_mp4),
    ], log_label=f"build_freeze {out_mp4.name}")


def build_bridge_blend_mp4(start_png: Path, end_png: Path, n_frames: int, out_mp4: Path) -> None:
    """Cross-fade bridge: linear alpha-blend from start_png to end_png over n_frames at FPS.

    Produces a smooth "double exposure" transition. Always works; no motion estimation.
    Trade-off: things in motion appear ghosted because both endpoints are visible at once.
    """
    duration = n_frames / FPS
    # The blend filter exposes T (time in seconds). At T=0 alpha=0 (all A=start),
    # at T=duration alpha=1 (all B=end). Frame i (i=0..N-1) lands at T=i/FPS.
    expr = f"A*(1-T/{duration:.6f})+B*(T/{duration:.6f})"
    run_ffmpeg([
        "ffmpeg", "-y", "-loglevel", "error",
        "-loop", "1", "-framerate", str(FPS), "-i", str(start_png),
        "-loop", "1", "-framerate", str(FPS), "-i", str(end_png),
        "-filter_complex", f"[0:v][1:v]blend=all_expr='{expr}'[v]",
        "-map", "[v]", "-t", f"{duration:.6f}", "-an", "-r", str(FPS),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-preset", "fast", "-crf", "18",
        str(out_mp4),
    ], log_label=f"build_blend {out_mp4.name}")


def build_bridge_mci_mp4(start_png: Path, end_png: Path, n_frames: int, out_mp4: Path) -> None:
    """Motion-compensated bridge using ffmpeg `minterpolate=mci`.

    Pipeline (worked out empirically — minterpolate refuses to emit frames on a
    bare 2-frame input):

      1. Pad the source: build a 10-frame video at FPS where the first 5 frames
         are start_png and the last 5 are end_png. The single visual change is
         between frames 4 and 5, but minterpolate sees enough temporal context
         on either side to estimate motion vectors.
      2. minterpolate to fps = FPS * (n_frames + 1) → output has roughly
         10*(n_frames+1) frames; the n_frames interpolated transition frames
         live at output indices [4*(n_frames+1) + 1 .. 5*(n_frames+1) - 1].
      3. Extract those output frames using `select=between(...)` and re-stamp
         them as a 30 FPS stream with `setpts=N/30/TB` — using `fps=30` would
         re-sample-drop frames inside a tiny 1/30 s window.

    Visual quality: warps pixels along estimated motion vectors. Looks much
    better than blend for small/local motion (gripper assembly shift), but the
    camera viewpoint can't actually move so the interpolation is approximate.
    """
    n_pad = 5  # copies of each endpoint on either side of the seam
    upsample = n_frames + 1
    interp_fps = FPS * upsample
    sel_first = n_pad * upsample - upsample + 1   # = (n_pad-1)*upsample + 1 = first interpolated frame
    sel_last = n_pad * upsample - 1               # last interpolated frame before end_png copy starts
    # sel_first..sel_last inclusive = exactly n_frames frames
    assert sel_last - sel_first + 1 == n_frames

    with tempfile.TemporaryDirectory(prefix="v5_mci_") as tmpd:
        tmp = Path(tmpd)
        # Stage 1: build padded 10-frame source.
        # We copy the two PNGs into a sequential naming so the image2 demuxer picks them up.
        for i in range(n_pad):
            shutil.copy(start_png, tmp / f"pad_{i:03d}.png")
        for i in range(n_pad, 2 * n_pad):
            shutil.copy(end_png, tmp / f"pad_{i:03d}.png")
        src_mp4 = tmp / "src.mp4"
        run_ffmpeg([
            "ffmpeg", "-y", "-loglevel", "error",
            "-framerate", str(FPS),
            "-i", str(tmp / "pad_%03d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            str(src_mp4),
        ], log_label="build_mci src")

        # Stage 2: minterpolate to interp_fps.
        interp_mp4 = tmp / "interp.mp4"
        run_ffmpeg([
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(src_mp4),
            "-vf", f"minterpolate=fps={interp_fps}:mi_mode=mci:mc_mode=aobmc:vsbmc=1",
            "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
            str(interp_mp4),
        ], log_label="build_mci minterp")

        # Stage 3: extract the n_frames interpolated frames and re-stamp at FPS.
        run_ffmpeg([
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(interp_mp4),
            "-vf", f"select='between(n,{sel_first},{sel_last})',setpts=N/{FPS}/TB",
            "-an", "-r", str(FPS),
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-preset", "fast", "-crf", "18",
            str(out_mp4),
        ], log_label=f"build_mci {out_mp4.name}")


BRIDGE_VIDEO_BUILDERS = {
    "freeze": build_bridge_freeze_mp4,
    "blend": build_bridge_blend_mp4,
    "mci": build_bridge_mci_mp4,
}


def concat_mp4s(clip_paths: list[Path], out_mp4: Path) -> None:
    """ffmpeg concat demuxer across a list of MP4 paths, re-encoded to clean 30fps PTS.

    Stream copy was the original strategy but it caused cumulative timing drift:
    the MCI bridge re-encoding produces clips that come out ~0.65 ms short of
    20/30 s, so after 30 concatenated segments the PTS grid lags ~20 ms behind
    where LeRobotDataset's frame_index → timestamp lookup expects it. Result:
    `FrameTimestampError: queried 85.3000, loaded 85.2980 (diff 0.0020 > tol 0.0001)`.

    Re-encoding with `-fps_mode cfr -r 30` regenerates PTS on an exact 1/30 s
    grid regardless of any input drift. Slower than stream copy but the only
    way to guarantee LeRobot can index the frames.
    """
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        list_path = Path(f.name)
        for p in clip_paths:
            f.write(f"file '{p.resolve()}'\n")
    try:
        run_ffmpeg([
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "concat", "-safe", "0", "-i", str(list_path),
            "-fflags", "+genpts",
            "-vsync", "cfr", "-r", str(FPS),
            "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-preset", "fast", "-crf", "18",
            str(out_mp4),
        ], log_label=f"concat {out_mp4.name}")
    finally:
        list_path.unlink(missing_ok=True)


def stitch_slot_videos(
    slot: str,
    approach_src: SourceDataset,
    place_src: SourceDataset,
    pair_list: list[tuple[int, int]],
    bridge_frames: int,
    bridge_video_mode: str,
    out_mp4: Path,
    workdir: Path,
) -> None:
    """Build one stitched MP4 covering all pairs for a slot.

    Bridge construction depends on ``bridge_video_mode`` (see BRIDGE_VIDEO_BUILDERS):

    - ``freeze``: bridge is a frozen approach-last frame. Cache the freeze MP4
      per approach episode (one clip reused across both placements).
    - ``blend`` / ``mci``: bridge depends on BOTH endpoints (approach-last +
      placement-first), so we build one clip per (a_ep, p_ep) pair.
    """
    if bridge_video_mode not in BRIDGE_VIDEO_BUILDERS:
        raise ValueError(
            f"bridge_video_mode={bridge_video_mode!r}; expected one of {sorted(BRIDGE_VIDEO_BUILDERS)}"
        )
    bridge_builder = BRIDGE_VIDEO_BUILDERS[bridge_video_mode]
    workdir.mkdir(parents=True, exist_ok=True)

    # 1. Extract source approach clips (one per unique approach_ep)
    approach_clips: dict[int, Path] = {}
    approach_lengths: dict[int, int] = {}
    for a_ep in sorted({a for a, _ in pair_list}):
        t0, t1 = approach_src.episode_video_window(a_ep)
        clip = workdir / f"approach_{a_ep}.mp4"
        extract_clip(approach_src.video_path, t0, t1, clip)
        approach_clips[a_ep] = clip
        approach_lengths[a_ep] = int(round((t1 - t0) * FPS))

    # 2. Extract source placement clips
    place_clips: dict[int, Path] = {}
    for p_ep in sorted({p for _, p in pair_list}):
        t0, t1 = place_src.episode_video_window(p_ep)
        clip = workdir / f"place_{p_ep}.mp4"
        extract_clip(place_src.video_path, t0, t1, clip)
        place_clips[p_ep] = clip

    # 3. Build seam PNGs: last frame of each approach clip + first frame of each placement clip
    approach_last_pngs: dict[int, Path] = {}
    for a_ep, clip in approach_clips.items():
        png = workdir / f"approach_last_a{a_ep}.png"
        extract_last_frame_png(clip, approach_lengths[a_ep], png)
        approach_last_pngs[a_ep] = png
    place_first_pngs: dict[int, Path] = {}
    for p_ep, clip in place_clips.items():
        png = workdir / f"place_first_p{p_ep}.png"
        extract_first_frame_png(clip, png)
        place_first_pngs[p_ep] = png

    # 4. Build bridge clips. Freeze can dedup by approach_ep; blend/mci depend
    # on both endpoints so we build one per pair (10 clips per slot).
    bridge_clips: dict[tuple[int, int], Path] = {}
    if bridge_video_mode == "freeze":
        # Dedup by approach_ep — end_png is ignored by the freeze builder.
        per_approach: dict[int, Path] = {}
        for a_ep, png in approach_last_pngs.items():
            clip = workdir / f"bridge_freeze_a{a_ep}.mp4"
            bridge_builder(png, png, bridge_frames, clip)
            per_approach[a_ep] = clip
        for a_ep, p_ep in pair_list:
            bridge_clips[(a_ep, p_ep)] = per_approach[a_ep]
    else:
        for a_ep, p_ep in pair_list:
            start_png = approach_last_pngs[a_ep]
            end_png = place_first_pngs[p_ep]
            clip = workdir / f"bridge_{bridge_video_mode}_a{a_ep}_p{p_ep}.mp4"
            bridge_builder(start_png, end_png, bridge_frames, clip)
            bridge_clips[(a_ep, p_ep)] = clip

    # 5. Assemble final concat list in pair order
    concat_inputs: list[Path] = []
    for a_ep, p_ep in pair_list:
        concat_inputs.append(approach_clips[a_ep])
        concat_inputs.append(bridge_clips[(a_ep, p_ep)])
        concat_inputs.append(place_clips[p_ep])
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    concat_mp4s(concat_inputs, out_mp4)


# --------------------------------------------------------------------------- #
# Per-slot materialization                                                    #
# --------------------------------------------------------------------------- #

def build_pair_episode_records(
    approach_src: SourceDataset,
    place_src: SourceDataset,
    bridge_frames: int,
    bridge_mode: str,
) -> list[dict]:
    """Build per-episode records (frames + video stats source pointers).

    Returns list of dicts, one per synthetic episode, in (a_ep, p_ep) cross-product
    order (a_ep varies fastest).
    """
    records: list[dict] = []
    ep_idx = 0
    for a_ep in range(approach_src.num_episodes):
        for p_ep in range(place_src.num_episodes):
            a_df = approach_src.episode_slice(a_ep)
            p_df = place_src.episode_slice(p_ep)
            ep_df = build_synthetic_episode(a_df, p_df, bridge_frames, bridge_mode)
            seam_d = seam_distance(a_df.iloc[-1], p_df.iloc[0])
            records.append({
                "ep_idx": ep_idx,
                "a_ep": a_ep,
                "p_ep": p_ep,
                "ep_df": ep_df,
                "length": int(len(ep_df)),
                "seam_distance_deg": seam_d,
                # We use approach's per-episode image stats as the per-synth-ep
                # image stats. Approach contributes the majority of frames in
                # most pairs and the bridge is a freeze on the approach-last
                # frame, so this is the most accurate single-source pick.
                "image_stats": approach_src.episode_image_stats(a_ep),
            })
            ep_idx += 1
    return records


def assign_frame_metadata(records: list[dict]) -> tuple[pd.DataFrame, list[dict]]:
    """Rewrite frame_index / episode_index / index / timestamp / task_index across all records.

    Returns (concatenated data DataFrame, per-episode metadata dicts with
    dataset_from_index / dataset_to_index / from_timestamp / to_timestamp).
    """
    global_idx = 0
    cumulative_frames = 0
    all_frames: list[pd.DataFrame] = []
    ep_meta: list[dict] = []
    for rec in records:
        df = rec["ep_df"].copy()
        n = len(df)
        df["frame_index"] = np.arange(n, dtype=np.int64)
        df["episode_index"] = np.int64(rec["ep_idx"])
        df["index"] = np.arange(global_idx, global_idx + n, dtype=np.int64)
        df["timestamp"] = (np.arange(n, dtype=np.float64) / FPS).astype(np.float32)
        df["task_index"] = np.int64(0)
        all_frames.append(df)

        ep_meta.append({
            "episode_index": int(rec["ep_idx"]),
            "length": n,
            "dataset_from_index": global_idx,
            "dataset_to_index": global_idx + n,
            "video_from_timestamp": cumulative_frames / FPS,
            "video_to_timestamp": (cumulative_frames + n) / FPS,
            "image_stats": rec["image_stats"],
            "seam_distance_deg": rec["seam_distance_deg"],
            "a_ep": rec["a_ep"],
            "p_ep": rec["p_ep"],
        })
        global_idx += n
        cumulative_frames += n

    return pd.concat(all_frames, ignore_index=True), ep_meta


# --------------------------------------------------------------------------- #
# Parquet writers                                                             #
# --------------------------------------------------------------------------- #

def write_data_parquet(df: pd.DataFrame, out_dir: Path) -> None:
    """Write data/chunk-000/file-000.parquet matching source dtypes exactly."""
    schema = pa.schema([
        ("action", pa.list_(pa.float32(), 6)),
        ("observation.state", pa.list_(pa.float32(), 6)),
        ("timestamp", pa.float32()),
        ("frame_index", pa.int64()),
        ("episode_index", pa.int64()),
        ("index", pa.int64()),
        ("task_index", pa.int64()),
    ])
    table = pa.table({
        "action": df["action"].tolist(),
        "observation.state": df["observation.state"].tolist(),
        "timestamp": df["timestamp"].astype(np.float32).to_numpy(),
        "frame_index": df["frame_index"].astype(np.int64).to_numpy(),
        "episode_index": df["episode_index"].astype(np.int64).to_numpy(),
        "index": df["index"].astype(np.int64).to_numpy(),
        "task_index": df["task_index"].astype(np.int64).to_numpy(),
    }, schema=schema)
    out_path = out_dir / "data" / "chunk-000" / "file-000.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out_path)


def write_tasks_parquet(out_dir: Path) -> None:
    """Single-task tasks.parquet.

    Source `tasks.parquet` carries pandas index metadata marking `task` as the
    pandas index (so `row['task']` resolves via the join at load time). We
    replicate that — writing via `pd.to_parquet` with `task` as the index is the
    least-surprising way to embed the right `index_columns` metadata.
    """
    out_path = out_dir / "meta" / "tasks.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        {"task_index": [0]},
        index=pd.Index([TASK_STRING], name="task"),
    )
    df.to_parquet(out_path)


def write_episodes_parquet(data_df: pd.DataFrame, ep_meta: list[dict], out_dir: Path) -> None:
    """Write episodes parquet with the full 93-field schema."""
    # Build per-episode action/state/timestamp/frame/episode/index/task_index stats from synthetic data.
    rows: list[dict] = []
    for em in ep_meta:
        f0, f1 = em["dataset_from_index"], em["dataset_to_index"]
        sub = data_df.iloc[f0:f1]
        actions = np.stack([np.asarray(a, dtype=np.float32) for a in sub["action"].tolist()])
        states = np.stack([np.asarray(s, dtype=np.float32) for s in sub["observation.state"].tolist()])
        timestamps = sub["timestamp"].to_numpy(dtype=np.float64)
        frame_indices = sub["frame_index"].to_numpy(dtype=np.int64)
        episode_indices = sub["episode_index"].to_numpy(dtype=np.int64)
        indices = sub["index"].to_numpy(dtype=np.int64)
        task_indices = sub["task_index"].to_numpy(dtype=np.int64)

        s_action = per_episode_numeric_stats(actions)
        s_state = per_episode_numeric_stats(states)
        s_ts = per_episode_numeric_stats(timestamps)
        s_fi = per_episode_numeric_stats(frame_indices)
        s_ei = per_episode_numeric_stats(episode_indices)
        s_idx = per_episode_numeric_stats(indices)
        s_ti = per_episode_numeric_stats(task_indices)
        s_img = em["image_stats"]

        row = {
            "episode_index": em["episode_index"],
            "tasks": [TASK_STRING],
            "length": em["length"],
            "data/chunk_index": 0,
            "data/file_index": 0,
            "dataset_from_index": em["dataset_from_index"],
            "dataset_to_index": em["dataset_to_index"],
            "videos/observation.images.front/chunk_index": 0,
            "videos/observation.images.front/file_index": 0,
            "videos/observation.images.front/from_timestamp": em["video_from_timestamp"],
            "videos/observation.images.front/to_timestamp": em["video_to_timestamp"],
            "meta/episodes/chunk_index": 0,
            "meta/episodes/file_index": 0,
        }

        for feat_key, stat_dict, count_as_int in [
            ("action", s_action, False),
            ("observation.state", s_state, False),
            ("timestamp", s_ts, False),
            ("frame_index", s_fi, True),
            ("episode_index", s_ei, True),
            ("index", s_idx, True),
            ("task_index", s_ti, True),
        ]:
            # min/max for int columns must be int64-typed lists (per schema).
            for stat in ("min", "max", "mean", "std", "count", "q01", "q10", "q50", "q90", "q99"):
                row[f"stats/{feat_key}/{stat}"] = stat_dict[stat]

        # Image stats — copied from approach source for this episode.
        for stat, val in s_img.items():
            row[f"stats/{IMAGE_KEY}/{stat}"] = val

        rows.append(row)

    # Build pyarrow table with the exact 93-field schema.
    # Easier: pull source schema and rebuild a table matching it.
    arrays: dict[str, list] = {col: [r[col] for r in rows] for col in rows[0].keys()}

    # Cast integer-stat columns to int64 (min/max/count). pyarrow will infer
    # double for the rest because they are python floats.
    int_stat_features = ("frame_index", "episode_index", "index", "task_index")
    schema_fields: list[pa.Field] = [
        ("episode_index", pa.int64()),
        ("tasks", pa.list_(pa.string())),
        ("length", pa.int64()),
        ("data/chunk_index", pa.int64()),
        ("data/file_index", pa.int64()),
        ("dataset_from_index", pa.int64()),
        ("dataset_to_index", pa.int64()),
        ("videos/observation.images.front/chunk_index", pa.int64()),
        ("videos/observation.images.front/file_index", pa.int64()),
        ("videos/observation.images.front/from_timestamp", pa.float64()),
        ("videos/observation.images.front/to_timestamp", pa.float64()),
    ]
    for feat in ("action", "observation.state", "timestamp", "frame_index",
                 "episode_index", "index", "task_index"):
        is_int = feat in int_stat_features
        # min/max/count are int for int columns; everything else float64.
        for stat in ("min", "max", "mean", "std", "count", "q01", "q10", "q50", "q90", "q99"):
            elem_type = pa.int64() if (is_int and stat in {"min", "max", "count"}) or (stat == "count") else pa.float64()
            schema_fields.append((f"stats/{feat}/{stat}", pa.list_(elem_type)))
    # Image stats: nested list<list<list<float64>>> for min/max/mean/std/q*; count is list<int64>.
    for stat in ("min", "max", "mean", "std", "count", "q01", "q10", "q50", "q90", "q99"):
        if stat == "count":
            schema_fields.append((f"stats/{IMAGE_KEY}/{stat}", pa.list_(pa.int64())))
        else:
            schema_fields.append((
                f"stats/{IMAGE_KEY}/{stat}",
                pa.list_(pa.list_(pa.list_(pa.float64()))),
            ))
    schema_fields.append(("meta/episodes/chunk_index", pa.int64()))
    schema_fields.append(("meta/episodes/file_index", pa.int64()))

    schema = pa.schema(schema_fields)
    # Coerce int-cell lists to int (pyarrow accepts python ints) and float-cell
    # lists already contain python floats.
    table = pa.table(arrays, schema=schema)

    out_path = out_dir / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out_path)


def write_stats_json(
    data_df: pd.DataFrame,
    approach_src: SourceDataset,
    place_src: SourceDataset,
    out_dir: Path,
) -> None:
    """Build global stats.json. Image stats come from source approach (dominant in frame mix)."""
    actions = np.stack([np.asarray(a, dtype=np.float32) for a in data_df["action"].tolist()])
    states = np.stack([np.asarray(s, dtype=np.float32) for s in data_df["observation.state"].tolist()])
    timestamps = data_df["timestamp"].to_numpy(dtype=np.float64)
    frame_indices = data_df["frame_index"].to_numpy(dtype=np.int64)
    episode_indices = data_df["episode_index"].to_numpy(dtype=np.int64)
    indices = data_df["index"].to_numpy(dtype=np.int64)
    task_indices = data_df["task_index"].to_numpy(dtype=np.int64)

    stats = {
        "task_index": global_numeric_stats(task_indices),
        "frame_index": global_numeric_stats(frame_indices),
        "index": global_numeric_stats(indices),
        "observation.state": global_numeric_stats(states),
        "action": global_numeric_stats(actions),
        IMAGE_KEY: approach_src.global_image_stats(),
        "episode_index": global_numeric_stats(episode_indices),
        "timestamp": global_numeric_stats(timestamps),
    }

    out_path = out_dir / "meta" / "stats.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(stats, indent=4))


def write_info_json(out_dir: Path, total_episodes: int, total_frames: int, approach_src: SourceDataset) -> None:
    info = deepcopy(approach_src.info)
    info["total_episodes"] = int(total_episodes)
    info["total_frames"] = int(total_frames)
    info["total_tasks"] = 1
    info["splits"] = {"train": f"0:{total_episodes}"}
    (out_dir / "meta").mkdir(parents=True, exist_ok=True)
    (out_dir / "meta" / "info.json").write_text(json.dumps(info, indent=4))


def write_readme(out_dir: Path, slot: str, total_episodes: int, total_frames: int) -> None:
    body = (
        "---\n"
        "license: apache-2.0\n"
        "task_categories:\n"
        "- robotics\n"
        "tags:\n"
        "- LeRobot\n"
        "configs:\n"
        "- config_name: default\n"
        "  data_files: data/*/*.parquet\n"
        "---\n\n"
        f"# dataset_v5_charuko_{slot}_full\n\n"
        f"Synthetic LeRobot dataset built by concatenating each "
        f"`dataset_v5_charuko_approach` episode with each "
        f"`dataset_v5_charuko_{slot}_1` episode (all-pairs cross product), with a "
        f"short linear-interpolated state+action bridge between them.\n\n"
        f"- **Episodes:** {total_episodes} (5 approaches × 2 placements)\n"
        f"- **Total frames:** {total_frames}\n"
        f"- **FPS:** {FPS}\n"
        f"- **Task string:** `{TASK_STRING}` — sentinel meant to be substituted at training time.\n\n"
        f"Generated by `tools/eval3_v5_concat_pairs.py` in the RobotLearningVLA "
        f"training repo.\n"
    )
    (out_dir / "README.md").write_text(body)


# --------------------------------------------------------------------------- #
# Slot orchestration                                                          #
# --------------------------------------------------------------------------- #

def report_pairs(slot: str, approach_src: SourceDataset, place_src: SourceDataset, bridge_frames: int) -> None:
    print(f"\n=== DRY RUN: slot={slot} ===")
    total = 0
    pairs = [(a, p) for a in range(approach_src.num_episodes) for p in range(place_src.num_episodes)]
    print(f"  pairs: {len(pairs)} (approach {approach_src.num_episodes} × place {place_src.num_episodes})")
    for a, p in pairs:
        a_df = approach_src.episode_slice(a)
        p_df = place_src.episode_slice(p)
        n = len(a_df) + bridge_frames + len(p_df)
        d = seam_distance(a_df.iloc[-1], p_df.iloc[0])
        print(f"    ep={a*place_src.num_episodes+p:2d}  a_ep={a} (n={len(a_df):3d}) + bridge={bridge_frames} + p_ep={p} (n={len(p_df):3d}) -> n={n:4d}  seam={d:5.1f}°")
        total += n
    print(f"  total synthetic frames: {total} ({total/FPS:.1f}s)")


def materialize_slot(
    slot: str,
    approach_src: SourceDataset,
    place_src: SourceDataset,
    out_dir: Path,
    bridge_frames: int,
    bridge_mode: str,
    bridge_video_mode: str,
    overwrite: bool,
) -> dict:
    if out_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"{out_dir} exists. Re-run with --overwrite to replace."
            )
        log.info("removing existing %s", out_dir)
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    log.info("== slot=%s: building per-pair records ==", slot)
    records = build_pair_episode_records(approach_src, place_src, bridge_frames, bridge_mode)
    data_df, ep_meta = assign_frame_metadata(records)

    pair_list = [(em["a_ep"], em["p_ep"]) for em in ep_meta]
    log.info("== slot=%s: %d episodes, %d frames; writing data parquet ==", slot, len(records), len(data_df))
    write_data_parquet(data_df, out_dir)

    log.info("== slot=%s: writing tasks parquet ==", slot)
    write_tasks_parquet(out_dir)

    log.info("== slot=%s: writing episodes parquet (per-ep stats) ==", slot)
    write_episodes_parquet(data_df, ep_meta, out_dir)

    log.info("== slot=%s: writing info.json + stats.json + README.md ==", slot)
    write_info_json(out_dir, total_episodes=len(records), total_frames=len(data_df), approach_src=approach_src)
    write_stats_json(data_df, approach_src, place_src, out_dir)
    write_readme(out_dir, slot, total_episodes=len(records), total_frames=len(data_df))

    log.info("== slot=%s: stitching video (ffmpeg, bridge_video_mode=%s) ==", slot, bridge_video_mode)
    video_out = out_dir / "videos" / IMAGE_KEY / "chunk-000" / "file-000.mp4"
    with tempfile.TemporaryDirectory(prefix=f"v5_concat_{slot}_") as tmp:
        stitch_slot_videos(
            slot, approach_src, place_src, pair_list, bridge_frames,
            bridge_video_mode, video_out, Path(tmp),
        )

    log.info("== slot=%s: materialized -> %s ==", slot, out_dir)
    return {
        "slot": slot,
        "out_dir": str(out_dir),
        "total_episodes": len(records),
        "total_frames": len(data_df),
        "seam_distances_deg": [em["seam_distance_deg"] for em in ep_meta],
    }


# --------------------------------------------------------------------------- #
# Hub push                                                                    #
# --------------------------------------------------------------------------- #

def push_to_hub(out_dir: Path, repo_id: str, tag: str) -> None:
    from huggingface_hub import HfApi
    api = HfApi()
    log.info("== push: create_repo %s ==", repo_id)
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
    log.info("== push: upload_folder %s -> %s ==", out_dir, repo_id)
    api.upload_folder(
        folder_path=str(out_dir),
        repo_id=repo_id,
        repo_type="dataset",
        ignore_patterns=["__pycache__", ".cache/*", ".cache/**"],
    )
    log.info("== push: create_tag %s tag=%s ==", repo_id, tag)
    # Delete existing tag if present (HF re-tag isn't idempotent in 1.x).
    try:
        api.delete_tag(repo_id, tag=tag, repo_type="dataset")
    except Exception:
        pass
    api.create_tag(repo_id=repo_id, tag=tag, repo_type="dataset")


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--slots", default="left,middle,right",
                    help="Comma-separated subset of {left,middle,right}.")
    ap.add_argument("--bridge-frames", type=int, default=20,
                    help="Number of linearly-interpolated bridge frames between approach-end and placement-start.")
    ap.add_argument("--bridge-mode", default="linear", choices=("linear", "hold"),
                    help="State+action bridge mode (data side). linear=interp, hold=freeze approach-last values.")
    ap.add_argument("--bridge-video-mode", default="freeze", choices=("freeze", "blend", "mci"),
                    help="Video bridge mode. freeze=hold approach-last frame (back-compat default); "
                         "blend=alpha cross-fade approach-last->placement-first; "
                         "mci=ffmpeg minterpolate motion-compensated interpolation (best fidelity).")
    ap.add_argument("--out-root", default=str(DATASETS_ROOT),
                    help="Where to materialize datasets (default: ./datasets/).")
    ap.add_argument("--repo-prefix", default="RobotLearningVLA/dataset_v5_charuko",
                    help="Hub repo prefix; output is <prefix>_<slot>_full.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Report pairs / seam distances / frame counts and exit. No writes.")
    ap.add_argument("--overwrite", action="store_true",
                    help="Replace existing output directories.")
    ap.add_argument("--push-to-hub", action="store_true",
                    help="After build, upload to RobotLearningVLA/ and create the v3.0 tag.")
    ap.add_argument("--tag", default=CODEBASE_VERSION,
                    help="Hub git tag matching meta/info.json:codebase_version (default v3.0).")
    args = ap.parse_args()

    slots = [s.strip() for s in args.slots.split(",") if s.strip()]
    for s in slots:
        if s not in ("left", "middle", "right"):
            ap.error(f"invalid slot: {s}")

    approach_src = SourceDataset("dataset_v5_charuko_approach")
    log.info("source approach: %d eps / %d frames", approach_src.num_episodes, approach_src.num_frames)

    out_root = Path(args.out_root)

    results = []
    for slot in slots:
        place_src = SourceDataset(f"dataset_v5_charuko_{slot}_1")
        log.info("source place [%s]: %d eps / %d frames", slot, place_src.num_episodes, place_src.num_frames)

        if args.dry_run:
            report_pairs(slot, approach_src, place_src, args.bridge_frames)
            continue

        out_dir = out_root / f"dataset_v5_charuko_{slot}_full"
        res = materialize_slot(
            slot=slot,
            approach_src=approach_src,
            place_src=place_src,
            out_dir=out_dir,
            bridge_frames=args.bridge_frames,
            bridge_mode=args.bridge_mode,
            bridge_video_mode=args.bridge_video_mode,
            overwrite=args.overwrite,
        )
        results.append(res)

        # Sanity-load via LeRobotDataset.
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
            repo_id = f"{args.repo_prefix}_{slot}_full"
            ds = LeRobotDataset(repo_id, root=str(out_dir), video_backend="pyav")
            log.info("verify: %s loads OK — episodes=%d frames=%d fps=%d",
                     repo_id, ds.num_episodes, ds.num_frames, ds.meta.fps)
        except Exception as e:
            log.error("verify FAILED for %s: %s: %s", out_dir, type(e).__name__, e)
            return 2

        if args.push_to_hub:
            repo_id = f"{args.repo_prefix}_{slot}_full"
            push_to_hub(out_dir, repo_id, args.tag)

    if args.dry_run:
        return 0
    print("\nSUMMARY")
    for r in results:
        seams = r["seam_distances_deg"]
        print(f"  {r['slot']:6s}  eps={r['total_episodes']:2d}  frames={r['total_frames']:5d}  "
              f"seam(min/med/max)={min(seams):.1f}/{np.median(seams):.1f}/{max(seams):.1f}°  "
              f"out={r['out_dir']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
