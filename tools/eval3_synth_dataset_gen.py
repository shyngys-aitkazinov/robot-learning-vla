#!/usr/bin/env python3
"""Synthetic LeRobot v3.0 dataset generator for Eval 3 ChArUco training data.

For each (target_celeb, target_position) pair (3 x 3 = 9 datasets), enumerate
250 unique (target_photo, other1_celeb, other1_photo, other2_celeb, other2_photo,
ordering) configurations and emit one LeRobot v3.0 episode per config. The
action / state trajectories are COPIED from a ChArUco source episode
(``datasets/dataset_v3_charuco_<position>_2``, 10 source episodes cycled 1:1
across the 250 configs). The camera frames are RE-COMPOSITED by warping the
three chosen celebrity images onto the three boards visible in the source
recording, using the existing pipeline from
``tools/eval3_charuco_episode_demo.py`` + the standardized global-lift.

Output: 9 directories under ``--out-root`` (default ``datasets/``):

    dataset_v3_synth_<celeb_slug>_<position>_2/

Each carries v3.0 schema (info.json + stats.json + tasks.parquet + episodes +
data + videos), single task string ``"Place the coke on <Celeb>"`` (canonical
form so ``scripts/eval3_dataset_prep.py`` task-augmenter accepts it), same
6-DOF action/state + 480x640 ``observation.images.front``, fps=30.

Per-dataset breakdown::

    250 configs = 5 (target_celeb photos)
                * 5 (other_celeb_1 photos)
                * 5 (other_celeb_2 photos)
                * 2 (orderings of other_celeb_1 vs other_celeb_2 on non-target boards)

Usage::

    # Smoke test (1 dataset, 5 episodes, ~5 min on M-series):
    python tools/eval3_synth_dataset_gen.py \\
        --target-celebs taylor_swift --target-positions left \\
        --n-configs-per-dataset 5

    # Full sweep on Brev with 9-way multiprocessing + HF push:
    python tools/eval3_synth_dataset_gen.py --n-workers 9 --push-to-hub

    # Dry run (print plan, write nothing):
    python tools/eval3_synth_dataset_gen.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import traceback
from copy import deepcopy
from dataclasses import dataclass, field
from itertools import permutations
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

# Reuse the compose pipeline from the demo (already does table-blend +
# warp + can-preservation + global-lift).
from eval3_charuco_episode_demo import (  # noqa: E402
    _apply_global_lift,
    _apply_table_blend,
    _center_crop_to_aspect,
    RedHSVMasker,
    compose_multi,
    lock_homographies_multi,
)
from eval3_charuco_check import build_board  # noqa: E402

try:
    import cv2.aruco as aruco
except ImportError as e:
    sys.exit(f"opencv-contrib-python required (cv2.aruco missing): {e}")


# ---------------------------------------------------------------------------
# Constants — match the captured ChArUco datasets and the existing pipeline.
# ---------------------------------------------------------------------------

CANONICAL_NAME = {
    "taylor_swift": "Taylor Swift",
    "barack_obama": "Barack Obama",
    "yann_lecun":   "Yann LeCun",
}

# Board geometry (matches all _2 captures + the demo defaults)
BOARD_SQUARES_X = 5
BOARD_SQUARES_Y = 7
BOARD_SQUARE_MM = 22.8
BOARD_MARKER_RATIO = 0.7
BOARD_CHROMA_MM = 60.0
BOARD_DICT = "4X4_50"
BOARD_W_MM = BOARD_SQUARES_X * BOARD_SQUARE_MM
BOARD_H_MM = BOARD_SQUARES_Y * BOARD_SQUARE_MM

HSV_LO = (35, 60, 40)
HSV_HI = (85, 255, 255)

POSITIONS = ("left", "middle", "right")
POSITION_TO_BOARD_IDX = {"left": 0, "middle": 1, "right": 2}

# v3.0 features schema — copied verbatim from
# datasets/dataset_v3_charuco_left_2/meta/info.json so the synthetic
# datasets are byte-for-byte schema-compatible with the source.
# NB: shapes MUST be tuples — lerobot's add_frame validator does
# strict `actual_shape != expected_shape` (tuple vs list mismatch fails).
FEATURES_SCHEMA = {
    "action": {
        "dtype": "float32",
        "shape": (6,),
        "names": [
            "shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
            "wrist_flex.pos", "wrist_roll.pos", "gripper.pos",
        ],
    },
    "observation.state": {
        "dtype": "float32",
        "shape": (6,),
        "names": [
            "shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
            "wrist_flex.pos", "wrist_roll.pos", "gripper.pos",
        ],
    },
    "observation.images.front": {
        "dtype": "video",
        "shape": (480, 640, 3),
        "names": ["height", "width", "channels"],
    },
}


# ---------------------------------------------------------------------------
# Config grid
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Config:
    """One generated episode's recipe."""
    target_celeb: str            # e.g. "taylor_swift"
    target_position: str         # "left" | "middle" | "right"
    target_photo_idx: int        # 0..4 (which of the 5 target photos)
    other1_celeb: str            # other celeb that goes on the FIRST non-target board
    other1_photo_idx: int
    other2_celeb: str            # other celeb on the SECOND non-target board
    other2_photo_idx: int
    other_swap: bool             # if True, other1 and other2 swap board slots


def build_config_grid(
    target_celeb: str, target_position: str, n_photos_per_celeb: int = 5,
) -> list[Config]:
    """Enumerate all configs for one (target_celeb, target_position) dataset.

    With ``n_photos_per_celeb=N``: ``N*N*N*2`` configs (= 250 for N=5, the ID
    pool; = 2000 for N=10, the ID+OOD merged pool). Deterministic — sort key
    is the Config tuple itself.
    """
    if target_celeb not in CANONICAL_NAME:
        sys.exit(f"unknown target_celeb={target_celeb!r}, want one of {list(CANONICAL_NAME)}")
    if target_position not in POSITIONS:
        sys.exit(f"unknown target_position={target_position!r}, want one of {POSITIONS}")
    others = sorted(s for s in CANONICAL_NAME if s != target_celeb)
    assert len(others) == 2, f"expected 2 other celebs, got {others}"
    other_a, other_b = others
    # other1 is always the alphabetically-first other celeb, other2 the second.
    # The `other_swap` flag is interpreted by _board_layout: when True, other1
    # and other2 trade their slot positions on the non-target boards. Do NOT
    # also swap (c1, c2) here — that'd double-swap and cancel out.
    configs: list[Config] = []
    for tp in range(n_photos_per_celeb):
        for ap in range(n_photos_per_celeb):
            for bp in range(n_photos_per_celeb):
                for swap in (False, True):
                    configs.append(Config(
                        target_celeb=target_celeb,
                        target_position=target_position,
                        target_photo_idx=tp,
                        other1_celeb=other_a,
                        other1_photo_idx=ap,
                        other2_celeb=other_b,
                        other2_photo_idx=bp,
                        other_swap=swap,
                    ))
    expected = n_photos_per_celeb ** 3 * 2
    assert len(configs) == expected, f"want {expected} configs, got {len(configs)}"
    return configs


def load_celebrity_pool(json_paths: list[Path]) -> tuple[dict[str, list[Path]], int]:
    """Load + merge celebrity JPG pools from one or more metadata JSONs.

    Returns:
        ({slug: [list of JPG paths]}, n_photos_per_celeb)
        where n_photos_per_celeb = min count across the 3 known celebs
        (so the config grid stays uniform).
    """
    pool: dict[str, list[Path]] = {s: [] for s in CANONICAL_NAME}
    for jp in json_paths:
        meta = json.loads(jp.read_text(encoding="utf-8"))
        for c in meta["celebrities"]:
            slug = c["slug"]
            if slug not in pool:
                continue  # silently skip non-eval3 celebs
            pool[slug].extend(sorted(Path(c["dir"]).glob("*.jpg")))
    counts = {slug: len(v) for slug, v in pool.items()}
    if min(counts.values()) == 0:
        missing = [s for s, n in counts.items() if n == 0]
        sys.exit(f"no photos found for celebs: {missing} (checked {json_paths})")
    n_photos = min(counts.values())
    if max(counts.values()) != n_photos:
        # Truncate to the minimum so the config grid is rectangular.
        print(f"[load_celebrity_pool] non-uniform counts {counts}; "
              f"truncating each celeb to {n_photos} photos for combinatorial uniformity",
              flush=True)
        for slug in pool:
            pool[slug] = pool[slug][:n_photos]
    return pool, n_photos


# ---------------------------------------------------------------------------
# Source episode reader
# ---------------------------------------------------------------------------

@dataclass
class SourceEpisodeMeta:
    episode_index: int
    length: int
    from_frame_global: int
    to_frame_global: int
    from_timestamp: float
    to_timestamp: float
    video_chunk: int
    video_file: int


def _maybe_download_source(
    source_root: Path, ds_name: str, hub_org: str,
) -> None:
    """Download ``hub_org/ds_name`` from HF Hub into ``source_root/ds_name`` if not present."""
    target = source_root / ds_name
    if target.is_dir():
        return
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        sys.exit("huggingface_hub is required for --source-hub-org; pip install huggingface_hub")
    repo_id = f"{hub_org}/{ds_name}"
    print(f"[download] {repo_id} -> {target} ...", flush=True)
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(target),
        ignore_patterns=["*.gitattributes", ".gitattributes"],
    )
    print(f"[download] done: {target}", flush=True)


def load_source_episodes(
    source_root: Path, position: str,
    source_suffix: str = "_2",
    source_prefix: str = "dataset_v3_charuco",
    source_hub_org: str = "",
) -> tuple[Path, list[SourceEpisodeMeta], np.ndarray, np.ndarray]:
    """Return (mp4_path, [SourceEpisodeMeta x N], action_array, state_array).

    Joint arrays are flat (concatenated across all source episodes); use the
    per-episode from/to_frame_global indices to slice them per episode.

    ``source_prefix`` (default ``dataset_v3_charuco``) + ``source_suffix``
    (default ``_2``) form the dataset dir name: ``{prefix}_{position}{suffix}``.
    When ``source_hub_org`` is set and the local dir is missing, the dataset is
    downloaded automatically from ``{source_hub_org}/{ds_name}`` on HF Hub.
    """
    ds_name = f"{source_prefix}_{position}{source_suffix}"
    if source_hub_org:
        _maybe_download_source(source_root, ds_name, source_hub_org)
    ds = source_root / ds_name
    if not ds.is_dir():
        sys.exit(f"source dataset missing: {ds}")
    # Episodes parquet — one per chunk (we know v3 ChArUco has chunk-000 only).
    ep_files = sorted(ds.glob("meta/episodes/chunk-*/file-*.parquet"))
    if not ep_files:
        sys.exit(f"no meta/episodes parquet under {ds}")
    cols = [
        "episode_index", "length",
        "dataset_from_index", "dataset_to_index",
        "videos/observation.images.front/chunk_index",
        "videos/observation.images.front/file_index",
        "videos/observation.images.front/from_timestamp",
        "videos/observation.images.front/to_timestamp",
    ]
    rows = []
    for f in ep_files:
        df = pq.read_table(f, columns=cols).to_pandas()
        for _, r in df.iterrows():
            rows.append(SourceEpisodeMeta(
                episode_index=int(r["episode_index"]),
                length=int(r["length"]),
                from_frame_global=int(r["dataset_from_index"]),
                to_frame_global=int(r["dataset_to_index"]),
                from_timestamp=float(r["videos/observation.images.front/from_timestamp"]),
                to_timestamp=float(r["videos/observation.images.front/to_timestamp"]),
                video_chunk=int(r["videos/observation.images.front/chunk_index"]),
                video_file=int(r["videos/observation.images.front/file_index"]),
            ))
    rows.sort(key=lambda m: m.episode_index)
    # Data parquet — flat action+state array
    data_files = sorted(ds.glob("data/chunk-*/file-*.parquet"))
    if not data_files:
        sys.exit(f"no data parquet under {ds}")
    data_dfs = [pq.read_table(f, columns=["action", "observation.state"]).to_pandas()
                for f in data_files]
    action = np.stack(np.concatenate([df["action"].to_numpy() for df in data_dfs])).astype(np.float32)
    state = np.stack(np.concatenate([df["observation.state"].to_numpy() for df in data_dfs])).astype(np.float32)
    # All sources have a single MP4 (chunk-000/file-000) in our v3 ChArUco corpus.
    chunk_idx = rows[0].video_chunk
    file_idx = rows[0].video_file
    mp4 = ds / "videos" / "observation.images.front" / f"chunk-{chunk_idx:03d}" / f"file-{file_idx:03d}.mp4"
    if not mp4.is_file():
        sys.exit(f"source video missing: {mp4}")
    return mp4, rows, action, state


def read_episode_frames(mp4_path: Path, ep: SourceEpisodeMeta, fps: float) -> list[np.ndarray]:
    """Read all frames of one source episode into memory as a list of BGR ndarrays.

    ChArUco episodes are ~500 frames = ~500 * 480*640*3 bytes = ~440 MB worst
    case — fits comfortably in a single worker's RAM. We read the whole episode
    at once so the compose loop can run linearly without seek overhead.
    """
    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        sys.exit(f"cv2.VideoCapture failed on {mp4_path}")
    start_frame = int(round(ep.from_timestamp * fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frames: list[np.ndarray] = []
    for _ in range(ep.length):
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if len(frames) != ep.length:
        # Fall back to sequential read from the top if seek was inexact.
        cap = cv2.VideoCapture(str(mp4_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frames = []
        for i in range(ep.to_frame_global):
            ok, frame = cap.read()
            if not ok:
                break
            if i >= ep.from_frame_global:
                frames.append(frame)
        cap.release()
    if len(frames) != ep.length:
        raise RuntimeError(
            f"expected {ep.length} frames from {mp4_path} ep{ep.episode_index} "
            f"(start_ts={ep.from_timestamp:.3f}), got {len(frames)}"
        )
    return frames


# ---------------------------------------------------------------------------
# Celebrity tile cache (per-worker)
# ---------------------------------------------------------------------------

class TileCache:
    """Pre-loads + blends celebrity tiles once per worker.

    Key: (celeb_slug, photo_idx). Value: BGR ndarray at the internal warp
    resolution (board_w_mm*10, board_h_mm*10) with per-tile table-blend applied.

    The per-tile blend noise is seeded by ``(seed_base, slug, photo_idx)`` so
    the same (slug, photo_idx) always produces the bit-identical tile regardless
    of worker, load order, or which other tiles were preloaded first. When the
    three v4 balanced workers share the same ``seed_base`` (see
    ``generate_one_balanced_dataset``) the celebrity face composites are
    pixel-identical across the three per-celebrity datasets. This is what makes
    the trainer see *strict* same-image-different-prompt counterfactuals
    (closes the "no counterfactual supervision" gap in
    ``outputs/eval3_celebrity_diagnosis/DIAGNOSIS_REPORT.md``).
    """

    def __init__(
        self,
        celebrity_jsons: list[Path],
        blend_args: dict,
        target_w_px: int,
        target_h_px: int,
        board_aspect: float,
        rng_seed: int,
    ):
        self._cache: dict[tuple[str, int], np.ndarray] = {}
        # Merge across all provided JSONs so OOD photos can be appended to the
        # ID pool transparently.
        self._jpgs_by_slug, self._n_photos = load_celebrity_pool(celebrity_jsons)
        self._blend_args = blend_args
        self._tw, self._th = target_w_px, target_h_px
        self._board_aspect = board_aspect
        # Stored as a base for the per-tile-deterministic RNG (see `get`).
        # Two workers with the same `_seed_base` produce bit-identical tiles
        # for the same (slug, photo_idx).
        self._seed_base = int(rng_seed)

    @property
    def n_photos_per_celeb(self) -> int:
        return self._n_photos

    def get(self, slug: str, photo_idx: int) -> np.ndarray:
        key = (slug, photo_idx)
        if key not in self._cache:
            jpgs = self._jpgs_by_slug.get(slug)
            if not jpgs:
                sys.exit(f"no JPGs found for celeb={slug!r} in celebrity-json")
            if not (0 <= photo_idx < len(jpgs)):
                sys.exit(f"photo_idx={photo_idx} out of range for {slug} (have {len(jpgs)})")
            raw = cv2.imread(str(jpgs[photo_idx]))
            if raw is None:
                sys.exit(f"failed to read {jpgs[photo_idx]}")
            cropped = _center_crop_to_aspect(raw, self._board_aspect)
            resized = cv2.resize(cropped, (self._tw, self._th), interpolation=cv2.INTER_AREA)
            tile_seed = abs(hash((self._seed_base, slug, int(photo_idx)))) % (2**31)
            tile_rng = np.random.default_rng(tile_seed)
            tile = _apply_table_blend(resized, rng=tile_rng, **self._blend_args)
            self._cache[key] = tile
        return self._cache[key]


# ---------------------------------------------------------------------------
# Per-dataset worker
# ---------------------------------------------------------------------------

@dataclass
class WorkerArgs:
    """All inputs for generate_one_dataset, picklable for multiprocessing."""
    target_celeb: str
    target_position: str            # ignored when balanced=True; covers all 3 slots
    n_configs: int                  # cap (-1 = use full grid)
    source_root: str
    out_root: str
    celebrity_jsons: list[str]      # one or more — merged at tile-load time
    blend_args: dict
    global_lift_args: dict
    vcodec: str
    push_to_hub: bool
    hub_org: str
    overwrite: bool
    output_suffix: str = ""         # appended after 'dataset_v3_synth' in the dataset name
    # Fix A (Eval3 celeb-confusion plan): emit one balanced dataset per celebrity
    # whose episodes cover ALL three placement slots. This breaks the
    # action = f(slot) shortcut documented in
    # ``docs/eval3/charuco_pipeline.md:281`` by varying the action target
    # within a single (task = "Place the coke on <Celeb>") dataset.
    balanced: bool = False
    output_version: str = "v3"      # "v3" -> dataset_v3_synth_*  ;  "v4" -> dataset_v4_synth_*
    # Source/output variant — default _2 (newer captures, current corpus).
    # The pinned generator sets both to _1 to target the older captures.
    source_suffix: str = "_2"
    output_postfix: str = "_2"
    # Prefix for the source ChArUco dataset dir names (default "dataset_v3_charuco").
    # Override e.g. to "dataset_v5_charuko" to use the v5 Hub captures.
    source_prefix: str = "dataset_v3_charuco"
    # If set, auto-download missing source datasets from this HF Hub org.
    source_hub_org: str = ""
    # Optional override for the config grid. When None (default), use the full
    # build_config_grid (5^3 * 2 = 250 for N=5, 2000 for N=10). When set, the
    # worker uses exactly this list of configs — the pinned generator passes
    # in a 5×N pinned-distractor grid here.
    config_grid_override: list | None = None
    # Intra-dataset sharding. Defaults (shard_count=1, is_shard=False) preserve
    # byte-identical behavior with the pre-sharding implementation. When N>1,
    # _run_with_sharding builds N WorkerArgs per (celeb, position) target,
    # each producing a distinct config slice to its own _shard_{i} dir.
    shard_index: int = 0
    shard_count: int = 1
    is_shard: bool = False


def _board_layout(target_position: str, other_celeb_1: str, other_celeb_2: str,
                  swap: bool) -> list[str]:
    """Return [celeb_on_left, celeb_on_middle, celeb_on_right] for one config."""
    layout = [None, None, None]
    target_idx = POSITION_TO_BOARD_IDX[target_position]
    layout[target_idx] = "<target>"  # placeholder, filled in by caller
    other_slots = [i for i in range(3) if i != target_idx]  # always sorted asc
    if swap:
        other_slots = other_slots[::-1]
    layout[other_slots[0]] = other_celeb_1
    layout[other_slots[1]] = other_celeb_2
    return layout


def build_balanced_config_grid(
    target_celeb: str, n_photos_per_celeb: int = 5,
) -> list[Config]:
    """Same combinatorics as ``build_config_grid`` but with target_position rotating
    through all 3 slots inside one dataset. Used by the Fix A "language-forcing"
    mode: by spanning all 3 placement slots in a single (task = celeb) dataset,
    the action label varies WITHIN the dataset and the trainer can no longer
    shortcut on "action = f(slot)".

    Total configs per celeb dataset: ``3 * n_photos_per_celeb^3 * 2``
    (= 750 for the default 5-photo pool).
    """
    if target_celeb not in CANONICAL_NAME:
        sys.exit(f"unknown target_celeb={target_celeb!r}, want one of {list(CANONICAL_NAME)}")
    others = sorted(s for s in CANONICAL_NAME if s != target_celeb)
    other_a, other_b = others
    configs: list[Config] = []
    for pos in POSITIONS:
        for tp in range(n_photos_per_celeb):
            for ap in range(n_photos_per_celeb):
                for bp in range(n_photos_per_celeb):
                    for swap in (False, True):
                        configs.append(Config(
                            target_celeb=target_celeb,
                            target_position=pos,
                            target_photo_idx=tp,
                            other1_celeb=other_a,
                            other1_photo_idx=ap,
                            other2_celeb=other_b,
                            other2_photo_idx=bp,
                            other_swap=swap,
                        ))
    expected = 3 * (n_photos_per_celeb ** 3) * 2
    assert len(configs) == expected, f"want {expected} configs, got {len(configs)}"
    return configs


def generate_one_dataset(args: WorkerArgs) -> dict:
    """Build one synthetic dataset end-to-end. Returns a summary dict."""
    t0 = time.time()
    target_celeb = args.target_celeb
    target_position = args.target_position
    canonical_task = f"Place the coke on {CANONICAL_NAME[target_celeb]}"
    suffix = f"_{args.output_suffix}" if args.output_suffix else ""
    out_name = f"dataset_v3_synth{suffix}_{target_celeb}_{target_position}{args.output_postfix}"
    # When this worker is producing a shard (one slice of N), write to a
    # _shard_{i} sibling dir. The orchestrator will merge all shards into the
    # canonical out_name dir after the pool finishes.
    if args.is_shard:
        out_name = f"{out_name}_shard_{args.shard_index}"
    out_root = Path(args.out_root)
    out_dir = out_root / out_name
    if out_dir.exists():
        if args.overwrite:
            import shutil
            shutil.rmtree(out_dir)
        else:
            return {
                "name": out_name, "skipped": True,
                "reason": f"output dir already exists: {out_dir}",
            }

    print(f"[{out_name}] START — task='{canonical_task}'", flush=True)

    # --- 1. Celebrity pool + tile cache + config grid ----------------------
    target_w_px = int(round(BOARD_W_MM * 10))
    target_h_px = int(round(BOARD_H_MM * 10))
    board_aspect = BOARD_H_MM / BOARD_W_MM
    tile_seed = abs(hash((target_celeb, target_position))) % (2**31)
    tile_cache = TileCache(
        celebrity_jsons=[Path(p) for p in args.celebrity_jsons],
        blend_args=args.blend_args,
        target_w_px=target_w_px,
        target_h_px=target_h_px,
        board_aspect=board_aspect,
        rng_seed=tile_seed,
    )
    n_photos = tile_cache.n_photos_per_celeb
    # Use the override grid if provided (pinned generator), else the full
    # build_config_grid product (default _2 generator).
    if args.config_grid_override is not None:
        full_grid = list(args.config_grid_override)
    else:
        full_grid = build_config_grid(target_celeb, target_position, n_photos_per_celeb=n_photos)
    n_configs = len(full_grid) if args.n_configs < 0 else min(args.n_configs, len(full_grid))
    # Even-split of [0:n_configs] across args.shard_count shards. Shards 0..rem-1
    # take base+1 configs; the rest take base. No gap, no overlap. For
    # shard_count=1 this reduces to start=0 / end=n_configs, byte-identical to
    # the pre-sharding `configs = full_grid[:n_configs]`.
    base = n_configs // args.shard_count
    rem = n_configs % args.shard_count
    shard_start = args.shard_index * base + min(args.shard_index, rem)
    shard_end = shard_start + base + (1 if args.shard_index < rem else 0)
    configs = full_grid[shard_start:shard_end]
    print(f"[{out_name}] pool: {n_photos} photos/celeb -> {len(full_grid)} full configs, "
          f"using {n_configs} (shard {args.shard_index + 1}/{args.shard_count}: "
          f"configs[{shard_start}:{shard_end}] = {len(configs)} eps)", flush=True)

    # --- 2. Source episodes + joints ---------------------------------------
    src_mp4, src_episodes, src_action, src_state = load_source_episodes(
        Path(args.source_root), target_position, source_suffix=args.source_suffix,
    )
    print(f"[{out_name}] source: {src_mp4.name}  "
          f"{len(src_episodes)} episodes, {len(src_action)} frames total", flush=True)
    board, dictionary = build_board(
        BOARD_SQUARES_X, BOARD_SQUARES_Y, BOARD_SQUARE_MM, BOARD_MARKER_RATIO, BOARD_DICT,
    )
    detector = aruco.ArucoDetector(dictionary, aruco.DetectorParameters())
    char_detector = aruco.CharucoDetector(board)
    masker = RedHSVMasker(HSV_LO, HSV_HI)

    # --- 4. Homography cache per source episode ----------------------------
    h_cache: dict[int, list[np.ndarray | None]] = {}

    def get_homographies(src_ep_idx: int, first_frame: np.ndarray) -> list[np.ndarray | None]:
        if src_ep_idx not in h_cache:
            Hs, _infos = lock_homographies_multi(first_frame, detector, char_detector, board, n_boards=3)
            h_cache[src_ep_idx] = Hs
        return h_cache[src_ep_idx]

    # --- 5. Create the output dataset --------------------------------------
    # Import here so a missing lerobot doesn't break --dry-run
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    # For shard outputs, repo_id is purely a local identifier (no push) and we
    # use the local out_name (with _shard_{i} suffix already applied).
    repo_id = f"{args.hub_org}/{out_name}" if (args.push_to_hub and not args.is_shard) else out_name
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=30,
        features=deepcopy(FEATURES_SCHEMA),
        root=out_dir,
        robot_type="so_follower",
        use_videos=True,
        vcodec=args.vcodec,
        streaming_encoding=True,  # bounded memory; no on-disk PNG buffer
    )

    # --- 6. Per-config generation loop -------------------------------------
    n_eps_total = len(configs)
    n_eps_done = 0
    n_frames_done = 0
    fps = 30.0
    n_source = len(src_episodes)
    for ci, cfg in enumerate(configs):
        # Cycle source episodes 1:1 across configs. Use the GLOBAL config
        # index (shard_start + ci) so config N picks the same source episode
        # whether it lands in shard 0's slot 666 or in the unsharded run's
        # slot 666. For shard_count=1 this reduces to `ci % n_source`.
        src_idx = (shard_start + ci) % n_source
        src = src_episodes[src_idx]
        # Read frames + slice joints
        frames = read_episode_frames(src_mp4, src, fps)
        action_slice = src_action[src.from_frame_global: src.to_frame_global]
        state_slice = src_state[src.from_frame_global: src.to_frame_global]
        assert len(frames) == len(action_slice) == len(state_slice) == src.length, (
            f"length mismatch ep{src.episode_index}: "
            f"frames={len(frames)} act={len(action_slice)} st={len(state_slice)} L={src.length}"
        )
        # Resolve celebrity assignment per board slot
        layout = _board_layout(
            target_position, cfg.other1_celeb, cfg.other2_celeb, cfg.other_swap,
        )
        target_idx = POSITION_TO_BOARD_IDX[target_position]
        # Photo-idx per board slot
        photo_idx_per_slot = [None, None, None]
        photo_idx_per_slot[target_idx] = cfg.target_photo_idx
        # The other_slots ordering matches _board_layout
        other_slots = [i for i in range(3) if i != target_idx]
        if cfg.other_swap:
            other_slots = other_slots[::-1]
        photo_idx_per_slot[other_slots[0]] = cfg.other1_photo_idx
        photo_idx_per_slot[other_slots[1]] = cfg.other2_photo_idx
        # Fill in the target celeb in layout
        layout[target_idx] = target_celeb
        # Build the 3 tiles in board-slot order (left, middle, right)
        tiles = [tile_cache.get(layout[i], photo_idx_per_slot[i]) for i in range(3)]
        # Lock homographies on this episode's frame 0
        Hs = get_homographies(src_idx, frames[0])
        if all(H is None for H in Hs):
            print(f"[{out_name}] WARN ep{ci}: source ep{src.episode_index} "
                  f"had no board locks; skipping", flush=True)
            continue
        # Compose every frame, write into the dataset
        for fi, frame in enumerate(frames):
            composed = compose_multi(
                frame, tiles, Hs, BOARD_W_MM, BOARD_H_MM, BOARD_CHROMA_MM,
                HSV_LO, HSV_HI, masker=masker,
            )
            composed = _apply_global_lift(composed, **args.global_lift_args)
            # lerobot expects RGB uint8 (480, 640, 3)
            composed_rgb = cv2.cvtColor(composed, cv2.COLOR_BGR2RGB)
            dataset.add_frame({
                "observation.images.front": composed_rgb,
                "observation.state": state_slice[fi],
                "action": action_slice[fi],
                "task": canonical_task,
            })
        dataset.save_episode()
        n_eps_done += 1
        n_frames_done += src.length
        if n_eps_done % 10 == 0 or n_eps_done == n_eps_total:
            elapsed = time.time() - t0
            fps_compose = n_frames_done / max(elapsed, 1e-6)
            eta = (n_eps_total - n_eps_done) * elapsed / max(n_eps_done, 1)
            print(f"[{out_name}] {n_eps_done}/{n_eps_total} eps "
                  f"({n_frames_done} frames, {fps_compose:.1f} fps, "
                  f"eta {eta / 60:.1f} min)", flush=True)

    dataset.finalize()
    elapsed = time.time() - t0
    summary = {
        "name": out_name,
        "skipped": False,
        "n_episodes": n_eps_done,
        "n_frames": n_frames_done,
        "elapsed_s": round(elapsed, 1),
        "out_dir": str(out_dir),
        "disk_mb": round(_dir_size_mb(out_dir), 1),
    }

    # Optional HF push (skipped for shards — the orchestrator pushes the
    # merged dataset once instead).
    if args.push_to_hub and not args.is_shard:
        upload_info = _push_to_hub(out_dir, repo_id)
        summary["hub_url"] = upload_info["hub_url"]
        summary["upload_elapsed_s"] = upload_info["elapsed_s"]

    print(f"[{out_name}] DONE — {summary}", flush=True)
    return summary


def generate_one_balanced_dataset(args: WorkerArgs) -> dict:
    """Fix A: build one dataset whose episodes COVER ALL THREE placement slots.

    Output naming follows ``args.output_version``:
      - "v3" -> ``dataset_v3_synth{suffix}_<celeb>_balanced_2`` (backwards name)
      - "v4" -> ``dataset_v4_synth{suffix}_<celeb>_balanced_1`` (recommended)

    Differences vs ``generate_one_dataset``:
      * config grid spans all 3 slots (``build_balanced_config_grid``)
      * source episodes are loaded for ALL three ``dataset_v3_charuco_<pos>_2``
      * per-config dispatch picks the source matching ``config.target_position``,
        so the action / state / camera frames all come from a left/middle/right
        ChArUco recording that matches where ``target_celeb`` is placed
      * task string is unchanged (one canonical celebrity per dataset);
        only the VISUAL placement and the ACTION TARGET vary together
    """
    t0 = time.time()
    target_celeb = args.target_celeb
    canonical_task = f"Place the coke on {CANONICAL_NAME[target_celeb]}"
    suffix = f"_{args.output_suffix}" if args.output_suffix else ""
    if args.output_version == "v4":
        out_name = f"dataset_v4_synth{suffix}_{target_celeb}_balanced_1"
    else:
        out_name = f"dataset_v3_synth{suffix}_{target_celeb}_balanced_2"
    out_root = Path(args.out_root)
    out_dir = out_root / out_name
    if out_dir.exists():
        if args.overwrite:
            import shutil
            shutil.rmtree(out_dir)
        else:
            return {
                "name": out_name, "skipped": True,
                "reason": f"output dir already exists: {out_dir}",
            }

    print(f"[{out_name}] START (balanced) — task='{canonical_task}'", flush=True)

    target_w_px = int(round(BOARD_W_MM * 10))
    target_h_px = int(round(BOARD_H_MM * 10))
    board_aspect = BOARD_H_MM / BOARD_W_MM
    # Celeb-INDEPENDENT seed base: all three balanced workers (Swift, LeCun,
    # Obama) share the same tile-cache seed, so (slug, photo_idx) -> tile
    # is bit-identical across the three v4 datasets. Combined with the
    # per-tile-deterministic RNG inside ``TileCache.get`` this means the
    # celebrity face composites are pixel-identical across the three
    # ``dataset_v4_synth_<celeb>_balanced_1`` datasets, giving the trainer
    # strict same-face-different-prompt counterfactuals (as opposed to
    # merely "similar visual layout, different prompt"). Background frames
    # still differ because the ChArUco source recording depends on the
    # target slot, but the discriminative pixels (the three warped face
    # tiles) are identical.
    tile_seed = abs(hash((args.output_version, "balanced", "v2_shared"))) % (2**31)
    tile_cache = TileCache(
        celebrity_jsons=[Path(p) for p in args.celebrity_jsons],
        blend_args=args.blend_args,
        target_w_px=target_w_px,
        target_h_px=target_h_px,
        board_aspect=board_aspect,
        rng_seed=tile_seed,
    )
    n_photos = tile_cache.n_photos_per_celeb
    full_grid = build_balanced_config_grid(target_celeb, n_photos_per_celeb=n_photos)
    n_configs = len(full_grid) if args.n_configs < 0 else min(args.n_configs, len(full_grid))
    configs = full_grid[:n_configs]
    print(f"[{out_name}] pool: {n_photos} photos/celeb -> {len(full_grid)} full configs, "
          f"using {n_configs}", flush=True)

    # Load all three ChArUco source datasets. Each gives (mp4, episodes, action, state).
    per_position: dict[str, tuple[Path, list[SourceEpisodeMeta], np.ndarray, np.ndarray]] = {}
    for pos in POSITIONS:
        src = load_source_episodes(Path(args.source_root), pos)
        per_position[pos] = src
        print(f"[{out_name}] source[{pos}]: {src[0].name}  "
              f"{len(src[1])} episodes, {len(src[2])} frames total", flush=True)

    board, dictionary = build_board(
        BOARD_SQUARES_X, BOARD_SQUARES_Y, BOARD_SQUARE_MM, BOARD_MARKER_RATIO, BOARD_DICT,
    )
    detector = aruco.ArucoDetector(dictionary, aruco.DetectorParameters())
    char_detector = aruco.CharucoDetector(board)
    masker = RedHSVMasker(HSV_LO, HSV_HI)

    # Homography cache keyed by (position, source_episode_index).
    h_cache: dict[tuple[str, int], list[np.ndarray | None]] = {}

    def get_homographies(pos: str, src_ep_idx: int, first_frame: np.ndarray) -> list[np.ndarray | None]:
        key = (pos, src_ep_idx)
        if key not in h_cache:
            Hs, _infos = lock_homographies_multi(first_frame, detector, char_detector, board, n_boards=3)
            h_cache[key] = Hs
        return h_cache[key]

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    repo_id = f"{args.hub_org}/{out_name}" if args.push_to_hub else out_name
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=30,
        features=deepcopy(FEATURES_SCHEMA),
        root=out_dir,
        robot_type="so_follower",
        use_videos=True,
        vcodec=args.vcodec,
        streaming_encoding=True,
    )

    n_eps_total = len(configs)
    n_eps_done = 0
    n_frames_done = 0
    fps = 30.0
    # Per-position episode cycling counter (so each source position uses its
    # 10 episodes evenly).
    n_used_per_pos = {pos: 0 for pos in POSITIONS}
    per_position_eps = {pos: per_position[pos][1] for pos in POSITIONS}

    for ci, cfg in enumerate(configs):
        pos = cfg.target_position
        src_mp4, src_episodes, src_action, src_state = per_position[pos]
        n_source = len(src_episodes)
        src_idx = n_used_per_pos[pos] % n_source
        n_used_per_pos[pos] += 1
        src = src_episodes[src_idx]

        frames = read_episode_frames(src_mp4, src, fps)
        action_slice = src_action[src.from_frame_global: src.to_frame_global]
        state_slice = src_state[src.from_frame_global: src.to_frame_global]
        assert len(frames) == len(action_slice) == len(state_slice) == src.length, (
            f"length mismatch pos={pos} ep{src.episode_index}: "
            f"frames={len(frames)} act={len(action_slice)} st={len(state_slice)} L={src.length}"
        )

        layout = _board_layout(pos, cfg.other1_celeb, cfg.other2_celeb, cfg.other_swap)
        target_idx = POSITION_TO_BOARD_IDX[pos]
        photo_idx_per_slot = [None, None, None]
        photo_idx_per_slot[target_idx] = cfg.target_photo_idx
        other_slots = [i for i in range(3) if i != target_idx]
        if cfg.other_swap:
            other_slots = other_slots[::-1]
        photo_idx_per_slot[other_slots[0]] = cfg.other1_photo_idx
        photo_idx_per_slot[other_slots[1]] = cfg.other2_photo_idx
        layout[target_idx] = target_celeb
        tiles = [tile_cache.get(layout[i], photo_idx_per_slot[i]) for i in range(3)]

        Hs = get_homographies(pos, src_idx, frames[0])
        if all(H is None for H in Hs):
            print(f"[{out_name}] WARN ep{ci}: source pos={pos} ep{src.episode_index} "
                  f"had no board locks; skipping", flush=True)
            continue

        for fi, frame in enumerate(frames):
            composed = compose_multi(
                frame, tiles, Hs, BOARD_W_MM, BOARD_H_MM, BOARD_CHROMA_MM,
                HSV_LO, HSV_HI, masker=masker,
            )
            composed = _apply_global_lift(composed, **args.global_lift_args)
            composed_rgb = cv2.cvtColor(composed, cv2.COLOR_BGR2RGB)
            dataset.add_frame({
                "observation.images.front": composed_rgb,
                "observation.state": state_slice[fi],
                "action": action_slice[fi],
                "task": canonical_task,
            })
        dataset.save_episode()
        n_eps_done += 1
        n_frames_done += src.length
        if n_eps_done % 25 == 0 or n_eps_done == n_eps_total:
            elapsed = time.time() - t0
            fps_compose = n_frames_done / max(elapsed, 1e-6)
            eta = (n_eps_total - n_eps_done) * elapsed / max(n_eps_done, 1)
            print(f"[{out_name}] {n_eps_done}/{n_eps_total} eps "
                  f"({n_frames_done} frames, {fps_compose:.1f} fps, "
                  f"eta {eta / 60:.1f} min)  pos_counts={n_used_per_pos}", flush=True)

    dataset.finalize()
    elapsed = time.time() - t0
    summary = {
        "name": out_name,
        "skipped": False,
        "balanced": True,
        "n_episodes": n_eps_done,
        "n_frames": n_frames_done,
        "elapsed_s": round(elapsed, 1),
        "out_dir": str(out_dir),
        "disk_mb": round(_dir_size_mb(out_dir), 1),
        "position_episode_counts": dict(n_used_per_pos),
    }

    if args.push_to_hub:
        upload_info = _push_to_hub(out_dir, repo_id)
        summary["hub_url"] = upload_info["hub_url"]
        summary["upload_elapsed_s"] = upload_info["elapsed_s"]

    print(f"[{out_name}] DONE — {summary}", flush=True)
    return summary


def _dir_size_mb(p: Path) -> float:
    total = 0
    for f in p.rglob("*"):
        if f.is_file():
            total += f.stat().st_size
    return total / (1024 * 1024)


def _push_to_hub(local_dir: Path, repo_id: str) -> dict:
    """Upload one finalized dataset to HF Hub + create v3.0 tag."""
    from huggingface_hub import HfApi
    api = HfApi()
    t0 = time.time()
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        folder_path=str(local_dir),
        repo_type="dataset",
        commit_message=f"Initial commit: synthetic ChArUco-derived training corpus",
    )
    api.create_tag(repo_id=repo_id, tag="v3.0", repo_type="dataset", exist_ok=True)
    return {
        "hub_url": f"https://huggingface.co/datasets/{repo_id}",
        "elapsed_s": round(time.time() - t0, 1),
    }


# ---------------------------------------------------------------------------
# CLI / dispatcher
# ---------------------------------------------------------------------------

def _dry_run(
    targets: list[tuple[str, str]], n_configs: int, source_root: Path,
    celebrity_jsons: list[Path], output_suffix: str, balanced: bool = False,
    output_version: str = "v3",
    shards_per_dataset: int = 1,
) -> None:
    """Print the plan without writing anything."""
    pool, n_photos = load_celebrity_pool(celebrity_jsons)
    full_grid_size = (3 if balanced else 1) * n_photos ** 3 * 2
    effective = full_grid_size if n_configs < 0 else min(n_configs, full_grid_size)
    suffix = f"_{output_suffix}" if output_suffix else ""
    print("=" * 72)
    mode_str = "balanced (Fix A: language-forcing)" if balanced else "per-slot (v3 legacy)"
    print(f"DRY RUN ({mode_str}) — {len(targets)} datasets x {effective} eps = "
          f"{len(targets) * effective} total")
    if shards_per_dataset > 1:
        # Show the per-shard split.
        base = effective // shards_per_dataset
        rem = effective % shards_per_dataset
        slice_lens = [base + (1 if i < rem else 0) for i in range(shards_per_dataset)]
        print(f"  sharding        : {shards_per_dataset} shards/dataset "
              f"-> {len(targets) * shards_per_dataset} parallel work units; "
              f"per-shard slice lengths = {slice_lens}")
        print(f"                    after gen, lerobot aggregate_datasets() merges "
              f"each dataset's shards into one final repo")
    print(f"  celebrity_jsons : {[str(p) for p in celebrity_jsons]}")
    print(f"  n_photos/celeb  : {n_photos}  ({', '.join(f'{s}={len(pool[s])}' for s in pool)})")
    grid_formula = f"3 * {n_photos}^3 * 2" if balanced else f"{n_photos}^3 * 2"
    print(f"  full grid size  : {full_grid_size}  (= {grid_formula})")
    print(f"  using           : {effective} configs/dataset")
    print(f"  output suffix   : {suffix!r}")
    print(f"  output version  : {output_version}")
    print("=" * 72)
    for celeb, pos in targets:
        if balanced:
            if output_version == "v4":
                out_name = f"dataset_v4_synth{suffix}_{celeb}_balanced_1"
            else:
                out_name = f"dataset_v3_synth{suffix}_{celeb}_balanced_2"
            task = f"Place the coke on {CANONICAL_NAME[celeb]}"
            configs = build_balanced_config_grid(celeb, n_photos_per_celeb=n_photos)[:effective]
            src_dirs = [source_root / f"dataset_v3_charuco_{p}_2" for p in POSITIONS]
            src_status = {p: ("OK" if d.is_dir() else "MISSING")
                          for p, d in zip(POSITIONS, src_dirs)}
            print(f"\n{out_name}  ({effective} eps, task='{task}')")
            print(f"  sources: " + "  ".join(f"{p}=[{src_status[p]}]" for p in POSITIONS))
            # Show first config per slot.
            print(f"  sample configs (1 per slot):")
            seen_slots: set[str] = set()
            for c in configs:
                if c.target_position in seen_slots:
                    continue
                seen_slots.add(c.target_position)
                print(f"    target=({c.target_celeb}@{c.target_position},photo={c.target_photo_idx})  "
                      f"other1=({c.other1_celeb},photo={c.other1_photo_idx})  "
                      f"other2=({c.other2_celeb},photo={c.other2_photo_idx})  "
                      f"swap={c.other_swap}")
                if len(seen_slots) == 3:
                    break
        else:
            out_name = f"dataset_v3_synth{suffix}_{celeb}_{pos}_2"
            task = f"Place the coke on {CANONICAL_NAME[celeb]}"
            configs = build_config_grid(celeb, pos, n_photos_per_celeb=n_photos)[:effective]
            src_ds = source_root / f"dataset_v3_charuco_{pos}_2"
            exists = "OK" if src_ds.is_dir() else "MISSING"
            print(f"\n{out_name}  ({effective} eps, task='{task}')")
            print(f"  source: {src_ds.name}  [{exists}]")
            print(f"  sample configs (first 3 + last):")
            for c in configs[:3] + ([configs[-1]] if len(configs) > 3 else []):
                print(f"    target=({c.target_celeb},photo={c.target_photo_idx})  "
                      f"other1=({c.other1_celeb},photo={c.other1_photo_idx})  "
                      f"other2=({c.other2_celeb},photo={c.other2_photo_idx})  "
                      f"swap={c.other_swap}")
    print()


def _worker_wrapper(args: WorkerArgs) -> dict:
    """Catches exceptions so one worker failure doesn't kill the pool."""
    fn = generate_one_balanced_dataset if args.balanced else generate_one_dataset
    label = f"{args.target_celeb}_{'balanced' if args.balanced else args.target_position}"
    try:
        result = fn(args)
        # Tag the result with the worker's (celeb, pos, shard_index) so
        # _run_with_sharding can group shards back to their parent dataset
        # without re-parsing the on-disk name. Harmless when balanced=True
        # (shard_count defaults to 1, is_shard=False).
        result["_target_celeb"] = args.target_celeb
        result["_target_position"] = args.target_position
        result["_shard_index"] = args.shard_index
        result["_shard_count"] = args.shard_count
        result["_is_shard"] = args.is_shard
        return result
    except SystemExit as e:
        return {"name": label,
                "error": f"SystemExit({e})", "skipped": True,
                "_target_celeb": args.target_celeb,
                "_target_position": args.target_position,
                "_shard_index": args.shard_index,
                "_shard_count": args.shard_count,
                "_is_shard": args.is_shard}
    except Exception as e:
        return {"name": label,
                "error": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc(), "skipped": True,
                "_target_celeb": args.target_celeb,
                "_target_position": args.target_position,
                "_shard_index": args.shard_index,
                "_shard_count": args.shard_count,
                "_is_shard": args.is_shard}


def _run_with_sharding(
    targets: list[tuple[str, str]],
    shards_per_dataset: int,
    n_workers: int,
    base_worker_args_kwargs: dict,
    push_to_hub: bool,
    hub_org: str,
    keep_shards: bool,
    output_suffix: str,
    out_root: Path,
) -> list[dict]:
    """Orchestrate sharded generation + per-dataset merge + push.

    Builds ``shards_per_dataset * len(targets)`` WorkerArgs, runs them through
    a single Pool.map, then for each target dataset whose shards ALL succeeded
    invokes ``lerobot.datasets.aggregate.aggregate_datasets`` to materialize a
    merged dataset at ``out_root / out_name``, optionally pushes to Hub once,
    and (unless --keep-shards) removes the per-shard dirs.

    Failure of any shard of dataset X skips ONLY that dataset's merge; the
    other 8 are unaffected.

    Returns a list of per-target summary dicts compatible with the SUMMARY
    block in main().
    """
    # 1. Build the shard worker args list (N * 9 entries).
    shard_args_list: list[WorkerArgs] = []
    for c, p in targets:
        for shard_idx in range(shards_per_dataset):
            shard_args_list.append(WorkerArgs(
                target_celeb=c,
                target_position=p,
                push_to_hub=False,           # shards never push individually
                hub_org=hub_org,
                output_suffix=output_suffix,
                shard_index=shard_idx,
                shard_count=shards_per_dataset,
                is_shard=True,
                **base_worker_args_kwargs,
            ))

    print(f"Generating {len(shard_args_list)} shard workers "
          f"({shards_per_dataset} shards/dataset x {len(targets)} datasets) "
          f"with {n_workers} parallel processes")
    t_global = time.time()
    if n_workers == 1:
        shard_results = [_worker_wrapper(wa) for wa in shard_args_list]
    else:
        with Pool(processes=n_workers) as pool:
            shard_results = pool.map(_worker_wrapper, shard_args_list)
    elapsed_shards = time.time() - t_global
    print(f"\nAll {len(shard_results)} shards finished in "
          f"{elapsed_shards / 60:.1f} min. Starting per-dataset merges...")

    # 2. Group shard results by (celeb, position).
    by_target: dict[tuple[str, str], list[dict]] = {}
    for r in shard_results:
        key = (r["_target_celeb"], r["_target_position"])
        by_target.setdefault(key, []).append(r)

    # 3. Merge each target's shards (if all succeeded) into the canonical dir.
    from lerobot.datasets.aggregate import aggregate_datasets

    final_results: list[dict] = []
    suffix = f"_{output_suffix}" if output_suffix else ""
    for (c, p), shards in by_target.items():
        shards = sorted(shards, key=lambda r: r["_shard_index"])
        final_name = f"dataset_v3_synth{suffix}_{c}_{p}_2"
        failed = [r for r in shards if r.get("error") or r.get("skipped")]
        if failed:
            print(f"\n[{final_name}] MERGE SKIPPED — {len(failed)}/{len(shards)} "
                  f"shards failed:")
            for r in failed:
                print(f"  shard {r['_shard_index']}: "
                      f"{r.get('error') or r.get('reason')}")
            final_results.append({
                "name": final_name, "skipped": True,
                "reason": f"{len(failed)}/{len(shards)} shards failed",
                "shard_failures": failed,
            })
            continue

        # All shards present. Run aggregate_datasets.
        shard_repo_ids = [s["name"] for s in shards]
        shard_roots = [Path(s["out_dir"]) for s in shards]
        aggr_repo_id = f"{hub_org}/{final_name}" if push_to_hub else final_name
        aggr_root = out_root / final_name
        # If a stale merged dir exists, blow it away — the shards are the
        # source of truth for this run.
        if aggr_root.exists():
            print(f"[{final_name}] removing stale merged dir {aggr_root}")
            shutil.rmtree(aggr_root)
        t_merge = time.time()
        print(f"\n[{final_name}] MERGING {len(shards)} shards into {aggr_root}")
        try:
            aggregate_datasets(
                repo_ids=shard_repo_ids,
                aggr_repo_id=aggr_repo_id,
                roots=shard_roots,
                aggr_root=aggr_root,
            )
        except Exception as e:
            print(f"[{final_name}] MERGE FAILED: {type(e).__name__}: {e}")
            traceback.print_exc()
            final_results.append({
                "name": final_name, "skipped": True,
                "error": f"merge failed: {type(e).__name__}: {e}",
            })
            continue
        merge_elapsed = time.time() - t_merge

        # Accumulate per-shard counters for the SUMMARY block.
        total_eps = sum(s["n_episodes"] for s in shards)
        total_frames = sum(s["n_frames"] for s in shards)
        merged_disk_mb = round(_dir_size_mb(aggr_root), 1)
        merged_summary = {
            "name": final_name,
            "skipped": False,
            "n_episodes": total_eps,
            "n_frames": total_frames,
            "elapsed_s": round(sum(s["elapsed_s"] for s in shards) + merge_elapsed, 1),
            "out_dir": str(aggr_root),
            "disk_mb": merged_disk_mb,
            "shards": len(shards),
            "merge_elapsed_s": round(merge_elapsed, 1),
        }
        print(f"[{final_name}] MERGED — {total_eps} eps, {total_frames} frames, "
              f"{merged_disk_mb:.1f} MB ({merge_elapsed:.1f}s merge)")

        # 4. Push the merged dataset (once).
        if push_to_hub:
            try:
                upload_info = _push_to_hub(aggr_root, aggr_repo_id)
                merged_summary["hub_url"] = upload_info["hub_url"]
                merged_summary["upload_elapsed_s"] = upload_info["elapsed_s"]
                print(f"[{final_name}] pushed -> {upload_info['hub_url']}")
            except Exception as e:
                # Don't kill the run; keep the merged dataset locally so the
                # user can manually retry push.
                print(f"[{final_name}] PUSH FAILED (merged dataset is still "
                      f"on disk at {aggr_root}): {type(e).__name__}: {e}")
                merged_summary["push_error"] = f"{type(e).__name__}: {e}"

        # 5. Clean up shard dirs (unless --keep-shards).
        if not keep_shards:
            for sd in shard_roots:
                try:
                    if sd.exists():
                        shutil.rmtree(sd)
                except Exception as e:
                    print(f"[{final_name}] warning: failed to remove shard "
                          f"dir {sd}: {e}")

        final_results.append(merged_summary)

    return final_results


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--celebrity-json",
                    default="datasets/in-distribution-eval-3.json",
                    help="Per-celebrity image inventory JSON. Accepts comma-separated "
                         "list to merge multiple pools (e.g. "
                         "'datasets/in-distribution-eval-3.json,datasets/out-distribution-eval-3.json' "
                         "merges ID+OOD into a 10-photos-per-celeb pool).")
    ap.add_argument("--source-root", type=Path, default=Path("datasets"),
                    help="Root containing dataset_v3_charuco_{left,middle,right}_2.")
    ap.add_argument("--out-root", type=Path, default=Path("datasets"),
                    help="Root for synthetic dataset_v3_synth_<celeb>_<pos>_2 dirs.")
    ap.add_argument("--target-celebs", default=",".join(CANONICAL_NAME.keys()),
                    help="Comma-separated target celebs to generate (default: all 3).")
    ap.add_argument("--target-positions", default=",".join(POSITIONS),
                    help="Comma-separated target positions (default: all 3).")
    ap.add_argument("--n-configs-per-dataset", type=int, default=-1,
                    help="Cap on configs per dataset. -1 (default) = use full "
                         "N*N*N*2 grid (250 for 5-photo pool, 2000 for 10-photo pool).")
    ap.add_argument("--output-suffix", default="",
                    help="Tag for the output dataset name. Empty (default) = "
                         "'dataset_v3_synth_<celeb>_<pos>_2'. Set e.g. 'ood' or 'mix' "
                         "to write 'dataset_v3_synth_ood_<celeb>_<pos>_2' alongside ID-only outputs.")
    ap.add_argument("--balanced", action="store_true",
                    help="Fix A: emit ONE balanced dataset per celebrity (3 outputs total) "
                         "whose episodes cover all three placement slots. Breaks the "
                         "action=f(slot) shortcut that lets v9 ChArUco SmolVLA ignore "
                         "the language token (see docs/eval3/charuco_pipeline.md:281 + "
                         "tools/eval3_diagnose_celeb_confusion.py). Action source is "
                         "matched to where the named celebrity is placed.")
    ap.add_argument("--output-version", choices=("v3", "v4"), default="v3",
                    help="Naming convention for outputs. 'v3' (default) keeps the "
                         "legacy 'dataset_v3_synth_*_2' prefix; 'v4' writes "
                         "'dataset_v4_synth_*_1' (recommended with --balanced).")
    ap.add_argument("--vcodec", default="h264",
                    help="Video codec for output MP4s (default h264, matches source). "
                         "Use 'libsvtav1' for smaller files at the cost of decode speed.")
    ap.add_argument("--n-workers", type=int, default=1,
                    help="Multiprocessing workers. With --shards-per-dataset=1 "
                         "(default) the upper limit is len(targets) (=9 for the "
                         "full sweep) — workers above that sit idle. With "
                         "--shards-per-dataset>1, the work unit count becomes "
                         "shards_per_dataset * len(targets), so e.g. 27 workers "
                         "for 3 shards x 9 datasets.")
    ap.add_argument("--shards-per-dataset", type=int, default=1,
                    help="If >1, split each output dataset's config grid into N "
                         "shards, run them in parallel, then merge with "
                         "lerobot.datasets.aggregate.aggregate_datasets and push "
                         "the merged dataset to Hub once. Default 1 = no "
                         "sharding (byte-identical to pre-sharding behavior).")
    ap.add_argument("--keep-shards", action="store_true",
                    help="With --shards-per-dataset>1, preserve the per-shard "
                         "*_shard_{i} dirs after merge. Default is to delete "
                         "them once the merged dataset is built (and pushed).")
    ap.add_argument("--push-to-hub", action="store_true",
                    help="After each dataset finishes, upload to HF Hub + create v3.0 tag. "
                         "With sharding, only the MERGED dataset is pushed (not each shard).")
    ap.add_argument("--hub-org", default="RobotLearningVLA",
                    help="HF org for the repo_id (default RobotLearningVLA).")
    ap.add_argument("--overwrite", action="store_true",
                    help="If output dir exists, delete it before writing.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the plan and exit (no IO).")

    # Inherit per-tile blend defaults from the demo
    ap.add_argument("--blend-contrast", type=float, default=0.88)
    ap.add_argument("--blend-brightness-offset", type=int, default=15)
    ap.add_argument("--blend-saturation", type=float, default=0.9)
    ap.add_argument("--blend-blur-kernel", type=int, default=3)
    ap.add_argument("--blend-noise-sigma", type=float, default=3.0)
    ap.add_argument("--blend-warmth", type=float, default=1.0)

    # Inherit global-lift defaults from the demo
    ap.add_argument("--global-lift-gain", type=float, default=1.25)
    ap.add_argument("--global-lift-offset", type=float, default=10.0)
    ap.add_argument("--global-lift-warmth", type=float, default=1.06)

    args = ap.parse_args()

    celebs = [s.strip() for s in args.target_celebs.split(",") if s.strip()]
    positions = [s.strip() for s in args.target_positions.split(",") if s.strip()]
    if args.balanced:
        # One output per celebrity; --target-positions is ignored (every output
        # spans all 3 slots internally). Keep a dummy "balanced" position string
        # so existing code paths still see a (celeb, pos) tuple.
        targets = [(c, "balanced") for c in celebs]
    else:
        targets = [(c, p) for c in celebs for p in positions]
    celebrity_jsons = [Path(s.strip()) for s in args.celebrity_json.split(",") if s.strip()]

    if args.dry_run:
        _dry_run(targets, args.n_configs_per_dataset, args.source_root,
                 celebrity_jsons, args.output_suffix,
                 balanced=args.balanced,
                 output_version=args.output_version,
                 shards_per_dataset=max(1, int(args.shards_per_dataset)))
        return

    # Validate inputs early.
    for jp in celebrity_jsons:
        if not jp.is_file():
            sys.exit(f"--celebrity-json entry missing: {jp}")
    if args.balanced:
        for p in POSITIONS:
            src_ds = args.source_root / f"dataset_v3_charuco_{p}_2"
            if not src_ds.is_dir():
                sys.exit(f"source dataset missing (balanced needs all 3): {src_ds}")
    else:
        for c, p in targets:
            src_ds = args.source_root / f"dataset_v3_charuco_{p}_2"
            if not src_ds.is_dir():
                sys.exit(f"source dataset missing: {src_ds}")

    blend_args = dict(
        contrast=args.blend_contrast,
        brightness_offset=args.blend_brightness_offset,
        saturation=args.blend_saturation,
        blur_kernel=args.blend_blur_kernel,
        noise_sigma=args.blend_noise_sigma,
        warmth=args.blend_warmth,
    )
    global_lift_args = dict(
        gain=args.global_lift_gain,
        offset=args.global_lift_offset,
        warmth=args.global_lift_warmth,
    )

    # Fields shared by every WorkerArgs in both the sharded and unsharded paths.
    # The sharded path adds shard_index/shard_count/is_shard/push_to_hub overrides.
    base_worker_kwargs = dict(
        n_configs=args.n_configs_per_dataset,
        source_root=str(args.source_root),
        out_root=str(args.out_root),
        celebrity_jsons=[str(jp) for jp in celebrity_jsons],
        blend_args=blend_args,
        global_lift_args=global_lift_args,
        vcodec=args.vcodec,
        overwrite=args.overwrite,
    )

    shards = max(1, int(args.shards_per_dataset))
    if args.balanced and shards > 1:
        sys.exit(
            "--balanced and --shards-per-dataset>1 are not yet compatible "
            "(the sharding orchestrator builds per-(celeb,position) WorkerArgs; "
            "balanced datasets are keyed by celeb only and have a different "
            "naming scheme). Run balanced with --shards-per-dataset=1."
        )
    t_global = time.time()
    if shards == 1:
        # Legacy path — byte-identical to pre-sharding behavior.
        worker_args_list = [
            WorkerArgs(
                target_celeb=c,
                target_position=p,
                push_to_hub=args.push_to_hub,
                hub_org=args.hub_org,
                output_suffix=args.output_suffix,
                **base_worker_kwargs,
            )
            for c, p in targets
        ]
        print(f"Generating {len(worker_args_list)} datasets "
              f"with {args.n_workers} workers  "
              f"(suffix={args.output_suffix!r}, n_configs={args.n_configs_per_dataset})")
        if args.n_workers == 1:
            results = [_worker_wrapper(wa) for wa in worker_args_list]
        else:
            with Pool(processes=args.n_workers) as pool:
                results = pool.map(_worker_wrapper, worker_args_list)
    else:
        # Sharded + merged path.
        results = _run_with_sharding(
            targets=targets,
            shards_per_dataset=shards,
            n_workers=args.n_workers,
            base_worker_args_kwargs=base_worker_kwargs,
            push_to_hub=args.push_to_hub,
            hub_org=args.hub_org,
            keep_shards=args.keep_shards,
            output_suffix=args.output_suffix,
            out_root=args.out_root,
        )
    elapsed_global = time.time() - t_global

    # Summary report
    print("\n" + "=" * 72)
    print(f"SUMMARY — total elapsed {elapsed_global / 60:.1f} min")
    print("=" * 72)
    total_eps = total_frames = total_mb = 0
    for r in results:
        if r.get("skipped"):
            print(f"  [SKIP] {r['name']}  ({r.get('reason') or r.get('error')})")
            continue
        print(f"  [OK]   {r['name']:<48s} eps={r['n_episodes']:3d} "
              f"frames={r['n_frames']:6d} disk={r['disk_mb']:.1f} MB  "
              f"({r['elapsed_s']:.0f}s)"
              + (f"  -> {r['hub_url']}" if "hub_url" in r else ""))
        total_eps += r["n_episodes"]
        total_frames += r["n_frames"]
        total_mb += r["disk_mb"]
    print(f"\nTotal: {total_eps} episodes  {total_frames} frames  "
          f"{total_mb / 1024:.2f} GB on disk")


if __name__ == "__main__":
    main()
