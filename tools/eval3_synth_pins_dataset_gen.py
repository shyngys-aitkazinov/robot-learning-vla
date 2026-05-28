#!/usr/bin/env python3
"""Pins-pool synthetic LeRobot v3.0 dataset generator for Eval 3 scale-up.

Sibling of ``tools/eval3_synth_dataset_gen.py``. Instead of the fixed 3-celeb
TOY pool, this generator samples DISTRACTOR celebrities from a much larger
Pins-Face-Recognition pool (top-30 / top-50 / full-105), producing 90-315
synthetic LeRobot v3.0 datasets ready for SmolVLA training.

## Sampling shape (structured)

For each (target_celeb, target_position):
    for each of the N = max_photos_per_celeb target_photos:
        for each of M = distractors_per_target_photo distractor scenes:
            pick 2 random distractor celebs from the pool (excluding target),
            pick 1 random photo for each (from their first N photos),
            pick a random swap flag,
            -> yield one episode

Total configs/dataset = N * M (default 10 * 50 = 500).

Coverage: every target_photo is GUARANTEED to appear exactly M times in the
output, paired with M distinct random distractor scenes. Distractor pool
per slot (top-30): C(29, 2) x 10 x 10 x 2 ~= 81,200 possible scenes; sampling
50 of these is essentially all unique.

## Output naming

``dataset_v3_synth_<output_suffix>_<celeb_slug>_<position>_2``

e.g. ``dataset_v3_synth_pins30_anne_hathaway_left_2``. Output suffix is
REQUIRED so pins runs don't collide with the existing TOY-only runs.

## Parallelism

One worker process per output dataset (LeRobotDataset writer isn't
multi-writer safe). Default ``--n-workers = min(os.cpu_count(), n_datasets)``.

## Usage

    # Dry-run plan for top-30:
    python tools/eval3_synth_pins_dataset_gen.py --dry-run

    # Smoke (1 dataset, M=5 -> 50 episodes, ~5 min):
    python tools/eval3_synth_pins_dataset_gen.py \\
        --target-celebs taylor_swift --target-positions left \\
        --distractors-per-target-photo 5 --overwrite

    # Full top-30 sweep on Brev with HF push:
    EVAL3_PINS_WORKERS=90 EVAL3_PINS_PUSH_TO_HUB=1 \\
      ./scripts/run_eval3_synth_pins_dataset_gen.sh
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import traceback
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

# Reuse compose pipeline from the demo
from eval3_charuco_episode_demo import (  # noqa: E402
    _apply_global_lift,
    _apply_table_blend,
    _center_crop_to_aspect,
    RedHSVMasker,
    compose_multi,
    lock_homographies_multi,
)
from eval3_charuco_check import build_board  # noqa: E402

# Reuse data plumbing + helpers from the TOY generator
from eval3_synth_dataset_gen import (  # noqa: E402
    BOARD_CHROMA_MM, BOARD_DICT, BOARD_H_MM, BOARD_MARKER_RATIO,
    BOARD_SQUARES_X, BOARD_SQUARES_Y, BOARD_SQUARE_MM, BOARD_W_MM,
    CANONICAL_NAME as TOY_CANONICAL_NAME,
    Config, FEATURES_SCHEMA, HSV_HI, HSV_LO,
    POSITIONS, POSITION_TO_BOARD_IDX,
    SourceEpisodeMeta, _board_layout, _dir_size_mb, _push_to_hub,
    load_source_episodes, read_episode_frames,
)

try:
    import cv2.aruco as aruco
except ImportError as e:
    sys.exit(f"opencv-contrib-python required (cv2.aruco missing): {e}")


# ---------------------------------------------------------------------------
# Pool loading + target celeb selection
# ---------------------------------------------------------------------------

def load_pool(pool_json: Path) -> dict[str, dict]:
    """Read a Pins-style JSON and return {slug: celeb_dict (with name, dir, jpgs)}.

    If the JSON entry has a ``quality_photos`` field (produced by
    ``tools/pins_quality_filter.py``), the photo list comes from there
    SORTED BEST-FIRST. Otherwise we fall back to globbing ``c["dir"]``
    in lexical order.
    """
    meta = json.loads(pool_json.read_text(encoding="utf-8"))
    pool: dict[str, dict] = {}
    used_quality = False
    for c in meta["celebrities"]:
        slug = c["slug"]
        if "quality_photos" in c and isinstance(c["quality_photos"], list) and len(c["quality_photos"]) > 0:
            jpgs = [Path(p) for p in c["quality_photos"]]
            used_quality = True
        else:
            jpgs = sorted(Path(c["dir"]).glob("*.jpg"))
        if not jpgs:
            print(f"[load_pool] WARN no JPGs for {slug}; skipping", flush=True)
            continue
        pool[slug] = {
            "name": c["name"],
            "slug": slug,
            "dir": c["dir"],
            "jpgs": jpgs,
            "n_images": len(jpgs),
        }
    if len(pool) < 3:
        sys.exit(f"--pool-json {pool_json} has only {len(pool)} usable celebs (need >=3 for distractor pairs)")
    if used_quality:
        print(f"[load_pool] using quality_photos field (best-first ranking) from {pool_json}", flush=True)
    return pool


def resolve_target_celebs(requested: str, pool: dict[str, dict]) -> list[str]:
    """'all' -> every slug in pool (sorted); else comma-list validated against pool."""
    if requested.strip().lower() == "all":
        return sorted(pool.keys())
    requested_slugs = [s.strip() for s in requested.split(",") if s.strip()]
    missing = [s for s in requested_slugs if s not in pool]
    if missing:
        sys.exit(f"--target-celebs include slugs not in pool: {missing}. "
                 f"Available: {sorted(pool.keys())[:10]}...")
    return requested_slugs


def canonical_task_for(slug: str, pool: dict[str, dict]) -> str:
    """Build the SmolVLA-canonical task string for a celeb. Falls back gracefully
    if the celeb is unknown (passes the augmenter untouched)."""
    name = pool[slug]["name"]
    return f"Place the coke on {name}"


# ---------------------------------------------------------------------------
# Structured sampling: iterate target_photos x sample M distractor scenes
# ---------------------------------------------------------------------------

def sample_pins_config_grid(
    target_celeb: str,
    target_position: str,
    pool: dict[str, dict],
    max_photos_per_celeb: int,
    distractors_per_target_photo: int,
    rng: random.Random,
) -> list[Config]:
    """Structured sampling: enumerate ALL target_photos x sample M distractor scenes each.

    Returns list of (N * M) Config instances where:
      N = min(max_photos_per_celeb, target_celeb's actual photo count)
      M = distractors_per_target_photo

    Each Config picks 2 random OTHER celebs from the pool (excluding target),
    1 random photo for each (from their first max_photos_per_celeb capped to
    the celeb's actual count), and a random swap. Deterministic given rng.
    """
    if target_celeb not in pool:
        sys.exit(f"target_celeb={target_celeb!r} not in pool ({sorted(pool.keys())[:5]}...)")
    other_slugs = [s for s in pool if s != target_celeb]
    if len(other_slugs) < 2:
        sys.exit(f"pool too small: need >=2 OTHER celebs, got {len(other_slugs)}")

    n_target = min(max_photos_per_celeb, pool[target_celeb]["n_images"])
    configs: list[Config] = []
    for tp in range(n_target):
        for _ in range(distractors_per_target_photo):
            d1, d2 = rng.sample(other_slugs, 2)
            n1 = min(max_photos_per_celeb, pool[d1]["n_images"])
            n2 = min(max_photos_per_celeb, pool[d2]["n_images"])
            p1 = rng.randrange(n1)
            p2 = rng.randrange(n2)
            swap = rng.choice([False, True])
            configs.append(Config(
                target_celeb=target_celeb,
                target_position=target_position,
                target_photo_idx=tp,
                other1_celeb=d1,
                other1_photo_idx=p1,
                other2_celeb=d2,
                other2_photo_idx=p2,
                other_swap=swap,
            ))
    return configs


# ---------------------------------------------------------------------------
# Per-worker tile cache (pool-aware, lazy)
# ---------------------------------------------------------------------------

class PinsTileCache:
    """Lazy per-(slug, photo_idx) tile cache for an arbitrarily large celeb pool.

    Photos are loaded + center-cropped to the board aspect + resized to the
    internal warp resolution + table-blended on first access; subsequent
    accesses hit the cache. Each entry is ~target_w * target_h * 3 bytes
    (1140 x 1600 x 3 ~= 5.5 MB). For top-30 with max_photos=10: at most
    30 * 10 = 300 tiles per worker = ~1.6 GB worst case (unlikely all hit).
    """

    def __init__(
        self,
        pool: dict[str, dict],
        max_photos_per_celeb: int,
        blend_args: dict,
        target_w_px: int,
        target_h_px: int,
        board_aspect: float,
        rng_seed: int,
    ):
        self._pool = pool
        self._max_photos = max_photos_per_celeb
        self._blend_args = blend_args
        self._tw = target_w_px
        self._th = target_h_px
        self._board_aspect = board_aspect
        self._rng = np.random.default_rng(rng_seed)
        self._cache: dict[tuple[str, int], np.ndarray] = {}

    def get(self, slug: str, photo_idx: int) -> np.ndarray:
        if slug not in self._pool:
            sys.exit(f"PinsTileCache: unknown celeb {slug!r}")
        key = (slug, photo_idx)
        if key in self._cache:
            return self._cache[key]
        jpgs = self._pool[slug]["jpgs"]
        capped = jpgs[: self._max_photos]
        if not (0 <= photo_idx < len(capped)):
            sys.exit(f"PinsTileCache: photo_idx={photo_idx} out of range "
                     f"for {slug} (have {len(capped)} after cap={self._max_photos})")
        raw = cv2.imread(str(capped[photo_idx]))
        if raw is None:
            sys.exit(f"PinsTileCache: failed to read {capped[photo_idx]}")
        cropped = _center_crop_to_aspect(raw, self._board_aspect)
        resized = cv2.resize(cropped, (self._tw, self._th), interpolation=cv2.INTER_AREA)
        tile = _apply_table_blend(resized, rng=self._rng, **self._blend_args)
        self._cache[key] = tile
        return tile


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

@dataclass
class WorkerArgs:
    """Picklable args for one (target_celeb, target_position) worker."""
    target_celeb: str
    target_position: str
    pool_json: str                # path to pool JSON (reloaded inside the worker)
    max_photos_per_celeb: int
    distractors_per_target_photo: int
    source_root: str
    out_root: str
    output_suffix: str            # REQUIRED for pins (e.g. 'pins30')
    blend_args: dict
    global_lift_args: dict
    vcodec: str
    push_to_hub: bool
    hub_org: str
    overwrite: bool
    seed: int
    # Source dataset family — defaults to v3 charuco _2 (the original Pins-pool
    # training corpus). Override to point at a different ChArUco capture, e.g.
    # source_prefix='dataset_v5_charuko_' + source_suffix='_full' for the v5
    # cross-product captures.
    source_prefix: str = "dataset_v3_charuco_"
    source_suffix: str = "_2"
    # Output dataset name template — ``{output_prefix}{output_suffix}_{celeb}_{pos}{output_postfix}``.
    # Defaults reproduce the historical pins layout
    # ``dataset_v3_synth_<suffix>_<celeb>_<pos>_2``.
    output_prefix: str = "dataset_v3_synth_"
    output_postfix: str = "_2"


def generate_one_pins_dataset(args: WorkerArgs) -> dict:
    """Build one synthetic Pins-pool dataset end-to-end."""
    t0 = time.time()
    pool = load_pool(Path(args.pool_json))
    pool_slugs = sorted(pool.keys())

    target_celeb = args.target_celeb
    target_position = args.target_position
    task_str = canonical_task_for(target_celeb, pool)
    out_name = f"{args.output_prefix}{args.output_suffix}_{target_celeb}_{target_position}{args.output_postfix}"
    out_dir = Path(args.out_root) / out_name

    if out_dir.exists():
        if args.overwrite:
            import shutil
            shutil.rmtree(out_dir)
        else:
            return {"name": out_name, "skipped": True,
                    "reason": f"output dir already exists: {out_dir}"}

    print(f"[{out_name}] START — task={task_str!r}", flush=True)

    # ---- 1. Configs (deterministic per dataset seed) ---------------------
    # Per-dataset seed = base seed XOR hash(celeb, position) so each output
    # is reproducible and independent.
    ds_seed = args.seed ^ (abs(hash((target_celeb, target_position))) % (2**31))
    cfg_rng = random.Random(ds_seed)
    configs = sample_pins_config_grid(
        target_celeb, target_position, pool,
        args.max_photos_per_celeb, args.distractors_per_target_photo,
        cfg_rng,
    )
    print(f"[{out_name}] {len(configs)} configs "
          f"(N={args.max_photos_per_celeb} target_photos x "
          f"M={args.distractors_per_target_photo} distractors)", flush=True)

    # ---- 2. Source episodes + joints -------------------------------------
    src_mp4, src_episodes, src_action, src_state = load_source_episodes(
        Path(args.source_root), target_position,
        source_suffix=args.source_suffix, source_prefix=args.source_prefix,
    )
    print(f"[{out_name}] source: {src_mp4.name}  {len(src_episodes)} eps, "
          f"{len(src_action)} frames total", flush=True)

    # ---- 3. Board + detectors + tile cache + masker ----------------------
    target_w_px = int(round(BOARD_W_MM * 10))
    target_h_px = int(round(BOARD_H_MM * 10))
    board_aspect = BOARD_H_MM / BOARD_W_MM
    tile_cache = PinsTileCache(
        pool=pool,
        max_photos_per_celeb=args.max_photos_per_celeb,
        blend_args=args.blend_args,
        target_w_px=target_w_px,
        target_h_px=target_h_px,
        board_aspect=board_aspect,
        rng_seed=ds_seed,
    )
    board, dictionary = build_board(
        BOARD_SQUARES_X, BOARD_SQUARES_Y, BOARD_SQUARE_MM, BOARD_MARKER_RATIO, BOARD_DICT,
    )
    detector = aruco.ArucoDetector(dictionary, aruco.DetectorParameters())
    char_detector = aruco.CharucoDetector(board)
    masker = RedHSVMasker(HSV_LO, HSV_HI)

    # ---- 4. Homography cache per source episode --------------------------
    h_cache: dict[int, list[np.ndarray | None]] = {}
    def get_homographies(src_ep_idx: int, frame0: np.ndarray):
        if src_ep_idx not in h_cache:
            Hs, _ = lock_homographies_multi(frame0, detector, char_detector, board, n_boards=3)
            h_cache[src_ep_idx] = Hs
        return h_cache[src_ep_idx]

    # ---- 5. Create output dataset ----------------------------------------
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from copy import deepcopy
    repo_id = f"{args.hub_org}/{out_name}" if args.push_to_hub else out_name
    dataset = LeRobotDataset.create(
        repo_id=repo_id, fps=30,
        features=deepcopy(FEATURES_SCHEMA),
        root=out_dir,
        robot_type="so_follower",
        use_videos=True,
        vcodec=args.vcodec,
        streaming_encoding=True,
    )

    # ---- 6. Per-config compose + write ------------------------------------
    target_idx = POSITION_TO_BOARD_IDX[target_position]
    fps = 30.0
    n_source = len(src_episodes)
    n_eps_done = 0
    n_frames_done = 0
    for ci, cfg in enumerate(configs):
        # Cycle source episodes 1:1 across configs
        src_idx = ci % n_source
        src = src_episodes[src_idx]
        frames = read_episode_frames(src_mp4, src, fps)
        actions = src_action[src.from_frame_global: src.to_frame_global]
        states = src_state[src.from_frame_global: src.to_frame_global]
        assert len(frames) == len(actions) == len(states) == src.length

        # Assemble board layout in (left, middle, right) order
        layout = _board_layout(target_position, cfg.other1_celeb, cfg.other2_celeb, cfg.other_swap)
        layout[target_idx] = target_celeb
        photo_idx_per_slot: list[int] = [-1, -1, -1]
        photo_idx_per_slot[target_idx] = cfg.target_photo_idx
        other_slots = [i for i in range(3) if i != target_idx]
        if cfg.other_swap:
            other_slots = other_slots[::-1]
        photo_idx_per_slot[other_slots[0]] = cfg.other1_photo_idx
        photo_idx_per_slot[other_slots[1]] = cfg.other2_photo_idx
        tiles = [tile_cache.get(layout[i], photo_idx_per_slot[i]) for i in range(3)]

        Hs = get_homographies(src_idx, frames[0])
        if all(H is None for H in Hs):
            print(f"[{out_name}] WARN cfg{ci}: src ep{src.episode_index} no board locks; skipping",
                  flush=True)
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
                "observation.state": states[fi],
                "action": actions[fi],
                "task": task_str,
            })
        dataset.save_episode()
        n_eps_done += 1
        n_frames_done += src.length
        if n_eps_done % 25 == 0 or n_eps_done == len(configs):
            elapsed = time.time() - t0
            fps_compose = n_frames_done / max(elapsed, 1e-6)
            eta = (len(configs) - n_eps_done) * elapsed / max(n_eps_done, 1)
            print(f"[{out_name}] {n_eps_done}/{len(configs)} eps "
                  f"({n_frames_done} fr, {fps_compose:.1f} fps, eta {eta / 60:.1f} min)",
                  flush=True)

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
    if args.push_to_hub:
        info = _push_to_hub(out_dir, repo_id)
        summary["hub_url"] = info["hub_url"]
        summary["upload_elapsed_s"] = info["elapsed_s"]

    print(f"[{out_name}] DONE — {summary}", flush=True)
    return summary


# ---------------------------------------------------------------------------
# CLI / dispatcher
# ---------------------------------------------------------------------------

def _dry_run(
    pool: dict[str, dict],
    targets: list[str],
    positions: list[str],
    max_photos_per_celeb: int,
    distractors_per_target_photo: int,
    output_suffix: str,
    source_root: Path,
    source_prefix: str = "dataset_v3_charuco_",
    source_suffix: str = "_2",
    output_prefix: str = "dataset_v3_synth_",
    output_postfix: str = "_2",
) -> None:
    n_configs = max_photos_per_celeb * distractors_per_target_photo
    n_datasets = len(targets) * len(positions)
    # Estimate per-source-position episode avg frame count for disk estimate
    source_avg_frames = 500  # measured average; refined below if sources exist
    src_avg_by_pos: dict[str, float] = {}
    for pos in positions:
        src_ds = source_root / f"{source_prefix}{pos}{source_suffix}"
        if src_ds.is_dir():
            try:
                info = json.loads((src_ds / "meta/info.json").read_text())
                src_avg_by_pos[pos] = info["total_frames"] / max(info["total_episodes"], 1)
            except Exception:
                src_avg_by_pos[pos] = source_avg_frames
    avg_frames_per_ep = (sum(src_avg_by_pos.values()) / len(src_avg_by_pos)) if src_avg_by_pos else source_avg_frames
    total_eps = n_datasets * n_configs
    total_frames = int(total_eps * avg_frames_per_ep)
    # ~3 MB per ~500-frame episode at libx264 -> 6 KB/frame
    disk_mb = total_frames * 6 / 1024  # MB
    suffix = output_suffix or "(empty)"

    print("=" * 80)
    print(f"DRY RUN — Pins-pool synthetic dataset generator")
    print("=" * 80)
    print(f"  pool celebs       : {len(pool)}  (post-load)")
    print(f"  target celebs     : {len(targets)} (e.g. {targets[:5]}{'...' if len(targets)>5 else ''})")
    print(f"  target positions  : {positions}")
    print(f"  N (target photos) : {max_photos_per_celeb}")
    print(f"  M (distractors)   : {distractors_per_target_photo}")
    print(f"  configs/dataset   : {n_configs}  (= N * M)")
    print(f"  output suffix     : {suffix!r}")
    print(f"  total datasets    : {n_datasets}")
    print(f"  total episodes    : {total_eps:,}")
    print(f"  avg frames/ep     : {int(avg_frames_per_ep)}  (estimated from "
          f"{', '.join(f'{p}={int(src_avg_by_pos[p])}' for p in src_avg_by_pos) or 'fallback'})")
    print(f"  total frames      : ~{total_frames:,}")
    print(f"  est. disk (libx264): ~{disk_mb / 1024:.1f} GB")
    print()
    # Per-celeb photo counts (post-cap)
    capped_counts = {s: min(pool[s]["n_images"], max_photos_per_celeb) for s in targets[:5]}
    print(f"  sample post-cap photo counts: {capped_counts}")
    if max(capped_counts.values()) < max_photos_per_celeb:
        print(f"  WARN: some targets have fewer photos than --max-photos-per-celeb={max_photos_per_celeb}")
    print()
    # Sample dataset names
    name_tpl = f"{output_prefix}{{suf}}_{{c}}_{{p}}{output_postfix}"
    sample_names = [
        name_tpl.format(suf=suffix if output_suffix else "SUFFIX", c=c, p=p)
        for c in targets[:3] for p in positions
    ]
    if len(targets) > 3:
        sample_names.append("...")
        sample_names.append(name_tpl.format(
            suf=suffix if output_suffix else "SUFFIX",
            c=targets[-1], p=positions[-1],
        ))
    print(f"  output dataset names:")
    for n in sample_names:
        print(f"    {n}")
    # Source check
    print()
    for pos in positions:
        src_name = f"{source_prefix}{pos}{source_suffix}"
        src_ds = source_root / src_name
        ok = "OK" if src_ds.is_dir() else "MISSING"
        print(f"  source {src_name}: [{ok}]")
    print()


def _worker_wrapper(args: WorkerArgs) -> dict:
    try:
        return generate_one_pins_dataset(args)
    except SystemExit as e:
        return {"name": f"{args.output_suffix}_{args.target_celeb}_{args.target_position}",
                "error": f"SystemExit({e})", "skipped": True}
    except Exception as e:
        return {"name": f"{args.output_suffix}_{args.target_celeb}_{args.target_position}",
                "error": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc(), "skipped": True}


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Prefer the quality-filtered JSON when present (auto-detect for friendlier UX).
    _DEFAULT_POOL = Path("datasets/pins-face-recognition-top30-quality.json")
    if not _DEFAULT_POOL.is_file():
        _DEFAULT_POOL = Path("datasets/pins-face-recognition-top30.json")
    ap.add_argument("--pool-json", type=Path, default=_DEFAULT_POOL,
                    help=f"Pool JSON (default: {_DEFAULT_POOL}). "
                         f"Use the *-quality.json variant produced by "
                         f"tools/pins_quality_filter.py to get best-quality "
                         f"photos sorted top-first.")
    ap.add_argument("--target-celebs", default="all",
                    help="Comma-separated target slugs OR literal 'all' (default).")
    ap.add_argument("--target-positions", default=",".join(POSITIONS),
                    help="Comma-separated positions (default: left,middle,right).")
    ap.add_argument("--max-photos-per-celeb", type=int, default=10,
                    help="N — cap on photos per celeb (default 10). Truncates the celeb's sorted JPG list.")
    ap.add_argument("--distractors-per-target-photo", type=int, default=50,
                    help="M — random distractor scenes per target_photo (default 50). "
                         "Total configs/dataset = N * M.")
    ap.add_argument("--output-suffix", default="pins30",
                    help="REQUIRED for pins runs (e.g. 'pins30', 'pins50', 'pinsfull'). "
                         "Becomes part of the dataset name to avoid colliding with TOY runs.")
    ap.add_argument("--source-root", type=Path, default=Path("datasets"),
                    help="Root containing dataset_v3_charuco_{left,middle,right}_2.")
    ap.add_argument("--out-root", type=Path, default=Path("datasets"),
                    help="Root for generated dataset_v3_synth_<suffix>_<celeb>_<pos>_2 dirs.")
    ap.add_argument("--vcodec", default="h264",
                    help="Video codec (h264 default, libsvtav1 for ~30%% smaller files).")
    ap.add_argument("--n-workers", type=int, default=0,
                    help="Worker processes (one per output dataset). "
                         "0 (default) = min(os.cpu_count(), n_datasets).")
    ap.add_argument("--push-to-hub", action="store_true",
                    help="Upload each dataset to HF Hub + create v3.0 tag after generation.")
    ap.add_argument("--hub-org", default="RobotLearningVLA",
                    help="HF org for repo_id (default RobotLearningVLA).")
    ap.add_argument("--overwrite", action="store_true",
                    help="Delete existing output dirs before writing.")
    ap.add_argument("--seed", type=int, default=42,
                    help="Base RNG seed (XOR'd with hash(celeb, position) per dataset).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print plan + disk estimate and exit.")

    # Per-tile blend defaults (pass-through to _apply_table_blend)
    ap.add_argument("--blend-contrast", type=float, default=0.88)
    ap.add_argument("--blend-brightness-offset", type=int, default=15)
    ap.add_argument("--blend-saturation", type=float, default=0.9)
    ap.add_argument("--blend-blur-kernel", type=int, default=3)
    ap.add_argument("--blend-noise-sigma", type=float, default=3.0)
    ap.add_argument("--blend-warmth", type=float, default=1.0)

    # Global-lift defaults (pass-through)
    ap.add_argument("--global-lift-gain", type=float, default=1.25)
    ap.add_argument("--global-lift-offset", type=float, default=10.0)
    ap.add_argument("--global-lift-warmth", type=float, default=1.06)

    # ChArUco source / output dataset name shape -----------------------------
    ap.add_argument("--source-prefix", default="dataset_v3_charuco_",
                    help="Prefix of the source dataset dir under --source-root. "
                         "Default 'dataset_v3_charuco_'. Pass 'dataset_v5_charuko_' "
                         "for the v5 cross-product captures (note the spelling).")
    ap.add_argument("--source-suffix", default="_2",
                    help="Suffix appended after the position in the source dir name. "
                         "Default '_2' (v3 newer corpus); pass '_full' for the v5 "
                         "cross-product captures.")
    ap.add_argument("--output-prefix", default="dataset_v3_synth_",
                    help="Prefix of generated dataset dir names. Default "
                         "'dataset_v3_synth_'. Pass 'dataset_v5_synth_' to mark "
                         "v5-derived outputs.")
    ap.add_argument("--output-postfix", default="_2",
                    help="Postfix appended after the position in the output dir name. "
                         "Default '_2'; pass '_full' to match v5 source naming.")

    args = ap.parse_args()

    if not args.pool_json.is_file():
        sys.exit(f"--pool-json {args.pool_json} not found")
    pool = load_pool(args.pool_json)
    target_celebs = resolve_target_celebs(args.target_celebs, pool)
    positions = [s.strip() for s in args.target_positions.split(",") if s.strip()]
    if not args.output_suffix:
        sys.exit("--output-suffix is REQUIRED for pins runs (e.g. 'pins30')")

    if args.dry_run:
        _dry_run(pool, target_celebs, positions,
                 args.max_photos_per_celeb, args.distractors_per_target_photo,
                 args.output_suffix, args.source_root,
                 source_prefix=args.source_prefix, source_suffix=args.source_suffix,
                 output_prefix=args.output_prefix, output_postfix=args.output_postfix)
        return

    # Validate sources before launching workers
    for pos in positions:
        src_ds = args.source_root / f"{args.source_prefix}{pos}{args.source_suffix}"
        if not src_ds.is_dir():
            sys.exit(f"source dataset missing: {src_ds}")

    # Pack worker args
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
    worker_args_list = [
        WorkerArgs(
            target_celeb=c,
            target_position=p,
            pool_json=str(args.pool_json),
            max_photos_per_celeb=args.max_photos_per_celeb,
            distractors_per_target_photo=args.distractors_per_target_photo,
            source_root=str(args.source_root),
            out_root=str(args.out_root),
            output_suffix=args.output_suffix,
            blend_args=blend_args,
            global_lift_args=global_lift_args,
            vcodec=args.vcodec,
            push_to_hub=args.push_to_hub,
            hub_org=args.hub_org,
            overwrite=args.overwrite,
            seed=args.seed,
            source_prefix=args.source_prefix,
            source_suffix=args.source_suffix,
            output_prefix=args.output_prefix,
            output_postfix=args.output_postfix,
        )
        for c in target_celebs for p in positions
    ]

    n_datasets = len(worker_args_list)
    if args.n_workers <= 0:
        n_workers = min(os.cpu_count() or 1, n_datasets)
    else:
        n_workers = min(args.n_workers, n_datasets)

    print(f"=== Pins-pool synth gen ===")
    print(f"  pool         : {args.pool_json} ({len(pool)} celebs)")
    print(f"  targets      : {len(target_celebs)} celebs x {len(positions)} positions = {n_datasets} datasets")
    print(f"  configs/ds   : {args.max_photos_per_celeb * args.distractors_per_target_photo} "
          f"(N={args.max_photos_per_celeb} x M={args.distractors_per_target_photo})")
    print(f"  workers      : {n_workers}")
    print(f"  output suffix: {args.output_suffix}")
    print(f"  push to hub  : {args.push_to_hub}")
    print()

    t_global = time.time()
    if n_workers == 1:
        results = [_worker_wrapper(wa) for wa in worker_args_list]
    else:
        with Pool(processes=n_workers) as pool_:
            results = pool_.map(_worker_wrapper, worker_args_list)
    elapsed = time.time() - t_global

    print("\n" + "=" * 80)
    print(f"SUMMARY — total elapsed {elapsed / 60:.1f} min")
    print("=" * 80)
    total_eps = total_frames = total_mb = 0
    for r in results:
        if r.get("skipped"):
            print(f"  [SKIP] {r['name']}  ({r.get('reason') or r.get('error')})")
            continue
        print(f"  [OK]   {r['name']:<60s} eps={r['n_episodes']:4d}  "
              f"frames={r['n_frames']:7d}  disk={r['disk_mb']:.1f} MB  "
              f"({r['elapsed_s']:.0f}s)"
              + (f"  -> {r['hub_url']}" if "hub_url" in r else ""))
        total_eps += r["n_episodes"]
        total_frames += r["n_frames"]
        total_mb += r["disk_mb"]
    print(f"\nTotal: {total_eps} episodes  {total_frames} frames  "
          f"{total_mb / 1024:.2f} GB on disk")


if __name__ == "__main__":
    main()
