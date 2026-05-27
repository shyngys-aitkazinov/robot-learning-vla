#!/usr/bin/env python3
"""RAG-variant synthetic dataset generator — drop-in replacement for
tools/eval3_synth_pins_dataset_gen.py with three extra CLI knobs:

    --source-prefix   PREFIX   (default: dataset_v5_charuko)
    --source-suffix   SUFFIX   (default: _1)
    --source-hub-org  ORG      (default: RobotLearningVLA)

These control which HuggingFace source recordings are used as background
scenes for ChArUco compositing.  The constructed dataset name is:

    {source_prefix}_{position}{source_suffix}
    e.g. dataset_v5_charuko_left_1  (RobotLearningVLA/dataset_v5_charuko_left_1)

If the local dir is absent and source_hub_org is set, the dataset is
downloaded from Hub automatically before workers start.

All other flags are identical to tools/eval3_synth_pins_dataset_gen.py;
see that file's --help for the full reference.

Usage (called by gen_rag_synth.sh):
    python scripts/eval3_rag/synth_pins_gen_rag.py \\
        --pool-json datasets/rag_pool.json \\
        --target-celebs elon_musk,marc_pollefeys \\
        --target-positions left,middle,right \\
        --max-photos-per-celeb 5 \\
        --distractors-per-target-photo 20 \\
        --output-suffix rag1 \\
        --source-prefix dataset_v5_charuko \\
        --source-suffix _1 \\
        --source-hub-org RobotLearningVLA \\
        --push-to-hub --hub-org yukk1
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from multiprocessing import Pool
from pathlib import Path

# ── path setup ────────────────────────────────────────────────────────────────
_SCRIPTS = Path(__file__).resolve().parent.parent   # scripts/
_ROOT    = _SCRIPTS.parent                          # robot-learning-vla/
_TOOLS   = _ROOT / "tools"
for _p in (_SCRIPTS, _TOOLS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# ── import from the original generator ───────────────────────────────────────
from eval3_synth_pins_dataset_gen import (  # noqa: E402
    WorkerArgs,
    PinsTileCache,
    resolve_target_celebs,
)
from eval3_synth_dataset_gen import (       # noqa: E402
    Config,
    POSITIONS,
    _maybe_download_source,
    load_source_episodes,
)


def load_pool(pool_json: Path) -> dict[str, dict]:
    """Like eval3_synth_pins_dataset_gen.load_pool but safe for rag_pool.json.

    rag_pool.json (built by build_rag_pool_json.py) has no 'dir' field —
    it uses quality_photos instead.  The original load_pool unconditionally
    accesses c["dir"] even when the quality_photos branch is taken, causing
    a KeyError.  This version uses c.get("dir", "") as the fallback.
    """
    meta = json.loads(pool_json.read_text(encoding="utf-8"))
    pool: dict[str, dict] = {}
    used_quality = False
    for c in meta["celebrities"]:
        slug = c["slug"]
        # Skip TOY (in-distribution) celebrities — they have real robot recordings
        # and should NOT be regenerated as synthetic data.
        if c.get("held_out", False):
            continue
        if "quality_photos" in c and isinstance(c["quality_photos"], list) and c["quality_photos"]:
            jpgs = [Path(p) for p in c["quality_photos"]]
            used_quality = True
        elif "dir" in c:
            jpgs = sorted(Path(c["dir"]).glob("*.jpg"))
        else:
            print(f"[load_pool] WARN {slug!r}: no quality_photos or dir; skipping", flush=True)
            continue
        if not jpgs:
            print(f"[load_pool] WARN no JPGs for {slug}; skipping", flush=True)
            continue
        pool[slug] = {
            "name": c["name"],
            "slug": slug,
            "dir": c.get("dir", ""),
            "jpgs": jpgs,
            "n_images": len(jpgs),
        }
    if len(pool) < 3:
        sys.exit(
            f"--pool-json {pool_json} has only {len(pool)} usable celebrities "
            f"(need >=3 for distractor pairs)"
        )
    if used_quality:
        print(f"[load_pool] using quality_photos (best-first) from {pool_json}", flush=True)
    return pool


# ── episode normalisation + augmented tile cache ──────────────────────────────
#
# Problem: celebrities with fewer than max_photos_per_celeb photos generate
# proportionally fewer training episodes (e.g. marc_pollefeys with 1 photo
# gets 5× fewer episodes than scarlett_johansson with 5).  This biases the
# model toward over-represented celebrities.
#
# Fix:
#   1. _normalized_config_grid — always generates max_photos * distractors
#      episodes per celebrity by cycling photo indices modulo actual count.
#   2. AugmentedPinsTileCache — for indices ≥ actual_count, returns a
#      deterministically augmented variant of the base photo (flip, brightness)
#      so the board shows diverse-but-recognisable appearances of the same face.

# Augmentation levels (deterministic, index-stable):
#   level 0 = flip horizontally           (different pose, same identity)
#   level 1 = brightness × 1.10           (lighting variation)
#   level 2 = flip + brightness × 1.10    (combined)
#   level 3 = brightness × 0.88           (darker)
_AUG_FLIP       = 0x1  # bit 0: horizontal flip
_AUG_BRIGHT_HI  = 0x2  # bit 1: +10 % brightness
_AUG_BRIGHT_LO  = 0x4  # bit 2: −12 % brightness
_AUG_FLAGS = [
    _AUG_FLIP,
    _AUG_BRIGHT_HI,
    _AUG_FLIP | _AUG_BRIGHT_HI,
    _AUG_BRIGHT_LO,
]


class AugmentedPinsTileCache(PinsTileCache):
    """PinsTileCache that fills missing photo slots with augmented variants.

    When ``photo_idx < n_actual_photos`` the tile is produced exactly as in
    the parent class.  When ``photo_idx >= n_actual_photos`` the cache wraps
    around (``base_idx = photo_idx % n_actual``) and applies a deterministic
    augmentation (flip / brightness) so each synthetic slot looks distinct
    while preserving face identity.
    """

    def get(self, slug: str, photo_idx: int):
        import cv2
        import numpy as np

        jpgs = self._pool[slug]["jpgs"]
        n_actual = min(len(jpgs), self._max_photos)

        if photo_idx < n_actual:
            return super().get(slug, photo_idx)

        # ── Augmentation path ──────────────────────────────────────────
        key = (slug, photo_idx)
        if key in self._cache:
            return self._cache[key]

        base_idx  = photo_idx % n_actual
        aug_level = (photo_idx - n_actual) // n_actual  # 0, 1, 2, …
        flags = _AUG_FLAGS[aug_level % len(_AUG_FLAGS)]

        # Get or compute the base tile via the parent (cached)
        base_tile = super().get(slug, base_idx).copy()

        aug = base_tile
        if flags & _AUG_FLIP:
            aug = cv2.flip(aug, 1)
        if flags & _AUG_BRIGHT_HI:
            aug = np.clip(aug.astype(np.float32) * 1.10, 0, 255).astype(np.uint8)
        elif flags & _AUG_BRIGHT_LO:
            aug = np.clip(aug.astype(np.float32) * 0.88, 0, 255).astype(np.uint8)

        self._cache[(slug, photo_idx)] = aug
        return aug


def _normalized_config_grid(
    target_celeb: str,
    target_position: str,
    pool: dict,
    max_photos_per_celeb: int,
    distractors_per_target_photo: int,
    rng,
) -> list:
    """Like sample_pins_config_grid but always produces max_photos * distractors episodes.

    For celebrities with fewer than max_photos actual photos, photo indices
    ≥ n_actual refer to augmented variants handled by AugmentedPinsTileCache.
    Distractor celebrities always use only their real (non-augmented) photos.

    This guarantees every celebrity contributes an equal episode budget to
    training, preventing over-represented celebrities from dominating learning.
    """
    import random as _random

    other_slugs = [s for s in pool if s != target_celeb]
    if len(other_slugs) < 2:
        raise ValueError(
            f"pool too small for {target_celeb!r}: need ≥2 other celebs, got {len(other_slugs)}"
        )

    # Always iterate over max_photos slots (augmented where needed).
    n_target = max_photos_per_celeb
    configs: list[Config] = []
    for tp in range(n_target):
        for _ in range(distractors_per_target_photo):
            d1, d2 = rng.sample(other_slugs, 2)
            # Distractors use only their actual photos, not augmented slots.
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


# ── patched worker that forwards source_prefix/suffix/hub_org ────────────────

def _patched_generate_one_pins_dataset(args: WorkerArgs) -> dict:
    """Identical to generate_one_pins_dataset but uses source_prefix/suffix/hub_org."""
    import random
    import numpy as np
    import cv2
    try:
        import cv2.aruco as aruco
    except ImportError as e:
        sys.exit(f"opencv-contrib-python required (cv2.aruco missing): {e}")

    from eval3_synth_pins_dataset_gen import (
        canonical_task_for,
    )
    from eval3_synth_dataset_gen import (
        BOARD_CHROMA_MM, BOARD_DICT, BOARD_H_MM, BOARD_MARKER_RATIO,
        BOARD_SQUARES_X, BOARD_SQUARES_Y, BOARD_SQUARE_MM, BOARD_W_MM,
        FEATURES_SCHEMA, HSV_HI, HSV_LO,
        POSITION_TO_BOARD_IDX,
        _board_layout, _dir_size_mb, _push_to_hub,
        read_episode_frames,
    )
    from eval3_charuco_episode_demo import (
        _apply_global_lift, _apply_table_blend, _center_crop_to_aspect,
        RedHSVMasker, compose_multi, lock_homographies_multi,
    )
    from eval3_charuco_check import build_board

    t0 = time.time()
    pool = load_pool(Path(args.pool_json))

    target_celeb   = args.target_celeb
    target_position = args.target_position
    task_str = canonical_task_for(target_celeb, pool)
    out_name = f"dataset_v3_synth_{args.output_suffix}_{target_celeb}_{target_position}_2"
    out_dir  = Path(args.out_root) / out_name

    if out_dir.exists():
        if args.overwrite:
            import shutil
            shutil.rmtree(out_dir)
        else:
            return {"name": out_name, "skipped": True,
                    "reason": f"output dir already exists: {out_dir}"}

    print(f"[{out_name}] START — task={task_str!r}", flush=True)

    ds_seed  = args.seed ^ (abs(hash((target_celeb, target_position))) % (2**31))
    cfg_rng  = random.Random(ds_seed)
    # Normalized grid: always max_photos × distractors episodes per celebrity;
    # celebrities with fewer photos use augmented variants to fill extra slots.
    configs  = _normalized_config_grid(
        target_celeb, target_position, pool,
        args.max_photos_per_celeb, args.distractors_per_target_photo,
        cfg_rng,
    )
    print(f"[{out_name}] {len(configs)} configs", flush=True)

    # ---- source episodes: use the patched source_prefix/suffix/hub_org ----
    src_mp4, src_episodes, src_action, src_state = load_source_episodes(
        Path(args.source_root), target_position,
        source_prefix=args.source_prefix,
        source_suffix=args.source_suffix,
        source_hub_org=args.source_hub_org,
    )
    print(f"[{out_name}] source: {src_mp4.name}  {len(src_episodes)} eps", flush=True)

    target_w_px  = int(round(BOARD_W_MM * 10))
    target_h_px  = int(round(BOARD_H_MM * 10))
    board_aspect = BOARD_H_MM / BOARD_W_MM
    # AugmentedPinsTileCache handles photo_idx ≥ n_actual by returning a
    # deterministically augmented variant so the board looks distinct each slot.
    tile_cache   = AugmentedPinsTileCache(
        pool=pool, max_photos_per_celeb=args.max_photos_per_celeb,
        blend_args=args.blend_args,
        target_w_px=target_w_px, target_h_px=target_h_px,
        board_aspect=board_aspect, rng_seed=ds_seed,
    )
    board, dictionary = build_board(
        BOARD_SQUARES_X, BOARD_SQUARES_Y, BOARD_SQUARE_MM, BOARD_MARKER_RATIO, BOARD_DICT,
    )
    detector      = aruco.ArucoDetector(dictionary, aruco.DetectorParameters())
    char_detector = aruco.CharucoDetector(board)
    masker        = RedHSVMasker(HSV_LO, HSV_HI)

    h_cache: dict[int, list] = {}
    def get_homographies(src_ep_idx: int, frame0: np.ndarray):
        if src_ep_idx not in h_cache:
            Hs, _ = lock_homographies_multi(frame0, detector, char_detector, board, n_boards=3)
            h_cache[src_ep_idx] = Hs
        return h_cache[src_ep_idx]

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from copy import deepcopy
    repo_id = f"{args.hub_org}/{out_name}" if args.push_to_hub else out_name
    dataset = LeRobotDataset.create(
        repo_id=repo_id, fps=30,
        features=deepcopy(FEATURES_SCHEMA),
        root=out_dir, robot_type="so_follower",
        use_videos=True, vcodec=args.vcodec, streaming_encoding=True,
    )

    target_idx  = POSITION_TO_BOARD_IDX[target_position]
    fps         = 30.0
    n_source    = len(src_episodes)
    n_eps_done  = 0
    n_frames_done = 0
    for ci, cfg in enumerate(configs):
        src_idx = ci % n_source
        src     = src_episodes[src_idx]
        frames  = read_episode_frames(src_mp4, src, fps)
        actions = src_action[src.from_frame_global: src.to_frame_global]
        states  = src_state[src.from_frame_global: src.to_frame_global]
        assert len(frames) == len(actions) == len(states) == src.length

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
            print(f"[{out_name}] WARN cfg{ci}: no board locks; skipping", flush=True)
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
        n_eps_done    += 1
        n_frames_done += src.length
        if n_eps_done % 25 == 0 or n_eps_done == len(configs):
            elapsed = time.time() - t0
            fps_c   = n_frames_done / max(elapsed, 1e-6)
            eta     = (len(configs) - n_eps_done) * elapsed / max(n_eps_done, 1)
            print(f"[{out_name}] {n_eps_done}/{len(configs)} eps "
                  f"({n_frames_done} fr, {fps_c:.1f} fps, eta {eta / 60:.1f} min)", flush=True)

    dataset.finalize()
    elapsed = time.time() - t0
    summary = {
        "name": out_name, "skipped": False,
        "n_episodes": n_eps_done, "n_frames": n_frames_done,
        "elapsed_s": round(elapsed, 1), "out_dir": str(out_dir),
        "disk_mb": round(_dir_size_mb(out_dir), 1),
    }
    if args.push_to_hub:
        info = _push_to_hub(out_dir, repo_id)
        summary["hub_url"]         = info["hub_url"]
        summary["upload_elapsed_s"] = info["elapsed_s"]

    print(f"[{out_name}] DONE — {summary}", flush=True)
    return summary


def _patched_worker_wrapper(args: WorkerArgs) -> dict:
    try:
        return _patched_generate_one_pins_dataset(args)
    except SystemExit as e:
        return {"name": f"{args.output_suffix}_{args.target_celeb}_{args.target_position}",
                "error": f"SystemExit({e})", "skipped": True}
    except Exception as e:
        return {"name": f"{args.output_suffix}_{args.target_celeb}_{args.target_position}",
                "error": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc(), "skipped": True}


# ── WorkerArgs extension — adds source_prefix / source_suffix / source_hub_org
# WorkerArgs is a dataclass defined in the original tool with positional fields.
# We subclass it and add the three new fields as keyword-only with defaults so
# the pickling contract is unchanged for the base worker pool.

from dataclasses import dataclass, field  # noqa: E402

@dataclass
class RagWorkerArgs(WorkerArgs):
    """WorkerArgs extended with source dataset routing knobs."""
    # "{source_prefix}_{position}{source_suffix}" → downloaded from source_hub_org if absent.
    # Default: dataset_v5_charuko_{pos}_1 from RobotLearningVLA.
    source_prefix:   str = "dataset_v5_charuko"
    source_suffix:   str = "_1"
    source_hub_org:  str = "RobotLearningVLA"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _DEFAULT_POOL = Path("datasets/rag_pool.json")
    if not _DEFAULT_POOL.is_file():
        _DEFAULT_POOL = Path("datasets/pins-face-recognition-top30-quality.json")
        if not _DEFAULT_POOL.is_file():
            _DEFAULT_POOL = Path("datasets/pins-face-recognition-top30.json")
    ap.add_argument("--pool-json", type=Path, default=_DEFAULT_POOL)
    ap.add_argument("--target-celebs", default="all")
    ap.add_argument("--target-positions", default=",".join(POSITIONS))
    ap.add_argument("--max-photos-per-celeb", type=int, default=5)
    ap.add_argument("--distractors-per-target-photo", type=int, default=20)
    ap.add_argument("--output-suffix", default="rag1")
    ap.add_argument("--source-root", type=Path, default=Path("datasets"),
                    help="Local root for source datasets (downloaded here if absent).")
    ap.add_argument("--source-prefix", default="dataset_v5_charuko",
                    help="Prefix for source dataset dir names "
                         "(default: dataset_v5_charuko). "
                         "Full name: {prefix}_{position}{suffix}.")
    ap.add_argument("--source-suffix", default="_1",
                    help="Suffix for source dataset dir names (default: _1).")
    ap.add_argument("--source-hub-org", default="RobotLearningVLA",
                    help="HF Hub org to auto-download missing source datasets "
                         "(default: RobotLearningVLA). Set empty to disable.")
    ap.add_argument("--out-root", type=Path, default=Path("datasets"))
    ap.add_argument("--vcodec", default="h264")
    ap.add_argument("--n-workers", type=int, default=0)
    ap.add_argument("--push-to-hub", action="store_true")
    ap.add_argument("--hub-org", default="yukk1")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--blend-contrast", type=float, default=0.88)
    ap.add_argument("--blend-brightness-offset", type=int, default=15)
    ap.add_argument("--blend-saturation", type=float, default=0.9)
    ap.add_argument("--blend-blur-kernel", type=int, default=3)
    ap.add_argument("--blend-noise-sigma", type=float, default=3.0)
    ap.add_argument("--blend-warmth", type=float, default=1.0)
    ap.add_argument("--global-lift-gain", type=float, default=1.25)
    ap.add_argument("--global-lift-offset", type=float, default=10.0)
    ap.add_argument("--global-lift-warmth", type=float, default=1.06)
    args = ap.parse_args()

    if not args.pool_json.is_file():
        sys.exit(f"--pool-json {args.pool_json} not found")
    pool           = load_pool(args.pool_json)
    target_celebs  = resolve_target_celebs(args.target_celebs, pool)
    positions      = [s.strip() for s in args.target_positions.split(",") if s.strip()]

    if args.dry_run:
        eps_per_pos = args.max_photos_per_celeb * args.distractors_per_target_photo
        total_eps   = eps_per_pos * len(positions)
        print(f"\n=== RAG Pins dry-run (normalized episode budget) ===")
        print(f"  pool        : {args.pool_json} ({len(pool)} celebs)")
        print(f"  episodes/ds : {eps_per_pos}  "
              f"({args.max_photos_per_celeb} slots × {args.distractors_per_target_photo} distractors)")
        print(f"  NOTE: all celebrities get exactly {eps_per_pos} episodes/position;")
        print(f"        under-represented ones use augmented photo variants.")
        print(f"\n  {'slug':<30}  {'actual':>6}  {'aug slots':>9}  episodes/pos")
        print(f"  {'-'*30}  {'------':>6}  {'---------':>9}  ------------")
        for slug in sorted(target_celebs):
            n_actual = min(args.max_photos_per_celeb, pool[slug]["n_images"])
            n_aug    = args.max_photos_per_celeb - n_actual
            print(f"  {slug:<30}  {n_actual:>6}  {n_aug:>9}  {eps_per_pos}")
        print(f"\n  source pattern : {args.source_prefix}_{{pos}}{args.source_suffix}")
        print(f"  source hub org : {args.source_hub_org or '(local only)'}")
        return

    # Pre-download source datasets in the main process so workers don't race.
    for pos in positions:
        src_name = f"{args.source_prefix}_{pos}{args.source_suffix}"
        if args.source_hub_org:
            _maybe_download_source(args.source_root, src_name, args.source_hub_org)
        src_ds = args.source_root / src_name
        if not src_ds.is_dir():
            sys.exit(
                f"source dataset missing: {src_ds}\n"
                f"  Set --source-hub-org to auto-download, e.g.:\n"
                f"  --source-hub-org RobotLearningVLA"
            )

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
        RagWorkerArgs(
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
            source_hub_org=args.source_hub_org,
        )
        for c in target_celebs for p in positions
    ]

    n_datasets = len(worker_args_list)
    n_workers  = (min(os.cpu_count() or 1, n_datasets)
                  if args.n_workers <= 0 else min(args.n_workers, n_datasets))

    print(f"=== RAG Pins-pool synth gen (v5 source) ===")
    print(f"  pool         : {args.pool_json} ({len(pool)} celebs)")
    print(f"  targets      : {len(target_celebs)} celebs x {len(positions)} positions = {n_datasets} datasets")
    print(f"  configs/ds   : {args.max_photos_per_celeb * args.distractors_per_target_photo}")
    print(f"  source       : {args.source_prefix}_{{pos}}{args.source_suffix}  "
          f"(hub: {args.source_hub_org or 'local only'})")
    print(f"  workers      : {n_workers}")
    print(f"  output suffix: {args.output_suffix}")
    print(f"  push to hub  : {args.push_to_hub}  (org: {args.hub_org})")
    print()

    t_global = time.time()
    if n_workers == 1:
        results = [_patched_worker_wrapper(wa) for wa in worker_args_list]
    else:
        with Pool(processes=n_workers) as p_:
            results = p_.map(_patched_worker_wrapper, worker_args_list)
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
        total_eps    += r["n_episodes"]
        total_frames += r["n_frames"]
        total_mb     += r["disk_mb"]
    print(f"\nTotal: {total_eps} episodes  {total_frames} frames  "
          f"{total_mb / 1024:.2f} GB on disk")


if __name__ == "__main__":
    main()
