#!/usr/bin/env python3
"""Pinned-grid variant of the Eval3 synth dataset generator.

Where the default generator (``tools/eval3_synth_dataset_gen.py``) builds a full
N×N×N×2 Cartesian product of (target_photo, other1_photo, other2_photo, swap),
this variant builds a "pinned" grid:

    for each target_photo in 0..N-1:                                 (5 values for ID pool)
        for each of K seeded distractor combos:                      (configurable, default 10)
            generate one episode

Each (celeb, target_photo, position) tuple gets exactly K episodes, all with
different distractor placements. The default K=10 yields **5×K = 50 episodes
per output dataset × 9 datasets = 450 total episodes** — a much smaller corpus
than the default _2 generator's 2,250.

Differences from the default generator:
  - Sources: ``dataset_v3_charuco_{left,middle,right}_1`` (the older ``_1``
    captures) instead of ``_2``.
  - Output names: ``dataset_v3_synth_pinned_<celeb>_<position>_1``.
  - Config grid: 5×K instead of 5^3 × 2.

Everything else (homography lock, tile composition, global lift, push-to-hub,
sharding orchestration, prep cache) is reused unchanged from the main tool.

Smoke (one dataset, K=4 → 20 episodes, no push):
    python tools/eval3_synth_dataset_gen_pinned.py \
        --target-celebs taylor_swift --target-positions left \
        --n-distractors-per-photo 4 --out-root /tmp/pinned_smoke

Full local sweep (default K=10, 9 datasets, ~450 episodes):
    python tools/eval3_synth_dataset_gen_pinned.py --n-workers 9

Dry run:
    python tools/eval3_synth_dataset_gen_pinned.py --dry-run
"""
from __future__ import annotations

import argparse
import random
import sys
import time
import traceback
from multiprocessing import Pool
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

# Reuse the full machinery from the _2 generator. We only override the config
# grid and pin source_suffix/output_postfix to "_1".
from eval3_synth_dataset_gen import (  # noqa: E402
    CANONICAL_NAME,
    POSITIONS,
    Config,
    WorkerArgs,
    _worker_wrapper,
    load_celebrity_pool,
)


# ---------------------------------------------------------------------------
# Pinned config grid
# ---------------------------------------------------------------------------

def build_pinned_config_grid(
    target_celeb: str,
    target_position: str,
    n_photos_per_celeb: int,
    n_distractors_per_photo: int,
    seed: int,
) -> list[Config]:
    """5 × n_distractors_per_photo configs per dataset.

    Distractor combos = (other1_photo_idx, other2_photo_idx, swap).
    The pool of distractor combos has size n_photos_per_celeb^2 * 2
    (e.g. 50 for n_photos=5). We deterministically pick K of them per
    dataset via a seeded RNG keyed on (target_celeb, target_position) so
    each dataset gets a stable but distinct distractor sample.

    If K >= pool size, the full pool is used (no repeats); the result
    becomes 5 × pool_size configs.
    """
    if target_celeb not in CANONICAL_NAME:
        sys.exit(f"unknown target_celeb={target_celeb!r}, want one of {list(CANONICAL_NAME)}")
    if target_position not in POSITIONS:
        sys.exit(f"unknown target_position={target_position!r}, want one of {POSITIONS}")
    others = sorted(s for s in CANONICAL_NAME if s != target_celeb)
    other_a, other_b = others

    # Full pool of distractor combos for this dataset.
    pool: list[tuple[int, int, bool]] = []
    for ap in range(n_photos_per_celeb):
        for bp in range(n_photos_per_celeb):
            for swap in (False, True):
                pool.append((ap, bp, swap))

    k = max(0, int(n_distractors_per_photo))
    rng = random.Random(f"{seed}|{target_celeb}|{target_position}")
    if k >= len(pool):
        chosen = list(pool)
    else:
        chosen = rng.sample(pool, k=k)
    chosen.sort()  # deterministic order regardless of sampling

    configs: list[Config] = []
    for tp in range(n_photos_per_celeb):
        for (ap, bp, swap) in chosen:
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
    return configs


# ---------------------------------------------------------------------------
# CLI + dispatcher
# ---------------------------------------------------------------------------

def _dry_run(
    targets: list[tuple[str, str]],
    n_distractors_per_photo: int,
    seed: int,
    source_root: Path,
    celebrity_jsons: list[Path],
    source_suffix: str = "_1",
    output_postfix: str = "_1",
    output_suffix: str = "pinned",
) -> None:
    pool, n_photos = load_celebrity_pool(celebrity_jsons)
    distractor_pool_size = n_photos ** 2 * 2
    effective_k = min(n_distractors_per_photo, distractor_pool_size)
    per_dataset = n_photos * effective_k
    print("=" * 72)
    print(f"DRY RUN (PINNED) — {len(targets)} datasets × {per_dataset} eps = "
          f"{len(targets) * per_dataset} total")
    print(f"  source_suffix      : {source_suffix}")
    print(f"  output_postfix     : {output_postfix}")
    print(f"  output_suffix      : {output_suffix!r}")
    print(f"  celebrity_jsons    : {[str(p) for p in celebrity_jsons]}")
    print(f"  n_photos/celeb     : {n_photos}")
    print(f"  distractor pool    : {distractor_pool_size}  (= {n_photos}^2 × 2)")
    print(f"  K (distractors/ph) : {effective_k}  (requested {n_distractors_per_photo})")
    print(f"  per-dataset eps    : {n_photos} × {effective_k} = {per_dataset}")
    print(f"  seed               : {seed}")
    print("=" * 72)
    for celeb, pos in targets:
        prefix = f"_{output_suffix}" if output_suffix else ""
        out_name = f"dataset_v3_synth{prefix}_{celeb}_{pos}{output_postfix}"
        task = f"Place the coke on {CANONICAL_NAME[celeb]}"
        configs = build_pinned_config_grid(
            celeb, pos,
            n_photos_per_celeb=n_photos,
            n_distractors_per_photo=effective_k,
            seed=seed,
        )
        src_ds = source_root / f"dataset_v3_charuco_{pos}{source_suffix}"
        exists = "OK" if src_ds.is_dir() else "MISSING"
        print(f"\n{out_name}  ({len(configs)} eps, task='{task}')")
        print(f"  source: {src_ds.name}  [{exists}]")
        print(f"  sample configs (first 3 + last):")
        for c in configs[:3] + ([configs[-1]] if len(configs) > 3 else []):
            print(f"    target=({c.target_celeb},photo={c.target_photo_idx})  "
                  f"other1=({c.other1_celeb},photo={c.other1_photo_idx})  "
                  f"other2=({c.other2_celeb},photo={c.other2_photo_idx})  "
                  f"swap={c.other_swap}")
    print()


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--celebrity-json",
                    default="datasets/in-distribution-eval-3.json",
                    help="Per-celebrity image inventory JSON (ID-only by default). "
                         "Accepts comma-separated list to merge multiple pools.")
    ap.add_argument("--source-root", type=Path, default=Path("datasets"),
                    help="Root containing dataset_v3_charuco_{left,middle,right}_1.")
    ap.add_argument("--out-root", type=Path, default=Path("datasets"),
                    help="Root for synthetic dataset_v3_synth_pinned_<celeb>_<pos>_1 dirs.")
    ap.add_argument("--target-celebs", default=",".join(CANONICAL_NAME.keys()))
    ap.add_argument("--target-positions", default=",".join(POSITIONS))
    ap.add_argument("--n-distractors-per-photo", type=int, default=10,
                    help="Number of distractor combos per (target_celeb, "
                         "target_photo) pair. Default 10 = 50 eps/dataset. "
                         "Max useful value is n_photos^2 × 2 (=50 for ID, =200 "
                         "for ID+OOD); values >= max use the full distractor pool.")
    ap.add_argument("--distractor-seed", type=int, default=42,
                    help="Seed for the per-dataset distractor sampler.")
    ap.add_argument("--vcodec", default="h264")
    ap.add_argument("--n-workers", type=int, default=1,
                    help="Multiprocessing workers (one per output dataset). "
                         "Cap at 9 for the full sweep.")
    ap.add_argument("--output-suffix", default="pinned",
                    help="Inserted after 'dataset_v3_synth' in the output name "
                         "(default 'pinned' → dataset_v3_synth_pinned_<celeb>_<pos>_<postfix>). "
                         "Use e.g. 'pinned_idood' for ID+OOD pinned variants.")
    ap.add_argument("--source-suffix", default="_1",
                    help="Source ChArUco variant suffix. Default '_1' "
                         "(reads dataset_v3_charuco_<pos>_1); pass '_2' for "
                         "the larger _2 captures.")
    ap.add_argument("--output-postfix", default="_1",
                    help="Postfix appended to the output name (default '_1'; "
                         "match the source variant unless you intentionally "
                         "want a cross-variant name).")
    ap.add_argument("--push-to-hub", action="store_true",
                    help="After each dataset finishes, upload to HF Hub + create v3.0 tag.")
    ap.add_argument("--hub-org", default="RobotLearningVLA")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry-run", action="store_true")

    # Blend / global-lift knobs — kept identical to the _2 generator defaults.
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

    celebs = [s.strip() for s in args.target_celebs.split(",") if s.strip()]
    positions = [s.strip() for s in args.target_positions.split(",") if s.strip()]
    targets = [(c, p) for c in celebs for p in positions]
    celebrity_jsons = [Path(s.strip()) for s in args.celebrity_json.split(",") if s.strip()]

    if args.dry_run:
        _dry_run(
            targets,
            args.n_distractors_per_photo,
            args.distractor_seed,
            args.source_root,
            celebrity_jsons,
            source_suffix=args.source_suffix,
            output_postfix=args.output_postfix,
            output_suffix=args.output_suffix,
        )
        return

    # Validate sources + JSONs.
    for jp in celebrity_jsons:
        if not jp.is_file():
            sys.exit(f"--celebrity-json entry missing: {jp}")
    for c, p in targets:
        src_ds = args.source_root / f"dataset_v3_charuco_{p}{args.source_suffix}"
        if not src_ds.is_dir():
            sys.exit(f"source dataset missing: {src_ds}")

    # Determine n_photos from the celebrity pool so the grid size matches.
    _pool, n_photos = load_celebrity_pool(celebrity_jsons)

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

    worker_args_list: list[WorkerArgs] = []
    for c, p in targets:
        configs = build_pinned_config_grid(
            c, p,
            n_photos_per_celeb=n_photos,
            n_distractors_per_photo=args.n_distractors_per_photo,
            seed=args.distractor_seed,
        )
        worker_args_list.append(WorkerArgs(
            target_celeb=c,
            target_position=p,
            n_configs=-1,  # use the whole override grid
            source_root=str(args.source_root),
            out_root=str(args.out_root),
            celebrity_jsons=[str(jp) for jp in celebrity_jsons],
            blend_args=blend_args,
            global_lift_args=global_lift_args,
            vcodec=args.vcodec,
            push_to_hub=args.push_to_hub,
            hub_org=args.hub_org,
            overwrite=args.overwrite,
            output_suffix=args.output_suffix,
            source_suffix=args.source_suffix,
            output_postfix=args.output_postfix,
            config_grid_override=configs,
        ))

    print(f"Generating {len(worker_args_list)} PINNED datasets "
          f"with {args.n_workers} workers  "
          f"(K={args.n_distractors_per_photo}, sources={args.source_suffix}, "
          f"output_suffix={args.output_suffix!r}, output_postfix={args.output_postfix})")
    t_global = time.time()
    if args.n_workers == 1:
        results = [_worker_wrapper(wa) for wa in worker_args_list]
    else:
        with Pool(processes=args.n_workers) as pool:
            results = pool.map(_worker_wrapper, worker_args_list)
    elapsed_global = time.time() - t_global

    # Summary
    print("\n" + "=" * 72)
    print(f"SUMMARY (PINNED) — total elapsed {elapsed_global / 60:.1f} min")
    print("=" * 72)
    total_eps = total_frames = total_mb = 0
    for r in results:
        if r.get("skipped"):
            print(f"  [SKIP] {r['name']}  ({r.get('reason') or r.get('error')})")
            continue
        print(f"  [OK]   {r['name']:<56s} eps={r['n_episodes']:3d} "
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
