#!/usr/bin/env python3
"""Build a small held-out eval set: 5 algvr researchers + 5 PINS top-30 quality
celebs, each at a single random position, ~4 episodes each.

Used by the §4.1 val watcher (set EVAL3_VAL_REPOS to the generated names). Held
out from training because:
  1. Output prefix is dataset_v5_synth_HOLDOUT_{algvr,pins30q5}_*_full, which
     does NOT match the launcher's training globs (dataset_v5_synth_algvr_*_full
     and dataset_v5_synth_pins30q5_*_full). The launcher will not pick these up.
  2. Distractor seed (default 4242) is distinct from training (seed 42), so the
     same (target_photo, position) configs draw a different distractor pool.

Output naming follows the same _left_/_middle_/_right_ convention that the val
watcher's slot-derivation regex expects.

Usage:
    python tools/eval3_build_holdout_eval_set.py                       # default
    python tools/eval3_build_holdout_eval_set.py --dry-run             # plan only
    python tools/eval3_build_holdout_eval_set.py --n-per-pool 3 --overwrite
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from eval3_synth_pins_dataset_gen import (  # noqa: E402
    WorkerArgs, generate_one_pins_dataset, load_pool,
)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--algvr-pool",
                    default="datasets/algvr-conference.json",
                    help="algvr-conference pool JSON")
    ap.add_argument("--pins30q5-pool",
                    default="datasets/pins-face-recognition-top30-quality.json",
                    help="PINS top-30 quality pool JSON")
    ap.add_argument("--source-root", default="datasets",
                    help="root holding dataset_v5_charuko_{left,middle,right}_full")
    ap.add_argument("--out-root", default="datasets",
                    help="output dir for the eval datasets")
    ap.add_argument("--n-per-pool", type=int, default=5,
                    help="how many celebs from each pool (default 5 + 5)")
    ap.add_argument("--n-target-photos", type=int, default=2,
                    help="target photos per celeb (N). default 2")
    ap.add_argument("--distractors-per-target-photo", type=int, default=2,
                    help="distractor scenes per target photo (M). default 2 -> 4 eps/ds")
    ap.add_argument("--selection-seed", type=int, default=42,
                    help="seed for picking celebs + positions")
    ap.add_argument("--distractor-seed", type=int, default=4242,
                    help="seed for distractor sampling (distinct from training seed 42)")
    ap.add_argument("--vcodec", default="h264")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="print plan, don't write anything")
    args = ap.parse_args()

    pools = {
        "algvr":    Path(args.algvr_pool),
        "pins30q5": Path(args.pins30q5_pool),
    }
    # Independent RNGs per (pool, axis) so seed re-use doesn't accidentally
    # cluster all positions to one slot when pool ordering happens to align.
    plan: list[dict] = []
    for tag, pool_json in pools.items():
        if not pool_json.is_file():
            sys.exit(f"pool JSON missing: {pool_json}")
        pool = load_pool(pool_json)
        slugs = sorted(pool.keys())
        if len(slugs) < args.n_per_pool:
            sys.exit(f"{tag} pool has only {len(slugs)} celebs, need {args.n_per_pool}")
        sel_rng = random.Random(f"{args.selection_seed}/{tag}/celebs")
        pos_rng = random.Random(f"{args.selection_seed}/{tag}/positions")
        chosen = sel_rng.sample(slugs, args.n_per_pool)
        # Force a roughly-balanced position distribution: with 5 picks, lay down
        # a 2/2/1 shuffle over (left, middle, right) so we always cover all three.
        if args.n_per_pool >= 3:
            base = ["left", "middle", "right"] * ((args.n_per_pool + 2) // 3)
            positions = base[: args.n_per_pool]
            pos_rng.shuffle(positions)
        else:
            positions = [pos_rng.choice(["left", "middle", "right"])
                         for _ in range(args.n_per_pool)]
        for slug, pos in zip(chosen, positions):
            plan.append({
                "tag":       tag,
                "pool_json": str(pool_json),
                "celeb":     slug,
                "name":      pool[slug]["name"],
                "position":  pos,
            })

    out_root = Path(args.out_root)
    print(f">> Held-out eval set plan ({len(plan)} datasets)")
    print(f"   selection seed   : {args.selection_seed}")
    print(f"   distractor seed  : {args.distractor_seed}  (training uses 42; this differs)")
    print(f"   N (target photos): {args.n_target_photos}")
    print(f"   M (distractors)  : {args.distractors_per_target_photo}")
    print(f"   eps/dataset      : {args.n_target_photos * args.distractors_per_target_photo}")
    print(f"   out-root         : {out_root}")
    print()
    for p in plan:
        out_name = f"dataset_v5_synth_holdout_{p['tag']}_{p['celeb']}_{p['position']}_full"
        print(f"   {p['tag']:8s}  {p['name']:30s} -> {out_name}")
        p["out_name"] = out_name

    if args.dry_run:
        print("\n(dry-run) exiting before any IO.")
        return 0

    print()
    print("== generating ==")
    t_total = time.time()
    results = []
    for i, p in enumerate(plan, 1):
        wargs = WorkerArgs(
            target_celeb=p["celeb"],
            target_position=p["position"],
            pool_json=p["pool_json"],
            max_photos_per_celeb=args.n_target_photos,
            distractors_per_target_photo=args.distractors_per_target_photo,
            source_root=args.source_root,
            out_root=str(out_root),
            output_suffix=f"holdout_{p['tag']}",
            blend_args={},
            global_lift_args={},
            vcodec=args.vcodec,
            push_to_hub=False,
            hub_org="RobotLearningVLA",
            overwrite=args.overwrite,
            seed=args.distractor_seed,
            source_prefix="dataset_v5_charuko_",
            source_suffix="_full",
            output_prefix="dataset_v5_synth_",
            output_postfix="_full",
        )
        print(f"[{i}/{len(plan)}] {p['out_name']}")
        t0 = time.time()
        try:
            res = generate_one_pins_dataset(wargs)
        except Exception as e:
            print(f"  !! FAILED: {e}")
            results.append({"name": p["out_name"], "error": str(e)})
            continue
        elapsed = time.time() - t0
        print(f"   done ({elapsed:.1f}s) — {res.get('n_episodes', '?')} eps, "
              f"{res.get('n_frames', '?')} frames, {res.get('disk_mb', '?')} MB")
        results.append(res)

    n_ok = sum(1 for r in results if not r.get("error") and not r.get("skipped"))
    n_skip = sum(1 for r in results if r.get("skipped"))
    n_fail = sum(1 for r in results if r.get("error"))
    print()
    print(f"== SUMMARY: {n_ok} ok / {n_skip} skipped / {n_fail} failed "
          f"(of {len(plan)}) — total {time.time() - t_total:.1f}s ==")

    print()
    print("== EVAL3_VAL_REPOS snippet (paste into the launcher invocation) ==")
    repos = ",".join(f"RobotLearningVLA/{p['out_name']}" for p in plan)
    print(f"EVAL3_VAL_REPOS={repos}")
    print(f"EVAL3_VAL_LOCAL_REPOS={repos}  # local-only, no Hub fetch needed")

    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
