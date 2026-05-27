#!/usr/bin/env python3
"""Build a pool JSON for eval3_synth_pins_dataset_gen.py from local datasets.

The pool JSON lists celebrity face images that the ChArUco-compositing synth
generator warps into robot table recordings to create synthetic training episodes.
Quality of the synthetic data depends directly on the resolution and variety of
the face images listed in ``quality_photos``.

This script aggregates images from all available local sources and sorts them by
resolution (largest first) within each celebrity, so the generator always uses the
highest-quality image available.

Sources scanned (in priority order for de-duplication):
  1. datasets/celeb_refdb/<slug>/          (consolidated DB — any images added via
                                            build_celeb_refdb.py --from-dir land here)
  2. datasets/in-distribution-eval-3/<slug>/ (1382–1645×2100px — TOY only)
  3. datasets/out-distribution-eval-3/<slug>/ (1575×2100px — TOY only)
  4. datasets/out-distribution-eval-3-pins/<slug>/ (302–564px — OOD only)

Quality summary of existing local sources::

    TOY celebrities (taylor_swift, barack_obama, yann_lecun)
      10 images each at 1382–1645×2100px — excellent for compositing.

    OOD celebrities (cristiano_ronaldo, lionel_messi, …)
      1 image each at 302–564px from out-distribution-eval-3-pins.
      To improve OOD diversity/quality before running the synth generator:

        # Option A — import any local named-folder face dataset:
        python scripts/eval3_rag/build_celeb_refdb.py \\
            --from-dir /path/to/face_dataset/with/named/folders

        # Option B — add individual images for a specific celebrity:
        python scripts/eval3_rag/build_celeb_refdb.py \\
            --add-images elon_musk=/path/to/elon1.jpg,/path/to/elon2.jpg

      After either option, re-run this script to regenerate the pool JSON.

Usage::

    # Build the pool JSON (run from repo root):
    python scripts/eval3_rag/build_rag_pool_json.py
    # → datasets/rag_pool.json

    # Custom output path:
    python scripts/eval3_rag/build_rag_pool_json.py --out-json /tmp/my_pool.json

    # Verify image resolution stats without writing:
    python scripts/eval3_rag/build_rag_pool_json.py --stats

Then pass it to the synth generator:
    EVAL3_RAG_POOL_JSON=datasets/rag_pool.json \\
    EVAL3_RAG_HF_ORG=yukk1 EVAL3_RAG_PUSH_TO_HUB=1 \\
      ./scripts/eval3_rag/gen_rag_synth.sh
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

_SCRIPT = Path(__file__).resolve()
_ROOT   = _SCRIPT.parent.parent.parent  # robot-learning-vla/

# TOY celebrities — held_out=True tells the synth generator to skip them
# (they already have real v4 robot recordings).
_TOY_SLUGS = {"taylor_swift", "barack_obama", "yann_lecun"}

_KNOWN_NAMES: dict[str, str] = {
    "taylor_swift":      "Taylor Swift",
    "barack_obama":      "Barack Obama",
    "yann_lecun":        "Yann LeCun",
    "elon_musk":         "Elon Musk",
    "lionel_messi":      "Lionel Messi",
    "cristiano_ronaldo": "Cristiano Ronaldo",
    "rihanna":           "Rihanna",
    "dwayne_johnson":    "Dwayne Johnson",
    "tom_cruise":        "Tom Cruise",
    "leonardo_dicaprio": "Leonardo DiCaprio",
    "robert_downey_jr":  "Robert Downey Jr",
    "scarlett_johansson":"Scarlett Johansson",
    "jennifer_lawrence": "Jennifer Lawrence",
}

# Source directories scanned in order.  Images found in earlier dirs are
# preferred when de-duplicating by filename stem.
_SOURCE_DIRS: list[Path] = [
    _ROOT / "datasets/celeb_refdb",
    _ROOT / "datasets/in-distribution-eval-3",
    _ROOT / "datasets/out-distribution-eval-3",
    _ROOT / "datasets/out-distribution-eval-3-pins",
]

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def _image_width(path: Path) -> int:
    """Return image width in pixels, or 0 on failure (used for sorting only)."""
    try:
        from PIL import Image
        return Image.open(path).size[0]
    except Exception:
        return 0


def _slug_to_name(slug: str) -> str:
    return _KNOWN_NAMES.get(slug, " ".join(w.capitalize() for w in slug.split("_")))


def build_pool_json(
    out_json: Path,
    *,
    source_dirs: list[Path] | None = None,
    min_images: int = 1,
    sort_by_resolution: bool = True,
    stats_only: bool = False,
) -> dict:
    """Aggregate celebrity images from all local source directories.

    Images for the same celebrity found in multiple source dirs are merged
    and deduplicated by filename stem.  The ``quality_photos`` list is sorted
    by image width descending so the synth generator uses the highest-resolution
    image first when ``--max-photos-per-celeb`` caps the count.

    Returns the pool dict.
    """
    dirs = source_dirs if source_dirs is not None else _SOURCE_DIRS

    def _file_key(p: Path) -> str:
        """Content fingerprint: first 4 KB hashed with SHA1 (fast, reliable dedup)."""
        h = hashlib.sha1()
        with p.open("rb") as f:
            h.update(f.read(4096))
        return h.hexdigest()

    # Collect images per slug, deduplicating by content hash so that the same
    # image stored in multiple source dirs (e.g. celeb_refdb/ and the original
    # out-distribution-eval-3-pins/) is counted only once.
    # {slug: {content_hash: Path}}  — first-found wins (priority order).
    slug_to_hash_path: dict[str, dict[str, Path]] = {}

    for src_dir in dirs:
        if not src_dir.is_dir():
            log.debug("source dir not found, skipping: %s", src_dir)
            continue
        for celeb_dir in sorted(src_dir.iterdir()):
            if not celeb_dir.is_dir():
                continue
            slug = celeb_dir.name
            for img in sorted(celeb_dir.iterdir()):
                if img.suffix.lower() not in _IMG_EXTS:
                    continue
                slug_to_hash_path.setdefault(slug, {})
                key = _file_key(img)
                if key not in slug_to_hash_path[slug]:
                    slug_to_hash_path[slug][key] = img

    celebrities = []
    for slug, hash_map in sorted(slug_to_hash_path.items()):
        imgs = list(hash_map.values())
        if len(imgs) < min_images:
            log.warning(
                "slug=%r: only %d image(s) — skipping (need >= %d)",
                slug, len(imgs), min_images,
            )
            continue

        if sort_by_resolution:
            imgs.sort(key=lambda p: _image_width(p), reverse=True)
        else:
            imgs.sort()

        widths = [_image_width(p) for p in imgs]
        log.info(
            "slug=%-25s  %2d image(s)  resolution: %s–%spx  held_out=%s",
            f"{slug!r}",
            len(imgs),
            min(widths),
            max(widths),
            slug in _TOY_SLUGS,
        )

        celebrities.append({
            "name":           _slug_to_name(slug),
            "slug":           slug,
            "n_images":       len(imgs),
            "quality_photos": [str(p) for p in imgs],
            "held_out":       slug in _TOY_SLUGS,
            "rank":           None,
            "category":       None,
        })

    if stats_only:
        ood = [c for c in celebrities if not c["held_out"]]
        toy = [c for c in celebrities if c["held_out"]]
        print(f"\nPool stats ({len(celebrities)} celebrities total):")
        print(f"  TOY ({len(toy)} celebs, skipped by synth generator):")
        for c in toy:
            widths = [_image_width(Path(p)) for p in c["quality_photos"]]
            print(f"    {c['slug']:<25}  {c['n_images']} imgs  {min(widths)}–{max(widths)}px")
        print(f"  OOD ({len(ood)} celebs, used by synth generator):")
        for c in ood:
            widths = [_image_width(Path(p)) for p in c["quality_photos"]]
            print(f"    {c['slug']:<25}  {c['n_images']} imgs  {min(widths)}–{max(widths)}px")
            if c["n_images"] < 3:
                print(f"    {'':25}  ^^^ only {c['n_images']} image(s) — synth will produce "
                      f"{c['n_images']}×N episodes instead of max_photos×N")
        return {"celebrities": celebrities}

    pool = {
        "dataset": "eval3_rag local pool",
        "source":  "built by scripts/eval3_rag/build_rag_pool_json.py",
        "sources_scanned": [str(d) for d in dirs],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_celebrities": len(celebrities),
        "total_images": sum(c["n_images"] for c in celebrities),
        "celebrities": celebrities,
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(pool, indent=2))

    n_ood = sum(1 for c in celebrities if not c["held_out"])
    n_toy = sum(1 for c in celebrities if c["held_out"])
    log.info(
        "Wrote %d celebrities (%d OOD + %d TOY) → %s",
        len(celebrities), n_ood, n_toy, out_json,
    )
    return pool


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build a synth-generator pool JSON from local celebrity datasets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument(
        "--out-json",
        default=str(_ROOT / "datasets/rag_pool.json"),
        metavar="FILE",
        help="output pool JSON path (default: datasets/rag_pool.json)",
    )
    ap.add_argument(
        "--min-images", type=int, default=1,
        help="skip celebrities with fewer than this many images (default: 1)",
    )
    ap.add_argument(
        "--no-sort-resolution", action="store_true",
        help="keep quality_photos sorted by filename instead of resolution",
    )
    ap.add_argument(
        "--stats", action="store_true",
        help="print image counts and resolution stats without writing the JSON",
    )
    args = ap.parse_args()

    pool = build_pool_json(
        Path(args.out_json),
        min_images=args.min_images,
        sort_by_resolution=not args.no_sort_resolution,
        stats_only=args.stats,
    )

    if args.stats:
        return

    print(f"\nPool JSON written: {args.out_json}")
    print(f"  {pool['total_celebrities']} celebrities, {pool['total_images']} images total")
    print(f"\nOOD synth episode estimate (default MAX_PHOTOS=5, DISTRACTORS=20):")
    for c in pool["celebrities"]:
        if c["held_out"]:
            continue
        n_photos = min(5, c["n_images"])
        episodes = n_photos * 20 * 3   # ×3 positions
        print(f"  {c['slug']:<25}  {c['n_images']} imgs → {n_photos}×20×3 = {episodes} episodes")
    print(f"\nUse with gen_rag_synth.sh:")
    print(f"  EVAL3_RAG_POOL_JSON={args.out_json} \\")
    print(f"  EVAL3_RAG_HF_ORG=yukk1 EVAL3_RAG_PUSH_TO_HUB=1 \\")
    print(f"    ./scripts/eval3_rag/gen_rag_synth.sh")


if __name__ == "__main__":
    main()
