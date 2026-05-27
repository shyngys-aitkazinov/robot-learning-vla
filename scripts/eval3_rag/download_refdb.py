#!/usr/bin/env python3
"""Pre-download portrait images for the RAG reference-face DB.

Builds an offline portrait cache in ``datasets/celeb_refdb/`` so the deploy
script never needs internet access on demo day.

Portrait retrieval priority (both here and at deploy time)
----------------------------------------------------------
1. Local ``celeb_refdb/<slug>/``                         → instant
2. HuggingFace named-identity face dataset (--from-hf)  → bulk offline prep
3. Wikipedia REST API (fallback for anyone not in HF)    → ~1-3 s, cached

Run this script before demo day.  Once portraits are cached, deploy is fully
offline regardless of network availability.

Usage
-----
# Pre-fetch portraits for specific people via Wikipedia (works for anyone):
python scripts/eval3_rag/download_refdb.py \\
    --slugs marc_pollefeys elon_musk taylor_swift yann_lecun

# Bulk-download from a HuggingFace named-identity dataset, then fall back
# to Wikipedia for anyone not found there:
python scripts/eval3_rag/download_refdb.py \\
    --from-hf nateraw/vggface2 \\
    --slugs elon_musk taylor_swift marc_pollefeys

# Check how many people are already cached offline:
python scripts/eval3_rag/download_refdb.py --coverage

# Dry-run any command:
python scripts/eval3_rag/download_refdb.py --slugs elon_musk --dry-run

Notes
-----
• Wikipedia covers millions of public figures including researchers, academics,
  and politicians — not just entertainment celebrities.
• HuggingFace face datasets cover entertainment celebrities at higher resolution
  but rarely include academics/researchers.
• Face detection (OpenCV) validates each Wikipedia image is actually a portrait;
  non-portrait images are discarded automatically.
• The script never re-downloads an already-cached portrait.
"""
from __future__ import annotations

import argparse
import io
import logging
import os
import re
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

_SCRIPT = Path(__file__).resolve()
_ROOT   = _SCRIPT.parent.parent.parent
sys.path.insert(0, str(_SCRIPT.parent.parent))  # make scripts/ importable


# ---------------------------------------------------------------------------
# Slug helpers
# ---------------------------------------------------------------------------

def _label_to_slug(raw: str) -> str:
    """Slugify any dataset class label or folder name — no hardcoded names."""
    name = raw.lower().strip()
    for prefix in ("pins_", "class_", "celeb_", "celebrity_"):
        if name.startswith(prefix):
            name = name[len(prefix):]
    return re.sub(r"[^a-z0-9]+", "_", name).strip("_")


def _slug_to_display(slug: str) -> str:
    return slug.replace("_", " ").title()


# ---------------------------------------------------------------------------
# Source 1: Wikipedia (works for any public figure)
# ---------------------------------------------------------------------------

def fetch_from_wikipedia(
    slugs: list[str],
    db_root: Path,
    *,
    dry_run: bool = False,
) -> dict[str, bool]:
    """Download Wikipedia portrait for each slug.

    Returns {slug: success}.
    Skips slugs that already have images in db_root.
    """
    sys.path.insert(0, str(_SCRIPT.parent.parent))
    from eval3_rag.reference_injector import (
        _fetch_portrait_online, _write_sentinel, _write_canonical_portrait,
    )

    results: dict[str, bool] = {}
    for slug in slugs:
        celeb_dir = db_root / slug
        existing = list(celeb_dir.glob("wiki_portrait.*")) if celeb_dir.exists() else []
        if existing:
            log.info("wiki: %s — already cached (%s)", slug, existing[0].name)
            # Retroactively stamp per-file sentinel for previously-downloaded portraits.
            for p in existing:
                _write_sentinel(celeb_dir, p.name)
            results[slug] = True
            continue

        if dry_run:
            log.info("[DRY-RUN] wiki: would fetch portrait for %r", slug)
            results[slug] = True
            continue

        path = _fetch_portrait_online(slug, db_root, validate_face=True)
        results[slug] = path is not None
        if not path:
            log.warning(
                "wiki: no portrait found for %r (%s). "
                "Place a photo manually in %s/",
                slug, _slug_to_display(slug), celeb_dir,
            )
    return results


# ---------------------------------------------------------------------------
# Source 2: HuggingFace named-identity dataset (optional, entertainment focus)
# ---------------------------------------------------------------------------

def fetch_from_hf(
    repo_id: str,
    slugs: list[str],
    db_root: Path,
    *,
    split: str = "train",
    image_col: str = "image",
    label_col: str = "label",
    max_per_person: int = 5,
    max_scan: int = 500_000,
    dry_run: bool = False,
) -> set[str]:
    """Stream portraits from a HuggingFace named-identity dataset.

    Returns the set of slugs that were satisfied from this source (so the
    caller knows which ones still need a Wikipedia fallback).

    Fails gracefully: if the dataset is unreachable or has no named-identity
    labels, logs a warning and returns an empty set — the caller falls back
    to Wikipedia for all requested slugs.
    """
    try:
        from datasets import load_dataset
        from datasets.exceptions import DatasetNotFoundError
    except ImportError:
        log.warning("'datasets' library not installed — skipping HF source. "
                    "Install with: pip install datasets")
        return set()

    from PIL import Image as _PILImage
    sys.path.insert(0, str(_SCRIPT.parent.parent))
    from eval3_rag.reference_injector import _write_sentinel, _write_canonical_portrait

    target_slugs = set(slugs)
    satisfied: set[str] = set()

    try:
        log.info("HF: connecting to %r …", repo_id)
        ds = load_dataset(repo_id, split=split, streaming=True, trust_remote_code=False)
    except Exception as exc:
        log.warning("HF: could not load %r (%s) — will use Wikipedia for all slugs.", repo_id, exc)
        return set()

    # Build integer ClassLabel ID → slug map when available.
    id_to_slug: dict[int, str] | None = None
    feats = getattr(ds, "features", None) or {}
    label_feat = feats.get(label_col)
    if label_feat is not None and hasattr(label_feat, "names"):
        id_to_slug = {}
        for idx, class_name in enumerate(label_feat.names):
            slug = _label_to_slug(class_name)
            if slug and slug in target_slugs:
                id_to_slug[idx] = slug
        n_matched = len(set(id_to_slug.values()))
        log.info("HF: %d classes in dataset, %d match requested slugs.", len(label_feat.names), n_matched)
        if n_matched == 0:
            log.warning(
                "HF: none of the requested slugs found in %r.\n"
                "  Requested : %s\n"
                "  Note      : class names are slugified before matching "
                "(e.g. 'Marc_Pollefeys' → 'marc_pollefeys').",
                repo_id, sorted(target_slugs),
            )
            return set()
    elif label_feat is None or not hasattr(label_feat, "names"):
        log.warning(
            "HF: dataset %r has no named-identity label column %r — "
            "can't match to person slugs. Will use Wikipedia instead.",
            repo_id, label_col,
        )
        return set()

    counts: dict[str, int] = {}
    scanned = 0

    for row in ds:
        if scanned >= max_scan:
            break
        scanned += 1

        raw_label = row.get(label_col)
        if raw_label is None:
            continue

        slug = id_to_slug.get(int(raw_label)) if id_to_slug else _label_to_slug(str(raw_label))
        if not slug or slug not in target_slugs:
            continue

        celeb_dir = db_root / slug
        on_disk  = len(list(celeb_dir.glob("fromhf_*"))) if celeb_dir.exists() else 0
        new_here = counts.get(slug, 0)
        if on_disk + new_here >= max_per_person:
            satisfied.add(slug)
            if satisfied >= target_slugs:
                log.info("HF: all requested slugs satisfied — stopping early.")
                break
            continue

        raw_img = row.get(image_col)
        if raw_img is None:
            continue
        try:
            pil_img = (
                raw_img.convert("RGB")
                if isinstance(raw_img, _PILImage.Image)
                else _PILImage.open(io.BytesIO(raw_img)).convert("RGB")
            )
        except Exception:
            continue

        w, h = pil_img.size
        n_next = on_disk + new_here + 1
        dst = celeb_dir / f"fromhf_{n_next:03d}.jpg"

        if dry_run:
            log.info("[DRY-RUN] hf %-30s  %dx%d  → %s", slug, w, h, dst.name)
        else:
            celeb_dir.mkdir(parents=True, exist_ok=True)
            pil_img.save(dst, "JPEG", quality=95)
            # HF face datasets contain pre-cropped identity images — mark validated.
            _write_sentinel(celeb_dir, dst.name)
            log.info("hf: %-30s  %dx%d  → %s", slug, w, h, dst.name)

        counts[slug] = new_here + 1
        if counts[slug] >= max_per_person:
            satisfied.add(slug)

    log.info("HF: scanned %d rows, found portraits for: %s", scanned, sorted(satisfied))

    # Pre-compute canonical portrait for each satisfied slug so inference is instant.
    for slug in satisfied:
        celeb_dir = db_root / slug
        try:
            imgs = [p for p in celeb_dir.iterdir()
                    if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}]
            _write_canonical_portrait(celeb_dir, imgs)
        except Exception:
            pass

    return satisfied


# ---------------------------------------------------------------------------
# Coverage report
# ---------------------------------------------------------------------------

def coverage_report(db_root: Path) -> None:
    if not db_root.is_dir():
        print(f"DB not found at {db_root}")
        print("Run this script to download portraits.")
        return

    rows: list[tuple[str, int, list[str]]] = []
    for d in sorted(db_root.iterdir()):
        if not d.is_dir():
            continue
        imgs = [p for p in d.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}]
        sources = sorted({
            "hf" if "fromhf" in p.name else
            "wiki" if "wiki" in p.name else
            "local"
            for p in imgs
        })
        rows.append((d.name, len(imgs), sources))

    total_people = len(rows)
    total_images = sum(n for _, n, _ in rows)
    print(f"\nOffline portrait DB: {db_root}")
    print(f"  {total_people} people  ·  {total_images} images total\n")
    if not rows:
        print("  (empty)")
        return
    print(f"  {'slug':<35}  {'images':>6}  sources")
    print(f"  {'-'*35}  {'------':>6}  -------")
    for slug, n, srcs in rows:
        print(f"  {slug:<35}  {n:>6}  {', '.join(srcs)}")
    print()
    tier = "large" if total_people >= 500 else "medium" if total_people >= 50 else "small"
    print(f"  Coverage: {tier} offline DB ({total_people} people).")
    if tier == "small":
        print("  At demo time, anyone NOT listed above will be fetched from Wikipedia (~2s).")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Pre-download portraits for offline use on demo day.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument(
        "--db-root",
        default=str(_ROOT / "datasets/celeb_refdb"),
        metavar="DIR",
        help="output directory (default: datasets/celeb_refdb)",
    )
    ap.add_argument(
        "--slugs", nargs="+", metavar="NAME",
        help=(
            "person slugs to fetch, e.g. --slugs marc_pollefeys elon_musk. "
            "Use underscores.  If omitted with --from-hf, downloads all identities "
            "found in that dataset."
        ),
    )
    ap.add_argument(
        "--from-hf", metavar="REPO_ID", default="",
        help=(
            "HuggingFace named-identity face dataset to try first "
            "(e.g. nateraw/vggface2).  "
            "Falls back to Wikipedia for any slug not found in the dataset."
        ),
    )
    ap.add_argument(
        "--hf-max-per-person", type=int, default=5, metavar="N",
        help="max HF images per person (default: 5)",
    )
    ap.add_argument(
        "--hf-max-scan", type=int, default=500_000, metavar="N",
        help="max HF rows to scan (default: 500000)",
    )
    ap.add_argument(
        "--no-wiki-fallback", action="store_true",
        help="do not fall back to Wikipedia for slugs missing from HF",
    )
    ap.add_argument(
        "--dry-run", action="store_true",
        help="show what would be downloaded without writing files",
    )
    ap.add_argument(
        "--coverage", action="store_true",
        help="show current offline DB coverage and exit",
    )
    args = ap.parse_args()

    db_root    = Path(args.db_root).resolve()
    slug_list  = args.slugs or []

    if args.coverage:
        coverage_report(db_root)
        return

    if not slug_list and not args.from_hf:
        ap.error("Provide --slugs <name ...> and/or --from-hf <repo_id>.")

    satisfied: set[str] = set()

    # ── Step 1: HuggingFace (optional, better quality for entertainment celebrities) ──
    if args.from_hf and slug_list:
        log.info("Step 1/2: trying HuggingFace dataset %r …", args.from_hf)
        satisfied = fetch_from_hf(
            args.from_hf, slug_list, db_root,
            max_per_person=args.hf_max_per_person,
            max_scan=args.hf_max_scan,
            dry_run=args.dry_run,
        )
    elif args.from_hf:
        log.info("--from-hf without --slugs: downloading all identities in dataset …")
        satisfied = fetch_from_hf(
            args.from_hf, slug_list or [], db_root,
            max_per_person=args.hf_max_per_person,
            max_scan=args.hf_max_scan,
            dry_run=args.dry_run,
        )

    # ── Step 2: Wikipedia fallback for slugs not covered by HF ───────────────
    wiki_needed = [s for s in slug_list if s not in satisfied]
    if wiki_needed and not args.no_wiki_fallback:
        if args.from_hf:
            log.info(
                "Step 2/2: Wikipedia fallback for %d slug(s) not in HF dataset: %s",
                len(wiki_needed), wiki_needed,
            )
        else:
            log.info("Fetching portraits from Wikipedia for %d slug(s) …", len(wiki_needed))

        wiki_results = fetch_from_wikipedia(wiki_needed, db_root, dry_run=args.dry_run)
        satisfied |= {s for s, ok in wiki_results.items() if ok}

    # ── Summary ───────────────────────────────────────────────────────────────
    if slug_list:
        print(f"\n{'='*55}")
        missing = [s for s in slug_list if s not in satisfied]
        print(f"  Requested : {len(slug_list)} people")
        print(f"  Cached    : {len(satisfied)} people")
        if missing:
            print(f"  Missing   : {missing}")
            print("  → Place photos manually in datasets/celeb_refdb/<slug>/")
        else:
            print("  All portraits available offline ✓")
        print(f"{'='*55}\n")

    if not args.dry_run:
        coverage_report(db_root)


if __name__ == "__main__":
    main()
