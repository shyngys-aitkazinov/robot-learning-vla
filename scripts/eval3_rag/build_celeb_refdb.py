#!/usr/bin/env python3
"""Build (or update) the celebrity reference-face DB used by the RAG SmolVLA.

Aggregates portrait images from existing local datasets into a unified DB at
``datasets/celeb_refdb/`` (override with ``--db-root``).

Images in celeb_refdb/ are used for:
  1. Synth data generation (pool JSON → ChArUco compositing)
  2. Training reference injection  (random sample per step)
  3. Inference reference injection (canonical/first image)

More unique images per celebrity → better synth diversity + better training
generalization. The pool JSON sorts images by resolution (largest first) so the
synth generator always composites from the highest-quality image available.

── Sources ──────────────────────────────────────────────────────────────────

A) Local project datasets (already on disk, no download needed):
   a. datasets/in-distribution-eval-3/       — course TOY portraits (~1400px)
   b. datasets/out-distribution-eval-3/      — Wikimedia TOY portraits (~1575px)
   c. datasets/out-distribution-eval-3-pins/ — 1 Pins image per OOD celeb (302–564px)

   OOD celebrities only have 1 image each from source (c). This limits synth to
   1×DISTRACTORS episodes per OOD dataset instead of MAX_PHOTOS×DISTRACTORS.

B) HuggingFace datasets (``--from-hf REPO_ID``, streams — no full download):
   Recommended for OOD celebrities. Streams the dataset, extracts images whose
   label matches a known celebrity slug, and saves them to celeb_refdb/.
   Requires: pip install datasets pillow

   Suitable datasets:
   • leondgarse/celeba_hq_face_recognition — 30k images at 1024×1024px.
     Has 6,217 named identities covering actors, musicians, athletes.
     label field = "label" (ClassLabel with names like "Cristiano_Ronaldo").
   • HuggingFaceM4/VGGFace2_sample         — sample of VGGFace2 (1024px).
   • Any HF image classification dataset with (image, label) fields where
     label names match celebrity slugs after normalisation.

C) Local named-folder datasets (``--from-dir DIR``):
   Any directory whose sub-folders are named after celebrities.
   Works with the full Pins dataset, VGGFace2, or custom collections.

Usage::

    # A — local build from the three existing project datasets (no download):
    python scripts/eval3_rag/build_celeb_refdb.py

    # B — stream from a HuggingFace dataset (recommended for OOD quality):
    python scripts/eval3_rag/build_celeb_refdb.py \\
        --from-hf leondgarse/celeba_hq_face_recognition \\
        --hf-max-per-celeb 10

    # B — with a different image column name or split:
    python scripts/eval3_rag/build_celeb_refdb.py \\
        --from-hf leondgarse/celeba_hq_face_recognition \\
        --hf-split train --hf-image-col image --hf-label-col label \\
        --hf-max-per-celeb 10

    # C — import from a local named-folder face dataset:
    python scripts/eval3_rag/build_celeb_refdb.py \\
        --from-dir /path/to/pins/105_classes_pins_dataset

    # Add individual images for a specific celebrity:
    python scripts/eval3_rag/build_celeb_refdb.py \\
        --add-images elon_musk=/path/to/elon1.jpg,/path/to/elon2.jpg

    # Dry-run any command without writing files:
    python scripts/eval3_rag/build_celeb_refdb.py --from-hf leondgarse/celeba_hq_face_recognition --dry-run

    # List available slugs from local sources:
    python scripts/eval3_rag/build_celeb_refdb.py --list

After adding images, rebuild the pool JSON:
    python scripts/eval3_rag/build_rag_pool_json.py --stats
    python scripts/eval3_rag/build_rag_pool_json.py
"""
from __future__ import annotations

import argparse
import io
import logging
import re
import shutil
import sys
from pathlib import Path


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

_SCRIPT = Path(__file__).resolve()
_ROOT   = _SCRIPT.parent.parent.parent  # …/robot-learning-vla/


# ---------------------------------------------------------------------------
# Slug normalisation helpers  (generic — no hardcoded names)
# ---------------------------------------------------------------------------

def _label_to_slug(raw: str) -> str:
    """Convert any dataset label or folder name to a filesystem slug.

    Strips common dataset prefixes (``pins_``, ``class_``, ``celeb_``) and
    slugifies the result.  Works for any person name — no hardcoded list.

    Examples::

        "Cristiano_Ronaldo"  → "cristiano_ronaldo"
        "pins_Taylor_Swift"  → "taylor_swift"
        "Marc Pollefeys"     → "marc_pollefeys"
    """
    name = raw.lower().strip()
    for prefix in ("pins_", "class_", "celeb_", "celebrity_"):
        if name.startswith(prefix):
            name = name[len(prefix):]
    return re.sub(r"[^a-z0-9]+", "_", name).strip("_")


# ---------------------------------------------------------------------------
# Local source dataset definitions
# ---------------------------------------------------------------------------

import json as _json

_IN_DIST_JSON  = _ROOT / "datasets/in-distribution-eval-3.json"
_OUT_DIST_JSON = _ROOT / "datasets/out-distribution-eval-3.json"
_PINS_OOD_JSON = _ROOT / "datasets/out-distribution-eval-3-pins.json"


def _load_sources() -> dict[str, list[Path]]:
    """Return {slug: [path, ...]} from the three local source datasets."""
    slug_to_paths: dict[str, list[Path]] = {}

    def _add(json_path: Path, image_field: str = "pdf_image") -> None:
        if not json_path.exists():
            log.warning("source JSON not found: %s", json_path)
            return
        data = _json.loads(json_path.read_text())
        root = Path(data.get("root", "."))
        for celeb in data.get("celebrities", []):
            slug = celeb.get("slug")
            if not slug:
                continue
            slug_to_paths.setdefault(slug, [])
            celeb_dir = _ROOT / root / slug
            if not celeb_dir.is_dir():
                single = celeb.get(image_field) or celeb.get("source_pins_image")
                if single:
                    p = (_ROOT / single).resolve()
                    if p.exists() and p not in slug_to_paths[slug]:
                        slug_to_paths[slug].append(p)
                continue
            imgs = sorted(celeb_dir.glob("*.jpg")) + sorted(celeb_dir.glob("*.png"))
            for p in imgs:
                if p not in slug_to_paths[slug]:
                    slug_to_paths[slug].append(p)

    _add(_IN_DIST_JSON,  "pdf_image")
    _add(_OUT_DIST_JSON, "pdf_image")
    _add(_PINS_OOD_JSON, "pdf_image")
    return slug_to_paths


# ---------------------------------------------------------------------------
# HuggingFace dataset fetcher
# ---------------------------------------------------------------------------

def fetch_from_hf(
    repo_id: str,
    db_root: Path,
    *,
    split: str = "train",
    image_col: str = "image",
    label_col: str = "label",
    max_per_celeb: int = 10,
    max_scan: int = 200_000,
    slug_filter: set[str] | None = None,
    dry_run: bool = False,
) -> dict[str, int]:
    """Stream images for target celebrities from a HuggingFace dataset.

    Works with any HF image-classification dataset where:
    - ``image_col`` contains PIL Images (or raw bytes)
    - ``label_col`` contains either integer ClassLabel IDs or string class names
      whose values (after normalisation) match celebrity slugs

    Images are saved as ``celeb_refdb/<slug>/fromhf_<NNN>.jpg``.
    Streaming means no full dataset download — only rows needed are fetched.

    Args:
        repo_id:        HuggingFace dataset repo (e.g. "leondgarse/celeba_hq_face_recognition")
        db_root:        celeb_refdb directory
        max_per_celeb:  stop collecting once this many new images saved per celebrity
        max_scan:       stop after scanning this many dataset rows (safety limit)
        slug_filter:    only collect these slugs; None = all known slugs

    Returns {slug: n_new_images_saved}.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        log.error("'datasets' library not installed.  Run: pip install datasets")
        return {}

    from PIL import Image as _PILImage

    # slug_filter=None means accept any person found in the dataset.
    # For large datasets (thousands of identities) always pass --slugs to limit scope.
    if slug_filter is None:
        log.warning(
            "HF stream: no --slugs filter — will download up to %d images for EVERY "
            "identity in the dataset.  Use --slugs name1,name2 to limit scope.",
            max_per_celeb,
        )
    target_slugs: set[str] | None = slug_filter

    log.info("HF stream: %r  split=%r  max_per_celeb=%d", repo_id, split, max_per_celeb)
    ds = load_dataset(repo_id, split=split, streaming=True, trust_remote_code=False)

    # Build integer-ID → slug map if the dataset has a ClassLabel feature.
    id_to_slug: dict[int, str] | None = None
    feats = getattr(ds, "features", None) or {}
    label_feat = feats.get(label_col)
    if label_feat is not None and hasattr(label_feat, "names"):
        id_to_slug = {}
        for idx, class_name in enumerate(label_feat.names):
            slug = _label_to_slug(class_name)
            if slug and (target_slugs is None or slug in target_slugs):
                id_to_slug[idx] = slug
        n_matched = len(set(id_to_slug.values()))
        log.info("ClassLabel feature: %d classes total, %d match target slugs",
                 len(label_feat.names), n_matched)
        if n_matched == 0 and target_slugs:
            log.warning(
                "No classes matched target slugs %s in dataset %r.  "
                "Check spelling — class names are slugified before matching "
                "(e.g. 'Elon_Musk' → 'elon_musk').",
                sorted(target_slugs), repo_id,
            )

    counts: dict[str, int] = {}  # slug → new images saved this run
    scanned = 0

    for row in ds:
        if scanned >= max_scan:
            log.info("HF stream: reached max_scan=%d — stopping", max_scan)
            break
        scanned += 1

        # Resolve label → slug
        raw_label = row.get(label_col)
        if raw_label is None:
            continue
        if id_to_slug is not None:
            slug = id_to_slug.get(int(raw_label))
        else:
            slug = _label_to_slug(str(raw_label)) or None

        if not slug:
            continue
        if target_slugs is not None and slug not in target_slugs:
            continue

        # Check quota
        celeb_out = db_root / slug
        already_saved = counts.get(slug, 0)
        already_on_disk = (
            len(list(celeb_out.glob("fromhf_*"))) if celeb_out.exists() else 0
        )
        if already_on_disk + already_saved >= max_per_celeb:
            if target_slugs is not None:
                remaining = [
                    s for s in target_slugs
                    if counts.get(s, 0) + (
                        len(list((db_root / s).glob("fromhf_*")))
                        if (db_root / s).exists() else 0
                    ) < max_per_celeb
                ]
                if not remaining:
                    log.info("HF stream: all target slugs satisfied — stopping early")
                    break
            continue

        # Decode image
        raw_img = row.get(image_col)
        if raw_img is None:
            continue
        try:
            if isinstance(raw_img, _PILImage.Image):
                pil_img = raw_img.convert("RGB")
            else:
                pil_img = _PILImage.open(io.BytesIO(raw_img)).convert("RGB")
        except Exception as exc:
            log.debug("image decode failed for slug=%r: %s", slug, exc)
            continue

        w, h = pil_img.size
        n_next = already_on_disk + already_saved + 1
        dst = celeb_out / f"fromhf_{n_next:03d}.jpg"

        if dry_run:
            log.info("[DRY-RUN] hf %r → %s  (%dx%d)", slug, dst.name, w, h)
        else:
            celeb_out.mkdir(parents=True, exist_ok=True)
            pil_img.save(dst, "JPEG", quality=95)
            log.info("hf: %-25s → %s  (%dx%d)", f"{slug!r}", dst.name, w, h)

        counts[slug] = already_saved + 1

    ok = sum(1 for n in counts.values() if n > 0)
    n_target = len(target_slugs) if target_slugs is not None else "all"
    log.info(
        "HF stream: scanned %d rows → %d/%s celebrities got new images",
        scanned, ok, n_target,
    )
    return counts


# ---------------------------------------------------------------------------
# --from-dir importer
# ---------------------------------------------------------------------------

def import_from_dir(
    face_dataset_dir: Path,
    db_root: Path,
    *,
    slug_filter: set[str] | None = None,
    dry_run: bool = False,
) -> dict[str, int]:
    """Import celebrity images from a named-folder face dataset into db_root.

    Scans sub-folders whose names match known celebrity slugs (after stripping
    dataset-specific prefixes like ``pins_``).  Images are copied into
    ``db_root/<slug>/fromdir_<NNN>.ext``.

    Returns {slug: n_images_written}.
    """
    if not face_dataset_dir.is_dir():
        log.error("--from-dir: directory not found: %s", face_dataset_dir)
        return {}

    _IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
    counts: dict[str, int] = {}
    for folder in sorted(face_dataset_dir.iterdir()):
        if not folder.is_dir():
            continue
        slug = _label_to_slug(folder.name)
        if not slug:
            log.debug("from-dir: empty slug for folder %r — skipping", folder.name)
            continue
        if slug_filter and slug not in slug_filter:
            continue

        imgs = sorted(
            p for p in folder.iterdir()
            if p.suffix.lower() in _IMG_EXTS
        )
        if not imgs:
            log.warning("from-dir: %r — no images in %s", slug, folder)
            continue

        celeb_out = db_root / slug
        existing = list(celeb_out.glob("fromdir_*")) if celeb_out.exists() else []
        next_idx = len(existing) + 1

        n = 0
        for i, src in enumerate(imgs, start=next_idx):
            dst = celeb_out / f"fromdir_{i:03d}{src.suffix.lower() or '.jpg'}"
            if dry_run:
                log.info("[DRY-RUN] from-dir %r: %s → %s", slug, src.name, dst.name)
            else:
                celeb_out.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
            n += 1

        counts[slug] = n
        log.info(
            "from-dir: %-25s  %3d image(s) from %s%s",
            f"{slug!r}", n, folder.name, "  [dry-run]" if dry_run else "",
        )

    return counts


# ---------------------------------------------------------------------------
# Local build logic
# ---------------------------------------------------------------------------

def build_db(
    db_root: Path,
    *,
    slug_filter: set[str] | None = None,
    extra: dict[str, list[Path]] | None = None,
    dry_run: bool = False,
) -> dict[str, int]:
    """Copy celebrity reference images from local source datasets into db_root."""
    sources = _load_sources()

    if extra:
        for slug, paths in extra.items():
            sources.setdefault(slug, [])
            for p in paths:
                if p not in sources[slug]:
                    sources[slug].append(p)

    if slug_filter:
        sources = {k: v for k, v in sources.items() if k in slug_filter}

    counts: dict[str, int] = {}
    for slug, src_paths in sorted(sources.items()):
        valid = [p for p in src_paths if p.exists()]
        if not valid:
            log.warning("slug=%r: no source images found", slug)
            continue

        celeb_out = db_root / slug
        if not dry_run:
            celeb_out.mkdir(parents=True, exist_ok=True)

        n = 0
        for i, src in enumerate(valid, start=1):
            dst = celeb_out / f"ref_{i:02d}{src.suffix.lower() or '.jpg'}"
            if dry_run:
                log.info("[DRY-RUN] %s → %s", src.name, dst)
            else:
                shutil.copy2(src, dst)
            n += 1

        counts[slug] = n
        log.info("slug=%-25s  %2d image(s)%s", f"{slug!r}", n,
                 "  [dry-run]" if dry_run else "")

    return counts


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_extra(spec: str) -> dict[str, list[Path]]:
    result: dict[str, list[Path]] = {}
    if not spec:
        return result
    for part in re.split(r";", spec):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            log.error("--add-images: expected 'slug=/path1,/path2', got %r", part)
            sys.exit(1)
        slug, paths_str = part.split("=", 1)
        slug = re.sub(r"[^a-z0-9]+", "_", slug.strip().lower())
        result[slug] = [Path(p.strip()) for p in paths_str.split(",") if p.strip()]
    return result


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build the celebrity reference-face DB for eval3_rag.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument(
        "--db-root", default=str(_ROOT / "datasets/celeb_refdb"),
        metavar="DIR",
        help="output directory (default: datasets/celeb_refdb)",
    )

    # ── local build ──────────────────────────────────────────────────────────
    local = ap.add_argument_group("local build (default mode)")
    local.add_argument(
        "--slugs", default="", metavar="SLUG[,SLUG...]",
        help="limit to these comma-separated slugs; default = all",
    )
    local.add_argument(
        "--add-images", default="", metavar="SLUG=/p1,/p2[;SLUG2=...]",
        help="inject extra local images for specific slugs (semicolon-separated)",
    )
    local.add_argument(
        "--list", action="store_true",
        help="print available slugs from local source datasets and exit",
    )

    # ── HuggingFace stream ────────────────────────────────────────────────────
    hf = ap.add_argument_group(
        "HuggingFace dataset stream (--from-hf)",
        "Stream images for target celebrities from a HuggingFace dataset.\n"
        "Only rows that match target celebrities are downloaded — no full dump.\n"
        "Recommended datasets:\n"
        "  leondgarse/celeba_hq_face_recognition  (1024px, named classes)\n"
        "  HuggingFaceM4/VGGFace2_sample          (1024px, named classes)\n"
        "Requires: pip install datasets pillow",
    )
    hf.add_argument(
        "--from-hf", default="", metavar="REPO_ID",
        help="HuggingFace dataset repo to stream from",
    )
    hf.add_argument(
        "--hf-split", default="train", metavar="SPLIT",
        help="dataset split (default: train)",
    )
    hf.add_argument(
        "--hf-image-col", default="image", metavar="COL",
        help="column name containing the image (default: image)",
    )
    hf.add_argument(
        "--hf-label-col", default="label", metavar="COL",
        help="column name containing the celebrity label (default: label)",
    )
    hf.add_argument(
        "--hf-max-per-celeb", type=int, default=10, metavar="N",
        help="max images to collect per celebrity (default: 10)",
    )
    hf.add_argument(
        "--hf-max-scan", type=int, default=500_000, metavar="N",
        help="max dataset rows to scan before giving up (default: 500000)",
    )

    # ── local named-folder import ─────────────────────────────────────────────
    imp = ap.add_argument_group("local named-folder import (--from-dir)")
    imp.add_argument(
        "--from-dir", default="", metavar="DIR",
        help=(
            "import from a local face dataset organised as <celeb_folder>/image.jpg. "
            "Works with the full Pins dataset, VGGFace2 test set, etc."
        ),
    )

    ap.add_argument("--dry-run", action="store_true",
                    help="print what would happen without writing files")

    args = ap.parse_args()
    db_root = Path(args.db_root).resolve()
    slug_filter = {s.strip() for s in args.slugs.split(",") if s.strip()} or None

    # ── --list ───────────────────────────────────────────────────────────────
    if args.list:
        sources = _load_sources()
        print("Available celebrity slugs in local source datasets:")
        for slug, paths in sorted(sources.items()):
            valid = sum(1 for p in paths if p.exists())
            print(f"  {slug:<30s}  {valid} source image(s)")
        return

    # ── --from-hf ────────────────────────────────────────────────────────────
    if args.from_hf:
        counts = fetch_from_hf(
            args.from_hf,
            db_root,
            split=args.hf_split,
            image_col=args.hf_image_col,
            label_col=args.hf_label_col,
            max_per_celeb=args.hf_max_per_celeb,
            max_scan=args.hf_max_scan,
            slug_filter=slug_filter,
            dry_run=args.dry_run,
        )
        total = sum(counts.values())
        print(f"\n{'[DRY-RUN] ' if args.dry_run else ''}"
              f"Saved {total} image(s) for {len(counts)} celebrity/ies → {db_root}")
        for slug, n in sorted(counts.items()):
            print(f"  {slug:<30s}  +{n} image(s)")
        if not args.dry_run and total:
            print("\nNow rebuild the pool JSON:")
            print("  python scripts/eval3_rag/build_rag_pool_json.py")
        return

    # ── --from-dir ────────────────────────────────────────────────────────────
    if args.from_dir:
        counts = import_from_dir(
            Path(args.from_dir).resolve(), db_root,
            slug_filter=slug_filter, dry_run=args.dry_run,
        )
        total = sum(counts.values())
        print(f"\n{'[DRY-RUN] ' if args.dry_run else ''}"
              f"Imported {total} image(s) for {len(counts)} celebrity/ies")
        if not args.dry_run and total:
            print("\nNow rebuild the pool JSON:")
            print("  python scripts/eval3_rag/build_rag_pool_json.py")
        return

    # ── default: local build ──────────────────────────────────────────────────
    extra = _parse_extra(args.add_images)
    log.info("DB root: %s%s", db_root, "  [DRY-RUN]" if args.dry_run else "")
    counts = build_db(db_root, slug_filter=slug_filter, extra=extra, dry_run=args.dry_run)

    total_slugs  = len(counts)
    total_images = sum(counts.values())
    print(f"\n{'[DRY-RUN] ' if args.dry_run else ''}Built {total_images} image(s)"
          f" across {total_slugs} celebrity slug(s) → {db_root}")
    if not args.dry_run and total_slugs:
        print("\nSlugs in DB:")
        for slug, n in sorted(counts.items()):
            print(f"  {slug:<30s}  {n} image(s)")
        print("\nNow rebuild the pool JSON:")
        print("  python scripts/eval3_rag/build_rag_pool_json.py")


if __name__ == "__main__":
    main()
