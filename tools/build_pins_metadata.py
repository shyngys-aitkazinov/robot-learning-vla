#!/usr/bin/env python3
"""Build a metadata index of the Pins Face Recognition dataset.

Scans every ``pins_<Name>/`` subdirectory under the dataset root and emits a
single JSON file recording, per celebrity:

  * ``name``               — canonical display name to use in task strings
                             (auto-titlecased from the slug; manual overrides
                             for known misspellings via ``NAME_OVERRIDES``).
  * ``slug``               — the directory name minus the ``pins_`` prefix
                             (what's actually on disk; never edit).
  * ``dir``                — repo-relative path to the directory.
  * ``n_images``           — count of ``.jpg`` files.
  * ``total_bytes``        — sum of file sizes on disk.
  * ``held_out``           — True if the canonical name matches a TOY
                             identity (Taylor Swift / Yann LeCun / Barack
                             Obama). The ChArUco post-processor must
                             EXCLUDE these from the OOD training pool.
  * ``aspect_counts``      — portrait / landscape / square histogram.
  * ``aspect_portrait_frac`` — convenience scalar for filtering.
  * ``long_edge_px``       — min / median / max of max(width, height).

The image-stat fields require opening each JPG header (PIL is lazy, only
reads dimensions); ~17.5k headers take ~15 s on a Mac SSD. Pass
``--no-image-stats`` to skip and just record counts.

Used by the ChArUco post-processor (docs/eval3/charuco_pipeline.md §5) to
draw a celebrity per episode and pick one of their images to warp onto the
board area.

Usage:
  python tools/build_pins_metadata.py
  python tools/build_pins_metadata.py --root datasets/pins-face-recognition/105_classes_pins_dataset \\
                                       --out datasets/pins-face-recognition.json
  python tools/build_pins_metadata.py --no-image-stats
"""
from __future__ import annotations

import argparse
import json
import statistics
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    raise SystemExit("Pillow required: uv pip install pillow")


# Manual overrides for slugs whose simple title-cased form is wrong.
# Keys are the lowercased Pins directory name (after stripping 'pins_').
# All names are pushed through `_to_ascii` afterwards (NFKD + ascii encode), so
# accents added here will be silently folded — there's no point spelling them
# with diacritics, but doing so is harmless.
NAME_OVERRIDES: dict[str, str] = {
    # --- Misspellings in the Pins source ---
    "kiernen shipka":         "Kiernan Shipka",       # Pins typo: "Kiernen" -> "Kiernan"
    "alycia dabnem carey":    "Alycia Debnam-Carey",  # Pins typo + dropped hyphen
    "henry cavil":            "Henry Cavill",         # Pins drops the second 'l'

    # --- Mc* capitalization (the letter after "Mc" is capitalised in canonical English usage) ---
    "jake mcdorman":          "Jake McDorman",
    "katharine mcphee":       "Katharine McPhee",

    # --- Stage-name preference ---
    # Pins records the legal first-three-names; the TA prompt is almost certainly
    # the stage name, so canonicalise.
    "shakira isabel mebarak": "Shakira",
}


def _to_ascii(s: str) -> str:
    """Strip diacritics so canonical names contain only ASCII letters.

    The TA prompts at eval time are typed by hand on a US keyboard and the
    SmolVLA tokenizer treats e.g. 'Á' and 'A' as distinct tokens — keeping
    every name ASCII avoids both classes of mismatch. NFKD decomposes
    characters like 'Á' into 'A' + combining-acute; the .encode/.decode
    drops the combining marks.
    """
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")

# Identity holdout list — the TOY celebrities scored in Eval 3 runs 1–6.
# Any celebrity whose canonical name matches an entry here is flagged
# held_out=True so the post-processor can exclude them from OOD pools.
TOY_HOLDOUT: set[str] = {"Taylor Swift", "Yann LeCun", "Barack Obama"}


def canonical_name(slug: str) -> str:
    key = slug.lower().strip()
    if key in NAME_OVERRIDES:
        name = NAME_OVERRIDES[key]
    elif slug == slug.lower():
        # Title-case the fully-lowercase slugs (e.g. "barack obama" -> "Barack Obama").
        name = slug.title()
    else:
        # Mixed-case slugs (e.g. "Adriana Lima", "K.J. Apa") are kept verbatim.
        name = slug
    return _to_ascii(name)


def image_stats(images: list[Path]) -> dict | None:
    widths: list[int] = []
    heights: list[int] = []
    long_edges: list[int] = []
    portrait = landscape = square = 0
    for p in images:
        try:
            with Image.open(p) as im:
                w, h = im.size
        except Exception:
            continue
        widths.append(w)
        heights.append(h)
        long_edges.append(max(w, h))
        if h > w:
            portrait += 1
        elif w > h:
            landscape += 1
        else:
            square += 1
    if not long_edges:
        return None
    total = portrait + landscape + square
    return {
        "aspect_counts": {"portrait": portrait, "landscape": landscape, "square": square},
        "aspect_portrait_frac": round(portrait / total, 3) if total else 0.0,
        "long_edge_px": {
            "min": min(long_edges),
            "median": int(statistics.median(long_edges)),
            "max": max(long_edges),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--root", type=Path,
                    default=Path("datasets/pins-face-recognition/105_classes_pins_dataset"))
    ap.add_argument("--out", type=Path,
                    default=Path("datasets/pins-face-recognition.json"))
    ap.add_argument("--no-image-stats", action="store_true",
                    help="Skip per-image header reads (no aspect / resolution fields).")
    args = ap.parse_args()

    if not args.root.is_dir():
        raise SystemExit(
            f"--root {args.root} does not exist. Run scripts/download_pins_faces.sh first."
        )

    pins_dirs = sorted(
        d for d in args.root.iterdir() if d.is_dir() and d.name.startswith("pins_")
    )
    if not pins_dirs:
        raise SystemExit(f"no pins_<Name>/ dirs under {args.root}")

    cwd = Path.cwd()
    print(f"Indexing {len(pins_dirs)} celebrity directories under {args.root} ...")

    celebrities: list[dict] = []
    total_images = 0
    total_bytes = 0
    for i, d in enumerate(pins_dirs, 1):
        slug = d.name[len("pins_"):]
        name = canonical_name(slug)
        images = sorted(d.glob("*.jpg"))
        bytes_ = sum(p.stat().st_size for p in images)
        try:
            rel = d.relative_to(cwd)
        except ValueError:
            rel = d
        entry: dict = {
            "name": name,
            "slug": slug,
            "dir": str(rel),
            "n_images": len(images),
            "total_bytes": bytes_,
            "held_out": name in TOY_HOLDOUT,
        }
        if not args.no_image_stats and images:
            stats = image_stats(images)
            if stats is not None:
                entry.update(stats)
        celebrities.append(entry)
        total_images += len(images)
        total_bytes += bytes_
        if i % 25 == 0 or i == len(pins_dirs):
            print(f"  ... {i}/{len(pins_dirs)} done")

    payload = {
        "dataset": "Pins Face Recognition",
        "source_url": "https://www.kaggle.com/datasets/hereisburak/pins-face-recognition",
        "root": str(args.root),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "total_celebrities": len(celebrities),
        "total_images": total_images,
        "total_bytes": total_bytes,
        "held_out_identities_target": sorted(TOY_HOLDOUT),
        "held_out_identities_present": sorted(
            c["name"] for c in celebrities if c["held_out"]
        ),
        "name_overrides_applied": dict(NAME_OVERRIDES),
        "celebrities": celebrities,
    }

    # Defensive: confirm every canonical name is pure ASCII before writing.
    non_ascii = [c["name"] for c in celebrities if not c["name"].isascii()]
    if non_ascii:
        raise SystemExit(
            f"non-ASCII names slipped through canonicalization: {non_ascii}. "
            f"Check NAME_OVERRIDES and _to_ascii() in this script."
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print()
    print(f"Wrote {args.out}")
    print(f"  total: {len(celebrities)} celebrities, {total_images} images, "
          f"{total_bytes / 1024 / 1024:.0f} MB on disk")
    print(f"  TOY holdout targets       : {sorted(TOY_HOLDOUT)}")
    print(f"  TOY holdout present in set: {payload['held_out_identities_present']}")
    n_overrides_used = sum(
        1 for c in celebrities if c["slug"].lower() in NAME_OVERRIDES
    )
    if n_overrides_used:
        print(f"  name overrides applied    : {n_overrides_used}")


if __name__ == "__main__":
    main()
