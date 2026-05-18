#!/usr/bin/env python3
"""Filter the Pins metadata to a curated "top 50 most globally famous" subset.

Same shape as ``tools/build_pins_top30.py`` (which produces a strict subset of
this list — ranks 1-30 are identical between the two files), just extended to
50 names. Adds 20 more celebrities who are widely recognizable globally but
slightly below the S/A-tier of the top-30 cut: GoT principals, MCU
secondaries, current music/pop names, and a few veteran A-listers.

Reads ``datasets/pins-face-recognition.json`` (built by
``tools/build_pins_metadata.py``), filters down to the 50 celebrities below,
and writes ``datasets/pins-face-recognition-top50.json`` with the same
per-celebrity schema plus two extra fields:

  * ``rank``      — 1..50, ordered by global pop-culture recognizability.
  * ``category``  — coarse bucket: actor / athlete / tech / music / tv.

Edit the ``TOP50`` list below to add/drop entries and re-run. The script
errors out cleanly if any name in ``TOP50`` doesn't match a ``name`` in
the source JSON, or if a TOY identity (Taylor Swift / Yann LeCun / Barack
Obama) slips in.

Usage:
  python tools/build_pins_top50.py
  python tools/build_pins_top50.py --src datasets/pins-face-recognition.json \\
                                    --out datasets/pins-face-recognition-top50.json
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


# Curated list — ordered by global pop-culture recognizability.
# Each entry is (canonical_name, category). Names MUST exactly match the
# ``name`` field in the source JSON; the script errors out if any miss.
# Ranks 1-30 are identical to tools/build_pins_top30.py — keep in sync.
TOP50: list[tuple[str, str]] = [
    # --- S tier (1-10): indisputably most famous globally ---
    ("Cristiano Ronaldo",   "athlete"),
    ("Lionel Messi",        "athlete"),
    ("Rihanna",             "music"),
    ("Elon Musk",           "tech"),
    ("Leonardo DiCaprio",   "actor"),
    ("Tom Cruise",          "actor"),
    ("Dwayne Johnson",      "actor"),
    ("Robert Downey Jr",    "actor"),
    ("Scarlett Johansson",  "actor"),
    ("Jennifer Lawrence",   "actor"),

    # --- A tier (11-20): A-list, current peak relevance + tech titans ---
    ("Margot Robbie",       "actor"),
    ("Zendaya",             "actor"),
    ("Emma Watson",         "actor"),
    ("Tom Holland",         "actor"),
    ("Keanu Reeves",        "actor"),
    ("Bill Gates",          "tech"),
    ("Mark Zuckerberg",     "tech"),
    ("Jeff Bezos",          "tech"),
    ("Chris Evans",         "actor"),
    ("Chris Hemsworth",     "actor"),

    # --- B tier (21-30): A-list veterans + huge franchise stars ---
    ("Hugh Jackman",        "actor"),
    ("Morgan Freeman",      "actor"),
    ("Johnny Depp",         "actor"),
    ("Christian Bale",      "actor"),
    ("Anne Hathaway",       "actor"),
    ("Emma Stone",          "actor"),
    ("Selena Gomez",        "music"),
    ("Millie Bobby Brown",  "actor"),
    ("Jason Momoa",         "actor"),
    ("Mark Ruffalo",        "actor"),

    # --- C tier (31-50): broad recognition, niche-popular, supporting franchises ---
    ("Natalie Portman",     "actor"),    # Black Swan + Star Wars, Oscar
    ("Tom Hardy",           "actor"),    # Mad Max / Venom / Dunkirk
    ("Gal Gadot",           "actor"),    # Wonder Woman
    ("Brie Larson",         "actor"),    # Captain Marvel + Oscar
    ("Henry Cavill",        "actor"),    # Superman + The Witcher
    ("Emilia Clarke",       "actor"),    # GoT — Daenerys
    ("Sophie Turner",       "actor"),    # GoT — Sansa
    ("Rami Malek",          "actor"),    # Bohemian Rhapsody Oscar + Mr Robot
    ("Ben Affleck",         "actor"),    # Batman + Argo Oscar
    ("Tom Hiddleston",      "actor"),    # Loki — current MCU lead
    ("Elizabeth Olsen",     "actor"),    # Scarlet Witch, MCU
    ("Miley Cyrus",         "music"),    # pop star
    ("Avril Lavigne",       "music"),    # pop-punk
    ("Megan Fox",           "actor"),    # Transformers, 2000s icon
    ("Penn Badgley",        "actor"),    # You + Gossip Girl
    ("Jeremy Renner",       "actor"),    # Hawkeye, MCU core
    ("Jimmy Fallon",        "tv"),       # late-night TV
    ("Anthony Mackie",      "actor"),    # Captain America (Falcon)
    ("Robert De Niro",      "actor"),    # legendary, decades-spanning
    ("Maisie Williams",     "actor"),    # GoT — Arya
]


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--src", type=Path, default=Path("datasets/pins-face-recognition.json"))
    ap.add_argument("--out", type=Path, default=Path("datasets/pins-face-recognition-top50.json"))
    args = ap.parse_args()

    if not args.src.exists():
        raise SystemExit(
            f"--src {args.src} not found. Run tools/build_pins_metadata.py first."
        )

    src = json.loads(args.src.read_text(encoding="utf-8"))
    by_name = {c["name"]: c for c in src["celebrities"]}

    missing = [n for (n, _) in TOP50 if n not in by_name]
    if missing:
        raise SystemExit(
            f"the following TOP50 names are not present in {args.src}: {missing}\n"
            f"(Source has {len(by_name)} celebrities — check spelling.)"
        )
    held_out = [n for (n, _) in TOP50 if by_name[n]["held_out"]]
    if held_out:
        raise SystemExit(
            f"TOP50 must not include held-out (TOY) identities: {held_out}"
        )
    if len(TOP50) != 50:
        raise SystemExit(f"TOP50 must have exactly 50 entries, has {len(TOP50)}")

    top_celebs: list[dict] = []
    for rank, (name, cat) in enumerate(TOP50, start=1):
        c = dict(by_name[name])
        c["rank"] = rank
        c["category"] = cat
        top_celebs.append(c)

    cat_counts: dict[str, int] = {}
    for _, cat in TOP50:
        cat_counts[cat] = cat_counts.get(cat, 0) + 1

    payload = {
        "dataset": "Pins Face Recognition — Top 50 (curated subset)",
        "source_json": str(args.src),
        "source_url": src.get("source_url"),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "total_celebrities": len(top_celebs),
        "total_images": sum(c["n_images"] for c in top_celebs),
        "total_bytes": sum(c["total_bytes"] for c in top_celebs),
        "category_counts": cat_counts,
        "selection_rationale": (
            "Ranked by global pop-culture recognizability, optimised for the "
            "Eval 3 OOD use case: celebrities the SmolVLM backbone is most "
            "likely to recognise zero-shot. Ranks 1-30 match build_pins_top30.py "
            "exactly; ranks 31-50 add broadly-recognisable supporting names "
            "(GoT principals, MCU secondaries, current music acts, veteran "
            "A-listers). Excludes TOY identities scored in runs 1-6."
        ),
        "celebrities": top_celebs,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"Wrote {args.out}")
    print(f"  {len(top_celebs)} celebrities, "
          f"{payload['total_images']} images, "
          f"{payload['total_bytes'] / 1024 / 1024:.0f} MB")
    print(f"  category breakdown: {cat_counts}")
    print(f"  top 5 by rank:")
    for c in top_celebs[:5]:
        print(f"    {c['rank']:2d}. {c['name']:25s} ({c['category']:8s})  "
              f"{c['n_images']} images")
    print(f"  bottom 5 by rank:")
    for c in top_celebs[-5:]:
        print(f"    {c['rank']:2d}. {c['name']:25s} ({c['category']:8s})  "
              f"{c['n_images']} images")


if __name__ == "__main__":
    main()
