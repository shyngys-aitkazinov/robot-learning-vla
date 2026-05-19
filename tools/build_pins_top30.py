#!/usr/bin/env python3
"""Filter the Pins metadata to a curated "top 30 most globally famous" subset.

Reads ``datasets/pins-face-recognition.json`` (built by
``tools/build_pins_metadata.py``), filters down to the 30 celebrities below,
and writes ``datasets/pins-face-recognition-top30.json`` with the same
per-celebrity schema plus two extra fields:

  * ``rank``      — 1..30, ordered by global pop-culture recognizability.
  * ``category``  — coarse bucket: actor / athlete / tech / music / model.

The ranking is a judgement call optimised for the Eval 3 OOD use case:
celebrities the SmolVLM backbone is most likely to recognise zero-shot and
that the TAs are most likely to use in runs 7–9. Edit ``TOP30`` below to
adjust.

Excluded: the two TOY identities present in Pins (Taylor Swift, Barack
Obama) — those are scored in runs 1–6 and must not appear in OOD training
pools (their ``held_out`` flag is also True in the source JSON).

Usage:
  python tools/build_pins_top30.py
  python tools/build_pins_top30.py --src datasets/pins-face-recognition.json \\
                                    --out datasets/pins-face-recognition-top30.json
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


# Curated list — ordered by global pop-culture recognizability.
# Each entry is (canonical_name, category). Names MUST exactly match the
# ``name`` field in the source JSON; the script errors out if any miss.
TOP30: list[tuple[str, str]] = [
    # --- S tier: indisputably most famous globally (athletes, music) ---
    ("Cristiano Ronaldo",   "athlete"),  # most-followed athlete worldwide
    ("Lionel Messi",        "athlete"),
    ("Rihanna",             "music"),    # music + Fenty empire
    ("Elon Musk",           "tech"),     # most-discussed public figure currently

    # --- S tier: A-list actors with massive franchise presence ---
    ("Leonardo DiCaprio",   "actor"),
    ("Tom Cruise",          "actor"),
    ("Dwayne Johnson",      "actor"),    # The Rock
    ("Robert Downey Jr",    "actor"),    # Iron Man + Oppenheimer Oscar
    ("Scarlett Johansson",  "actor"),
    ("Jennifer Lawrence",   "actor"),

    # --- A tier: A-list, current peak relevance ---
    ("Margot Robbie",       "actor"),    # Barbie 2023
    ("Zendaya",             "actor"),    # Euphoria / Dune / Spider-Man
    ("Emma Watson",         "actor"),    # Hermione + activism profile
    ("Tom Holland",         "actor"),    # Spider-Man (current)
    ("Keanu Reeves",        "actor"),    # John Wick / Matrix / cultural icon

    # --- A tier: tech titans ---
    ("Bill Gates",          "tech"),
    ("Mark Zuckerberg",     "tech"),
    ("Jeff Bezos",          "tech"),

    # --- A tier: MCU core (high recognition from franchise exposure) ---
    ("Chris Evans",         "actor"),    # Captain America
    ("Chris Hemsworth",     "actor"),    # Thor

    # --- B tier: A-list veterans + huge franchise stars ---
    ("Hugh Jackman",        "actor"),    # Wolverine
    ("Morgan Freeman",      "actor"),    # legendary
    ("Johnny Depp",         "actor"),    # Pirates of the Caribbean
    ("Christian Bale",      "actor"),    # Batman + Oscar
    ("Anne Hathaway",       "actor"),
    ("Emma Stone",          "actor"),    # 2x Oscar winner
    ("Selena Gomez",        "music"),    # music + most-followed on Instagram historically
    ("Millie Bobby Brown",  "actor"),    # Stranger Things lead
    ("Jason Momoa",         "actor"),    # Aquaman / GoT
    ("Mark Ruffalo",        "actor"),    # Hulk + Oscar nominee
]


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--src", type=Path, default=Path("datasets/pins-face-recognition.json"))
    ap.add_argument("--out", type=Path, default=Path("datasets/pins-face-recognition-top30.json"))
    args = ap.parse_args()

    if not args.src.exists():
        raise SystemExit(
            f"--src {args.src} not found. Run tools/build_pins_metadata.py first."
        )

    src = json.loads(args.src.read_text(encoding="utf-8"))
    by_name = {c["name"]: c for c in src["celebrities"]}

    # Verify every TOP30 name exists in the source, and not held-out.
    missing = [n for (n, _) in TOP30 if n not in by_name]
    if missing:
        raise SystemExit(
            f"the following TOP30 names are not present in {args.src}: {missing}\n"
            f"(Source has {len(by_name)} celebrities — check spelling.)"
        )
    held_out = [n for (n, _) in TOP30 if by_name[n]["held_out"]]
    if held_out:
        raise SystemExit(
            f"TOP30 must not include held-out (TOY) identities: {held_out}"
        )
    if len(TOP30) != 30:
        raise SystemExit(f"TOP30 must have exactly 30 entries, has {len(TOP30)}")

    # Build the filtered list, preserving the TOP30 order and adding rank+category.
    top_celebs = []
    for rank, (name, cat) in enumerate(TOP30, start=1):
        c = dict(by_name[name])  # shallow copy
        c["rank"] = rank
        c["category"] = cat
        top_celebs.append(c)

    cat_counts: dict[str, int] = {}
    for _, cat in TOP30:
        cat_counts[cat] = cat_counts.get(cat, 0) + 1

    payload = {
        "dataset": "Pins Face Recognition — Top 30 (curated subset)",
        "source": None,
        "source_url": src.get("source_url"),
        "source_json": str(args.src),
        "root": src.get("root"),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "total_celebrities": len(top_celebs),
        "total_images": sum(c["n_images"] for c in top_celebs),
        "total_bytes": sum(c["total_bytes"] for c in top_celebs),
        "category_counts": cat_counts,
        "selection_rationale": (
            "Ranked by global pop-culture recognizability, optimised for the "
            "Eval 3 OOD use case: celebrities the SmolVLM backbone is most "
            "likely to recognise zero-shot. Excludes TOY identities scored "
            "in runs 1–6."
        ),
        "held_out_identities_target": src.get("held_out_identities_target"),
        "held_out_identities_present": src.get("held_out_identities_present"),
        "name_overrides_applied": src.get("name_overrides_applied"),
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


if __name__ == "__main__":
    main()
