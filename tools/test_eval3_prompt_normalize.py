#!/usr/bin/env python3
"""Tests for eval3_prompt_normalize (stdlib only)."""

from __future__ import annotations

import sys
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from eval3_prompt_normalize import CANONICAL_TASKS, normalize_eval3_task  # noqa: E402


def test_canonical_unchanged() -> None:
    for task in CANONICAL_TASKS:
        r = normalize_eval3_task(task)
        assert r.normalized == task
        assert not r.changed


def test_the_insertion() -> None:
    r = normalize_eval3_task("Place the coke on the Barack Obama")
    assert r.normalized == "Place the coke on Barack Obama"
    assert r.changed


def test_case_insensitive() -> None:
    r = normalize_eval3_task("place the coke on taylor swift")
    assert r.normalized == "Place the coke on Taylor Swift"
    assert r.changed


def test_extra_whitespace() -> None:
    r = normalize_eval3_task("  Place   the coke on   Yann LeCun  ")
    assert r.normalized == "Place the coke on Yann LeCun"


def test_slot_left_unchanged() -> None:
    r = normalize_eval3_task("Place the coke on the left print")
    assert r.normalized == "Place the coke on the left print"
    assert not r.matched_celeb


def test_unknown_passthrough() -> None:
    r = normalize_eval3_task("Pick up the bottle")
    assert r.normalized == "Pick up the bottle"


def main() -> int:
    test_canonical_unchanged()
    test_the_insertion()
    test_case_insensitive()
    test_extra_whitespace()
    test_slot_left_unchanged()
    test_unknown_passthrough()
    print("eval3_prompt_normalize: all tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
