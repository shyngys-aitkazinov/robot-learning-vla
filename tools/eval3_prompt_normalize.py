#!/usr/bin/env python3
"""Normalize Eval 3 deploy prompts to the canonical training distribution.

v10 training uses ``EVAL3_TASK_AUG_CANONICAL_P=1.0``, so the policy sees
``Place the coke on <Celeb>`` (no ``the`` before the name). Eval-day wording
often includes ``the`` or different casing — this module rewrites those variants
before ``predict_action`` without a VLM (Eval 1-style spatial routing is out of
scope for Eval 3 celebrity names).
"""

from __future__ import annotations

import re
from typing import NamedTuple

# Keep in sync with tools/eval3_check_deploy_command.py and scripts/eval3_dataset_prep.py.
CANONICAL_TASKS: frozenset[str] = frozenset(
    {
        "Place the coke on Taylor Swift",
        "Place the coke on Yann LeCun",
        "Place the coke on Barack Obama",
    }
)

CELEB_CANONICAL: dict[str, str] = {
    "taylor swift": "Place the coke on Taylor Swift",
    "yann lecun": "Place the coke on Yann LeCun",
    "barack obama": "Place the coke on Barack Obama",
}

# Aliases seen in recordings / eval sheets.
CELEB_ALIASES: dict[str, str] = {
    "taylor": "taylor swift",
    "swift": "taylor swift",
    "lecun": "yann lecun",
    "yann": "yann lecun",
    "obama": "barack obama",
    "barack": "barack obama",
}

_PLACE_RE = re.compile(
    r"^\s*place\s+the\s+coke\s+on\s+(?:the\s+)?(.+?)\s*(?:print)?\s*$",
    re.IGNORECASE,
)


class NormalizeResult(NamedTuple):
    raw: str
    normalized: str
    changed: bool
    matched_celeb: str | None


def _collapse_ws(text: str) -> str:
    return " ".join(text.strip().split())


def _match_celeb(fragment: str) -> str | None:
    key = _collapse_ws(fragment).lower()
    if key in CELEB_CANONICAL:
        return key
    if key in CELEB_ALIASES:
        return CELEB_ALIASES[key]
    for celeb_key in CELEB_CANONICAL:
        if celeb_key in key or key in celeb_key:
            return celeb_key
    return None


def normalize_eval3_task(task: str, *, allow_slot_print: bool = False) -> NormalizeResult:
    """Map user task text to a canonical Eval 3 celebrity prompt when possible.

    Parameters
    ----------
    task:
        Raw instruction from CLI, stdin, or eval sheet.
    allow_slot_print:
        If False (default), strings like ``Place the coke on the left print`` are
        left unchanged so ``--target_slot`` overrides remain explicit.
    """
    raw = str(task)
    collapsed = _collapse_ws(raw)

    if collapsed in CANONICAL_TASKS:
        return NormalizeResult(raw=raw, normalized=collapsed, changed=False, matched_celeb=None)

    # Legacy exact replacements (same as eval3_external_vla_data.canonicalize_task).
    legacy = {
        "Place the coke on the Taylor Swift": "Place the coke on Taylor Swift",
        "Place the coke on the Yann LeCun": "Place the coke on Yann LeCun",
        "Place the coke on the Barack Obama": "Place the coke on Barack Obama",
    }
    if collapsed in legacy:
        target = legacy[collapsed]
        celeb = _match_celeb(collapsed)
        return NormalizeResult(raw=raw, normalized=target, changed=True, matched_celeb=celeb)

    m = _PLACE_RE.match(collapsed)
    if not m:
        return NormalizeResult(raw=raw, normalized=collapsed, changed=collapsed != raw, matched_celeb=None)

    fragment = m.group(1).strip()
    if not allow_slot_print and fragment.lower() in {"left", "middle", "right", "left print", "middle print", "right print"}:
        return NormalizeResult(raw=raw, normalized=collapsed, changed=collapsed != raw, matched_celeb=None)

    celeb_key = _match_celeb(fragment)
    if celeb_key is None:
        return NormalizeResult(raw=raw, normalized=collapsed, changed=collapsed != raw, matched_celeb=None)

    target = CELEB_CANONICAL[celeb_key]
    return NormalizeResult(raw=raw, normalized=target, changed=target != collapsed, matched_celeb=celeb_key)
