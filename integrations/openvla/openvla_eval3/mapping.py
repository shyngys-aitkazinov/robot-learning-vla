"""SO-101 embodiment bridge — placeholders until stats are fitted from Hub/replay."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

# Typical SO-101 follower keys (LeRobot datasets); verify against meta.features["action"].names.
DEFAULT_SO101_ACTION_KEYS = (
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
)


def load_embodiment_stats(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def openvla_action_to_so101(
    raw: np.ndarray,
    *,
    stats: dict[str, Any],
    motion_gain: float = 1.0,
    openvla_slice: tuple[int, int] | None = None,
) -> dict[str, float]:
    """Map OpenVLA continuous vector → SO-101 goal dict.

    Without calibrated ``stats["mapping"]`` this applies only crude slicing/clamping.

    Parameters
    ----------
    raw:
        Flat action vector from ``predict_action``.
    stats:
        Loaded JSON from ``configs/embodiment_stats.json`` (see example schema).
    motion_gain:
        Scale applied before clamp (use <<1 for first hardware touches).
    openvla_slice:
        Optional ``(start, end)`` indices to take from ``raw`` if OpenVLA dim != 6.
    """
    vec = np.asarray(raw, dtype=np.float64).reshape(-1)
    if openvla_slice is not None:
        start, end = openvla_slice
        vec = vec[start:end]

    mapping = stats.get("mapping") or {}
    # Explicit per-output formula: {"shoulder_pan.pos": {"src_idx": 0, "scale": 1.0, "offset": 0.0}}
    keys = tuple(stats.get("so101_joint_names") or DEFAULT_SO101_ACTION_KEYS)
    limits = stats.get("joint_limits_deg") or {}

    out: dict[str, float] = {}
    if mapping:
        for key in keys:
            rule = mapping.get(key)
            if not rule:
                continue
            idx = int(rule["src_idx"])
            scale = float(rule.get("scale", 1.0))
            offset = float(rule.get("offset", 0.0))
            val = float(vec[idx]) * scale + offset
            lim = limits.get(key)
            if lim and isinstance(lim, (list, tuple)) and len(lim) == 2:
                val = float(np.clip(val, lim[0], lim[1]))
            out[key] = val * float(motion_gain)
        return out

    # Fallback: align first min(len(vec), len(keys)) dimensions (unsafe — for debugging only).
    n = min(len(vec), len(keys))
    for i in range(n):
        key = keys[i]
        val = float(vec[i]) * float(motion_gain)
        lim = limits.get(key)
        if lim and isinstance(lim, (list, tuple)) and len(lim) == 2:
            val = float(np.clip(val, lim[0], lim[1]))
        out[key] = val
    return out
