"""Resolve torch device for Eval3 local scripts (Apple Silicon MPS, CUDA, CPU)."""

from __future__ import annotations

import torch


def resolve_eval3_device(arg: str = "auto") -> torch.device:
    """Prefer MPS on macOS when available, then CUDA, else CPU."""
    p = (arg or "auto").strip().lower()
    if p != "auto":
        return torch.device(arg)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
