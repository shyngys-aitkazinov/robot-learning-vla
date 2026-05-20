"""fresh_start — a from-scratch SmolVLA fine-tuning pipeline for lerobot.

Importing this package immediately applies the GROOT import shim (see
:mod:`fresh_start.groot_shim`) so that ``import lerobot.policies`` works on a
machine with ``transformers`` installed.

Modules:
    groot_shim      — import-time workaround for lerobot's GROOT crash
    config          — all configurable options (draccus dataclasses)
    transforms_ext  — custom torchvision-v2 image transforms
    augmentation    — builds lerobot's image-transforms config from AugmentationConfig
    merge_datasets  — merge the per-celebrity datasets into one training corpus
    train           — the training launcher (entry point)
    verify          — staged sanity checks + smoke train
"""

from . import groot_shim

groot_shim.apply()

__all__ = ["groot_shim"]
