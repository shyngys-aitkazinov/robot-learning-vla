#!/usr/bin/env python3
"""Training entry point for the Eval3 RAG (reference-face) SmolVLA variant.

This script applies three import-time patches in the required order and then
delegates to ``lerobot-train`` (``lerobot.scripts.lerobot_train.main``):

1. ``eval3_lerobot_shim.apply()``      — fixes the transformers import-order crash
2. ``eval3_concat_patch.apply_concat_patch()`` — enables multi-dataset training
3. ``apply_rag_concat_patch()``        — layers reference-face injection on top

The resulting dataset exposes ``observation.images.front_refface`` per row.
The rename_map in the launcher routes it to ``camera2``; ``policy.empty_cameras=1``
leaves camera3 as an empty pad.

Usage (via the shell launcher)::

    ./scripts/eval3_rag/run_train_rag.sh

Or directly::

    EVAL3_RAG_REFDB=datasets/celeb_refdb \\
    EVAL3_EXTRA_REPOS=RobotLearningVLA/dataset_v4_taylor_middle,...  \\
      python scripts/eval3_rag/train_rag.py \\
        --policy.path=lerobot/smolvla_base \\
        --dataset.repo_id=RobotLearningVLA/dataset_v4_taylor_left \\
        --rename_map='{"observation.images.front":"observation.images.camera1","observation.images.front_refface":"observation.images.camera2"}' \\
        --policy.empty_cameras=1 \\
        --policy.freeze_vision_encoder=true \\
        --steps=50000 --batch_size=16
"""

import sys
from pathlib import Path

# ── path setup ───────────────────────────────────────────────────────────────
_SCRIPTS = Path(__file__).resolve().parent.parent   # eval1_train/robot-learning-vla/scripts/
_ROOT    = _SCRIPTS.parent                          # eval1_train/robot-learning-vla/
for _p in (_SCRIPTS, _ROOT / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# ── 1. lerobot import-order shim ─────────────────────────────────────────────
import eval3_lerobot_shim  # noqa: E402
eval3_lerobot_shim.apply()

# ── 2. multi-dataset concat patch ────────────────────────────────────────────
import eval3_concat_patch  # noqa: E402
eval3_concat_patch.apply_concat_patch()

# ── 3. RAG reference-face layer ──────────────────────────────────────────────
from eval3_rag.concat_patch_rag import apply_rag_concat_patch  # noqa: E402
apply_rag_concat_patch()

# ── 4. delegate to lerobot-train ─────────────────────────────────────────────
from lerobot.scripts.lerobot_train import main  # noqa: E402

if __name__ == "__main__":
    main()
