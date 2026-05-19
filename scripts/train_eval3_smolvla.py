#!/usr/bin/env python3
"""Fine-tune SmolVLA for Eval 3 (same CLI as ``lerobot-train``).

Install once in this venv::

    uv pip install transformers accelerate sentencepiece

Usage matches upstream, for example::

    python scripts/train_eval3_smolvla.py \\
      --policy.path=lerobot/smolvla_base \\
      --policy.push_to_hub=false \\
      --dataset.repo_id=RobotLearningVLA/taylor_swift_1 \\
      --dataset.video_backend=pyav \\
      --job_name=eval3_smolvla \\
      --steps=50000 \\
      --batch_size=8 \\
      --policy.device=mps \\
      --policy.compile_model=false \\
      --output_dir=outputs/train/eval3_smolvla
"""

from __future__ import annotations

import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from eval3_lerobot_shim import apply as _eval3_shim_apply

_eval3_shim_apply()

from eval3_concat_patch import apply_concat_patch as _eval3_concat_apply  # noqa: E402

_eval3_concat_apply()

# Optional auxiliary position-classification head (forces SmolVLA's
# hidden state to encode language-image binding). No-op unless
# EVAL3_AUX_POS_LOSS_WEIGHT > 0.
from eval3_smolvla_aux_head import apply as _eval3_aux_head_apply  # noqa: E402

_eval3_aux_head_apply()

from eval3_train_hub_patch import apply_hub_patch as _eval3_hub_apply  # noqa: E402

_eval3_hub_apply()

from lerobot.scripts.lerobot_train import main  # noqa: E402


if __name__ == "__main__":
    main()
