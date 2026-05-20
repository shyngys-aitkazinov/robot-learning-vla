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

# Slot-bottleneck head XOR the legacy aux head — mutually exclusive (both
# monkey-patch SmolVLAPolicy.forward). The slot bottleneck (prefix-side
# classifier on image+language tokens, no suffix/action leak) is selected
# when EVAL3_SLOT_LOSS_WEIGHT > 0; otherwise fall back to the legacy aux
# head (itself a no-op unless EVAL3_AUX_POS_LOSS_WEIGHT > 0).
import os as _os  # noqa: E402

if _os.environ.get("EVAL3_SLOT_LOSS_WEIGHT", "0").strip() not in ("", "0", "0.0"):
    from eval3_smolvla_slot_bottleneck import apply as _eval3_slot_apply  # noqa: E402

    _eval3_slot_apply()
else:
    from eval3_smolvla_aux_head import apply as _eval3_aux_head_apply  # noqa: E402

    _eval3_aux_head_apply()

from eval3_train_hub_patch import apply_hub_patch as _eval3_hub_apply  # noqa: E402

_eval3_hub_apply()

from lerobot.scripts.lerobot_train import main  # noqa: E402


if __name__ == "__main__":
    main()
