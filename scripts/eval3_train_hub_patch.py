"""Optional per-checkpoint Hugging Face upload for Eval3 SmolVLA training.

Enable with::

    EVAL3_HUB_PUSH=1 EVAL3_HUB_REPO=RobotLearningVLA/my-model ./scripts/run_eval3_smolvla_aug_train.sh

Each save uploads ``checkpoint_dir/pretrained_model`` and tags ``step-XXXXXX``.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Callable


def _truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def _detect_train_utils():
    try:
        import lerobot.scripts.lerobot_train as train_mod  # noqa: F401
        from lerobot.utils.train_utils import save_checkpoint

        return save_checkpoint
    except ImportError:
        pass
    try:
        from lerobot.common.utils.train_utils import save_checkpoint

        return save_checkpoint
    except ImportError as exc:
        raise ImportError("Could not import lerobot save_checkpoint for hub patch.") from exc


def apply_hub_patch() -> None:
    if not _truthy("EVAL3_HUB_PUSH"):
        return
    repo_id = os.environ.get("EVAL3_HUB_REPO", "").strip()
    if not repo_id:
        logging.warning("EVAL3_HUB_PUSH is set but EVAL3_HUB_REPO is empty; hub patch disabled.")
        return

    orig_save: Callable[..., Any] = _detect_train_utils()

    def _upload_pretrained(checkpoint_dir: Path, step: int) -> None:
        pretrained = checkpoint_dir / "pretrained_model"
        if not pretrained.is_dir():
            logging.warning("[eval3 hub] No pretrained_model at %s — skip upload", pretrained)
            return
        from huggingface_hub import HfApi, create_repo

        api = HfApi()
        create_repo(repo_id, repo_type="model", exist_ok=True)
        logging.info("[eval3 hub] Uploading step %s → %s", step, repo_id)
        api.upload_folder(
            folder_path=str(pretrained),
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"Eval3 SmolVLA checkpoint step {int(step)}",
        )
        tag = f"step-{int(step):06d}"
        api.create_tag(repo_id, tag=tag, repo_type="model")
        logging.info("[eval3 hub] Tagged %s as %s", repo_id, tag)

    def patched_save_checkpoint(checkpoint_dir, step, cfg, policy, **kwargs):
        orig_save(checkpoint_dir=checkpoint_dir, step=step, cfg=cfg, policy=policy, **kwargs)
        try:
            _upload_pretrained(Path(checkpoint_dir), int(step))
        except Exception as exc:
            logging.warning("[eval3 hub] Upload failed at step %s: %s", step, exc)

    # Patch both the utils module and lerobot_train if it imported save_checkpoint directly.
    import lerobot.utils.train_utils as train_utils

    train_utils.save_checkpoint = patched_save_checkpoint
    try:
        import lerobot.scripts.lerobot_train as train_mod

        if hasattr(train_mod, "save_checkpoint"):
            train_mod.save_checkpoint = patched_save_checkpoint
    except ImportError:
        pass

    logging.info("[eval3 hub] Patched save_checkpoint → %s (tags step-XXXXXX)", repo_id)
