#!/usr/bin/env python3
"""Closed-loop deploy for the Eval3 RAG SmolVLA checkpoint.

Wraps the standard ``eval3_vla_deploy`` entry point with two additions:

1. **Reference face injection** — patches ``build_dataset_frame`` so every
   observation frame gets ``observation.images.front_refface`` set to the
   celebrity's reference portrait from the local DB.  The rename_map routes
   it to ``camera2`` for the frozen vision encoder.

2. **Task canonicalisation** — the model was trained with
   ``EVAL3_TASK_AUG_CANONICAL_P=1.0``, meaning the task string NEVER
   contained a celebrity name during training.  Passing "Place the coke on
   Elon Musk" at inference is out-of-distribution.  This script therefore
   replaces the task in ``--task`` with a canonical form (no celebrity name)
   before handing off to the deploy main, while still using the name to look
   up the reference image.

Training distribution:
  task    = "Place the coke on the person in the reference photo"  (canonical)
  camera2 = reference portrait of the target celebrity

Inference (this script):
  --task "Place the coke on Elon Musk"
       ↓ slug extracted  →  elon_musk
       ↓ reference image loaded from datasets/celeb_refdb/elon_musk/
       ↓ task replaced   →  "Place the coke on the person in the reference photo"
  model receives the same (task, camera2) distribution it was trained on.

Env vars
--------
EVAL3_RAG_REFDB           path to celeb_refdb (default: datasets/celeb_refdb)
EVAL3_RAG_HW              "H,W" reference image size (default: 480,640)
EVAL3_CANONICAL_TASK      override the canonical task string used at inference
                          (default: "Place the coke on the person in the
                           reference photo")
EVAL3_KEEP_TASK_NAME      set to 1 to skip canonicalisation and pass the
                          original --task string to the model as-is (ablation)

Usage::

    EVAL3_RAG_REFDB=datasets/celeb_refdb \\
      python scripts/eval3_rag/deploy_rag.py \\
        --policy.path=outputs/train/eval3_rag/checkpoints/050000/pretrained_model \\
        --rename_map='{"observation.images.front":"observation.images.camera1","observation.images.front_refface":"observation.images.camera2"}' \\
        --task="Place the coke on Elon Musk" \\
        --episode_time_s=20

Or via the launcher::

    ./scripts/eval3_rag/run_deploy_rag.sh "Place the coke on Elon Musk"
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

# ── path setup ────────────────────────────────────────────────────────────
_SCRIPTS = Path(__file__).resolve().parent.parent
_ROOT    = _SCRIPTS.parent
for _p in (_SCRIPTS, _ROOT / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# ── apply import-time shim (must happen before lerobot.policies) ─────────
import eval3_lerobot_shim  # noqa: E402
eval3_lerobot_shim.apply()

# ── parse task / rag-db from argv early so we can load the face image ────
from eval3_rag.reference_injector import (  # noqa: E402
    ReferenceImageInjector,
    task_to_celeb_slug,
)


def _argv_value(flag: str) -> str | None:
    for i, a in enumerate(sys.argv):
        if a.startswith(f"{flag}="):
            return a.split("=", 1)[1]
        if a == flag and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return None


def _replace_argv_task(new_task: str) -> None:
    """Replace the --task value in sys.argv with *new_task* in-place."""
    for i, a in enumerate(sys.argv):
        if a.startswith("--task="):
            sys.argv[i] = f"--task={new_task}"
            return
        if a == "--task" and i + 1 < len(sys.argv):
            sys.argv[i + 1] = new_task
            return
    # Not present in argv — append it so the deploy script sees it
    sys.argv.extend(["--task", new_task])


_task_raw = _argv_value("--task") or ""
_rag_db   = os.environ.get("EVAL3_RAG_REFDB", "datasets/celeb_refdb").strip()
_hw_raw   = os.environ.get("EVAL3_RAG_HW", "480,640").strip()
try:
    _h, _w = (int(x) for x in _hw_raw.split(","))
except Exception:
    _h, _w = 480, 640

# Default canonical task matches what EVAL3_TASK_AUG_CANONICAL_P=1.0
# produces during training — no celebrity name, forces reliance on camera2.
_CANONICAL_TASK = os.environ.get(
    "EVAL3_CANONICAL_TASK",
    "Place the coke on the person in the reference photo",
).strip()
_KEEP_TASK_NAME = os.environ.get("EVAL3_KEEP_TASK_NAME", "0").strip() in ("1", "true")

# Auto-fetch portraits from Wikipedia for anyone not in the local celeb_refdb/.
# Enabled by default: a missing portrait means black camera2 and guaranteed
# task failure.  Wikipedia is not a VLM — it just serves an image file.
# Disable with EVAL3_RAG_AUTO_FETCH=0 if you want a fully-offline strict run.
_AUTO_FETCH = os.environ.get("EVAL3_RAG_AUTO_FETCH", "1").strip() not in ("0", "false")


# ── resolve celebrity slug and load reference image ───────────────────────

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

_slug = task_to_celeb_slug(_task_raw)

if _slug:
    log.info("deploy_rag: task=%r  →  slug=%r", _task_raw, _slug)
else:
    log.warning("deploy_rag: cannot infer celebrity slug from task=%r", _task_raw)


def _load_refface(slug: str | None, hw: tuple[int, int]) -> "np.ndarray | None":
    """Load the canonical reference face as numpy uint8 (H,W,3).

    Searches all local dataset directories in priority order.  When no local
    image is found and EVAL3_RAG_AUTO_FETCH=1 (default), fetches a portrait
    from Wikipedia and caches it permanently — so any public figure with a
    Wikipedia page works out of the box at demo day.

    This fetch happens once at script startup, before the 20 s task timer
    begins.  All subsequent runs use the cached JPEG.

    To pre-download portraits offline in advance (recommended):
        python scripts/eval3_rag/download_refdb.py --slugs <name ...>

    Disable Wikipedia fallback with EVAL3_RAG_AUTO_FETCH=0.
    """
    if not slug:
        return None
    try:
        inj = ReferenceImageInjector(
            slug,
            image_hw=hw,
            random_sample=False,
            auto_fetch=_AUTO_FETCH,
            validate_face=False,  # no image recognition at inference — rule compliance
        )
        arr = inj.load_numpy()
        log.info("deploy_rag: reference face  slug=%r  shape=%s", slug, arr.shape)
        return arr
    except FileNotFoundError:
        log.error(
            "deploy_rag: no portrait found for slug=%r.\n"
            "  → Pre-download: python scripts/eval3_rag/download_refdb.py --slugs %s\n"
            "  → Or ensure internet access so Wikipedia auto-fetch can run.",
            slug, slug,
        )
        return None


_refface_img = _load_refface(_slug, (_h, _w))

# ── canonicalise the task string in argv ─────────────────────────────────
# Replace the celebrity-name task with the canonical form that the model
# was trained on (EVAL3_TASK_AUG_CANONICAL_P=1.0 → no name during training).
if _KEEP_TASK_NAME:
    log.info("deploy_rag: EVAL3_KEEP_TASK_NAME=1 — keeping original task: %r", _task_raw)
else:
    _replace_argv_task(_CANONICAL_TASK)
    log.info(
        "deploy_rag: task canonicalised\n"
        "  original : %r\n"
        "  → model sees: %r\n"
        "  (set EVAL3_KEEP_TASK_NAME=1 to skip canonicalisation)",
        _task_raw, _CANONICAL_TASK,
    )

# ── patch build_dataset_frame to inject front_refface ────────────────────
# build_dataset_frame lives in lerobot.utils.feature_utils (not .datasets).
# eval3_vla_deploy.py imports it as:
#   from lerobot.datasets.feature_utils import build_dataset_frame
# which fails on current lerobot builds.  We pre-inject the patched function
# into lerobot.datasets.feature_utils so that import resolves correctly,
# then import eval3_vla_deploy (which picks up our version), then also patch
# lerobot.policies.utils which is the call-site used inside the control loop.
import lerobot.utils.feature_utils as _futils       # noqa: E402
import lerobot.datasets.feature_utils as _ds_futils  # noqa: E402

_orig_build_dataset_frame = _futils.build_dataset_frame


def _rag_build_dataset_frame(features, obs, **kwargs):
    frame = _orig_build_dataset_frame(features, obs, **kwargs)
    if _refface_img is not None:
        frame["observation.images.front_refface"] = _refface_img
    return frame


# Patch canonical location + datasets alias before eval3_vla_deploy imports it
_futils.build_dataset_frame    = _rag_build_dataset_frame
_ds_futils.build_dataset_frame = _rag_build_dataset_frame  # satisfies eval3_vla_deploy import

# ── delegate to the standard deploy main() ───────────────────────────────
import eval3_vla_deploy  # noqa: E402  (now sees patched build_dataset_frame)

# Also patch the reference inside policies.utils (used by SmolVLA control loop)
try:
    import lerobot.policies.utils as _pol_utils  # noqa: E402
    _pol_utils.build_dataset_frame = _rag_build_dataset_frame
except Exception:
    pass

if __name__ == "__main__":
    eval3_vla_deploy.build_dataset_frame = _rag_build_dataset_frame  # type: ignore[attr-defined]
    eval3_vla_deploy.main()
