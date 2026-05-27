"""RAG dataset wrapper — layers celebrity reference-face injection on top of
the existing eval3_concat_patch pipeline without modifying it.

Call order in train_rag.py::

    eval3_lerobot_shim.apply()
    eval3_concat_patch.apply_concat_patch()   # regular multi-dataset build
    apply_rag_concat_patch()                  # wraps result with Eval3RagWrapper

``Eval3RagWrapper`` reads each row's ``task`` string, infers the celebrity slug,
and injects a random reference face as ``observation.images.front_refface``.
The training rename_map routes ``front_refface`` → ``camera2`` so SmolVLA's
frozen vision encoder processes it alongside the live scene (``camera1``).

The model trains to answer:
    "Given reference face (camera2) and live scene (camera1), where is that
    face on the table?  Move there."

── Why camera2 augmentation matters ────────────────────────────────────────

At training time camera2 is a clean portrait from the local DB.
At inference time the same celebrity appears in camera1 as a *printed photo
on a table* — different lighting, reflection, scale, and print artefacts.

Applying printed-photo-like augmentation specifically to camera2 (heavier
than the camera1 scene augmentation) teaches the model:

  "These two different-looking images of Elon Musk should produce the same
   trajectory, despite colour shift / blur / monochrome print."

Without this, the model may use appearance shortcuts that break at test time.

Note: the trajectory-identity bug (all same-slot celebs share identical
actions) means BC can still ignore camera2 entirely and predict correctly.
The real fix for that is data-side (multi-celebrity scenes, or per-celebrity
endpoint offsets in the synth generator).  The augmentation here ensures that
*once the model learns to use camera2*, it does so robustly.

Env vars
--------
EVAL3_RAG_REFDB      path to celebrity reference-image directory (required)
EVAL3_RAG_HW         "H,W" load size, default "480,640"
EVAL3_CAM2_AUG       1/0 enable camera2-specific printed-photo augmentation
                     (default 1 at training, 0 at inference)
EVAL3_CAM2_AUG_P     probability of applying each individual transform (0–1,
                     default 0.7); higher = more aggressive appearance changes
"""
from __future__ import annotations

import logging
import os
import random
import sys
from pathlib import Path
from typing import Any

import torch

# Ensure scripts/ is importable so eval3_rag.reference_injector resolves.
_SCRIPTS = Path(__file__).resolve().parent.parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from eval3_rag.reference_injector import ReferenceImageInjector, task_to_celeb_slug


# ---------------------------------------------------------------------------
# Camera2 printed-photo augmentation
# ---------------------------------------------------------------------------

def _build_cam2_augmentation(p: float):
    """Return a callable that augments a (3,H,W) float tensor in [0,1].

    Simulates the appearance of a celebrity portrait *after* being printed
    and photographed by a robot camera:
      • colour temperature / white-balance shift  (colourjitter)
      • monochrome / greyscale print              (randomgrayscale)
      • motion blur / out-of-focus               (gaussianblur)
      • tilt / slight perspective of printed card (randomaffine)
      • photocopy artefacts / contrast changes    (colourjitter contrast)

    All transforms are applied stochastically with probability *p* so the
    model sees the *same* reference face under many different appearances.
    """
    try:
        import torchvision.transforms.functional as TF
        import torchvision.transforms as T
    except ImportError:
        logging.warning("torchvision not available — camera2 augmentation disabled")
        return None

    def augment(img: torch.Tensor) -> torch.Tensor:
        """img: (3, H, W) float [0,1] — returns augmented tensor same shape."""
        rng = random.random

        # NOTE: horizontal flip intentionally omitted — the physical print has a fixed
        # orientation, so a mirrored reference is actively harmful for identity matching.

        # 1. Colour temperature shift (warm / cool print, different lighting)
        if rng() < p:
            br = 1.0 + random.uniform(-0.35, 0.35)
            co = 1.0 + random.uniform(-0.35, 0.35)
            sa = 1.0 + random.uniform(-0.40, 0.40)
            hu = random.uniform(-0.06, 0.06)
            img = TF.adjust_brightness(img, br)
            img = TF.adjust_contrast(img, co)
            img = TF.adjust_saturation(img, sa)
            img = TF.adjust_hue(img, hu)

        # 3. Greyscale / monochrome print (probability matches typical B&W prints)
        if rng() < p * 0.5:
            grey = TF.rgb_to_grayscale(img, num_output_channels=3)
            # Blend so it may be partially desaturated, not always full grey
            alpha = random.uniform(0.5, 1.0)
            img = alpha * grey + (1 - alpha) * img

        # 4. Gaussian blur (camera focus distance, paper texture)
        if rng() < p:
            ks = random.choice([3, 5, 7])
            sigma = random.uniform(0.5, 2.0)
            img = TF.gaussian_blur(img, kernel_size=ks, sigma=sigma)

        # 5. Slight affine (card tilt, perspective when photographing)
        if rng() < p * 0.6:
            angle = random.uniform(-8, 8)
            dx = random.uniform(-0.05, 0.05)
            dy = random.uniform(-0.05, 0.05)
            sc = random.uniform(0.90, 1.10)
            img = TF.affine(img, angle=angle, translate=[dx * img.shape[-1], dy * img.shape[-2]],
                            scale=sc, shear=0, interpolation=T.InterpolationMode.BILINEAR)

        # Clamp to [0,1] after float arithmetic
        return img.clamp(0.0, 1.0)

    return augment


# Module-level singleton (built lazily on first call)
_CAM2_AUG_FUNC = None
_CAM2_AUG_BUILT = False


def _get_cam2_aug(p: float):
    global _CAM2_AUG_FUNC, _CAM2_AUG_BUILT
    if not _CAM2_AUG_BUILT:
        _CAM2_AUG_FUNC = _build_cam2_augmentation(p)
        _CAM2_AUG_BUILT = True
    return _CAM2_AUG_FUNC


# ---------------------------------------------------------------------------
# Eval3RagWrapper
# ---------------------------------------------------------------------------

class Eval3RagWrapper:
    """Proxy that injects ``observation.images.front_refface`` into every row.

    Sits on top of the ``ConcatLeRobotDataset`` produced by the existing
    concat-patch.  Reads ``row["task"]`` to determine which person's reference
    image to load, injects it as ``front_refface``, and **also overwrites the
    task string** with a canonical form that contains no person name.

    The task overwrite is the key step that makes training match inference:

      Training (before this wrapper):  ``"Place the coke on Taylor Swift"``
      After this wrapper:              ``"Place the coke on the person in the reference photo"``
      Inference (deploy_rag.py):       ``"Place the coke on the person in the reference photo"``

    Without this overwrite the model would see celebrity names at training time
    but a generic task at inference — an out-of-distribution mismatch.

    Camera2 augmentation is applied when ``cam2_aug=True`` (training mode).
    """

    def __init__(
        self,
        dataset,
        rag_db: str,
        image_hw: tuple[int, int] = (480, 640),
        cam2_aug: bool = True,
        cam2_aug_p: float = 0.7,
        canonical_task: str = "Place the coke on the person in the reference photo",
    ):
        self._dataset        = dataset
        self._rag_db         = str(rag_db)
        self._image_hw       = tuple(image_hw)
        self._cam2_aug       = cam2_aug
        self._cam2_aug_p     = float(cam2_aug_p)
        self._canonical_task = str(canonical_task)
        # slug -> injector (or None when slug not in DB)
        self._injectors: dict[str, ReferenceImageInjector | None] = {}
        try:
            self._available_slugs: set[str] = {
                d for d in os.listdir(rag_db)
                if os.path.isdir(os.path.join(rag_db, d))
            }
        except FileNotFoundError:
            logging.warning(
                "Eval3RagWrapper: EVAL3_RAG_REFDB=%r not found — "
                "front_refface absent from all samples.  "
                "Run: python scripts/eval3_rag/build_celeb_refdb.py",
                rag_db,
            )
            self._available_slugs = set()
        self._warned_missing: set[str] = set()

    # ── reference-image helpers ──────────────────────────────────────────────

    def _get_injector(self, slug: str) -> ReferenceImageInjector | None:
        if slug in self._injectors:
            return self._injectors[slug]
        if slug not in self._available_slugs:
            if slug not in self._warned_missing:
                logging.warning(
                    "Eval3RagWrapper: slug=%r not in DB (%s) — "
                    "front_refface skipped for this celebrity.",
                    slug, self._rag_db,
                )
                self._warned_missing.add(slug)
            self._injectors[slug] = None
            return None
        inj = ReferenceImageInjector(
            slug, self._rag_db, self._image_hw,
            seed=hash(slug) & 0xFFFF,
            random_sample=True,   # different image each step at training time
        )
        self._injectors[slug] = inj
        return inj

    # ── dataset protocol ─────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, idx) -> Any:
        row  = self._dataset[idx]
        task = row.get("task", "") or ""
        slug = task_to_celeb_slug(str(task))

        if slug is None:
            # OOD fallback: try to infer slug from repo_id embedded in the row
            repo = row.get("dataset_index", "")
            if repo:
                from eval3_rag.reference_injector import _repo_id_to_celeb_slug
                slug = _repo_id_to_celeb_slug(str(repo))

        if slug is not None:
            inj = self._get_injector(slug)
            if inj is not None:
                ref_tensor = inj()  # (3, H, W) float [0,1], randomly sampled image
                if self._cam2_aug:
                    aug = _get_cam2_aug(self._cam2_aug_p)
                    if aug is not None:
                        ref_tensor = aug(ref_tensor)
                row = dict(row)
                row["observation.images.front_refface"] = ref_tensor
                # Overwrite the task: strip the person's name so the model
                # never sees celebrity names and must rely on camera2 instead.
                # This MUST match the canonical task used at inference in deploy_rag.py.
                row["task"] = self._canonical_task

        return row

    # ── trainer-required attribute proxies ───────────────────────────────────

    @property
    def meta(self):
        return self._dataset.meta

    @property
    def num_frames(self) -> int:
        return self._dataset.num_frames

    @property
    def num_episodes(self) -> int:
        return self._dataset.num_episodes

    @property
    def episodes(self):
        return self._dataset.episodes

    @property
    def repo_id(self) -> str:
        return self._dataset.repo_id

    @property
    def features(self):
        return self._dataset.features

    def __getattr__(self, name: str):
        if "_dataset" in self.__dict__:
            return getattr(self._dataset, name)
        raise AttributeError(name)


# ---------------------------------------------------------------------------
# Patch installation
# ---------------------------------------------------------------------------

def apply_rag_concat_patch() -> None:
    """Wrap the already-patched make_dataset output with Eval3RagWrapper.

    Must be called AFTER ``eval3_concat_patch.apply_concat_patch()``.
    No-op when ``EVAL3_RAG_REFDB`` is unset or empty.

    Env vars read here:
      EVAL3_RAG_REFDB       — path to celeb_refdb
      EVAL3_RAG_HW          — "H,W" (default 480,640)
      EVAL3_CANONICAL_TASK  — task string the model sees during training and inference
                              (default: "Place the coke on the person in the reference photo")
      EVAL3_CAM2_AUG        — 0/1 enable camera2 printed-photo augmentation (default 1)
      EVAL3_CAM2_AUG_P      — per-transform probability (default 0.7)
    """
    rag_db = os.environ.get("EVAL3_RAG_REFDB", "").strip()
    if not rag_db:
        logging.info("apply_rag_concat_patch: EVAL3_RAG_REFDB not set — RAG layer skipped.")
        return

    hw_raw = os.environ.get("EVAL3_RAG_HW", "480,640").strip()
    try:
        h_s, w_s = hw_raw.split(",")
        image_hw: tuple[int, int] = (int(h_s), int(w_s))
    except Exception:
        logging.warning(
            "apply_rag_concat_patch: cannot parse EVAL3_RAG_HW=%r — using (480,640)", hw_raw
        )
        image_hw = (480, 640)

    canonical_task = os.environ.get(
        "EVAL3_CANONICAL_TASK",
        "Place the coke on the person in the reference photo",
    ).strip()
    cam2_aug   = os.environ.get("EVAL3_CAM2_AUG",   "1").strip() not in ("0", "false", "")
    cam2_aug_p = float(os.environ.get("EVAL3_CAM2_AUG_P", "0.7").strip())

    from lerobot.datasets import factory as _factory

    _prev = _factory.make_dataset

    def _rag_make_dataset(cfg):
        base    = _prev(cfg)
        wrapped = Eval3RagWrapper(
            base,
            rag_db=rag_db,
            image_hw=image_hw,
            cam2_aug=cam2_aug,
            cam2_aug_p=cam2_aug_p,
            canonical_task=canonical_task,
        )
        logging.info(
            "apply_rag_concat_patch: wrapped %d frames "
            "(db=%s hw=%s cam2_aug=%s p=%.2f canonical_task=%r available_slugs=%s)",
            wrapped.num_frames, rag_db, image_hw, cam2_aug, cam2_aug_p,
            canonical_task, sorted(wrapped._available_slugs),
        )
        return wrapped

    _factory.make_dataset = _rag_make_dataset
    try:
        import lerobot.scripts.lerobot_train as _train_mod
        if hasattr(_train_mod, "make_dataset"):
            _train_mod.make_dataset = _rag_make_dataset
    except Exception:
        pass

    logging.info(
        "apply_rag_concat_patch: installed "
        "(EVAL3_RAG_REFDB=%s hw=%s cam2_aug=%s p=%.2f)",
        rag_db, image_hw, cam2_aug, cam2_aug_p,
    )
