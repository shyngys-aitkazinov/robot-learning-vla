#!/usr/bin/env python3
"""Render before/after augmentation previews for the Eval 3 training pipeline.

Per dataset (Swift, LeCun, Obama), pick one frame mid-episode and apply each of the
6 individual torchvision augmentations from the trainer config, plus a "full stack"
pass that samples up to 3 augmentations from the full set (matches what the trainer
does at each step). The output is a 2x4 grid PNG per dataset under
``outputs/eval3_deep_analysis_v2/augmentation_preview/``:

    ┌──────────────┬──────────────┬──────────────┬──────────────┐
    │ original     │ brightness   │ contrast     │ saturation   │
    ├──────────────┼──────────────┼──────────────┼──────────────┤
    │ hue          │ sharpness    │ affine       │ full stack   │
    └──────────────┴──────────────┴──────────────┴──────────────┘

Visual check: prints + faces remain recognisable, room context is meaningfully
perturbed (so the model can't trivially memorise it).

Also prints 30 augmented task strings per celebrity using the same RNG seed the
trainer uses, so we can see what the tokenizer will see.

Usage:
   python tools/eval3_visualize_augmentation.py
   python tools/eval3_visualize_augmentation.py --frame-idx 400
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS = _REPO_ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from eval3_lerobot_shim import apply as _shim_apply  # noqa: E402

_shim_apply()

from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from lerobot.datasets.transforms import (  # noqa: E402
    ImageTransformConfig,
    ImageTransformsConfig,
    ImageTransforms,
    make_transform_from_config,
)
from eval3_dataset_prep import make_task_augmenter  # noqa: E402


# Exact augmentation kwargs the trainer uses (must mirror run_eval3_smolvla_aug_train.sh).
SINGLE_TFS = {
    "brightness": ImageTransformConfig(weight=2.0, type="ColorJitter", kwargs={"brightness": (0.6, 1.4)}),
    "contrast":   ImageTransformConfig(weight=2.0, type="ColorJitter", kwargs={"contrast":   (0.6, 1.4)}),
    "saturation": ImageTransformConfig(weight=1.0, type="ColorJitter", kwargs={"saturation": (0.5, 1.5)}),
    "hue":        ImageTransformConfig(weight=1.0, type="ColorJitter", kwargs={"hue":        (-0.05, 0.05)}),
    "sharpness":  ImageTransformConfig(weight=1.0, type="SharpnessJitter", kwargs={"sharpness": (0.5, 1.5)}),
    "affine":     ImageTransformConfig(weight=1.0, type="RandomAffine", kwargs={"degrees": (-3.0, 3.0), "translate": (0.03, 0.03)}),
}


def _img_chw_float_to_pil(img_chw: torch.Tensor) -> Image.Image:
    arr = img_chw.detach().cpu().numpy()
    arr = (arr.transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def _load_font(size: int):
    for c in [
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]:
        try:
            return ImageFont.truetype(c, size)
        except Exception:
            pass
    return ImageFont.load_default()


def render_grid(img_chw: torch.Tensor, label: str, out_path: Path) -> None:
    """Render a 2x4 grid of single-aug variants + the full stack."""
    cells: list[tuple[str, Image.Image]] = [("original", _img_chw_float_to_pil(img_chw))]
    for name, cfg in SINGLE_TFS.items():
        t = make_transform_from_config(cfg)
        out = t(img_chw)
        cells.append((name, _img_chw_float_to_pil(out)))

    # "full stack" via RandomSubsetApply (n_subset=3) — same as trainer
    full_cfg = ImageTransformsConfig(enable=True, max_num_transforms=3, random_order=False, tfs=SINGLE_TFS)
    full_t = ImageTransforms(full_cfg)
    cells.append(("full stack (3 sampled)", _img_chw_float_to_pil(full_t(img_chw))))

    cw, ch = cells[0][1].size
    pad = 6
    label_h = 28
    grid_w = 4 * cw + 5 * pad
    grid_h = 2 * (ch + label_h) + 3 * pad
    canvas = Image.new("RGB", (grid_w, grid_h + 50), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    font_h = _load_font(18)
    font_c = _load_font(14)
    draw.text((pad, 6), label, fill=(0, 0, 0), font=font_h)
    for i, (name, img) in enumerate(cells):
        col = i % 4; row = i // 4
        x = pad + col * (cw + pad)
        y = 50 + pad + row * (ch + label_h + pad)
        canvas.paste(img, (x, y))
        draw.text((x + 4, y + ch + 4), name, fill=(40, 40, 40), font=font_c)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    print(f"  wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="outputs/eval3_deep_analysis_v2/augmentation_preview")
    ap.add_argument("--frame-frac", type=float, default=0.4, help="Frame index as a fraction of the chosen episode")
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--n-task-samples", type=int, default=30)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    repos = [
        ("RobotLearningVLA/taylor_swift_1", "swift", "Taylor Swift"),
        ("RobotLearningVLA/yann_lecun_1",   "lecun", "Yann LeCun"),
        ("RobotLearningVLA/barack_obama_1", "obama", "Barack Obama"),
    ]
    print("== Image augmentation previews ==")
    torch.manual_seed(0)
    for repo, slug, name in repos:
        ds = LeRobotDataset(repo, video_backend="pyav", episodes=[args.episode])
        idx = max(0, int((len(ds) - 1) * args.frame_frac))
        row = ds[idx]
        img = row["observation.images.front"]
        out_png = out_dir / f"{slug}.png"
        render_grid(img, f"{name}   ep{args.episode}   frame {idx}/{len(ds)}", out_png)

    # Task-string samples
    print("\n== Task-string augmentation samples (matches trainer seed) ==")
    task_samples: dict[str, list[str]] = {}
    for _, slug, name in repos:
        aug = make_task_augmenter()
        canonical = f"Place the coke on the {name}"
        samples = [aug(canonical) for _ in range(args.n_task_samples)]
        task_samples[name] = samples
        print(f"\n  {name} -> {args.n_task_samples} samples from canonical {canonical!r}:")
        from collections import Counter
        ctr = Counter(samples)
        for k, v in ctr.most_common():
            print(f"    {v:3d}  {k!r}")

    task_json = out_dir / "task_aug_samples.json"
    with open(task_json, "w") as f:
        json.dump(task_samples, f, indent=2)
    print(f"\n  wrote {task_json}")


if __name__ == "__main__":
    main()
