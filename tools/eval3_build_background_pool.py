#!/usr/bin/env python3
"""Generate a synthetic background pool for Eval3 background-replacement augmentation.

We could stream Places365 or COCO, but for our purpose (decorrelate background
from prompt) the backgrounds don't need to be photo-realistic — they just need
to be visually diverse so the model can't latch on to any one pattern. Synthetic
backgrounds also guarantee no people / coke cans / faces leak in, which is a
real risk with downloaded photo datasets.

Generates ~500 backgrounds across 5 families:
  1. Solid color (random RGB)
  2. Two-color linear gradients
  3. Tiled colored noise
  4. Horizontal-stripe "wall-like" patterns
  5. Random checker / shapes

Each image is 640 x 480 to match the training-frame resolution.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def gen_solid(rng: np.random.Generator, size=(480, 640)) -> np.ndarray:
    h, w = size
    color = rng.integers(20, 240, size=3, dtype=np.int32)
    return np.broadcast_to(color, (h, w, 3)).astype(np.uint8).copy()


def gen_gradient(rng: np.random.Generator, size=(480, 640)) -> np.ndarray:
    h, w = size
    c0 = rng.integers(20, 240, size=3, dtype=np.int32)
    c1 = rng.integers(20, 240, size=3, dtype=np.int32)
    axis = rng.choice(["h", "v", "d"])
    if axis == "h":
        t = np.linspace(0, 1, w)[None, :, None]
    elif axis == "v":
        t = np.linspace(0, 1, h)[:, None, None]
    else:
        yy, xx = np.meshgrid(np.linspace(0, 1, h), np.linspace(0, 1, w), indexing="ij")
        t = (yy + xx)[..., None] / 2
    arr = c0 * (1 - t) + c1 * t
    return arr.clip(0, 255).astype(np.uint8)


def gen_noise_tiled(rng: np.random.Generator, size=(480, 640)) -> np.ndarray:
    h, w = size
    tile_size = int(rng.integers(8, 48))
    base = rng.integers(40, 200, size=3, dtype=np.int32)
    th = (h + tile_size - 1) // tile_size
    tw = (w + tile_size - 1) // tile_size
    small = rng.integers(-40, 40, size=(th, tw, 3), dtype=np.int32) + base
    big = np.kron(small, np.ones((tile_size, tile_size, 1), dtype=np.int32))[:h, :w]
    return big.clip(0, 255).astype(np.uint8)


def gen_stripes(rng: np.random.Generator, size=(480, 640)) -> np.ndarray:
    h, w = size
    img = np.zeros((h, w, 3), dtype=np.int32)
    base = rng.integers(40, 200, size=3, dtype=np.int32)
    img[:] = base
    horizontal = bool(rng.integers(0, 2))
    bands = int(rng.integers(2, 12))
    for b in range(bands):
        band_color = rng.integers(20, 240, size=3, dtype=np.int32)
        if horizontal:
            y0 = (h * b) // bands
            y1 = (h * (b + 1)) // bands
            img[y0:y1] = band_color
        else:
            x0 = (w * b) // bands
            x1 = (w * (b + 1)) // bands
            img[:, x0:x1] = band_color
    return img.clip(0, 255).astype(np.uint8)


def gen_shapes(rng: np.random.Generator, size=(480, 640)) -> np.ndarray:
    h, w = size
    base_color = tuple(int(c) for c in rng.integers(40, 200, size=3))
    img = Image.new("RGB", (w, h), base_color)
    draw = ImageDraw.Draw(img)
    n_shapes = int(rng.integers(3, 14))
    for _ in range(n_shapes):
        x0, y0 = int(rng.integers(0, w)), int(rng.integers(0, h))
        x1, y1 = x0 + int(rng.integers(20, 200)), y0 + int(rng.integers(20, 200))
        x1, y1 = min(w, x1), min(h, y1)
        color = tuple(int(c) for c in rng.integers(20, 240, size=3))
        if rng.choice(["rect", "ell"]) == "rect":
            draw.rectangle([x0, y0, x1, y1], fill=color)
        else:
            draw.ellipse([x0, y0, x1, y1], fill=color)
    return np.array(img)


GENERATORS = [
    ("solid", gen_solid),
    ("gradient", gen_gradient),
    ("noise", gen_noise_tiled),
    ("stripes", gen_stripes),
    ("shapes", gen_shapes),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="outputs/eval3_backgrounds")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    counts = {name: 0 for name, _ in GENERATORS}
    for i in range(args.n):
        name, gen = GENERATORS[i % len(GENERATORS)]
        arr = gen(rng)
        Image.fromarray(arr).save(out_dir / f"bg_{i:04d}_{name}.png", optimize=False)
        counts[name] += 1
        if (i + 1) % 50 == 0:
            print(f"  generated {i+1}/{args.n}")

    print("\nDone. Per-family counts:")
    for name, n in counts.items():
        print(f"  {name:>10}: {n}")
    print(f"  total dir: {out_dir} ({sum(counts.values())} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
