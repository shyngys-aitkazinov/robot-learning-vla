"""Custom torchvision-v2 image transforms for visual augmentation.

lerobot's ``make_transform_from_config`` builds a transform by name via
``getattr(torchvision.transforms.v2, cfg.type)``. :func:`register` adds the
custom transforms below to that namespace so they are usable from the
augmentation config exactly like the built-in ones — without editing lerobot.

Transforms operate on float image tensors in ``[0, 1]`` with shape
``(..., C, H, W)`` — the layout lerobot hands to image transforms (decoded
video frames are float32 ``[0, 1]`` CHW).
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torchvision.transforms import v2


class SpatialLighting(v2.Transform):
    """Spatially-varying illumination: vignette + directional brightness gradient.

    More realistic than global ``ColorJitter`` brightness — it models uneven
    scene lighting (a lamp to one side, darker corners) by multiplying the image
    with a smooth per-pixel illumination map.

    Args:
        vignette: ``(min, max)`` range for corner-darkening strength. ``0`` =
            none, ``1`` = strong. Sampled uniformly per call.
        gradient: ``(min, max)`` range for directional brightness-ramp strength.
            ``0`` = none. Sampled uniformly per call; direction is random.
    """

    def __init__(
        self,
        vignette: tuple[float, float] | list[float] = (0.0, 0.5),
        gradient: tuple[float, float] | list[float] = (0.0, 0.4),
    ) -> None:
        super().__init__()
        self.vignette = (float(vignette[0]), float(vignette[1]))
        self.gradient = (float(gradient[0]), float(gradient[1]))

    def make_params(self, flat_inputs: list[Any]) -> dict[str, Any]:
        return {
            "vignette": torch.empty(1).uniform_(*self.vignette).item(),
            "gradient": torch.empty(1).uniform_(*self.gradient).item(),
            "angle": torch.empty(1).uniform_(0.0, 2.0 * math.pi).item(),
        }

    def transform(self, inpt: Any, params: dict[str, Any]) -> Any:
        if not isinstance(inpt, torch.Tensor) or inpt.ndim < 2:
            return inpt

        h, w = inpt.shape[-2], inpt.shape[-1]
        device = inpt.device
        ys = torch.linspace(-1.0, 1.0, h, device=device, dtype=torch.float32)
        xs = torch.linspace(-1.0, 1.0, w, device=device, dtype=torch.float32)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")

        illum = torch.ones((h, w), device=device, dtype=torch.float32)

        vignette = params["vignette"]
        if vignette > 0.0:
            radius = torch.sqrt(gx * gx + gy * gy) / math.sqrt(2.0)
            illum = illum * (1.0 - vignette * radius.clamp(0.0, 1.0))

        gradient = params["gradient"]
        if gradient > 0.0:
            ramp = gx * math.cos(params["angle"]) + gy * math.sin(params["angle"])
            illum = illum * (1.0 + gradient * 0.5 * ramp)

        illum = illum.clamp(0.05, 1.5)
        out = inpt.to(torch.float32) * illum
        return out.clamp_(0.0, 1.0).to(inpt.dtype)


# Custom transforms exposed to lerobot's `make_transform_from_config` by name.
CUSTOM_TRANSFORMS: dict[str, type[v2.Transform]] = {
    "SpatialLighting": SpatialLighting,
}


def register() -> None:
    """Make the custom transforms resolvable via ``getattr(v2, name)``.

    Idempotent. Call once in the main process before the dataset / its
    ``ImageTransforms`` are built (dataloader workers inherit the result).
    """
    for name, cls in CUSTOM_TRANSFORMS.items():
        if getattr(v2, name, None) is not cls:
            setattr(v2, name, cls)


if __name__ == "__main__":
    register()
    t = SpatialLighting(vignette=[0.4, 0.4], gradient=[0.3, 0.3])
    img = torch.rand(3, 480, 640)
    out = t(img)
    assert out.shape == img.shape and float(out.min()) >= 0.0 and float(out.max()) <= 1.0
    print("transforms_ext OK:", out.shape, out.dtype, "registered:", list(CUSTOM_TRANSFORMS))
