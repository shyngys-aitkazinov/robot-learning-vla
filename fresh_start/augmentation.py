"""Build lerobot's image-transforms config from :class:`AugmentationConfig`.

lerobot applies augmentation through ``ImageTransformsConfig`` — a dict of
named ``ImageTransformConfig`` entries (``type`` + ``kwargs`` + sampling
``weight``). Per frame it samples up to ``max_num_transforms`` of them. Each
``type`` resolves to a ``torchvision.transforms.v2`` class (or a custom one
registered by :mod:`transforms_ext`), so noise / blur / lighting are all
plain config — no monkey-patching.

This module converts our friendly :class:`AugmentationConfig` into that
lerobot structure.
"""

from __future__ import annotations

from lerobot.datasets.transforms import (
    ImageTransformConfig,
    ImageTransforms,
    ImageTransformsConfig,
)

from config import AugmentationConfig


def build_image_transforms_config(aug: AugmentationConfig) -> ImageTransformsConfig:
    """Translate :class:`AugmentationConfig` into a lerobot ``ImageTransformsConfig``.

    Only the enabled groups are emitted. lerobot's defaults (``affine``,
    ``sharpness``) are intentionally not included — ``affine`` can shift the
    can/target out of frame for a fixed single camera, and blur already covers
    softness.
    """
    tfs: dict[str, ImageTransformConfig] = {}

    if aug.lighting.enable:
        light = aug.lighting
        # One ColorJitter per attribute so the sampler can pick any subset.
        for name, key, value in (
            ("brightness", "brightness", light.brightness),
            ("contrast", "contrast", light.contrast),
            ("saturation", "saturation", light.saturation),
            ("hue", "hue", light.hue),
        ):
            tfs[name] = ImageTransformConfig(
                weight=light.weight,
                type="ColorJitter",
                kwargs={key: list(value)},
            )

    if aug.noise.enable:
        tfs["gaussian_noise"] = ImageTransformConfig(
            weight=aug.noise.weight,
            type="GaussianNoise",
            kwargs={"mean": aug.noise.mean, "sigma": aug.noise.sigma, "clip": True},
        )

    if aug.blur.enable:
        tfs["gaussian_blur"] = ImageTransformConfig(
            weight=aug.blur.weight,
            type="GaussianBlur",
            kwargs={"kernel_size": aug.blur.kernel_size, "sigma": list(aug.blur.sigma)},
        )

    if aug.spatial_lighting.enable:
        spatial = aug.spatial_lighting
        # Custom transform — transforms_ext.register() must run first.
        tfs["spatial_lighting"] = ImageTransformConfig(
            weight=spatial.weight,
            type="SpatialLighting",
            kwargs={"vignette": list(spatial.vignette), "gradient": list(spatial.gradient)},
        )

    return ImageTransformsConfig(
        enable=aug.enable and len(tfs) > 0,
        max_num_transforms=aug.max_num_transforms,
        random_order=aug.random_order,
        tfs=tfs,
    )


def build_image_transforms(aug: AugmentationConfig) -> ImageTransforms:
    """Build a ready-to-apply lerobot ``ImageTransforms`` callable (for previews)."""
    return ImageTransforms(build_image_transforms_config(aug))


def summarize(aug: AugmentationConfig) -> str:
    """One-line human summary of the active augmentation stack."""
    cfg = build_image_transforms_config(aug)
    if not cfg.enable:
        return "augmentation: DISABLED"
    active = ", ".join(f"{n}({c.type})" for n, c in cfg.tfs.items())
    return (
        f"augmentation: {len(cfg.tfs)} transforms "
        f"[sample <= {cfg.max_num_transforms}, random_order={cfg.random_order}] -> {active}"
    )
