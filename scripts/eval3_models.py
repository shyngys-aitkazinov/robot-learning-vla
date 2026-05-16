"""Tiny CNN + proprio → action BC head for Eval 3 pipeline sanity (not a course-compliant VLA alone)."""

from __future__ import annotations

import torch
from torch import nn


class Eval3TinyBC(nn.Module):
    """Vision + proprio behavioral cloning head."""

    def __init__(
        self,
        action_dim: int,
        state_dim: int,
        image_size: int = 256,
        channels: int = 3,
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.image_size = image_size
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=8, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=6, stride=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.Flatten(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, channels, image_size, image_size)
            flat_dim = self.encoder(dummy).shape[1]
        self.head = nn.Sequential(
            nn.Linear(flat_dim + state_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, action_dim),
        )

    def forward(self, images: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        z = self.encoder(images)
        x = torch.cat([z, state], dim=-1)
        return self.head(x)


def find_first_image_key(sample: dict) -> str | None:
    keys = [k for k in sample if k.startswith("observation.images.")]
    return sorted(keys)[0] if keys else None
