"""TinyMiniPitchNet fallback model definition."""
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

MODEL_FILENAME = "model.safetensors"


class TinyMiniPitchNet(nn.Module):
    """A lightweight convolutional fallback network for pitch estimation."""

    def __init__(self) -> None:
        super().__init__()
        channels = [1, 16, 32, 48]
        layers: list[nn.Module] = []
        for in_ch, out_ch in zip(channels[:-1], channels[1:]):
            layers.append(
                nn.Conv1d(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=15,
                    padding=7,
                    bias=True,
                )
            )
            layers.append(nn.ReLU())
        self.feature_extractor = nn.Sequential(*layers)
        self.head = nn.Linear(channels[-1], 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """Run the network on audio frames shaped as (batch, channels, samples)."""

        features = self.feature_extractor(x)
        pooled = features.mean(dim=-1)
        return self.head(pooled)


def get_model_dir() -> Path:
    """Return the directory that should contain the fallback weights."""

    return Path(__file__).parent


__all__ = ["TinyMiniPitchNet", "MODEL_FILENAME", "get_model_dir"]
