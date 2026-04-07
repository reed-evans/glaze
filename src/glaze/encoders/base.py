from __future__ import annotations

from typing import Protocol

import torch


class FeatureEncoder(Protocol):
    """Abstract interface for feature encoders used in Glaze cloaking."""

    @property
    def device(self) -> torch.device: ...

    @property
    def feature_dim(self) -> int: ...

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images to feature vectors.

        Args:
            images: Float tensor [B, 3, H, W] normalized to [-1, 1]

        Returns:
            Feature tensor [B, D] where D is feature_dim
        """
        ...

    def to(self, device: torch.device) -> FeatureEncoder: ...
