from __future__ import annotations

import torch
from diffusers import AutoencoderKL


class VAEEncoder:
    """Stable Diffusion VAE-based feature encoder for Glaze cloaking.

    Uses the VAE encoder's mean latent as the feature extractor Φ in the
    Glaze perturbation objective.  Spatial dims are flattened so the output
    is a 1-D feature vector per image.
    """

    def __init__(
        self,
        model_id: str = "stabilityai/stable-diffusion-2-1",
    ) -> None:
        self._model_id = model_id
        self._vae: AutoencoderKL = AutoencoderKL.from_pretrained(
            model_id, subfolder="vae"
        ).eval()

        for param in self._vae.parameters():
            param.requires_grad_(False)

        self._feature_dim: int = self._compute_feature_dim()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_feature_dim(self) -> int:
        """Run a dummy forward pass to determine the flattened feature size."""
        dummy = torch.zeros(1, 3, 512, 512, device=self.device, dtype=torch.float32)
        with torch.inference_mode():
            latent = self._vae.encode(dummy).latent_dist.mean  # [1, C, H/8, W/8]
        return int(latent.numel())  # C * (512/8) * (512/8)

    # ------------------------------------------------------------------
    # FeatureEncoder protocol
    # ------------------------------------------------------------------

    @property
    def device(self) -> torch.device:
        return next(self._vae.parameters()).device

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    @torch.inference_mode()
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images to flattened VAE latent means.

        Args:
            images: Float tensor [B, 3, H, W] normalised to [-1, 1].
                    SD VAE expects this exact normalisation.

        Returns:
            Float tensor [B, C*H/8*W/8] of flattened latent means.
        """
        if images.ndim == 3:
            images = images.unsqueeze(0)

        images = images.to(dtype=torch.float32, device=self.device)
        latent = self._vae.encode(images).latent_dist.mean  # [B, C, H/8, W/8]
        return latent.flatten(start_dim=1)  # [B, C*H/8*W/8]

    def to(self, device: torch.device) -> VAEEncoder:
        self._vae = self._vae.to(device)
        return self

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"VAEEncoder(model_id={self._model_id!r}, "
            f"feature_dim={self.feature_dim}, device={self.device})"
        )
