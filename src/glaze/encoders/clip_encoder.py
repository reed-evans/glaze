from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPVisionModelWithProjection


class CLIPEncoder:
    """CLIP-based feature encoder for Glaze cloaking.

    Uses the CLIP vision encoder (ViT-L/14 by default) as the feature
    extractor Φ in the Glaze perturbation objective.
    """

    def __init__(self, model_id: str = "openai/clip-vit-large-patch14") -> None:
        self._model_id = model_id
        self._model = CLIPVisionModelWithProjection.from_pretrained(model_id).eval()
        self._processor = CLIPProcessor.from_pretrained(model_id)

        for param in self._model.parameters():
            param.requires_grad_(False)

    # ------------------------------------------------------------------
    # FeatureEncoder protocol
    # ------------------------------------------------------------------

    @property
    def device(self) -> torch.device:
        return next(self._model.parameters()).device

    @property
    def feature_dim(self) -> int:
        return self._model.config.projection_dim  # 768 for ViT-L/14

    @torch.inference_mode()
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images to L2-normalised CLIP embeddings.

        Args:
            images: Float tensor [B, 3, H, W] normalised to [-1, 1].

        Returns:
            Float tensor [B, 768] of L2-normalised image embeddings.
        """
        # Add batch dim if a single image was passed
        if images.ndim == 3:
            images = images.unsqueeze(0)

        # Convert [-1, 1] → [0, 1] then to uint8 PIL-compatible range
        # CLIPProcessor expects pixel_values in [0, 1] float or uint8 numpy/PIL.
        # We bypass the processor's resize/normalise by supplying pixel_values
        # directly after scaling, but we must match the processor's normalisation.
        # Easier: convert to [0,255] uint8, pass as list of tensors, let processor handle it.
        pixel_values = (images.clamp(-1.0, 1.0) + 1.0) / 2.0  # [B, 3, H, W] in [0,1]

        # Use the processor's normalisation stats but skip resize (assume caller
        # handles resolution); pass do_resize=True so the processor handles 224×224.
        import numpy as np
        from PIL import Image

        pil_images = [
            Image.fromarray(
                (img.permute(1, 2, 0).cpu().float().numpy() * 255).astype(np.uint8)
            )
            for img in pixel_values
        ]

        inputs = self._processor(images=pil_images, return_tensors="pt")
        pixel_values_proc = inputs["pixel_values"].to(dtype=torch.float32, device=self.device)

        outputs = self._model(pixel_values=pixel_values_proc)
        embeddings = outputs.image_embeds  # [B, 768]
        return F.normalize(embeddings, p=2, dim=-1)

    def to(self, device: torch.device) -> CLIPEncoder:
        self._model = self._model.to(device)
        return self

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"CLIPEncoder(model_id={self._model_id!r}, "
            f"feature_dim={self.feature_dim}, device={self.device})"
        )
