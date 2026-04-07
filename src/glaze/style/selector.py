from __future__ import annotations

import random
from dataclasses import dataclass, field

import torch

from glaze.encoders.base import FeatureEncoder

# ---------------------------------------------------------------------------
# Curated style prompts spanning diverse aesthetic territories
# ---------------------------------------------------------------------------

DEFAULT_STYLE_PROMPTS: list[str] = [
    "in the style of abstract expressionism, bold brushstrokes",
    "in the style of pointillism, tiny colored dots",
    "in the style of cubism, geometric fragmented shapes",
    "in the style of art nouveau, flowing organic lines",
    "in the style of pop art, bold colors flat design",
    "in the style of watercolor painting, soft washes",
    "in the style of ukiyo-e Japanese woodblock print",
    "in the style of surrealism, dreamlike imagery",
    "in the style of baroque painting, dramatic chiaroscuro",
    "in the style of pixel art, 8-bit retro game style",
    "in the style of comic book art, bold outlines halftone dots",
    "in the style of pencil sketch, graphite cross-hatching",
    "in the style of fauvism, unnaturally bright colors",
    "in the style of art deco, geometric elegance 1920s",
    "in the style of impressionist painting, loose dabs of color",
    "in the style of hyperrealism, photorealistic oil painting",
    "in the style of gothic medieval illuminated manuscript",
    "in the style of vaporwave aesthetic, neon synthwave colors",
    "in the style of chinese ink wash painting, sumi-e",
    "in the style of constructivism, bold geometric poster art",
]


@dataclass(frozen=True)
class StyleSelectorConfig:
    use_distance_selection: bool = True
    percentile_low: float = 0.50
    percentile_high: float = 0.75
    style_prompts: tuple[str, ...] = field(
        default_factory=lambda: tuple(DEFAULT_STYLE_PROMPTS)
    )


class StyleSelector:
    """Selects target style T for Glaze cloaking.

    Per the Glaze paper, the target style should be 'sufficiently different'
    from the source image's style but not maximally distant — i.e., drawn
    from the 50th–75th percentile of pairwise distances in feature space.

    When *use_distance_selection* is False (or CLIP text encoding is
    unavailable), the selector falls back to uniform random selection.
    """

    def __init__(
        self,
        encoder: FeatureEncoder,
        style_prompts: list[str] | None = None,
        use_distance_selection: bool = True,
    ) -> None:
        self._encoder = encoder
        self._style_prompts: list[str] = style_prompts or list(DEFAULT_STYLE_PROMPTS)
        self._use_distance_selection = use_distance_selection
        # Lazily computed text embeddings, one per style prompt
        self._style_embeddings: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def select_target_style(
        self,
        source_features: torch.Tensor,
        percentile_low: float = 0.50,
        percentile_high: float = 0.75,
    ) -> str:
        """Select a target style prompt from the curated list.

        When distance selection is enabled, picks a style whose feature-space
        distance from *source_features* falls in [percentile_low, percentile_high].
        Falls back to random selection when distance selection is disabled or
        if no styles fall in the target band.

        Args:
            source_features: [D] feature vector of the source image.
            percentile_low: Lower bound of the acceptable distance percentile.
            percentile_high: Upper bound of the acceptable distance percentile.

        Returns:
            A style prompt string from the curated list.
        """
        if not self._use_distance_selection:
            return self._random_style()

        try:
            return self._distance_based_select(
                source_features, percentile_low, percentile_high
            )
        except Exception:
            # Graceful fallback: random selection if encoding fails
            return self._random_style()

    def to(self, device: str | torch.device) -> StyleSelector:
        """Move encoder to *device* and invalidate cached embeddings."""
        self._encoder = self._encoder.to(torch.device(str(device)))
        self._style_embeddings = None  # recompute on next call
        return self

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _distance_based_select(
        self,
        source_features: torch.Tensor,
        percentile_low: float,
        percentile_high: float,
    ) -> str:
        embeddings = self._get_style_embeddings()  # [N, D]

        # Compute cosine distances (1 - cosine_similarity) to source
        src = source_features.to(embeddings.device)
        if src.ndim == 1:
            src = src.unsqueeze(0)  # [1, D]

        import torch.nn.functional as F

        src_norm = F.normalize(src, p=2, dim=-1)               # [1, D]
        emb_norm = F.normalize(embeddings, p=2, dim=-1)        # [N, D]
        cosine_sim = (emb_norm @ src_norm.T).squeeze(-1)       # [N]
        distances = 1.0 - cosine_sim                           # [N]

        low_thresh = torch.quantile(distances, percentile_low).item()
        high_thresh = torch.quantile(distances, percentile_high).item()

        mask = (distances >= low_thresh) & (distances <= high_thresh)
        candidate_indices = mask.nonzero(as_tuple=False).squeeze(-1).tolist()

        if not candidate_indices:
            return self._random_style()

        chosen_idx = random.choice(candidate_indices)
        return self._style_prompts[chosen_idx]

    def _get_style_embeddings(self) -> torch.Tensor:
        """Lazily encode style prompts via CLIP text encoder."""
        if self._style_embeddings is not None:
            return self._style_embeddings

        self._style_embeddings = self._encode_style_prompts()
        return self._style_embeddings

    def _encode_style_prompts(self) -> torch.Tensor:
        """Encode all style prompts to feature vectors using CLIP text encoder."""
        from transformers import CLIPTextModel, CLIPTokenizer

        # Resolve device from encoder
        device = self._encoder.device

        # Use the same CLIP model that backs the image encoder when possible.
        # We target the standard CLIP ViT-L/14 text tower; this is a reasonable
        # default that pairs with CLIPEncoder's vision tower.
        model_id = "openai/clip-vit-large-patch14"

        tokenizer = CLIPTokenizer.from_pretrained(model_id)
        text_model = CLIPTextModel.from_pretrained(model_id).to(device).eval()

        for param in text_model.parameters():
            param.requires_grad_(False)

        inputs = tokenizer(
            self._style_prompts,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        import torch.nn.functional as F

        with torch.inference_mode():
            outputs = text_model(**inputs)
            # Use pooled output (EOS token) as the style embedding
            embeddings = outputs.pooler_output  # [N, D]
            embeddings = F.normalize(embeddings, p=2, dim=-1)

        return embeddings

    def _random_style(self) -> str:
        return random.choice(self._style_prompts)

    def __repr__(self) -> str:
        return (
            f"StyleSelector("
            f"num_styles={len(self._style_prompts)}, "
            f"use_distance_selection={self._use_distance_selection})"
        )
