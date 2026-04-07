from __future__ import annotations

from dataclasses import dataclass

import lpips
import torch
from tqdm import tqdm

from glaze.encoders.base import FeatureEncoder


@dataclass
class GlazeConfig:
    """Configuration for the Glaze optimization loop."""

    perturbation_budget: float = 0.05
    iterations: int = 500
    learning_rate: float = 1e-3
    alpha_init: float = 1.0
    alpha_multiplier: float = 1.1
    alpha_update_interval: int = 50
    device: str = "cuda"
    lpips_net: str = "alex"  # "alex" (default, fastest), "vgg", or "squeeze"

    @property
    def torch_device(self) -> torch.device:
        return torch.device(self.device)


class GlazeOptimizer:
    """Computes style-cloaking perturbations per the Glaze paper.

    Solves:
        min_δ ||Φ(Ω(x,T)) - Φ(x + δ)||₂²  +  α·max(LPIPS(x, x+δ) - p, 0)

    where Φ is a frozen feature encoder, Ω(x,T) is the source image rendered
    in target style T (precomputed), and p is the perturbation budget.
    """

    def __init__(self, encoder: FeatureEncoder, config: GlazeConfig | None = None) -> None:
        self.encoder = encoder
        self.config = config or GlazeConfig()
        self._lpips_fn: lpips.LPIPS | None = None
        # Optional warm-start delta; set by VideoGlazer before calling cloak_image
        self._delta_init: torch.Tensor | None = None

    def _get_lpips(self) -> lpips.LPIPS:
        """Lazy-init and cache the LPIPS model."""
        if self._lpips_fn is None:
            self._lpips_fn = lpips.LPIPS(net=self.config.lpips_net).to(self.config.torch_device)
            for param in self._lpips_fn.parameters():
                param.requires_grad_(False)
        return self._lpips_fn

    def compute_losses(
        self,
        delta: torch.Tensor,
        source: torch.Tensor,
        target_features: torch.Tensor,
        alpha: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute total, feature, and constraint losses.

        Args:
            delta: Perturbation tensor [1, 3, H, W], requires_grad=True.
            source: Original image [1, 3, H, W] in [-1, 1].
            target_features: Precomputed Φ(Ω(x,T)) [1, D].
            alpha: Current penalty weight.

        Returns:
            Tuple of (total_loss, feature_loss, constraint_loss).
        """
        cloaked = torch.clamp(source + delta, -1.0, 1.0)

        cloaked_features = self.encoder.encode(cloaked)
        feature_loss = torch.sum((target_features - cloaked_features) ** 2)

        lpips_fn = self._get_lpips()
        lpips_dist = lpips_fn(source, cloaked)
        constraint_loss = torch.clamp(lpips_dist - self.config.perturbation_budget, min=0.0)

        total_loss = feature_loss + alpha * constraint_loss
        return total_loss, feature_loss, constraint_loss

    def cloak_image(
        self,
        source: torch.Tensor,
        target_features: torch.Tensor,
        progress: bool = True,
    ) -> torch.Tensor:
        """Compute cloaked version of source image.

        Args:
            source: Original image [1, 3, H, W] in [-1, 1].
            target_features: Precomputed Φ(Ω(x,T)) [1, D].
            progress: Whether to show a tqdm progress bar.

        Returns:
            Cloaked image [1, 3, H, W] in [-1, 1].
        """
        cfg = self.config
        device = cfg.torch_device

        source = source.to(device)
        target_features = target_features.to(device).detach()

        if self._delta_init is not None:
            init = self._delta_init.to(device).detach().clone()
            # Resize warm-start delta if resolution differs
            if init.shape != source.shape:
                import torch.nn.functional as F
                init = F.interpolate(init, size=source.shape[2:], mode="bilinear", align_corners=False)
            delta = init.requires_grad_(True)
            self._delta_init = None  # consume once
        else:
            delta = torch.zeros_like(source, requires_grad=True, device=device)
        optimizer = torch.optim.Adam([delta], lr=cfg.learning_rate)
        alpha = cfg.alpha_init

        bar = tqdm(
            range(cfg.iterations),
            desc="Glazing",
            disable=not progress,
            dynamic_ncols=True,
        )

        for step in bar:
            optimizer.zero_grad()

            total_loss, feature_loss, constraint_loss = self.compute_losses(
                delta, source, target_features, alpha
            )

            total_loss.backward()
            optimizer.step()

            # Clamp cloaked image to valid range by projecting delta
            with torch.no_grad():
                delta.clamp_(
                    -1.0 - source,
                    1.0 - source,
                )

            # Adaptive alpha: tighten the budget constraint when violated
            if (step + 1) % cfg.alpha_update_interval == 0:
                with torch.no_grad():
                    lpips_fn = self._get_lpips()
                    cloaked = torch.clamp(source + delta, -1.0, 1.0)
                    lpips_dist = lpips_fn(source, cloaked).item()
                if lpips_dist > cfg.perturbation_budget:
                    alpha *= cfg.alpha_multiplier

            bar.set_postfix(
                {
                    "feat": f"{feature_loss.item():.4f}",
                    "lpips": f"{constraint_loss.item():.4f}",
                    "α": f"{alpha:.3f}",
                }
            )

        with torch.no_grad():
            cloaked_image = torch.clamp(source + delta, -1.0, 1.0)

        return cloaked_image
