from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image

# Prevent the tokenizers Rust library from forking parallelism threads,
# which causes semaphore leaks on macOS when the process exits.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


@dataclass(frozen=True)
class StyleTransferConfig:
    model_id: str = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    ip_adapter_model: str = "h94/IP-Adapter"
    ip_adapter_subfolder: str = "models"
    ip_adapter_weight_name: str = "ip-adapter_sd15.bin"
    device: str = "cuda"


class StyleTransfer:
    """Transfers the style of a reference image onto a source image.

    Computes Ω(x, T): source image x rendered in target style T, where T is
    a reference style *image* (not a text prompt).

    Uses Stable Diffusion img2img conditioned via IP-Adapter so that the
    target style is derived from the pixel content of T rather than a text
    description. This matches the Glaze paper's formulation: the target
    feature Φ(Ω(x,T)) should reflect T's visual style directly.

    The pipeline is loaded once and reused across calls for efficiency.
    """

    def __init__(
        self,
        model_id: str = "stable-diffusion-v1-5/stable-diffusion-v1-5",
        ip_adapter_model: str = "h94/IP-Adapter",
        ip_adapter_subfolder: str = "models",
        ip_adapter_weight_name: str = "ip-adapter_sd15.bin",
        device: str = "cuda",
        use_ip_adapter: bool = True,
    ) -> None:
        self._model_id = model_id
        self._ip_adapter_model = ip_adapter_model
        self._ip_adapter_subfolder = ip_adapter_subfolder
        self._ip_adapter_weight_name = ip_adapter_weight_name
        self._device = device
        self._use_ip_adapter = use_ip_adapter
        self._pipeline = None  # lazy init

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def transfer(
        self,
        source: torch.Tensor,
        style_image: torch.Tensor,
        strength: float = 0.5,
        num_inference_steps: int = 20,
        guidance_scale: float = 7.5,
        ip_adapter_scale: float = 0.6,
        style_prompt: str = "",
        max_size: int = 512,
    ) -> torch.Tensor:
        """Transfer the visual style of style_image onto source.

        Args:
            source: Float tensor [1, 3, H, W] in [-1, 1]. Content is preserved.
            style_image: Float tensor [1, 3, H, W] in [-1, 1]. Style reference
                image. Used directly when use_ip_adapter=True; ignored otherwise.
            strength: How strongly to apply the style (0.0–1.0).
            num_inference_steps: Denoising steps for the img2img pipeline.
            guidance_scale: Classifier-free guidance scale.
            ip_adapter_scale: IP-Adapter conditioning weight (ignored when
                use_ip_adapter=False).
            style_prompt: Text prompt used as fallback when use_ip_adapter=False.

        Returns:
            Float tensor [1, 3, H, W] in [-1, 1].
        """
        pipeline = self._get_pipeline()
        source = self._fit_to_size(source, max_size)
        source_pil = self._tensor_to_pil(source)

        if self._use_ip_adapter:
            pipeline.set_ip_adapter_scale(ip_adapter_scale)
            call_kwargs: dict = dict(
                prompt="",
                negative_prompt="",
                image=source_pil,
                ip_adapter_image=self._tensor_to_pil(style_image),
            )
        else:
            call_kwargs = dict(
                prompt=style_prompt,
                image=source_pil,
            )

        with torch.inference_mode():
            result = pipeline(
                **call_kwargs,
                strength=strength,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
            )

        output_pil: Image.Image = result.images[0]
        return self._pil_to_tensor(output_pil, device=source.device)

    def to(self, device: str | torch.device) -> StyleTransfer:
        self._device = str(device)
        if self._pipeline is not None:
            self._pipeline = self._pipeline.to(self._device)
        return self

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_pipeline(self):  # type: ignore[return]
        """Lazy-load the SD img2img pipeline with IP-Adapter."""
        if self._pipeline is None:
            from diffusers import StableDiffusionImg2ImgPipeline

            if "cuda" in self._device:
                dtype = torch.float16
            elif "mps" in self._device:
                dtype = torch.float16  # MPS supports float16 and it's faster
            else:
                dtype = torch.float32
            pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
                self._model_id,
                torch_dtype=dtype,
                safety_checker=None,
            ).to(self._device)

            if self._use_ip_adapter:
                pipeline.load_ip_adapter(
                    self._ip_adapter_model,
                    subfolder=self._ip_adapter_subfolder,
                    weight_name=self._ip_adapter_weight_name,
                )

            pipeline.set_progress_bar_config(disable=True)
            self._pipeline = pipeline

        return self._pipeline

    @staticmethod
    def _fit_to_size(tensor: torch.Tensor, max_size: int) -> torch.Tensor:
        """Resize tensor so the longer side equals max_size, rounded to nearest 64."""
        import torch.nn.functional as F
        _, _, h, w = tensor.shape
        scale = max_size / max(h, w)
        new_h = round(h * scale / 64) * 64
        new_w = round(w * scale / 64) * 64
        if (new_h, new_w) == (h, w):
            return tensor
        return F.interpolate(tensor, size=(new_h, new_w), mode="bilinear", align_corners=False)

    @staticmethod
    def _tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
        """Convert [1, 3, H, W] tensor in [-1, 1] to a PIL RGB image."""
        img = tensor.squeeze(0).float().cpu()
        img = (img.clamp(-1.0, 1.0) + 1.0) / 2.0
        img = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        return Image.fromarray(img)

    @staticmethod
    def _pil_to_tensor(image: Image.Image, device: torch.device) -> torch.Tensor:
        """Convert a PIL RGB image to a [1, 3, H, W] tensor in [-1, 1]."""
        arr = np.array(image.convert("RGB"), dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1)
        tensor = tensor * 2.0 - 1.0
        return tensor.unsqueeze(0).to(device)

    def __repr__(self) -> str:
        loaded = self._pipeline is not None
        return (
            f"StyleTransfer(model_id={self._model_id!r}, "
            f"device={self._device!r}, pipeline_loaded={loaded})"
        )
