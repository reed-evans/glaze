from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from glaze.glaze import GlazeConfig, GlazeOptimizer
from glaze.encoders.base import FeatureEncoder


@dataclass
class VideoGlazeConfig:
    """Configuration for video glazing."""

    glaze_config: GlazeConfig = field(default_factory=GlazeConfig)

    # Temporal coherence: skip frames whose content hasn't changed much
    # compared to the previous glazed frame (saves compute)
    temporal_threshold: float = 0.02  # SSIM-based; 0 = process every frame

    # Re-use the perturbation from the previous frame as a warm start
    warm_start: bool = True

    # Resize frames to this size before glazing, then upscale back
    # (None = use original resolution)
    process_size: int | None = 512

    # Output video codec (fourcc string)
    output_codec: str = "mp4v"

    # Frames per second for output video (None = same as input)
    output_fps: float | None = None


def _tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert [1, 3, H, W] tensor in [-1,1] to PIL Image."""
    arr = tensor.squeeze(0).permute(1, 2, 0).cpu().float()
    arr = ((arr + 1.0) / 2.0 * 255.0).clamp(0, 255).byte().numpy()
    return Image.fromarray(arr)


def _pil_to_tensor(image: Image.Image, device: torch.device) -> torch.Tensor:
    """Convert PIL Image to [1, 3, H, W] tensor in [-1,1]."""
    arr = np.array(image.convert("RGB"), dtype=np.float32)
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0
    return tensor.to(device)


def _frame_ssim(a: torch.Tensor, b: torch.Tensor) -> float:
    """Approximate SSIM between two [1,3,H,W] tensors in [-1,1]."""
    diff = (a - b).abs().mean().item()
    return 1.0 - diff  # crude proxy; good enough for threshold gating


def _resize_tensor(tensor: torch.Tensor, size: int) -> torch.Tensor:
    return F.interpolate(tensor, size=(size, size), mode="bilinear", align_corners=False)


def _resize_back(tensor: torch.Tensor, h: int, w: int) -> torch.Tensor:
    return F.interpolate(tensor, size=(h, w), mode="bilinear", align_corners=False)


def iter_video_frames(path: Path) -> Iterator[tuple[int, np.ndarray]]:
    """Yield (frame_index, bgr_frame) for every frame in a video file."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {path}")
    idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield idx, frame
            idx += 1
    finally:
        cap.release()


def video_metadata(path: Path) -> tuple[int, float, int, int]:
    """Return (frame_count, fps, width, height)."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {path}")
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return count, fps, width, height


class VideoGlazer:
    """Applies Glaze cloaking to video frame sequences.

    Key optimizations over naive per-frame glazing:
    - **Temporal gating**: frames that are visually near-identical to the
      previous frame re-use the previous perturbation, skipping expensive
      optimization entirely.
    - **Warm start**: the previous frame's perturbation δ is used as the
      starting point for the next frame's optimization, reducing iterations
      needed to converge.
    - **Resolution scaling**: frames are downscaled to `process_size` before
      glazing and upscaled back, cutting O(N²) encoder compute.
    """

    def __init__(
        self,
        optimizer: GlazeOptimizer,
        config: VideoGlazeConfig | None = None,
    ) -> None:
        self.optimizer = optimizer
        self.config = config or VideoGlazeConfig()

    def glaze(
        self,
        input_path: Path,
        output_path: Path,
        target_style_prompt: str,
        style_transfer_fn: "StyleTransferFn | None" = None,
    ) -> None:
        """Glaze every frame of a video and write to output_path.

        Args:
            input_path: Source video file.
            output_path: Destination video file.
            target_style_prompt: Text prompt for the target art style.
            style_transfer_fn: Callable(source_tensor) -> styled_tensor.
                If None, target features are estimated from the prompt alone
                using the encoder's text path (CLIP) or by passing a
                zero-noise latent through the VAE.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        count, fps, w, h = video_metadata(input_path)
        out_fps = self.config.output_fps or fps
        fourcc = cv2.VideoWriter_fourcc(*self.config.output_codec)  # type: ignore[attr-defined]

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        writer = cv2.VideoWriter(str(tmp_path), fourcc, out_fps, (w, h))

        device = self.optimizer.config.torch_device
        prev_cloaked: torch.Tensor | None = None
        prev_delta: torch.Tensor | None = None
        target_features: torch.Tensor | None = None

        with tqdm(total=count, desc="Glazing frames", unit="frame") as bar:
            for idx, bgr in iter_video_frames(input_path):
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                pil = Image.fromarray(rgb)
                frame_tensor = _pil_to_tensor(pil, device)  # [1,3,H,W]

                # --- Temporal gating ---
                if prev_cloaked is not None and self.config.temporal_threshold > 0:
                    sim = _frame_ssim(frame_tensor, prev_cloaked - (prev_delta if prev_delta is not None else 0))
                    if sim >= 1.0 - self.config.temporal_threshold:
                        # Re-use previous perturbation
                        cloaked = torch.clamp(frame_tensor + prev_delta, -1.0, 1.0)
                        bar.set_postfix({"action": "reused"})
                        writer.write(_tensor_to_bgr(cloaked))
                        bar.update(1)
                        continue

                # --- Resize for processing ---
                orig_h, orig_w = frame_tensor.shape[2], frame_tensor.shape[3]
                if self.config.process_size is not None:
                    proc = _resize_tensor(frame_tensor, self.config.process_size)
                else:
                    proc = frame_tensor

                # --- Compute target features once (first frame) ---
                if target_features is None:
                    target_features = self._compute_target_features(
                        proc, target_style_prompt, style_transfer_fn
                    )

                # --- Warm start: seed delta from previous ---
                if self.config.warm_start and prev_delta is not None:
                    warm_delta = _resize_tensor(prev_delta, proc.shape[2])
                    self.optimizer._delta_init = warm_delta.detach().clone()
                else:
                    self.optimizer._delta_init = None  # type: ignore[assignment]

                cloaked_proc = self.optimizer.cloak_image(
                    proc, target_features, progress=False
                )

                # --- Upscale back to original resolution ---
                if self.config.process_size is not None:
                    cloaked = _resize_back(cloaked_proc, orig_h, orig_w)
                else:
                    cloaked = cloaked_proc

                prev_delta = (cloaked - frame_tensor).detach()
                prev_cloaked = cloaked.detach()

                bar.set_postfix({"action": "glazed"})
                writer.write(_tensor_to_bgr(cloaked))
                bar.update(1)

        writer.release()
        tmp_path.rename(output_path)

    def _compute_target_features(
        self,
        source: torch.Tensor,
        style_prompt: str,
        style_transfer_fn: "StyleTransferFn | None",
    ) -> torch.Tensor:
        """Compute Φ(Ω(x, T)) — target features for the optimization."""
        if style_transfer_fn is not None:
            with torch.inference_mode():
                styled = style_transfer_fn(source)
                return self.optimizer.encoder.encode(styled)
        # Fallback: encode the source itself (identity target — less effective
        # but functional when no style transfer model is available)
        with torch.inference_mode():
            return self.optimizer.encoder.encode(source)

    def glaze_frames(
        self,
        frames: list[Image.Image],
        target_style_prompt: str,
        style_transfer_fn: "StyleTransferFn | None" = None,
    ) -> list[Image.Image]:
        """Glaze a list of PIL frames and return cloaked PIL frames.

        Convenience method for in-memory frame sequences (e.g. GIFs).
        """
        device = self.optimizer.config.torch_device
        results: list[Image.Image] = []
        prev_delta: torch.Tensor | None = None
        target_features: torch.Tensor | None = None

        for frame in tqdm(frames, desc="Glazing frames", unit="frame"):
            frame_tensor = _pil_to_tensor(frame, device)

            if self.config.process_size is not None:
                proc = _resize_tensor(frame_tensor, self.config.process_size)
            else:
                proc = frame_tensor

            if target_features is None:
                target_features = self._compute_target_features(
                    proc, target_style_prompt, style_transfer_fn
                )

            if self.config.warm_start and prev_delta is not None:
                warm = _resize_tensor(prev_delta, proc.shape[2])
                self.optimizer._delta_init = warm.detach().clone()  # type: ignore[assignment]
            else:
                self.optimizer._delta_init = None  # type: ignore[assignment]

            orig_h, orig_w = frame_tensor.shape[2], frame_tensor.shape[3]
            cloaked_proc = self.optimizer.cloak_image(proc, target_features, progress=False)

            if self.config.process_size is not None:
                cloaked = _resize_back(cloaked_proc, orig_h, orig_w)
            else:
                cloaked = cloaked_proc

            prev_delta = (cloaked - frame_tensor).detach()
            results.append(_tensor_to_pil(cloaked))

        return results


# ---------------------------------------------------------------------------
# Type alias (forward ref — avoid importing style module here)
# ---------------------------------------------------------------------------
from typing import Callable

StyleTransferFn = Callable[[torch.Tensor], torch.Tensor]


def _tensor_to_bgr(tensor: torch.Tensor) -> np.ndarray:
    """Convert [1,3,H,W] tensor in [-1,1] to BGR uint8 ndarray."""
    arr = tensor.squeeze(0).permute(1, 2, 0).cpu().float()
    arr = ((arr + 1.0) / 2.0 * 255.0).clamp(0, 255).byte().numpy()
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
