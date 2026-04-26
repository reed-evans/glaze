"""Video-level similarity metrics for style-mimicry evaluation.

Given two videos (typically a reference artist clip and a generator's output),
compute three per-frame metrics and average them:

    clip_similarity  — cosine similarity of CLIP image embeddings. Higher
                       means the two videos look more alike to CLIP.
    lpips            — AlexNet-backed learned perceptual distance.
                       Lower means closer. Matches what Glaze optimizes.
    style_loss       — Gatys-style Gram-matrix distance in VGG16 feature
                       space. Lower means more stylistically similar.

Both videos are uniformly sub-sampled to the same number of frames and resized
to a common square resolution before comparison, so frame-rate and resolution
mismatches between reference and generation don't dominate the signal.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import lpips
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.models import VGG16_Weights, vgg16
from transformers import CLIPModel, CLIPProcessor

from runner.video_io import read_frames, resize_square, subsample

CLIP_MODEL_ID = "openai/clip-vit-large-patch14"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class VideoMetrics:
    clip_similarity: float
    lpips: float
    style_loss: float
    num_frames_compared: int


class _GramStyleLoss:
    """Gatys-style loss: L2 distance between Gram matrices at selected VGG16 layers."""

    # conv1_2, conv2_2, conv3_3, conv4_3 — classic Gatys layer selection.
    STYLE_LAYER_INDICES = (3, 8, 15, 22)

    def __init__(self, device: str) -> None:
        self.device = device
        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features.to(device).eval()
        for p in vgg.parameters():
            p.requires_grad = False
        self.vgg = vgg
        self.mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)

    @staticmethod
    def _gram(feat: torch.Tensor) -> torch.Tensor:
        b, c, h, w = feat.shape
        flat = feat.view(b, c, h * w)
        return flat @ flat.transpose(1, 2) / (c * h * w)

    def __call__(self, img_a: torch.Tensor, img_b: torch.Tensor) -> float:
        a = (img_a - self.mean) / self.std
        b = (img_b - self.mean) / self.std
        total = torch.zeros((), device=self.device)
        for i, layer in enumerate(self.vgg):
            a = layer(a)
            b = layer(b)
            if i in self.STYLE_LAYER_INDICES:
                total = total + F.mse_loss(self._gram(a), self._gram(b))
            if i >= self.STYLE_LAYER_INDICES[-1]:
                break
        return float(total.detach().cpu())


def _to_tensor(rgb: np.ndarray, device: str) -> torch.Tensor:
    t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    return t.unsqueeze(0).to(device)


class Evaluator:
    """Owns the heavy models so comparing many video pairs amortizes setup cost."""

    def __init__(self, device: str = "cuda", frame_size: int = 224, num_frames: int = 16) -> None:
        if num_frames <= 0:
            raise ValueError("num_frames must be positive")
        if frame_size <= 0:
            raise ValueError("frame_size must be positive")
        self.device = device
        self.frame_size = frame_size
        self.num_frames = num_frames
        self.clip_model = CLIPModel.from_pretrained(CLIP_MODEL_ID).to(device).eval()
        self.clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
        self.lpips_fn = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
        self.style_fn = _GramStyleLoss(device=device)

    @torch.inference_mode()
    def _clip_embed(self, frame: np.ndarray) -> torch.Tensor:
        img = Image.fromarray(frame)
        inputs = self.clip_processor(images=img, return_tensors="pt").to(self.device)
        feat = self.clip_model.get_image_features(**inputs)
        return F.normalize(feat, dim=-1)

    @torch.inference_mode()
    def compare(self, video_a: Path, video_b: Path) -> VideoMetrics:
        frames_a = subsample(read_frames(video_a), self.num_frames)
        frames_b = subsample(read_frames(video_b), self.num_frames)
        n = min(len(frames_a), len(frames_b))
        if n == 0:
            raise ValueError(f"No frames to compare between {video_a} and {video_b}")

        clip_sims: list[float] = []
        lpips_vals: list[float] = []
        style_vals: list[float] = []

        for fa, fb in zip(frames_a[:n], frames_b[:n]):
            ra = resize_square(fa, self.frame_size)
            rb = resize_square(fb, self.frame_size)

            ea = self._clip_embed(ra)
            eb = self._clip_embed(rb)
            clip_sims.append(float((ea * eb).sum().item()))

            ta = _to_tensor(ra, self.device) * 2.0 - 1.0
            tb = _to_tensor(rb, self.device) * 2.0 - 1.0
            lpips_vals.append(float(self.lpips_fn(ta, tb).item()))

            sa = _to_tensor(ra, self.device)
            sb = _to_tensor(rb, self.device)
            style_vals.append(self.style_fn(sa, sb))

        return VideoMetrics(
            clip_similarity=float(np.mean(clip_sims)),
            lpips=float(np.mean(lpips_vals)),
            style_loss=float(np.mean(style_vals)),
            num_frames_compared=n,
        )
