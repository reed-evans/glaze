"""Shared video IO utilities for the runner."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import cv2
import numpy as np


def read_frames(path: Path) -> list[np.ndarray]:
    """Decode an mp4 into a list of HxWx3 uint8 RGB frames."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")
    frames: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise ValueError(f"No frames decoded from {path}")
    return frames


def write_frames(frames: Sequence[np.ndarray], path: Path, fps: int) -> None:
    """Write HxWx3 uint8 RGB frames to an mp4 using OpenCV's mp4v codec."""
    if not frames:
        raise ValueError("Cannot write an empty frame list")
    h, w = frames[0].shape[:2]
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"OpenCV VideoWriter failed to open {path}")
    try:
        for f in frames:
            writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def subsample(frames: Sequence[np.ndarray], n: int) -> list[np.ndarray]:
    """Take n frames uniformly spaced across the sequence (or all if shorter)."""
    if len(frames) <= n:
        return list(frames)
    indices = np.linspace(0, len(frames) - 1, n).astype(int)
    return [frames[i] for i in indices]


def resize_square(frame: np.ndarray, size: int) -> np.ndarray:
    return cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)
