"""Run the video Evaluator on a pair of mp4 files and print the metrics."""

from __future__ import annotations

from pathlib import Path

from runner.metrics import Evaluator


def main() -> None:
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ev = Evaluator(device=device, frame_size=224, num_frames=16)
    metrics = ev.compare(
        Path("outputs/from_original.mp4"),
        Path("outputs/from_glazed.mp4"),
    )

    print(f"CLIP similarity:  {metrics.clip_similarity:.4f}  (higher = more similar)")
    print(f"LPIPS distance:   {metrics.lpips:.4f}        (lower  = more similar)")
    print(f"Style loss:       {metrics.style_loss:.6f}      (lower  = more similar)")
    print(f"Frames compared:  {metrics.num_frames_compared}")


if __name__ == "__main__":
    main()
