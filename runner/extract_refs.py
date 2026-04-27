"""Extract a single reference frame from each video as a PNG.

Image-conditioned video generators (Runway, Pika, AnimateDiff+IP-Adapter)
take an image, not a video. This script picks one representative frame per
clip and writes it as a PNG, using the SAME frame index for both original
and glazed versions of a clip so the only variable in downstream generation
is the glaze perturbation itself.

Usage:
    # Extract from originals (gifs):
    python -m runner.extract_refs --videos clips --out refs/original --frame median
    # Extract from glazed (mp4s):
    python -m runner.extract_refs --videos clips/glazed --out refs/glazed --frame median --suffix _glazed
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal

import cv2

from runner.video_io import read_frames

FrameChoice = Literal["first", "median", "last"]


def pick_frame_index(num_frames: int, choice: FrameChoice) -> int:
    if num_frames < 1:
        raise ValueError("Video has no frames")
    if choice == "first":
        return 0
    if choice == "last":
        return num_frames - 1
    if choice == "median":
        return num_frames // 2
    raise ValueError(f"Unknown frame choice: {choice}")


def extract(video_path: Path, out_path: Path, choice: FrameChoice) -> None:
    frames = read_frames(video_path)
    idx = pick_frame_index(len(frames), choice)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(frames[idx], cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(out_path), bgr):
        raise RuntimeError(f"Failed to write {out_path}")
    print(f"[done] {video_path.name} → {out_path.name} (frame {idx} of {len(frames)})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract a single reference frame from each video.")
    parser.add_argument("--videos", type=Path, required=True, help="Directory of source videos (.gif/.mp4).")
    parser.add_argument("--out", type=Path, required=True, help="Output directory for PNGs.")
    parser.add_argument("--frame", choices=("first", "median", "last"), default="median")
    parser.add_argument(
        "--suffix",
        default="",
        help="Filename suffix to strip from input stem when naming output (e.g. '_glazed').",
    )
    parser.add_argument(
        "--ext",
        nargs="+",
        default=("gif", "mp4"),
        help="Video extensions to scan.",
    )
    args = parser.parse_args()

    video_paths: list[Path] = []
    for ext in args.ext:
        video_paths.extend(sorted(args.videos.glob(f"*.{ext}")))
    if not video_paths:
        raise SystemExit(f"No videos found in {args.videos} with extensions {args.ext}")

    for vp in video_paths:
        stem = vp.stem.removesuffix(args.suffix) if args.suffix else vp.stem
        out_path = args.out / f"{stem}.png"
        extract(vp, out_path, args.frame)


if __name__ == "__main__":
    main()
