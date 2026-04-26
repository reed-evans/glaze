"""ModelScope text-to-video batch runner.

Generates videos using damo-vilab/text-to-video-ms-1.7b for each (clip, version)
pair with deterministic seeds, writing files named {clip}_modelscope_{version}.mp4
so that aggregate.py can pair originals and glazed outputs.

IMPORTANT — honesty about what this script tests:
    ModelScope 1.7b is a *pure text-to-video* model. It takes no image or video
    conditioning. If both the 'original' and 'glazed' runs use the same prompt
    (as they should, for a fair comparison), the only source of variation is
    sampling noise — glaze cannot flow through a text prompt. Use this as a
    control/baseline to confirm generator variance, and reach for an image- or
    video-conditioned pipeline (AnimateDiff + IP-Adapter, I2VGen-XL, SVD) when
    you need to probe glaze's real effect.

Usage:
    python -m runner.modelscope --prompts runner/prompts.json --out outputs/
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

import numpy as np
import torch
from diffusers import DiffusionPipeline

from runner.video_io import write_frames

MODEL_ID = "damo-vilab/text-to-video-ms-1.7b"
Version = Literal["original", "glazed"]


@dataclass(frozen=True)
class GenerationJob:
    clip: str
    version: Version
    prompt: str
    seed: int


def load_prompts(prompts_path: Path) -> dict[str, str]:
    with prompts_path.open("r") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object mapping clip→prompt in {prompts_path}")
    for k, v in data.items():
        if not isinstance(k, str) or not isinstance(v, str):
            raise ValueError(f"Prompts must be strings; got {type(k).__name__}:{type(v).__name__}")
    return data


def plan_jobs(
    prompts: dict[str, str],
    versions: Iterable[Version],
    seed: int,
) -> list[GenerationJob]:
    return [
        GenerationJob(clip=clip, version=version, prompt=prompt, seed=seed)
        for clip, prompt in prompts.items()
        for version in versions
    ]


def _frames_to_uint8(frames: object) -> list[np.ndarray]:
    """Normalize diffusers' varied output types to a list of HxWx3 uint8 arrays."""
    if isinstance(frames, list) and frames and isinstance(frames[0], np.ndarray) and frames[0].ndim == 4:
        frames = frames[0]
    if isinstance(frames, np.ndarray):
        if frames.ndim != 4:
            raise ValueError(f"Expected (F,H,W,3) array, got shape {frames.shape}")
        arr = frames
        if arr.dtype != np.uint8:
            arr = (arr.clip(0.0, 1.0) * 255.0).astype(np.uint8)
        return [arr[i] for i in range(arr.shape[0])]
    if isinstance(frames, list):
        out: list[np.ndarray] = []
        for f in frames:
            if hasattr(f, "convert"):
                out.append(np.array(f.convert("RGB")))
            elif isinstance(f, np.ndarray):
                out.append(f if f.dtype == np.uint8 else (f.clip(0.0, 1.0) * 255.0).astype(np.uint8))
            else:
                raise TypeError(f"Unsupported frame type from pipeline: {type(f).__name__}")
        return out
    raise TypeError(f"Unsupported .frames type: {type(frames).__name__}")


def generate(
    pipe: DiffusionPipeline,
    job: GenerationJob,
    out_dir: Path,
    num_frames: int,
    num_inference_steps: int,
    fps: int,
    device: str,
) -> Path:
    out_path = out_dir / f"{job.clip}_modelscope_{job.version}.mp4"
    if out_path.exists():
        print(f"[skip] {out_path.name} already exists")
        return out_path

    gen = torch.Generator(device=device).manual_seed(job.seed)
    result = pipe(
        job.prompt,
        num_frames=num_frames,
        num_inference_steps=num_inference_steps,
        generator=gen,
    )
    frames = _frames_to_uint8(result.frames)
    write_frames(frames, out_path, fps=fps)
    print(f"[done] {out_path.name} ({len(frames)} frames)")
    return out_path


def build_pipeline(model_id: str, dtype: str, device: str, cpu_offload: bool) -> DiffusionPipeline:
    torch_dtype = torch.float16 if dtype == "fp16" else torch.float32
    variant = "fp16" if dtype == "fp16" else None
    pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch_dtype, variant=variant)
    if cpu_offload and device == "cuda":
        pipe.enable_model_cpu_offload()
    else:
        pipe = pipe.to(device)
    return pipe


def main() -> None:
    parser = argparse.ArgumentParser(description="ModelScope text-to-video batch runner")
    parser.add_argument("--prompts", type=Path, required=True, help="JSON mapping clip_name → prompt")
    parser.add_argument("--out", type=Path, required=True, help="Output directory")
    parser.add_argument(
        "--versions",
        nargs="+",
        choices=("original", "glazed"),
        default=("original", "glazed"),
        help="Which versions to emit (ModelScope ignores visual inputs; this just controls filenames).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Base seed; identical across versions for fairness.")
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--num-inference-steps", type=int, default=25)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--dtype", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument(
        "--no-cpu-offload",
        action="store_true",
        help="Disable diffusers' cpu-offload. Keep enabled on <16GB VRAM.",
    )
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    prompts = load_prompts(args.prompts)
    jobs = plan_jobs(prompts, args.versions, args.seed)

    pipe = build_pipeline(
        model_id=args.model_id,
        dtype=args.dtype,
        device=args.device,
        cpu_offload=not args.no_cpu_offload,
    )

    for job in jobs:
        generate(
            pipe=pipe,
            job=job,
            out_dir=args.out,
            num_frames=args.num_frames,
            num_inference_steps=args.num_inference_steps,
            fps=args.fps,
            device=args.device,
        )


if __name__ == "__main__":
    main()
