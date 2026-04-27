"""DAMO image-to-video runner (I2VGen-XL).

Replaces the original text-only ModelScope pipeline (damo-vilab/text-to-video-ms-1.7b)
with its image-conditioned successor I2VGen-XL (ali-vilab/i2vgen-xl) so that glaze
perturbations in the reference frame can actually reach the generator. Both models
come from DAMO/Ali-Vilab — calling the generator label 'modelscope' in output filenames
preserves continuity with aggregate.py and the experiment design.

Conditioning model: I2VGen-XL is keyframe-style (the reference image becomes the
opening frame of the generated clip), not a pure style adapter like IP-Adapter.
That means the test measures: does a glazed keyframe lead the model to animate
in a different style than the un-glazed keyframe? It is a weaker signal than a
true style reference, but it is the strongest signal available without leaving
the DAMO model family.

Inputs per (clip, version):
    refs/original/{clip}.png    — un-glazed median frame from clips/{clip}.gif
    refs/glazed/{clip}.png      — glazed median frame from clips/glazed/{clip}_glazed.mp4
    runner/prompts.json["_content_prompt"]  — single shared content prompt

Output naming: {clip}_modelscope_{original|glazed}.mp4

Usage:
    python -m runner.modelscope \\
        --prompts runner/prompts.json \\
        --refs-original refs/original \\
        --refs-glazed refs/glazed \\
        --out outputs
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

import numpy as np
import torch
from diffusers import I2VGenXLPipeline
from diffusers.utils import load_image

from runner.video_io import write_frames

MODEL_ID = "ali-vilab/i2vgen-xl"
GENERATOR_LABEL = "modelscope"
Version = Literal["original", "glazed"]


@dataclass(frozen=True)
class GenerationJob:
    clip: str
    version: Version
    prompt: str
    image_path: Path
    seed: int


def load_content_prompt(prompts_path: Path | None, override: str | None) -> str:
    if override:
        return override
    if prompts_path is None:
        raise ValueError("Provide --content-prompt or a --prompts file with a '_content_prompt' key.")
    with prompts_path.open("r") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{prompts_path} must be a JSON object")
    prompt = data.get("_content_prompt")
    if not isinstance(prompt, str) or not prompt:
        raise ValueError(
            f"{prompts_path} must define '_content_prompt' as a non-empty string, "
            f"or pass --content-prompt on the command line."
        )
    return prompt


def discover_refs(refs_dir: Path) -> dict[str, Path]:
    if not refs_dir.is_dir():
        raise FileNotFoundError(f"Not a directory: {refs_dir}")
    return {p.stem: p for p in sorted(refs_dir.glob("*.png"))}


def plan_jobs(
    content_prompt: str,
    versions: Iterable[Version],
    refs_by_version: dict[Version, dict[str, Path]],
    seed: int,
) -> list[GenerationJob]:
    all_clips = sorted({clip for refs in refs_by_version.values() for clip in refs})
    jobs: list[GenerationJob] = []
    for clip in all_clips:
        for version in versions:
            refs = refs_by_version.get(version, {})
            if clip not in refs:
                print(f"[skip] no {version} reference for clip {clip!r}")
                continue
            jobs.append(
                GenerationJob(
                    clip=clip,
                    version=version,
                    prompt=content_prompt,
                    image_path=refs[clip],
                    seed=seed,
                )
            )
    return jobs


def _frames_to_uint8(frames: object) -> list[np.ndarray]:
    if isinstance(frames, list) and frames and isinstance(frames[0], np.ndarray) and frames[0].ndim == 4:
        frames = frames[0]
    if isinstance(frames, np.ndarray):
        if frames.ndim != 4:
            raise ValueError(f"Expected (F,H,W,3) array, got shape {frames.shape}")
        arr = frames if frames.dtype == np.uint8 else (frames.clip(0.0, 1.0) * 255.0).astype(np.uint8)
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
    pipe: I2VGenXLPipeline,
    job: GenerationJob,
    out_dir: Path,
    num_frames: int,
    num_inference_steps: int,
    guidance_scale: float,
    negative_prompt: str,
    fps: int,
    device: str,
) -> Path:
    out_path = out_dir / f"{job.clip}_{GENERATOR_LABEL}_{job.version}.mp4"
    if out_path.exists():
        print(f"[skip] {out_path.name} already exists")
        return out_path

    image = load_image(str(job.image_path))
    gen = torch.Generator(device=device).manual_seed(job.seed)
    result = pipe(
        prompt=job.prompt,
        image=image,
        num_frames=num_frames,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        negative_prompt=negative_prompt,
        generator=gen,
    )
    frames = _frames_to_uint8(result.frames)
    write_frames(frames, out_path, fps=fps)
    print(f"[done] {out_path.name} ({len(frames)} frames from {job.image_path.name})")
    return out_path


def build_pipeline(model_id: str, dtype: str, device: str, cpu_offload: bool) -> I2VGenXLPipeline:
    torch_dtype = torch.float16 if dtype == "fp16" else torch.float32
    variant = "fp16" if dtype == "fp16" else None
    pipe = I2VGenXLPipeline.from_pretrained(model_id, torch_dtype=torch_dtype, variant=variant)
    if cpu_offload and device == "cuda":
        pipe.enable_model_cpu_offload()
    else:
        pipe = pipe.to(device)
    return pipe


def main() -> None:
    parser = argparse.ArgumentParser(description="I2VGen-XL image-to-video batch runner")
    parser.add_argument("--prompts", type=Path, default=None, help="JSON file with a '_content_prompt' key.")
    parser.add_argument("--content-prompt", default=None, help="Inline content prompt; overrides --prompts.")
    parser.add_argument("--refs-original", type=Path, required=True, help="Directory of un-glazed reference PNGs.")
    parser.add_argument("--refs-glazed", type=Path, required=True, help="Directory of glazed reference PNGs.")
    parser.add_argument("--out", type=Path, required=True, help="Output directory.")
    parser.add_argument(
        "--versions",
        nargs="+",
        choices=("original", "glazed"),
        default=("original", "glazed"),
    )
    parser.add_argument("--seed", type=int, default=42, help="Shared across versions for fairness.")
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=9.0)
    parser.add_argument(
        "--negative-prompt",
        default="distorted, discontinuous, ugly, blurry, low quality, low resolution",
    )
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--dtype", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--no-cpu-offload", action="store_true", help="Disable diffusers' cpu-offload.")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    content_prompt = load_content_prompt(args.prompts, args.content_prompt)
    refs_by_version: dict[Version, dict[str, Path]] = {
        "original": discover_refs(args.refs_original),
        "glazed": discover_refs(args.refs_glazed),
    }
    jobs = plan_jobs(content_prompt, args.versions, refs_by_version, args.seed)
    if not jobs:
        raise SystemExit("No jobs planned — check that --refs-original and --refs-glazed are populated.")

    pipe = build_pipeline(
        model_id=args.model_id,
        dtype=args.dtype,
        device=args.device,
        cpu_offload=not args.no_cpu_offload,
    )

    print(f"Content prompt: {content_prompt!r}")
    print(f"Planned {len(jobs)} jobs.")

    for job in jobs:
        generate(
            pipe=pipe,
            job=job,
            out_dir=args.out,
            num_frames=args.num_frames,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            negative_prompt=args.negative_prompt,
            fps=args.fps,
            device=args.device,
        )


if __name__ == "__main__":
    main()
