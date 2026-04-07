from __future__ import annotations

from pathlib import Path

import click
import torch


def _default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@click.group()
def main() -> None:
    """Glaze video/image cloaking CLI — protect your art from AI style mimicry."""


# ---------------------------------------------------------------------------
# style-transfer command
# ---------------------------------------------------------------------------

@main.command("style-transfer")
@click.argument("source_path", type=click.Path(exists=True, path_type=Path))
@click.argument("style_path", type=click.Path(exists=True, path_type=Path))
@click.argument("output_path", type=click.Path(path_type=Path))
@click.option(
    "--strength",
    type=float,
    default=0.5,
    show_default=True,
    help="How strongly to apply the style (0.0–1.0). Higher deviates more from source content.",
)
@click.option(
    "--ip-adapter-scale",
    type=float,
    default=0.6,
    show_default=True,
    help="IP-Adapter style conditioning weight (0.0 = ignore style image, 1.0 = full style).",
)
@click.option(
    "--steps",
    type=int,
    default=20,
    show_default=True,
    help="Number of diffusion denoising steps.",
)
@click.option(
    "--guidance-scale",
    type=float,
    default=7.5,
    show_default=True,
    help="Classifier-free guidance scale.",
)
@click.option(
    "--model-id",
    default="stable-diffusion-v1-5/stable-diffusion-v1-5",
    show_default=True,
    help="HuggingFace model ID for the SD img2img pipeline.",
)
@click.option(
    "--size",
    type=int,
    default=512,
    show_default=True,
    help="Resize the longer side of the source image to this value before processing. "
         "SD v1.5 is designed for 512; use 768 for slightly higher quality.",
)
@click.option(
    "--no-ip-adapter",
    is_flag=True,
    default=False,
    help=(
        "Skip IP-Adapter and use a text prompt instead. "
        "Faster to test; requires --style-prompt."
    ),
)
@click.option(
    "--style-prompt",
    default="in the style of impressionist painting",
    show_default=True,
    help="Text prompt used when --no-ip-adapter is set.",
)
@click.option(
    "--device",
    default=_default_device(),
    show_default=True,
    help="Torch device (cuda / cpu / mps).",
)
def style_transfer_cmd(
    source_path: Path,
    style_path: Path,
    output_path: Path,
    strength: float,
    ip_adapter_scale: float,
    steps: int,
    guidance_scale: float,
    model_id: str,
    size: int,
    no_ip_adapter: bool,
    style_prompt: str,
    device: str,
) -> None:
    """Transfer the style of STYLE_PATH onto SOURCE_PATH and save to OUTPUT_PATH.

    SOURCE_PATH  Content image to restyle.
    STYLE_PATH   Reference image whose visual style is applied.
    OUTPUT_PATH  Destination for the styled output image.

    Example:

        glaze style-transfer photo.png monet.jpg styled.png --strength 0.6
    """
    import numpy as np
    from PIL import Image
    from glaze.style.transfer import StyleTransfer

    dev = torch.device(device)

    def load_image(path: Path) -> torch.Tensor:
        arr = np.array(Image.open(path).convert("RGB"), dtype=np.float32)
        t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0
        return t.to(dev)

    use_ip_adapter = not no_ip_adapter
    if use_ip_adapter:
        click.echo(f"Loading style transfer pipeline with IP-Adapter ({model_id})…")
    else:
        click.echo(f"Loading style transfer pipeline, no IP-Adapter ({model_id})…")

    transfer = StyleTransfer(model_id=model_id, device=device, use_ip_adapter=use_ip_adapter)

    source = load_image(source_path)
    style = load_image(style_path)

    click.echo(
        f"Transferring style  [{style_path.name} → {source_path.name}]  "
        f"strength={strength}"
        + (f"  ip_adapter_scale={ip_adapter_scale}" if use_ip_adapter else f"  prompt={style_prompt!r}")
    )

    import traceback as tb
    try:
        click.echo("Step 1/3: loading pipeline…")
        _ = transfer._get_pipeline()
        click.echo("Step 2/3: running inference…")
        result = transfer.transfer(
            source=source,
            style_image=style,
            strength=strength,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            ip_adapter_scale=ip_adapter_scale,
            style_prompt=style_prompt,
            max_size=size,
        )
        click.echo("Step 3/3: saving output…")
        out_arr = result.squeeze(0).permute(1, 2, 0).cpu().float()
        out_arr = ((out_arr + 1.0) / 2.0 * 255.0).clamp(0, 255).byte().numpy()
        Image.fromarray(out_arr).save(output_path)
        click.echo(f"Saved → {output_path}")
    except Exception:
        click.echo("\n--- ERROR ---", err=True)
        click.echo(tb.format_exc(), err=True)
        raise SystemExit(1)


# ---------------------------------------------------------------------------
# image command
# ---------------------------------------------------------------------------

@main.command("image")
@click.argument("input_path", type=click.Path(exists=True, path_type=Path))
@click.argument("output_path", type=click.Path(path_type=Path))
@click.option(
    "--encoder",
    type=click.Choice(["clip", "vae"]),
    default="clip",
    show_default=True,
    help="Feature encoder: 'clip' (CLIP ViT-L/14) or 'vae' (Stable Diffusion VAE).",
)
@click.option(
    "--style-prompt",
    default=None,
    help="Target style text prompt. If omitted, a style is auto-selected.",
)
@click.option(
    "--budget",
    type=float,
    default=0.05,
    show_default=True,
    help="Perturbation budget (LPIPS threshold). Higher = stronger protection, more visible.",
)
@click.option(
    "--iterations",
    type=int,
    default=500,
    show_default=True,
    help="Optimization iterations.",
)
@click.option(
    "--max-size",
    type=int,
    default=None,
    show_default=True,
    help=(
        "Resize the image so its longer side is at most this many pixels before glazing, "
        "then upscale the result back. Required when using --encoder vae on large images "
        "(the VAE attention layer is O(n²) in spatial size). "
        "CLIP encoder is safe at any resolution. Defaults to 512 when --encoder vae is used."
    ),
)
@click.option(
    "--device",
    default=_default_device(),
    show_default=True,
    help="Torch device (cuda / cpu / mps).",
)
def glaze_image(
    input_path: Path,
    output_path: Path,
    encoder: str,
    style_prompt: str | None,
    budget: float,
    iterations: int,
    max_size: int | None,
    device: str,
) -> None:
    """Cloak a single IMAGE to protect its style from AI mimicry.

    INPUT_PATH  Source image (PNG/JPG/WEBP).
    OUTPUT_PATH Destination for the cloaked image.
    """
    from PIL import Image
    import numpy as np
    from glaze.glaze import GlazeConfig, GlazeOptimizer
    from glaze.style.selector import StyleSelector, DEFAULT_STYLE_PROMPTS

    click.echo(f"Loading encoder: {encoder}")
    enc = _load_encoder(encoder, device)

    config = GlazeConfig(
        perturbation_budget=budget,
        iterations=iterations,
        device=device,
    )
    optimizer = GlazeOptimizer(enc, config)

    # Select target style
    if style_prompt is None:
        import random
        style_prompt = random.choice(DEFAULT_STYLE_PROMPTS)
        click.echo(f"Auto-selected target style: {style_prompt}")
    else:
        click.echo(f"Target style: {style_prompt}")

    # Load image → tensor
    dev = torch.device(device)
    img = Image.open(input_path).convert("RGB")
    orig_w, orig_h = img.size
    arr = np.array(img, dtype=np.float32)
    source = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0
    source = source.to(dev)

    # VAE has O(n²) attention — enforce a size limit to avoid OOM.
    # CLIP resizes internally to 224×224 so it's always safe.
    effective_max_size = max_size or (512 if encoder == "vae" else None)
    if effective_max_size is not None:
        source = _fit_to_size(source, effective_max_size)
        if source.shape[2:] != (orig_h, orig_w):
            click.echo(f"Resized to {source.shape[3]}×{source.shape[2]} for processing.")

    # Compute target features (identity fallback — no style transfer model required)
    with torch.inference_mode():
        target_features = enc.encode(source)

    click.echo(f"Glazing image ({source.shape[3]}×{source.shape[2]})…")
    cloaked = optimizer.cloak_image(source, target_features, progress=True)

    # Upscale back to original resolution if we downscaled
    import torch.nn.functional as F
    if cloaked.shape[2:] != (orig_h, orig_w):
        cloaked = F.interpolate(cloaked, size=(orig_h, orig_w), mode="bilinear", align_corners=False)

    # Save
    out_arr = cloaked.squeeze(0).permute(1, 2, 0).cpu().float()
    out_arr = ((out_arr + 1.0) / 2.0 * 255.0).clamp(0, 255).byte().numpy()
    Image.fromarray(out_arr).save(output_path)
    click.echo(f"Saved cloaked image → {output_path}")


# ---------------------------------------------------------------------------
# video command
# ---------------------------------------------------------------------------

@main.command("video")
@click.argument("input_path", type=click.Path(exists=True, path_type=Path))
@click.argument("output_path", type=click.Path(path_type=Path))
@click.option(
    "--encoder",
    type=click.Choice(["clip", "vae"]),
    default="clip",
    show_default=True,
    help="Feature encoder.",
)
@click.option(
    "--style-prompt",
    default=None,
    help="Target style text prompt. Auto-selected if omitted.",
)
@click.option(
    "--budget",
    type=float,
    default=0.05,
    show_default=True,
    help="Perturbation budget (LPIPS threshold).",
)
@click.option(
    "--iterations",
    type=int,
    default=200,
    show_default=True,
    help="Optimization iterations per frame. Lower is faster but weaker.",
)
@click.option(
    "--process-size",
    type=int,
    default=512,
    show_default=True,
    help="Resize frames to this size before glazing (speeds up processing).",
)
@click.option(
    "--temporal-threshold",
    type=float,
    default=0.02,
    show_default=True,
    help="Frame similarity threshold. Frames within this delta re-use prior perturbation.",
)
@click.option(
    "--no-warm-start",
    is_flag=True,
    default=False,
    help="Disable warm-start initialization from previous frame's perturbation.",
)
@click.option(
    "--device",
    default=_default_device(),
    show_default=True,
    help="Torch device.",
)
def glaze_cmd(
    input_path: Path,
    output_path: Path,
    encoder: str,
    style_prompt: str | None,
    budget: float,
    iterations: int,
    process_size: int,
    temporal_threshold: float,
    no_warm_start: bool,
    device: str,
) -> None:
    """Cloak every frame of a VIDEO to protect its style from AI mimicry.

    INPUT_PATH  Source video file.
    OUTPUT_PATH Destination for the cloaked video.
    """
    import random
    from glaze.glaze import GlazeConfig, GlazeOptimizer
    from glaze.video import VideoGlazer, VideoGlazeConfig
    from glaze.style.selector import DEFAULT_STYLE_PROMPTS

    if style_prompt is None:
        style_prompt = random.choice(DEFAULT_STYLE_PROMPTS)
        click.echo(f"Auto-selected target style: {style_prompt}")
    else:
        click.echo(f"Target style: {style_prompt}")

    click.echo(f"Loading encoder: {encoder}")
    enc = _load_encoder(encoder, device)

    glaze_config = GlazeConfig(
        perturbation_budget=budget,
        iterations=iterations,
        device=device,
    )
    video_config = VideoGlazeConfig(
        glaze_config=glaze_config,
        temporal_threshold=temporal_threshold,
        warm_start=not no_warm_start,
        process_size=process_size,
    )
    optimizer = GlazeOptimizer(enc, glaze_config)
    glazer = VideoGlazer(optimizer, video_config)

    click.echo(f"Processing {input_path} → {output_path}")
    glazer.glaze(input_path, output_path, style_prompt)
    click.echo("Done.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fit_to_size(tensor: torch.Tensor, max_size: int) -> torch.Tensor:
    """Resize so the longer side equals max_size, rounded to nearest 64."""
    import torch.nn.functional as F
    _, _, h, w = tensor.shape
    scale = max_size / max(h, w)
    new_h = round(h * scale / 64) * 64
    new_w = round(w * scale / 64) * 64
    if (new_h, new_w) == (h, w):
        return tensor
    return F.interpolate(tensor, size=(new_h, new_w), mode="bilinear", align_corners=False)


def _load_encoder(name: str, device: str) -> "FeatureEncoder":  # noqa: F821
    dev = torch.device(device)
    if name == "clip":
        from glaze.encoders.clip_encoder import CLIPEncoder
        enc = CLIPEncoder()
        return enc.to(dev)
    elif name == "vae":
        from glaze.encoders.vae_encoder import VAEEncoder
        enc = VAEEncoder()
        return enc.to(dev)
    raise ValueError(f"Unknown encoder: {name}")
