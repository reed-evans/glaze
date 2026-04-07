# glaze

A Python/PyTorch implementation of the [Glaze](https://glaze.cs.uchicago.edu/) image-cloaking algorithm, extended for video. Glaze adds imperceptible perturbations to images that cause AI style-mimicry models to misidentify the artist's style, protecting artwork from being used to train style-copying models without consent.

## How it works

Glaze solves the following optimization for each image:

```
min_δ  ‖Φ(Ω(x,T)) − Φ(x+δ)‖²  +  α · max(LPIPS(x, x+δ) − p, 0)
```

- **x** is the original image, **δ** is the perturbation being optimized
- **Φ** is a frozen feature encoder (CLIP or VAE) — the same kind of encoder a diffusion model uses internally to understand style
- **Ω(x, T)** is the source image rendered in a target style T (computed via Stable Diffusion img2img + IP-Adapter)
- The first term pulls the cloaked image's features toward the target style in feature space
- The second term keeps the perturbation invisible by penalizing any LPIPS perceptual distance above budget **p**
- **α** is an adaptive penalty weight that tightens automatically when the budget is violated

The result is an image that looks unchanged to a human but appears to be in a completely different style to an AI.

## Installation

Requires Python 3.10+, PyTorch 2.0+.

```bash
git clone https://github.com/your-org/glaze
cd glaze
```

If using uv:

```bash
uv sync
uv run glaze [SUBCOMMAND] ...
```

Otherwise:

```bash
pip install -e .
glaze [SUBCOMMAND] ...
```

For development dependencies (pytest, black, ruff, mypy):

```bash
pip install -e ".[dev]"
```

A HuggingFace account and token are required to download model weights. Set the token before running any command:

```bash
export HF_TOKEN=your_token_here
# or
huggingface-cli login
```

Models are downloaded automatically on first use and cached in `~/.cache/huggingface/`.

---

## Commands

### `glaze image`

Cloak a single image to protect its style from AI mimicry.

```
glaze image INPUT_PATH OUTPUT_PATH [OPTIONS]
```

**Arguments**

| Argument | Description |
|---|---|
| `INPUT_PATH` | Source image (PNG, JPG, or WEBP) |
| `OUTPUT_PATH` | Destination for the cloaked image |

**Options**

| Option | Default | Description |
|---|---|---|
| `--encoder` | `clip` | Feature encoder: `clip` (CLIP ViT-L/14) or `vae` (Stable Diffusion VAE). See [Choosing an encoder](#choosing-an-encoder). |
| `--budget` | `0.05` | Perturbation budget (LPIPS threshold). Higher = stronger protection but more visible. `0.05` is imperceptible to most people; `0.1`–`0.2` for stronger protection. |
| `--iterations` | `500` | Adam optimizer steps. The paper uses 500; fewer iterations converge faster but give weaker cloaking. |
| `--max-size` | auto | Resize the longer side to at most this many pixels before glazing, then upscale the result back. See [Image size and memory](#image-size-and-memory). |
| `--style-prompt` | auto | Target style text prompt. If omitted, a style is randomly selected from a curated list of 20 diverse art styles that are distant from typical photographic content. |
| `--device` | auto | Torch device. Auto-detected in priority order: `cuda` → `mps` → `cpu`. See [Device selection](#device-selection). |

**Examples**

```bash
# Basic — cloak a painting with auto-selected style
glaze image artwork.png cloaked.png

# Explicit style and stronger budget
glaze image artwork.png cloaked.png \
    --style-prompt "in the style of cubism, geometric fragmented shapes" \
    --budget 0.08

# Use the VAE encoder with explicit size cap
glaze image artwork.png cloaked.png \
    --encoder vae \
    --max-size 512
```

---

### `glaze video`

Cloak every frame of a video to protect its style from AI mimicry.

```
glaze video INPUT_PATH OUTPUT_PATH [OPTIONS]
```

**Arguments**

| Argument | Description |
|---|---|
| `INPUT_PATH` | Source video file |
| `OUTPUT_PATH` | Destination for the cloaked video |

**Options**

| Option | Default | Description |
|---|---|---|
| `--encoder` | `clip` | Feature encoder. Same choice as `image` command. |
| `--budget` | `0.05` | Perturbation budget (LPIPS threshold). |
| `--iterations` | `200` | Optimizer steps per frame. Intentionally lower than the image default (500) — video has many frames and temporal coherence reduces the need for full convergence on each one. |
| `--process-size` | `512` | Resize frames to this size before glazing. Frames are upscaled back to original resolution after. Smaller = faster; larger = more faithful cloaking. |
| `--temporal-threshold` | `0.02` | Frame similarity threshold. When a frame's pixel-level difference from the previous frame is within this threshold, the previous perturbation is re-used without re-running the optimizer. Eliminates redundant work in static or slow-moving shots. Set to `0` to process every frame independently. |
| `--no-warm-start` | off | By default, each frame's optimizer is initialized from the previous frame's perturbation δ rather than from zeros. This warm start typically halves the iterations needed to converge on similar frames. Pass this flag to disable it. |
| `--style-prompt` | auto | Target style prompt, auto-selected if omitted. |
| `--device` | auto | Torch device. |

**Examples**

```bash
# Basic
glaze video input.mp4 output.mp4

# Faster processing at lower quality
glaze video input.mp4 output.mp4 \
    --iterations 100 \
    --process-size 384

# Process every frame fully with no temporal shortcuts
glaze video input.mp4 output.mp4 \
    --temporal-threshold 0 \
    --no-warm-start \
    --iterations 500
```

---

### `glaze style-transfer`

Transfer the visual style of a reference image onto a source image. This is the component that computes Ω(x, T) — the source image rendered in target style T. Exposed as a standalone command primarily for testing and experimentation.

```
glaze style-transfer SOURCE_PATH STYLE_PATH OUTPUT_PATH [OPTIONS]
```

**Arguments**

| Argument | Description |
|---|---|
| `SOURCE_PATH` | Content image whose structure is preserved |
| `STYLE_PATH` | Reference image whose visual style is applied |
| `OUTPUT_PATH` | Destination for the styled output |

**Options**

| Option | Default | Description |
|---|---|---|
| `--strength` | `0.5` | How far the diffusion process is allowed to wander from the source content (0.0 = no change, 1.0 = ignore source). |
| `--ip-adapter-scale` | `0.6` | IP-Adapter conditioning weight. Controls how strongly the style reference image influences the output (0.0 = no style influence, 1.0 = maximum style). |
| `--steps` | `20` | Number of diffusion denoising steps. More steps = higher quality, slower. |
| `--guidance-scale` | `7.5` | Classifier-free guidance scale. |
| `--size` | `512` | Resize the longer side of the source image to this value before processing. See [Image size and memory](#image-size-and-memory). |
| `--no-ip-adapter` | off | Skip IP-Adapter and use `--style-prompt` as a text prompt instead. Useful for testing the pipeline without downloading IP-Adapter weights. |
| `--style-prompt` | `"in the style of impressionist painting"` | Text prompt used when `--no-ip-adapter` is set. |
| `--model-id` | `stable-diffusion-v1-5/stable-diffusion-v1-5` | HuggingFace model ID for the SD img2img pipeline. See [Model ID history](#model-id-history). |
| `--device` | auto | Torch device. |

**Examples**

```bash
# Full IP-Adapter style transfer
glaze style-transfer photo.png van-gogh.jpg styled.png

# Stronger style application
glaze style-transfer photo.png van-gogh.jpg styled.png \
    --strength 0.7 \
    --ip-adapter-scale 0.8

# Test without IP-Adapter (faster, text-guided only)
glaze style-transfer photo.png van-gogh.jpg styled.png \
    --no-ip-adapter \
    --style-prompt "in the style of post-impressionism, swirling brushstrokes"
```

---

## Implementation notes

### Choosing an encoder

Two feature encoders are available, both sourced from HuggingFace. The encoder is Φ in the optimization — it determines what "style" means in feature space.

**`clip` (default)** uses CLIP ViT-L/14 (`openai/clip-vit-large-patch14`). The CLIPProcessor always resizes images to 224×224 internally before encoding, so it is safe to use at any input resolution without hitting memory issues.

**`vae`** uses the VAE from Stable Diffusion 2.1 (`stabilityai/stable-diffusion-2-1`). The VAE encoder has a self-attention layer in its mid-block whose memory cost is O(n²) in the spatial dimensions of the input. This makes it unsafe at large resolutions — see [Image size and memory](#image-size-and-memory).

CLIP is the better default for most use cases. The VAE encoder is provided for experimentation; it encodes in SD's native latent space, which may improve cloaking effectiveness against SD-based mimicry models specifically.

### Image size and memory

The Stable Diffusion VAE (used by both `--encoder vae` and the style transfer pipeline) contains a self-attention layer that allocates an attention matrix proportional to the square of the spatial dimensions of the input. Passing a large image directly causes a memory error like:

```
RuntimeError: Invalid buffer size: 67.60 GiB
```

SD v1.5 was designed for 512×512 images. All commands that use the VAE automatically cap input resolution:

- `style-transfer`: always resizes to `--size` (default 512) before processing
- `image` with `--encoder vae`: automatically applies a 512px cap; override with `--max-size`
- `image` with `--encoder clip`: no resizing applied (CLIP handles it internally at 224×224)
- `video`: always resizes frames to `--process-size` (default 512) before glazing

When resizing is applied, the cloaked or styled result is upscaled back to the original resolution before saving. Resize dimensions are rounded to the nearest multiple of 64, which is required by the VAE's strided convolutions.

### Device selection

The `--device` option is auto-detected on startup in priority order: `cuda` → `mps` → `cpu`.

**Do not use `--device cpu` on Apple Silicon.** The PyTorch CPU kernels for certain attention operations in the diffusion pipeline trigger an illegal instruction on ARM, causing a silent segfault — the process dies without printing a Python traceback, leaving no output file and only an OS-level semaphore leak warning:

```
UserWarning: resource_tracker: There appear to be 1 leaked semaphore objects
to clean up at shutdown: {'/mp-3xxsfog1'}
```

This warning is the only output you will see, which makes the failure appear mysterious. The fix is to use `--device mps`, which routes computation through Apple's Metal Performance Shaders. MPS is significantly faster than CPU for this workload and does not have the stability issues.

If you are not on Apple Silicon and have no CUDA GPU, `--device cpu` is safe on x86.

### LPIPS network

The perceptual constraint in the Glaze objective uses LPIPS (Learned Perceptual Image Patch Similarity) to measure how visible the perturbation is. The `lpips` library supports three backbone networks: AlexNet (`alex`), VGG-16 (`vgg`), and SqueezeNet (`squeeze`).

This implementation uses AlexNet (`alex`), which is the library's own default. The original LPIPS paper found AlexNet correlates *better* with human perceptual judgement than VGG despite being a much simpler model, and it runs 2–3× faster — a meaningful difference when the LPIPS loss is evaluated at every optimizer step across hundreds of video frames. VGG is available via `GlazeConfig(lpips_net="vgg")` if you have a specific reason to prefer it.

### Model ID history

The style transfer pipeline originally used `runwayml/stable-diffusion-v1-5` as the default model ID. This repository was taken down on HuggingFace. The current default is `stable-diffusion-v1-5/stable-diffusion-v1-5`, which is the canonical replacement. If you have stale cached weights at `~/.cache/huggingface/hub/models--runwayml*`, they can be safely deleted.

### Style transfer vs. identity target features

When running `glaze image` or `glaze video` without a style transfer function, the target features Φ(Ω(x,T)) fall back to encoding the source image itself. This means the optimizer is pushing the cloaked image's features away from the source in a random direction rather than toward a specific target style. It still produces a valid perturbation but is less principled than using actual style transfer.

For maximum effectiveness, provide a style reference image. The `style-transfer` command exists to let you verify that pipeline works before integrating it into a full glaze run.

---

## Architecture

```
src/glaze_video/
├── encoders/
│   ├── base.py          # FeatureEncoder Protocol (pluggable interface)
│   ├── clip_encoder.py  # CLIP ViT-L/14
│   └── vae_encoder.py   # Stable Diffusion VAE
├── style/
│   ├── transfer.py      # SD img2img + IP-Adapter style transfer Ω(x,T)
│   └── selector.py      # Target style selection (50–75th percentile distance)
├── glaze.py             # Core Adam optimization loop
├── video.py             # VideoGlazer with temporal optimizations
└── cli.py               # Click CLI entry points
```

### Temporal optimizations (video)

Applying the full Glaze optimization to every frame of a video naively would take hours per minute of footage. `VideoGlazer` implements two optimizations that make this practical:

**Temporal gating**: frames that are visually near-identical to the previous frame (within `--temporal-threshold`) skip the optimizer entirely and re-use the previous perturbation. This eliminates redundant work in static shots and slow pans.

**Warm start**: instead of initializing δ from zeros for each frame, the previous frame's perturbation is used as the starting point. Frames in a video are typically very similar, so the optimizer starts close to a good solution and converges in far fewer iterations.
