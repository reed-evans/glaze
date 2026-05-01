"""Convert an mp4 file into an animated gif."""

from __future__ import annotations

from pathlib import Path

import click
import cv2
from PIL import Image

from .video_io import read_frames


def mp4_to_gif(
    src: Path,
    dst: Path,
    fps: int = 15,
    width: int | None = None,
    loop: int = 0,
) -> None:
    """Convert ``src`` mp4 to an animated gif at ``dst``.

    Args:
        src: Source mp4 path.
        dst: Output gif path.
        fps: Target frames per second for the gif.
        width: If set, resize frames to this width while preserving aspect ratio.
        loop: 0 means loop forever, otherwise the number of loops.
    """
    cap = cv2.VideoCapture(str(src))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or float(fps)
    cap.release()

    frames = read_frames(src)

    if fps < src_fps:
        step = max(1, round(src_fps / fps))
        frames = frames[::step]

    if width is not None:
        resized: list[Image.Image] = []
        for frame in frames:
            img = Image.fromarray(frame)
            ratio = width / img.width
            new_size = (width, max(1, round(img.height * ratio)))
            resized.append(img.resize(new_size, Image.LANCZOS))
        pil_frames = resized
    else:
        pil_frames = [Image.fromarray(frame) for frame in frames]

    duration_ms = round(1000 / fps)
    dst.parent.mkdir(parents=True, exist_ok=True)
    pil_frames[0].save(
        dst,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration_ms,
        loop=loop,
        optimize=True,
        disposal=2,
    )


@click.command()
@click.argument("src", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("dst", type=click.Path(dir_okay=False, path_type=Path))
@click.option("--fps", default=15, show_default=True, help="GIF frame rate.")
@click.option("--width", type=int, default=None, help="Resize width in pixels.")
@click.option("--loop", default=0, show_default=True, help="Loop count (0 = forever).")
def main(src: Path, dst: Path, fps: int, width: int | None, loop: int) -> None:
    """Convert SRC mp4 into an animated DST gif."""
    mp4_to_gif(src, dst, fps=fps, width=width, loop=loop)
    click.echo(f"Wrote {dst}")


if __name__ == "__main__":
    main()
