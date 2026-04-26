"""Aggregate style-mimicry metrics across all generated videos.

Expects:
    references_dir/ {clip}.mp4                               — original artist clips
    outputs_dir/    {clip}_{generator}_{version}.mp4         — generator outputs
                                                              (version ∈ original, glazed)

For each (clip, generator) pair it compares both the original- and glazed-derived
generation back against the reference clip, writes one CSV row per pair with the
metrics and their delta, and prints a summary. Delta columns are signed so you
can see at a glance whether glazing reduced similarity to the original style:

    clip_sim_delta = clip_sim_orig - clip_sim_glazed    (positive → glaze worked)
    lpips_delta    = lpips_glazed  - lpips_orig         (positive → glaze worked)
    style_delta    = style_glazed  - style_orig         (positive → glaze worked)

Usage:
    python -m runner.aggregate --references clips/original --outputs outputs/ --csv results.csv
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from runner.metrics import Evaluator, VideoMetrics

FILENAME_RE = re.compile(r"^(?P<clip>.+)_(?P<generator>[^_]+)_(?P<version>original|glazed)\.mp4$")


@dataclass(frozen=True)
class GeneratedFile:
    path: Path
    clip: str
    generator: str
    version: str


def discover_outputs(outputs_dir: Path) -> list[GeneratedFile]:
    if not outputs_dir.is_dir():
        raise FileNotFoundError(f"Not a directory: {outputs_dir}")
    found: list[GeneratedFile] = []
    for p in sorted(outputs_dir.glob("*.mp4")):
        m = FILENAME_RE.match(p.name)
        if not m:
            print(f"[skip] filename does not match pattern: {p.name}", file=sys.stderr)
            continue
        found.append(
            GeneratedFile(
                path=p,
                clip=m.group("clip"),
                generator=m.group("generator"),
                version=m.group("version"),
            )
        )
    return found


def _row(clip: str, generator: str, orig: VideoMetrics, glazed: VideoMetrics) -> list[object]:
    return [
        clip,
        generator,
        f"{orig.clip_similarity:.4f}",
        f"{glazed.clip_similarity:.4f}",
        f"{orig.clip_similarity - glazed.clip_similarity:+.4f}",
        f"{orig.lpips:.4f}",
        f"{glazed.lpips:.4f}",
        f"{glazed.lpips - orig.lpips:+.4f}",
        f"{orig.style_loss:.4f}",
        f"{glazed.style_loss:.4f}",
        f"{glazed.style_loss - orig.style_loss:+.4f}",
        orig.num_frames_compared,
    ]


HEADER = [
    "clip",
    "generator",
    "clip_sim_orig",
    "clip_sim_glazed",
    "clip_sim_delta",
    "lpips_orig",
    "lpips_glazed",
    "lpips_delta",
    "style_orig",
    "style_glazed",
    "style_delta",
    "frames_compared",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate style-mimicry metrics into a CSV.")
    parser.add_argument("--references", type=Path, required=True, help="Directory of original reference clips (named {clip}.mp4).")
    parser.add_argument("--outputs", type=Path, required=True, help="Directory of generated videos.")
    parser.add_argument("--csv", type=Path, required=True, help="Output CSV path.")
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--frame-size", type=int, default=224)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()

    files = discover_outputs(args.outputs)
    if not files:
        raise SystemExit(f"No matching files found in {args.outputs}")

    evaluator = Evaluator(device=args.device, frame_size=args.frame_size, num_frames=args.num_frames)

    pairs: dict[tuple[str, str], dict[str, VideoMetrics]] = {}
    for f in files:
        ref = args.references / f"{f.clip}.mp4"
        if not ref.exists():
            print(f"[skip] reference missing: {ref}", file=sys.stderr)
            continue
        metrics = evaluator.compare(ref, f.path)
        pairs.setdefault((f.clip, f.generator), {})[f.version] = metrics
        print(
            f"[metric] {f.path.name}: "
            f"clip={metrics.clip_similarity:.4f}  "
            f"lpips={metrics.lpips:.4f}  "
            f"style={metrics.style_loss:.4f}"
        )

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    rows_written = 0
    with args.csv.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(HEADER)
        for (clip, generator), versions in sorted(pairs.items()):
            orig = versions.get("original")
            glazed = versions.get("glazed")
            if orig is None or glazed is None:
                print(
                    f"[warn] incomplete pair for {clip}/{generator}: have {sorted(versions)}",
                    file=sys.stderr,
                )
                continue
            writer.writerow(_row(clip, generator, orig, glazed))
            rows_written += 1

    print(f"wrote {rows_written} rows → {args.csv}")


if __name__ == "__main__":
    main()
