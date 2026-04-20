#!/usr/bin/env python3
"""Create a labeled side-by-side comparison video from multiple mp4s."""

import argparse
import shlex
import subprocess
from pathlib import Path


def _escape_drawtext(text: str) -> str:
    return (
        text.replace("\\", r"\\")
        .replace(":", r"\:")
        .replace("'", r"\'")
        .replace("[", r"\[")
        .replace("]", r"\]")
        .replace(",", r"\,")
    )


def _build_filter(labels, height, font_size):
    chains = []
    inputs = []
    x_offset = "0"
    layout = []

    for idx, label in enumerate(labels):
        escaped = _escape_drawtext(label)
        chains.append(
            f"[{idx}:v]"
            f"scale=-2:{height}:force_original_aspect_ratio=decrease,"
            f"pad=ceil(iw/2)*2:{height}:(ow-iw)/2:(oh-ih)/2:black,"
            f"drawtext=text='{escaped}':x=20:y=20:fontsize={font_size}:"
            f"fontcolor=white:box=1:boxcolor=black@0.6:boxborderw=8"
            f"[v{idx}]"
        )
        inputs.append(f"[v{idx}]")
        layout.append(f"{x_offset}_0")
        x_offset += f"+w{idx}"
        if idx < len(labels) - 1:
            pass

    chains.append(
        "".join(inputs)
        + f"xstack=inputs={len(labels)}:layout={'|'.join(layout)}[vout]"
    )
    return ";".join(chains)


def main():
    parser = argparse.ArgumentParser(
        description="Create a labeled side-by-side comparison mp4."
    )
    parser.add_argument("videos", nargs="+", help="Input video paths.")
    parser.add_argument(
        "--labels",
        nargs="*",
        help="Optional labels matching the input videos. Defaults to video stems.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output mp4 path.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=512,
        help="Output tile height. Default: 512.",
    )
    parser.add_argument(
        "--font-size",
        type=int,
        default=28,
        help="Label font size. Default: 28.",
    )
    parser.add_argument(
        "--mute",
        action="store_true",
        help="Do not keep audio from the first input.",
    )
    args = parser.parse_args()

    if len(args.videos) < 2:
        raise SystemExit("Need at least two videos to compare.")

    video_paths = [Path(v).expanduser().resolve() for v in args.videos]
    missing = [str(v) for v in video_paths if not v.exists()]
    if missing:
        raise SystemExit(f"Missing input videos: {', '.join(missing)}")

    if args.labels and len(args.labels) != len(video_paths):
        raise SystemExit("--labels must match the number of videos.")

    labels = args.labels or [v.stem for v in video_paths]
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    cmd = ["ffmpeg", "-y"]
    for video in video_paths:
        cmd += ["-i", str(video)]

    cmd += ["-filter_complex", _build_filter(labels, args.height, args.font_size)]
    cmd += ["-map", "[vout]"]
    if not args.mute:
        cmd += ["-map", "0:a?"]
    cmd += [
        "-c:v",
        "libx264",
        "-crf",
        "18",
        "-preset",
        "medium",
        "-pix_fmt",
        "yuv420p",
    ]
    if not args.mute:
        cmd += ["-c:a", "aac", "-shortest"]
    cmd += [str(output)]

    print("Running:\n" + " ".join(shlex.quote(part) for part in cmd))
    subprocess.run(cmd, check=True)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
