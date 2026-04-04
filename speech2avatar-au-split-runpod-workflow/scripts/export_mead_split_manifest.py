#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mead_root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stem_to_split: dict[str, str] = {}
    counts = {split: 0 for split in SPLITS}

    for split in SPLITS:
        split_root = args.mead_root / split
        if not split_root.exists():
            continue
        for video_path in sorted(split_root.rglob("*.mp4")):
            stem_to_split[video_path.stem] = split
            counts[split] += 1

    if not stem_to_split:
        raise RuntimeError(f"No MEAD videos found under {args.mead_root}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "mead_root": str(args.mead_root),
                "counts": counts,
                "stem_to_split": stem_to_split,
            },
            indent=2,
            sort_keys=True,
        )
    )
    print(f"wrote {args.output}")
    print(counts)


if __name__ == "__main__":
    main()
