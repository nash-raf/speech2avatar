#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path


SPLITS = ("train", "val", "test")
MODALITIES = {
    "motion": ".pt",
    "audio": ".npy",
    "smirk": ".pt",
    "gaze": ".npy",
    "au": ".npy",
}


def build_stem_index(mead_root: Path) -> dict[str, str]:
    stem_to_split: dict[str, str] = {}
    for split in SPLITS:
        split_root = mead_root / split
        if not split_root.exists():
            continue
        for video_path in sorted(split_root.rglob("*.mp4")):
            stem_to_split[video_path.stem] = split
    if not stem_to_split:
        raise RuntimeError(f"No MEAD videos found under {mead_root}")
    return stem_to_split


def load_stem_index_from_manifest(manifest_path: Path) -> dict[str, str]:
    payload = json.loads(manifest_path.read_text())
    stem_to_split = payload.get("stem_to_split", {})
    if not stem_to_split:
        raise RuntimeError(f"No split mapping found in manifest: {manifest_path}")
    return {str(stem): str(split) for stem, split in stem_to_split.items()}


def materialize(src: Path, dst: Path, mode: str) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()

    if mode == "symlink":
        dst.symlink_to(src)
        return
    if mode == "hardlink":
        os.link(src, dst)
        return
    if mode == "copy":
        shutil.copy2(src, dst)
        return
    raise ValueError(f"Unsupported mode: {mode}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--flat_root", required=True, type=Path, help="Flat generator dataset root")
    parser.add_argument("--mead_root", default=None, type=Path, help="MEAD root with train/val/test")
    parser.add_argument(
        "--split_manifest",
        default=None,
        type=Path,
        help="JSON manifest mapping file stem to split, used instead of scanning MEAD",
    )
    parser.add_argument("--output_root", required=True, type=Path, help="Split-preserving output dataset root")
    parser.add_argument(
        "--mode",
        choices=("symlink", "hardlink", "copy"),
        default="symlink",
        help="How to place files in the split dataset",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.split_manifest is not None:
        stem_to_split = load_stem_index_from_manifest(args.split_manifest)
    elif args.mead_root is not None:
        stem_to_split = build_stem_index(args.mead_root)
    else:
        raise RuntimeError("Provide either --mead_root or --split_manifest")

    for split in SPLITS:
        for modality in MODALITIES:
            (args.output_root / split / modality).mkdir(parents=True, exist_ok=True)

    summary: dict[str, dict[str, int]] = {
        split: {modality: 0 for modality in MODALITIES} for split in SPLITS
    }
    missing: list[tuple[str, str]] = []
    unknown: list[str] = []

    for modality, suffix in MODALITIES.items():
        src_dir = args.flat_root / modality
        if not src_dir.exists():
            raise RuntimeError(f"Missing modality directory: {src_dir}")

        for src_path in sorted(src_dir.glob(f"*{suffix}")):
            split = stem_to_split.get(src_path.stem)
            if split is None:
                unknown.append(src_path.stem)
                continue
            dst_path = args.output_root / split / modality / src_path.name
            materialize(src_path.resolve(), dst_path, args.mode)
            summary[split][modality] += 1

    expected_counts = {split: 0 for split in SPLITS}
    for split in stem_to_split.values():
        expected_counts[split] += 1

    for split in SPLITS:
        for modality in MODALITIES:
            if summary[split][modality] != expected_counts[split]:
                missing.append((split, modality))

    print("[SUMMARY]")
    for split in SPLITS:
        counts = " ".join(f"{modality}={summary[split][modality]}" for modality in MODALITIES)
        print(f"{split}: expected={expected_counts[split]} {counts}")

    if unknown:
        print(f"[WARN] {len(unknown)} files had no MEAD split mapping.")
    if missing:
        raise RuntimeError(f"Incomplete split dataset detected: {missing}")


if __name__ == "__main__":
    main()
