#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
from typing import Iterable, List

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from tqdm import tqdm


def resolve_existing_path(candidates: Iterable[Path]) -> Path | None:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
LOCAL_ROOT = Path("/home/user/D")
RUNPOD_ROOT = Path("/workspace")

DEFAULT_MOSHI_REPO = resolve_existing_path(
    [
        RUNPOD_ROOT / "moshi",
        LOCAL_ROOT / "moshi",
    ]
)
DEFAULT_WAV_ROOT = resolve_existing_path(
    [
        RUNPOD_ROOT / "MEAD_wav",
        LOCAL_ROOT / "MEAD_wav",
    ]
)
DEFAULT_DATASET_ROOT = resolve_existing_path(
    [
        RUNPOD_ROOT / "generator_dataset",
        LOCAL_ROOT / "generator_dataset",
    ]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract Mimi continuous latents from 24 kHz wav files and save IMTalker-style npy features."
    )
    parser.add_argument(
        "--wav_root",
        type=Path,
        default=DEFAULT_WAV_ROOT,
        help="Root containing split folders like train/ and val/ with wav files.",
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=DEFAULT_DATASET_ROOT / "audio_mimi" if DEFAULT_DATASET_ROOT else None,
        help="Directory to write flat <stem>.npy feature files.",
    )
    parser.add_argument(
        "--motion_root",
        type=Path,
        default=DEFAULT_DATASET_ROOT / "motion" if DEFAULT_DATASET_ROOT else None,
        help="Optional motion root. If <motion_root>/<stem>.pt exists, audio features are interpolated to that exact length.",
    )
    parser.add_argument(
        "--moshi_repo",
        type=Path,
        default=DEFAULT_MOSHI_REPO,
        help="Path to the local Moshi repo.",
    )
    parser.add_argument(
        "--hf_repo",
        type=str,
        default="kyutai/moshiko-pytorch-bf16",
        help="Hugging Face repo used to load Mimi weights.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run Mimi on.",
    )
    parser.add_argument(
        "--target_fps",
        type=float,
        default=25.0,
        help="Fallback fps used to infer feature length when no motion file is found.",
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Use quantized-decoded Mimi latents instead of unquantized continuous latents.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing npy files.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val"],
        help="Split subfolders under wav_root to process.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit for quick testing.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.wav_root is None or not args.wav_root.exists():
        raise FileNotFoundError("Could not find --wav_root. Pass it explicitly.")
    if args.output_root is None:
        raise FileNotFoundError("Could not infer --output_root. Pass it explicitly.")
    if args.moshi_repo is None or not args.moshi_repo.exists():
        raise FileNotFoundError("Could not find --moshi_repo. Pass it explicitly.")


def import_moshi_loaders(moshi_repo: Path):
    moshi_pkg_root = moshi_repo / "moshi"
    if str(moshi_pkg_root) not in sys.path:
        sys.path.insert(0, str(moshi_pkg_root))
    from moshi.models import loaders  # type: ignore

    return loaders


def discover_wavs(wav_root: Path, splits: List[str], limit: int | None) -> List[Path]:
    wavs: List[Path] = []
    for split in splits:
        split_root = wav_root / split
        if not split_root.exists():
            print(f"[Warn] Split not found, skipping: {split_root}")
            continue
        wavs.extend(sorted(split_root.rglob("*.wav")))
    if limit is not None:
        wavs = wavs[:limit]
    return wavs


def load_wav_24k_mono(wav_path: Path) -> torch.Tensor:
    wav, sr = torchaudio.load(str(wav_path))
    if wav.numel() == 0:
        return torch.zeros(1, 1, 0, dtype=torch.float32)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != 24000:
        wav = torchaudio.functional.resample(wav, sr, 24000)
    return wav.unsqueeze(0).float()


def target_frames_for_stem(stem: str, motion_root: Path | None, wav_num_samples: int, target_fps: float) -> int:
    if motion_root is not None:
        motion_path = motion_root / f"{stem}.pt"
        if motion_path.exists():
            motion = torch.load(motion_path, map_location="cpu")
            return int(len(motion))
    return max(1, int(round(wav_num_samples * target_fps / 24000.0)))


def encode_mimi_features(mimi, wav: torch.Tensor, device: str, quantize: bool, target_frames: int) -> np.ndarray:
    wav = wav.to(device)
    with torch.inference_mode():
        latent = mimi.encode_to_latent(wav, quantize=quantize).float()
        # Motion length is the source of truth. Interpolate Mimi time steps to exactly match it.
        latent = F.interpolate(latent, size=target_frames, mode="linear", align_corners=True)
    feat = latent[0].transpose(0, 1).contiguous().cpu().numpy().astype(np.float32)
    return feat


def main() -> None:
    args = parse_args()
    validate_args(args)

    loaders = import_moshi_loaders(args.moshi_repo)

    args.output_root.mkdir(parents=True, exist_ok=True)

    wav_paths = discover_wavs(args.wav_root, args.splits, args.limit)
    if not wav_paths:
        raise RuntimeError(f"No wav files found under {args.wav_root} for splits {args.splits}.")

    print(f"[Info] Loading Mimi from {args.hf_repo} on {args.device}")
    checkpoint = loaders.CheckpointInfo.from_hf_repo(args.hf_repo)
    mimi = checkpoint.get_mimi(device=args.device)
    mimi.eval()

    print(f"[Info] Mimi sample_rate={mimi.sample_rate} frame_rate={mimi.frame_rate} frame_size={mimi.frame_size}")
    print(f"[Info] Found {len(wav_paths)} wav files")

    written = 0
    skipped = 0
    failed = 0

    for wav_path in tqdm(wav_paths, desc="Extracting Mimi audio features"):
        stem = wav_path.stem
        out_path = args.output_root / f"{stem}.npy"
        if out_path.exists() and not args.overwrite:
            skipped += 1
            continue

        try:
            wav = load_wav_24k_mono(wav_path)
            target_frames = target_frames_for_stem(
                stem=stem,
                motion_root=args.motion_root,
                wav_num_samples=wav.shape[-1],
                target_fps=args.target_fps,
            )
            feat = encode_mimi_features(
                mimi=mimi,
                wav=wav,
                device=args.device,
                quantize=args.quantize,
                target_frames=target_frames,
            )
            np.save(out_path, feat)
            written += 1
        except Exception as exc:
            failed += 1
            print(f"[Warn] Failed on {wav_path}: {exc}")

    print("[Done] Mimi feature extraction complete")
    print(f"[Done] written={written} skipped={skipped} failed={failed} output_root={args.output_root}")


if __name__ == "__main__":
    main()
