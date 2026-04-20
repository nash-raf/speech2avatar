from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import librosa
import numpy as np
import torch
from tqdm import tqdm
from transformers import Wav2Vec2FeatureExtractor


def _motion_len(path: Path) -> int:
    obj = torch.load(path, map_location="cpu")
    if torch.is_tensor(obj):
        return int(obj.shape[0])
    if isinstance(obj, dict):
        for key in ("motion", "latents", "x", "z"):
            if key in obj and torch.is_tensor(obj[key]):
                return int(obj[key].shape[0])
    return int(len(obj))


def _load_stems(root: Path, split: str | None, limit: int | None) -> list[str]:
    if split:
        stems = [
            Path(line.strip()).stem
            for line in (root / split).read_text().splitlines()
            if line.strip()
        ]
    else:
        stems = sorted(p.stem for p in (root / "motion").glob("*.pt"))
    if limit is not None and limit > 0:
        stems = stems[:limit]
    return stems


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=str)
    parser.add_argument(
        "--imtalker_root",
        default="/workspace/IMTalker",
        type=str,
        help="Repo root containing generator/wav2vec2.py.",
    )
    parser.add_argument(
        "--wav2vec_model_path",
        default="/workspace/IMTalker/checkpoints/wav2vec2-base-960h",
        type=str,
    )
    parser.add_argument("--split", default=None, type=str)
    parser.add_argument("--limit", default=None, type=int)
    parser.add_argument("--out_subdir", default="audio_wav2vec", type=str)
    parser.add_argument("--meta_subdir", default="audio_wav2vec_meta", type=str)
    parser.add_argument("--sampling_rate", default=16000, type=int)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    imtalker_root = Path(args.imtalker_root)
    sys.path.insert(0, str(imtalker_root))
    from generator.wav2vec2 import Wav2VecModel

    root = Path(args.root)
    audio_dir = root / "audio_raw"
    motion_dir = root / "motion"
    out_dir = root / args.out_subdir
    meta_dir = root / args.meta_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    stems = _load_stems(root, args.split, args.limit)
    if not stems:
        raise RuntimeError("No stems found to process")

    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
        args.wav2vec_model_path,
        local_files_only=True,
    )
    model = Wav2VecModel.from_pretrained(
        args.wav2vec_model_path,
        local_files_only=True,
    ).to(args.device).eval()
    model.feature_extractor._freeze_parameters()

    ok = 0
    skipped = 0
    errors = 0

    with torch.no_grad():
        for stem in tqdm(stems, desc=f"extract wav2vec {args.split or 'all'}"):
            wav_path = audio_dir / f"{stem}.wav"
            motion_path = motion_dir / f"{stem}.pt"
            out_path = out_dir / f"{stem}.npy"
            meta_path = meta_dir / f"{stem}.json"

            if out_path.exists() and not args.overwrite:
                ok += 1
                continue
            if not wav_path.exists() or not motion_path.exists():
                skipped += 1
                continue

            try:
                target_len = _motion_len(motion_path)
                wav, _ = librosa.load(str(wav_path), sr=args.sampling_rate, mono=True)
                inputs = feature_extractor(
                    wav,
                    sampling_rate=args.sampling_rate,
                    return_tensors="pt",
                ).input_values.to(args.device)

                out = model(inputs, seq_len=target_len, output_hidden_states=False)
                feat = (
                    out.last_hidden_state.squeeze(0)
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32, copy=False)
                )
                if feat.shape != (target_len, 768):
                    raise RuntimeError(
                        f"Expected wav2vec shape {(target_len, 768)}, got {feat.shape}"
                    )

                np.save(out_path, feat)
                meta_path.write_text(
                    json.dumps(
                        {
                            "clip": stem,
                            "source_wav": str(wav_path),
                            "motion_path": str(motion_path),
                            "target_motion_len": target_len,
                            "sampling_rate": args.sampling_rate,
                            "wav2vec_model_path": args.wav2vec_model_path,
                            "feature_shape": list(feat.shape),
                            "mode": "normal_original_imtalker_teacher_wav2vec_full_clip",
                        },
                        indent=2,
                    )
                )
                ok += 1
            except Exception as exc:
                errors += 1
                print(f"[extract_wav2vec_teacher] error stem={stem}: {exc}")

    print(
        f"[extract_wav2vec_teacher] done ok={ok} skipped={skipped} "
        f"errors={errors} out={out_dir}"
    )
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
