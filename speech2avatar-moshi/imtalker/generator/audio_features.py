import os
import sys
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn.functional as F
from transformers import Wav2Vec2FeatureExtractor


DEFAULT_MIMI_HF_REPO = "kyutai/moshiko-pytorch-bf16"


def _default_moshi_repo() -> Path:
    return Path(__file__).resolve().parents[2] / "moshi"


def _ensure_moshi_importable(moshi_repo: str | None = None) -> None:
    repo = Path(moshi_repo) if moshi_repo else _default_moshi_repo()
    pkg_root = repo / "moshi"
    if pkg_root.exists() and str(pkg_root) not in sys.path:
        sys.path.insert(0, str(pkg_root))


def _get_mimi_model(device: torch.device | str, cache: dict | None, moshi_repo: str | None, hf_repo: str):
    if cache is not None and "mimi_model" in cache:
        return cache["mimi_model"]

    _ensure_moshi_importable(moshi_repo)
    from moshi.models import loaders

    checkpoint = loaders.CheckpointInfo.from_hf_repo(hf_repo)
    mimi = checkpoint.get_mimi(device=str(device))
    mimi.eval()

    if cache is not None:
        cache["mimi_model"] = mimi
    return mimi


def load_audio_conditioning(opt, audio_path: str | None, audio_feat_path: str | None, device, cache: dict | None = None):
    if audio_feat_path and os.path.exists(audio_feat_path):
        a_feat = torch.from_numpy(np.load(audio_feat_path)).float()
        return {"a_feat": a_feat}

    if not audio_path or not os.path.exists(audio_path):
        raise FileNotFoundError(f"Missing audio input: {audio_path}")

    audio_feat_dim = getattr(opt, "audio_feat_dim", 768)
    if audio_feat_dim == 512:
        mimi = _get_mimi_model(
            device=device,
            cache=cache,
            moshi_repo=getattr(opt, "moshi_repo", None),
            hf_repo=getattr(opt, "mimi_hf_repo", DEFAULT_MIMI_HF_REPO),
        )
        speech_array, _ = librosa.load(audio_path, sr=24000, mono=True)
        wav = torch.from_numpy(speech_array).unsqueeze(0).unsqueeze(0).float().to(device)
        target_frames = max(1, int(round(wav.shape[-1] * opt.fps / 24000.0)))
        with torch.inference_mode():
            latent = mimi.encode_to_latent(wav, quantize=False).float()
            latent = F.interpolate(latent, size=target_frames, mode="linear", align_corners=True)
        a_feat = latent[0].transpose(0, 1).contiguous().cpu()
        return {"a_feat": a_feat}

    if cache is not None and "wav2vec_preprocessor" in cache:
        preprocessor = cache["wav2vec_preprocessor"]
    else:
        preprocessor = Wav2Vec2FeatureExtractor.from_pretrained(
            opt.wav2vec_model_path, local_files_only=True
        )
        if cache is not None:
            cache["wav2vec_preprocessor"] = preprocessor

    speech_array, sampling_rate = librosa.load(audio_path, sr=opt.sampling_rate)
    raw = preprocessor(
        speech_array,
        sampling_rate=sampling_rate,
        return_tensors="pt"
    ).input_values[0]
    return {"a": raw.unsqueeze(0).to(device)}
