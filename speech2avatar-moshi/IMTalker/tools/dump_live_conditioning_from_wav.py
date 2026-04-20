from __future__ import annotations

import json
from pathlib import Path

import numpy as np
try:
    import soundfile as sf  # type: ignore
except ImportError:
    sf = None  # noqa: N816
import torch
try:
    import torchaudio
except ImportError:
    torchaudio = None
import librosa

from launch_live import LaunchOptions
from live_pipeline import LiveMoshiIMTalkerSession


class DumpOptions(LaunchOptions):
    def initialize(self, parser):
        parser = super().initialize(parser)
        parser.add_argument("--wav_path", required=True, type=str)
        parser.add_argument("--dump_dir", required=True, type=str)
        parser.add_argument(
            "--warmup",
            action="store_true",
            help="Run session warmup before dumping live conditioning.",
        )
        return parser


def _load_audio_native(path: Path) -> tuple[np.ndarray, int]:
    if sf is not None:
        try:
            audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1, dtype=np.float32)
            return np.asarray(audio, dtype=np.float32), int(sr)
        except Exception:
            pass
    if torchaudio is not None:
        try:
            audio_t, sr = torchaudio.load(str(path))
            sr = int(sr)
            if audio_t.shape[0] > 1:
                audio_t = audio_t.mean(dim=0, keepdim=True)
            return audio_t.squeeze(0).float().cpu().numpy(), sr
        except Exception:
            pass
    audio, sr = librosa.load(str(path), sr=None, mono=True)
    return audio.astype(np.float32, copy=False), int(sr)


def _load_wav_mono_24k(path: Path, target_sr: int) -> torch.Tensor:
    audio, sr = _load_audio_native(path)
    wav = torch.from_numpy(np.asarray(audio, dtype=np.float32)).float()
    if sr != target_sr:
        if torchaudio is not None:
            wav = torchaudio.functional.resample(wav.unsqueeze(0), sr, target_sr).squeeze(0)
        else:
            wav = torch.from_numpy(
                librosa.resample(audio, orig_sr=sr, target_sr=target_sr, res_type="kaiser_fast").astype(np.float32)
            )
        sr = target_sr
    if sr != target_sr:
        raise RuntimeError(f"Expected sample rate {target_sr}, got {sr}")
    return wav.contiguous()


def main() -> None:
    opt = DumpOptions().parse()
    opt.rank = opt.device

    out_dir = Path(opt.dump_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    session = LiveMoshiIMTalkerSession(
        opt,
        generator_path=opt.generator_path,
        renderer_path=opt.renderer_path,
        ref_path=opt.ref_path,
        crop=opt.crop,
        nfe=opt.nfe,
        a_cfg_scale=opt.a_cfg_scale,
        moshi_repo=opt.moshi_repo,
        mimi_hf_repo=opt.hf_repo,
    )
    if opt.warmup:
        session.warmup()
    session.reset_reply()

    wav_path = Path(opt.wav_path)
    wav = _load_wav_mono_24k(wav_path, int(session.mimi.sample_rate))

    raw_chunks: list[np.ndarray] = []
    aligned_chunks: list[np.ndarray] = []
    chunk_meta: list[dict] = []

    cursor = 0
    chunk_index = 0
    while cursor < wav.numel():
        end = min(cursor + session.chunk_samples, wav.numel())
        original_num_samples = int(end - cursor)
        chunk = wav[cursor:end].clone()
        if chunk.numel() < session.chunk_samples:
            chunk = torch.nn.functional.pad(chunk, (0, session.chunk_samples - chunk.numel()))

        latents = session._encode_reply_pcm_to_latents(
            chunk,
            original_num_samples=original_num_samples,
        )
        target_frames = session._target_frames_for_pcm(original_num_samples)
        max_overlap_frames = max(session.render_window_frames - target_frames, 0)
        overlap_frames = min(session.overlap_frames, max_overlap_frames)

        latents_for_align = latents
        used_overlap_context = False
        if (
            session.overlap_context_latent_count > 0
            and session._overlap_latents is not None
            and session._overlap_latents.shape[0] >= session.overlap_context_latent_count
        ):
            latents_for_align = torch.cat([session._overlap_latents, latents], dim=0)
            used_overlap_context = True
        elif overlap_frames > 0:
            overlap_frames = 0

        aligned_chunk = session._align_raw_mimi_to_target_frames(
            latents_for_align,
            target_frames + overlap_frames,
        )

        if session.overlap_context_latent_count > 0:
            session._overlap_latents = latents[-session.overlap_context_latent_count :].clone()
        else:
            session._overlap_latents = None

        emitted_aligned = aligned_chunk[overlap_frames:].contiguous() if overlap_frames > 0 else aligned_chunk

        raw_np = latents.cpu().numpy().astype(np.float32, copy=False)
        aligned_np = emitted_aligned.cpu().numpy().astype(np.float32, copy=False)

        np.save(out_dir / f"chunk_{chunk_index:03d}_raw.npy", raw_np)
        np.save(out_dir / f"chunk_{chunk_index:03d}_aligned.npy", aligned_np)

        raw_chunks.append(raw_np)
        aligned_chunks.append(aligned_np)
        chunk_meta.append(
            {
                "chunk_index": chunk_index,
                "original_num_samples": original_num_samples,
                "raw_latent_shape": list(raw_np.shape),
                "target_frames": int(target_frames),
                "overlap_frames": int(overlap_frames),
                "used_overlap_context": bool(used_overlap_context),
                "aligned_emitted_shape": list(aligned_np.shape),
            }
        )

        cursor += session.chunk_samples
        chunk_index += 1

    raw_full = np.concatenate(raw_chunks, axis=0) if raw_chunks else np.zeros((0, opt.audio_feat_dim), dtype=np.float32)
    aligned_full = (
        np.concatenate(aligned_chunks, axis=0) if aligned_chunks else np.zeros((0, opt.audio_feat_dim), dtype=np.float32)
    )

    np.save(out_dir / "live_raw_concat.npy", raw_full)
    np.save(out_dir / "live_aligned_concat.npy", aligned_full)

    meta = {
        "wav_path": str(wav_path),
        "wav_num_samples": int(wav.numel()),
        "sample_rate": int(session.mimi.sample_rate),
        "fps": float(opt.fps),
        "chunk_sec": float(session.chunk_sec),
        "chunk_samples": int(session.chunk_samples),
        "render_window_frames": int(session.render_window_frames),
        "emit_step_frames": int(session.emit_step_frames),
        "overlap_frames": int(session.overlap_frames),
        "raw_concat_shape": list(raw_full.shape),
        "aligned_concat_shape": list(aligned_full.shape),
        "num_chunks": int(len(chunk_meta)),
        "chunks": chunk_meta,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    print("saved:", out_dir)
    print("live_raw_concat shape:", raw_full.shape)
    print("live_aligned_concat shape:", aligned_full.shape)


if __name__ == "__main__":
    main()
