from __future__ import annotations

import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import librosa
import numpy as np
import torch
import torch.nn.functional as F
import torchvision

from generator.FM import FMGenerator
from generator.generate import DataProcessor
from renderer.models import IMTRenderer


def _default_moshi_repo() -> Path:
    return Path(__file__).resolve().parents[1] / "moshi"


def _ensure_moshi_importable(moshi_repo: str | None = None) -> None:
    repo = Path(moshi_repo) if moshi_repo else _default_moshi_repo()
    pkg_root = repo / "moshi"
    if pkg_root.exists() and str(pkg_root) not in sys.path:
        sys.path.insert(0, str(pkg_root))


@dataclass
class RenderedChunk:
    chunk_index: int
    frames: torch.Tensor
    sample_latents: torch.Tensor
    conditioning_latents: torch.Tensor


class LiveMoshiIMTalkerSession:
    def __init__(
        self,
        opt,
        generator_path: str,
        renderer_path: str,
        ref_path: str,
        *,
        crop: bool = True,
        nfe: int = 2,
        a_cfg_scale: float = 1.0,
        moshi_repo: str | None = None,
        mimi_hf_repo: str = "kyutai/moshiko-pytorch-bf16",
    ):
        self.opt = opt
        self.device = torch.device(getattr(opt, "rank", "cuda"))
        self.nfe = nfe
        self.a_cfg_scale = a_cfg_scale
        self.crop = crop
        self.chunk_frames = int(round(opt.wav2vec_sec * opt.fps))
        self.moshi_repo = moshi_repo
        self.mimi_hf_repo = mimi_hf_repo

        self.fm = FMGenerator(opt).to(self.device).eval()
        self.renderer = IMTRenderer(opt).to(self.device).eval()
        self.data_processor = DataProcessor(opt)

        self._load_generator_weights(generator_path)
        self._load_renderer_weights(renderer_path)

        self.ref_x = None
        self.f_r = None
        self.g_r = None
        self.prepare_reference(ref_path)

        self.mimi = self._load_mimi()
        self.mimi.streaming_forever(1)
        self.mimi.reset_streaming()
        self.mimi_frame_size = int(self.mimi.sample_rate / self.mimi.frame_rate)
        self.chunk_raw_mimi_frames = max(1, int(round(self.chunk_frames * self.mimi.frame_rate / opt.fps)))

        self.raw_pcm_buffer = np.zeros((0,), dtype=np.float32)
        self.raw_mimi_latent_buffer = torch.zeros((0, opt.audio_feat_dim), dtype=torch.float32)
        self.fm_stream_state = None
        self.chunk_index = 0
        self.frame_callback: Callable[[RenderedChunk], None] | None = None

    def _load_mimi(self):
        _ensure_moshi_importable(self.moshi_repo)
        from moshi.models import loaders

        checkpoint = loaders.CheckpointInfo.from_hf_repo(self.mimi_hf_repo)
        mimi = checkpoint.get_mimi(device=str(self.device))
        mimi.eval()
        return mimi

    def _load_generator_weights(self, checkpoint_path: str, prefix: str = "model.") -> None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)
        if "model" in state_dict:
            state_dict = state_dict["model"]

        stripped_state_dict = {
            k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)
        }
        model_state_dict = self.fm.state_dict()
        loadable = {
            k: v
            for k, v in stripped_state_dict.items()
            if k in model_state_dict and model_state_dict[k].shape == v.shape
        }
        self.fm.load_state_dict(loadable, strict=False)

    def _load_renderer_weights(self, checkpoint_path: str) -> None:
        renderer_ckpt = torch.load(checkpoint_path, map_location="cpu")["state_dict"]
        ae_state_dict = {k.replace("gen.", ""): v for k, v in renderer_ckpt.items() if k.startswith("gen.")}
        self.renderer.load_state_dict(ae_state_dict, strict=False)

    @torch.no_grad()
    def prepare_reference(self, ref_path: str) -> None:
        image = self.data_processor.default_img_loader(ref_path)
        if self.crop:
            image = self.data_processor.process_img(image)
        s_tensor = self.data_processor.transform(image).unsqueeze(0).to(self.device)
        self.f_r, self.g_r = self.renderer.dense_feature_encoder(s_tensor)
        self.ref_x = self.renderer.latent_token_encoder(s_tensor)

    def _align_raw_mimi_to_imtalker_fps(self, latents: torch.Tensor) -> torch.Tensor:
        if latents.shape[0] == self.chunk_frames:
            return latents
        aligned = F.interpolate(
            latents.transpose(0, 1).unsqueeze(0),
            size=self.chunk_frames,
            mode="linear",
            align_corners=True,
        )[0].transpose(0, 1).contiguous()
        return aligned

    @torch.no_grad()
    def _decode_sample_to_frames(self, sample: torch.Tensor) -> torch.Tensor:
        total_frames = sample.shape[1]
        ta_r = self.renderer.adapt(self.ref_x, self.g_r)
        m_r = self.renderer.latent_token_decoder(ta_r)
        frames = []
        for frame_idx in range(total_frames):
            ta_c = self.renderer.adapt(sample[:, frame_idx, ...], self.g_r)
            m_c = self.renderer.latent_token_decoder(ta_c)
            frames.append(self.renderer.decode(m_c, m_r, self.f_r))
        return torch.stack(frames, dim=1).squeeze(0)

    @torch.no_grad()
    def _render_next_ready_chunks(self) -> list[RenderedChunk]:
        rendered = []
        while self.raw_mimi_latent_buffer.shape[0] >= self.chunk_raw_mimi_frames:
            raw_chunk = self.raw_mimi_latent_buffer[: self.chunk_raw_mimi_frames]
            self.raw_mimi_latent_buffer = self.raw_mimi_latent_buffer[self.chunk_raw_mimi_frames :]
            aligned_chunk = self._align_raw_mimi_to_imtalker_fps(raw_chunk)
            sample, self.fm_stream_state = self.fm.sample(
                {"ref_x": self.ref_x, "a_feat": aligned_chunk.to(self.device)},
                a_cfg_scale=self.a_cfg_scale,
                nfe=self.nfe,
                stream_state=self.fm_stream_state,
                return_state=True,
            )
            frames = self._decode_sample_to_frames(sample)
            result = RenderedChunk(
                chunk_index=self.chunk_index,
                frames=frames.detach().cpu(),
                sample_latents=sample.detach().cpu(),
                conditioning_latents=aligned_chunk.detach().cpu(),
            )
            self.chunk_index += 1
            rendered.append(result)
            if self.frame_callback is not None:
                self.frame_callback(result)
        return rendered

    @torch.no_grad()
    def push_mimi_latents(self, latents: torch.Tensor | np.ndarray, *, aligned: bool = False) -> list[RenderedChunk]:
        if isinstance(latents, np.ndarray):
            latents = torch.from_numpy(latents)
        latents = latents.detach().cpu().float()
        if latents.ndim == 3:
            latents = latents[0].transpose(0, 1).contiguous()
        if aligned:
            aligned_chunk = latents
            sample, self.fm_stream_state = self.fm.sample(
                {"ref_x": self.ref_x, "a_feat": aligned_chunk.to(self.device)},
                a_cfg_scale=self.a_cfg_scale,
                nfe=self.nfe,
                stream_state=self.fm_stream_state,
                return_state=True,
            )
            frames = self._decode_sample_to_frames(sample)
            result = RenderedChunk(
                chunk_index=self.chunk_index,
                frames=frames.detach().cpu(),
                sample_latents=sample.detach().cpu(),
                conditioning_latents=aligned_chunk.detach().cpu(),
            )
            self.chunk_index += 1
            if self.frame_callback is not None:
                self.frame_callback(result)
            return [result]

        self.raw_mimi_latent_buffer = torch.cat([self.raw_mimi_latent_buffer, latents], dim=0)
        return self._render_next_ready_chunks()

    @torch.no_grad()
    def push_mimi_codes(self, codes: torch.Tensor) -> list[RenderedChunk]:
        if codes.ndim == 2:
            codes = codes.unsqueeze(0)
        codes = codes.to(self.device)
        latents = self.mimi.decode_latent(codes)[0].transpose(0, 1).contiguous()
        return self.push_mimi_latents(latents, aligned=False)

    @torch.no_grad()
    def push_pcm_chunk(self, pcm: torch.Tensor | np.ndarray, sample_rate: int | None = None) -> list[RenderedChunk]:
        if isinstance(pcm, torch.Tensor):
            pcm_np = pcm.detach().cpu().float().numpy()
        else:
            pcm_np = np.asarray(pcm, dtype=np.float32)
        pcm_np = np.squeeze(pcm_np)
        if pcm_np.ndim != 1:
            raise ValueError(f"Expected mono PCM, got shape {pcm_np.shape}")

        if sample_rate is not None and sample_rate != self.mimi.sample_rate:
            pcm_np = librosa.resample(pcm_np, orig_sr=sample_rate, target_sr=self.mimi.sample_rate)

        self.raw_pcm_buffer = np.concatenate([self.raw_pcm_buffer, pcm_np.astype(np.float32)])
        latents = []
        while self.raw_pcm_buffer.shape[0] >= self.mimi_frame_size:
            frame = self.raw_pcm_buffer[: self.mimi_frame_size]
            self.raw_pcm_buffer = self.raw_pcm_buffer[self.mimi_frame_size :]
            wav = torch.from_numpy(frame).unsqueeze(0).unsqueeze(0).float().to(self.device)
            latent = self.mimi.encode_to_latent(wav, quantize=False)[0].transpose(0, 1).contiguous().cpu()
            latents.append(latent)
        if not latents:
            return []
        return self.push_mimi_latents(torch.cat(latents, dim=0), aligned=False)

    async def handle_moshi_output(self, tokens: torch.Tensor, pcm: torch.Tensor, latents: torch.Tensor) -> list[RenderedChunk]:
        del tokens
        return self.push_mimi_latents(latents[0].transpose(0, 1).contiguous(), aligned=False)

    async def handle_user_audio(self, latents: torch.Tensor) -> list[RenderedChunk]:
        """While the user is speaking, render a neutral/listening face by feeding silence.
        latents: [B, D, T] from mimi.decode_latent — we use its shape but replace with zeros."""
        silence = torch.zeros_like(latents[0]).transpose(0, 1).contiguous()  # [T, D] zeros
        return self.push_mimi_latents(silence, aligned=False)

    def save_chunk_video(self, rendered_chunk: RenderedChunk, output_path: str, audio_path: str | None = None) -> str:
        frames = rendered_chunk.frames.permute(0, 2, 3, 1).detach().clamp(-1, 1)
        frames = (frames * 255).byte()
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp_path = tmp.name
        torchvision.io.write_video(tmp_path, frames, fps=self.opt.fps)
        if audio_path:
            os.system(f'ffmpeg -loglevel quiet -y -i "{tmp_path}" -i "{audio_path}" -c:v copy -c:a aac "{output_path}"')
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        else:
            os.replace(tmp_path, output_path)
        return output_path
