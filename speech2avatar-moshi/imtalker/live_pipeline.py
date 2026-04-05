from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import time
import librosa
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from scipy.io import wavfile

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
        self.chunk_sec = float(getattr(opt, "chunk_sec", 1.5))
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
        self.idle_frame = None
        self.prepare_reference(ref_path)

        self._render_stream = torch.cuda.Stream(self.device)
        self.mimi = self._load_mimi()
        self.mimi_frame_size = int(self.mimi.sample_rate / self.mimi.frame_rate)
        self.chunk_raw_mimi_frames = max(1, int(round(self.chunk_frames * self.mimi.frame_rate / opt.fps)))
        self.chunk_samples = max(self.mimi_frame_size, int(round(self.chunk_sec * self.mimi.sample_rate)))

        self.raw_pcm_buffer = np.zeros((0,), dtype=np.float32)
        self.raw_mimi_latent_buffer = torch.zeros((0, opt.audio_feat_dim), dtype=torch.float32)
        self.frame_callback: Callable[[RenderedChunk], None] | None = None
        self.video_callback: Callable[[int, str], None] | None = None
        self.segment_callback: Callable[[int, int, str], None] | None = None
        self.turn_done_callback: Callable[[int], None] | None = None

        self.fm_stream_state = None
        self.chunk_index = 0
        self.turn_index = 0
        self.current_turn_id: int | None = None
        self.current_segment_index = 0
        self._reply_pcm_buffer = torch.zeros((0,), dtype=torch.float32)
        self._render_queue: list[tuple[int, int, torch.Tensor]] = []
        self._render_queue_event = asyncio.Event()
        self._render_worker_task: asyncio.Task | None = None
        self._reply_active = False
        self._reply_done = False

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
        self.idle_frame = s_tensor[0].detach().cpu()
        self.f_r, self.g_r = self.renderer.dense_feature_encoder(s_tensor)
        self.g_r = self.g_r
        self.ref_x = self.renderer.latent_token_encoder(s_tensor)

    def _align_raw_mimi_to_target_frames(self, latents: torch.Tensor, target_frames: int) -> torch.Tensor:
        if latents.shape[0] == target_frames:
            return latents
        aligned = F.interpolate(
            latents.transpose(0, 1).unsqueeze(0),
            size=target_frames,
            mode="linear",
            align_corners=True,
        )[0].transpose(0, 1).contiguous()
        return aligned

    def _ensure_mimi_streaming(self) -> None:
        if getattr(self.mimi, "_streaming_state", None) is None:
            self.mimi.streaming_forever(1)
        self.mimi.reset_streaming()

    @torch.no_grad()
    def _encode_reply_pcm_to_latents(self, pcm: torch.Tensor) -> torch.Tensor:
        pcm = pcm.detach().cpu().float().reshape(-1)
        if pcm.numel() == 0:
            return torch.zeros((0, self.opt.audio_feat_dim), dtype=torch.float32)

        self._ensure_mimi_streaming()

        latents = []
        start = 0
        total = pcm.numel()
        while start < total:
            end = min(start + self.mimi_frame_size, total)
            frame = pcm[start:end]
            if frame.numel() < self.mimi_frame_size:
                frame = F.pad(frame, (0, self.mimi_frame_size - frame.numel()))
            wav = frame.unsqueeze(0).unsqueeze(0).to(self.device)
            latent = self.mimi.encode_to_latent(wav, quantize=False)[0].transpose(0, 1).contiguous().cpu()
            latents.append(latent)
            start = end

        return torch.cat(latents, dim=0)

    @torch.no_grad()
    def _decode_sample_to_frames(self, sample: torch.Tensor) -> torch.Tensor:
        total_frames = sample.shape[1]  # sample: [1, T, 32]
        batch_size = getattr(self, '_render_batch_size', 4)  # mini-batch to avoid OOM

        # Reference motion maps (computed once, same for every frame)
        ta_r = self.renderer.adapt(self.ref_x, self.g_r)          # [1, 32]
        m_r = self.renderer.latent_token_decoder(ta_r)             # tuple of 4 spatial maps

        # --- Batch adapt + latent_token_decoder for ALL frames at once (cheap MLPs) ---
        g_r_exp = self.g_r.expand(total_frames, -1)                # [T, 512]
        sample_flat = sample.squeeze(0)                            # [T, 32]
        ta_c_all = self.renderer.adapt(sample_flat, g_r_exp)       # [T, 32]
        m_c_all = self.renderer.latent_token_decoder(ta_c_all)     # tuple of 4: each [T, C, H, W]

        # --- Mini-batched decode (CrossAttention + SynthesisNetwork) ---
        all_frames = []
        for start in range(0, total_frames, batch_size):
            end = min(start + batch_size, total_frames)
            bs = end - start
            m_c_batch = tuple(m[start:end] for m in m_c_all)
            m_r_batch = tuple(m.expand(bs, -1, -1, -1) for m in m_r)
            f_r_batch = [f.expand(bs, -1, -1, -1) for f in self.f_r]
            out = self.renderer.decode(m_c_batch, m_r_batch, f_r_batch)
            all_frames.append(out)

        return torch.cat(all_frames, dim=0)  # [T, 3, H, W]

    @torch.no_grad()
    def _render_reply_chunk(self, pcm_chunk: torch.Tensor) -> RenderedChunk:
        t_chunk_start = time.perf_counter()

        # Run all GPU work on a dedicated stream to avoid conflicting with Moshi's CUDA Graphs
        with torch.cuda.stream(self._render_stream):
            t0 = time.perf_counter()
            latents = self._encode_reply_pcm_to_latents(pcm_chunk)
            t_mimi = time.perf_counter() - t0
            if latents.shape[0] == 0:
                raise RuntimeError("Cannot render empty reply chunk")

            target_frames = max(1, int(round(latents.shape[0] * self.opt.fps / self.mimi.frame_rate)))
            aligned_chunk = self._align_raw_mimi_to_target_frames(latents, target_frames)

            t0 = time.perf_counter()
            stream_state = self.fm_stream_state
            sample, next_state = self.fm.sample(
                {"ref_x": self.ref_x, "a_feat": aligned_chunk.to(self.device)},
                a_cfg_scale=self.a_cfg_scale,
                nfe=self.nfe,
                stream_state=stream_state,
                return_state=True,
            )
            self._render_stream.synchronize()
            t_generator = time.perf_counter() - t0
            self.fm_stream_state = next_state

            t0 = time.perf_counter()
            frames = self._decode_sample_to_frames(sample)
            self._render_stream.synchronize()
            t_renderer = time.perf_counter() - t0

        result = RenderedChunk(
            chunk_index=self.chunk_index,
            frames=frames.detach().cpu(),
            sample_latents=sample.detach().cpu(),
            conditioning_latents=aligned_chunk.detach().cpu(),
        )
        self.chunk_index += 1

        t_total = time.perf_counter() - t_chunk_start
        print(f"[TIMING] chunk {result.chunk_index:03d} | "
              f"mimi_encode={t_mimi:.3f}s  generator={t_generator:.3f}s  renderer={t_renderer:.3f}s  "
              f"total={t_total:.3f}s  ({target_frames} frames)")

        if self.frame_callback is not None:
            self.frame_callback(result)
        return result

    def _ensure_turn_started(self) -> int:
        if self._reply_active and self.current_turn_id is not None:
            return self.current_turn_id
        self.current_turn_id = self.turn_index
        self.turn_index += 1
        self.current_segment_index = 0
        self.fm_stream_state = None
        self._reply_pcm_buffer = torch.zeros((0,), dtype=torch.float32)
        self._reply_active = True
        self._reply_done = False
        return self.current_turn_id

    def _enqueue_ready_segments(self) -> None:
        if not self._reply_active or self.current_turn_id is None:
            return
        while self._reply_pcm_buffer.numel() >= self.chunk_samples:
            chunk = self._reply_pcm_buffer[: self.chunk_samples].clone()
            self._reply_pcm_buffer = self._reply_pcm_buffer[self.chunk_samples :]
            self._render_queue.append((self.current_turn_id, self.current_segment_index, chunk))
            self.current_segment_index += 1
        if self._render_queue:
            self._render_queue_event.set()

    async def _drain_render_queue(self) -> None:
        while True:
            if self._reply_done and not self._render_queue:
                break
            await self._render_queue_event.wait()
            while self._render_queue:
                turn_id, segment_index, pcm_chunk = self._render_queue.pop(0)

                def _render_and_save() -> str:
                    t_seg_start = time.perf_counter()
                    rendered = self._render_reply_chunk(pcm_chunk)
                    output_dir = tempfile.mkdtemp(prefix=f"imtalker_segment_{turn_id:04d}_")
                    output_path = os.path.join(output_dir, f"segment_{segment_index:04d}.mp4")
                    path = self.save_response_video(rendered.frames, pcm_chunk, output_path)
                    t_seg_total = time.perf_counter() - t_seg_start
                    pcm_sec = pcm_chunk.numel() / self.mimi.sample_rate
                    print(f"[TIMING] segment {turn_id:04d}/{segment_index:04d} TOTAL={t_seg_total:.3f}s  "
                          f"(audio={pcm_sec:.2f}s, ratio={t_seg_total/max(pcm_sec,0.01):.1f}x realtime)")
                    return path

                try:
                    output_path = await asyncio.to_thread(_render_and_save)
                except Exception as exc:
                    print(f"[IMTalker] failed to render segment {turn_id:04d}/{segment_index:04d}: {exc}")
                    continue
                if self.segment_callback is not None:
                    self.segment_callback(turn_id, segment_index, output_path)
            self._render_queue_event.clear()
            if self._reply_done and not self._render_queue:
                break
        self._render_worker_task = None
        if self._reply_active and self.current_turn_id is not None and self.turn_done_callback is not None:
            self.turn_done_callback(self.current_turn_id)
        self._reply_active = False
        self._reply_done = False
        self.current_turn_id = None
        self.current_segment_index = 0
        self.fm_stream_state = None
        self._reply_pcm_buffer = torch.zeros((0,), dtype=torch.float32)

    def _ensure_render_worker(self) -> None:
        if self._render_worker_task is None or self._render_worker_task.done():
            self._render_worker_task = asyncio.create_task(self._drain_render_queue())

    @torch.no_grad()
    def push_mimi_latents(self, latents: torch.Tensor | np.ndarray, *, aligned: bool = False) -> list[RenderedChunk]:
        del latents, aligned
        return []

    @torch.no_grad()
    def push_mimi_codes(self, codes: torch.Tensor) -> list[RenderedChunk]:
        del codes
        return []

    @torch.no_grad()
    def push_pcm_chunk(self, pcm: torch.Tensor | np.ndarray, sample_rate: int | None = None) -> list[RenderedChunk]:
        self._ensure_mimi_streaming()
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
        return []

    async def handle_moshi_output(self, tokens: torch.Tensor, pcm: torch.Tensor, latents: torch.Tensor) -> list[RenderedChunk]:
        """Buffer Moshi reply PCM and progressively publish AV segments."""
        del tokens, latents
        turn_id = self._ensure_turn_started()
        flat_pcm = pcm.detach().cpu().float().reshape(-1)
        self._reply_pcm_buffer = torch.cat([self._reply_pcm_buffer, flat_pcm], dim=0)
        buffered_sec = self._reply_pcm_buffer.numel() / self.mimi.sample_rate
        queue_before = len(self._render_queue)
        self._enqueue_ready_segments()
        queue_after = len(self._render_queue)
        if queue_after > queue_before:
            print(f"[TIMING] moshi_output turn={turn_id} | +{flat_pcm.numel()} samples | "
                  f"buffer={buffered_sec:.2f}s | enqueued {queue_after-queue_before} segment(s) | "
                  f"render_queue={queue_after}")
        if self._render_queue:
            self._ensure_render_worker()
        return []

    async def handle_user_audio(self, latents: torch.Tensor) -> list[RenderedChunk]:
        del latents
        return []

    async def finalize_pending_reply(self) -> str | None:
        """Flush the final partial reply chunk and wait for segment publication."""
        if not self._reply_active:
            return None

        if self.current_turn_id is None:
            return None

        if self._reply_pcm_buffer.numel() > 0:
            chunk = self._reply_pcm_buffer.clone()
            self._reply_pcm_buffer = torch.zeros((0,), dtype=torch.float32)
            self._render_queue.append((self.current_turn_id, self.current_segment_index, chunk))
            self.current_segment_index += 1
            self._render_queue_event.set()

        self._reply_done = True
        if self._render_queue:
            self._ensure_render_worker()
        if self._render_worker_task is not None:
            self._render_queue_event.set()
            await self._render_worker_task
        else:
            if self.turn_done_callback is not None:
                self.turn_done_callback(self.current_turn_id)
            self._reply_active = False
            self._reply_done = False
            self.current_turn_id = None
            self.current_segment_index = 0
            self.fm_stream_state = None
            self._reply_pcm_buffer = torch.zeros((0,), dtype=torch.float32)
        return None

    def save_idle_video(self, output_path: str, duration_sec: float = 2.0) -> str:
        if self.idle_frame is None:
            raise RuntimeError("Reference image has not been prepared")
        num_frames = max(1, int(round(duration_sec * self.opt.fps)))
        frames = self.idle_frame.unsqueeze(0).repeat(num_frames, 1, 1, 1)
        frames_hwc = frames.permute(0, 2, 3, 1).detach().clamp(-1, 1)
        frames_hwc = (frames_hwc * 255).byte()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        torchvision.io.write_video(output_path, frames_hwc, fps=self.opt.fps)
        return output_path

    def save_response_video(self, frames: torch.Tensor, pcm: torch.Tensor, output_path: str) -> str:
        t_save_start = time.perf_counter()

        frames_hwc = frames.permute(0, 2, 3, 1).detach().clamp(-1, 1)
        frames_hwc = (frames_hwc * 255).byte().numpy()
        n, h, w, c = frames_hwc.shape
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        t0 = time.perf_counter()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
            tmp_wav_path = tmp_wav.name
        wavfile.write(tmp_wav_path, int(self.mimi.sample_rate), pcm.detach().cpu().float().numpy())
        t_write_wav = time.perf_counter() - t0

        # Pipe raw RGB frames directly to ffmpeg — avoids torchvision.io.write_video overhead
        t0 = time.perf_counter()
        cmd = [
            "ffmpeg", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{w}x{h}", "-r", str(self.opt.fps),
            "-i", "pipe:0",
            "-i", tmp_wav_path,
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "18",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-shortest",
            output_path,
        ]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        # Use communicate() to avoid pipe deadlock — it handles large writes safely
        raw_bytes = frames_hwc.tobytes()
        _, stderr = proc.communicate(input=raw_bytes)
        t_ffmpeg = time.perf_counter() - t0

        if os.path.exists(tmp_wav_path):
            os.remove(tmp_wav_path)
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg pipe mux failed: {stderr.decode().strip()}")

        t_total = time.perf_counter() - t_save_start
        print(f"[TIMING] save_video | write_wav={t_write_wav:.3f}s  ffmpeg_pipe={t_ffmpeg:.3f}s  total={t_total:.3f}s")
        return output_path
