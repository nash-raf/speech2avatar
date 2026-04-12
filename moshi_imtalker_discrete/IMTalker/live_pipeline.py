from __future__ import annotations

import asyncio
import os
import queue as _queue_mod
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import cv2
import face_alignment
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from generator.FM import FMGenerator
from renderer.models import IMTRenderer


def _default_moshi_repo() -> Path:
    return Path(__file__).resolve().parent.parent / "moshi"


def _maybe_torch_compile_renderer(ae: IMTRenderer) -> None:
    """Opt-in renderer compilation via IMTALKER_TORCH_COMPILE env var.

    Values:
        frame_decoder | 1 | true   - compile SynthesisNetwork only (CUDA graphs)
        decode_default             - compile full decode() with mode="default"
        0 | false | off | ""      - disabled
    """
    raw = os.environ.get("IMTALKER_TORCH_COMPILE", "").strip().lower()
    if not raw or raw in ("0", "false", "no", "off"):
        return
    if not hasattr(torch, "compile"):
        print("[IMTalker] torch.compile not available; skipping renderer compile.")
        return

    target = "frame_decoder" if raw in ("1", "true", "yes", "on") else raw
    if target not in ("frame_decoder", "decode_default"):
        print(
            "[IMTalker] IMTALKER_TORCH_COMPILE must be frame_decoder, "
            "decode_default, or 1; got %r; skip." % (raw,)
        )
        return
    try:
        if target == "decode_default":
            ae.decode = torch.compile(ae.decode, mode="default", fullgraph=False)
            print("[IMTalker] torch.compile: IMTRenderer.decode, mode=default (no CUDA graphs).")
        else:
            ae.frame_decoder = torch.compile(
                ae.frame_decoder, mode="reduce-overhead", fullgraph=False
            )
            print("[IMTalker] torch.compile: SynthesisNetwork (frame_decoder), mode=reduce-overhead.")
    except Exception as exc:
        print(f"[IMTalker] torch.compile failed ({exc}); using eager renderer.")


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


@dataclass
class PendingRenderChunk:
    chunk_index: int
    reply_generation: int
    pcm: torch.Tensor
    valid_samples: int


class ReferenceDataProcessor:
    def __init__(self):
        self.fa = face_alignment.FaceAlignment(
            face_alignment.LandmarksType.TWO_D, flip_input=False
        )
        self.transform = transforms.Compose(
            [
                transforms.Resize((512, 512)),
                transforms.ToTensor(),
            ]
        )

    @torch.no_grad()
    def process_img(self, img: Image.Image) -> Image.Image:
        img_arr = np.array(img)
        h, w = img_arr.shape[:2]
        bboxes = self.fa.face_detector.detect_from_image(img_arr)
        valid_bboxes = [
            (int(x1), int(y1), int(x2), int(y2), score)
            for (x1, y1, x2, y2, score) in bboxes
            if score > 0.95
        ]
        if not valid_bboxes:
            raise ValueError("No face detected in the reference image.")

        x1, y1, x2, y2, _ = valid_bboxes[0]
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        half_w = int((x2 - x1) * 0.8)
        half_h = int((y2 - y1) * 0.8)
        half = max(half_w, half_h)

        x1_new = max(cx - half, 0)
        x2_new = min(cx + half, w)
        y1_new = max(cy - half, 0)
        y2_new = min(cy + half, h)

        side = min(x2_new - x1_new, y2_new - y1_new)
        x2_new = x1_new + side
        y2_new = y1_new + side

        crop_img = img_arr[y1_new:y2_new, x1_new:x2_new]
        return Image.fromarray(crop_img)

    def default_img_loader(self, path: str) -> Image.Image:
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return Image.fromarray(img)


class LiveMoshiIMTalkerSession:
    """Progressively render Moshi reply PCM with the discrete IMTalker fork.

    Pipeline: reply PCM -> Mimi streaming encode -> codebook 0 token IDs ->
    MoshiTokenEncoder embed+interp -> FM.sample(a_feat=...) -> renderer.
    """

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
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")
        self.nfe = nfe
        self.a_cfg_scale = a_cfg_scale
        self.crop = crop
        self.debug_session = bool(getattr(opt, "debug_session", False))
        self.chunk_frames = int(round(opt.wav2vec_sec * opt.fps))
        self.chunk_sec = float(getattr(opt, "chunk_sec", 1.5))
        self.render_workers = max(1, int(getattr(opt, "render_workers", 2)))
        self.moshi_repo = moshi_repo
        self.mimi_hf_repo = mimi_hf_repo

        self.fm = FMGenerator(opt).to(self.device).eval()
        self.renderer = IMTRenderer(opt).to(self.device).eval()
        self.data_processor = ReferenceDataProcessor()

        self._load_generator_weights(generator_path)
        self._load_renderer_weights(renderer_path)
        _maybe_torch_compile_renderer(self.renderer)

        self.ref_x = None
        self.f_r = None
        self.g_r = None
        self.idle_frame = None
        self.prepare_reference(ref_path)

        self.mimi = self._load_mimi()
        self.mimi_frame_size = int(self.mimi.sample_rate / self.mimi.frame_rate)
        self.render_window_frames = self.chunk_frames
        self.chunk_samples = max(
            self.mimi_frame_size, int(round(self.chunk_sec * self.mimi.sample_rate))
        )
        self.emit_step_frames = max(1, int(round(self.chunk_samples * self.opt.fps / self.mimi.sample_rate)))
        self.overlap_frames = max(0, self.render_window_frames - self.emit_step_frames)
        self.overlap_token_count = int(
            round(self.overlap_frames * self.mimi.frame_rate / self.opt.fps)
        )
        self._av_audio_sample_rate = 48000
        self._av_samples_per_frame = int(round(self._av_audio_sample_rate / self.opt.fps))

        # av_session is set externally by the fMP4 HTTP handler when a viewer
        # connects. Render output is pushed there as (frames_uint8_hwc, pcm48_int16);
        # while None, render work is skipped (no point burning GPU with no viewer).
        self.av_session = None  # type: ignore[assignment]

        self.fm_stream_state = None
        self._frame_residual = 0.0
        self._mimi_reply_stream_ready = False
        self._reply_pcm_buffer = torch.zeros((0,), dtype=torch.float32)
        self._reply_wall_t0 = 0.0
        self._reply_pcm_generated = 0
        self._reply_chunks_enqueued = 0
        self._reply_chunks_rendered = 0
        self._reply_chunks_pushed = 0
        self._reply_generation = 0
        self._max_render_queue = 0
        self._reply_ending = False
        self._prev_chunk_last_latent = None
        self._prev_chunk_tail_frames = None
        self._overlap_tokens = None
        self._render_queue: _queue_mod.Queue[PendingRenderChunk] = _queue_mod.Queue()
        self._render_threads: list[threading.Thread] = []
        self._render_stop = threading.Event()
        self._next_chunk_index = 0
        self._next_fm_chunk_index = 0
        self._next_emit_chunk_index = 0
        self._ready_chunks: dict[int, tuple[RenderedChunk, torch.Tensor]] = {}
        self._fm_cv = threading.Condition()
        self._emit_lock = threading.Lock()

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
        ae_state_dict = {
            k.replace("gen.", ""): v for k, v in renderer_ckpt.items() if k.startswith("gen.")
        }
        self.renderer.load_state_dict(ae_state_dict, strict=False)

    @torch.no_grad()
    def prepare_reference(self, ref_path: str) -> None:
        image = self.data_processor.default_img_loader(ref_path)
        if self.crop:
            image = self.data_processor.process_img(image)
        s_tensor = self.data_processor.transform(image).unsqueeze(0).to(self.device)
        self.idle_frame = s_tensor[0].detach().cpu()
        # Pre-converted idle frame for the fMP4 producer: H,W,3 uint8 in [0,255].
        # ToTensor() gives [0,1], so a plain clamp(0,1)*255 round is correct here.
        idle_u8 = (
            self.idle_frame.clamp(0, 1).mul(255).round().to(torch.uint8)
        )
        self.idle_frame_uint8 = (
            idle_u8.permute(1, 2, 0).contiguous().numpy()
        )
        self.f_r, self.g_r = self.renderer.dense_feature_encoder(s_tensor)
        self.ref_x = self.renderer.latent_token_encoder(s_tensor)

    @torch.no_grad()
    def warmup(self, n_passes: int = 2) -> float:
        """Run dummy chunks through generator+renderer to fill CUDA graph cache."""
        t0 = time.perf_counter()
        fake_pcm = torch.zeros(self.chunk_samples, dtype=torch.float32)
        warmup_stream = torch.cuda.Stream(self.device)
        for i in range(n_passes):
            try:
                self._mimi_reply_stream_ready = False
                self._render_reply_chunk(
                    fake_pcm,
                    chunk_index=i,
                    reply_generation=self._reply_generation,
                    render_stream=warmup_stream,
                )
            except Exception as exc:
                print(f"[IMTalker] warmup pass {i} error (non-fatal): {exc}")
        self.fm_stream_state = None
        self._frame_residual = 0.0
        self._mimi_reply_stream_ready = False
        self._next_chunk_index = 0
        self._next_fm_chunk_index = 0
        self._next_emit_chunk_index = 0
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        self._prev_chunk_last_latent = None
        self._prev_chunk_tail_frames = None
        self._overlap_tokens = None
        elapsed = time.perf_counter() - t0
        print(f"[IMTalker] warmup done ({n_passes} passes, {elapsed:.2f}s)")
        return elapsed

    def _ensure_mimi_streaming(self) -> None:
        if getattr(self.mimi, "_streaming_state", None) is None:
            self.mimi.streaming_forever(1)
        if not self._mimi_reply_stream_ready:
            self.mimi.reset_streaming()
            self._mimi_reply_stream_ready = True

    def _target_frames_for_pcm(self, pcm_num_samples: int) -> int:
        exact = (pcm_num_samples * float(self.opt.fps) / float(self.mimi.sample_rate)) + self._frame_residual
        target_frames = max(1, int(exact))
        self._frame_residual = exact - target_frames
        return target_frames

    @torch.no_grad()
    def _encode_reply_pcm_to_tokens(
        self,
        pcm: torch.Tensor,
        *,
        original_num_samples: int | None = None,
    ) -> torch.Tensor:
        pcm = pcm.detach().cpu().float().reshape(-1)
        if pcm.numel() == 0:
            return torch.zeros((0,), dtype=torch.long)

        self._ensure_mimi_streaming()
        if original_num_samples is None:
            original_num_samples = int(pcm.numel())
        total = pcm.numel()
        remainder = total % self.mimi_frame_size
        if remainder:
            pcm = F.pad(pcm, (0, self.mimi_frame_size - remainder))
        wav = pcm.unsqueeze(0).unsqueeze(0).to(self.device)
        codes = self.mimi.encode(wav)
        tokens_cb0 = codes[0, 0].long().cpu()
        expected_frames = (
            int(original_num_samples) + self.mimi_frame_size - 1
        ) // self.mimi_frame_size
        return tokens_cb0[:expected_frames]

    @torch.no_grad()
    def _decode_sample_to_frames(self, sample: torch.Tensor) -> torch.Tensor:
        total_frames = sample.shape[1]
        batch_size = getattr(self, "_render_batch_size", 4)

        ta_r = self.renderer.adapt(self.ref_x, self.g_r)
        m_r = self.renderer.latent_token_decoder(ta_r)

        g_r_exp = self.g_r.expand(total_frames, -1)
        sample_flat = sample.squeeze(0)
        ta_c_all = self.renderer.adapt(sample_flat, g_r_exp)
        m_c_all = self.renderer.latent_token_decoder(ta_c_all)

        all_frames = []
        for start in range(0, total_frames, batch_size):
            end = min(start + batch_size, total_frames)
            bs = end - start
            m_c_batch = tuple(m[start:end] for m in m_c_all)
            m_r_batch = tuple(m.expand(bs, -1, -1, -1) for m in m_r)
            f_r_batch = [f.expand(bs, -1, -1, -1) for f in self.f_r]
            out = self.renderer.decode(m_c_batch, m_r_batch, f_r_batch).clone()
            all_frames.append(out)

        return torch.cat(all_frames, dim=0)

    @torch.no_grad()
    def _render_reply_chunk(
        self,
        pcm_chunk: torch.Tensor,
        *,
        chunk_index: int,
        reply_generation: int,
        render_stream: torch.cuda.Stream,
        original_num_samples: int | None = None,
    ) -> RenderedChunk:
        t_chunk_start = time.perf_counter()
        render_num_samples = int(original_num_samples or pcm_chunk.numel())
        with self._fm_cv:
            while (
                chunk_index != self._next_fm_chunk_index
                and not self._render_stop.is_set()
                and reply_generation == self._reply_generation
            ):
                self._fm_cv.wait(timeout=0.05)
            if reply_generation != self._reply_generation:
                raise RuntimeError("Skipping stale reply chunk")
            if self._render_stop.is_set():
                raise RuntimeError("Render worker stopping")

            with torch.cuda.stream(render_stream):
                t0 = time.perf_counter()
                tokens = self._encode_reply_pcm_to_tokens(
                    pcm_chunk,
                    original_num_samples=render_num_samples,
                )
                t_mimi = time.perf_counter() - t0
                if tokens.shape[0] == 0:
                    raise RuntimeError("Cannot render empty reply chunk")

                target_frames = self._target_frames_for_pcm(render_num_samples)
                max_overlap_frames = max(self.render_window_frames - target_frames, 0)
                overlap_frames = min(self.overlap_frames, max_overlap_frames)
                tokens_for_encode = tokens
                if (
                    overlap_frames > 0
                    and self.overlap_token_count > 0
                    and self._overlap_tokens is not None
                    and self._overlap_tokens.shape[0] >= self.overlap_token_count
                ):
                    tokens_for_encode = torch.cat([self._overlap_tokens, tokens], dim=0)
                else:
                    overlap_frames = 0

                # cb0 token IDs -> embed + linear interp to video frame rate.
                # Token IDs cannot be linearly interpolated, so this MUST go
                # through the MoshiTokenEncoder embedding before any resampling.
                tokens_dev = tokens_for_encode.unsqueeze(0).to(self.device)
                a_feat = self.fm.audio_encoder.inference(
                    tokens_dev, seq_len=target_frames + overlap_frames
                )
                if self.overlap_token_count > 0:
                    self._overlap_tokens = tokens[-self.overlap_token_count:].clone()
                else:
                    self._overlap_tokens = None

                t0 = time.perf_counter()
                stream_state = self.fm_stream_state
                sample, next_state = self.fm.sample(
                    {'ref_x': self.ref_x, 'a_feat': a_feat},
                    a_cfg_scale=self.a_cfg_scale,
                    nfe=self.nfe,
                    stream_state=stream_state,
                    return_state=True,
                )
                render_stream.synchronize()
                t_generator = time.perf_counter() - t0
                self.fm_stream_state = next_state
                if overlap_frames > 0:
                    sample = sample[:, overlap_frames:, :]
                    a_feat = a_feat[:, overlap_frames:, :]

            boundary_l2 = None
            first_latent = sample[0, 0].detach().cpu()
            if self._prev_chunk_last_latent is not None:
                boundary_l2 = torch.norm(first_latent - self._prev_chunk_last_latent).item()
            self._prev_chunk_last_latent = sample[0, -1].detach().cpu().clone()
            self._next_fm_chunk_index += 1
            self._fm_cv.notify_all()

        with torch.cuda.stream(render_stream):
            t0 = time.perf_counter()
            frames = self._decode_sample_to_frames(sample)
            render_stream.synchronize()
            t_renderer = time.perf_counter() - t0

        result = RenderedChunk(
            chunk_index=chunk_index,
            frames=frames.detach().cpu(),
            sample_latents=sample.detach().cpu(),
            conditioning_latents=a_feat.detach().cpu(),
        )

        t_total = time.perf_counter() - t_chunk_start
        boundary_msg = "" if boundary_l2 is None else f"  boundary_l2={boundary_l2:.3f}"
        print(
            f"[TIMING] chunk {result.chunk_index:03d} | "
            f"mimi_encode={t_mimi:.3f}s  generator={t_generator:.3f}s  "
            f"renderer={t_renderer:.3f}s  total={t_total:.3f}s  ({target_frames} frames)"
            f"{boundary_msg}"
        )
        return result

    # ---------------------------------------------------------------- #
    # av_session push path
    # ---------------------------------------------------------------- #
    def _push_to_av(self, frames_t: torch.Tensor, pcm24k_t: torch.Tensor) -> None:
        """Convert a rendered chunk and push it to the active fMP4 session.

        Skips silently if no viewer is connected.
        """
        av = self.av_session
        if av is None:
            return
        # frames: [N,3,H,W] in [0,1] on CPU. Convert to [N,H,W,3] uint8.
        x = frames_t.detach().clamp(0, 1).mul(255).round().to(torch.uint8)
        frames_np = x.permute(0, 2, 3, 1).contiguous().cpu().numpy()

        # Resample mimi-rate float PCM (24 kHz) to 48 kHz int16 mono.
        pcm_np = pcm24k_t.detach().cpu().numpy().astype(np.float32, copy=False)
        if pcm_np.ndim != 1:
            pcm_np = pcm_np.reshape(-1)
        src_sr = int(self.mimi.sample_rate)
        if src_sr == self._av_audio_sample_rate:
            pcm48 = pcm_np
        else:
            pcm48 = librosa.resample(
                pcm_np, orig_sr=src_sr, target_sr=self._av_audio_sample_rate
            )
        target_audio_samples = int(frames_np.shape[0] * self._av_samples_per_frame)
        if pcm48.shape[0] > target_audio_samples:
            pcm48 = pcm48[:target_audio_samples]
        elif pcm48.shape[0] < target_audio_samples:
            pcm48 = np.pad(pcm48, (0, target_audio_samples - pcm48.shape[0]))
        pcm48 = np.clip(pcm48, -1.0, 1.0)
        pcm48_int16 = (pcm48 * 32767.0).astype(np.int16)
        self._reply_chunks_pushed += 1
        if self.debug_session:
            frame_sec = frames_np.shape[0] / max(float(self.opt.fps), 1.0)
            audio_sec = pcm48_int16.shape[0] / max(float(self._av_audio_sample_rate), 1.0)
            delta_ms = (audio_sec - frame_sec) * 1000.0
            av_state = av.debug_state() if hasattr(av, "debug_state") else {}
            print(
                f"[DEBUG/imtalker] push_av | chunk={self._reply_chunks_pushed - 1:03d} "
                f"frames={frames_np.shape[0]} video={frame_sec:.3f}s "
                f"audio={audio_sec:.3f}s delta={delta_ms:+.1f}ms "
                f"play_q={av_state.get('play_queue_len', '?')} av_q={av_state.get('av_q_size', '?')}"
            )
        av.push_chunk(frames_np, pcm48_int16)

    def debug_state(self) -> dict:
        elapsed = max(time.perf_counter() - self._reply_wall_t0, 0.0) if self._reply_wall_t0 else 0.0
        generated_sec = self._reply_pcm_generated / max(float(self.mimi.sample_rate), 1.0)
        return {
            "render_queue_len": self._render_queue.qsize(),
            "max_render_queue": self._max_render_queue,
            "reply_pcm_buffer_samples": int(self._reply_pcm_buffer.numel()),
            "reply_pcm_generated_samples": int(self._reply_pcm_generated),
            "reply_generated_sec": round(generated_sec, 3),
            "reply_elapsed_sec": round(elapsed, 3),
            "generation_rate_x": round(generated_sec / elapsed, 3) if elapsed > 0 else 0.0,
            "chunks_enqueued": self._reply_chunks_enqueued,
            "chunks_rendered": self._reply_chunks_rendered,
            "chunks_pushed": self._reply_chunks_pushed,
            "frame_residual": round(self._frame_residual, 6),
            "reply_ending": self._reply_ending,
        }

    def _maybe_end_reply_hold(self) -> None:
        av = self.av_session
        if av is None or not self._reply_ending:
            return
        if self._reply_pcm_buffer.numel() != 0:
            return
        if not self._render_queue.empty():
            return
        if self._ready_chunks:
            return
        if self._reply_chunks_pushed < self._reply_chunks_enqueued:
            return
        if hasattr(av, "end_reply"):
            av.end_reply()
        self._reply_ending = False

    def reset_reply(self) -> None:
        """Drop any pending state from a prior Moshi turn.

        Called when a new Moshi WS connection starts (so the next reply renders
        with a fresh fm_stream_state). The render worker keeps running across
        turns; any chunk currently being rendered finishes naturally, but no
        further chunks from the prior turn will be enqueued.
        """
        while True:
            try:
                self._render_queue.get_nowait()
            except _queue_mod.Empty:
                break
        self._reply_generation += 1
        with self._fm_cv:
            self._next_fm_chunk_index = 0
            self._fm_cv.notify_all()
        with self._emit_lock:
            self._ready_chunks.clear()
            self._next_emit_chunk_index = 0
        self.fm_stream_state = None
        self._frame_residual = 0.0
        self._mimi_reply_stream_ready = False
        if getattr(self.mimi, "_streaming_state", None) is not None:
            self.mimi.reset_streaming()
        self._reply_pcm_buffer = torch.zeros((0,), dtype=torch.float32)
        self._reply_wall_t0 = 0.0
        self._reply_pcm_generated = 0
        self._reply_chunks_enqueued = 0
        self._reply_chunks_rendered = 0
        self._reply_chunks_pushed = 0
        self._max_render_queue = 0
        self._reply_ending = False
        self._next_chunk_index = 0
        self._prev_chunk_last_latent = None
        self._prev_chunk_tail_frames = None
        self._overlap_tokens = None
        av = self.av_session
        if av is not None and hasattr(av, "end_reply"):
            av.end_reply()

    def _enqueue_chunk(self, chunk: torch.Tensor, *, valid_samples: int | None = None) -> None:
        if valid_samples is None:
            valid_samples = int(chunk.numel())
        chunk_index = self._next_chunk_index
        self._next_chunk_index += 1
        self._render_queue.put(
            PendingRenderChunk(
                chunk_index=chunk_index,
                reply_generation=self._reply_generation,
                pcm=chunk,
                valid_samples=int(valid_samples),
            )
        )
        self._reply_chunks_enqueued += 1
        self._max_render_queue = max(self._max_render_queue, self._render_queue.qsize())

    def _enqueue_ready_chunks(self) -> None:
        """Slice _reply_pcm_buffer into chunk_samples-sized renderable chunks."""
        while self._reply_pcm_buffer.numel() >= self.chunk_samples:
            chunk = self._reply_pcm_buffer[: self.chunk_samples].clone()
            self._reply_pcm_buffer = self._reply_pcm_buffer[self.chunk_samples :]
            self._enqueue_chunk(chunk)

    def _flush_ready_chunks_locked(self) -> None:
        while True:
            ready = self._ready_chunks.pop(self._next_emit_chunk_index, None)
            if ready is None:
                break
            rendered, pcm_for_push = ready
            if self._prev_chunk_tail_frames is not None and rendered.frames.shape[0] > 0:
                blend_n = min(3, self._prev_chunk_tail_frames.shape[0], rendered.frames.shape[0])
                alpha = torch.linspace(
                    0.0, 1.0, blend_n + 2, dtype=rendered.frames.dtype
                )[1:-1].view(blend_n, 1, 1, 1)
                prev_tail = self._prev_chunk_tail_frames[-blend_n:]
                rendered.frames[:blend_n] = (
                    (1.0 - alpha) * prev_tail + alpha * rendered.frames[:blend_n]
                )
            self._prev_chunk_tail_frames = rendered.frames[-3:].clone()
            self._push_to_av(rendered.frames, pcm_for_push)
            if self.debug_session:
                print(
                    f"[DEBUG/imtalker] emit_done | pushed_idx={self._next_emit_chunk_index:03d} "
                    f"queue_remaining={self._render_queue.qsize()} pushed={self._reply_chunks_pushed}"
                )
            self._next_emit_chunk_index += 1
        self._maybe_end_reply_hold()

    def _render_worker_loop(self, worker_id: int) -> None:
        worker_stream = torch.cuda.Stream(self.device)
        while not self._render_stop.is_set():
            try:
                pending = self._render_queue.get(timeout=0.1)
            except _queue_mod.Empty:
                continue
            pcm_chunk = pending.pcm
            chunk_samples = int(pending.valid_samples)
            padded_samples = int(pcm_chunk.numel())
            if self.debug_session:
                print(
                    f"[DEBUG/imtalker] render_start[{worker_id}] | queue_before={self._render_queue.qsize() + 1} "
                    f"queue_after_pop={self._render_queue.qsize()} chunk_samples={chunk_samples} "
                    f"padded_samples={padded_samples}"
                )
            try:
                rendered = self._render_reply_chunk(
                    pcm_chunk,
                    chunk_index=pending.chunk_index,
                    reply_generation=pending.reply_generation,
                    render_stream=worker_stream,
                    original_num_samples=chunk_samples,
                )
            except Exception as exc:
                print(f"[IMTalker] render error: {exc}")
                continue
            with self._emit_lock:
                if pending.reply_generation != self._reply_generation:
                    continue
                self._reply_chunks_rendered += 1
                self._ready_chunks[pending.chunk_index] = (
                    rendered,
                    pcm_chunk[:chunk_samples],
                )
                if self.debug_session:
                    print(
                        f"[DEBUG/imtalker] render_done[{worker_id}] | rendered={self._reply_chunks_rendered} "
                        f"ready={len(self._ready_chunks)} queue_remaining={self._render_queue.qsize()}"
                    )
                self._flush_ready_chunks_locked()

    def _ensure_render_worker(self) -> None:
        alive = [t for t in self._render_threads if t.is_alive()]
        self._render_threads = alive
        if len(self._render_threads) >= self.render_workers:
            return
        self._render_stop.clear()
        while len(self._render_threads) < self.render_workers:
            worker_id = len(self._render_threads)
            t = threading.Thread(
                target=self._render_worker_loop,
                args=(worker_id,),
                daemon=True,
                name=f"imtalker-render-{worker_id}",
            )
            t.start()
            self._render_threads.append(t)

    async def handle_moshi_output(
        self, tokens: torch.Tensor, pcm: torch.Tensor, latents: torch.Tensor
    ) -> list[RenderedChunk]:
        """Buffer Moshi reply PCM and enqueue chunks for the render worker."""
        del tokens, latents
        # No viewer connected -> drop the PCM. Saves GPU and avoids growing
        # fm_stream_state context for an audience that isn't there.
        if self.av_session is None:
            return []
        flat_pcm = pcm.detach().cpu().float().reshape(-1)
        if self._reply_wall_t0 == 0.0:
            self._reply_wall_t0 = time.perf_counter()
        self._reply_pcm_generated += int(flat_pcm.numel())
        self._reply_pcm_buffer = torch.cat([self._reply_pcm_buffer, flat_pcm], dim=0)
        self._reply_ending = False
        queue_before = self._render_queue.qsize()
        self._enqueue_ready_chunks()
        queue_after = self._render_queue.qsize()
        if queue_after > queue_before:
            buffered_sec = self._reply_pcm_buffer.numel() / self.mimi.sample_rate
            elapsed = max(time.perf_counter() - self._reply_wall_t0, 1e-6)
            generated_sec = self._reply_pcm_generated / self.mimi.sample_rate
            msg = (
                f"[TIMING] moshi_output | +{flat_pcm.numel()} samples | "
                f"residual={buffered_sec:.2f}s | enqueued {queue_after - queue_before} chunk(s) | "
                f"render_queue={queue_after}"
            )
            if self.debug_session:
                msg += (
                    f" | generated={generated_sec:.2f}s wall={elapsed:.2f}s "
                    f"rate={generated_sec / elapsed:.2f}x max_queue={self._max_render_queue}"
                )
            print(msg)
        if not self._render_queue.empty():
            self._ensure_render_worker()
        return []

    async def handle_user_audio(self, latents: torch.Tensor) -> list[RenderedChunk]:
        del latents
        return []

    async def finalize_pending_reply(self) -> None:
        """Flush any tail PCM into the render queue.

        Does NOT block on the worker — the av_session keeps streaming idle
        frames to the browser regardless, and the worker pushes the tail chunk
        whenever the renderer finishes. This avoids blocking the Moshi WS
        teardown on a slow render.
        """
        tail_samples = int(self._reply_pcm_buffer.numel())
        if tail_samples == 0:
            self._reply_ending = True
            self._maybe_end_reply_hold()
            return
        if self.av_session is None:
            self._reply_pcm_buffer = torch.zeros((0,), dtype=torch.float32)
            self._reply_ending = False
            return
        tail_sec = tail_samples / max(self.mimi.sample_rate, 1)
        print(
            f"[TIMING] finalize | tail_pcm={tail_samples} samples ({tail_sec:.2f}s) | "
            f"queued_before_flush={self._render_queue.qsize()}"
        )
        chunk = self._reply_pcm_buffer.clone()
        self._reply_pcm_buffer = torch.zeros((0,), dtype=torch.float32)
        if tail_samples < self.chunk_samples:
            chunk = F.pad(chunk, (0, self.chunk_samples - tail_samples))
        self._reply_ending = True
        self._enqueue_chunk(chunk, valid_samples=tail_samples)
        self._ensure_render_worker()
