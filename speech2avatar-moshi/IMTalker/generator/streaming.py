"""Streaming inference for IMTalker.

Reference image is fixed for the session; audio arrives in chunks; RGB frames
are emitted with bounded latency.  Reuses InferenceAgent's compiled renderer
and decode_image batching path.

Audio chunk = num_frames_for_clip / fps = 50 / 25 = 2.0 s by default
(matches the FMT training window).  See generator/FM.py:sample_chunk for the
chunk-level state contract.
"""
from __future__ import annotations

import collections
import os
import time
from dataclasses import dataclass
from typing import Callable, Iterator, Optional

import numpy as np
import torch


@dataclass
class ChunkTimings:
    chunk_idx: int
    n_audio_samples: int
    n_emitted_frames: int
    t_audio_to_tensor_ms: float
    t_generator_ms: float
    t_decode_ms: float
    t_total_ms: float

    def line(self) -> str:
        return (
            "[stream] chunk=%d frames=%d audio_samples=%d | "
            "to_tensor=%.1fms generator=%.1fms decode=%.1fms total=%.1fms"
            % (
                self.chunk_idx,
                self.n_emitted_frames,
                self.n_audio_samples,
                self.t_audio_to_tensor_ms,
                self.t_generator_ms,
                self.t_decode_ms,
                self.t_total_ms,
            )
        )


def _cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class StreamingInferenceAgent:
    """Wraps an InferenceAgent for chunked audio → frames inference.

    Lifecycle:
        agent = StreamingInferenceAgent(inference_agent, ref_path,
                                        chunk_frames=50, crop=True)
        agent.warmup()
        for audio_chunk_np in audio_source:          # 1-D float32 @ 16 kHz
            agent.feed_audio(audio_chunk_np)
            for frames in agent.drain():             # frames: [N, 3, 512, 512]
                send_to_consumer(frames)

    The reference image (face crop, identity encode, motion ref) is computed
    once in __init__.  After warmup, .step() / .drain() pay only the per-chunk
    generator + renderer cost.
    """

    def __init__(
        self,
        inference_agent,           # type: ignore[no-untyped-def]
        ref_path: str,
        chunk_frames: int = 50,
        crop: bool = True,
        a_cfg_scale: float = 1.0,
        nfe: int = 10,
        seed: int = 25,
        debug_stream: bool = False,
    ):
        self.agent = inference_agent
        self.opt = inference_agent.opt
        self.device = self.opt.rank
        self.fps = self.opt.fps
        self.sr = self.opt.sampling_rate

        self.fm = inference_agent.fm
        self.ae = inference_agent.ae
        self.dp = inference_agent.data_processor

        self.N = self.fm.num_frames_for_clip                 # 50
        self.K = self.fm.num_prev_frames                     # 10
        if not (1 <= chunk_frames <= self.N):
            raise ValueError(
                f"chunk_frames must be in [1, {self.N}]; got {chunk_frames}"
            )
        self.chunk_frames = chunk_frames

        # Audio samples per emitted-frame's worth of audio.  We always FEED
        # the model N frames of audio (sr * N / fps samples) to match the
        # training window, but we only EMIT chunk_frames per call.
        self.samples_per_chunk = int(round(self.sr * self.N / self.fps))
        self.samples_per_emit  = int(round(self.sr * self.chunk_frames / self.fps))

        self.a_cfg_scale = a_cfg_scale
        self.nfe = nfe
        self.seed = seed
        self.debug_stream = debug_stream

        # --- one-shot reference encode (the expensive once-per-session part) ---
        # DataProcessor.preprocess() requires an audio path, so we do the
        # image-only work inline to avoid touching audio at all here.
        with torch.no_grad():
            data = self._encode_reference_image(ref_path, crop=crop)
            self.f_r, self.t_r, self.g_r = self.agent.encode_image(
                data["s"].to(self.device)
            )
        # encode_image returns t_r as [1, 32] — already batched.
        assert self.t_r.dim() == 2, f"expected t_r [1,32], got {tuple(self.t_r.shape)}"

        # Streaming state
        self.state = self.fm.make_initial_stream_state(self.t_r)
        self.audio_buffer: collections.deque = collections.deque()
        self.audio_buffer_len = 0
        self.chunk_idx = 0
        self._timings: list[ChunkTimings] = []

    # ------------------------------------------------------------------ #
    # one-shot helpers
    # ------------------------------------------------------------------ #
    def _encode_reference_image(self, ref_path: str, crop: bool) -> dict:
        """Subset of DataProcessor.preprocess that does only image work."""
        from PIL import Image  # local imports — keeps cold start lean
        import cv2

        img_bgr = cv2.imread(ref_path)
        img = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        if crop:
            img = self.dp.process_img(img)
        s_tensor = self.dp.transform(img).unsqueeze(0)
        return {"s": s_tensor}

    # ------------------------------------------------------------------ #
    # warmup — runs one full chunk through the pipeline so torch.compile
    # captures CUDA graphs and inductor caches before the first real chunk.
    # ------------------------------------------------------------------ #
    def warmup(self) -> float:
        t0 = time.perf_counter()
        fake = np.zeros(self.samples_per_chunk, dtype=np.float32)
        # Run twice: 1st triggers compile, 2nd verifies steady-state replay
        for _ in range(2):
            saved_state = {k: v.clone() for k, v in self.state.items()}
            _ = self._run_one_chunk(fake)
            self.state = saved_state
        # Warmup must not affect chunk numbering or buffer state
        self.chunk_idx = 0
        self._timings.clear()
        self.audio_buffer.clear()
        self.audio_buffer_len = 0
        _cuda_sync()
        return time.perf_counter() - t0

    # ------------------------------------------------------------------ #
    # public API
    # ------------------------------------------------------------------ #
    def feed_audio(self, samples: np.ndarray) -> None:
        """Push float32 mono samples at self.sr (16 kHz)."""
        if samples.dtype != np.float32:
            samples = samples.astype(np.float32, copy=False)
        if samples.ndim != 1:
            samples = samples.reshape(-1)
        self.audio_buffer.append(samples)
        self.audio_buffer_len += samples.shape[0]

    def has_chunk(self) -> bool:
        return self.audio_buffer_len >= self.samples_per_emit

    def drain(self) -> Iterator[torch.Tensor]:
        """Yield as many frame chunks as the buffer currently contains."""
        while self.has_chunk():
            yield self.step()

    def step(self) -> torch.Tensor:
        """Consume one chunk of audio, return [chunk_frames, 3, 512, 512]."""
        if not self.has_chunk():
            raise RuntimeError(
                f"step() called with only {self.audio_buffer_len} samples in buffer "
                f"(need {self.samples_per_emit})"
            )
        audio_chunk = self._take_samples(self.samples_per_emit)
        # Pad up to samples_per_chunk so the model sees its full N=50 window.
        if audio_chunk.shape[0] < self.samples_per_chunk:
            pad = np.zeros(
                self.samples_per_chunk - audio_chunk.shape[0], dtype=np.float32
            )
            audio_chunk = np.concatenate([audio_chunk, pad])
        return self._run_one_chunk(audio_chunk)

    def flush(self) -> Optional[torch.Tensor]:
        """Emit any remaining buffered audio as a (possibly partial) chunk.

        The model is still given its full N-frame window (zero-padded).  Returns
        None if the buffer is empty.
        """
        if self.audio_buffer_len == 0:
            return None
        n = min(self.audio_buffer_len, self.samples_per_emit)
        audio_chunk = self._take_samples(n)
        # Trim emit_frames to the actual audio we have.
        emit_frames_actual = max(1, int(round(n * self.fps / self.sr)))
        emit_frames_actual = min(emit_frames_actual, self.chunk_frames)
        if audio_chunk.shape[0] < self.samples_per_chunk:
            pad = np.zeros(
                self.samples_per_chunk - audio_chunk.shape[0], dtype=np.float32
            )
            audio_chunk = np.concatenate([audio_chunk, pad])
        return self._run_one_chunk(audio_chunk, emit_frames=emit_frames_actual)

    @property
    def timings(self) -> list[ChunkTimings]:
        return list(self._timings)

    # ------------------------------------------------------------------ #
    # internals
    # ------------------------------------------------------------------ #
    def _take_samples(self, n: int) -> np.ndarray:
        """Pop n samples off the front of the deque."""
        out = np.empty(n, dtype=np.float32)
        filled = 0
        while filled < n and self.audio_buffer:
            head = self.audio_buffer[0]
            need = n - filled
            if head.shape[0] <= need:
                out[filled:filled + head.shape[0]] = head
                filled += head.shape[0]
                self.audio_buffer.popleft()
            else:
                out[filled:] = head[:need]
                self.audio_buffer[0] = head[need:]
                filled = n
        self.audio_buffer_len -= n
        if filled < n:
            # buffer underflow — shouldn't happen given has_chunk() guard
            out[filled:] = 0.0
        return out

    @torch.no_grad()
    def _run_one_chunk(
        self, audio_chunk: np.ndarray, emit_frames: Optional[int] = None
    ) -> torch.Tensor:
        if emit_frames is None:
            emit_frames = self.chunk_frames

        t_chunk0 = time.perf_counter()

        # 1. audio numpy → tensor on device
        _cuda_sync()
        t0 = time.perf_counter()
        audio_t = (
            torch.from_numpy(audio_chunk).to(self.device, non_blocking=True).unsqueeze(0)
        )
        _cuda_sync()
        t_audio_to_tensor = time.perf_counter() - t0

        # 2. generator: ODE sample one chunk + carry state
        t0 = time.perf_counter()
        sample, new_state = self.fm.sample_chunk(
            audio_samples=audio_t,
            ref_x=self.t_r,
            state=self.state,
            emit_frames=emit_frames,
            a_cfg_scale=self.a_cfg_scale,
            nfe=self.nfe,
            seed=self.seed,
        )
        self.state = new_state
        _cuda_sync()
        t_gen = time.perf_counter() - t0

        # 3. renderer: decode the chunk's motion latents to RGB
        t0 = time.perf_counter()
        decoded = self.agent.decode_image(self.f_r, self.t_r, sample, self.g_r)
        frames = decoded["d_hat"]  # [emit_frames, 3, 512, 512]
        _cuda_sync()
        t_dec = time.perf_counter() - t0

        if self.debug_stream:
            if not torch.isfinite(frames).all():
                bad = (~torch.isfinite(frames)).sum().item()
                print(f"[stream] WARN chunk={self.chunk_idx} non-finite count={bad}")

        t_total = time.perf_counter() - t_chunk0
        self._timings.append(
            ChunkTimings(
                chunk_idx=self.chunk_idx,
                n_audio_samples=int(audio_chunk.shape[0]),
                n_emitted_frames=int(frames.shape[0]),
                t_audio_to_tensor_ms=t_audio_to_tensor * 1e3,
                t_generator_ms=t_gen * 1e3,
                t_decode_ms=t_dec * 1e3,
                t_total_ms=t_total * 1e3,
            )
        )
        if self.debug_stream:
            print(self._timings[-1].line())

        self.chunk_idx += 1
        return frames

    # ------------------------------------------------------------------ #
    # convenience: stream from a wav file end-to-end (file-driven test)
    # ------------------------------------------------------------------ #
    def stream_from_wav(
        self,
        wav_path: str,
        on_frames: Optional[Callable[[torch.Tensor, ChunkTimings], None]] = None,
    ) -> int:
        """Read a wav, feed it as if it were a live stream, run all chunks.

        Returns total emitted frame count.  If on_frames is given it is called
        with (frames, timings) for each chunk.
        """
        from generator.generate import _load_audio_native, _resample_mono

        speech_native, sr_native = _load_audio_native(wav_path)
        speech = _resample_mono(speech_native, sr_native, self.sr)
        if not isinstance(speech, np.ndarray):
            speech = np.asarray(speech, dtype=np.float32)
        speech = speech.astype(np.float32, copy=False)

        # Push the entire wav as if it had arrived live.  In a real loop you
        # would call feed_audio() repeatedly with small slices and call
        # drain() between feeds; the math is identical.
        self.feed_audio(speech)
        total_frames = 0
        for frames in self.drain():
            total_frames += int(frames.shape[0])
            if on_frames is not None:
                on_frames(frames, self._timings[-1])
        # Drain any tail less than one full emit chunk.
        tail = self.flush()
        if tail is not None:
            total_frames += int(tail.shape[0])
            if on_frames is not None:
                on_frames(tail, self._timings[-1])
        return total_frames
