"""
launch_live.py - combined Moshi + IMTalker launcher for sibling repos.

Expected layout:
    /workspace/IMTalker
    /workspace/moshi

Open in browser:
    http://localhost:8998/        - combined avatar page
    http://localhost:8998/moshi   - hidden controller UI (debug/fallback)
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import fractions
import io
import inspect
import os
import random
import sys
import tarfile
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
import typing as tp

import aiohttp
from aiohttp import web
from huggingface_hub import hf_hub_download
import numpy as np
import sphn
import torch

# Moshi creates some CUDA graph / torch.compile wrappers at import time. On the
# 5090 runtime these wrappers can poison the CUDA context and crash later during
# Mimi/LM init, so use eager Moshi kernels unless the caller explicitly overrides
# these environment variables before starting Python.
os.environ.setdefault("NO_CUDA_GRAPH", "1")
os.environ.setdefault("NO_TORCH_COMPILE", "1")

# fMP4 streaming constants — single muxed timeline so the browser does not
# have to align two clocks. 48 kHz / 20 ms packets matches the realtime_server
# pipeline so the browser-side AAC decoder paths are well-trodden.
AUDIO_SR_OUT = 48000
AUDIO_PTIME = 0.020
AUDIO_SAMPLES_PER_PKT = int(AUDIO_SR_OUT * AUDIO_PTIME)  # 960
AUDIO_SAMPLES_PER_FRAME = AUDIO_SAMPLES_PER_PKT * 2  # 1920 = 48k / 25fps

_MOSHI_REPO = Path(__file__).resolve().parent.parent / "moshi"
_MOSHI_PKG = _MOSHI_REPO / "moshi"
if _MOSHI_PKG.exists() and str(_MOSHI_PKG) not in sys.path:
    sys.path.insert(0, str(_MOSHI_PKG))

sys.path.insert(0, str(Path(__file__).resolve().parent))

from generator.options.base_options import BaseOptions
from live_pipeline import LiveMoshiIMTalkerSession

from moshi.client_utils import log
from moshi.models import LMGen, LMModel, MimiModel, loaders
from moshi.run_inference import get_condition_tensors


def seed_all(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False


class LaunchOptions(BaseOptions):
    def initialize(self, parser):
        super().initialize(parser)
        parser.set_defaults(audio_feat_dim=512, nfe=5, a_cfg_scale=1.0, use_stream_state=True)
        parser.add_argument("--ref_path", required=True, type=str)
        parser.add_argument("--generator_path", required=True, type=str)
        parser.add_argument("--renderer_path", required=True, type=str)
        parser.add_argument(
            "--use_ema",
            action="store_true",
            default=True,
            help="When available, merge EMA-shadow weights into the full generator checkpoint for inference.",
        )
        parser.add_argument(
            "--no_use_ema",
            action="store_false",
            dest="use_ema",
            help="Disable EMA-merged generator loading and use raw checkpoint weights only.",
        )
        parser.add_argument("--moshi_repo", default=str(_MOSHI_REPO), type=str)
        parser.add_argument("--hf_repo", default="kyutai/moshiko-pytorch-bf16", type=str)
        parser.add_argument("--moshi_weight", default=None, type=str)
        parser.add_argument("--mimi_weight", default=None, type=str)
        parser.add_argument("--tokenizer", default=None, type=str)
        parser.add_argument("--lora_weight", default=None, type=str)
        parser.add_argument("--config_path", default=None, type=str)
        parser.add_argument("--host", default="0.0.0.0", type=str)
        parser.add_argument("--port", default=8998, type=int)
        parser.add_argument("--output_dir", default=None, type=str)
        parser.add_argument("--device", default="cuda", type=str)
        parser.add_argument("--crop", action="store_true")
        parser.add_argument(
            "--static_pose",
            default=None,
            type=str,
            help="Comma-separated 3-vector to repeat as pose on every frame, e.g. '0,0,0'.",
        )
        parser.add_argument(
            "--static_cam",
            default=None,
            type=str,
            help="Comma-separated 3-vector to repeat as camera on every frame, e.g. '0,0,0'.",
        )
        parser.add_argument(
            "--static_gaze",
            default=None,
            type=str,
            help="Comma-separated 2-vector to repeat as gaze on every frame, e.g. '0,0'.",
        )
        parser.add_argument(
            "--max_sentences",
            default=6,
            type=int,
            help="Maximum number of sentences Moshi should speak per reply. "
            "<=0 disables the sentence cap. Default 6 (was 1, which closed the "
            "session immediately after the first '.','!','?').",
        )
        parser.add_argument(
            "--max_text_tokens",
            default=200,
            type=int,
            help="Hard cap on Moshi text tokens per reply. <=0 disables the cap. "
            "Default 200 (was 40, which truncated mid-thought).",
        )
        parser.add_argument(
            "--debug_session",
            action="store_true",
            help="Verbose per-step logging of recv_loop / decode_and_send / "
            "limit checks. Helpful for diagnosing why a session closed early.",
        )
        parser.add_argument(
            "--skip_warmup",
            action="store_true",
            help="Skip IMTalker/Moshi startup warmup for faster debugging. "
            "First live response may be slower.",
        )
        parser.add_argument(
            "--dump_reply_dir",
            default=None,
            type=str,
            help="If set, save each Moshi reply to this directory as a .wav plus "
            "a raw latent .pt sidecar for offline replay/debugging.",
        )
        parser.add_argument(
            "--chunk_sec",
            default=1.0,
            type=float,
            help="Live PCM accumulation window for progressive AV chunks",
        )
        parser.add_argument(
            "--boundary_blend_frames",
            default=4,
            type=int,
            help="How many frames to cross-fade at each live chunk boundary.",
        )
        parser.add_argument(
            "--use_stream_state",
            action="store_true",
            help="Carry FM chunk-to-chunk state across live chunks. Enabled by default for stable streaming seams.",
        )
        parser.add_argument(
            "--no_use_stream_state",
            action="store_false",
            dest="use_stream_state",
            help=argparse.SUPPRESS,
        )
        parser.add_argument(
            "--render_batch_size",
            default=4,
            type=int,
            help="How many video frames the renderer decodes at once per chunk.",
        )
        parser.add_argument(
            "--ws_audio_buffer_ms",
            default=80.0,
            type=float,
            help="Initial client-side audio jitter buffer for the WS viewer in milliseconds.",
        )
        parser.add_argument(
            "--text_topk",
            default=25,
            type=int,
            help="Top-k for Moshi text token sampling. Lower = faster but less diverse.",
        )
        parser.add_argument(
            "--text_temperature",
            default=0.7,
            type=float,
            help="Temperature for Moshi text token sampling.",
        )
        parser.add_argument(
            "--half",
            action="store_const",
            const=torch.float16,
            default=torch.bfloat16,
            dest="dtype",
            help="Run Moshi inference with float16 instead of bfloat16.",
        )
        parser.add_argument(
            "--no_fuse_lora",
            action="store_false",
            dest="fuse_lora",
            default=True,
            help="Do not fuse LoRA layers into linear layers.",
        )
        parser.add_argument(
            "--no_moshi_cuda_graph",
            action="store_true",
            default=False,
            help="Disable Moshi CUDAGraph. Default on for speed; use this if your "
            "GPU/PyTorch build hits a graph-capture runtime error.",
        )
        parser.add_argument(
            "--moshi_cuda_graph",
            action="store_false",
            dest="no_moshi_cuda_graph",
            help=argparse.SUPPRESS,
        )
        return parser


class TranscriptStore:
    """Thread-safe rolling transcript exposed via /api/stream_state.

    The fMP4 stream carries video+audio; this store only tracks assistant
    text so the page can render a live caption alongside the stream.
    """

    def __init__(self):
        self._lock = threading.RLock()
        self.transcript_turn_id = -1
        self.transcript_version = 0
        self.assistant_text = ""

    def start_transcript_turn(self) -> int:
        with self._lock:
            self.transcript_turn_id += 1
            self.assistant_text = ""
            self.transcript_version += 1
            return self.transcript_turn_id

    def append_assistant_text(self, piece: str) -> None:
        if not piece:
            return
        with self._lock:
            self.assistant_text += piece
            self.transcript_version += 1

    def state(self, extra: dict | None = None) -> dict:
        with self._lock:
            data = {
                "assistant_text": self.assistant_text,
                "transcript_turn_id": self.transcript_turn_id,
                "transcript_version": self.transcript_version,
            }
            if extra:
                data.update(extra)
            return data


# Backwards-compatible alias so MoshiAvatarServerState's type hint still works.
AVSegmentStore = TranscriptStore


class FMP4StreamSession:
    """Per-viewer fMP4 streaming session.

    Owns the producer task that paces frames at wall-clock 25fps. Reply
    chunks are pushed in via push_chunk() (called from the renderer worker
    in LiveMoshiIMTalkerSession). When no reply chunk is queued, the
    producer fills with the idle reference frame + silent audio so the
    media timeline never stalls — that's what keeps the browser's MSE
    SourceBuffer fed and prevents stutter at segment boundaries.

    The fMP4 pump task lives in launch_live's stream handler and reads
    from av_q, muxing into a PyAV container.
    """

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        idle_frame_uint8: np.ndarray,
        fps: int,
        av_buffer_ticks: int = 50,  # 2 s slack at 25 fps
        debug: bool = False,
    ):
        self.loop = loop
        self.idle_frame_uint8 = idle_frame_uint8
        self.fps = fps
        self.debug = debug
        # Single queue of (vf_uint8_hwc, ap1_int16, ap2_int16) tuples so the
        # video/audio drop-oldest stays sample-accurate (independent queues
        # would risk dropping audio without dropping the matching video and
        # vice versa, breaking sync).
        self.av_q: asyncio.Queue = asyncio.Queue(maxsize=av_buffer_ticks)
        # Reply chunks waiting to be paced out into the av_q.
        # Each entry is [frames_uint8_hwc, pcm48_int16, pos_in_frames].
        self._play_queue: deque = deque()
        self._play_lock = threading.Lock()
        self._stop = asyncio.Event()
        self._task: asyncio.Task | None = None
        self.frames_emitted = 0
        self.dropped_ticks = 0
        self.reply_ticks_emitted = 0
        self.idle_ticks_emitted = 0
        self.chunks_pushed = 0
        self.max_play_queue = 0
        self.max_av_q = 0
        self._last_mode = "idle"
        self._last_mode_change_t = time.monotonic()

    def debug_state(self) -> dict:
        with self._play_lock:
            play_queue_len = len(self._play_queue)
        return {
            "frames_emitted": self.frames_emitted,
            "dropped_ticks": self.dropped_ticks,
            "reply_ticks_emitted": self.reply_ticks_emitted,
            "idle_ticks_emitted": self.idle_ticks_emitted,
            "chunks_pushed": self.chunks_pushed,
            "play_queue_len": play_queue_len,
            "max_play_queue": self.max_play_queue,
            "av_q_size": self.av_q.qsize(),
            "max_av_q": self.max_av_q,
            "mode": self._last_mode,
        }

    # ------------------------------------------------------------------ #
    def push_chunk(self, frames_np: np.ndarray, pcm48_int16: np.ndarray, meta: dict | None = None) -> None:
        """Called from the render worker (asyncio loop thread).

        Frames: [N,H,W,3] uint8. PCM: 1D int16 at 48 kHz mono.
        """
        with self._play_lock:
            self._play_queue.append([frames_np, pcm48_int16, 0, meta or {}])
            self.chunks_pushed += 1
            self.max_play_queue = max(self.max_play_queue, len(self._play_queue))
            if self.debug:
                audio_sec = pcm48_int16.shape[0] / max(float(AUDIO_SR_OUT), 1.0)
                video_sec = frames_np.shape[0] / max(float(self.fps), 1.0)
                print(
                    f"[DEBUG/fmp4] push_chunk | idx={self.chunks_pushed - 1:03d} "
                    f"frames={frames_np.shape[0]} video={video_sec:.3f}s "
                    f"audio={audio_sec:.3f}s play_q={len(self._play_queue)}"
                )

    # ------------------------------------------------------------------ #
    def start(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._produce())

    def stop(self) -> None:
        self._stop.set()

    # ------------------------------------------------------------------ #
    async def _produce(self) -> None:
        spf_a48 = AUDIO_SAMPLES_PER_FRAME  # 1920 samples per video frame
        silence = np.zeros(spf_a48, dtype=np.int16)
        pace_t0 = time.monotonic()
        pace_frame_no = 0
        current = None  # [frames_np, pcm48_int16, pos, meta]
        try:
            while not self._stop.is_set():
                if current is None:
                    with self._play_lock:
                        if self._play_queue:
                            current = self._play_queue.popleft()
                            # Start pacing this reply chunk from "now" instead of
                            # trying to catch up from a long idle interval.
                            pace_t0 = time.monotonic()
                            pace_frame_no = 0

                if current is not None:
                    frames_np, pcm48, pos, meta = current
                    if pos >= frames_np.shape[0]:
                        current = None
                        continue
                    if pos == 0 and self.debug:
                        now = time.perf_counter()
                        enqueue_ts = float(meta.get("enqueue_wall_ts", 0.0) or 0.0)
                        push_ts = float(meta.get("push_wall_ts", 0.0) or 0.0)
                        chunk_age = max(now - enqueue_ts, 0.0) if enqueue_ts > 0.0 else -1.0
                        push_to_play = max(now - push_ts, 0.0) if push_ts > 0.0 else -1.0
                        print(
                            f"[DEBUG/fmp4] play_start | chunk={meta.get('chunk_index', '?')} "
                            f"chunk_age={chunk_age:.3f}s push_to_play={push_to_play:.3f}s "
                            f"play_q={len(self._play_queue)} av_q={self.av_q.qsize()}"
                        )
                    vf = frames_np[pos]
                    a_start = pos * spf_a48
                    a_end = a_start + spf_a48
                    ap = pcm48[a_start:a_end]
                    if ap.shape[0] < spf_a48:
                        ap = np.concatenate(
                            [ap, np.zeros(spf_a48 - ap.shape[0], dtype=np.int16)]
                        )
                    current[2] = pos + 1
                    self.reply_ticks_emitted += 1
                    if self._last_mode != "reply" and self.debug:
                        now = time.monotonic()
                        print(
                            f"[DEBUG/fmp4] mode -> reply | "
                            f"idle_for={now - self._last_mode_change_t:.3f}s "
                            f"play_q={len(self._play_queue)} av_q={self.av_q.qsize()}"
                        )
                        self._last_mode_change_t = now
                    self._last_mode = "reply"
                else:
                    vf = self.idle_frame_uint8
                    ap = silence
                    self.idle_ticks_emitted += 1
                    if self._last_mode != "idle" and self.debug:
                        now = time.monotonic()
                        print(
                            f"[DEBUG/fmp4] mode -> idle | "
                            f"reply_for={now - self._last_mode_change_t:.3f}s "
                            f"play_q={len(self._play_queue)} av_q={self.av_q.qsize()}"
                        )
                        self._last_mode_change_t = now
                    self._last_mode = "idle"

                ap1 = ap[:AUDIO_SAMPLES_PER_PKT].copy()
                ap2 = ap[AUDIO_SAMPLES_PER_PKT : AUDIO_SAMPLES_PER_PKT * 2].copy()
                self._push_tick((vf, ap1, ap2))

                self.frames_emitted += 1
                pace_frame_no += 1
                target = pace_t0 + pace_frame_no / self.fps
                dt = target - time.monotonic()
                if dt > 0:
                    await asyncio.sleep(dt)
        finally:
            # EOS sentinel for the pump task.
            with contextlib.suppress(asyncio.QueueFull):
                self.av_q.put_nowait(None)

    def _push_tick(self, tick: tuple) -> None:
        self.max_av_q = max(self.max_av_q, self.av_q.qsize())
        if self.av_q.full():
            try:
                self.av_q.get_nowait()
                self.dropped_ticks += 1
            except asyncio.QueueEmpty:
                pass
        try:
            self.av_q.put_nowait(tick)
            self.max_av_q = max(self.max_av_q, self.av_q.qsize())
        except asyncio.QueueFull:
            self.dropped_ticks += 1


class _AsyncByteSink:
    """File-like sink that forwards muxed bytes into an asyncio queue.

    PyAV's container.mux() runs synchronously and writes to its file-like
    sink. We bounce those writes into the asyncio loop so the aiohttp
    response writer (also on the loop) can drain them. Identical pattern
    to realtime_server.py's _AsyncByteSink.
    """

    def __init__(self, loop: asyncio.AbstractEventLoop, queue: asyncio.Queue):
        self.loop = loop
        self.q = queue
        self._closed = False

    def write(self, data) -> int:
        payload = bytes(data)
        self.loop.call_soon_threadsafe(self.q.put_nowait, payload)
        return len(payload)

    def flush(self) -> None:
        return None

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.loop.call_soon_threadsafe(self.q.put_nowait, None)


def _add_video_stream(container, fps: int, *, log_codec: bool = True):
    """Add an H.264 video stream — NVENC if available, libx264 fallback.

    Same defensive int(round(float())) coercion as realtime_server's
    FMP4Server._add_video_stream so a stray --fps float can't break
    PyAV's to_avrational.
    """
    fps = int(round(float(fps)))
    try:
        vstream = container.add_stream("h264_nvenc", rate=fps)
        vstream.options = {
            "preset": "p1",
            "tune": "ll",
            "rc": "cbr",
            "b": "4M",
            "g": str(fps),
            "bf": "0",
            "delay": "0",
        }
        codec_name = "h264_nvenc"
    except Exception:
        vstream = container.add_stream("libx264", rate=fps)
        vstream.options = {
            "preset": "ultrafast",
            "tune": "zerolatency",
            "g": str(fps),
            "bf": "0",
        }
        codec_name = "libx264"

    vstream.width = 512
    vstream.height = 512
    vstream.pix_fmt = "yuv420p"
    if log_codec:
        print(f"[launch/fmp4] video encoder = {codec_name}")
    return vstream


def _warm_fmp4_muxer(av_module, fps: int, *, warm_seconds: float = 1.5) -> None:
    """Prime PyAV's frame conversion + encoder path before a real viewer starts."""
    from av.audio.frame import AudioFrame
    from av.video.frame import VideoFrame

    sink = io.BytesIO()
    container = av_module.open(
        sink,
        mode="w",
        format="mp4",
        options={
            "movflags": "frag_keyframe+empty_moov+default_base_moof+omit_tfhd_offset",
            "frag_duration": "40000",
        },
    )
    try:
        vstream = _add_video_stream(container, fps, log_codec=False)
        astream = container.add_stream("aac", rate=AUDIO_SR_OUT)
        astream.layout = "mono"
        astream.bit_rate = 96_000

        warm_v = np.zeros((vstream.height, vstream.width, 3), dtype=np.uint8)
        warm_a = np.zeros(AUDIO_SAMPLES_PER_PKT, dtype=np.int16)
        total_ticks = max(1, int(round(fps * warm_seconds)))
        v_pts = 0
        a_pts = 0
        for _ in range(total_ticks):
            vframe = VideoFrame.from_ndarray(warm_v, format="rgb24")
            vframe.pts = v_pts
            vframe.time_base = fractions.Fraction(1, fps)
            v_pts += 1
            for packet in vstream.encode(vframe):
                container.mux(packet)

            for _ in range(2):
                aframe = AudioFrame.from_ndarray(
                    warm_a.reshape(1, -1), format="s16", layout="mono"
                )
                aframe.sample_rate = AUDIO_SR_OUT
                aframe.pts = a_pts
                aframe.time_base = fractions.Fraction(1, AUDIO_SR_OUT)
                a_pts += AUDIO_SAMPLES_PER_PKT
                for packet in astream.encode(aframe):
                    container.mux(packet)

        for packet in vstream.encode(None):
            container.mux(packet)
        for packet in astream.encode(None):
            container.mux(packet)
    finally:
        container.close()


@dataclass
class MoshiAvatarServerState:
    model_type: str
    mimi: MimiModel
    text_tokenizer: tp.Any
    lm_gen: LMGen
    lock: asyncio.Lock
    output_handler: tp.Any

    def __init__(
        self,
        model_type: str,
        mimi: MimiModel,
        text_tokenizer,
        lm: LMModel,
        cfg_coef: float,
        device: str | torch.device,
        output_handler=None,
        user_audio_handler=None,
        max_sentences: int = 1,
        max_text_tokens: int = 40,
        send_audio_to_client: bool = True,
        transcript_store: AVSegmentStore | None = None,
        debug_session: bool = False,
        **kwargs,
    ):
        self.model_type = model_type
        self.mimi = mimi
        self.text_tokenizer = text_tokenizer
        condition_tensors = get_condition_tensors(model_type, lm, batch_size=1, cfg_coef=cfg_coef)
        self.max_sentences = max_sentences
        self.max_text_tokens = max_text_tokens
        self.lm_gen = LMGen(
            lm,
            cfg_coef=cfg_coef,
            condition_tensors=condition_tensors,
            on_text_hook=self._on_text_hook,
            **kwargs,
        )

        self.device = device
        self.output_handler = output_handler
        self.user_audio_handler = user_audio_handler
        self.send_audio_to_client = send_audio_to_client
        self.transcript_store = transcript_store
        self.debug_session = debug_session
        self.frame_size = int(self.mimi.sample_rate / self.mimi.frame_rate)
        self.lock = asyncio.Lock()
        self._session_t0 = 0.0
        self._mic_chunks_in = 0
        self._mic_samples_in = 0
        self._gen_pcm_samples = 0

        self.mimi.streaming_forever(1)
        self.lm_gen.streaming_forever(1)

        self._should_close = False
        self._output_text_tokens = 0
        self._output_sentences = 0
        self._force_eos = False
        self._finalize_called = False

    def _debug_log(self, msg: str) -> None:
        if self.debug_session:
            print(f"[DEBUG/moshi] {msg}")

    def generation_rate(self) -> float:
        if self._session_t0 <= 0.0:
            return 0.0
        elapsed = time.perf_counter() - self._session_t0
        if elapsed < 0.1:
            return 0.0
        gen_sec = self._gen_pcm_samples / float(self.mimi.sample_rate)
        return gen_sec / elapsed

    def debug_state(self) -> dict:
        elapsed = 0.0
        if self._session_t0 > 0.0:
            elapsed = max(0.0, time.perf_counter() - self._session_t0)
        generated_sec = self._gen_pcm_samples / float(self.mimi.sample_rate)
        return {
            "mic_chunks_in": self._mic_chunks_in,
            "mic_samples_in": self._mic_samples_in,
            "reply_generated_sec": round(generated_sec, 3),
            "reply_elapsed_sec": round(elapsed, 3),
            "generation_rate_x": round(self.generation_rate(), 3),
            "reply_sentences": self._output_sentences,
            "reply_tokens": self._output_text_tokens,
        }

    def _reset_connection(self):
        import time as _time

        self._should_close = False
        self._output_text_tokens = 0
        self._output_sentences = 0
        self._force_eos = False
        self._finalize_called = False
        self._session_t0 = _time.perf_counter()
        self._mic_chunks_in = 0
        self._mic_samples_in = 0
        self._gen_pcm_samples = 0
        if self.transcript_store is not None:
            self.transcript_store.start_transcript_turn()
        # Also reset the IMTalker render state so each new Moshi turn starts
        # from a clean fm_stream_state. The fMP4 stream itself stays open
        # across turns — only the renderer's per-reply temporal context is
        # cleared.
        if self.output_handler is not None and hasattr(self.output_handler, "__self__"):
            reset = getattr(self.output_handler.__self__, "reset_reply", None)
            if callable(reset):
                reset()
        log(
            "info",
            f"session reset | max_sentences={self.max_sentences} "
            f"max_text_tokens={self.max_text_tokens} "
            f"frame_size={self.frame_size} mimi_sr={self.mimi.sample_rate}",
        )

    async def _finalize_pending_reply(self):
        if self._finalize_called:
            return
        self._finalize_called = True
        if self.output_handler is not None and hasattr(self.output_handler, "__self__"):
            finalize = getattr(self.output_handler.__self__, "finalize_pending_reply", None)
            if finalize is not None:
                maybe_awaitable = finalize()
                if inspect.isawaitable(maybe_awaitable):
                    await maybe_awaitable

    def _on_text_hook(self, text_token: torch.Tensor) -> None:
        if text_token.numel() == 0:
            return
        if self._force_eos:
            text_token.fill_(self.text_tokenizer.eos_id())

    def warmup(self):
        for _ in range(4):
            chunk = torch.zeros(1, 1, self.frame_size, dtype=torch.float32, device=self.device)
            codes = self.mimi.encode(chunk)
            for c in range(codes.shape[-1]):
                tokens = self.lm_gen.step(codes[:, :, c : c + 1])
                if tokens is None:
                    continue
                _ = self.mimi.decode(tokens[:, 1:])
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    async def _safe_send_ws_bytes(
        self,
        ws: web.WebSocketResponse,
        payload: bytes,
    ) -> bool:
        if ws.closed:
            self._should_close = True
            return False
        try:
            await ws.send_bytes(payload)
            return True
        except (
            aiohttp.client_exceptions.ClientConnectionResetError,
            ConnectionResetError,
            RuntimeError,
        ) as exc:
            self._should_close = True
            log("info", f"client disconnected during send ({exc})")
            return False

    async def decode_and_send(
        self,
        tokens: torch.Tensor,
        ws: web.WebSocketResponse,
        opus_writer: sphn.OpusStreamWriter,
    ):
        assert tokens.shape[1] == self.lm_gen.lm_model.dep_q + 1

        if self._should_close:
            return

        text_token_id = int(tokens[0, 0, 0].item())
        eos_id = self.text_tokenizer.eos_id()
        token_limit = False
        sentence_limit = False
        if text_token_id not in (0, 3, eos_id):
            piece = self.text_tokenizer.id_to_piece(text_token_id).replace("▁", " ")
            self._output_text_tokens += 1
            self._output_sentences += piece.count(".") + piece.count("!") + piece.count("?")
            token_limit = (
                self.max_text_tokens > 0 and self._output_text_tokens >= self.max_text_tokens
            )
            sentence_limit = (
                self.max_sentences > 0 and self._output_sentences >= self.max_sentences
            )

        main_latents = self.mimi.decode_latent(tokens[:, 1:])
        main_pcm = self.mimi.decode(tokens[:, 1:])
        self._gen_pcm_samples += int(main_pcm.shape[-1])
        if self.output_handler is not None:
            maybe_awaitable = self.output_handler(tokens, main_pcm, main_latents)
            if inspect.isawaitable(maybe_awaitable):
                await maybe_awaitable
        main_pcm = main_pcm.detach().cpu()
        if self.send_audio_to_client:
            opus_bytes = opus_writer.append_pcm(main_pcm[0, 0].numpy())
            if len(opus_bytes) > 0:
                if not await self._safe_send_ws_bytes(ws, b"\x01" + opus_bytes):
                    return
        if text_token_id not in (0, 3, eos_id):
            text_piece = self.text_tokenizer.id_to_piece(text_token_id).replace("▁", " ")
            if self.transcript_store is not None:
                self.transcript_store.append_assistant_text(text_piece)
            msg = b"\x02" + bytes(text_piece, encoding="utf8")
            log("info", f"text token '{text_piece}'")
            if not await self._safe_send_ws_bytes(ws, msg):
                return

        if token_limit or sentence_limit:
            self._should_close = True
            self._force_eos = True
            reason = []
            if token_limit:
                reason.append(
                    f"token_limit({self._output_text_tokens}/{self.max_text_tokens})"
                )
            if sentence_limit:
                reason.append(
                    f"sentence_limit({self._output_sentences}/{self.max_sentences})"
                )
            log(
                "info",
                f"reply limit reached - {' & '.join(reason)} | "
                f"closing session (raise --max_sentences / --max_text_tokens to "
                f"allow longer replies)",
            )
        elif text_token_id == eos_id:
            self._should_close = True
            log(
                "info",
                f"assistant reply ended (eos) | "
                f"sentences={self._output_sentences} tokens={self._output_text_tokens} "
                f"- closing session",
            )
        elif self.debug_session and text_token_id not in (0, 3, eos_id):
            self._debug_log(
                f"step | tokens={self._output_text_tokens}/{self.max_text_tokens} "
                f"sentences={self._output_sentences}/{self.max_sentences}"
            )

    async def recv_loop(
        self,
        ws: web.WebSocketResponse,
        opus_reader: sphn.OpusStreamReader,
        opus_writer: sphn.OpusStreamWriter,
    ):
        all_pcm_data = None
        skip_frames = 1
        keepalive_stop = asyncio.Event()

        async def _keepalive_sender():
            while not keepalive_stop.is_set():
                try:
                    await asyncio.wait_for(keepalive_stop.wait(), timeout=5.0)
                    break
                except asyncio.TimeoutError:
                    pass
                if ws.closed:
                    break
                try:
                    await ws.send_bytes(b"\x06")
                except Exception:
                    break

        keepalive_task = asyncio.create_task(_keepalive_sender())
        try:
            async for message in ws:
                if message.type == aiohttp.WSMsgType.ERROR:
                    log("error", f"{ws.exception()}")
                    break
                if message.type == aiohttp.WSMsgType.CLOSED:
                    break
                if message.type != aiohttp.WSMsgType.BINARY:
                    log("error", f"unexpected message type {message.type}")
                    continue
                payload = message.data
                if not isinstance(payload, bytes):
                    log("error", f"unsupported message type {type(payload)}")
                    continue
                if len(payload) == 0:
                    log("warning", "empty message")
                    continue
                kind = payload[0]
                if kind != 1:
                    log("warning", f"unknown message kind {kind}")
                    continue
                pcm = opus_reader.append_bytes(payload[1:])
                if pcm is None or pcm.shape[-1] == 0:
                    continue
                self._mic_chunks_in += 1
                self._mic_samples_in += int(pcm.shape[-1])
                if self.debug_session and self._mic_chunks_in % 25 == 0:
                    self._debug_log(
                        f"recv | mic_chunks={self._mic_chunks_in} "
                        f"mic_samples={self._mic_samples_in} "
                        f"pending_pcm={(0 if all_pcm_data is None else all_pcm_data.shape[-1])}"
                    )
                if all_pcm_data is None:
                    all_pcm_data = pcm
                else:
                    all_pcm_data = np.concatenate((all_pcm_data, pcm))
                while all_pcm_data.shape[-1] >= self.frame_size:
                    chunk = all_pcm_data[: self.frame_size]
                    all_pcm_data = all_pcm_data[self.frame_size :]
                    chunk_t = torch.from_numpy(chunk).to(device=self.device)[None, None]
                    codes = self.mimi.encode(chunk_t)
                    if skip_frames:
                        self.mimi.reset_streaming()
                        skip_frames -= 1
                    for c in range(codes.shape[-1]):
                        tokens = self.lm_gen.step(codes[:, :, c : c + 1])
                        if tokens is None:
                            continue
                        await self.decode_and_send(tokens, ws, opus_writer)
                    if self._should_close:
                        await self._finalize_pending_reply()
                        import time as _time

                        elapsed = _time.perf_counter() - self._session_t0
                        log(
                            "info",
                            f"session complete - closing websocket | "
                            f"elapsed={elapsed:.2f}s mic_chunks={self._mic_chunks_in} "
                            f"mic_samples={self._mic_samples_in} "
                            f"reply_sentences={self._output_sentences} "
                            f"reply_tokens={self._output_text_tokens}",
                        )
                        return
                    # Give the producer and HTTP stream pump a chance to run
                    # between Moshi frames instead of monopolizing the loop.
                    await asyncio.sleep(0)
        finally:
            keepalive_stop.set()
            keepalive_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await keepalive_task
            await self._finalize_pending_reply()
            import time as _time

            elapsed = _time.perf_counter() - self._session_t0
            close_code = getattr(ws, "close_code", None)
            log(
                "info",
                f"connection closed | elapsed={elapsed:.2f}s "
                f"mic_chunks={self._mic_chunks_in} mic_samples={self._mic_samples_in} "
                f"reply_sentences={self._output_sentences} "
                f"reply_tokens={self._output_text_tokens} close_code={close_code}",
            )
            if self.debug_session and self.output_handler is not None:
                owner = getattr(self.output_handler, "__self__", None)
                if owner is not None and hasattr(owner, "debug_state"):
                    print(f"[DEBUG/moshi] session_summary | {owner.debug_state()}")

    async def handle_chat(self, request):
        ws = web.WebSocketResponse()
        try:
            await ws.prepare(request)
        except (
            aiohttp.client_exceptions.ClientConnectionResetError,
            ConnectionResetError,
            RuntimeError,
        ) as exc:
            log("info", f"client disconnected before websocket ready ({exc})")
            return web.Response(status=204)

        log("info", "accepted connection")

        async with self.lock:
            opus_writer = sphn.OpusStreamWriter(self.mimi.sample_rate)
            opus_reader = sphn.OpusStreamReader(self.mimi.sample_rate)
            self.mimi.reset_streaming()
            self.lm_gen.reset_streaming()
            self._reset_connection()
            if not await self._safe_send_ws_bytes(ws, b"\x00"):
                return ws
            await self.recv_loop(ws, opus_reader, opus_writer)
        log("info", "done with connection")
        return ws


_VIEWER_HTML = """\
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Moshi + IMTalker</title>
  <style>
    :root { color-scheme: dark; }
    body {
      margin: 0;
      min-height: 100vh;
      display: grid;
      place-items: center;
      background:
        radial-gradient(circle at top, rgba(44,108,138,0.35), transparent 45%),
        linear-gradient(180deg, #07090c 0%, #11161d 100%);
      font-family: "IBM Plex Mono", monospace;
      color: #d8e2ea;
    }
    .shell {
      width: min(92vw, 980px);
      display: grid;
      gap: 16px;
      justify-items: center;
    }
    .stage {
      position: relative;
      width: min(92vw, 720px);
      aspect-ratio: 1 / 1;
      border: 1px solid rgba(216,226,234,0.18);
      border-radius: 18px;
      background: #000;
      overflow: hidden;
      box-shadow: 0 30px 60px rgba(0,0,0,0.45);
    }
    video {
      width: 100%;
      height: 100%;
      object-fit: contain;
      background: #000;
    }
    .overlay {
      position: absolute;
      left: 16px;
      right: 16px;
      bottom: 16px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 12px 14px;
      border-radius: 12px;
      background: rgba(0, 0, 0, 0.42);
      backdrop-filter: blur(10px);
      font-size: 13px;
    }
    .controls { display: flex; gap: 12px; align-items: center; }
    button {
      border: 1px solid rgba(216,226,234,0.35);
      border-radius: 999px;
      background: rgba(255,255,255,0.05);
      color: #ecf2f7;
      padding: 11px 18px;
      font: inherit;
      cursor: pointer;
    }
    button:hover { background: rgba(255,255,255,0.12); }
    .hidden-controller {
      position: absolute;
      width: 1px;
      height: 1px;
      left: -9999px;
      top: -9999px;
      opacity: 0;
      pointer-events: none;
    }
    .hint {
      font-size: 12px;
      color: #9caab5;
      text-align: center;
      line-height: 1.5;
      max-width: 720px;
    }
    .transcript {
      width: min(92vw, 720px);
      min-height: 72px;
      padding: 14px 16px;
      border-radius: 14px;
      border: 1px solid rgba(216,226,234,0.16);
      background: rgba(255,255,255,0.04);
      color: #ecf2f7;
      white-space: pre-wrap;
      line-height: 1.5;
    }
    .transcript-label {
      display: block;
      margin-bottom: 8px;
      font-size: 11px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: #9caab5;
    }
    .badge {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 8px 12px;
      border-radius: 999px;
      border: 1px solid rgba(216,226,234,0.18);
      background: rgba(255,255,255,0.04);
      font-size: 12px;
    }
    .dot {
      width: 8px;
      height: 8px;
      border-radius: 999px;
      background: #4fd38f;
      box-shadow: 0 0 12px rgba(79,211,143,0.75);
    }
  </style>
</head>
<body>
  <div class="shell">
    <div class="badge"><span class="dot"></span><span id="mode">Idle</span></div>
    <div class="stage">
      <video id="avatar" playsinline autoplay muted></video>
      <div class="overlay">
        <div id="status">Press Start. Audio unlocks on click.</div>
        <div class="controls">
          <button id="startBtn">Start</button>
          <button id="resetBtn">Reset</button>
        </div>
      </div>
    </div>
    <div class="transcript" id="transcriptBox">
      <span class="transcript-label">Assistant Transcript</span>
      <span id="transcriptText">Assistant text will appear here as Moshi responds.</span>
    </div>
    <p class="hint">
      Single muxed fMP4 over chunked HTTP, played via MediaSource Extensions.
      Video and audio share one PTS timeline so lip-sync is enforced by the
      container, not the page. The producer fills idle gaps with the
      reference frame so the timeline never stalls between Moshi replies.
    </p>
    <iframe id="controller" class="hidden-controller" allow="microphone; autoplay"></iframe>
  </div>
  <script>
    const avatar = document.getElementById('avatar');
    const startBtn = document.getElementById('startBtn');
    const resetBtn = document.getElementById('resetBtn');
    const statusEl = document.getElementById('status');
    const modeEl = document.getElementById('mode');
    const controller = document.getElementById('controller');
    const transcriptTextEl = document.getElementById('transcriptText');
    const DEBUG_VIEW = new URLSearchParams(window.location.search).get('debug') === '1';

    const MIME_CANDIDATES = [
      'video/mp4; codecs="avc1.640015, mp4a.40.2"',
      'video/mp4; codecs="avc1.4D401E, mp4a.40.2"',
      'video/mp4; codecs="avc1.42E01E, mp4a.40.2"',
    ];
    const MIME = (("MediaSource" in window)
      ? MIME_CANDIDATES.find(m => MediaSource.isTypeSupported(m))
      : null) || "";

    let controllerStarted = false;
    let streamStarted = false;
    let lastTranscriptTurnId = -1;
    let lastTranscriptVersion = -1;
    let abortCtl = null;
    let currentMSUrl = null;
    let firstFrameLogged = false;
    let lastDebugSnapshot = '';

    function getBufferedEnd() {
      try {
        if (!avatar.buffered || avatar.buffered.length === 0) return 0;
        return avatar.buffered.end(avatar.buffered.length - 1);
      } catch (_) {
        return 0;
      }
    }
    function logAvatarState(label) {
      if (!DEBUG_VIEW) return;
      console.log('[avatar-debug]', label, {
        currentTime: Number(avatar.currentTime || 0).toFixed(3),
        bufferedEnd: Number(getBufferedEnd()).toFixed(3),
        readyState: avatar.readyState,
        networkState: avatar.networkState,
        paused: avatar.paused,
      });
    }

    function setTranscript(text) {
      transcriptTextEl.textContent = text || 'Assistant text will appear here as Moshi responds.';
    }
    function setStatus(text, mode) {
      statusEl.textContent = text;
      if (mode) modeEl.textContent = mode;
    }

    async function startStream() {
      if (streamStarted) return;
      if (!MIME) {
        setStatus('MediaSource / H264+AAC not supported in this browser.', 'Error');
        return;
      }
      streamStarted = true;
      firstFrameLogged = false;

      if (abortCtl) abortCtl.abort();
      abortCtl = new AbortController();

      const ms = new MediaSource();
      if (currentMSUrl) URL.revokeObjectURL(currentMSUrl);
      currentMSUrl = URL.createObjectURL(ms);
      avatar.src = currentMSUrl;
      const t0 = performance.now();

      ms.addEventListener('sourceopen', async () => {
        let sb;
        try { sb = ms.addSourceBuffer(MIME); }
        catch (e) { setStatus('addSourceBuffer failed: ' + e.message, 'Error'); return; }
        sb.mode = 'sequence';

        const queue = [];
        let appending = false;
        let playAttempted = false;
        function pump() {
          if (appending || !queue.length || sb.updating) return;
          appending = true;
          try { sb.appendBuffer(queue.shift()); }
          catch (e) { console.warn('appendBuffer error', e); appending = false; }
        }
        sb.addEventListener('updateend', () => {
          appending = false;
          logAvatarState('sourcebuffer-updateend');
          if (!playAttempted) {
            playAttempted = true;
            avatar.play().catch(e => console.warn('play blocked', e));
          }
          pump();
        });

        avatar.addEventListener('playing', () => {
          if (firstFrameLogged) return;
          firstFrameLogged = true;
          console.log('[avatar] first frame in', (performance.now() - t0).toFixed(0), 'ms');
          setStatus('Streaming. Speak any time.', 'Listening');
        }, { once: true });
        ['waiting', 'stalled', 'canplay', 'pause', 'seeking', 'ended', 'error'].forEach(evt => {
          avatar.addEventListener(evt, () => logAvatarState(evt));
        });

        try {
          const resp = await fetch('/stream.mp4?t=' + Date.now(), { signal: abortCtl.signal });
          if (!resp.ok) {
            setStatus('Stream HTTP ' + resp.status, 'Error');
            streamStarted = false;
            return;
          }
          const reader = resp.body.getReader();
          while (true) {
            const { value, done } = await reader.read();
            if (done) break;
            queue.push(value);
            pump();
          }
          const finish = () => { try { if (ms.readyState === 'open') ms.endOfStream(); } catch (_) {} };
          if (sb.updating || queue.length) sb.addEventListener('updateend', finish, { once: true });
          else finish();
        } catch (e) {
          if (e.name !== 'AbortError') {
            setStatus('Stream error: ' + e.message, 'Error');
            streamStarted = false;
          }
        }
      }, { once: true });
    }

    function startController() {
      if (controllerStarted) return;
      controllerStarted = true;
      controller.src = '/moshi#/?embed=1&ts=' + Date.now();
    }

    async function pollState() {
      try {
        const res = await fetch('/api/stream_state?ts=' + Date.now(), { cache: 'no-store' });
        if (!res.ok) return;
        const meta = await res.json();
        if (meta.transcript_turn_id !== lastTranscriptTurnId
            || meta.transcript_version !== lastTranscriptVersion) {
          lastTranscriptTurnId = meta.transcript_turn_id;
          lastTranscriptVersion = meta.transcript_version;
          setTranscript(meta.assistant_text || '');
        }
        if (DEBUG_VIEW && (meta.av_debug || meta.imtalker_debug)) {
          const snapshot = JSON.stringify({
            av: meta.av_debug || null,
            imtalker: meta.imtalker_debug || null,
          });
          if (snapshot !== lastDebugSnapshot) {
            lastDebugSnapshot = snapshot;
            console.log('[avatar-debug] stream_state', meta);
          }
        }
      } catch (_) {}
    }

    async function primeMicrophoneFromParentGesture() {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        stream.getTracks().forEach(t => t.stop());
      } catch (e) {
        console.warn('[avatar] mic prime failed:', e);
      }
    }

    startBtn.addEventListener('click', async () => {
      await primeMicrophoneFromParentGesture();
      avatar.muted = false;
      await startStream();
      startController();
    });

    resetBtn.addEventListener('click', () => {
      if (abortCtl) abortCtl.abort();
      streamStarted = false;
      controllerStarted = false;
      controller.src = 'about:blank';
      avatar.pause();
      if (currentMSUrl) URL.revokeObjectURL(currentMSUrl);
      currentMSUrl = null;
      avatar.removeAttribute('src');
      avatar.load();
      setTranscript('');
      setStatus('Reset. Press Start to begin again.', 'Idle');
    });

    controller.addEventListener('load', () => {
      try { controller.contentWindow?.postMessage({ type: 'moshi-mic-primed' }, '*'); }
      catch (_) {}
    });

    setInterval(pollState, 500);
    pollState();
  </script>
</body>
</html>
"""


def build_imtalker_session(opt) -> LiveMoshiIMTalkerSession:
    print("[launch] Loading IMTalker session...")
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
    if getattr(opt, "skip_warmup", False):
        print("[launch] Skipping IMTalker warmup (--skip_warmup).")
    else:
        session.warmup()
    print("[launch] IMTalker session ready.")
    return session


# ====================================================================== #
# fMP4 stream handler
# ====================================================================== #
async def serve_fmp4_stream(
    request: web.Request,
    session: LiveMoshiIMTalkerSession,
    av_lock: asyncio.Lock,
) -> web.StreamResponse:
    """One muxed fMP4 stream per browser viewer.

    Tears down any prior viewer's session, opens a fresh PyAV mp4 muxer
    backed by an async byte sink, and pumps the FMP4StreamSession's av_q
    into the muxer until the client disconnects. The IMTalker render
    worker pushes reply chunks into the FMP4StreamSession in parallel.
    """
    try:
        import av
        from av.audio.frame import AudioFrame
        from av.video.frame import VideoFrame
    except Exception as e:  # pragma: no cover
        return web.Response(
            status=500,
            text=(
                "PyAV is not installed in this environment. "
                "Install with: pip install av\n"
                f"Error: {e}"
            ),
        )

    fps = int(round(float(getattr(session.opt, "fps", 25))))
    loop = asyncio.get_running_loop()

    t_warm = time.perf_counter()
    if getattr(session.opt, "skip_warmup", False):
        if bool(getattr(session.opt, "debug_session", False)):
            print("[launch/fmp4] skipping muxer warmup (--skip_warmup)")
    else:
        _warm_fmp4_muxer(av, fps)
        if bool(getattr(session.opt, "debug_session", False)):
            print(f"[launch/fmp4] warmup={time.perf_counter() - t_warm:.3f}s")

    # Tear down any prior viewer session, then publish the new one to the
    # IMTalker render worker. One viewer at a time.
    async with av_lock:
        if session.av_session is not None:
            print("[launch/fmp4] tearing down previous viewer session")
            try:
                session.av_session.stop()
            except Exception:
                pass
            session.av_session = None

        # Reset render-side state so the next reply starts clean for this
        # viewer (the prior viewer may have left mid-turn).
        session.reset_reply()

        av_session = FMP4StreamSession(
            loop=loop,
            idle_frame_uint8=session.idle_frame_uint8,
            fps=fps,
            debug=bool(getattr(session.opt, "debug_session", False)),
        )
        av_session.start()
        session.av_session = av_session

    # Build the muxer + async sink for this viewer.
    out_q: asyncio.Queue = asyncio.Queue()
    sink = _AsyncByteSink(loop, out_q)
    try:
        container = av.open(
            sink,
            mode="w",
            format="mp4",
            options={
                "movflags": "frag_keyframe+empty_moov+default_base_moof+omit_tfhd_offset",
                "frag_duration": "40000",
            },
        )
        vstream = _add_video_stream(container, fps)
        astream = container.add_stream("aac", rate=AUDIO_SR_OUT)
        astream.layout = "mono"
        astream.bit_rate = 96_000
    except Exception:
        if session.av_session is av_session:
            av_session.stop()
            session.av_session = None
        raise

    resp = web.StreamResponse(
        status=200,
        headers={
            "Content-Type": "video/mp4",
            "Cache-Control": "no-store",
            "X-Accel-Buffering": "no",
        },
    )
    await resp.prepare(request)

    async def pump():
        v_pts = 0
        a_pts = 0
        frames_sent = 0

        def _encode_tick(vf_np, ap1, ap2):
            nonlocal v_pts, a_pts, frames_sent
            vframe = VideoFrame.from_ndarray(vf_np, format="rgb24")
            vframe.pts = v_pts
            vframe.time_base = fractions.Fraction(1, fps)
            v_pts += 1
            frames_sent += 1
            for p in vstream.encode(vframe):
                container.mux(p)

            for ap in (ap1, ap2):
                if ap.shape[0] < AUDIO_SAMPLES_PER_PKT:
                    ap = np.concatenate(
                        [ap, np.zeros(AUDIO_SAMPLES_PER_PKT - ap.shape[0], dtype=np.int16)]
                    )
                aframe = AudioFrame.from_ndarray(
                    ap.reshape(1, -1), format="s16", layout="mono"
                )
                aframe.sample_rate = AUDIO_SR_OUT
                aframe.pts = a_pts
                aframe.time_base = fractions.Fraction(1, AUDIO_SR_OUT)
                a_pts += AUDIO_SAMPLES_PER_PKT
                for p in astream.encode(aframe):
                    container.mux(p)

        try:
            while True:
                first_tick = await av_session.av_q.get()
                if first_tick is None:
                    break
                pending_ticks = [first_tick]
                end_stream = False
                while True:
                    try:
                        tick = av_session.av_q.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                    if tick is None:
                        end_stream = True
                        break
                    pending_ticks.append(tick)

                for vf_np, ap1, ap2 in pending_ticks:
                    _encode_tick(vf_np, ap1, ap2)
                if end_stream:
                    break
        except asyncio.CancelledError:
            raise
        finally:
            with contextlib.suppress(Exception):
                for p in vstream.encode(None):
                    container.mux(p)
                for p in astream.encode(None):
                    container.mux(p)
                container.close()
            sink.close()
            print(f"[launch/fmp4] stream end, frames_sent={frames_sent}")

    pumper = asyncio.create_task(pump())
    try:
        while True:
            chunk = await out_q.get()
            if chunk is None:
                break
            await resp.write(chunk)
    except (asyncio.CancelledError, ConnectionResetError):
        print("[launch/fmp4] client disconnected")
    finally:
        pumper.cancel()
        with contextlib.suppress(Exception):
            await pumper
        async with av_lock:
            if session.av_session is av_session:
                av_session.stop()
                session.av_session = None
    return resp


def build_moshi_state(opt, session: LiveMoshiIMTalkerSession, transcript: TranscriptStore):
    seed_all(42424242)

    print("[launch] Loading Moshi checkpoint...")
    checkpoint_kwargs = {}
    if opt.config_path:
        checkpoint_kwargs["config_path"] = opt.config_path
    if opt.lora_weight:
        checkpoint_kwargs["lora_weights"] = opt.lora_weight
    checkpoint_info = loaders.CheckpointInfo.from_hf_repo(
        opt.hf_repo,
        opt.moshi_weight,
        opt.mimi_weight,
        opt.tokenizer,
        **checkpoint_kwargs,
    )

    text_tokenizer = checkpoint_info.get_text_tokenizer()

    print("[launch] Loading Moshi Mimi...")
    mimi = checkpoint_info.get_mimi(device=opt.device)
    print("[launch] Loading Moshi LM...")
    lm = checkpoint_info.get_moshi(device=opt.device, dtype=opt.dtype, fuse_lora=opt.fuse_lora)
    lm_gen_kwargs = dict(checkpoint_info.lm_gen_config)
    lm_gen_kwargs["top_k_text"] = opt.text_topk
    lm_gen_kwargs["temp_text"] = opt.text_temperature

    state = MoshiAvatarServerState(
        checkpoint_info.model_type,
        mimi,
        text_tokenizer,
        lm,
        cfg_coef=1.0,
        device=opt.device,
        output_handler=session.handle_moshi_output,
        user_audio_handler=session.handle_user_audio,
        max_sentences=opt.max_sentences,
        max_text_tokens=opt.max_text_tokens,
        send_audio_to_client=False,
        transcript_store=transcript,
        debug_session=opt.debug_session,
        **lm_gen_kwargs,
    )
    print("[launch] Warming up Moshi...")
    if getattr(opt, "skip_warmup", False):
        print("[launch] Skipping Moshi warmup (--skip_warmup).")
    else:
        state.warmup()
    return state


def main():
    opt = LaunchOptions().parse()
    opt.rank = opt.device

    if opt.no_moshi_cuda_graph:
        os.environ.setdefault("NO_CUDA_GRAPH", "1")
        os.environ.setdefault("NO_TORCH_COMPILE", "1")

    transcript = TranscriptStore()
    session = build_imtalker_session(opt)
    state = build_moshi_state(opt, session, transcript)

    # Serializes viewer connect/disconnect so two near-simultaneous opens
    # don't trample each other's session.av_session.
    av_lock = asyncio.Lock()

    local_dist = Path(opt.moshi_repo) / "client" / "dist"
    if local_dist.exists():
        static_path = str(local_dist)
        print(f"[launch] Using local Moshi client build: {static_path}")
    else:
        dist_tgz = Path(hf_hub_download("kyutai/moshi-artifacts", "dist.tgz"))
        dist = dist_tgz.parent / "dist"
        if not dist.exists():
            with tarfile.open(dist_tgz, "r:gz") as tar:
                tar.extractall(path=dist_tgz.parent)
        static_path = str(dist)
        print(
            "[launch] WARNING: HF Moshi UI is BrowserRouter-only; /moshi will be blank. "
            "Build moshi/client (`npm run build`) and restart to use local dist."
        )

    app = web.Application()
    app.router.add_get("/api/chat", state.handle_chat)
    def _stream_state_response(_request):
        extra = {
            "av_debug": (
                session.av_session.debug_state()
                if session.av_session is not None and hasattr(session.av_session, "debug_state")
                else None
            ),
            "moshi_debug": state.debug_state() if hasattr(state, "debug_state") else None,
            "imtalker_debug": session.debug_state() if hasattr(session, "debug_state") else None,
        }
        return web.json_response(transcript.state(extra=extra))

    app.router.add_get("/api/stream_state", _stream_state_response)
    app.router.add_get(
        "/stream.mp4",
        lambda r: serve_fmp4_stream(r, session, av_lock),
    )

    async def on_shutdown(_app):
        if session.av_session is not None:
            with contextlib.suppress(Exception):
                session.av_session.stop()
            session.av_session = None

    app.on_shutdown.append(on_shutdown)

    async def serve_moshi_index(_request):
        index_path = os.path.join(static_path, "index.html")
        html = Path(index_path).read_text(encoding="utf-8")
        # Stock bundle must be rebuilt from moshi/client after the HashRouter
        # fix (see app.tsx). Bare /moshi still needs a default hash so RR sees /?embed=1.
        bootstrap = """<script>
if (!window.location.hash) {
  var q = window.location.search || '?embed=1';
  if (q.charAt(0) !== '?') q = '?' + q;
  window.location.replace(window.location.pathname + '#/' + q);
}
</script>"""
        if "</head>" in html:
            html = html.replace("</head>", bootstrap + "\n</head>", 1)
        else:
            html = bootstrap + html
        return web.Response(text=html, content_type="text/html")

    app.router.add_get("/", lambda r: web.Response(text=_VIEWER_HTML, content_type="text/html"))
    app.router.add_get("/viewer", lambda r: web.Response(text=_VIEWER_HTML, content_type="text/html"))
    app.router.add_get("/moshi", serve_moshi_index)
    app.router.add_get("/moshi/", serve_moshi_index)
    app.router.add_get("/moshi/index.html", serve_moshi_index)
    app.router.add_static(
        "/moshi/assets",
        path=os.path.join(static_path, "assets"),
        follow_symlinks=True,
        name="moshi_static",
    )
    # Compatibility fallback for client bundles built with base="/" (for example
    # the HF prebuilt dist). Proper local builds should reference /moshi/assets/,
    # but exposing /assets avoids 404/HTML-as-JS failures until the client is rebuilt.
    app.router.add_static(
        "/assets",
        path=os.path.join(static_path, "assets"),
        follow_symlinks=True,
        name="moshi_static_root_assets",
    )
    favicon_path = os.path.join(static_path, "favicon.ico")
    if os.path.exists(favicon_path):
        app.router.add_get("/favicon.ico", lambda r: web.FileResponse(favicon_path))

    print(f"\n[launch] Combined avatar -> http://localhost:{opt.port}/")
    print(f"[launch] Hidden Moshi UI  -> http://localhost:{opt.port}/moshi\n")
    web.run_app(app, host=opt.host, port=opt.port)


if __name__ == "__main__":
    with torch.no_grad():
        main()
