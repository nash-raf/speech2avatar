from __future__ import annotations

import asyncio
import contextlib
import fractions
import threading
import time
from collections import deque

import numpy as np
from aiortc import MediaStreamTrack
from aiortc.mediastreams import MediaStreamError
from av.audio.frame import AudioFrame
from av.video.frame import VideoFrame

AUDIO_SR_OUT = 48000
AUDIO_PTIME = 0.020
AUDIO_SAMPLES_PER_PKT = int(AUDIO_SR_OUT * AUDIO_PTIME)  # 960
AUDIO_SAMPLES_PER_FRAME = AUDIO_SAMPLES_PER_PKT * 2  # 1920 at 25 fps


class _QueuedTrack(MediaStreamTrack):
    kind = ""

    def __init__(self, kind: str, *, maxsize: int = 128):
        super().__init__()
        self.kind = kind
        self.q: asyncio.Queue = asyncio.Queue(maxsize=maxsize)
        self.frames_sent = 0
        self.frames_dropped = 0

    async def recv(self):
        item = await self.q.get()
        if item is None:
            raise MediaStreamError
        self.frames_sent += 1
        return item

    def put_nowait_drop_oldest(self, item) -> None:
        if self.q.full():
            try:
                self.q.get_nowait()
                self.frames_dropped += 1
            except asyncio.QueueEmpty:
                pass
        try:
            self.q.put_nowait(item)
        except asyncio.QueueFull:
            self.frames_dropped += 1

    def close(self) -> None:
        with contextlib.suppress(asyncio.QueueFull):
            self.q.put_nowait(None)
        self.stop()


class WebRTCVideoTrack(_QueuedTrack):
    def __init__(self):
        super().__init__("video", maxsize=64)


class WebRTCAudioTrack(_QueuedTrack):
    def __init__(self):
        super().__init__("audio", maxsize=128)


class WebRTCStreamSession:
    """Per-viewer WebRTC session with paced idle/reply frame production.

    Mirrors the current fMP4 session shape so LiveMoshiIMTalkerSession can
    keep using `session.av_session.push_chunk(...)` unchanged.
    """

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        idle_frame_uint8: np.ndarray,
        fps: int,
        debug: bool = False,
    ):
        self.loop = loop
        self.idle_frame_uint8 = idle_frame_uint8
        self.fps = int(round(float(fps)))
        self.debug = debug
        self.video_track = WebRTCVideoTrack()
        self.audio_track = WebRTCAudioTrack()
        self._play_queue: deque = deque()
        self._play_lock = threading.Lock()
        self._stop = asyncio.Event()
        self._task: asyncio.Task | None = None
        self.frames_emitted = 0
        self.reply_ticks_emitted = 0
        self.idle_ticks_emitted = 0
        self.chunks_pushed = 0
        self.max_play_queue = 0
        self._last_mode = "idle"
        self._last_mode_change_t = time.monotonic()

    def debug_state(self) -> dict:
        return {
            "frames_emitted": self.frames_emitted,
            "reply_ticks_emitted": self.reply_ticks_emitted,
            "idle_ticks_emitted": self.idle_ticks_emitted,
            "chunks_pushed": self.chunks_pushed,
            "play_queue_len": len(self._play_queue),
            "max_play_queue": self.max_play_queue,
            "video_q_size": self.video_track.q.qsize(),
            "audio_q_size": self.audio_track.q.qsize(),
            "video_drops": self.video_track.frames_dropped,
            "audio_drops": self.audio_track.frames_dropped,
            "mode": self._last_mode,
        }

    def start(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._produce())

    def stop(self) -> None:
        self._stop.set()
        self.video_track.close()
        self.audio_track.close()

    def push_chunk(
        self, frames_np: np.ndarray, pcm48_int16: np.ndarray, meta: dict | None = None
    ) -> None:
        with self._play_lock:
            self._play_queue.append([frames_np, pcm48_int16, 0, meta or {}])
            self.chunks_pushed += 1
            self.max_play_queue = max(self.max_play_queue, len(self._play_queue))
            if self.debug:
                audio_sec = pcm48_int16.shape[0] / max(float(AUDIO_SR_OUT), 1.0)
                video_sec = frames_np.shape[0] / max(float(self.fps), 1.0)
                print(
                    f"[DEBUG/webrtc] push_chunk | idx={self.chunks_pushed - 1:03d} "
                    f"frames={frames_np.shape[0]} video={video_sec:.3f}s "
                    f"audio={audio_sec:.3f}s play_q={len(self._play_queue)}"
                )

    async def _produce(self) -> None:
        silence = np.zeros(AUDIO_SAMPLES_PER_FRAME, dtype=np.int16)
        pace_t0 = time.monotonic()
        pace_frame_no = 0
        current = None
        v_pts = 0
        a_pts = 0
        try:
            while not self._stop.is_set():
                if current is None:
                    with self._play_lock:
                        if self._play_queue:
                            current = self._play_queue.popleft()
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
                            f"[DEBUG/webrtc] play_start | chunk={meta.get('chunk_index', '?')} "
                            f"chunk_age={chunk_age:.3f}s push_to_play={push_to_play:.3f}s "
                            f"play_q={len(self._play_queue)} "
                            f"video_q={self.video_track.q.qsize()} audio_q={self.audio_track.q.qsize()}"
                        )
                    vf_np = frames_np[pos]
                    a_start = pos * AUDIO_SAMPLES_PER_FRAME
                    a_end = a_start + AUDIO_SAMPLES_PER_FRAME
                    ap = pcm48[a_start:a_end]
                    if ap.shape[0] < AUDIO_SAMPLES_PER_FRAME:
                        ap = np.concatenate(
                            [ap, np.zeros(AUDIO_SAMPLES_PER_FRAME - ap.shape[0], dtype=np.int16)]
                        )
                    current[2] = pos + 1
                    self.reply_ticks_emitted += 1
                    if self._last_mode != "reply" and self.debug:
                        now = time.monotonic()
                        print(
                            f"[DEBUG/webrtc] mode -> reply | "
                            f"idle_for={now - self._last_mode_change_t:.3f}s "
                            f"play_q={len(self._play_queue)}"
                        )
                        self._last_mode_change_t = now
                    self._last_mode = "reply"
                else:
                    vf_np = self.idle_frame_uint8
                    ap = silence
                    self.idle_ticks_emitted += 1
                    if self._last_mode != "idle" and self.debug:
                        now = time.monotonic()
                        print(
                            f"[DEBUG/webrtc] mode -> idle | "
                            f"reply_for={now - self._last_mode_change_t:.3f}s "
                            f"play_q={len(self._play_queue)}"
                        )
                        self._last_mode_change_t = now
                    self._last_mode = "idle"

                vframe = VideoFrame.from_ndarray(vf_np, format="rgb24")
                vframe.pts = v_pts
                vframe.time_base = fractions.Fraction(1, self.fps)
                v_pts += 1
                self.video_track.put_nowait_drop_oldest(vframe)

                for offset in (0, AUDIO_SAMPLES_PER_PKT):
                    chunk = ap[offset : offset + AUDIO_SAMPLES_PER_PKT]
                    if chunk.shape[0] < AUDIO_SAMPLES_PER_PKT:
                        chunk = np.concatenate(
                            [chunk, np.zeros(AUDIO_SAMPLES_PER_PKT - chunk.shape[0], dtype=np.int16)]
                        )
                    aframe = AudioFrame.from_ndarray(
                        chunk.reshape(1, -1), format="s16", layout="mono"
                    )
                    aframe.sample_rate = AUDIO_SR_OUT
                    aframe.pts = a_pts
                    aframe.time_base = fractions.Fraction(1, AUDIO_SR_OUT)
                    a_pts += AUDIO_SAMPLES_PER_PKT
                    self.audio_track.put_nowait_drop_oldest(aframe)

                self.frames_emitted += 1
                pace_frame_no += 1
                target = pace_t0 + pace_frame_no / self.fps
                dt = target - time.monotonic()
                if dt > 0:
                    await asyncio.sleep(dt)
        finally:
            self.video_track.close()
            self.audio_track.close()
