from __future__ import annotations

import asyncio
import contextlib
import fractions
import json
import struct
import threading
import time
from collections import deque

import av
import numpy as np
from av.audio.frame import AudioFrame
from av.video.frame import VideoFrame

AUDIO_SR_OUT = 48000
AUDIO_PTIME = 0.020
AUDIO_SAMPLES_PER_PKT = int(AUDIO_SR_OUT * AUDIO_PTIME)  # 960
AUDIO_SAMPLES_PER_FRAME = AUDIO_SAMPLES_PER_PKT * 2  # 1920 at 25 fps

MSG_VIDEO_INIT = 0x01
MSG_VIDEO_NAL = 0x02
MSG_AUDIO_OPUS = 0x03
MSG_SYNC_META = 0x04
MSG_AUDIO_INIT = 0x05


def _pack_ws_message(kind: int, timestamp_us: int, payload: bytes) -> bytes:
    return struct.pack(">BIQ", kind, len(payload), int(timestamp_us)) + payload


def _codec_string_from_avcc(extradata: bytes | None) -> str:
    if extradata and len(extradata) >= 4 and extradata[0] == 0x01:
        return f"avc1.{extradata[1]:02x}{extradata[2]:02x}{extradata[3]:02x}"
    return "avc1.42E01F"


def _make_opus_head(
    *,
    channels: int = 1,
    pre_skip: int = 312,
    sample_rate: int = AUDIO_SR_OUT,
    output_gain: int = 0,
) -> bytes:
    return (
        b"OpusHead"
        + bytes([1, channels])
        + int(pre_skip).to_bytes(2, "little", signed=False)
        + int(sample_rate).to_bytes(4, "little", signed=False)
        + int(output_gain).to_bytes(2, "little", signed=True)
        + bytes([0])
    )


def _looks_like_annexb(payload: bytes | None) -> bool:
    if not payload:
        return False
    return payload.startswith(b"\x00\x00\x00\x01") or payload.startswith(b"\x00\x00\x01")


class WSStreamSession:
    """Per-viewer WebSocket stream session.

    Mirrors the current fMP4 / WebRTC `av_session` contract closely so the
    existing LiveMoshiIMTalkerSession can keep calling `push_chunk(...)`
    without any renderer changes.
    """

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        idle_frame_uint8: np.ndarray,
        fps: int,
        *,
        debug: bool = False,
        video_bitrate: int = 800_000,
        audio_bitrate: int = 32_000,
        out_queue_size: int = 512,
    ):
        self.loop = loop
        self.idle_frame_uint8 = idle_frame_uint8
        self.fps = int(round(float(fps)))
        self.debug = debug
        self.video_bitrate = int(video_bitrate)
        self.audio_bitrate = int(audio_bitrate)
        self.out_q: asyncio.Queue[bytes | None] = asyncio.Queue(maxsize=out_queue_size)

        self._play_queue: deque = deque()
        self._play_lock = threading.Lock()
        self._stop = asyncio.Event()
        self._task: asyncio.Task | None = None

        self._video_encoder = None
        self._audio_encoder = None
        self._video_init_payload: bytes | None = None
        self._audio_init_payload: bytes | None = None
        self._video_codec = "avc1.42E01F"
        self._stream_init_sent = False

        self.frames_emitted = 0
        self.reply_ticks_emitted = 0
        self.idle_ticks_emitted = 0
        self.chunks_pushed = 0
        self.video_packets_sent = 0
        self.audio_packets_sent = 0
        self.meta_packets_sent = 0
        self.out_q_drops = 0
        self.max_play_queue = 0
        self.max_out_q = 0
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
            "out_q_size": self.out_q.qsize(),
            "max_out_q": self.max_out_q,
            "video_packets_sent": self.video_packets_sent,
            "audio_packets_sent": self.audio_packets_sent,
            "meta_packets_sent": self.meta_packets_sent,
            "out_q_drops": self.out_q_drops,
            "mode": self._last_mode,
        }

    def start(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._produce())

    def stop(self) -> None:
        self._stop.set()

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
                    f"[DEBUG/ws] push_chunk | idx={self.chunks_pushed - 1:03d} "
                    f"frames={frames_np.shape[0]} video={video_sec:.3f}s "
                    f"audio={audio_sec:.3f}s play_q={len(self._play_queue)}"
                )

    def _ensure_encoders(self) -> None:
        if self._video_encoder is None:
            venc = av.CodecContext.create("libx264", "w")
            venc.width = int(self.idle_frame_uint8.shape[1])
            venc.height = int(self.idle_frame_uint8.shape[0])
            venc.pix_fmt = "yuv420p"
            venc.time_base = fractions.Fraction(1, self.fps)
            venc.framerate = fractions.Fraction(self.fps, 1)
            venc.bit_rate = self.video_bitrate
            venc.options = {
                "preset": "ultrafast",
                "tune": "zerolatency",
                "g": str(self.fps),
                "bf": "0",
                "x264-params": (
                    f"keyint={self.fps}:min-keyint={self.fps}:"
                    "scenecut=0:rc-lookahead=0:repeat-headers=1"
                ),
            }
            venc.open()
            self._video_encoder = venc
            if venc.extradata:
                self._video_init_payload = bytes(venc.extradata)
                self._video_codec = _codec_string_from_avcc(venc.extradata)

        if self._audio_encoder is None:
            aenc = av.CodecContext.create("libopus", "w")
            aenc.sample_rate = AUDIO_SR_OUT
            aenc.rate = AUDIO_SR_OUT
            aenc.layout = "mono"
            aenc.format = "s16"
            aenc.time_base = fractions.Fraction(1, AUDIO_SR_OUT)
            aenc.bit_rate = self.audio_bitrate
            aenc.options = {
                "application": "voip",
                "frame_duration": "20",
            }
            aenc.open()
            self._audio_encoder = aenc
            self._audio_init_payload = _make_opus_head()

    def _stream_init_json(self) -> bytes:
        return json.dumps(
            {
                "meta_type": "stream_init",
                "video_codec": self._video_codec,
                "width": int(self.idle_frame_uint8.shape[1]),
                "height": int(self.idle_frame_uint8.shape[0]),
                "fps": self.fps,
                "audio_codec": "opus",
                "audio_sr": AUDIO_SR_OUT,
                "audio_channels": 1,
                "audio_ptime_ms": int(round(AUDIO_PTIME * 1000.0)),
            }
        ).encode("utf-8")

    def _emit_stream_init(self) -> None:
        if self._stream_init_sent:
            return
        self._ensure_encoders()
        self._queue_message(_pack_ws_message(MSG_SYNC_META, 0, self._stream_init_json()))
        self.meta_packets_sent += 1
        if self._audio_init_payload:
            self._queue_message(_pack_ws_message(MSG_AUDIO_INIT, 0, self._audio_init_payload))
        self._stream_init_sent = True

    def _queue_message(self, payload: bytes | None) -> None:
        if self.out_q.full():
            try:
                self.out_q.get_nowait()
                self.out_q_drops += 1
            except asyncio.QueueEmpty:
                pass
        try:
            self.out_q.put_nowait(payload)
            self.max_out_q = max(self.max_out_q, self.out_q.qsize())
        except asyncio.QueueFull:
            self.out_q_drops += 1

    def _emit_video_frame(self, frame_np: np.ndarray, timestamp_us: int) -> None:
        self._ensure_encoders()
        if self._video_encoder is None:
            return
        vframe = VideoFrame.from_ndarray(frame_np, format="rgb24")
        vframe.pts = self.frames_emitted
        vframe.time_base = fractions.Fraction(1, self.fps)
        for packet in self._video_encoder.encode(vframe):
            is_key = bool(getattr(packet, "is_keyframe", False))
            raw = bytes(packet)
            if is_key:
                extradata = self._video_encoder.extradata
                print(f"[VDEBUG] keyframe | extradata={bytes(extradata[:8]).hex() if extradata else None} nal[:8]={raw[:8].hex()}", flush=True)
            if is_key and self._video_encoder.extradata:
                payload = bytes(self._video_encoder.extradata)
                self._queue_message(_pack_ws_message(MSG_VIDEO_INIT, timestamp_us, payload))
            self._queue_message(_pack_ws_message(MSG_VIDEO_NAL, timestamp_us, raw))
            self.video_packets_sent += 1

    def _emit_audio_frame(self, pcm_chunk: np.ndarray, pts: int) -> None:
        self._ensure_encoders()
        if self._audio_encoder is None:
            return
        aframe = AudioFrame.from_ndarray(
            pcm_chunk.reshape(1, -1), format="s16", layout="mono"
        )
        aframe.sample_rate = AUDIO_SR_OUT
        aframe.pts = pts
        aframe.time_base = fractions.Fraction(1, AUDIO_SR_OUT)
        timestamp_us = int(round((pts / float(AUDIO_SR_OUT)) * 1_000_000.0))
        for packet in self._audio_encoder.encode(aframe):
            self._queue_message(_pack_ws_message(MSG_AUDIO_OPUS, timestamp_us, bytes(packet)))
            self.audio_packets_sent += 1

    async def _produce(self) -> None:
        silence = np.zeros(AUDIO_SAMPLES_PER_FRAME, dtype=np.int16)
        pace_t0 = time.monotonic()
        pace_frame_no = 0
        current = None
        audio_pts = 0
        try:
            self._emit_stream_init()
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
                    if pos == 0:
                        now = time.perf_counter()
                        enqueue_ts = float(meta.get("enqueue_wall_ts", 0.0) or 0.0)
                        push_ts = float(meta.get("push_wall_ts", 0.0) or 0.0)
                        chunk_age = max(now - enqueue_ts, 0.0) if enqueue_ts > 0.0 else -1.0
                        push_to_stream = max(now - push_ts, 0.0) if push_ts > 0.0 else -1.0
                        sync_payload = json.dumps(
                            {
                                "meta_type": "chunk_start",
                                "chunk_index": meta.get("chunk_index", -1),
                                "chunk_age": chunk_age,
                                "push_to_stream": push_to_stream,
                                "play_queue_len": len(self._play_queue),
                            }
                        ).encode("utf-8")
                        self._queue_message(_pack_ws_message(MSG_SYNC_META, 0, sync_payload))
                        self.meta_packets_sent += 1
                        if self.debug:
                            print(
                                "[DBG/wsq] "
                                f"chunk={meta.get('chunk_index', '?')} "
                                f"chunk_age={chunk_age:.3f}s "
                                f"push_to_stream={push_to_stream:.3f}s "
                                f"play_queue_len={len(self._play_queue)} "
                                f"max_play_queue={self.max_play_queue} "
                                f"out_q_size={self.out_q.qsize()} "
                                f"max_out_q={self.max_out_q} "
                                f"frames_emitted={self.frames_emitted} "
                                f"reply_ticks_emitted={self.reply_ticks_emitted} "
                                f"idle_ticks_emitted={self.idle_ticks_emitted} "
                                f"video_packets_sent={self.video_packets_sent} "
                                f"audio_packets_sent={self.audio_packets_sent} "
                                f"out_q_drops={self.out_q_drops} "
                                f"mode={self._last_mode}"
                            )
                        if self.debug:
                            print(
                                f"[DEBUG/ws] play_start | chunk={meta.get('chunk_index', '?')} "
                                f"chunk_age={chunk_age:.3f}s "
                                f"push_to_stream={push_to_stream:.3f}s "
                                f"play_q={len(self._play_queue)} out_q={self.out_q.qsize()}"
                            )

                    frame_np = frames_np[pos]
                    a_start = pos * AUDIO_SAMPLES_PER_FRAME
                    a_end = a_start + AUDIO_SAMPLES_PER_FRAME
                    audio_frame = pcm48[a_start:a_end]
                    if audio_frame.shape[0] < AUDIO_SAMPLES_PER_FRAME:
                        audio_frame = np.concatenate(
                            [
                                audio_frame,
                                np.zeros(
                                    AUDIO_SAMPLES_PER_FRAME - audio_frame.shape[0],
                                    dtype=np.int16,
                                ),
                            ]
                        )
                    current[2] = pos + 1
                    self.reply_ticks_emitted += 1
                    if self._last_mode != "reply" and self.debug:
                        now = time.monotonic()
                        print(
                            f"[DEBUG/ws] mode -> reply | "
                            f"idle_for={now - self._last_mode_change_t:.3f}s "
                            f"play_q={len(self._play_queue)} out_q={self.out_q.qsize()}"
                        )
                        self._last_mode_change_t = now
                    self._last_mode = "reply"
                else:
                    frame_np = self.idle_frame_uint8
                    audio_frame = silence
                    self.idle_ticks_emitted += 1
                    if self._last_mode != "idle" and self.debug:
                        now = time.monotonic()
                        print(
                            f"[DEBUG/ws] mode -> idle | "
                            f"reply_for={now - self._last_mode_change_t:.3f}s "
                            f"play_q={len(self._play_queue)} out_q={self.out_q.qsize()}"
                        )
                        self._last_mode_change_t = now
                    self._last_mode = "idle"

                video_ts_us = int(round((self.frames_emitted / float(self.fps)) * 1_000_000.0))
                self._emit_video_frame(frame_np, video_ts_us)
                self._emit_audio_frame(audio_frame[:AUDIO_SAMPLES_PER_PKT], audio_pts)
                audio_pts += AUDIO_SAMPLES_PER_PKT
                self._emit_audio_frame(audio_frame[AUDIO_SAMPLES_PER_PKT:], audio_pts)
                audio_pts += AUDIO_SAMPLES_PER_PKT

                self.frames_emitted += 1
                pace_frame_no += 1
                target = pace_t0 + pace_frame_no / self.fps
                dt = target - time.monotonic()
                if dt > 0:
                    await asyncio.sleep(dt)
        finally:
            with contextlib.suppress(Exception):
                self._flush_encoders(audio_pts)
            self._queue_message(None)

    def _flush_encoders(self, audio_pts: int) -> None:
        if self._video_encoder is not None:
            timestamp_us = int(round((self.frames_emitted / float(self.fps)) * 1_000_000.0))
            for packet in self._video_encoder.encode(None):
                self._queue_message(_pack_ws_message(MSG_VIDEO_NAL, timestamp_us, bytes(packet)))
                self.video_packets_sent += 1
        if self._audio_encoder is not None:
            timestamp_us = int(round((audio_pts / float(AUDIO_SR_OUT)) * 1_000_000.0))
            for packet in self._audio_encoder.encode(None):
                self._queue_message(_pack_ws_message(MSG_AUDIO_OPUS, timestamp_us, bytes(packet)))
                self.audio_packets_sent += 1
