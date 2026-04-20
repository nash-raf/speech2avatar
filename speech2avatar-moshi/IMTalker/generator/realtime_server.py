"""Real-time streaming server for IMTalker.

Sends StreamingInferenceAgent output to a viewer with minimum extra delay.

Modes (--transport):
  fmp4    — muxed H.264 + AAC over chunked HTTP (default).
  webrtc  — aiortc + browser viewer.
  mjpeg   — legacy HTTP fallback; works through proxies but drifts.
  local   — cv2.imshow window. Truly zero transport. No audio.

Architecture:
    [producer thread]
        StreamingInferenceAgent.step() -> [N,3,512,512] in [0,1]
        slice 48 kHz mono PCM aligned to wall clock
              |
              v
    [asyncio.Queue]   <- thread-safe push via loop.call_soon_threadsafe
              |
              v
    [VideoTrack.recv() / AudioTrack.recv()]
        stamp PTS from a single monotonic reference, hand to aiortc.

The producer paces output to wall-clock so the queues stay small (≈1 chunk
of slack) regardless of how fast inference runs. Drops oldest on overflow.

For maximum speed before launching:
    export IMTALKER_TORCH_COMPILE=1
    export IMTALKER_DECODE_BATCH=8

NVENC: aiortc currently uses libx264 internally. At 512x512 / 25 fps with
the ultrafast+zerolatency tuning that aiortc applies, libx264 is well under
5 ms / frame on modern CPUs and is NOT the latency bottleneck — inference
is. Wiring h264_nvenc into aiortc requires replacing its internal
H264Encoder; tracked as a follow-up.
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import fractions
import json
import os
import sys
import threading
import time
from typing import List, Optional

import numpy as np
import torch

# project imports
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from generator.generate import (  # noqa: E402
    InferenceAgent,
    InferenceOptions,
    _load_audio_native,
    _resample_mono,
)
from generator.streaming import StreamingInferenceAgent  # noqa: E402

AUDIO_SAMPLE_RATE = 48000
AUDIO_PTIME = 0.020  # 20 ms — Opus packet size
AUDIO_SAMPLES_PER_PKT = int(AUDIO_SAMPLE_RATE * AUDIO_PTIME)  # 960


def _frames_to_uint8_hwc(frames_t: torch.Tensor) -> np.ndarray:
    """[N, 3, H, W] in [0, 1] (Sigmoid output) -> [N, H, W, 3] uint8."""
    x = frames_t.detach().clamp(0, 1).mul(255).to(torch.uint8)
    return x.permute(0, 2, 3, 1).contiguous().cpu().numpy()


# ====================================================================== #
# StreamSession — owns the producer thread and the per-track queues
# ====================================================================== #
class StreamSession:
    def __init__(
        self,
        stream_agent: StreamingInferenceAgent,
        audio_source_16k: np.ndarray,
        loop: asyncio.AbstractEventLoop,
        video_buffer_frames: int = 50,    # 2 s @ 25 fps of slack
        audio_buffer_pkts: int = 200,     # 4 s of 20 ms packets of slack
    ):
        self.stream_agent = stream_agent
        self.audio_source_16k = audio_source_16k.astype(np.float32, copy=False)
        self.fps = stream_agent.fps
        self.sr_in = stream_agent.sr
        self.loop = loop

        self.video_q: asyncio.Queue = asyncio.Queue(maxsize=video_buffer_frames)
        self.audio_q: asyncio.Queue = asyncio.Queue(maxsize=audio_buffer_pkts)

        # Resample full audio to 48 kHz once for the audio track.
        a48 = _resample_mono(self.audio_source_16k, self.sr_in, AUDIO_SAMPLE_RATE)
        self.audio_48k_int16 = (np.clip(a48, -1.0, 1.0) * 32767).astype(np.int16)

        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.frames_emitted = 0
        self.dropped_video = 0
        self.dropped_audio = 0

    # ------------------------------------------------------------------ #
    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._produce, daemon=True, name="imtalker-producer"
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)

    # ------------------------------------------------------------------ #
    def _produce(self) -> None:
        agent = self.stream_agent
        samples_per_emit = agent.samples_per_emit
        n_in = self.audio_source_16k.shape[0]
        spf_a48 = AUDIO_SAMPLE_RATE / self.fps  # 48k samples per video frame

        wall_t0 = time.monotonic()
        pos = 0
        chunk_idx = 0

        while pos < n_in and not self._stop.is_set():
            end = min(pos + samples_per_emit, n_in)
            chunk = self.audio_source_16k[pos:end]
            full = chunk.shape[0] == samples_per_emit
            agent.feed_audio(chunk)

            try:
                out = agent.step() if full else agent.flush()
            except Exception as e:
                print(f"[realtime] producer error: {e}")
                break
            if out is None:
                break

            frames_np = _frames_to_uint8_hwc(out)
            n_frames = frames_np.shape[0]

            for i in range(n_frames):
                # Pace to wall-clock so the queue stays small.
                target = wall_t0 + (self.frames_emitted + 1) / self.fps
                while True:
                    dt = target - time.monotonic()
                    if dt <= 0 or self._stop.is_set():
                        break
                    time.sleep(min(dt, 0.005))
                if self._stop.is_set():
                    return

                self._submit_video(frames_np[i])

                a_start = int(round(self.frames_emitted * spf_a48))
                a_end = int(round((self.frames_emitted + 1) * spf_a48))
                self._submit_audio_for_frame(self.audio_48k_int16[a_start:a_end])

                self.frames_emitted += 1

            chunk_idx += 1
            pos = end

        # End-of-stream sentinels.
        self._submit_video(None)
        self._submit_audio_pkt(None)
        print(
            f"[realtime] producer finished: chunks={chunk_idx} "
            f"frames={self.frames_emitted} dropped_v={self.dropped_video} "
            f"dropped_a={self.dropped_audio}"
        )

    # ------------------------------------------------------------------ #
    def _submit_video(self, frame_or_none) -> None:
        def _do():
            if frame_or_none is None:
                with contextlib.suppress(asyncio.QueueFull):
                    self.video_q.put_nowait(None)
                return
            if self.video_q.full():
                try:
                    self.video_q.get_nowait()
                    self.dropped_video += 1
                except asyncio.QueueEmpty:
                    pass
            try:
                self.video_q.put_nowait(frame_or_none)
            except asyncio.QueueFull:
                self.dropped_video += 1

        self.loop.call_soon_threadsafe(_do)

    def _submit_audio_for_frame(self, pcm_int16: np.ndarray) -> None:
        # 1 video frame at 25 fps == 1920 samples == 2 Opus packets exactly.
        n = pcm_int16.shape[0]
        i = 0
        while i < n:
            j = min(i + AUDIO_SAMPLES_PER_PKT, n)
            pkt = pcm_int16[i:j]
            if pkt.shape[0] < AUDIO_SAMPLES_PER_PKT:
                pkt = np.concatenate(
                    [pkt, np.zeros(AUDIO_SAMPLES_PER_PKT - pkt.shape[0], dtype=np.int16)]
                )
            self._submit_audio_pkt(pkt.copy())
            i = j

    def _submit_audio_pkt(self, pkt) -> None:
        def _do():
            if pkt is None:
                with contextlib.suppress(asyncio.QueueFull):
                    self.audio_q.put_nowait(None)
                return
            if self.audio_q.full():
                try:
                    self.audio_q.get_nowait()
                    self.dropped_audio += 1
                except asyncio.QueueEmpty:
                    pass
            try:
                self.audio_q.put_nowait(pkt)
            except asyncio.QueueFull:
                self.dropped_audio += 1

        self.loop.call_soon_threadsafe(_do)


# ====================================================================== #
# Tracks (lazy import of aiortc/av so local mode has no hard dep)
# ====================================================================== #
def _make_tracks(session: StreamSession):
    from aiortc import MediaStreamTrack
    from aiortc.mediastreams import MediaStreamError
    from av import AudioFrame, VideoFrame

    class IMTalkerVideoTrack(MediaStreamTrack):
        kind = "video"

        def __init__(self) -> None:
            super().__init__()
            self._idx = 0

        async def recv(self):
            frame_np = await session.video_q.get()
            if frame_np is None:
                self.stop()
                raise MediaStreamError("video EOS")
            vf = VideoFrame.from_ndarray(frame_np, format="rgb24")
            vf.pts = self._idx
            vf.time_base = fractions.Fraction(1, session.fps)
            self._idx += 1
            return vf

    class IMTalkerAudioTrack(MediaStreamTrack):
        kind = "audio"

        def __init__(self) -> None:
            super().__init__()
            self._samples_sent = 0

        async def recv(self):
            pkt = await session.audio_q.get()
            if pkt is None:
                self.stop()
                raise MediaStreamError("audio EOS")
            af = AudioFrame.from_ndarray(pkt.reshape(1, -1), format="s16", layout="mono")
            af.sample_rate = AUDIO_SAMPLE_RATE
            af.pts = self._samples_sent
            af.time_base = fractions.Fraction(1, AUDIO_SAMPLE_RATE)
            self._samples_sent += pkt.shape[0]
            return af

    return IMTalkerVideoTrack(), IMTalkerAudioTrack()


# ====================================================================== #
# WebRTC server
# ====================================================================== #
VIEWER_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>IMTalker live</title><style>
body{background:#111;color:#ddd;font-family:system-ui,sans-serif;margin:0;padding:24px;}
video{background:#000;width:512px;height:512px;display:block;margin-top:12px;border-radius:8px;}
button{font:inherit;padding:8px 16px;border-radius:6px;border:0;background:#3b82f6;color:#fff;cursor:pointer;}
button:disabled{opacity:.5;cursor:default;}
pre{font-size:12px;color:#888;white-space:pre-wrap;}
</style></head>
<body>
<h2>IMTalker live</h2>
<button id="go">Connect</button>
<video id="v" autoplay playsinline></video>
<pre id="log"></pre>
<script>
const log = (...a)=>{document.getElementById("log").textContent += a.join(" ")+"\\n"};
document.getElementById("go").onclick = async () => {
  document.getElementById("go").disabled = true;
  const pc = new RTCPeerConnection({iceServers: __ICE_SERVERS__});
  pc.addTransceiver("video",{direction:"recvonly"});
  pc.addTransceiver("audio",{direction:"recvonly"});
  pc.ontrack = (ev)=>{
    log("track:", ev.track.kind);
    const v = document.getElementById("v");
    if (!v.srcObject) v.srcObject = new MediaStream();
    v.srcObject.addTrack(ev.track);
  };
  const offer = await pc.createOffer();
  await pc.setLocalDescription(offer);
  await new Promise(r=>{
    if (pc.iceGatheringState==="complete") return r();
    pc.addEventListener("icegatheringstatechange", ()=>{
      if (pc.iceGatheringState==="complete") r();
    });
  });
  const resp = await fetch("/offer",{method:"POST",headers:{"content-type":"application/json"},
    body:JSON.stringify({sdp:pc.localDescription.sdp,type:pc.localDescription.type})});
  const answer = await resp.json();
  await pc.setRemoteDescription(answer);
  log("connected");
};
</script></body></html>"""


class WebRTCServer:
    def __init__(self, stream_agent, audio_source_16k, host, port, ice_servers: list):
        self.stream_agent = stream_agent
        self.audio_source = audio_source_16k
        self.host = host
        self.port = port
        # ice_servers: list of dicts in W3C RTCIceServer shape, e.g.
        #   [{"urls": "stun:stun.l.google.com:19302"},
        #    {"urls": "turns:turn.example.com:443?transport=tcp",
        #     "username": "u", "credential": "p"}]
        # Both the browser RTCPeerConnection AND the aiortc RTCPeerConnection
        # need this — relay only works if BOTH ends know the TURN server.
        self.ice_servers = ice_servers
        self.pcs: set = set()
        self.session: Optional[StreamSession] = None

    async def index(self, request):
        from aiohttp import web
        html = VIEWER_HTML.replace("__ICE_SERVERS__", json.dumps(self.ice_servers))
        return web.Response(content_type="text/html", text=html)

    async def offer(self, request):
        from aiohttp import web
        from aiortc import RTCConfiguration, RTCIceServer, RTCPeerConnection, RTCSessionDescription

        body = await request.json()
        offer_sdp = RTCSessionDescription(sdp=body["sdp"], type=body["type"])
        rtc_ice = [
            RTCIceServer(
                urls=s["urls"],
                username=s.get("username"),
                credential=s.get("credential"),
            )
            for s in self.ice_servers
        ]
        pc = RTCPeerConnection(configuration=RTCConfiguration(iceServers=rtc_ice))
        self.pcs.add(pc)

        # One session at a time — tear down any prior one.
        if self.session is not None:
            print("[realtime] tearing down previous session")
            self.session.stop()
            self.session = None

        loop = asyncio.get_event_loop()
        self.session = StreamSession(self.stream_agent, self.audio_source, loop)
        v_track, a_track = _make_tracks(self.session)
        pc.addTrack(v_track)
        pc.addTrack(a_track)

        @pc.on("connectionstatechange")
        async def on_state():
            print(f"[realtime] pc state: {pc.connectionState}")
            if pc.connectionState in ("failed", "closed", "disconnected"):
                if self.session is not None:
                    self.session.stop()
                    self.session = None
                with contextlib.suppress(Exception):
                    await pc.close()
                self.pcs.discard(pc)

        await pc.setRemoteDescription(offer_sdp)
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        while pc.iceGatheringState != "complete":
            await asyncio.sleep(0.05)

        # Reset stream_agent state to a clean slate for the new viewer.
        # The reference encode is reused, but the temporal carry resets so
        # the new session starts from silence.
        self.stream_agent.state = self.stream_agent.fm.make_initial_stream_state(
            self.stream_agent.t_r
        )
        self.stream_agent.audio_buffer.clear()
        self.stream_agent.audio_buffer_len = 0
        self.stream_agent.chunk_idx = 0

        self.session.start()
        return web.json_response(
            {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
        )

    async def on_shutdown(self, app):
        if self.session is not None:
            self.session.stop()
        for pc in list(self.pcs):
            with contextlib.suppress(Exception):
                await pc.close()
        self.pcs.clear()

    def run(self) -> None:
        from aiohttp import web

        app = web.Application()
        app.router.add_get("/", self.index)
        app.router.add_post("/offer", self.offer)
        app.on_shutdown.append(self.on_shutdown)
        # 0.0.0.0 / :: mean "all interfaces" — you cannot open that host in a browser.
        p = self.port
        print(f"[realtime] HTTP + WebRTC listening on {self.host}:{p}")
        print(f"  → Open in browser: http://127.0.0.1:{p}/  (same machine)")
        print(
            "  → From your laptop (RunPod):  ssh -L %d:127.0.0.1:%d user@pod  then  http://localhost:%d/"
            % (p, p, p)
        )
        web.run_app(app, host=self.host, port=self.port, print=None)


# ====================================================================== #
# MJPEG-over-HTTP server — works through any HTTP(S) proxy (RunPod, ngrok)
# Trades a bit of bandwidth for "actually reaches the browser today".
# Audio is served as a parallel <audio src="/audio.wav"> element; the JS
# starts <img> and <audio> at the same instant for ~hand-aligned A/V.
# ====================================================================== #
MJPEG_VIEWER_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>IMTalker live (MJPEG)</title><style>
body{background:#111;color:#ddd;font-family:system-ui,sans-serif;margin:0;padding:24px;}
img{background:#000;width:512px;height:512px;display:block;margin-top:12px;border-radius:8px;}
button{font:inherit;padding:6px 12px;border-radius:6px;border:0;background:#3b82f6;color:#fff;cursor:pointer;margin:4px 4px 4px 0;}
button:disabled{opacity:.5;cursor:default;}
button.active{background:#10b981;}
pre{font-size:12px;color:#888;white-space:pre-wrap;}
.clips{margin:8px 0;}
.note{font-size:12px;color:#888;max-width:560px;}
</style></head>
<body>
<h2>IMTalker live (MJPEG)</h2>
<div class="clips" id="clips"></div>
<img id="v" />
<audio id="a" preload="auto"></audio>
<pre id="log">Pick a clip above. Stream is HTTP MJPEG; works through any proxy.</pre>
<p class="note">
Note: MJPEG video and HTML5 audio run on independent clocks (no muxed PTS),
and the proxy adds buffering. Lip-sync is approximate. For tight sync use
WebRTC + TURN, or generate the offline MP4 via generate.py.
</p>
<script>
const CLIPS = __CLIPS__;
const log = (...a)=>{document.getElementById("log").textContent += a.join(" ")+"\\n"};
let firstFrameSeenAt = null;
let runId = 0;

function startClip(name, btn) {
  // disable all clip buttons during a run
  document.querySelectorAll(".clips button").forEach(b => {
    b.classList.remove("active");
  });
  btn.classList.add("active");

  runId += 1;
  const myRun = runId;
  firstFrameSeenAt = null;
  const t0 = performance.now();
  const v = document.getElementById("v");
  const a = document.getElementById("a");

  // Stop the prior audio so we don't overlap.
  try { a.pause(); } catch(e) {}

  // 1. Prime the audio element with the same wav.
  a.src = "/audio/file?aud=" + encodeURIComponent(name);
  a.currentTime = 0;
  a.load();

  // 2. Hook first-frame onload BEFORE setting src.
  v.onload = () => {
    if (myRun !== runId) return;
    if (firstFrameSeenAt !== null) return;
    firstFrameSeenAt = performance.now();
    const ms = (firstFrameSeenAt - t0).toFixed(0);
    log("[" + name + "] first frame in " + ms + " ms");
    // Sync mitigation: only start audio after the first JPEG arrived.
    a.play().catch(e => log("audio play blocked:", e.message));
  };

  // 3. Kick off the MJPEG stream (cache-busted).
  v.src = "/stream.mjpg?aud=" + encodeURIComponent(name) + "&t=" + Date.now();
  log("[" + name + "] requested");
}

const container = document.getElementById("clips");
CLIPS.forEach(name => {
  const b = document.createElement("button");
  b.textContent = name;
  b.onclick = () => startClip(name, b);
  container.appendChild(b);
});
if (CLIPS.length === 0) {
  container.textContent = "No clips found in assets/. Server logs have details.";
}
</script></body></html>"""


FMP4_VIEWER_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>IMTalker live (fMP4)</title>
<style>
body{font-family:system-ui;margin:24px;max-width:760px}
.clips{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:12px}
button{padding:8px 12px;cursor:pointer}
video{width:512px;height:512px;background:#000;border-radius:8px}
pre{background:#f5f5f5;padding:8px;border-radius:6px;white-space:pre-wrap}
.note{color:#555;font-size:13px;margin-top:8px}
</style></head><body>
<h2>IMTalker live (fMP4)</h2>
<div class="clips" id="clips"></div>
<video id="v" controls autoplay playsinline></video>
<pre id="log">Pick a clip. Single muxed MP4 — video and audio share one PTS timeline.</pre>
<p class="note">Transport: fragmented MP4 over chunked HTTP, played via MediaSource. Works through any HTTP/HTTPS proxy. Lip-sync is enforced by the container, not the page.</p>
<script>
const CLIPS = __CLIPS__;
const MIME_CANDIDATES = [
  'video/mp4; codecs="avc1.640015, mp4a.40.2"',
  'video/mp4; codecs="avc1.4D401E, mp4a.40.2"',
  'video/mp4; codecs="avc1.42E01E, mp4a.40.2"',
];
const MIME = MIME_CANDIDATES.find(m => "MediaSource" in window && MediaSource.isTypeSupported(m)) || "";
let runId = 0, abortCtl = null, currentUrl = null;

function log(...a){const el=document.getElementById("log");el.textContent+="\\n"+a.join(" ");el.scrollTop=el.scrollHeight}

async function startClip(name){
  runId += 1; const myRun = runId;
  if (abortCtl) abortCtl.abort();
  abortCtl = new AbortController();

  const v = document.getElementById("v");
  const t0 = performance.now();
  let firstFrameLogged = false;
  let playAttempted = false;

  try { v.pause(); } catch (e) {}
  if (currentUrl) {
    URL.revokeObjectURL(currentUrl);
    currentUrl = null;
  }

  if (!MIME) {
    log("MSE/H264+AAC not supported in this browser"); return;
  }
  const ms = new MediaSource();
  currentUrl = URL.createObjectURL(ms);
  v.src = currentUrl;

  ms.addEventListener("sourceopen", async () => {
    if (myRun !== runId) return;
    const sb = ms.addSourceBuffer(MIME);
    sb.mode = "sequence";

    const queue = [];
    let appending = false;
    function pump(){
      if (appending || !queue.length || sb.updating) return;
      appending = true;
      sb.appendBuffer(queue.shift());
    }
    sb.addEventListener("updateend", () => {
      appending = false;
      if (!playAttempted) {
        playAttempted = true;
        v.play().catch(e => log("play blocked:", e.message));
      }
      pump();
    });

    v.addEventListener("playing", () => {
      if (myRun !== runId) return;
      if (firstFrameLogged) return;
      firstFrameLogged = true;
      log("[" + name + "] first frame in " + (performance.now()-t0).toFixed(0) + " ms");
    }, { once: true });

    try {
      const resp = await fetch("/stream.mp4?aud=" + encodeURIComponent(name) + "&t=" + Date.now(),
                               { signal: abortCtl.signal });
      if (!resp.ok) { log("HTTP", resp.status); return; }
      const reader = resp.body.getReader();
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        if (myRun !== runId) { reader.cancel(); break; }
        queue.push(value);
        pump();
      }
      const finish = () => { try { if (ms.readyState === "open") ms.endOfStream(); } catch(e){} };
      if (sb.updating || queue.length) sb.addEventListener("updateend", finish, { once: true });
      else finish();
    } catch (e) {
      if (e.name !== "AbortError") log("stream error:", e.message);
    }
  }, { once: true });
}

const container = document.getElementById("clips");
CLIPS.forEach(name => {
  const b = document.createElement("button");
  b.textContent = name;
  b.onclick = () => startClip(name);
  container.appendChild(b);
});
if (CLIPS.length === 0) {
  container.textContent = "No clips found in assets/. Server logs have details.";
}
</script></body></html>"""


_AUDIO_EXTS = (".wav", ".mp3", ".m4a", ".flac", ".ogg")


def _scan_assets(assets_dir: str, default_audio_path: Optional[str]) -> List[str]:
    """Return a sorted list of audio basenames in assets_dir.

    The default --aud_path is included even if it lives elsewhere — its basename
    is added and resolved by _resolve_clip below.
    """
    found: list[str] = []
    if assets_dir and os.path.isdir(assets_dir):
        for name in sorted(os.listdir(assets_dir)):
            if name.lower().endswith(_AUDIO_EXTS):
                found.append(name)
    if default_audio_path:
        b = os.path.basename(default_audio_path)
        if b and b not in found:
            found.append(b)
    return found


class MJPEGServer:
    def __init__(
        self,
        stream_agent,
        host,
        port,
        assets_dir: str,
        default_audio_path: Optional[str],
        target_sr: int,
    ):
        self.stream_agent = stream_agent
        self.host = host
        self.port = port
        self.assets_dir = os.path.abspath(assets_dir) if assets_dir else ""
        self.default_audio_path = default_audio_path
        self.target_sr = target_sr
        self.clips = _scan_assets(self.assets_dir, default_audio_path)
        # Cache resampled 16k arrays so picking the same clip twice is instant.
        self._audio_cache: dict[str, np.ndarray] = {}
        self.session: Optional[StreamSession] = None
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------ #
    def _resolve_clip(self, basename: str) -> Optional[str]:
        """Validate basename and return the absolute path on disk, or None.

        Allowlist: only basenames returned by _scan_assets() are accepted, and
        os.path.basename() strips any path separators / `..` / drive letters.
        """
        if not basename:
            return None
        clean = os.path.basename(basename)
        if clean != basename:
            return None
        if clean not in self.clips:
            return None
        # Prefer assets_dir; fall back to the default --aud_path basename match.
        cand = os.path.join(self.assets_dir, clean) if self.assets_dir else ""
        if cand and os.path.exists(cand):
            return cand
        if self.default_audio_path and os.path.basename(self.default_audio_path) == clean:
            return self.default_audio_path
        return None

    def _load_clip_16k(self, path: str) -> np.ndarray:
        if path in self._audio_cache:
            return self._audio_cache[path]
        a, sr = _load_audio_native(path)
        a16 = _resample_mono(a, sr, self.target_sr).astype(np.float32, copy=False)
        self._audio_cache[path] = a16
        return a16

    # ------------------------------------------------------------------ #
    async def index(self, request):
        from aiohttp import web
        html = MJPEG_VIEWER_HTML.replace("__CLIPS__", json.dumps(self.clips))
        return web.Response(content_type="text/html", text=html)

    async def audio_file(self, request):
        from aiohttp import web
        basename = request.query.get("aud", "")
        path = self._resolve_clip(basename)
        if not path:
            return web.Response(status=404, text="not allowed")
        return web.FileResponse(
            path, headers={"Cache-Control": "no-cache, no-store, must-revalidate"}
        )

    async def stream(self, request):
        from aiohttp import web
        import cv2

        basename = request.query.get("aud", "")
        path = self._resolve_clip(basename)
        if not path:
            return web.Response(status=404, text="not allowed")

        # Serialize stream starts so two near-simultaneous clicks don't trample
        # stream_agent.state. Each new request tears the prior session down.
        async with self._lock:
            if self.session is not None:
                print("[realtime/mjpeg] tearing down previous session")
                self.session.stop()
                self.session = None

            # Per-request audio: load + resample to 16k mono float (cached).
            try:
                audio_16k = self._load_clip_16k(path)
            except Exception as e:
                return web.Response(status=500, text=f"audio load error: {e}")

            # Reset stream_agent state for a fresh playback (reuses ref encode).
            self.stream_agent.state = self.stream_agent.fm.make_initial_stream_state(
                self.stream_agent.t_r
            )
            self.stream_agent.audio_buffer.clear()
            self.stream_agent.audio_buffer_len = 0
            self.stream_agent.chunk_idx = 0

            loop = asyncio.get_event_loop()
            self.session = StreamSession(self.stream_agent, audio_16k, loop)
            self.session.start()
            session = self.session

        boundary = "imtalkerframe"
        resp = web.StreamResponse(
            status=200,
            reason="OK",
            headers={
                "Content-Type": f"multipart/x-mixed-replace; boundary={boundary}",
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Connection": "close",
            },
        )
        await resp.prepare(request)
        n_sent = 0
        try:
            while True:
                frame_np = await session.video_q.get()
                if frame_np is None:
                    break
                bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
                ok, jpg = cv2.imencode(
                    ".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 80]
                )
                if not ok:
                    continue
                head = (
                    f"--{boundary}\r\n"
                    f"Content-Type: image/jpeg\r\n"
                    f"Content-Length: {len(jpg)}\r\n\r\n"
                ).encode()
                await resp.write(head)
                await resp.write(jpg.tobytes())
                await resp.write(b"\r\n")
                n_sent += 1
        except (asyncio.CancelledError, ConnectionResetError):
            print(f"[realtime/mjpeg] client disconnected ({basename})")
        finally:
            if self.session is session:
                self.session.stop()
                self.session = None
            else:
                # A newer request already took over; just leave it.
                pass
            print(f"[realtime/mjpeg] {basename} end, frames_sent={n_sent}")
        return resp

    async def on_shutdown(self, app):
        if self.session is not None:
            self.session.stop()
            self.session = None

    def run(self) -> None:
        from aiohttp import web

        app = web.Application()
        app.router.add_get("/", self.index)
        app.router.add_get("/stream.mjpg", self.stream)
        app.router.add_get("/audio/file", self.audio_file)
        app.on_shutdown.append(self.on_shutdown)
        p = self.port
        print(f"[realtime/mjpeg] HTTP listening on {self.host}:{p}")
        print(f"[realtime/mjpeg] assets_dir = {self.assets_dir or '<none>'}")
        print(f"[realtime/mjpeg] clips      = {self.clips}")
        print(f"  → On the pod:           http://127.0.0.1:{p}/")
        print(f"  → RunPod proxy URL:     https://<pod-id>-{p}.proxy.runpod.net/")
        web.run_app(app, host=self.host, port=self.port, print=None)


class _AsyncByteSink:
    """File-like sink that forwards muxed bytes into an asyncio queue."""

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


class FMP4Server(MJPEGServer):
    async def index(self, request):
        from aiohttp import web

        html = FMP4_VIEWER_HTML.replace("__CLIPS__", json.dumps(self.clips))
        return web.Response(content_type="text/html", text=html)

    def _add_video_stream(self, container, fps):
        # Defensive: PyAV's to_avrational rejects bare floats with
        # AttributeError: 'float' object has no attribute 'numerator'.
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
        print(f"[realtime/fmp4] video encoder = {codec_name}")
        return vstream

    async def stream(self, request):
        from aiohttp import web
        from av.audio.frame import AudioFrame
        from av.video.frame import VideoFrame
        import av

        basename = request.query.get("aud", "")
        path = self._resolve_clip(basename)
        if not path:
            return web.Response(status=404, text="not allowed")

        async with self._lock:
            if self.session is not None:
                print("[realtime/fmp4] tearing down previous session")
                self.session.stop()
                self.session = None

            try:
                audio_16k = self._load_clip_16k(path)
            except Exception as e:
                return web.Response(status=500, text=f"audio load error: {e}")

            self.stream_agent.state = self.stream_agent.fm.make_initial_stream_state(
                self.stream_agent.t_r
            )
            self.stream_agent.audio_buffer.clear()
            self.stream_agent.audio_buffer_len = 0
            self.stream_agent.chunk_idx = 0

            loop = asyncio.get_event_loop()
            self.session = StreamSession(self.stream_agent, audio_16k, loop)
            self.session.start()
            session = self.session

        out_q: asyncio.Queue = asyncio.Queue()
        sink = _AsyncByteSink(asyncio.get_running_loop(), out_q)
        try:
            container = av.open(
                sink,
                mode="w",
                format="mp4",
                options={
                    "movflags": "frag_keyframe+empty_moov+default_base_moof+omit_tfhd_offset",
                    "frag_duration": "200000",
                },
            )

            # PyAV's add_stream(rate=...) → to_avrational expects an int/Fraction,
            # not a Python float. self.stream_agent.fps comes from --fps (float,
            # default 25.0), so coerce here. The same int is reused for the
            # video PTS time_base inside pump() below.
            fps = int(round(float(self.stream_agent.fps)))
            sr = AUDIO_SAMPLE_RATE
            samples_per_pkt = AUDIO_SAMPLES_PER_PKT
            vstream = self._add_video_stream(container, fps)
            astream = container.add_stream("aac", rate=sr)
            astream.layout = "mono"
            astream.bit_rate = 96_000
        except Exception:
            if self.session is session:
                self.session.stop()
                self.session = None
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
            video_done = False
            audio_done = False
            frames_sent = 0
            try:
                while not (video_done and audio_done):
                    got = False

                    if not audio_done:
                        try:
                            pkt = session.audio_q.get_nowait()
                            got = True
                            if pkt is None:
                                audio_done = True
                            else:
                                af = AudioFrame.from_ndarray(
                                    pkt.reshape(1, -1), format="s16", layout="mono"
                                )
                                af.sample_rate = sr
                                af.pts = a_pts
                                af.time_base = fractions.Fraction(1, sr)
                                a_pts += samples_per_pkt
                                for p in astream.encode(af):
                                    container.mux(p)
                        except asyncio.QueueEmpty:
                            pass

                    if not video_done:
                        try:
                            vf_np = session.video_q.get_nowait()
                            got = True
                            if vf_np is None:
                                video_done = True
                            else:
                                vf = VideoFrame.from_ndarray(vf_np, format="rgb24")
                                vf.pts = v_pts
                                vf.time_base = fractions.Fraction(1, fps)
                                v_pts += 1
                                frames_sent += 1
                                for p in vstream.encode(vf):
                                    container.mux(p)
                        except asyncio.QueueEmpty:
                            pass

                    if not got:
                        await asyncio.sleep(0.002)
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
                print(f"[realtime/fmp4] {basename} end, frames_sent={frames_sent}")

        pumper = asyncio.create_task(pump())
        try:
            while True:
                chunk = await out_q.get()
                if chunk is None:
                    break
                await resp.write(chunk)
        except (asyncio.CancelledError, ConnectionResetError):
            print(f"[realtime/fmp4] client disconnected ({basename})")
        finally:
            pumper.cancel()
            with contextlib.suppress(Exception):
                await pumper
            if self.session is session:
                self.session.stop()
                self.session = None
        return resp

    def run(self) -> None:
        from aiohttp import web

        app = web.Application()
        app.router.add_get("/", self.index)
        app.router.add_get("/stream.mp4", self.stream)
        app.on_shutdown.append(self.on_shutdown)
        p = self.port
        print(f"[realtime/fmp4] HTTP listening on {self.host}:{p}")
        print(f"[realtime/fmp4] assets_dir = {self.assets_dir or '<none>'}")
        print(f"[realtime/fmp4] clips      = {self.clips}")
        print(f"  → On the pod:           http://127.0.0.1:{p}/")
        print(f"  → RunPod proxy URL:     https://<pod-id>-{p}.proxy.runpod.net/")
        web.run_app(app, host=self.host, port=self.port, print=None)


# ====================================================================== #
# Local mode (cv2.imshow) — truly zero transport, no audio
# ====================================================================== #
def run_local(stream_agent: StreamingInferenceAgent, audio_source_16k: np.ndarray) -> None:
    import cv2

    samples_per_emit = stream_agent.samples_per_emit
    n_in = audio_source_16k.shape[0]
    fps = stream_agent.fps
    pos = 0
    n_emitted = 0
    wall_t0 = time.monotonic()
    print("[realtime/local] press 'q' or ESC to quit")
    while pos < n_in:
        end = min(pos + samples_per_emit, n_in)
        chunk = audio_source_16k[pos:end]
        full = chunk.shape[0] == samples_per_emit
        stream_agent.feed_audio(chunk)
        out = stream_agent.step() if full else stream_agent.flush()
        if out is None:
            break
        frames_np = _frames_to_uint8_hwc(out)
        for i in range(frames_np.shape[0]):
            target = wall_t0 + (n_emitted + 1) / fps
            while True:
                dt = target - time.monotonic()
                if dt <= 0:
                    break
                time.sleep(min(dt, 0.005))
            bgr = cv2.cvtColor(frames_np[i], cv2.COLOR_RGB2BGR)
            cv2.imshow("IMTalker", bgr)
            k = cv2.waitKey(1) & 0xFF
            if k in (ord("q"), 27):
                cv2.destroyAllWindows()
                return
            n_emitted += 1
        pos = end
    cv2.destroyAllWindows()


# ====================================================================== #
# main
# ====================================================================== #
def main() -> None:
    torch.set_float32_matmul_precision("high")

    base_parser = argparse.ArgumentParser(add_help=False)
    base_parser = InferenceOptions().initialize(base_parser)
    parser = argparse.ArgumentParser(parents=[base_parser])
    parser.add_argument(
        "--transport",
        choices=["fmp4", "mjpeg", "webrtc", "local"],
        default="fmp4",
        help="fmp4 (default, muxed H.264+AAC over HTTP, lip-synced); "
             "mjpeg (legacy, drifts); webrtc (needs TURN on RunPod); "
             "local=cv2.imshow window.",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument(
        "--ice_servers",
        type=str,
        default=None,
        help="JSON list of RTCIceServer dicts, e.g. "
             "'[{\"urls\":\"stun:stun.l.google.com:19302\"},"
             "{\"urls\":\"turns:turn.example.com:443?transport=tcp\","
             "\"username\":\"u\",\"credential\":\"p\"}]'. Required for cross-NAT WebRTC.",
    )
    parser.add_argument(
        "--assets_dir",
        type=str,
        default="./assets",
        help="Directory the browser viewer scans for selectable audio clips. "
             "Allowlisted by basename only. Default: ./assets",
    )
    opt = parser.parse_args()
    opt.rank, opt.ngpus = 0, 1

    if not opt.ref_path or not os.path.exists(opt.ref_path):
        raise SystemExit("--ref_path is required and must exist")
    if not opt.aud_path or not os.path.exists(opt.aud_path):
        raise SystemExit("--aud_path is required and must exist")

    print("[realtime] building inference agent...")
    inf = InferenceAgent(opt)

    print("[realtime] building streaming agent + reference encode...")
    stream_agent = StreamingInferenceAgent(
        inf,
        ref_path=opt.ref_path,
        chunk_frames=opt.chunk_frames,
        crop=opt.crop,
        a_cfg_scale=opt.a_cfg_scale,
        nfe=opt.nfe,
        seed=opt.seed,
        debug_stream=opt.debug_stream,
    )
    if not opt.no_warmup:
        t = stream_agent.warmup()
        print(f"[realtime] warmup={t * 1e3:.1f}ms")

    print("[realtime] loading audio source...")
    audio_native, sr_native = _load_audio_native(opt.aud_path)
    audio_16k = _resample_mono(audio_native, sr_native, stream_agent.sr).astype(
        np.float32, copy=False
    )

    if opt.transport == "local":
        run_local(stream_agent, audio_16k)
        return

    if opt.transport == "mjpeg":
        try:
            import aiohttp  # noqa: F401
        except ImportError as e:
            raise SystemExit(f"mjpeg transport requires aiohttp: pip install aiohttp\n{e}")
        # Note: audio_16k loaded above was only used for warmup; MJPEG mode picks
        # its audio per-request from --assets_dir based on the clicked clip.
        server = MJPEGServer(
            stream_agent,
            host=opt.host,
            port=opt.port,
            assets_dir=opt.assets_dir,
            default_audio_path=opt.aud_path,
            target_sr=stream_agent.sr,
        )
        server.run()
        return

    if opt.transport == "fmp4":
        try:
            import aiohttp  # noqa: F401
            import av  # noqa: F401
        except ImportError as e:
            raise SystemExit(
                "fmp4 transport requires aiohttp + av:\n"
                "  pip install aiohttp av\n"
                f"missing: {e}"
            )
        server = FMP4Server(
            stream_agent,
            host=opt.host,
            port=opt.port,
            assets_dir=opt.assets_dir,
            default_audio_path=opt.aud_path,
            target_sr=stream_agent.sr,
        )
        server.run()
        return

    # WebRTC path
    try:
        import aiohttp  # noqa: F401
        import aiortc  # noqa: F401
        import av  # noqa: F401
    except ImportError as e:
        raise SystemExit(
            "webrtc transport requires aiortc + aiohttp + av:\n"
            "  pip install aiortc aiohttp av\n"
            f"missing: {e}"
        )

    if opt.ice_servers:
        try:
            ice_servers = json.loads(opt.ice_servers)
            if not isinstance(ice_servers, list):
                raise ValueError("ice_servers must be a JSON list")
        except Exception as e:
            raise SystemExit(f"--ice_servers must be JSON list of RTCIceServer dicts: {e}")
    else:
        ice_servers = [{"urls": "stun:stun.l.google.com:19302"}]
        print(
            "[realtime] WARN: no --ice_servers given. STUN-only WebRTC will FAIL "
            "behind RunPod's HTTPS proxy (UDP not relayed). Use --transport fmp4 "
            "or supply a TURN server."
        )
    server = WebRTCServer(stream_agent, audio_16k, opt.host, opt.port, ice_servers)
    server.run()


if __name__ == "__main__":
    main()
