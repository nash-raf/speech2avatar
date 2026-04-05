"""
launch_live.py — combined AV Moshi + IMTalker launcher.

Open in browser:
    http://localhost:8998/        — combined avatar page
    http://localhost:8998/moshi   — hidden controller UI (debug/fallback)
"""

from __future__ import annotations

import os
import sys
import tarfile
import tempfile
import threading
from pathlib import Path

import torch

_MOSHI_REPO = Path(__file__).resolve().parent.parent / "moshi"
_MOSHI_PKG = _MOSHI_REPO / "moshi"
if _MOSHI_PKG.exists() and str(_MOSHI_PKG) not in sys.path:
    sys.path.insert(0, str(_MOSHI_PKG))

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generator.options.base_options import BaseOptions
from live_pipeline import LiveMoshiIMTalkerSession


class LaunchOptions(BaseOptions):
    def initialize(self, parser):
        super().initialize(parser)
        parser.set_defaults(audio_feat_dim=512, nfe=5, a_cfg_scale=1.0)
        parser.add_argument("--ref_path", required=True, type=str)
        parser.add_argument("--generator_path", required=True, type=str)
        parser.add_argument("--renderer_path", required=True, type=str)
        parser.add_argument("--hf_repo", default="kyutai/moshiko-pytorch-bf16", type=str)
        parser.add_argument("--moshi_weight", default=None, type=str)
        parser.add_argument("--mimi_weight", default=None, type=str)
        parser.add_argument("--tokenizer", default=None, type=str)
        parser.add_argument("--host", default="0.0.0.0", type=str)
        parser.add_argument("--port", default=8998, type=int)
        parser.add_argument("--output_dir", default=None, type=str)
        parser.add_argument("--device", default="cuda", type=str)
        parser.add_argument("--crop", action="store_true")
        parser.add_argument("--max_sentences", default=1, type=int,
                            help="Maximum number of sentences Moshi should speak per reply")
        parser.add_argument("--max_text_tokens", default=40, type=int,
                            help="Hard cap on Moshi text tokens per reply")
        parser.add_argument("--chunk_sec", default=1.5, type=float,
                            help="Live PCM accumulation window for progressive AV chunks")
        return parser


class AVSegmentStore:
    def __init__(self):
        self._lock = threading.RLock()
        self.idle_path: str | None = None
        self.idle_version = 0
        self.current_turn = -1
        self.done = True
        self.segments: dict[int, dict[int, str]] = {}

    def set_idle(self, path: str):
        with self._lock:
            self.idle_path = path
            self.idle_version += 1

    def start_turn_if_needed(self, turn_id: int):
        with self._lock:
            if self.current_turn != turn_id:
                self.current_turn = turn_id
                self.done = False
                self.segments[turn_id] = {}

    def add_segment(self, turn_id: int, segment_index: int, path: str):
        with self._lock:
            self.start_turn_if_needed(turn_id)
            self.segments.setdefault(turn_id, {})[segment_index] = path

    def finish_turn(self, turn_id: int):
        with self._lock:
            if self.current_turn == turn_id:
                self.done = True

    def state(self) -> dict:
        with self._lock:
            segment_indices = sorted(self.segments.get(self.current_turn, {}).keys())
            return {
                "idle_ready": self.idle_path is not None,
                "idle_version": self.idle_version,
                "turn_id": self.current_turn,
                "done": self.done,
                "segments": segment_indices,
            }

    def get_idle_path(self) -> str | None:
        with self._lock:
            return self.idle_path

    def get_segment_path(self, turn_id: int, segment_index: int) -> str | None:
        with self._lock:
            return self.segments.get(turn_id, {}).get(segment_index)


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
      width: min(92vw, 760px);
      aspect-ratio: 16 / 9;
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
    .controls {
      display: flex;
      gap: 12px;
      align-items: center;
    }
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
      <video id="avatar" playsinline autoplay muted loop></video>
      <div class="overlay">
        <div id="status">Press Start to begin. The hidden controller handles microphone capture.</div>
        <div class="controls">
          <button id="startBtn">Start</button>
          <button id="resetBtn">Reset</button>
        </div>
      </div>
    </div>
    <p class="hint">
      Speak after starting. The browser will stay silent until an avatar AV chunk is ready, and then you will hear only the combined avatar playback.
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

    let controllerStarted = false;
    let idleVersion = -1;
    let currentTurn = -1;
    let lastSeenSegment = -1;
    let turnDone = true;
    let pending = [];
    let loading = new Set();
    let playing = false;
    let controllerReloadScheduled = false;
    let lastControllerResetTurn = -1;
    let currentPlaybackUrl = null;

    function setStatus(text, mode) {
      statusEl.textContent = text;
      if (mode) modeEl.textContent = mode;
    }

    function revokePlaybackUrlIfNeeded() {
      if (currentPlaybackUrl && currentPlaybackUrl.startsWith('blob:')) {
        URL.revokeObjectURL(currentPlaybackUrl);
      }
      currentPlaybackUrl = null;
    }

    function clearPendingSegments() {
      for (const item of pending) {
        URL.revokeObjectURL(item.url);
      }
      pending = [];
    }

    async function ensureIdle(meta, opts) {
      if (!meta.idle_ready) return;
      const force = opts && opts.force;
      // pollState runs often; do not replace an active reply segment with idle. playNext passes force
      // when the reply queue is drained and we intentionally return to idle.
      if (!force && avatar.dataset.mode === 'speaking') return;
      if (idleVersion === meta.idle_version && avatar.dataset.mode === 'idle') return;
      idleVersion = meta.idle_version;
      avatar.pause();
      revokePlaybackUrlIfNeeded();
      avatar.src = `/api/idle_video?v=${meta.idle_version}`;
      currentPlaybackUrl = avatar.src;
      avatar.muted = true;
      avatar.loop = true;
      avatar.dataset.mode = 'idle';
      try { await avatar.play(); } catch (_) {}
      setStatus('Listening for your next turn…', 'Idle');
    }

    async function fetchSegment(turnId, segmentIndex) {
      const key = `${turnId}:${segmentIndex}`;
      if (loading.has(key)) return;
      loading.add(key);
      try {
        const res = await fetch(`/api/segment/${turnId}/${segmentIndex}?t=${Date.now()}`, { cache: 'no-store' });
        if (!res.ok) return;
        const blob = await res.blob();
        pending.push({ turnId, segmentIndex, url: URL.createObjectURL(blob) });
        pending.sort((a, b) => a.segmentIndex - b.segmentIndex);
      } finally {
        loading.delete(key);
      }
      // Kick playback after fetch completes (and loading count decremented)
      if (!playing) playNext();
    }

    async function playNext() {
      if (!pending.length) {
        // Don't declare turn complete while segment fetches are still in flight
        if (loading.size > 0) {
          playing = false;
          console.log('[avatar] playNext: pending empty but', loading.size, 'fetches in flight — waiting');
          return;
        }
        playing = false;
        if (turnDone) {
          console.log('[avatar] playNext: all segments played, turn done — returning to idle');
          await ensureIdle({ idle_ready: true, idle_version: idleVersion }, { force: true });
          maybeReloadController();
        }
        return;
      }
      const next = pending.shift();
      playing = true;
      avatar.pause();
      revokePlaybackUrlIfNeeded();
      avatar.src = next.url;
      currentPlaybackUrl = next.url;
      avatar.muted = false;
      avatar.loop = false;
      avatar.dataset.mode = 'speaking';
      console.log('[avatar] playNext: playing segment', next.segmentIndex, '| pending:', pending.length, '| loading:', loading.size);
      setStatus(`Playing reply chunk ${next.segmentIndex + 1}`, 'Speaking');
      try { await avatar.play(); } catch (_) {}
    }

    function startController() {
      if (controllerStarted) return;
      controllerStarted = true;
      controller.src = `/moshi#/?embed=1&ts=${Date.now()}`;
      setStatus('Controller connected. Speak into your mic.', 'Listening');
    }

    function maybeReloadController() {
      if (!controllerStarted || controllerReloadScheduled) return;
      if (currentTurn < 0 || lastControllerResetTurn === currentTurn) return;
      controllerReloadScheduled = true;
      lastControllerResetTurn = currentTurn;
      setTimeout(() => {
        controllerReloadScheduled = false;
        controller.src = `/moshi#/?embed=1&ts=${Date.now()}`;
        setStatus('Ready for another turn. Speak again.', 'Listening');
      }, 900);
    }

    async function pollState() {
      try {
        const res = await fetch(`/api/stream_state?ts=${Date.now()}`, { cache: 'no-store' });
        if (!res.ok) return;
        const meta = await res.json();
        // Only sync idle from polling when not actively playing a segment (see ensureIdle speaking guard).
        if (!playing) {
          await ensureIdle(meta);
        }
        if (meta.turn_id !== currentTurn) {
          clearPendingSegments();
          playing = false;
          currentTurn = meta.turn_id;
          lastSeenSegment = -1;
          turnDone = meta.done;
        } else {
          turnDone = meta.done;
        }
        if (meta.turn_id >= 0) {
          for (const segmentIndex of meta.segments) {
            if (segmentIndex > lastSeenSegment) {
              lastSeenSegment = segmentIndex;
              fetchSegment(meta.turn_id, segmentIndex);
            }
          }
        }
        if (!playing && pending.length) {
          playNext();
        }
        if (!playing && turnDone && currentTurn >= 0 && !pending.length && !loading.size) {
          maybeReloadController();
        }
      } catch (_) {}
    }

    avatar.addEventListener('ended', () => {
      playNext();
    });

    async function primeMicrophoneFromParentGesture() {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        stream.getTracks().forEach((t) => t.stop());
        console.info('[avatar] Microphone permission primed on parent (same origin as iframe).');
      } catch (e) {
        console.warn('[avatar] Parent could not acquire microphone — iframe auto-connect may fail:', e);
      }
    }

    function notifyIframeMicPrimed() {
      try {
        controller.contentWindow?.postMessage({ type: 'moshi-mic-primed' }, '*');
      } catch (_) {}
    }

    controller.addEventListener('load', () => notifyIframeMicPrimed());

    startBtn.addEventListener('click', async () => {
      await primeMicrophoneFromParentGesture();
      startController();
      await pollState();
    });

    resetBtn.addEventListener('click', async () => {
      clearPendingSegments();
      revokePlaybackUrlIfNeeded();
      currentTurn = -1;
      lastSeenSegment = -1;
      lastControllerResetTurn = -1;
      playing = false;
      controllerStarted = false;
      controller.src = 'about:blank';
      await ensureIdle({ idle_ready: true, idle_version }, { force: true });
      setStatus('Reset complete. Press Start to begin again.', 'Idle');
    });

    setInterval(pollState, 500);
    pollState();
  </script>
</body>
</html>
"""


def build_imtalker_session(opt, segments: AVSegmentStore) -> LiveMoshiIMTalkerSession:
    print("[launch] Loading IMTalker session…")
    session = LiveMoshiIMTalkerSession(
        opt,
        generator_path=opt.generator_path,
        renderer_path=opt.renderer_path,
        ref_path=opt.ref_path,
        crop=opt.crop,
        nfe=opt.nfe,
        a_cfg_scale=opt.a_cfg_scale,
        moshi_repo=str(_MOSHI_REPO),
        mimi_hf_repo=opt.hf_repo,
    )

    output_dir = opt.output_dir or os.path.join(tempfile.gettempdir(), "imtalker_stream_segments")
    os.makedirs(output_dir, exist_ok=True)
    idle_path = os.path.join(output_dir, "idle_loop.mp4")
    session.save_idle_video(idle_path)
    segments.set_idle(idle_path)

    def _on_segment(turn_id: int, segment_index: int, path: str):
        segments.add_segment(turn_id, segment_index, path)
        print(f"[IMTalker] turn {turn_id:04d} segment {segment_index:04d} ready → {path}")

    def _on_turn_done(turn_id: int):
        segments.finish_turn(turn_id)
        print(f"[IMTalker] turn {turn_id:04d} finished")

    session.segment_callback = _on_segment
    session.turn_done_callback = _on_turn_done
    print("[launch] IMTalker session ready.")
    return session


def build_moshi_state(opt, session: LiveMoshiIMTalkerSession):
    from moshi.models import loaders
    from moshi.server import ServerState, seed_all

    seed_all(42424242)

    print("[launch] Loading Moshi checkpoint…")
    checkpoint_info = loaders.CheckpointInfo.from_hf_repo(
        opt.hf_repo, opt.moshi_weight, opt.mimi_weight, opt.tokenizer,
    )

    text_tokenizer = checkpoint_info.get_text_tokenizer()

    print("[launch] Loading Moshi Mimi…")
    mimi = checkpoint_info.get_mimi(device=opt.device)
    print("[launch] Loading Moshi LM…")
    lm = checkpoint_info.get_moshi(device=opt.device, dtype=torch.bfloat16)

    state = ServerState(
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
        **checkpoint_info.lm_gen_config,
    )
    print("[launch] Warming up Moshi…")
    state.warmup()
    return state


def main():
    from aiohttp import web
    from huggingface_hub import hf_hub_download

    opt = LaunchOptions().parse()
    opt.rank = opt.device

    segments = AVSegmentStore()
    session = build_imtalker_session(opt, segments)
    state = build_moshi_state(opt, session)

    # Use locally-built Moshi client (built with base="/moshi/" and hash routing)
    _local_dist = Path(__file__).resolve().parent.parent / "moshi" / "client" / "dist"
    if _local_dist.exists():
        static_path = str(_local_dist)
        print(f"[launch] Using local Moshi client build: {static_path}")
    else:
        dist_tgz = Path(hf_hub_download("kyutai/moshi-artifacts", "dist.tgz"))
        dist = dist_tgz.parent / "dist"
        if not dist.exists():
            with tarfile.open(dist_tgz, "r:gz") as tar:
                tar.extractall(path=dist_tgz.parent)
        static_path = str(dist)
        print(f"[launch] WARNING: Using HF pre-built bundle (no hash routing fix)")

    app = web.Application()
    app.router.add_get("/api/chat", state.handle_chat)
    app.router.add_get("/api/stream_state", lambda r: web.json_response(segments.state()))
    app.router.add_get(
        "/api/idle_video",
        lambda r: web.FileResponse(segments.get_idle_path()) if segments.get_idle_path() else web.Response(status=404),
    )
    app.router.add_get(
        r"/api/segment/{turn:\d+}/{segment:\d+}",
        lambda r: web.FileResponse(
            segments.get_segment_path(int(r.match_info["turn"]), int(r.match_info["segment"]))
        ) if segments.get_segment_path(int(r.match_info["turn"]), int(r.match_info["segment"])) else web.Response(status=404),
    )

    app.router.add_get("/", lambda r: web.Response(text=_VIEWER_HTML, content_type="text/html"))
    app.router.add_get("/viewer", lambda r: web.Response(text=_VIEWER_HTML, content_type="text/html"))
    app.router.add_get("/moshi", lambda r: web.FileResponse(os.path.join(static_path, "index.html")))
    app.router.add_get("/moshi/", lambda r: web.FileResponse(os.path.join(static_path, "index.html")))
    app.router.add_static("/moshi/assets", path=os.path.join(static_path, "assets"),
                          follow_symlinks=True, name="moshi_static")

    print(f"\n[launch] Combined avatar → http://localhost:{opt.port}/")
    print(f"[launch] Hidden Moshi UI  → http://localhost:{opt.port}/moshi\n")
    web.run_app(app, host=opt.host, port=opt.port)


if __name__ == "__main__":
    main()
