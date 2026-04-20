"""
launch_live_webrtc.py - parallel WebRTC launcher for Moshi + IMTalker.

Expected layout:
    /workspace/IMTalker
    /workspace/moshi

Open in browser:
    http://localhost:8998/        - combined avatar page (WebRTC)
    http://localhost:8998/moshi   - hidden controller UI (debug/fallback)
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import tarfile
from pathlib import Path

from aiohttp import web
from huggingface_hub import hf_hub_download

from launch_live import (
    LaunchOptions,
    TranscriptStore,
    _MOSHI_REPO,
    build_imtalker_session,
    build_moshi_state,
)
from live_pipeline_webrtc import WebRTCStreamSession

try:
    from aiortc import (
        RTCConfiguration,
        RTCIceServer,
        RTCPeerConnection,
        RTCSessionDescription,
    )
except Exception:  # pragma: no cover - runtime dependency check
    RTCConfiguration = None
    RTCIceServer = None
    RTCPeerConnection = None
    RTCSessionDescription = None


def _default_ice_servers() -> list:
    return [{"urls": "stun:stun.l.google.com:19302"}]


def load_ice_servers() -> list:
    """W3C-shaped list for browser + aiortc. Override with WEBRTC_ICE_SERVERS_JSON env (JSON array)."""
    raw = os.environ.get("WEBRTC_ICE_SERVERS_JSON", "").strip()
    if not raw:
        return _default_ice_servers()
    try:
        parsed = json.loads(raw)
        if not isinstance(parsed, list):
            raise ValueError("expected a JSON array")
        return parsed
    except Exception as e:
        print(f"[launch/webrtc] WEBRTC_ICE_SERVERS_JSON parse error ({e}); using default STUN")
        return _default_ice_servers()


def _make_rtc_configuration() -> "RTCConfiguration":
    ice = load_ice_servers()
    rtc_ice = [
        RTCIceServer(
            urls=s["urls"],
            username=s.get("username"),
            credential=s.get("credential"),
        )
        for s in ice
    ]
    return RTCConfiguration(iceServers=rtc_ice)


_WEBRTC_VIEWER_HTML = """\
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Moshi + IMTalker (WebRTC)</title>
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
    .shell { width: min(92vw, 980px); display: grid; gap: 16px; justify-items: center; }
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
      <video id="avatar" playsinline autoplay></video>
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
      WebRTC transport for low-latency avatar streaming. Same Moshi controller and
      IMTalker render path as the fMP4 version, but media is delivered over a peer
      connection instead of MSE/fMP4 buffering.
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

    let controllerStarted = false;
    let lastTranscriptTurnId = -1;
    let lastTranscriptVersion = -1;
    let currentPc = null;
    let remoteStream = null;
    let lastDebugSnapshot = '';

    function setTranscript(text) {
      transcriptTextEl.textContent = text || 'Assistant text will appear here as Moshi responds.';
    }
    function setStatus(text, mode) {
      statusEl.textContent = text;
      if (mode) modeEl.textContent = mode;
    }

    function logAvatarVideoState(tag) {
      console.log('[avatar]', tag, {
        paused: avatar.paused,
        readyState: avatar.readyState,
        videoWidth: avatar.videoWidth,
        videoHeight: avatar.videoHeight,
      });
    }

    async function tryPlayAvatar(reason) {
      try {
        await avatar.play();
        logAvatarVideoState('play ok (' + reason + ')');
      } catch (e) {
        console.warn('[avatar] play failed', reason, e);
      }
    }

    avatar.addEventListener('loadedmetadata', () => {
      logAvatarVideoState('loadedmetadata');
      tryPlayAvatar('loadedmetadata');
    });

    async function waitForIceGatheringComplete(pc) {
      if (pc.iceGatheringState === 'complete') return;
      await new Promise((resolve) => {
        function checkState() {
          if (pc.iceGatheringState === 'complete') {
            pc.removeEventListener('icegatheringstatechange', checkState);
            resolve();
          }
        }
        pc.addEventListener('icegatheringstatechange', checkState);
      });
    }

    async function logDebugIceStats(tag) {
      if (!DEBUG_VIEW || !currentPc) return;
      try {
        const report = await currentPc.getStats();
        const byId = {};
        report.forEach((s) => {
          byId[s.id] = s;
        });
        const pairs = [];
        let videoInbound = null;
        report.forEach((s) => {
          if (s.type === 'candidate-pair') {
            const loc = s.localCandidateId ? byId[s.localCandidateId] : null;
            const rem = s.remoteCandidateId ? byId[s.remoteCandidateId] : null;
            pairs.push({
              state: s.state,
              nominated: s.nominated,
              bytesReceived: s.bytesReceived,
              localType: loc?.candidateType,
              remoteType: rem?.candidateType,
            });
          }
          if (s.type === 'inbound-rtp' && s.kind === 'video') {
            videoInbound = {
              framesDecoded: s.framesDecoded,
              bytesReceived: s.bytesReceived,
              packetsReceived: s.packetsReceived,
            };
          }
        });
        console.log('[webrtc] stats', Date.now(), tag, {
          iceConnectionState: currentPc.iceConnectionState,
          connectionState: currentPc.connectionState,
          candidatePairs: pairs,
          videoInbound,
        });
      } catch (e) {
        console.warn('[webrtc] stats snapshot failed', tag, e);
      }
    }

    async function startWebRTC() {
      if (currentPc) return;
      setStatus('Negotiating WebRTC...', 'Connecting');
      remoteStream = new MediaStream();
      currentPc = new RTCPeerConnection({ iceServers: __ICE_SERVERS__ });
      currentPc.addTransceiver('video', { direction: 'recvonly' });
      currentPc.addTransceiver('audio', { direction: 'recvonly' });
      currentPc.onicegatheringstatechange = () => {
        console.log('[webrtc]', Date.now(), 'iceGatheringState', currentPc.iceGatheringState);
      };
      currentPc.oniceconnectionstatechange = () => {
        console.log('[webrtc]', Date.now(), 'iceConnectionState', currentPc.iceConnectionState);
        logDebugIceStats('after iceConnectionState');
      };
      currentPc.onicecandidate = (ev) => {
        if (!DEBUG_VIEW) return;
        if (ev.candidate) console.log('[webrtc] localCandidate', ev.candidate.candidate);
        else console.log('[webrtc] localCandidate (null, gathering done)');
      };
      currentPc.ontrack = (event) => {
        remoteStream.addTrack(event.track);
        avatar.srcObject = remoteStream;
        if (DEBUG_VIEW) console.log('[webrtc] ontrack', event.track.kind, 'streams=', event.streams?.length);
        logAvatarVideoState('after ontrack (' + event.track.kind + ')');
        tryPlayAvatar('ontrack');
        setStatus('Streaming. Speak any time.', 'Listening');
      };
      currentPc.onconnectionstatechange = () => {
        const state = currentPc.connectionState;
        console.log('[webrtc]', Date.now(), 'connectionState', state);
        logDebugIceStats('after connectionState');
        if (state === 'connected') {
          startController();
        }
        if (state === 'failed' || state === 'closed' || state === 'disconnected') {
          setStatus('WebRTC ' + state + '.', 'Idle');
        }
      };

      const offer = await currentPc.createOffer();
      await currentPc.setLocalDescription(offer);
      await waitForIceGatheringComplete(currentPc);

      const resp = await fetch('/offer', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          sdp: currentPc.localDescription.sdp,
          type: currentPc.localDescription.type,
        }),
      });
      if (!resp.ok) {
        const text = await resp.text();
        throw new Error(text || ('HTTP ' + resp.status));
      }
      const answer = await resp.json();
      await currentPc.setRemoteDescription(answer);
      if (DEBUG_VIEW) {
        setTimeout(() => logDebugIceStats('t+12s post setRemoteDescription'), 12000);
      }
    }

    function closeWebRTC() {
      if (currentPc) {
        try { currentPc.close(); } catch (_) {}
        currentPc = null;
      }
      if (avatar.srcObject) {
        avatar.pause();
        avatar.srcObject = null;
      }
      remoteStream = null;
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
            console.log('[webrtc] stream_state', meta);
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
      try {
        await primeMicrophoneFromParentGesture();
        avatar.muted = false;
        await startWebRTC();
      } catch (e) {
        console.error(e);
        setStatus('WebRTC error: ' + e.message, 'Error');
        closeWebRTC();
      }
    });

    resetBtn.addEventListener('click', () => {
      closeWebRTC();
      controllerStarted = false;
      controller.src = 'about:blank';
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


def _webrtc_viewer_html() -> str:
    return _WEBRTC_VIEWER_HTML.replace("__ICE_SERVERS__", json.dumps(load_ice_servers()))


def _resolve_static_path(opt) -> str:
    local_dist = Path(opt.moshi_repo) / "client" / "dist"
    if local_dist.exists():
        static_path = str(local_dist)
        print(f"[launch] Using local Moshi client build: {static_path}")
        return static_path

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
    return static_path


def _make_stream_state_handler(session, transcript, state):
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

    return _stream_state_response


async def _serve_moshi_index(_request, static_path: str):
    index_path = os.path.join(static_path, "index.html")
    html = Path(index_path).read_text(encoding="utf-8")
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


async def _handle_offer(
    request: web.Request,
    session,
    rtc_lock: asyncio.Lock,
    current_peer: dict,
):
    if (
        RTCPeerConnection is None
        or RTCSessionDescription is None
        or RTCConfiguration is None
        or RTCIceServer is None
    ):
        return web.Response(
            status=500,
            text="aiortc is not installed. Install with: pip install aiortc",
        )

    params = await request.json()
    if "sdp" not in params or "type" not in params:
        return web.Response(status=400, text="Expected JSON offer with sdp and type")

    loop = asyncio.get_running_loop()
    fps = int(round(float(getattr(session.opt, "fps", 25))))
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    async with rtc_lock:
        old_pc = current_peer.get("pc")
        old_sink = current_peer.get("sink")
        if old_sink is not None:
            old_sink.stop()
        if old_pc is not None:
            with contextlib.suppress(Exception):
                await old_pc.close()
        current_peer.clear()

        session.reset_reply()
        sink = WebRTCStreamSession(
            loop=loop,
            idle_frame_uint8=session.idle_frame_uint8,
            fps=fps,
            debug=bool(getattr(session.opt, "debug_session", False)),
        )
        sink.start()
        session.av_session = sink

        ice_debug = bool(getattr(session.opt, "debug_session", False))
        pc = RTCPeerConnection(configuration=_make_rtc_configuration())
        current_peer["pc"] = pc
        current_peer["sink"] = sink

        @pc.on("icegatheringstatechange")
        async def on_icegatheringstatechange():
            print(f"[launch/webrtc] iceGatheringState={pc.iceGatheringState}")

        @pc.on("iceconnectionstatechange")
        async def on_iceconnectionstatechange():
            print(f"[launch/webrtc] iceConnectionState={pc.iceConnectionState}")

        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            state = pc.connectionState
            print(f"[launch/webrtc] connectionState={state}")
            if state in {"failed", "closed", "disconnected"}:
                async with rtc_lock:
                    if current_peer.get("pc") is pc:
                        current_peer.clear()
                        if session.av_session is sink:
                            session.av_session = None
                    sink.stop()
                with contextlib.suppress(Exception):
                    await pc.close()

        pc.addTrack(sink.video_track)
        pc.addTrack(sink.audio_track)
        await pc.setRemoteDescription(offer)
        await pc.setLocalDescription(await pc.createAnswer())
        while pc.iceGatheringState != "complete":
            await asyncio.sleep(0.02)
        if ice_debug:
            sdp_out = pc.localDescription.sdp
            n_cand = sum(1 for line in sdp_out.splitlines() if line.startswith("a=candidate"))
            print(f"[launch/webrtc] answer SDP: {n_cand} candidate lines, {len(sdp_out)} bytes")

        return web.json_response(
            {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
        )


def main():
    opt = LaunchOptions().parse()
    opt.rank = opt.device

    transcript = TranscriptStore()
    session = build_imtalker_session(opt)
    state = build_moshi_state(opt, session, transcript)
    static_path = _resolve_static_path(opt)

    rtc_lock = asyncio.Lock()
    current_peer: dict = {}

    app = web.Application()
    app.router.add_get("/api/chat", state.handle_chat)
    app.router.add_get("/api/stream_state", _make_stream_state_handler(session, transcript, state))

    async def handle_offer(request):
        return await _handle_offer(request, session, rtc_lock, current_peer)

    async def serve_moshi_index(request):
        return await _serve_moshi_index(request, static_path)

    app.router.add_post("/offer", handle_offer)

    async def on_shutdown(_app):
        sink = current_peer.get("sink")
        pc = current_peer.get("pc")
        if sink is not None:
            sink.stop()
        if pc is not None:
            with contextlib.suppress(Exception):
                await pc.close()
        current_peer.clear()
        session.av_session = None

    app.on_shutdown.append(on_shutdown)

    async def serve_webrtc_viewer(_request):
        return web.Response(text=_webrtc_viewer_html(), content_type="text/html")

    app.router.add_get("/", serve_webrtc_viewer)
    app.router.add_get("/viewer", serve_webrtc_viewer)
    app.router.add_get("/moshi", serve_moshi_index)
    app.router.add_get("/moshi/", serve_moshi_index)
    app.router.add_get("/moshi/index.html", serve_moshi_index)
    app.router.add_static(
        "/moshi/assets",
        path=os.path.join(static_path, "assets"),
        follow_symlinks=True,
        name="moshi_static",
    )
    app.router.add_static(
        "/assets",
        path=os.path.join(static_path, "assets"),
        follow_symlinks=True,
        name="moshi_static_root_assets",
    )
    favicon_path = os.path.join(static_path, "favicon.ico")
    if os.path.exists(favicon_path):
        app.router.add_get(
            "/favicon.ico",
            lambda _r, p=favicon_path: web.FileResponse(path=p),
        )

    print(f"[launch] Combined avatar -> http://localhost:{opt.port}/")
    print(f"[launch] Hidden Moshi UI  -> http://localhost:{opt.port}/moshi\n")
    web.run_app(app, host=opt.host, port=opt.port)


if __name__ == "__main__":
    main()
