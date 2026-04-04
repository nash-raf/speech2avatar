"""
launch_live.py — Moshi + IMTalker live bridge launcher.

Usage:
    python launch_live.py \
        --ref_path assets/source_5.png \
        --generator_path checkpoints/generator.ckpt \
        --renderer_path checkpoints/renderer.ckpt \
        --hf_repo kyutai/moshiko-pytorch-bf16 \
        [--output_dir /tmp/live_chunks] \
        [--host 0.0.0.0] [--port 8998]

Open in browser:
    http://localhost:8998          — Moshi voice chat UI
    http://localhost:8998/viewer   — IMTalker live face stream
"""

from __future__ import annotations

import asyncio
import io
import os
import sys
import tarfile
from pathlib import Path

import torch
from PIL import Image

# ── make moshi importable ──────────────────────────────────────────────────
_MOSHI_REPO = Path(__file__).resolve().parent.parent / "moshi"
_MOSHI_PKG  = _MOSHI_REPO / "moshi"
if _MOSHI_PKG.exists() and str(_MOSHI_PKG) not in sys.path:
    sys.path.insert(0, str(_MOSHI_PKG))

# ── imtalker imports ───────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))
from generator.options.base_options import BaseOptions
from live_pipeline import LiveMoshiIMTalkerSession


# ──────────────────────────────────────────────────────────────────────────
# Options
# ──────────────────────────────────────────────────────────────────────────
class LaunchOptions(BaseOptions):
    def initialize(self, parser):
        super().initialize(parser)
        parser.add_argument("--ref_path",       required=True,  type=str)
        parser.add_argument("--generator_path", required=True,  type=str)
        parser.add_argument("--renderer_path",  required=True,  type=str)
        parser.add_argument("--hf_repo",        default="kyutai/moshiko-pytorch-bf16", type=str)
        parser.add_argument("--moshi_weight",   default=None,   type=str)
        parser.add_argument("--mimi_weight",    default=None,   type=str)
        parser.add_argument("--tokenizer",      default=None,   type=str)
        parser.add_argument("--host",           default="0.0.0.0", type=str)
        parser.add_argument("--port",           default=8998,   type=int)
        parser.add_argument("--nfe",            default=5,      type=int)
        parser.add_argument("--a_cfg_scale",    default=1.0,    type=float)
        parser.add_argument("--output_dir",     default=None,   type=str,
                            help="Also save each chunk as chunk_NNNN.mp4 here")
        parser.add_argument("--device",         default="cuda", type=str)
        parser.add_argument("--crop",           action="store_true")
        parser.add_argument("--jpeg_quality",   default=75,     type=int,
                            help="JPEG quality for MJPEG stream (1-95)")
        return parser


# ──────────────────────────────────────────────────────────────────────────
# MJPEG frame broadcaster
# ──────────────────────────────────────────────────────────────────────────
class MJPEGBroadcaster:
    """Holds a set of active streaming responses and pushes JPEG frames to all."""

    def __init__(self, jpeg_quality: int = 75):
        self.jpeg_quality = jpeg_quality
        self._clients: set[asyncio.Queue] = set()
        self._last_jpeg: bytes | None = None   # shown to clients that connect late

    def _tensor_to_jpeg(self, frame_chw: torch.Tensor) -> bytes:
        """frame_chw: [C, H, W] float in [-1, 1]."""
        frame = frame_chw.clamp(-1, 1).add(1).mul(127.5).byte()
        arr = frame.permute(1, 2, 0).cpu().numpy()        # [H, W, C]
        img = Image.fromarray(arr, mode="RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=self.jpeg_quality)
        return buf.getvalue()

    def push_chunk(self, frames: torch.Tensor):
        """frames: [T, C, H, W]. Pushes every frame to all connected clients."""
        for t in range(frames.shape[0]):
            jpeg = self._tensor_to_jpeg(frames[t])
            self._last_jpeg = jpeg
            dead = set()
            for q in self._clients:
                try:
                    q.put_nowait(jpeg)
                except asyncio.QueueFull:
                    dead.add(q)   # slow client — drop it
            self._clients -= dead

    async def stream_response(self, request):
        """aiohttp handler — streams MJPEG to one client."""
        from aiohttp import web
        resp = web.StreamResponse()
        resp.content_type = "multipart/x-mixed-replace; boundary=frame"
        await resp.prepare(request)

        q: asyncio.Queue = asyncio.Queue(maxsize=30)
        self._clients.add(q)

        # Send the last known frame immediately so the browser isn't blank
        if self._last_jpeg is not None:
            await _write_jpeg(resp, self._last_jpeg)

        try:
            while True:
                jpeg = await q.get()
                await _write_jpeg(resp, jpeg)
        except (ConnectionResetError, asyncio.CancelledError):
            pass
        finally:
            self._clients.discard(q)
        return resp


async def _write_jpeg(resp, jpeg: bytes):
    header = (
        b"--frame\r\n"
        b"Content-Type: image/jpeg\r\n"
        b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n"
    )
    await resp.write(header + jpeg + b"\r\n")


# ──────────────────────────────────────────────────────────────────────────
# Viewer HTML (served at /viewer)
# ──────────────────────────────────────────────────────────────────────────
_VIEWER_HTML = """\
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>IMTalker Live</title>
  <style>
    body  { margin:0; background:#111; display:flex; align-items:center; justify-content:center; height:100vh; }
    img   { max-width:90vw; max-height:90vh; border-radius:8px; box-shadow:0 0 30px #000; }
    p     { color:#888; font-family:monospace; text-align:center; margin-top:12px; font-size:13px; }
  </style>
</head>
<body>
  <div>
    <img id="feed" src="/api/video_feed" alt="Waiting for IMTalker…">
    <p>IMTalker live output &mdash; speak into Moshi to animate the face</p>
  </div>
  <script>
    // If MJPEG drops, auto-reconnect after 2 s
    const img = document.getElementById('feed');
    img.onerror = () => setTimeout(() => { img.src = '/api/video_feed?' + Date.now(); }, 2000);
  </script>
</body>
</html>
"""


# ──────────────────────────────────────────────────────────────────────────
# Build session + moshi state
# ──────────────────────────────────────────────────────────────────────────
def build_imtalker_session(opt, broadcaster: MJPEGBroadcaster) -> LiveMoshiIMTalkerSession:
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

    if opt.output_dir:
        os.makedirs(opt.output_dir, exist_ok=True)

    def _on_chunk(chunk):
        # Push every frame to MJPEG stream
        broadcaster.push_chunk(chunk.frames)
        # Optionally save to disk too
        if opt.output_dir:
            out = os.path.join(opt.output_dir, f"chunk_{chunk.chunk_index:04d}.mp4")
            session.save_chunk_video(chunk, out)
            print(f"[IMTalker] chunk {chunk.chunk_index:04d} → {out}")
        else:
            print(f"[IMTalker] chunk {chunk.chunk_index:04d} rendered ({chunk.frames.shape[0]} frames)")

    session.frame_callback = _on_chunk
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

    mimi = session.mimi   # reuse — no double load
    text_tokenizer = checkpoint_info.get_text_tokenizer()

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
        **checkpoint_info.lm_gen_config,
    )
    print("[launch] Warming up Moshi…")
    state.warmup()
    return state


# ──────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────
def main():
    from aiohttp import web
    from huggingface_hub import hf_hub_download

    opt = LaunchOptions().parse()
    opt.rank = opt.device

    broadcaster = MJPEGBroadcaster(jpeg_quality=opt.jpeg_quality)
    session     = build_imtalker_session(opt, broadcaster)
    state       = build_moshi_state(opt, session)

    # ── Moshi web UI static bundle ─────────────────────────────────────────
    dist_tgz = Path(hf_hub_download("kyutai/moshi-artifacts", "dist.tgz"))
    dist = dist_tgz.parent / "dist"
    if not dist.exists():
        with tarfile.open(dist_tgz, "r:gz") as tar:
            tar.extractall(path=dist_tgz.parent)
    static_path = str(dist)

    # ── aiohttp routes ─────────────────────────────────────────────────────
    app = web.Application()

    # Moshi voice chat
    app.router.add_get("/api/chat", state.handle_chat)
    app.router.add_get("/", lambda r: web.FileResponse(os.path.join(static_path, "index.html")))
    app.router.add_static("/assets", path=os.path.join(static_path, "assets"),
                          follow_symlinks=True, name="moshi_static")

    # IMTalker live face stream
    app.router.add_get("/api/video_feed", broadcaster.stream_response)
    app.router.add_get("/viewer", lambda r: web.Response(text=_VIEWER_HTML, content_type="text/html"))

    print(f"\n[launch] Moshi chat  → http://localhost:{opt.port}/")
    print(f"[launch] Face stream → http://localhost:{opt.port}/viewer\n")
    web.run_app(app, host=opt.host, port=opt.port)


if __name__ == "__main__":
    main()
