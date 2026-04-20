"""
launch_live_ws.py - phase 2 WebSocket transport launcher for Moshi + IMTalker.

Expected layout:
    /workspace/IMTalker
    /workspace/moshi

Open in browser:
    http://localhost:9000/        - WS viewer page
    http://localhost:9000/moshi   - legacy Moshi controller UI (debug only)
"""

from __future__ import annotations

import asyncio
import contextlib
import os
from pathlib import Path

from aiohttp import ClientConnectionResetError, web

from launch_live import LaunchOptions, TranscriptStore, build_imtalker_session, build_moshi_state
from launch_live_webrtc import _resolve_static_path, _serve_moshi_index
from live_pipeline_ws import WSStreamSession

_WS_VIEWER_HTML = """\
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Moshi + IMTalker (WS Phase 2)</title>
  <style>
    :root { color-scheme: dark; }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      background:
        radial-gradient(circle at top, rgba(44,108,138,0.35), transparent 45%),
        linear-gradient(180deg, #07090c 0%, #11161d 100%);
      font-family: "IBM Plex Mono", monospace;
      color: #d8e2ea;
    }
    .shell {
      width: min(96vw, 1240px);
      margin: 24px auto 40px;
      display: grid;
      gap: 18px;
    }
    .row {
      display: grid;
      grid-template-columns: minmax(480px, 700px) minmax(280px, 1fr);
      gap: 18px;
      align-items: start;
    }
    .row.single {
      grid-template-columns: 1fr;
    }
    .panel {
      border-radius: 18px;
      border: 1px solid rgba(216,226,234,0.18);
      background: rgba(0, 0, 0, 0.5);
      padding: 18px 20px;
      box-shadow: 0 30px 60px rgba(0,0,0,0.35);
    }
    .status {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 8px 12px;
      border-radius: 999px;
      border: 1px solid rgba(216,226,234,0.18);
      background: rgba(255,255,255,0.04);
      font-size: 12px;
      width: fit-content;
    }
    .dot {
      width: 8px;
      height: 8px;
      border-radius: 999px;
      background: #d3a34f;
      box-shadow: 0 0 12px rgba(211,163,79,0.75);
    }
    .dot.live {
      background: #4fd38f;
      box-shadow: 0 0 12px rgba(79,211,143,0.75);
    }
    .controls {
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
      align-items: center;
      margin-bottom: 14px;
    }
    button, a.button {
      border: 1px solid rgba(216,226,234,0.35);
      border-radius: 999px;
      background: rgba(255,255,255,0.05);
      color: #ecf2f7;
      padding: 11px 18px;
      font: inherit;
      cursor: pointer;
      text-decoration: none;
      display: inline-flex;
      align-items: center;
      justify-content: center;
    }
    button:hover, a.button:hover { background: rgba(255,255,255,0.12); }
    button:disabled { opacity: 0.5; cursor: default; }
    .stage {
      position: relative;
      width: 100%;
      aspect-ratio: 1 / 1;
      border-radius: 18px;
      overflow: hidden;
      border: 1px solid rgba(216,226,234,0.12);
      background: #000;
      display: grid;
      place-items: center;
    }
    canvas {
      width: 100%;
      height: 100%;
      display: block;
      background: #000;
    }
    .stage-note {
      position: absolute;
      left: 14px;
      bottom: 14px;
      padding: 6px 10px;
      border-radius: 999px;
      background: rgba(0,0,0,0.55);
      border: 1px solid rgba(216,226,234,0.14);
      font-size: 12px;
      color: #d8e2ea;
      pointer-events: none;
    }
    .grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(120px, 1fr));
      gap: 12px;
    }
    .metric {
      border-radius: 12px;
      border: 1px solid rgba(216,226,234,0.14);
      background: rgba(255,255,255,0.03);
      padding: 12px 14px;
    }
    .metric-label {
      font-size: 11px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: #9caab5;
      margin-bottom: 8px;
    }
    .metric-value {
      font-size: 20px;
      color: #ecf2f7;
    }
    .transcript {
      min-height: 128px;
      white-space: pre-wrap;
      line-height: 1.6;
      color: #ecf2f7;
    }
    pre {
      margin: 0;
      min-height: 180px;
      max-height: 280px;
      padding: 14px 16px;
      border-radius: 14px;
      border: 1px solid rgba(216,226,234,0.16);
      background: rgba(255,255,255,0.04);
      color: #ecf2f7;
      white-space: pre-wrap;
      line-height: 1.5;
      overflow: auto;
    }
    .hint {
      font-size: 12px;
      color: #9caab5;
      line-height: 1.6;
    }
    .debug-hidden {
      display: none;
    }
    @media (max-width: 980px) {
      .row {
        grid-template-columns: 1fr;
      }
    }
  </style>
</head>
<body>
  <div class="shell">
    <div class="status"><span id="dot" class="dot"></span><span id="mode">Idle</span></div>
    <div class="row">
      <div class="panel">
        <div class="controls">
          <button id="connectBtn">Connect</button>
          <button id="disconnectBtn">Disconnect</button>
        </div>
        <div class="stage">
          <canvas id="avatarCanvas" width="512" height="512"></canvas>
          <div class="stage-note" id="stageNote">Waiting for stream</div>
        </div>
      </div>
      <div class="panel">
        <div class="grid">
          <div class="metric"><div class="metric-label">Video Packets</div><div class="metric-value" id="videoPackets">0</div></div>
          <div class="metric"><div class="metric-label">Audio Packets</div><div class="metric-value" id="audioPackets">0</div></div>
          <div class="metric"><div class="metric-label">Frames Decoded</div><div class="metric-value" id="framesDecoded">0</div></div>
          <div class="metric"><div class="metric-label">Frames Rendered</div><div class="metric-value" id="framesRendered">0</div></div>
          <div class="metric"><div class="metric-label">Video Queue</div><div class="metric-value" id="videoQueueDepth">0</div></div>
          <div class="metric"><div class="metric-label">A/V Drift</div><div class="metric-value" id="driftMs">n/a</div></div>
        </div>
      </div>
    </div>
    <div class="row single">
      <div class="panel">
        <div class="metric-label">Assistant Transcript</div>
        <div class="transcript" id="transcriptBox">Waiting for transcript...</div>
      </div>
      <!-- Keep the WS log mounted but hidden for public demos. Remove `debug-hidden`
           below when you want the debug panel back. -->
      <div class="panel debug-hidden">
        <div class="metric-label">WS Log</div>
        <pre id="logBox">Phase 2 viewer ready. Connect to begin decode.</pre>
      </div>
    </div>
    <!-- Public-demo cleanup: footer copy intentionally hidden for now. Restore the
         paragraph below if you want the inline architecture note back later.
    <p class="hint">
      Final single-page flow: the page opens `/ws/stream` for avatar render and `/api/chat` for direct mic uplink
      and transcript tokens. No iframe, no second tab, one button only.
    </p>
    -->
  </div>
  <script src="/vendor/recorder.min.js"></script>
  <script>
    const MSG_VIDEO_INIT = 0x01;
    const MSG_VIDEO_NAL = 0x02;
    const MSG_AUDIO_OPUS = 0x03;
    const MSG_SYNC_META = 0x04;
    const MSG_AUDIO_INIT = 0x05;

    const CHAT_HANDSHAKE = 0x00;
    const CHAT_AUDIO = 0x01;
    const CHAT_TEXT = 0x02;
    const CHAT_CONTROL = 0x03;
    const CHAT_METADATA = 0x04;
    const CHAT_ERROR = 0x05;
    const CHAT_PING = 0x06;
    const WS_AUDIO_START_BUFFER_MS = __WS_AUDIO_START_BUFFER_MS__;

    const connectBtn = document.getElementById('connectBtn');
    const disconnectBtn = document.getElementById('disconnectBtn');
    const modeEl = document.getElementById('mode');
    const dotEl = document.getElementById('dot');
    const stageNoteEl = document.getElementById('stageNote');
    const transcriptBox = document.getElementById('transcriptBox');
    const logBox = document.getElementById('logBox');
    const canvas = document.getElementById('avatarCanvas');
    const ctx = canvas.getContext('2d', { alpha: false });
    const metricEls = {
      videoPackets: document.getElementById('videoPackets'),
      audioPackets: document.getElementById('audioPackets'),
      framesDecoded: document.getElementById('framesDecoded'),
      framesRendered: document.getElementById('framesRendered'),
      videoQueueDepth: document.getElementById('videoQueueDepth'),
      driftMs: document.getElementById('driftMs'),
    };

    let streamWs = null;
    let chatWs = null;
    let transcriptPieces = [];
    let renderHandle = 0;
    let driftTimer = null;
    let reconnecting = false;
    let micPrimed = false;
    let micRecorder = null;
    let chatHandshakeSeen = false;

    let videoDecoder = null;
    let audioDecoder = null;
    let audioDecoderReady = null;
    let audioCtx = null;
    let audioStartAtSec = null;
    let audioWorkletNode = null;
    let audioWorkletReady = null;
    let audioRendererStarted = false;
    let audioPlayedFrames = 0;
    let audioBufferedFrames = 0;
    let audioUnderflows = 0;
    let audioBaseTimestampUs = null;
    let audioSampleRate = 48000;
    let audioSampleRateLogged = false;
    let pendingVideoKey = false;
    let pendingAnnexBInit = null;
    let streamInfo = null;
    let audioInitPayload = null;
    let videoQueue = [];
    let gotFirstKeyframe = false;
    let lastPresentedTsUs = null;

    // DEBUG_VIDEO_INIT: set false or delete this block after extradata diagnosis
    const DEBUG_VIDEO_INIT = true;

    const metrics = {
      videoPackets: 0,
      audioPackets: 0,
      framesDecoded: 0,
      framesRendered: 0,
      videoQueueDepth: 0,
      driftMs: null,
    };

    function setMode(text, live) {
      modeEl.textContent = text;
      dotEl.classList.toggle('live', !!live);
      stageNoteEl.textContent = text;
    }

    function appendLog(line) {
      const stamp = new Date().toLocaleTimeString();
      logBox.textContent = '[' + stamp + '] ' + line + '\\n' + logBox.textContent;
    }

    function updateMetrics() {
      metricEls.videoPackets.textContent = String(metrics.videoPackets);
      metricEls.audioPackets.textContent = String(metrics.audioPackets);
      metricEls.framesDecoded.textContent = String(metrics.framesDecoded);
      metricEls.framesRendered.textContent = String(metrics.framesRendered);
      metricEls.videoQueueDepth.textContent = String(metrics.videoQueueDepth);
      metricEls.driftMs.textContent = metrics.driftMs == null ? 'n/a' : metrics.driftMs.toFixed(1) + ' ms';
    }

    function parsePacket(buf) {
      const dv = new DataView(buf);
      const kind = dv.getUint8(0);
      const length = dv.getUint32(1, false);
      const timestampUs = Number(dv.getBigUint64(5, false));
      const payload = new Uint8Array(buf, 13, length);
      return { kind, length, timestampUs, payload };
    }

    function parseChatMessage(data) {
      const payload = data.slice(1);
      switch (data[0]) {
        case CHAT_HANDSHAKE:
          return { type: 'handshake' };
        case CHAT_AUDIO:
          return { type: 'audio', data: payload };
        case CHAT_TEXT:
          return { type: 'text', data: new TextDecoder().decode(payload) };
        case CHAT_METADATA:
          try {
            return { type: 'metadata', data: JSON.parse(new TextDecoder().decode(payload)) };
          } catch (_err) {
            return { type: 'metadata', data: null };
          }
        case CHAT_ERROR:
          return { type: 'error', data: new TextDecoder().decode(payload) };
        case CHAT_PING:
          return { type: 'ping' };
        default:
          return { type: 'unknown', rawType: data[0], data: payload };
      }
    }

    function encodeChatAudio(payload) {
      const msg = new Uint8Array(payload.length + 1);
      msg[0] = CHAT_AUDIO;
      msg.set(payload, 1);
      return msg;
    }

    function looksLikeAnnexB(payload) {
      return payload.length >= 4 &&
        ((payload[0] === 0x00 && payload[1] === 0x00 && payload[2] === 0x00 && payload[3] === 0x01) ||
         (payload[0] === 0x00 && payload[1] === 0x00 && payload[2] === 0x01));
    }

    function audioClockUs() {
      if (audioRendererStarted && audioBaseTimestampUs != null) {
        return audioBaseTimestampUs + Math.round((audioPlayedFrames / audioSampleRate) * 1_000_000);
      }
      if (!audioCtx || audioStartAtSec == null) return null;
      return Math.max((audioCtx.currentTime - audioStartAtSec) * 1_000_000, 0);
    }

    async function configureVideoDecoderMaybe() {
      if (!streamInfo || videoDecoder) return;
      if (!('VideoDecoder' in window)) {
        appendLog('VideoDecoder not available in this browser.');
        setMode('No WebCodecs', false);
        return;
      }

      const baseConfig = {
        codec: streamInfo.video_codec,
        optimizeForLatency: true,
      };
      let chosenConfig = { ...baseConfig, hardwareAcceleration: 'prefer-hardware' };

      try {
        const supported = await VideoDecoder.isConfigSupported(chosenConfig);
        chosenConfig = supported.config;
      } catch (_err) {
        chosenConfig = baseConfig;
      }

      videoDecoder = new VideoDecoder({
        output(frame) {
          metrics.framesDecoded += 1;
          videoQueue.push(frame);
          if (videoQueue.length > 16) {
            const dropped = videoQueue.shift();
            if (dropped) dropped.close();
          }
          metrics.videoQueueDepth = videoQueue.length;
          updateMetrics();
        },
        error(err) {
          appendLog('VideoDecoder error: ' + err.message);
        },
      });
      videoDecoder.configure(chosenConfig);
      appendLog('Video decoder configured: ' + chosenConfig.codec);
    }

    async function configureAudioDecoderMaybe() {
      if (!streamInfo || !audioInitPayload || audioDecoder) return;
      if (audioDecoderReady) {
        await audioDecoderReady;
        return;
      }
      if (!('AudioDecoder' in window)) {
        appendLog('AudioDecoder not available in this browser.');
        return;
      }
      if (!audioCtx) {
        appendLog('AudioContext missing; connect via button to enable audio.');
        return;
      }
      await ensureAudioWorklet();

      const baseConfig = {
        codec: streamInfo.audio_codec,
        sampleRate: streamInfo.audio_sr,
        numberOfChannels: streamInfo.audio_channels,
        description: audioInitPayload.buffer.slice(
          audioInitPayload.byteOffset,
          audioInitPayload.byteOffset + audioInitPayload.byteLength
        ),
      };
      audioDecoderReady = (async () => {
        let chosenConfig = baseConfig;
        try {
          const supported = await AudioDecoder.isConfigSupported(baseConfig);
          chosenConfig = supported.config;
        } catch (_err) {}

        audioDecoder = new AudioDecoder({
          output(data) {
            scheduleAudioData(data);
          },
          error(err) {
            appendLog('AudioDecoder error: ' + err.message);
          },
        });
        audioDecoder.configure(chosenConfig);
        appendLog('Audio decoder configured: ' + chosenConfig.codec);
      })();

      try {
        await audioDecoderReady;
      } finally {
        audioDecoderReady = null;
      }
    }

    function scheduleAudioData(audioData) {
      if (audioWorkletNode) {
        queueAudioToWorklet(audioData);
        return;
      }
      scheduleAudioDataOneShot(audioData);
    }

    function scheduleAudioDataOneShot(audioData) {
      if (!audioCtx) {
        audioData.close();
        return;
      }
      const when = ensureAudioSchedule(audioData.timestamp);
      if (when < audioCtx.currentTime - 0.02) {
        audioData.close();
        return;
      }

      const buffer = audioCtx.createBuffer(audioData.numberOfChannels, audioData.numberOfFrames, audioData.sampleRate);
      for (let ch = 0; ch < audioData.numberOfChannels; ch += 1) {
        const pcm = new Float32Array(audioData.numberOfFrames);
        audioData.copyTo(pcm, { planeIndex: ch, format: 'f32-planar' });
        buffer.copyToChannel(pcm, ch);
      }

      const source = audioCtx.createBufferSource();
      source.buffer = buffer;
      source.connect(audioCtx.destination);
      source.start(when);
      source.onended = () => source.disconnect();
      metrics.audioPackets += 1;
      updateMetrics();
      audioData.close();
    }

    async function ensureAudioWorklet() {
      if (!audioCtx || !('audioWorklet' in audioCtx) || typeof AudioWorkletNode === 'undefined') {
        return;
      }
      if (audioWorkletNode) return;
      if (!audioWorkletReady) {
        audioWorkletReady = (async () => {
          await audioCtx.audioWorklet.addModule('/vendor/ws_audio_worklet.js');
          const node = new AudioWorkletNode(audioCtx, 'ws-audio-playback', {
            numberOfInputs: 0,
            numberOfOutputs: 1,
            outputChannelCount: [1],
            processorOptions: {
              startBufferFrames: Math.max(
                128,
                Math.round((audioCtx.sampleRate * WS_AUDIO_START_BUFFER_MS) / 1000)
              ),
            },
          });
          node.connect(audioCtx.destination);
          node.port.onmessage = (event) => {
            const msg = event.data || {};
            if (msg.type === 'stats') {
              audioPlayedFrames = msg.playedFrames || 0;
              audioBufferedFrames = msg.bufferedFrames || 0;
              const nextUnderflows = msg.underflows || 0;
              if (nextUnderflows > audioUnderflows) {
                appendLog('Audio underflow x' + nextUnderflows + ' (buffered=' + audioBufferedFrames + 'f)');
              }
              audioUnderflows = nextUnderflows;
              if (msg.started && !audioRendererStarted) {
                audioRendererStarted = true;
                appendLog('Audio worklet playback started.');
              }
            }
          };
          audioWorkletNode = node;
          appendLog('Audio worklet ready (' + WS_AUDIO_START_BUFFER_MS.toFixed(0) + ' ms buffer).');
        })().catch((err) => {
          audioWorkletReady = null;
          appendLog('Audio worklet unavailable, falling back to packet scheduling: ' + err.message);
        });
      }
      await audioWorkletReady;
    }

    function queueAudioToWorklet(audioData) {
      if (!audioWorkletNode) {
        scheduleAudioDataOneShot(audioData);
        return;
      }
      if (audioBaseTimestampUs == null) {
        audioBaseTimestampUs = audioData.timestamp;
        appendLog('Audio worklet primed with stream timestamp ' + audioBaseTimestampUs + 'us.');
      }
      if (!audioSampleRateLogged && audioData.sampleRate !== audioSampleRate) {
        appendLog('Audio decoder sample rate=' + audioData.sampleRate + 'Hz');
        audioSampleRateLogged = true;
      }
      audioSampleRate = audioData.sampleRate || audioSampleRate;

      const mono = new Float32Array(audioData.numberOfFrames);
      if (audioData.numberOfChannels === 1) {
        audioData.copyTo(mono, { planeIndex: 0, format: 'f32-planar' });
      } else {
        const scratch = new Float32Array(audioData.numberOfFrames);
        for (let ch = 0; ch < audioData.numberOfChannels; ch += 1) {
          scratch.fill(0);
          audioData.copyTo(scratch, { planeIndex: ch, format: 'f32-planar' });
          for (let i = 0; i < scratch.length; i += 1) {
            mono[i] += scratch[i];
          }
        }
        const inv = 1 / audioData.numberOfChannels;
        for (let i = 0; i < mono.length; i += 1) {
          mono[i] *= inv;
        }
      }

      audioWorkletNode.port.postMessage({
        type: 'push',
        pcm: mono,
      }, [mono.buffer]);
      metrics.audioPackets += 1;
      updateMetrics();
      audioData.close();
    }

    function ensureAudioSchedule(timestampUs) {
      if (!audioCtx) return 0;
      if (audioStartAtSec == null) {
        audioStartAtSec = audioCtx.currentTime + 0.05 - (timestampUs / 1_000_000);
        appendLog('Audio clock started with 50ms jitter buffer.');
      }
      return audioStartAtSec + (timestampUs / 1_000_000);
    }

    function clearVideoQueue() {
      while (videoQueue.length) {
        const frame = videoQueue.shift();
        if (frame) frame.close();
      }
      metrics.videoQueueDepth = 0;
      updateMetrics();
    }

    function renderLoop() {
      renderHandle = window.requestAnimationFrame(renderLoop);
      if (!videoQueue.length) return;

      const nowUs = audioClockUs();
      let frameToDraw = null;

      if (nowUs == null) {
        while (videoQueue.length > 1) {
          const stale = videoQueue.shift();
          if (stale) stale.close();
        }
        frameToDraw = videoQueue.shift() || null;
      } else {
        while (videoQueue.length) {
          const next = videoQueue[0];
          if (next.timestamp < nowUs - 100_000) {
            const stale = videoQueue.shift();
            if (stale) stale.close();
            continue;
          }
          if (next.timestamp <= nowUs + 16_000) {
            if (frameToDraw) frameToDraw.close();
            frameToDraw = videoQueue.shift();
            continue;
          }
          break;
        }
      }

      if (!frameToDraw) {
        metrics.videoQueueDepth = videoQueue.length;
        updateMetrics();
        return;
      }

      ctx.drawImage(frameToDraw, 0, 0, canvas.width, canvas.height);
      lastPresentedTsUs = frameToDraw.timestamp;
      metrics.framesRendered += 1;
      metrics.videoQueueDepth = videoQueue.length;
      updateMetrics();
      frameToDraw.close();
    }

    function startRenderLoop() {
      if (!renderHandle) {
        renderHandle = window.requestAnimationFrame(renderLoop);
      }
    }

    function stopRenderLoop() {
      if (renderHandle) {
        window.cancelAnimationFrame(renderHandle);
        renderHandle = 0;
      }
    }

    function renderTranscript() {
      transcriptBox.textContent = transcriptPieces.length
        ? transcriptPieces.join('')
        : 'Waiting for transcript...';
    }

    async function primeMicrophoneFromClick() {
      if (micPrimed) return;
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        stream.getTracks().forEach((track) => track.stop());
        micPrimed = true;
      } catch (err) {
        appendLog('Mic prime failed: ' + err.message);
      }
    }

    function startDriftLogging() {
      if (driftTimer) return;
      driftTimer = window.setInterval(() => {
        const nowUs = audioClockUs();
        if (nowUs == null || lastPresentedTsUs == null) {
          metrics.driftMs = null;
        } else {
          metrics.driftMs = (lastPresentedTsUs - nowUs) / 1000.0;
        }
        updateMetrics();
      }, 1000);
    }

    function stopDriftLogging() {
      if (driftTimer) {
        window.clearInterval(driftTimer);
        driftTimer = null;
      }
    }

    async function handleMessage(pkt) {
      if (pkt.kind === MSG_SYNC_META) {
        const meta = JSON.parse(new TextDecoder().decode(pkt.payload));
        if (meta.meta_type === 'stream_init') {
          streamInfo = meta;
          appendLog('stream_init video=' + meta.video_codec + ' ' + meta.width + 'x' + meta.height + ' fps=' + meta.fps);
          await configureVideoDecoderMaybe();
          await configureAudioDecoderMaybe();
        } else if (meta.meta_type === 'chunk_start') {
          appendLog('chunk ' + meta.chunk_index + ' age=' + Number(meta.chunk_age).toFixed(3) + 's q=' + meta.play_queue_len);
        }
        return;
      }

      if (pkt.kind === MSG_AUDIO_INIT) {
        audioInitPayload = pkt.payload;
        appendLog('audio init received (' + pkt.payload.length + ' bytes)');
        await configureAudioDecoderMaybe();
        return;
      }

      if (pkt.kind === MSG_VIDEO_INIT) {
        if (DEBUG_VIDEO_INIT) {
          const p = pkt.payload;
          const n = Math.min(8, p.length);
          let hex = '';
          for (let i = 0; i < n; i++) {
            hex += p[i].toString(16).padStart(2, '0');
            if (i + 1 < n) hex += ' ';
          }
          const asAnnexB = looksLikeAnnexB(p);
          const treatAs = asAnnexB ? 'Annex B (will prepend to next key NAL)' : 'not Annex-B start codes (no prepend; typical AVCC extradata)';
          const dbgLine = 'DEBUG MSG_VIDEO_INIT first8_hex=[' + hex + '] treat_as=' + treatAs;
          appendLog(dbgLine);
          console.log('[viewer]', dbgLine);
        }
        pendingVideoKey = true;
        pendingAnnexBInit = looksLikeAnnexB(pkt.payload) ? pkt.payload : null;
        return;
      }

      if (pkt.kind === MSG_VIDEO_NAL) {
        metrics.videoPackets += 1;
        updateMetrics();
        await configureVideoDecoderMaybe();
        if (!videoDecoder || videoDecoder.state === 'closed') return;

        const isKey = pendingVideoKey || !gotFirstKeyframe;
        if (!isKey && !gotFirstKeyframe) {
          return;
        }

        let data = pkt.payload;
        if (isKey && pendingAnnexBInit && pendingAnnexBInit.length) {
          const merged = new Uint8Array(pendingAnnexBInit.length + data.length);
          merged.set(pendingAnnexBInit, 0);
          merged.set(data, pendingAnnexBInit.length);
          data = merged;
        }

        try {
          const chunk = new EncodedVideoChunk({
            type: isKey ? 'key' : 'delta',
            timestamp: pkt.timestampUs,
            data,
          });
          videoDecoder.decode(chunk);
          if (isKey) gotFirstKeyframe = true;
        } catch (err) {
          appendLog('video decode failed: ' + err.message);
        } finally {
          pendingVideoKey = false;
          pendingAnnexBInit = null;
        }
        return;
      }

      if (pkt.kind === MSG_AUDIO_OPUS) {
        await configureAudioDecoderMaybe();
        if (!audioDecoder || audioDecoder.state === 'closed') return;
        try {
          const chunk = new EncodedAudioChunk({
            type: 'key',
            timestamp: pkt.timestampUs,
            data: pkt.payload,
          });
          audioDecoder.decode(chunk);
        } catch (err) {
          appendLog('audio decode failed: ' + err.message);
        }
      }
    }

    function buildChatUrl() {
      const scheme = window.location.protocol === 'https:' ? 'wss' : 'ws';
      const url = new URL(scheme + '://' + window.location.host + '/api/chat');
      return url.toString();
    }

    function stopMicRecorder() {
      if (micRecorder) {
        try { micRecorder.stop(); } catch (_err) {}
        micRecorder = null;
      }
    }

    function startMicRecorder() {
      if (micRecorder || !chatWs || chatWs.readyState !== WebSocket.OPEN) return;
      if (!window.Recorder) {
        appendLog('Recorder library is missing.');
        return;
      }
      const sampleRate = audioCtx ? audioCtx.sampleRate : 48000;
      const RecorderCtor = window.Recorder;
      const recorderOptions = {
        mediaTrackConstraints: {
          audio: {
            echoCancellation: true,
            noiseSuppression: true,
            autoGainControl: true,
            channelCount: 1,
          },
          video: false,
        },
        encoderPath: '/vendor/encoderWorker.min.js',
        bufferLength: Math.round(960 * sampleRate / 24000),
        encoderFrameSize: 20,
        encoderSampleRate: 24000,
        maxFramesPerPage: 2,
        numberOfChannels: 1,
        recordingGain: 1,
        resampleQuality: 3,
        encoderComplexity: 0,
        encoderApplication: 2049,
        streamPages: true,
      };
      let chunkCount = 0;
      micRecorder = new RecorderCtor(recorderOptions);
      micRecorder.ondataavailable = (data) => {
        if (!chatWs || chatWs.readyState !== WebSocket.OPEN) return;
        const payload = data instanceof Uint8Array ? data : new Uint8Array(data);
        if (chunkCount < 5) {
          appendLog('mic opus chunk ' + chunkCount + ' (' + payload.length + ' bytes)');
        }
        chunkCount += 1;
        chatWs.send(encodeChatAudio(payload));
      };
      micRecorder.onstart = () => appendLog('Mic recorder started.');
      micRecorder.onstop = () => appendLog('Mic recorder stopped.');
      micRecorder.start();
    }

    async function connectChat() {
      if (chatWs && (chatWs.readyState === WebSocket.OPEN || chatWs.readyState === WebSocket.CONNECTING)) return;
      transcriptPieces = [];
      renderTranscript();
      chatHandshakeSeen = false;
      const url = buildChatUrl();
      chatWs = new WebSocket(url);
      chatWs.binaryType = 'arraybuffer';

      chatWs.onopen = () => {
        appendLog('Chat websocket opened.');
      };

      chatWs.onmessage = (event) => {
        const bytes = new Uint8Array(event.data);
        const msg = parseChatMessage(bytes);
        if (msg.type === 'handshake') {
          chatHandshakeSeen = true;
          appendLog('Chat handshake received.');
          startMicRecorder();
          return;
        }
        if (msg.type === 'text') {
          transcriptPieces.push(msg.data);
          renderTranscript();
          return;
        }
        if (msg.type === 'metadata') {
          appendLog('Chat metadata received.');
          return;
        }
        if (msg.type === 'error') {
          appendLog('Chat error: ' + msg.data);
          return;
        }
      };

      chatWs.onerror = () => {
        appendLog('Chat websocket error.');
      };

      chatWs.onclose = () => {
        stopMicRecorder();
        chatWs = null;
        if (!reconnecting) {
          appendLog('Chat session ended. Click Connect for another turn.');
          connectBtn.disabled = false;
          if (streamWs && streamWs.readyState === WebSocket.OPEN) {
            setMode('Ready For Next Turn', true);
          } else {
            setMode('Idle', false);
          }
        }
      };
    }

    async function connect() {
      const streamAlreadyOpen = streamWs && (streamWs.readyState === WebSocket.OPEN || streamWs.readyState === WebSocket.CONNECTING);
      const chatAlreadyOpen = chatWs && (chatWs.readyState === WebSocket.OPEN || chatWs.readyState === WebSocket.CONNECTING);
      if (streamAlreadyOpen && chatAlreadyOpen) return;
      if (!('VideoDecoder' in window) || !('AudioDecoder' in window)) {
        appendLog('WebCodecs not available. Use Brave/Chromium for this viewer.');
        setMode('Unsupported Browser', false);
        return;
      }

      if (!audioCtx) {
        audioCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 48000 });
      }
      await audioCtx.resume();

      setMode('Connecting', false);
      connectBtn.disabled = true;

      if (!streamAlreadyOpen) {
        const scheme = window.location.protocol === 'https:' ? 'wss' : 'ws';
        streamWs = new WebSocket(scheme + '://' + window.location.host + '/ws/stream');
        streamWs.binaryType = 'arraybuffer';

        streamWs.onopen = () => {
          setMode('Streaming', true);
          appendLog('Avatar stream connected.');
          startRenderLoop();
          startDriftLogging();
        };

        streamWs.onclose = () => {
          appendLog('Avatar stream closed.');
          streamWs = null;
          if (!reconnecting) {
            connectBtn.disabled = false;
            setMode('Closed', false);
          }
        };

        streamWs.onerror = () => {
          setMode('Error', false);
          appendLog('Avatar stream error.');
        };

        streamWs.onmessage = async (event) => {
          if (!(event.data instanceof ArrayBuffer)) {
            appendLog('non-binary packet: ' + typeof event.data);
            return;
          }
          const pkt = parsePacket(event.data);
          await handleMessage(pkt);
        };
      }

      if (!chatAlreadyOpen) {
        await connectChat();
      }
    }

    async function disconnect() {
      reconnecting = true;
      stopRenderLoop();
      stopDriftLogging();
      stopMicRecorder();
      if (chatWs) {
        chatWs.close();
        chatWs = null;
      }
      if (streamWs) {
        streamWs.close();
        streamWs = null;
      }
      if (videoDecoder && videoDecoder.state !== 'closed') videoDecoder.close();
      if (audioDecoder && audioDecoder.state !== 'closed') audioDecoder.close();
      audioDecoderReady = null;
      if (audioWorkletNode) {
        try { audioWorkletNode.disconnect(); } catch (_err) {}
      }
      if (audioCtx) {
        try { await audioCtx.close(); } catch (_err) {}
      }
      videoDecoder = null;
      audioDecoder = null;
      audioWorkletNode = null;
      audioWorkletReady = null;
      audioRendererStarted = false;
      audioPlayedFrames = 0;
      audioBufferedFrames = 0;
      audioUnderflows = 0;
      audioBaseTimestampUs = null;
      audioSampleRate = 48000;
      audioSampleRateLogged = false;
      audioCtx = null;
      audioStartAtSec = null;
      streamInfo = null;
      audioInitPayload = null;
      pendingVideoKey = false;
      pendingAnnexBInit = null;
      gotFirstKeyframe = false;
      lastPresentedTsUs = null;
      metrics.videoPackets = 0;
      metrics.audioPackets = 0;
      metrics.framesDecoded = 0;
      metrics.framesRendered = 0;
      metrics.videoQueueDepth = 0;
      metrics.driftMs = null;
      transcriptPieces = [];
      clearVideoQueue();
      renderTranscript();
      updateMetrics();
      ctx.fillStyle = '#000';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      setMode('Idle', false);
      connectBtn.disabled = false;
      reconnecting = false;
    }

    connectBtn.addEventListener('click', () => {
      primeMicrophoneFromClick().then(() => {
        return connect();
      }).catch((err) => {
        appendLog('connect failed: ' + err.message);
        setMode('Connect Failed', false);
        connectBtn.disabled = false;
      });
    });

    disconnectBtn.addEventListener('click', () => {
      disconnect().catch((err) => appendLog('disconnect failed: ' + err.message));
    });

    renderTranscript();
    updateMetrics();
  </script>
</body>
</html>
"""


async def serve_ws_stream(
    request: web.Request,
    session,
    ws_lock: asyncio.Lock,
) -> web.WebSocketResponse:
    ws = web.WebSocketResponse(max_msg_size=8 * 1024 * 1024)
    await ws.prepare(request)

    loop = asyncio.get_running_loop()
    fps = int(round(float(getattr(session.opt, "fps", 25))))

    async with ws_lock:
        if session.av_session is not None:
            print("[launch/ws] tearing down previous viewer session")
            try:
                session.av_session.stop()
            except Exception:
                pass
            session.av_session = None

        session.reset_reply()
        stream_session = WSStreamSession(
            loop=loop,
            idle_frame_uint8=session.idle_frame_uint8,
            fps=fps,
            debug=bool(getattr(session.opt, "debug_session", False)),
        )
        stream_session.start()
        session.av_session = stream_session

    try:
        while True:
            payload = await stream_session.out_q.get()
            if payload is None:
                break
            await ws.send_bytes(payload)
    except (asyncio.CancelledError, ConnectionResetError, ClientConnectionResetError):
        print("[launch/ws] client disconnected")
    finally:
        async with ws_lock:
            if session.av_session is stream_session:
                stream_session.stop()
                session.av_session = None
        with contextlib.suppress(Exception):
            await ws.close()
    return ws


def _render_ws_viewer_html(opt) -> str:
    return _WS_VIEWER_HTML.replace(
        "__WS_AUDIO_START_BUFFER_MS__",
        f"{float(getattr(opt, 'ws_audio_buffer_ms', 80.0)):.1f}",
    )


def _stream_state_response_factory(session, transcript, state):
    def _handler(_request):
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

    return _handler


def main():
    opt = LaunchOptions().parse()
    opt.rank = opt.device

    transcript = TranscriptStore()
    session = build_imtalker_session(opt)
    state = build_moshi_state(opt, session, transcript)
    static_path = _resolve_static_path(opt)
    vendor_path = str(Path(__file__).resolve().parent / "web_vendor")

    ws_lock = asyncio.Lock()
    app = web.Application()
    app.router.add_get("/api/chat", state.handle_chat)
    app.router.add_get("/api/stream_state", _stream_state_response_factory(session, transcript, state))
    viewer_html = _render_ws_viewer_html(opt)
    app.router.add_get("/", lambda _r, html=viewer_html: web.Response(text=html, content_type="text/html"))
    app.router.add_get("/viewer", lambda _r, html=viewer_html: web.Response(text=html, content_type="text/html"))

    async def handle_ws_stream(request):
        return await serve_ws_stream(request, session, ws_lock)

    async def serve_moshi_index(request):
        return await _serve_moshi_index(request, static_path)

    app.router.add_get("/ws/stream", handle_ws_stream)
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
    if os.path.isdir(vendor_path):
        app.router.add_static(
            "/vendor",
            path=vendor_path,
            follow_symlinks=True,
            name="ws_vendor_assets",
        )
    favicon_path = os.path.join(static_path, "favicon.ico")
    if os.path.exists(favicon_path):
        app.router.add_get(
            "/favicon.ico",
            lambda _r, p=favicon_path: web.FileResponse(path=p),
        )

    async def on_shutdown(_app):
        if session.av_session is not None:
            with contextlib.suppress(Exception):
                session.av_session.stop()
            session.av_session = None

    app.on_shutdown.append(on_shutdown)

    print(f"[launch] WS viewer -> http://localhost:{opt.port}/")
    print(f"[launch] Hidden Moshi UI -> http://localhost:{opt.port}/moshi")
    print(f"[launch] WS stream route -> ws://localhost:{opt.port}/ws/stream\n")
    web.run_app(app, host=opt.host, port=opt.port)


if __name__ == "__main__":
    main()
