# Task 001 interim: pod sync and baseline prep

Date: 2026-04-20
Writer state: stopped because this chat reached the workflow tool-call budget.

## What was read

- Requested repo-local `.cursor/skills/imtalker-moshi/SKILL.md` was missing in `/home/user/D/working_yay/both/IMTalker`.
- Installed project skill was read instead: `/home/user/.codex/skills/imtalker-moshi/SKILL.md`.
- Read `WORKFLOW.md`.
- Read `experiments/2026-04-19-live-state-snapshot.md`.
- Read `tasks/001-lipsync-live-diagnosis-and-fix.md`.
- Read `runbooks/live-launch.md`.

## Pod -> local sync

The pod-side divergence was resolved first as requested.

Used pod:

```bash
export POD_HOST=root@74.15.1.150
export POD_PORT=30998
export POD_KEY=$HOME/.ssh/id_ed25519
export POD_IMT=/workspace/IMTalker
```

Pulled:

```bash
rsync -avz --delete \
  --exclude '__pycache__' --exclude '*.pyc' \
  -e "ssh -p $POD_PORT -i $POD_KEY" \
  "$POD_HOST:$POD_IMT/generator/" \
  /home/user/D/working_yay/both/IMTalker/generator/

rsync -avz \
  --include='live_pipeline.py' --include='launch_live*.py' --exclude='*' \
  -e "ssh -p $POD_PORT -i $POD_KEY" \
  "$POD_HOST:$POD_IMT/" \
  /home/user/D/working_yay/both/IMTalker/
```

Committed locally:

```text
26ad915 sync: pull pod-side fixes (prev_x0 carry, ...)
```

Staged only the requested sync surface:

- `generator/**/*.py`
- `live_pipeline.py`
- `launch_live*.py`

Not staged: existing untracked docs, tasks, runbooks, logs, pycache, `live_pipeline_ws.py`, etc.

## Confirmed from synced code

- `generator/FM.py` now contains stream-state `prev_x0` carry.
- `generator/FM.py` emits `[DBG/x0] ... carry_frames=...` when `debug_session` is enabled.
- `LaunchOptions` default `chunk_sec` is `1.0`; do not change it.
- `launch_live.py` sets `NO_CUDA_GRAPH=1` and `NO_TORCH_COMPILE=1` by default.
- `--skip_warmup` exists but was not used yet.

## Baseline reproduction prep

No baseline run was launched yet.

Pod checks:

- `/workspace/IMTalker/assets/audio_3.wav` exists.
- `/workspace/IMTalker/tools/ws_capture.py` exists.
- `/workspace/bin/boundary_summary.py` exists.
- Plain `python3` lacks `aiohttp`, `sphn`, and `soundfile`.
- The venv `/workspace/preprocess_5090/bin/activate` has `aiohttp`, `sphn`, `soundfile`, `torchaudio`, and `websockets`.
- Port 8998 did not appear occupied at the time of checking.

## Non-browser driving approach

The live browser normally opens two sockets:

- `/ws/stream` for avatar/audio packet playback.
- `/api/chat` for mic uplink.

For a CLI reproduction, run both:

1. Drain `/ws/stream` using `tools/ws_capture.py`.
2. Send `audio_3.wav` to `/api/chat` as Opus pages, using `sphn.OpusStreamWriter`.

Chat packet shape from `launch_live_ws.py`:

- Server sends one-byte handshake `0x00`.
- Client sends binary messages where byte 0 is `CHAT_AUDIO` (`0x01`) and the rest is an Opus payload page.

The driver should:

- Load `/workspace/IMTalker/assets/audio_3.wav`.
- Resample to Moshi/Mimi sample rate if needed, expected 24 kHz.
- Feed chunks through `sphn.OpusStreamWriter(24000).append_pcm(...)`.
- Send each returned Opus page as `b"\x01" + opus_bytes`.
- Sleep according to audio duration or send fast enough to mimic the browser recorder.
- Keep `/ws/stream` connected during the whole reply, otherwise queue/mode behavior is not representative.

## Next exact step

Launch baseline server on pod from `/workspace/IMTalker`:

```bash
source /workspace/preprocess_5090/bin/activate
export PYTHONPATH=/workspace/IMTalker
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p dump_reply
rm -f live_ws_debug.log

python -u launch_live_ws.py \
  --generator_path /workspace/exps/phase2_bridge_finetune/checkpoints/step=050000.ckpt \
  --renderer_path /workspace/IMTalker/checkpoints/renderer.ckpt \
  --ref_path /workspace/sources/source_5.png \
  --audio_adapter_mode bridge_to_768 \
  --audio_feat_dim 512 \
  --adapter_hidden_dim 1024 \
  --bridge_ckpt /workspace/exps/mimi_bridge_768_control/bridge_pretrained.pt \
  --fix_noise_seed --seed 25 \
  --debug_session \
  --dump_reply_dir /workspace/IMTalker/dump_reply \
  --chunk_sec 1.0 \
  2>&1 | tee live_ws_debug.log
```

Then connect the two CLI clients above for about 30 seconds, stop the server, pull logs/dumps, and summarize with:

```bash
/workspace/bin/boundary_summary.py /workspace/IMTalker/live_ws_debug.log
```

Do not paste raw logs into chat.

## Remaining task order

1. Baseline debug run and `boundary_summary.py` output.
2. Patch P1 overlap accumulation in `live_pipeline.py`.
3. Rsync local -> pod and rerun.
4. Confirm `used_overlap_context=True` from chunk 2 onward and boundary stats improve.
5. Add gated `[DBG/queue]`, `[DBG/mode]`, and `[DBG/timing]` instrumentation.
6. Diagnose/fix queue/mode scheduler.
7. Diagnose Moshi timing and render overlap; do not reduce `nfe` below 5.
8. Final live run, mp4 capture, final experiment report, local commit.
