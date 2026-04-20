# Moshi + IMTalker on RunPod

This tree now includes a self-contained combined launcher for the sibling layout:

```text
/workspace/IMTalker
/workspace/moshi
```

It keeps the current compatibility path for the avatar:

```text
Moshi reply PCM -> Mimi encode_to_latent(quantize=False) -> IMTalker FM.sample(a_feat=...) -> renderer
```

No direct `main_latents` wiring was added in this task.

## Files to copy

Copy these files from this local tree into the pod's `IMTalker` tree:

```text
IMTalker/launch_live.py
IMTalker/live_pipeline.py
IMTalker/run_live_avatar.sh
IMTalker/LIVE_MOSHI_RUNPOD.md
IMTalker/generator/FM.py
IMTalker/generator/options/base_options.py
```

## Milestone A: Moshi-only live

From the pod:

```bash
cd /workspace/moshi/moshi
python3 -m moshi.server --host 0.0.0.0 --port 8998
```

Open:

```text
http://127.0.0.1:8998
```

If you are off-pod, tunnel or proxy that port. This is the stock Moshi server/UI path and does not involve IMTalker.

## Download the private generator

Your generator repo is private, so use a token with access:

```bash
export HF_TOKEN=...
huggingface-cli login --token "$HF_TOKEN"
mkdir -p /workspace/IMTalker/ckpts_mimi
huggingface-cli download \
  niloy629/imtalker_mimi_step100000 \
  step=300000.ckpt \
  --local-dir /workspace/IMTalker/ckpts_mimi
```

Expected checkpoint path:

```text
/workspace/IMTalker/ckpts_mimi/step=300000.ckpt
```

## Milestone B: Moshi + IMTalker live avatar

Run from the pod:

```bash
cd /workspace/IMTalker
chmod +x run_live_avatar.sh
./run_live_avatar.sh \
  --host 0.0.0.0 \
  --port 8998 \
  --ref_path /workspace/IMTalker/assets/source_5.png \
  --generator_path /workspace/IMTalker/ckpts_mimi/step=300000.ckpt \
  --renderer_path /workspace/IMTalker/checkpoints/renderer.ckpt \
  --hf_repo kyutai/moshiko-pytorch-bf16 \
  --crop
```

Direct Python equivalent:

```bash
cd /workspace/IMTalker
PYTHONPATH="/workspace/IMTalker:/workspace/moshi/moshi:${PYTHONPATH}" \
python3 launch_live.py \
  --moshi_repo /workspace/moshi \
  --host 0.0.0.0 \
  --port 8998 \
  --ref_path /workspace/IMTalker/assets/source_5.png \
  --generator_path /workspace/IMTalker/ckpts_mimi/step=300000.ckpt \
  --renderer_path /workspace/IMTalker/checkpoints/renderer.ckpt \
  --hf_repo kyutai/moshiko-pytorch-bf16 \
  --crop
```

Open:

```text
http://127.0.0.1:8998/
```

Hidden Moshi UI:

```text
http://127.0.0.1:8998/moshi
```

## Notes

- `launch_live.py` vendors the needed Moshi websocket/output-hook behavior locally, so you do not need to patch the sibling `/workspace/moshi` repo just to try the avatar flow.
- The combined launcher defaults IMTalker to `audio_feat_dim=512` for Mimi-style conditioning.
- The existing raw-audio Wav2Vec IMTalker path is still intact for the old checkpoints. The new Mimi-conditioned path is only used by `launch_live.py`.
- If `/workspace/moshi/client/dist` does not exist, the launcher falls back to the prebuilt Moshi client bundle from `kyutai/moshi-artifacts`.

## Quick verification

1. `python3 -m py_compile launch_live.py live_pipeline.py generator/FM.py generator/options/base_options.py`
2. `cd /workspace/moshi/moshi && python3 -m moshi.server --host 0.0.0.0 --port 8998`
3. `cd /workspace/IMTalker && ./run_live_avatar.sh ...`
4. Open the combined page and verify:
   - the hidden Moshi controller connects
   - the browser asks for mic permission
   - reply segments appear under the avatar page
   - audio is heard only through avatar playback
