# Moshi + IMTalker (discrete fork) on RunPod

This is the live integration for the **discrete** Moshi+IMTalker fork. It
mirrors the continuous live setup but the audio path is:

```text
Moshi reply PCM
  -> Mimi streaming encode (kyutai/moshiko-pytorch-bf16)
  -> codebook 0 token IDs
  -> MoshiTokenEncoder.embed + linear_interpolation to video frame rate
  -> FMGenerator.sample(a_feat=...)
  -> IMTRenderer
  -> fMP4 stream to browser
```

Pod layout this targets:

```text
/workspace/IMTalker          ← discrete-fork IMTalker tree
/workspace/moshi             ← Kyutai Moshi/Mimi (sibling)
```

## Files to copy onto the pod

From this repo's `IMTalker/` into the pod's `/workspace/IMTalker/`:

```text
IMTalker/launch_live.py
IMTalker/live_pipeline.py
IMTalker/run_live_avatar.sh
IMTalker/LIVE_MOSHI_RUNPOD.md
IMTalker/generator/FM.py
IMTalker/tools/smoke_mimi_discrete.py
```

(`generator/options/base_options.py` already has `audio_feat_dim=512` and
`audio_token_rate=12.5` in this fork — no change needed there.)

## Checkpoints

Discrete generator (default for the runbook):

```text
/workspace/IMTalker/ckpts_mimi/discrete_step=300000.ckpt
```

Frozen renderer (unchanged across forks):

```text
/workspace/IMTalker/checkpoints/renderer.ckpt
```

## Quick verification

```bash
cd /workspace/IMTalker
python3 -m py_compile launch_live.py live_pipeline.py generator/FM.py
python3 tools/smoke_mimi_discrete.py --moshi_repo /workspace/moshi
```

The smoke check should print `OK` and report 25 tokens for 2 s of audio
at 12.5 Hz, with `cb0.shape=(1, 25)` and `dtype=torch.int64`.

## Milestone A: Moshi-only live (sanity)

```bash
cd /workspace/moshi/moshi
python3 -m moshi.server --host 0.0.0.0 --port 8998
```

Open `http://127.0.0.1:8998`. This is the stock Moshi UI; no IMTalker.

## Milestone B: Moshi + discrete IMTalker live avatar

```bash
cd /workspace/IMTalker
chmod +x run_live_avatar.sh
./run_live_avatar.sh \
  --host 0.0.0.0 \
  --port 8998 \
  --ref_path /workspace/IMTalker/assets/source_5.png \
  --generator_path /workspace/IMTalker/ckpts_mimi/discrete_step=300000.ckpt \
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
  --generator_path /workspace/IMTalker/ckpts_mimi/discrete_step=300000.ckpt \
  --renderer_path /workspace/IMTalker/checkpoints/renderer.ckpt \
  --hf_repo kyutai/moshiko-pytorch-bf16 \
  --crop
```

Open:

```text
http://127.0.0.1:8998/
```

Hidden Moshi controller UI:

```text
http://127.0.0.1:8998/moshi
```

## Notes

- This fork uses **codebook 0 only** for live audio conditioning (matches
  how the discrete generator was trained).
- `MoshiTokenEncoder` (an `nn.Embedding(2048, 512)` lookup + linear interp)
  is the embedding layer the discrete generator was trained with. Live
  inference calls it directly so the embedding weights from the checkpoint
  are reused exactly.
- Token IDs cannot be linearly interpolated, so the embed -> interpolate
  order in `live_pipeline._render_reply_chunk` is load-bearing. Do not
  reorder.
- The `app.py` and `generator/generate.py` utilities in this repo are
  obsolete (still wav2vec-based). Do not run them; use this live path.
