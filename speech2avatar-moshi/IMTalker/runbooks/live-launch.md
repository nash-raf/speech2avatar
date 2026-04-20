# Live WebSocket launch (debug mode)

Use this exact recipe when diagnosing or demoing the live stream.

## Prereqs on pod

- Current generator checkpoint: `/workspace/exps/phase2_bridge_finetune/checkpoints/step=050000.ckpt`
- Renderer: `/workspace/IMTalker/checkpoints/renderer.ckpt`
- Bridge: `/workspace/exps/mimi_bridge_768_control/bridge_pretrained.pt`
- Reference image: `/workspace/sources/source_5.png`

## Debug launch

```bash
ssh -p $POD_PORT $POD_HOST
cd /workspace/IMTalker
mkdir -p dump_reply
rm -f live_ws_debug.log

python launch_live_ws.py \
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
  2>&1 | tee live_ws_debug.log
```

Drive a known reply (e.g. play `audio_3.wav` through the client) for ~30 s, then Ctrl-C.

## After the run

```bash
# locally
rsync -avz -e "ssh -p $POD_PORT" \
  $POD_HOST:/workspace/IMTalker/live_ws_debug.log \
  $POD_HOST:/workspace/IMTalker/dump_reply/ \
  /home/user/D/working_yay/both/IMTalker/logs/

# summarise (do this instead of reading the raw log)
ssh -p $POD_PORT $POD_HOST '/workspace/bin/boundary_summary.py /workspace/IMTalker/live_ws_debug.log'
```

## What to look at

- Lines tagged `[TIMING] chunk NNN ... boundary_l2=X.XXX` — seam quality.
- Lines tagged `[DBG/mimi]` — confirms PCM residual buffering (`full_samples>0`, `remainder<mimi_frame_size`).
- Lines tagged `[DBG/x0]` — only present after Bug A fix lands.
- Lines tagged `[TIMING]` with large `chunk_age` — indicates the render queue falling behind; usually means nfe too high or renderer not compiled.
