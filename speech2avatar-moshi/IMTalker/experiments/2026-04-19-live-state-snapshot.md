# Live streaming state snapshot (2026-04-19)

This is the current state of the live WebSocket pipeline as observed from the latest debug run on the pod. Read before starting Task 001.

## What's working

- **Offline path is stable.** `generator.eval_bridge_frozen` with the Phase 2 bridge checkpoint (`/workspace/exps/phase2_bridge_finetune/checkpoints/step=050000.ckpt`) produces good renders. So the model and checkpoint are fine. The remaining bugs are in online orchestration.
- Checkpoint loads cleanly: bridge adapter params loaded, `audio_projection` params loaded, EMA merged.
- Live uses Mimi-native path: `audio_adapter_mode=bridge_to_768`, `audio_feat_dim=512`, `adapter_hidden_dim=1024`.
- `use_stream_state=True` default.
- `prev_x` first-chunk anchor to `ref_x` — landed.
- `prev_x0` carry across chunks — landed on pod, logs confirm `carry_frames=25` from chunk 1. **Not yet synced back to local `FM.py`** (local is behind pod for this change; writer must verify and unify).
- Mimi PCM residual buffering — landed in `live_pipeline._encode_reply_pcm_to_latents`.
- CUDA graphs / `torch.compile` disabled (crashed during capture). `--skip_warmup` being used.

## Hard constraint

- **`chunk_sec = 1.0` is NON-NEGOTIABLE.** Target is real-time live streaming. All fixes must accommodate 1s chunks. Do not propose increasing `chunk_sec`.

## Main symptoms (live, not offline)

### S1. Moshi generation sub-real-time

```
generated=22.00s wall=32.75s rate=0.67x
```

For 22 s of reply audio Moshi took ~33 s of wall clock. Anything sub-1.0× cascades into stalls regardless of what IMTalker does downstream.

### S2. Playback scheduler misbehaves

Browser flips:
```
mode -> reply
mode -> idle
mode -> reply
...
```
with `play_queue_len=0`, while `out_q_size` grows to ~75 rendered packets. Producer/consumer mismatch between the render output and the play queue.

### S3. Boundary spikes during speech

```
chunk 10: boundary_l2=2.560
chunk 11: 3.495
chunk 12: 6.972        (rms=0.03773, first_half_delta=0.5542, second_half_delta=0.5900)
chunk 13: 2.810
chunk 17: 2.281
```

Silent / early chunks stay moderate (0.2–0.8). Spikes correlate with **speech activity** and per-chunk conditioning deltas, and occur with both `remainder=0` and `remainder=960`. So this is NOT the old PCM zero-pad bug (that fix is already in).

### S4. IMTalker render is fast enough

Each 1s video chunk renders in ~0.39–0.50 s. So the renderer itself is not the bottleneck.

## Key smoking gun (already diagnosed, not yet fixed)

### The overlap / context path is silently disabled every chunk

Logs show:
```
used_overlap_context=False
overlap_frames_used=0
```
for **every** chunk, even though config prints:
```
render_window_frames=50
emit_step_frames=25
overlap_frames=25
overlap_latent_count=12
overlap_context_latent_count=24
```

Root cause is in `live_pipeline.py:655`:
```python
self._overlap_latents = latents[-self.overlap_context_latent_count:].clone()
```
`self.overlap_context_latent_count = 24`, but `latents.shape[0]` for a 1-second chunk is only 12–13 raw Mimi latents (Mimi is 12.5 Hz). So `latents[-24:]` returns at most 12–13 frames. The gate on line 617:
```python
and self._overlap_latents.shape[0] >= self.overlap_context_latent_count
```
is 13 >= 24 → False → overlap path skipped forever.

Net effect: the intended seam-smoothing via prior-chunk Mimi context never activates. The FM generator sees disconnected 1-second Mimi audio windows back-to-back, with only the 10-frame `prev_x/prev_a` context to bridge them. That is plausibly the dominant cause of S3 (boundary spikes during speech-activity transitions).

## Minor alignment oddity

Raw Mimi latent length alternates between 12 and 13 for 1-s chunks (Mimi 12.5 Hz + residual buffering). Each chunk then gets aligned to 25 video frames via linear interpolation. Not obviously wrong, but means each chunk's audio-frame layout differs slightly.

## What the writer should fix, in order of suspected impact

1. **(S3) Overlap accumulation in `live_pipeline._render_reply_chunk`.** Replace the overwriting `self._overlap_latents = latents[-N:]` with an accumulator: append new latents and keep the last `overlap_context_latent_count` frames across all chunks in the reply. First-chunk behavior unchanged (None → skip gate).
2. **(S2) Investigate `out_q_size` vs `play_queue_len` mismatch.** Trace from renderer output → packet encode → WebSocket send → client play queue. Find the step that silently drops or holds packets when `mode` flips to `idle`. Likely in `live_pipeline_ws.py` or `launch_live_ws.py`.
3. **(S1) Profile where Moshi time goes.** Is it GPU contention with IMTalker on a single GPU? If so, either put IMTalker on a separate CUDA stream (it already uses `render_stream` — verify it's actually overlapping), or consider pinning Moshi / Mimi / IMTalker to different SMs via CUDA `StreamPriority`. If the pod has only one GPU and is saturated, the only real fix is a bigger GPU or moving one of the stacks off.

Minor arch / training notes:

- Training is **allowed but discouraged**. Fixes should be code-only where possible. If a minor tweak (e.g., a tiny projection) needs a quick fine-tune to not regress, that's acceptable, but flag it in the report.
