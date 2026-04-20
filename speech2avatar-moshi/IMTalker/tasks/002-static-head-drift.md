# Task 002: Static-head mode that actually stays static during speech

## Context

- Read `@.cursor/skills/imtalker-moshi/SKILL.md` first. Then this file.
- Static-head flags already exist: `--static_pose_first`, `--static_gaze_first`, `--static_cam_first` (also the `_zero` variants). They pin the per-frame pose/gaze/cam conditioning vectors to either zero or the first frame. `--static_cam_zero` causes face distortion (OOD); `--static_cam_first` is the sane choice.
- **Observed problem:** even with pose/gaze/cam pinned, the generated 32-d motion latent still drifts during speech. The head/upper-face moves away from the reference and drifts back to neutral at the end of the utterance. First chunk is additionally biased (was partially addressed by the `prev_x_t = repeat_first_step(ref_x)` anchor in `FM.sample()`).
- Root cause: the 32-d motion latent encodes full-face motion, not just lip motion. Audio conditioning drives non-mouth directions too, because it was trained on HDTF where speakers naturally move.

## Goal

A demo-grade static head: during speech, **only the mouth/lips move**. Eyes, head pose, upper face remain on the reference motion. No drift back-and-forth across utterances.

Acceptance:
- Offline render on `source_5 + audio_3` with `--static_pose_first --static_gaze_first --static_cam_first`:
  - `upper_face_motion_mean` (from `eval_bridge_frozen.py` metrics) drops to `<= 60%` of the non-static baseline.
  - `mouth_open_max / mouth_open_mean` ratio stays `>= 2.2` (i.e., we didn't kill lip motion).
  - Eyeball check: head visibly still, mouth clearly moves.
- Live render preserves the same behaviour (no regression in lip sync from Task 001).

## Approach

Two layered fixes. Try **A first**, it's the cheap 3-hour version. If A is visibly not enough, add **B**.

### A. Residual damping around reference motion (fast)

Idea: instead of letting the generator emit absolute motion latent `m_gen`, treat output as `m_ref + delta` and damp `delta` on a per-dim scale.

- In `FM.sample()` (or cleaner: in `live_pipeline._decode_sample_to_frames` / its equivalent wrapper), compute
  ```python
  delta = sample - ref_x          # ref_x broadcast over time
  sample_out = ref_x + mask * delta
  ```
  where `mask` is a 32-d vector in `[0, 1]` per dim, and `ref_x` is the renderer's reference motion latent for the current session (already available in `live_pipeline`).
- Two ways to get `mask`:
  1. **Manual binary mask (fastest).** Perturb each of the 32 dims on a held-out clip, render, and record which dims move the mouth vs which move upper face / pose. Mask = 1 for mouth dims, 0 elsewhere. Save as `/workspace/exps/mouth_mask.pt`.
  2. **Variance-correlation mask (cheap and principled).** On `~500` training motion latents:
     - Compute `motion` minus per-clip mean = residuals.
     - Compute a 1-d "mouth-open signal" per frame from SMIRK/landmarks (distance between upper/lower lip landmark, or existing `mouth_open` metric code).
     - For each of the 32 dims, compute `|corr(residual[:, d], mouth_open_signal)|`.
     - Normalise to `[0, 1]` (divide by max). Optionally threshold at 0.3 for a sharper mask. Save as `/workspace/exps/mouth_mask.pt`.
  - Gate via CLI flag `--mouth_mask_path` (default unset = identity mask = old behaviour).
- Wire it in **both** paths: offline `eval_bridge_frozen.py` and `live_pipeline.py`'s per-chunk sample handling. One helper function, two call sites.

### B. First-frame anchor on residual (only if A alone drifts)

If after A the non-mouth directions still wander slightly, anchor the *residual's running mean* to zero over a small window:

```python
# ema_delta: running mean of (1 - mask) * delta over last ~50 frames, alpha=0.1
sample_out = ref_x + mask * delta + (1 - mask) * (delta - ema_delta)
```

This keeps short-term expressive motion on unmasked dims but kills long-term drift. Make it a flag `--anchor_unmasked_delta` off by default.

## Files to touch

- `generator/FM.py` — optionally apply mask inside `sample()`. Cleaner if possible; otherwise do it in the caller.
- `live_pipeline.py` — post-sample hook. `ref_x` is already stored on the session as `self.ref_x`.
- `generator/eval_bridge_frozen.py` — add CLI, apply mask, same hook.
- `generator/options/base_options.py` — add `--mouth_mask_path` (str, default None), `--anchor_unmasked_delta` (bool, default False).
- New script `tools/build_mouth_mask.py` — offline one-shot: reads N training stems, builds variance-correlation mask, saves `.pt`.

## Steps (suggested order)

1. Write `tools/build_mouth_mask.py`. Target ≤150 lines. Save `mouth_mask.pt` to pod exps dir.
2. Add the CLI flag + apply-mask hook in `FM.sample()` or caller.
3. Wire into `eval_bridge_frozen.py`. Run offline with/without mask on `source_5 + audio_3`. Compare metrics. Eyeball video.
4. If metrics meet the bar, wire into `live_pipeline.py` with the same flag. Test live. Confirm no lip-sync regression.
5. If step 3 falls short, add the `--anchor_unmasked_delta` option and retest.

## Acceptance check commands

```bash
# non-static baseline (for comparison)
python -m generator.eval_bridge_frozen \
  --resume_ckpt /workspace/exps/phase2_bridge_finetune/checkpoints/step=050000.ckpt \
  --renderer_path /workspace/IMTalker/checkpoints/renderer.ckpt \
  --ref_path /workspace/sources/source_5.png \
  --aud_feat_path /workspace/hdtf_preprocess/audio_rt_aligned/audio_3.npy \
  --audio_subdir audio_rt_aligned --audio_feat_dim 512 \
  --audio_adapter_mode bridge_to_768 --adapter_hidden_dim 1024 \
  --bridge_ckpt /workspace/exps/mimi_bridge_768_control/bridge_pretrained.pt \
  --output_path /workspace/out/baseline.mp4 \
  --metrics_json_path /workspace/out/baseline.json \
  --crop --fix_noise_seed --seed 25

# static with mask (after fix)
python -m generator.eval_bridge_frozen \
  [...same as above...] \
  --static_pose_first --static_gaze_first --static_cam_first \
  --mouth_mask_path /workspace/exps/mouth_mask.pt \
  --output_path /workspace/out/static_masked.mp4 \
  --metrics_json_path /workspace/out/static_masked.json
```

## Report

Write `experiments/YYYY-MM-DD-002-static-head-drift.md` with:

- Mask-building method used (binary dim-probe vs variance-correlation).
- The top-K mouth-correlated dim indices and their correlations.
- Metrics diff: `upper_face_motion_mean`, `mouth_open_max`, `mouth_open_max/mean` ratio — baseline vs static+mask.
- Eyeball verdict from watching the mp4.
- Whether anchor-unmasked-delta was also needed.
- Live-stream test outcome.

## Do not

- Do not retrain the generator. Mask is an inference-time post-hoc projection.
- Do not touch the renderer.
- Do not use `--static_cam_zero`. Proven OOD.
- Do not enable the mask by default. It must stay opt-in via flag so non-static renders are unchanged.
