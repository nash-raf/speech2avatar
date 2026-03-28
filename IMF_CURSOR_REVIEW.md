# iMF Generator Review Note

This branch replaces IMTalker's standard flow-matching generator with an iMF-style generator for the `generator/` stack only.

## Scope

Modified files:

- `generator/FMT.py`
- `generator/FM.py`
- `generator/train.py`
- `generator/generate.py`
- `generator/options/base_options.py`

Not changed:

- `generator/dataset.py`
- `generator/wav2vec2.py`
- `renderer/`

## Intended Design

This version is for the non-AU setup.

Conditioning used:

- audio
- gaze
- pose
- cam
- reference motion latent

AU was intentionally left out for this branch.

## Main Architectural Changes

### `generator/FMT.py`

- Removed AdaLN conditioning blocks
- Added RMSNorm-based transformer blocks
- Added zero-init residual gates (`attn_scale`, `mlp_scale`)
- Added QK-norm attention
- Added SwiGLU MLP
- Added scalar prefix-token conditioning for:
  - `h = t - r`
  - `omega`
  - `t_min`
  - `t_max`
- Added dual heads:
  - `u` head for mean flow
  - `v` head for instantaneous velocity
- `v` head is skippable at eval via `return_v=False`

### `generator/FM.py`

- Replaced ODE sampling with finite-step iMF update
- Preserved chunked generation with previous-context frames
- Preserved `cat([ref_x_repeated, x], dim=-1)` before token projection
- Audio CFG now means:
  - conditioned branch = real audio
  - unconditional branch = zeroed audio
- Made `cam` optional:
  - if missing in batch, it falls back to zeros

### `generator/train.py`

- Replaced standard FM loss with iMF-style objective
- Added:
  - `sample_tr()`
  - CFG scale sampling
  - CFG interval sampling
  - JVP-based `u + h * stopgrad(du_dt)` target construction
- Kept EMA logic
- Kept partial checkpoint loading by shape match
- Logs:
  - `train_loss`
  - `loss_u`
  - `loss_v`
  - `val_loss`
  - `val_loss_u`
  - `val_loss_v`

### `generator/generate.py`

- Uses the new finite-step `FM.sample()`
- Checkpoint loading now skips same-name / mismatched-shape tensors instead of crashing

### `generator/options/base_options.py`

Added iMF-related args:

- `aux_head_depth`
- `P_mean`
- `P_std`
- `data_proportion`
- `cfg_beta`
- `norm_p`
- `norm_eps`
- `num_time_tokens`
- `num_cfg_tokens`
- `num_interval_tokens`

Also changed:

- `nfe` default is now `1`

## Important Implementation Notes

- This is adapted from `imeanflow/`, but translated to 1D temporal tokens rather than 2D image patches.
- Per-frame conditions are added to motion tokens before the transformer, not injected through AdaLN.
- Prefix tokens are only for scalar schedule/guidance variables.
- RoPE remains 1D over the full prefix + motion sequence.
- Attention currently has `self.fused_attn = False` because forward-mode AD / JVP compatibility was more reliable that way in smoke testing.

## What Was Verified

- `python3 -m py_compile` passes for the modified generator files
- A synthetic `_compute_loss()` smoke test runs end-to-end
- `generator/generate.py --help` still runs

## What Still Needs Human Review

- Whether the PyTorch JVP translation is faithful enough to the intended iMF objective
- Whether the prefix-token design is the right adaptation for this sequence setting
- Whether the train/inference behavior matches the intended 1-step or low-step iMF solver
- Whether partial checkpoint loading should be stricter or looser
- Whether `cam` should remain active or be fully removed for the no-AU / original-style setup

## Suggested First Real Test

1. Run a very short training smoke test
2. Confirm `loss_u` and `loss_v` decrease
3. Save a checkpoint
4. Run `generator/generate.py` with `nfe=1`
5. Check whether rendered motion is stable and lip-sync remains reasonable
