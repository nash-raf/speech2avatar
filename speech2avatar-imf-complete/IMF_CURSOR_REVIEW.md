# iMF Generator Review Note

This branch replaces IMTalker's standard flow-matching generator with an iMF-style generator for the `generator/` stack only.

Parity target for review:

- local reference objective: `/home/user/D/imeanflow/imf.py`
- local reference backbone: `/home/user/D/imeanflow/models/imfDiT.py`

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
  - adaptive weighting using `(loss + norm_eps) ** norm_p`
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

## Mathematical Parity Checklist

Expected training invariants against `imeanflow/imf.py`:

- Interpolation:
  - `z_t = (1 - t) * x + t * e`
  - `v_t = e - x`
- Ordered time sampling:
  - sample `t` and `r`
  - enforce `t >= r`
  - for the `fm_mask` subset, force `r = t`
- Guided target:
  - `v_g = v_t + (1 - 1 / omega) * (v_c - v_u)`
  - for the `fm_mask` subset, use the full interval by forcing `t_min = 0` and `t_max = 1`
  - outside the sampled interval, guidance is disabled by replacing `omega` with `1`
- JVP:
  - JVP is taken on the `u` branch only
  - tangent direction is `(z_t, h) -> (v_c, 1)` in this PyTorch adaptation
  - since `h = t - r`, this corresponds to the intended `(z_t, t, r) -> (v_c, 1, 0)` direction from the JAX formulation
  - this assumes the model is differentiated with respect to the same explicit forward inputs used by the JVP call, with `h` treated as an independent scalar input at JVP time
- Compound target:
  - `V = u + h * stopgrad(du_dt)`
- Loss:
  - `loss_u` compares `V` to `stopgrad(v_g)`
  - `loss_v` compares `v` to `stopgrad(v_g)`
  - final loss is `loss_u + loss_v`
  - adaptive weighting is applied before the mean reduction

## Important Implementation Notes

- This is adapted from `imeanflow/`, but translated to 1D temporal tokens rather than 2D image patches.
- Per-frame conditions are added to motion tokens before the transformer, not injected through AdaLN.
- Prefix tokens are only for scalar schedule/guidance variables.
- RoPE remains 1D over the full prefix + motion sequence.
- Attention currently has `self.fused_attn = False` because forward-mode AD / JVP compatibility was more reliable that way in smoke testing.

## CFG and Dropout Mapping

- CFG is over audio only
- Unconditional branch:
  - audio is zeroed
  - pose, gaze, cam, and ref motion remain present
- Conditioning dropout analog:
  - current flag name is `audio_dropout_prob`
  - only audio is dropped
  - when audio is dropped, the guided target is reset to unguided `v_t`
- Pose, gaze, and cam are not dropped by the current implementation

## Inference Contract

- `nfe=1` is now the default intended mode
- Sampling uses repeated finite-step updates over a fixed grid from `1 -> 0`
- For `nfe > 1`, the same update rule is applied repeatedly on that grid:
  - `z_r = z_t - (t - r) * u(...)`
- At inference time, each chunk uses fixed scalar guidance inputs:
  - `omega = max(a_cfg_scale, 1.0)`
  - `t_min = 0`
  - `t_max = 1`
- Those inference scalars stay fixed within a chunk while only `h = t - r` changes step to step

## Defaults Snapshot

Current notable defaults from `base_options.py` and `train.py`:

| Parameter | Default |
| --- | --- |
| `aux_head_depth` | `4` |
| `fmt_depth` | `8` |
| `nfe` | `1` |
| `P_mean` | `-0.4` |
| `P_std` | `1.0` |
| `data_proportion` | `0.5` |
| `cfg_beta` | `1.0` |
| `norm_p` | `1.0` |
| `norm_eps` | `0.01` |
| `num_time_tokens` | `4` |
| `num_cfg_tokens` | `4` |
| `num_interval_tokens` | `2` |
| `audio_dropout_prob` | `0.1` |
| `omega s_max` in training sampler | `7.0` |
| `t_min` sampling range | `[0, 0.5]` for non-`fm_mask` samples |
| `t_max` sampling range | `[0.5, 1.0]` for non-`fm_mask` samples |

## What Was Verified

Static / synthetic only:

- `python3 -m py_compile` passes for the modified generator files
- A synthetic `_compute_loss()` smoke test runs end-to-end
- `generator/generate.py --help` still runs

Not yet verified:

- real-data convergence
- real checkpoint rendering quality
- 1-step rendered behavior on real audio and conditions

Validation behavior:

- validation currently uses the same full iMF loss path as training
- it is not a simplified proxy loss
- validation keeps grad enabled around the step because the JVP-based objective needs autograd machinery

## Checkpoint Expectations

- Partial checkpoint loading is intentional
- Expected to load reasonably from old FM checkpoints if names and shapes still match:
  - audio encoder
  - audio / gaze / pose / cam projection layers
  - any unchanged top-level tensors with identical names and shapes
- Expected to initialize fresh:
  - prefix token banks
  - shared / dual-head iMF transformer blocks
  - new final heads
  - any tensor whose name or shape no longer matches the old FM checkpoint
- `generate.py` now skips same-name shape mismatches instead of crashing
- `train.py` also performs shape-filtered partial loading

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
