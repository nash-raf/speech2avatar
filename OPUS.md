# Speech2Avatar Training Review Brief

This document is for a thorough Opus review of the current `speech2avatar` repo state, with emphasis on the **generator training path**, the **AU-conditioning changes**, and the **dataset assumptions needed for fine-tuning**.

The goal is not a style review. The goal is to determine whether the current training code is internally consistent and whether there are any hidden bugs, regressions, or mismatches before AU-conditioned fine-tuning begins.

## Repo Identity

This repo is **not vanilla IMTalker** anymore.

It is best understood as:

- an IMTalker-derived fork
- with AU-conditioned generator changes added
- with static-head-oriented inference behavior for the standalone audio path

Current repo root:

- `/home/user/D/speech2avatar`

Reference codebase that must not be modified:

- `/home/user/D/IMTalker`

## What Changed From Original IMTalker

The generator fork now differs from original IMTalker in these important ways:

1. AU conditioning exists end-to-end in generator training and sampling.
2. The dataset loader supports an `au/` modality and zero-AU fallback.
3. Static pose augmentation exists during training.
4. Fine-tuning-specific freezing and differential learning rates were added.
5. Standalone inference now defaults to a **static pose sequence**, while camera remains null unless explicitly supplied elsewhere.
6. The command-line training code now accepts both `--dataset_path` and a hidden `--dataset_pat` compatibility alias.

## Files That Matter Most

Please read these carefully:

- `/home/user/D/speech2avatar/generator/options/base_options.py`
- `/home/user/D/speech2avatar/generator/dataset.py`
- `/home/user/D/speech2avatar/generator/FM.py`
- `/home/user/D/speech2avatar/generator/FMT.py`
- `/home/user/D/speech2avatar/generator/train.py`
- `/home/user/D/speech2avatar/generator/generate.py`
- `/home/user/D/speech2avatar/scripts/train_au_generator.sh`
- `/home/user/D/speech2avatar/scripts/runpod_dataset_common.py`
- `/home/user/D/speech2avatar/scripts/prepare_generator_dataset.py`
- `/home/user/D/speech2avatar/latest_imtalker.md`

For original-reference comparison only:

- `/home/user/D/IMTalker/generator/FM.py`
- `/home/user/D/IMTalker/generator/FMT.py`
- `/home/user/D/IMTalker/generator/dataset.py`
- `/home/user/D/IMTalker/generator/train.py`
- `/home/user/D/IMTalker/generator/generate.py`
- `/home/user/D/IMTalker/generator/options/base_options.py`

## Current Training Objective

The intended fine-tuning setup is:

- start from pretrained IMTalker generator checkpoint
- add AU conditioning
- fine-tune for more explicit facial control
- preserve a generally static head

The intended dataset modalities are:

- `motion/{stem}.pt` -> tensor `[T, 32]`
- `audio/{stem}.npy` -> array `[T, 768]`
- `smirk/{stem}.pt` -> dict with:
  - `pose_params`: `[T, 3]`
  - `cam`: `[T, 3]`
- `gaze/{stem}.npy` -> array `[T, 2]`
- `au/{stem}.npy` -> array `[T, 17]`

All modalities are expected to be frame-aligned by shared stem and shared `T`.

## Current Option Defaults

Please verify these are coherent with the pretrained checkpoint and training assumptions.

From `generator/options/base_options.py`:

- `fps = 25.0`
- `wav2vec_sec = 2`
- `num_prev_frames = 10`
- `dim_motion = 32`
- `dim_c = 32`
- `dim_h = 512`
- `fmt_depth = 8`
- `num_heads = 8`
- `num_aus = 17`
- `au_dropout_prob = 0.1`
- `static_pose_aug_prob = 0.3`
- `freeze_first_n_blocks = 4`

This means the model currently expects:

- current clip length = `2 * 25 = 50` frames
- previous context = `10` frames
- total transformer context = `60` frames

Please verify this matches original IMTalker’s pretrained regime closely enough for fine-tuning.

## Dataset Loader Behavior

File:

- `/home/user/D/speech2avatar/generator/dataset.py`

### Current behavior

The dataset:

1. uses `motion/*.pt` as the anchor modality
2. requires `audio`, `gaze`, and `smirk` to exist for each stem
3. treats `au` as optional
4. checks lengths eagerly by loading each file
5. keeps only samples with `min_len >= required_len`
6. picks a random contiguous window of:
   - `num_prev_frames + num_frames_for_clip`

### Returned tensors

Per sample it returns:

- `m_now`
- `a_now`
- `gaze`
- `pose`
- `cam`
- `gaze_now`
- `pose_now`
- `cam_now`
- `m_prev`
- `a_prev`
- `gaze_prev`
- `pose_prev`
- `cam_prev`
- `au`
- `au_now`
- `au_prev`
- `m_ref`

### Important implementation details

- AU fallback:
  - if `au/{stem}.npy` is missing, it returns zeros of shape `[required_len, num_aus]`
- static pose augmentation:
  - with probability `static_pose_aug_prob`, it replaces the sampled pose/cam clip with the first frame repeated across the full clip
- reference motion:
  - `m_ref` is one random frame sampled from the full motion segment

### Review questions

Please verify:

1. whether using `motion` as the anchor modality is still the right design
2. whether the eager load-and-check approach is acceptable or a hidden bottleneck
3. whether the returned key set is coherent with the model forward path
4. whether static pose augmentation is implemented correctly for a static-head objective
5. whether AU zero fallback is safe for backward compatibility

## FMGenerator Behavior

File:

- `/home/user/D/speech2avatar/generator/FM.py`

### What it does now

The generator:

- encodes audio with wav2vec2
- projects conditioning streams to `dim_c`
- passes everything into `FlowMatchingTransformer`
- samples motion latents with ODE integration

### New conditioning streams

These projections now exist:

- `audio_projection`
- `gaze_projection`
- `pose_projection`
- `cam_projection`
- `au_projection`

### Training forward path

`forward()` now reads:

- motion current/previous
- audio current/previous
- gaze current/previous
- pose current/previous
- cam current/previous
- AU current/previous

and projects all of them before calling `FMT.forward()`.

### Sampling path

`sample()`:

- computes audio features at frame-aligned length `T`
- aligns pose/gaze/cam/AU inputs to that length
- defaults missing conditioning streams to zeros in projected space
- chunks generation into windows of `wav2vec_sec * fps`

### Current inference fallback behavior

Important current behavior:

- if `pose` is missing at the generator level, `FM.sample()` itself still defaults to zero pose conditioning
- however the standalone inference wrapper in `generator/generate.py` now injects a **static pose sequence before calling `sample()`**
- current standalone inference sets:
  - `pose` = static repeated sequence
  - `cam` = `None`
  - `gaze` = optional if explicitly provided

### Review questions

Please verify:

1. whether AU projection is integrated correctly
2. whether the chunking logic still matches original IMTalker expectations
3. whether the null-conditioning fallback remains safe for gaze/cam/AU
4. whether the static-pose behavior should live in `generate.py` instead of `FM.sample()`
5. whether the current inference semantics are consistent enough for training and demo use

## FlowMatchingTransformer Behavior

File:

- `/home/user/D/speech2avatar/generator/FMT.py`

### Architecture notes

This remains the main transformer backbone with:

- RoPE attention
- timestep embedding
- AdaLN-conditioned residual blocks
- decoder head back to motion latent space

### AU-related changes

The current `forward()` accepts:

- `au`
- `prev_au`

The conditioning sum is now:

- `c = self.c_embedder(a + pose + cam + gaze + au)`

### Dropout behavior

Current dropout behavior:

- `a`, `pose`, `cam`, `gaze` all use `audio_dropout_prob`
- `au` uses `au_dropout_prob`
- previous-context streams use `0.5`

### CFG behavior

`forward_with_cfg()` currently:

- zeros only the audio branch for the unconditional half
- duplicates pose/gaze/cam/AU in both halves

So AU acts as always-on conditioning while audio is the only branch toggled for CFG.

### Important nuance to review

Current code uses:

- `audio_dropout_prob` for pose, cam, and gaze dropout too

Please assess whether that is intentional and correct, or whether separate dropout probabilities should exist for non-audio conditions.

### Review questions

Please verify:

1. whether AU conditioning is inserted at the right point
2. whether the previous/current AU concatenation is handled correctly
3. whether dropout semantics are sound
4. whether the CFG implementation is still mathematically coherent
5. whether there are any hidden shape issues in previous/current concatenation

## Training System Behavior

File:

- `/home/user/D/speech2avatar/generator/train.py`

### Main training loop

The Lightning module:

- builds `FMGenerator`
- applies fine-tuning freeze
- maintains EMA shadow weights
- trains with flow-matching objective on noised motion

### Current losses

Training loss is:

- `fm_loss = L1(pred_flow_anchor, gt_flow)`
- `velocity_loss = L1(framewise_diff(pred), framewise_diff(gt))`
- total = `fm_loss + velocity_loss`

### Batch-key compatibility

`_prepare_batch()` currently aliases:

- `gaze_now -> gaze`
- `pose_now -> pose`
- `cam_now -> cam`
- `au_now -> au`

### Checkpoint loading behavior

`load_ckpt()`:

- prefers EMA weights if present
- strips `model.` prefix if needed
- loads only matching-shaped params
- tolerates missing keys

This is intended to allow the new `au_projection` to stay randomly initialized while the rest comes from pretrained IMTalker weights.

### Fine-tuning freeze

`_apply_finetune_freeze()`:

- freezes the first `freeze_first_n_blocks` transformer blocks

### Optimizer behavior

`configure_optimizers()` creates two param groups:

1. full LR:
   - `au_projection`
2. reduced LR:
   - `audio_projection`
   - `gaze_projection`
   - `pose_projection`
   - `cam_projection`
   - entire `fmt`

### Important thing to inspect

Please verify whether the intended “reduced LR for remaining FMT blocks, decoder, c_embedder, pose/cam/gaze projections, audio_projection” is actually achieved by the current grouping.

Potential subtlety:

- reduced LR includes the whole `fmt`
- frozen params should be ignored because `requires_grad=False`
- but please verify there is no missing trainable module that accidentally receives no optimizer group

### TrainOptions behavior

Current CLI behavior:

- primary arg: `--dataset_path`
- hidden compatibility alias: `--dataset_pat`

Launch command provided in repo:

- `/home/user/D/speech2avatar/scripts/train_au_generator.sh`

### Review questions

Please verify:

1. whether the optimizer groups match the design intent
2. whether any trainable params are accidentally excluded
3. whether EMA logic is correct after partial checkpoint loading
4. whether flow-matching target construction still matches original IMTalker
5. whether `devices=-1` is a good default for current expected training usage

## Standalone Inference Behavior

File:

- `/home/user/D/speech2avatar/generator/generate.py`

### Why this matters for review

This file is not the main training loop, but it reflects current intended conditioning semantics and can expose mismatches between training and inference.

### Current behavior

- loads renderer and generator checkpoints
- preprocesses reference image and audio
- optionally loads pose/gaze/AU files
- currently overrides pose to a static repeated sequence
- currently sets camera to `None`
- leaves gaze optional
- leaves AU optional

### Current static-head policy

The current standalone policy is:

- static pose
- null camera
- optional gaze
- optional AU

Please verify whether that policy is consistent with the desired static-head fine-tuning goal and whether camera should remain null or be static at inference.

## Training Launcher

File:

- `/home/user/D/speech2avatar/scripts/train_au_generator.sh`

Current launcher:

```bash
python generator/train.py \
    --dataset_path /home/user/D/generator_dataset \
    --exp_name au_finetune_v1 \
    --batch_size 16 \
    --iter 500000 \
    --lr 1e-5 \
    --resume_ckpt ./checkpoints/generator.ckpt \
    --num_aus 17 \
    --au_dropout_prob 0.1 \
    --static_pose_aug_prob 0.3 \
    --freeze_first_n_blocks 4 \
    --save_freq 10000 \
    --display_freq 5000
```

Please assess whether this is a sensible default starting point for AU fine-tuning.

## Dataset Preparation Scripts

Files:

- `/home/user/D/speech2avatar/scripts/prepare_generator_dataset.py`
- `/home/user/D/speech2avatar/scripts/runpod_dataset_common.py`

### Important nuance

Please do **not** assume the checked-in RunPod helper is exactly the same as the pod-local helper variants that were used successfully.

In practice, the pod runs for `motion_audio`, `smirk`, and `gaze` required some live troubleshooting and stage-specific environment isolation. The checked-in helper should be reviewed as its own artifact, not assumed to be production-perfect because the data extraction succeeded on patched pods.

### Current checked-in `runpod_dataset_common.py`

Current checked-in behavior:

- lazy-imports some heavy stage-specific modules
- still uses MediaPipe-based `crop_face()` for non-motion stages
- still uses SMIRK helper `utils.mediapipe_utils`
- still expects classic OpenFace via `CPP_OPENFACE_BIN` for full AU extraction, else falls back to OpenFace-3.0 8-AU mapping

### Review questions

Please verify:

1. whether the checked-in script is runnable as-is
2. whether it still over-couples stage dependencies
3. whether `crop_face()` should stay MediaPipe-based or be simplified for already face-cropped MEAD
4. whether the script’s successful pod behavior depended on uncommitted local patching

## Current Dataset State Known To User

Local repo-visible state at the moment:

- `/home/user/D/generator_dataset/au` is complete locally
- local `/home/user/D/generator_dataset/motion`, `audio`, `smirk`, `gaze` are not all complete in this filesystem snapshot
- however those modalities were processed on pods and uploaded as tar files separately

The user’s intended final training dataset structure remains:

```text
/workspace/generator_dataset/
├── motion/
├── audio/
├── smirk/
├── gaze/
└── au/
```

with flat files by stem in each modality folder.

## Most Important Review Questions

Please answer these directly and concretely:

1. Is the current AU-conditioned training path internally consistent?
2. Is the dataset key naming now fully consistent across dataset, FM, FMT, train, and generate?
3. Are there any trainable parameters missing from optimizer groups?
4. Is the current static-head strategy coherent for fine-tuning and inference?
5. Is the current `generate.py` static-pose / null-cam policy the right default?
6. Is there any likely mismatch between training conditioning semantics and inference conditioning semantics?
7. Is the checked-in RunPod helper trustworthy as-is, or should it be refactored before reuse?
8. Are there any hidden shape, device, or checkpoint-loading bugs still present?

## What To Prioritize

Prioritize:

- real bugs
- training/inference mismatches
- optimizer-group mistakes
- bad checkpoint loading assumptions
- conditioning-shape issues
- AU ordering problems
- frame alignment assumptions
- dataset structure mismatches
- places where successful pod execution depended on ad-hoc local patching

Please be strict. The user wants confidence before training, not a polite pass.
