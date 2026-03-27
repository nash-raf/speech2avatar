# Speech2Avatar AU + RunPod Handoff

This file is a verification brief for another model to review the current state of the repo and the RunPod preprocessing plan.

## Goal

There are two connected goals in this repo:

1. Add AU-conditioned fine-tuning support to the IMTalker-style generator in `speech2avatar/generator/`.
2. Prepare a full generator dataset from MEAD on RunPod using:
   - original motion latents from the renderer motion encoder
   - Wav2Vec2 audio features
   - SMIRK for pose/cam
   - L2CS-Net for gaze
   - OpenFace-based AU extraction

## Files Changed

Core training/inference changes:

- `generator/FM.py`
- `generator/FMT.py`
- `generator/dataset.py`
- `generator/generate.py`
- `generator/options/base_options.py`
- `generator/train.py`

Dataset / RunPod scripts:

- `scripts/prepare_generator_dataset.py`
- `scripts/train_au_generator.sh`
- `scripts/runpod_dataset_common.py`
- `scripts/runpod_setup_environment.sh`
- `scripts/runpod_download_mead.sh`
- `scripts/runpod_unpack_mead.sh`
- `scripts/runpod_download_models.sh`
- `scripts/runpod_prepare_motion_audio.sh`
- `scripts/runpod_prepare_smirk.sh`
- `scripts/runpod_prepare_gaze_l2cs.sh`
- `scripts/runpod_prepare_au_openface.sh`
- `scripts/runpod_prepare_exact_dataset.sh`

Git hygiene:

- `.gitignore`

## Generator Changes To Verify

Please verify these behaviors:

### 1. AU conditioning added end-to-end

- `base_options.py`
  - adds `--num_aus`
  - adds `--au_dropout_prob`
  - adds `--static_pose_aug_prob`
  - adds `--freeze_first_n_blocks`
  - fixes `--dataset_pat` to `--dataset_path`

- `dataset.py`
  - loads `au/{stem}.npy` if present
  - falls back to zeros if AU file is missing
  - returns AU current/previous segments
  - includes static pose augmentation by repeating first pose/cam frame with probability `static_pose_aug_prob`
  - preserves backward compatibility for datasets without AU files

- `FM.py`
  - adds `au_projection`
  - `forward()` reads AU inputs and projects them
  - `sample()` accepts optional `data["au"]`
  - if AU is missing at inference, creates zero AU conditioning

- `FMT.py`
  - `forward()` accepts `au` and `prev_au`
  - applies AU dropout using `au_dropout_prob`
  - concatenates previous and current AU sequences
  - conditioning sum includes AU
  - `forward_with_cfg()` duplicates AU in both CFG branches rather than zeroing AU in the unconditional branch

- `train.py`
  - normalizes batch key naming so dataset outputs match model expectations
  - freezes first `freeze_first_n_blocks` FMT blocks
  - uses differential learning rates:
    - full LR for `au_projection`
    - reduced LR for other unfrozen generator modules

- `generate.py`
  - adds `--au_path`
  - loads AU `.npy` and forwards it into sampling

### 2. Naming mismatch fix

There was a likely mismatch between dataset keys like `gaze_now` and model expectations like `gaze`.

Please verify that training and inference now use a consistent mapping for:

- `gaze`
- `gaze_prev`
- `pose`
- `pose_prev`
- `cam`
- `cam_prev`
- `au`
- `au_prev`

### 3. Batch-size-1 safety

Please verify that a previous `x.squeeze()`-style issue was fixed so the generator forward path does not collapse batch dimension when `B=1`.

## RunPod Dataset Pipeline To Verify

The user wants split scripts rather than one large script, so that failures are isolated by stage.

### Expected RunPod order

From `/workspace/speech2avatar`:

```bash
ROOT_DIR=/workspace ./scripts/runpod_setup_environment.sh
ROOT_DIR=/workspace ./scripts/runpod_download_mead.sh
ROOT_DIR=/workspace ./scripts/runpod_unpack_mead.sh
ROOT_DIR=/workspace ./scripts/runpod_download_models.sh

ROOT_DIR=/workspace GPU_ID=0 ./scripts/runpod_prepare_motion_audio.sh
ROOT_DIR=/workspace GPU_ID=0 ./scripts/runpod_prepare_smirk.sh
ROOT_DIR=/workspace GPU_ID=0 ./scripts/runpod_prepare_gaze_l2cs.sh
ROOT_DIR=/workspace GPU_ID=0 CPP_OPENFACE_BIN=/path/to/FeatureExtraction ./scripts/runpod_prepare_au_openface.sh
```

### Shared stage runner

Please inspect `scripts/runpod_dataset_common.py`.

It is intended to support these stages:

- `motion_audio`
- `smirk`
- `gaze`
- `au`

All stages should write into one flat dataset root:

- `generator_dataset/motion`
- `generator_dataset/audio`
- `generator_dataset/smirk`
- `generator_dataset/gaze`
- `generator_dataset/au`

### Motion/audio stage expectations

- motion:
  - uses original renderer checkpoint
  - feeds each frame into `renderer.latent_token_encoder`
  - writes `{stem}.pt`

- audio:
  - extracts 16k mono wav via `ffmpeg`
  - runs Wav2Vec2
  - interpolates to exactly `T` video frames
  - writes `{stem}.npy`

### SMIRK stage expectations

- uses SMIRK encoder checkpoint
- extracts `pose_params` and `cam`
- writes dict with:
  - `pose_params`
  - `cam`

Please verify that the script includes the MediaPipe asset required by SMIRK:

- `assets/face_landmarker.task`

### L2CS stage expectations

- uses `L2CSNet_gaze360.pkl`
- predicts pitch/yaw
- writes `[T, 2]`

### AU stage expectations

Important nuance:

- `OpenFace-3.0` is not the classic C++ OpenFace `FeatureExtraction` pipeline.
- The linked OpenFace-3.0 repo appears to expose only 8 AUs, not the full HM-Talker 17.

Please verify that the AU stage behaves like this:

1. If `CPP_OPENFACE_BIN` points to classic OpenFace `FeatureExtraction`, use that first.
2. Otherwise fall back to OpenFace-3.0.
3. OpenFace-3.0 fallback maps its smaller AU set into the 17-AU output vector, with unsupported AUs zero-filled.

Please verify whether the exact AU order matches this target:

- `AU01`
- `AU02`
- `AU04`
- `AU05`
- `AU06`
- `AU07`
- `AU09`
- `AU10`
- `AU12`
- `AU14`
- `AU15`
- `AU17`
- `AU20`
- `AU23`
- `AU25`
- `AU26`
- `AU45`

### Hugging Face MEAD source

The intended download source is:

- `https://huggingface.co/datasets/NoahMartinezXiang/MEAD/resolve/main/MEAD.zip`

Please verify the download/unpack scripts are correct and do not unpack into an unexpected nested directory structure.

## Known Local State

Local full preprocessing was not completed.

Partial local outputs exist from smoke tests only:

- `audio`: partial
- `smirk`: partial
- `gaze`: partial
- `au`: partial
- `motion`: not completed locally

This is expected because the user plans to run the full pipeline on RunPod.

## Review Questions

Please answer these clearly:

1. Are the AU-conditioning generator changes internally consistent and safe?
2. Is the batch-key mapping now correct across dataset, train, FM, and generate paths?
3. Is the RunPod split-script flow coherent and runnable in principle?
4. Are there any obvious missing dependencies, import issues, or path issues?
5. Is the OpenFace logic honest about the difference between classic C++ OpenFace and OpenFace-3.0?
6. Is `runpod_prepare_exact_dataset.sh` still worth keeping, or should the repo keep only the split scripts?
7. Are there any likely runtime bugs that should be fixed before using RunPod?

## What Matters Most

Prioritize finding:

- real bugs
- broken imports
- wrong tensor shapes
- wrong AU ordering
- incorrect frame alignment
- dangerous assumptions in RunPod paths or model downloads

The goal is not a style review. The goal is to catch things that would make preprocessing or AU fine-tuning fail.
