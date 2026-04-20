# IMTalker Two-Fork Summary

We now have two separate IMTalker training forks, and both are working.

## Fork locations

- Discrete audio fork:
  - `/home/user/D/imtalkerTries/imtalker_discrete`
- Continuous audio fork:
  - `/home/user/D/imtalkerTries/imtalker_continuous`

## Why there are two forks

We wanted to test two different ways of conditioning IMTalker on Moshi/Mimi audio features.

The preprocessed dataset provides two audio forms:

- `audio_discrete/`
  - Mimi Codebook 0 token IDs
  - raw 12.5 Hz sequence
- `audio_continuous/`
  - Mimi pre-quantizer dense latents
  - shape `[T, 512]`
  - raw 12.5 Hz sequence

Because these are different input types, we split IMTalker into two clean codebases instead of mixing both paths into one training fork.

## 1. Discrete fork

Path:

- `/home/user/D/imtalkerTries/imtalker_discrete`

What it does:

- trains IMTalker from cached discrete Mimi/Moshi token IDs
- uses an embedding-based audio encoder
- takes integer audio tokens as input
- upsamples them dynamically from 12.5 Hz to video frame rate during the forward pass
- then sends them through IMTalker’s `audio_projection` and FMT

Key idea:

- token IDs -> `nn.Embedding(...)` -> continuous hidden vectors -> temporal interpolation -> IMTalker

## 2. Continuous fork

Path:

- `/home/user/D/imtalkerTries/imtalker_continuous`

What it does:

- trains IMTalker from cached continuous Mimi latents
- takes dense `[T, 512]` float audio features directly
- dynamically interpolates them from 12.5 Hz to video frame rate during the forward pass
- then feeds them into IMTalker’s `audio_projection` and FMT

Key idea:

- continuous Mimi latents -> temporal interpolation -> IMTalker

## Shared behavior

Both forks now support:

- loading cached `.npy` audio features instead of raw wav + wav2vec extraction
- dataset slicing that respects the mismatch between:
  - audio token rate: 12.5 Hz
  - motion/frame rate: video FPS
- `--audio_subdir`
- training from the preprocessed HDTF dataset layout
- optional auxiliary-condition disabling
- working training on the pod
- working cached-feature inference

## Important training conclusion

One major lesson from the training work:

- `--freeze_non_audio` was not the right strategy for a full audio-modality replacement

Why:

- freezing most of the model kept it too close to the predict-zero / non-adapting baseline

What worked better:

- full unfreeze training

Once we switched to full unfreeze, loss dropped properly and visual output started looking correct.

## Current status

Both forks are now in a usable state:

- discrete fork: working
- continuous fork: working

So at this point we have:

1. one IMTalker fork for discrete Mimi tokens
2. one IMTalker fork for continuous Mimi latents

and both can train from the cached HDTF preprocessing outputs.
