#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  cat <<'EOF'
Usage:
  bash tools/run_teacher_student_infer.sh <CLIP_ID> <REF_IMAGE> [ROOT]

Example:
  bash tools/run_teacher_student_infer.sh \
    WDA_BrendaLawrence_000_5427_5507 \
    /workspace/refs/brenda.jpg \
    /workspace/hdtf_preprocess

What it does:
  - runs wav2vec teacher inference on GPU 0
  - runs Mimi student replay inference on GPU 1
  - waits for both to finish
  - writes outputs under:
      /workspace/out_teacher
      /workspace/out_student

Requirements:
  - run from /workspace/IMTalker
  - python env at /workspace/preprocess_5090
EOF
  exit 1
fi

CLIP_ID="$1"
REF_IMAGE="$2"
ROOT="${3:-/workspace/hdtf_preprocess}"

if [[ ! -f "$REF_IMAGE" ]]; then
  echo "ref image not found: $REF_IMAGE" >&2
  exit 1
fi

if [[ ! -f "$ROOT/audio_raw/${CLIP_ID}.wav" ]]; then
  echo "audio not found: $ROOT/audio_raw/${CLIP_ID}.wav" >&2
  exit 1
fi

if [[ ! -f "$ROOT/audio_rt_aligned/${CLIP_ID}.npy" ]]; then
  echo "student aligned features not found: $ROOT/audio_rt_aligned/${CLIP_ID}.npy" >&2
  exit 1
fi

if [[ ! -f "$ROOT/audio_wav2vec/${CLIP_ID}.npy" ]]; then
  echo "teacher wav2vec features not found: $ROOT/audio_wav2vec/${CLIP_ID}.npy" >&2
  exit 1
fi

if [[ ! -f "$ROOT/smirk/${CLIP_ID}.pt" ]]; then
  echo "smirk file not found: $ROOT/smirk/${CLIP_ID}.pt" >&2
  exit 1
fi

if [[ ! -f "$ROOT/gaze/${CLIP_ID}.npy" ]]; then
  echo "gaze file not found: $ROOT/gaze/${CLIP_ID}.npy" >&2
  exit 1
fi

cd /workspace/IMTalker
source /workspace/preprocess_5090/bin/activate
export PYTHONPATH=/workspace/IMTalker

mkdir -p /workspace/out_teacher /workspace/out_student /workspace/refs_tmp

REF_EXT="${REF_IMAGE##*.}"
REF_MATCH="/workspace/refs_tmp/${CLIP_ID}.${REF_EXT}"
ln -sfn "$(readlink -f "$REF_IMAGE")" "$REF_MATCH"

STUDENT_META_ARGS=()
if [[ -f "$ROOT/audio_rt_meta/${CLIP_ID}.json" ]]; then
  STUDENT_META_ARGS+=(--meta_json "$ROOT/audio_rt_meta/${CLIP_ID}.json")
else
  echo "[run] audio_rt_meta missing for $CLIP_ID, continuing without --meta_json"
fi

echo "[run] clip=$CLIP_ID"
echo "[run] ref_symlink=$REF_MATCH"
echo "[run] launching teacher on GPU0 and student on GPU1"

CUDA_VISIBLE_DEVICES=0 python generator/generate.py \
  --ref_path "$REF_MATCH" \
  --aud_path "$ROOT/audio_raw/${CLIP_ID}.wav" \
  --pose_path "$ROOT/smirk" \
  --gaze_path "$ROOT/gaze" \
  --generator_path "/workspace/IMTalker/checkpoints/generator.ckpt" \
  --renderer_path "/workspace/IMTalker/checkpoints/renderer.ckpt" \
  --wav2vec_model_path "/workspace/IMTalker/checkpoints/wav2vec2-base-960h" \
  --a_cfg_scale 2.0 \
  --nfe 10 \
  --crop \
  --res_dir "/workspace/out_teacher" \
  >"/workspace/out_teacher/${CLIP_ID}.log" 2>&1 &
PID_TEACHER=$!

CUDA_VISIBLE_DEVICES=1 python tools/replay_live_conditioning.py \
  --ref_path "$REF_MATCH" \
  --aligned_npy "$ROOT/audio_rt_aligned/${CLIP_ID}.npy" \
  --wav_path "$ROOT/audio_raw/${CLIP_ID}.wav" \
  --pose_pt "$ROOT/smirk/${CLIP_ID}.pt" \
  --gaze_npy "$ROOT/gaze/${CLIP_ID}.npy" \
  --generator_path "/workspace/IMTalker/ckpts_mimi/cont_12000_adamw_64_batch.ckpt" \
  --renderer_path "/workspace/IMTalker/checkpoints/renderer.ckpt" \
  --audio_feat_dim 512 \
  --a_cfg_scale 1.0 \
  --nfe 5 \
  --crop \
  "${STUDENT_META_ARGS[@]}" \
  --output_mp4 "/workspace/out_student/${CLIP_ID}_silent.mp4" \
  --output_muxed_mp4 "/workspace/out_student/${CLIP_ID}.mp4" \
  >"/workspace/out_student/${CLIP_ID}.log" 2>&1 &
PID_STUDENT=$!

FAIL=0

wait "$PID_TEACHER" || FAIL=1
wait "$PID_STUDENT" || FAIL=1

echo "[done] teacher_log=/workspace/out_teacher/${CLIP_ID}.log"
echo "[done] student_log=/workspace/out_student/${CLIP_ID}.log"
echo "[done] teacher_mp4=/workspace/out_teacher/${CLIP_ID}.mp4"
echo "[done] student_mp4=/workspace/out_student/${CLIP_ID}.mp4"

if [[ $FAIL -ne 0 ]]; then
  echo "[error] one or both inference jobs failed" >&2
  exit 1
fi

echo "[ok] both inference jobs finished"
