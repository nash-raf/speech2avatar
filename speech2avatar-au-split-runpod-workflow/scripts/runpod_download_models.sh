#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/workspace}"
S2A_DIR="${S2A_DIR:-${ROOT_DIR}/speech2avatar}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${S2A_DIR}/checkpoints}"
MODEL_DIR="${MODEL_DIR:-${ROOT_DIR}/model_cache}"

mkdir -p "${CHECKPOINT_DIR}" "${MODEL_DIR}" "${CHECKPOINT_DIR}/wav2vec2-base-960h"
cd "${ROOT_DIR}"
source preprocess/bin/activate

download_if_missing() {
  local url="$1"
  local dest="$2"
  if [[ ! -s "${dest}" ]]; then
    mkdir -p "$(dirname "${dest}")"
    wget -c -O "${dest}" "${url}"
  fi
}

download_if_missing "https://huggingface.co/cbsjtu01/IMTalker/resolve/main/renderer.ckpt" "${CHECKPOINT_DIR}/renderer.ckpt"
download_if_missing "https://huggingface.co/cbsjtu01/IMTalker/resolve/main/generator.ckpt" "${CHECKPOINT_DIR}/generator.ckpt"
download_if_missing "https://huggingface.co/cbsjtu01/IMTalker/resolve/main/wav2vec2-base-960h/config.json" "${CHECKPOINT_DIR}/wav2vec2-base-960h/config.json"
download_if_missing "https://huggingface.co/cbsjtu01/IMTalker/resolve/main/wav2vec2-base-960h/pytorch_model.bin" "${CHECKPOINT_DIR}/wav2vec2-base-960h/pytorch_model.bin"
download_if_missing "https://huggingface.co/cbsjtu01/IMTalker/resolve/main/wav2vec2-base-960h/preprocessor_config.json" "${CHECKPOINT_DIR}/wav2vec2-base-960h/preprocessor_config.json"
download_if_missing "https://huggingface.co/cbsjtu01/IMTalker/resolve/main/wav2vec2-base-960h/feature_extractor_config.json" "${CHECKPOINT_DIR}/wav2vec2-base-960h/feature_extractor_config.json"

if [[ ! -s "${MODEL_DIR}/SMIRK_em1.pt" ]]; then
  gdown --fuzzy "https://drive.google.com/file/d/1T65uEd9dVLHgVw5KiUYL66NUee-MCzoE/view?usp=sharing" -O "${MODEL_DIR}/SMIRK_em1.pt"
fi

if [[ ! -s "${MODEL_DIR}/L2CSNet_gaze360.pkl" ]]; then
  rm -rf "${MODEL_DIR}/l2cs_drive"
  gdown --folder "https://drive.google.com/drive/folders/17p6ORr-JQJcw-eYtG2WGNiuS_qVKwdWd?usp=sharing" -O "${MODEL_DIR}/l2cs_drive"
  if [[ -f "${MODEL_DIR}/l2cs_drive/L2CSNet_gaze360.pkl" ]]; then
    cp "${MODEL_DIR}/l2cs_drive/L2CSNet_gaze360.pkl" "${MODEL_DIR}/L2CSNet_gaze360.pkl"
  fi
fi

echo "Checkpoints and pretrained models are ready"
