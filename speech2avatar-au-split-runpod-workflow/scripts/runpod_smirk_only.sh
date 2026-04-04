#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/workspace}"
REPO_URL="${REPO_URL:-https://github.com/nash-raf/speech2avatar.git}"
REPO_DIR="${REPO_DIR:-${ROOT_DIR}/speech2avatar}"
TOOLS_DIR="${TOOLS_DIR:-${ROOT_DIR}/external_tools}"
SMIRK_DIR="${SMIRK_DIR:-${TOOLS_DIR}/smirk}"
MODEL_DIR="${MODEL_DIR:-${ROOT_DIR}/model_cache}"
MEAD_ZIP="${MEAD_ZIP:-${ROOT_DIR}/MEAD.zip}"
MEAD_DIR="${MEAD_DIR:-${ROOT_DIR}/MEAD}"
MEAD_URL="${MEAD_URL:-https://huggingface.co/datasets/NoahMartinezXiang/MEAD/resolve/main/MEAD.zip}"
GPU_ID="${GPU_ID:-0}"

apt-get update && apt-get install -y git ffmpeg libgl1 libglib2.0-0 wget

cd "${ROOT_DIR}"

if [[ ! -d "${REPO_DIR}" ]]; then
  git clone "${REPO_URL}" "${REPO_DIR}"
fi

python -m venv preprocess_smirk
source preprocess_smirk/bin/activate

python -m pip install --upgrade pip setuptools wheel

pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 \
  --index-url https://download.pytorch.org/whl/cu121

pip install \
  "numpy==1.26.4" \
  "opencv-python<4.10.0.0" \
  "mediapipe==0.10.14" \
  "timm==1.0.9" \
  "einops==0.8.0" \
  "scikit-image==0.24.0" \
  "pandas==2.2.3" \
  "tqdm==4.66.2" \
  "gdown==5.2.0"

mkdir -p "${TOOLS_DIR}" "${MODEL_DIR}"

if [[ ! -d "${SMIRK_DIR}" ]]; then
  git clone --depth 1 https://github.com/georgeretsi/smirk "${SMIRK_DIR}"
fi

mkdir -p "${SMIRK_DIR}/assets"
if [[ ! -s "${SMIRK_DIR}/assets/face_landmarker.task" ]]; then
  wget -c -O "${SMIRK_DIR}/assets/face_landmarker.task" \
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
fi

if [[ ! -s "${MODEL_DIR}/SMIRK_em1.pt" ]]; then
  gdown --fuzzy "https://drive.google.com/file/d/1T65uEd9dVLHgVw5KiUYL66NUee-MCzoE/view?usp=sharing" -O "${MODEL_DIR}/SMIRK_em1.pt"
fi

if [[ ! -f "${MEAD_ZIP}" ]]; then
  wget -c -O "${MEAD_ZIP}" "${MEAD_URL}"
fi

if [[ ! -d "${MEAD_DIR}/train" ]]; then
  mkdir -p "${MEAD_DIR}"
  unzip -q "${MEAD_ZIP}" -d "${MEAD_DIR}"
fi

if [[ ! -d "${MEAD_DIR}/train" && -d "${MEAD_DIR}/MEAD/train" ]]; then
  shopt -s dotglob nullglob
  mv "${MEAD_DIR}/MEAD/"* "${MEAD_DIR}/"
  rmdir "${MEAD_DIR}/MEAD"
  shopt -u dotglob nullglob
fi

if [[ ! -d "${MEAD_DIR}/train" ]]; then
  echo "Expected ${MEAD_DIR}/train after unzip, but it was not found." >&2
  exit 1
fi

cd "${REPO_DIR}"

python scripts/runpod_dataset_common.py \
  --stage smirk \
  --root_dir "${ROOT_DIR}" \
  --gpu_id "${GPU_ID}"
