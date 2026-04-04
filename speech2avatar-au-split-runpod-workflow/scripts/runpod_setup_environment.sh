#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/workspace}"
S2A_DIR="${S2A_DIR:-${ROOT_DIR}/speech2avatar}"
TOOLS_DIR="${TOOLS_DIR:-${ROOT_DIR}/external_tools}"
SMIRK_DIR="${SMIRK_DIR:-${TOOLS_DIR}/smirk}"
L2CS_DIR="${L2CS_DIR:-${TOOLS_DIR}/L2CS-Net}"
OPENFACE3_DIR="${OPENFACE3_DIR:-${TOOLS_DIR}/OpenFace-3.0}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MEDIAPIPE_FACE_TASK_URL="${MEDIAPIPE_FACE_TASK_URL:-https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task}"

mkdir -p "${ROOT_DIR}"
cd "${ROOT_DIR}"

if [[ ! -d preprocess ]]; then
  "${PYTHON_BIN}" -m venv preprocess
fi

source preprocess/bin/activate

python -m pip install --upgrade pip setuptools wheel
python -m pip install \
  torch \
  torchvision \
  torchaudio \
  pytorch-lightning==2.2.1 \
  torchdiffeq==0.2.5 \
  huggingface_hub \
  transformers \
  librosa \
  pandas \
  timm \
  tqdm \
  gdown \
  opencv-python-headless \
  scikit-image \
  av \
  einops \
  mediapipe \
  face_alignment

mkdir -p "${TOOLS_DIR}"

if [[ ! -d "${SMIRK_DIR}" ]]; then
  git clone --depth 1 https://github.com/georgeretsi/smirk "${SMIRK_DIR}"
fi
if [[ ! -d "${L2CS_DIR}" ]]; then
  git clone --depth 1 https://github.com/Ahmednull/L2CS-Net "${L2CS_DIR}"
fi
if [[ ! -d "${OPENFACE3_DIR}" ]]; then
  git clone --depth 1 https://github.com/CMU-MultiComp-Lab/OpenFace-3.0 "${OPENFACE3_DIR}"
fi

python -m pip install -r "${SMIRK_DIR}/requirements.txt" || true
python -m pip install -e "${L2CS_DIR}"
python -m pip install -r "${OPENFACE3_DIR}/requirements.txt" || true

mkdir -p "${SMIRK_DIR}/assets"
if [[ ! -s "${SMIRK_DIR}/assets/face_landmarker.task" ]]; then
  wget -c -O "${SMIRK_DIR}/assets/face_landmarker.task" "${MEDIAPIPE_FACE_TASK_URL}"
fi

echo "Environment ready in ${ROOT_DIR}/preprocess"
