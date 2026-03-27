#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/workspace}"
MEAD_ZIP="${MEAD_ZIP:-${ROOT_DIR}/MEAD.zip}"
MEAD_URL="${MEAD_URL:-https://huggingface.co/datasets/NoahMartinezXiang/MEAD/resolve/main/MEAD.zip}"

mkdir -p "${ROOT_DIR}"
cd "${ROOT_DIR}"
source preprocess/bin/activate

wget -c -O "${MEAD_ZIP}" "${MEAD_URL}"

echo "Downloaded ${MEAD_ZIP}"
