#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/workspace}"
MEAD_ZIP="${MEAD_ZIP:-${ROOT_DIR}/MEAD.zip}"
MEAD_DIR="${MEAD_DIR:-${ROOT_DIR}/MEAD}"

cd "${ROOT_DIR}"
source preprocess/bin/activate

mkdir -p "${MEAD_DIR}"
unzip -q "${MEAD_ZIP}" -d "${MEAD_DIR}"

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

echo "Unpacked MEAD into ${MEAD_DIR}"
