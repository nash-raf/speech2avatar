#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/workspace}"
MEAD_ZIP="${MEAD_ZIP:-${ROOT_DIR}/MEAD.zip}"
MEAD_DIR="${MEAD_DIR:-${ROOT_DIR}/MEAD}"

cd "${ROOT_DIR}"
source preprocess/bin/activate

mkdir -p "${MEAD_DIR}"
unzip -q "${MEAD_ZIP}" -d "${MEAD_DIR}"

echo "Unpacked MEAD into ${MEAD_DIR}"
