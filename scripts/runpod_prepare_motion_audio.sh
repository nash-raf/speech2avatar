#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/workspace}"
S2A_DIR="${S2A_DIR:-${ROOT_DIR}/speech2avatar}"
cd "${S2A_DIR}"
source "${ROOT_DIR}/preprocess/bin/activate"

python scripts/runpod_dataset_common.py \
  --stage motion_audio \
  --root_dir "${ROOT_DIR}" \
  "$@"
