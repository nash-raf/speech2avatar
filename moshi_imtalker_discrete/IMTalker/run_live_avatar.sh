#!/usr/bin/env bash
set -euo pipefail

# Run from ANY working directory by passing the FULL path to this script, e.g. on RunPod:
#   bash /workspace/IMTalker/run_live_avatar.sh --ref_path ... --generator_path ... --renderer_path ...
# (Running `bash run_live_avatar.sh` only works if your current directory contains this file.)
#
# Optional: set checkpoints once, then call with no path args (flags you pass later still win):
#   export IMTALKER_REF_PATH=/path/to/ref.png
#   export IMTALKER_GENERATOR_PATH=/path/to/gen.ckpt
#   export IMTALKER_RENDERER_PATH=/path/to/ren.ckpt
#   bash /workspace/IMTalker/run_live_avatar.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAUNCH_PY="${SCRIPT_DIR}/launch_live.py"
if [[ ! -f "${LAUNCH_PY}" ]]; then
  echo "error: launch_live.py missing beside this script (${SCRIPT_DIR})" >&2
  exit 1
fi

MOSHI_REPO="${MOSHI_REPO:-$(cd "${SCRIPT_DIR}/../moshi" && pwd)}"

export PYTHONPATH="${SCRIPT_DIR}:${MOSHI_REPO}/moshi${PYTHONPATH:+:${PYTHONPATH}}"

_PATH_ARGS=()
[[ -n "${IMTALKER_REF_PATH:-}" ]] && _PATH_ARGS+=(--ref_path "${IMTALKER_REF_PATH}")
[[ -n "${IMTALKER_GENERATOR_PATH:-}" ]] && _PATH_ARGS+=(--generator_path "${IMTALKER_GENERATOR_PATH}")
[[ -n "${IMTALKER_RENDERER_PATH:-}" ]] && _PATH_ARGS+=(--renderer_path "${IMTALKER_RENDERER_PATH}")

# Reply-length caps. Old defaults (1 sentence / 40 tokens) closed the Moshi
# session after the very first '.','!','?'. The launcher defaults are now
# 4 / 200; tune via env or pass --max_sentences / --max_text_tokens directly.
_LIMIT_ARGS=()
[[ -n "${IMTALKER_MAX_SENTENCES:-}" ]] && _LIMIT_ARGS+=(--max_sentences "${IMTALKER_MAX_SENTENCES}")
[[ -n "${IMTALKER_MAX_TEXT_TOKENS:-}" ]] && _LIMIT_ARGS+=(--max_text_tokens "${IMTALKER_MAX_TEXT_TOKENS}")
[[ "${IMTALKER_DEBUG_SESSION:-0}" == "1" ]] && _LIMIT_ARGS+=(--debug_session)

exec python3 "${LAUNCH_PY}" --moshi_repo "${MOSHI_REPO}" "${_PATH_ARGS[@]}" "${_LIMIT_ARGS[@]}" "$@"
