#!/usr/bin/env bash
# Print last N lines of a file (default 80).
# Usage: tail_log.sh <path> [N]
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: tail_log.sh <path> [N]" >&2
  exit 2
fi

N="${2:-80}"
tail -n "$N" "$1"
