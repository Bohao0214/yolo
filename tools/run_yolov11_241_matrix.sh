#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "[notice] tools/run_yolov11_241_matrix.sh was renamed to tools/run_yolov11_241_module_combo.sh; forwarding automatically." >&2
exec bash "${ROOT_DIR}/tools/run_yolov11_241_module_combo.sh" "$@"
