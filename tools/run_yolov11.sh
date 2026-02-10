#!/usr/bin/env bash
set -euo pipefail

export LANG=C.UTF-8
export LC_ALL=C.UTF-8

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${1:-${ROOT_DIR}/configs/yolo11/defect.yaml}"

python "${ROOT_DIR}/src/train.py" --config "${CONFIG}"

# bash tools/run_yolov11.sh configs/yolo11/defect.yaml
