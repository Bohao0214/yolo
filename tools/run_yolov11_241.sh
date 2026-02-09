#!/usr/bin/env bash
set -euo pipefail

export LANG=C.UTF-8
export LC_ALL=C.UTF-8

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

BASE_CONFIG_DEFAULT="${ROOT_DIR}/configs/yolo11/enhance241/defect241.yaml"
BASE_CONFIG="${BASE_CONFIG:-${BASE_CONFIG_DEFAULT}}"
PYTHON_BIN="${PYTHON_BIN:-python}"

usage() {
  cat <<'USAGE'
Usage:
  bash tools/run_yolov11_241.sh [base_config.yaml] [switches...]

Switches (implemented):
  a3    Enable 2.4.1 a3 (SPDConvDownsample P3 downsample)
  b1    Enable 2.4.1 b1 (ASFF-lite P4->P3 fuse)
  b2    Enable 2.4.1 b2-safe (residual P4->P3 fuse, baseline-safe init)
  b3    Enable 2.4.1 b3 (NASFPNLite P5->P4 + P4->P3)
  d1    Enable 2.4.1 d1 (Add P2 detect head)

Examples:
  bash tools/run_yolov11_241.sh
  bash tools/run_yolov11_241.sh a3
  bash tools/run_yolov11_241.sh b1
  bash tools/run_yolov11_241.sh b2
  bash tools/run_yolov11_241.sh b3
  bash tools/run_yolov11_241.sh d1
  bash tools/run_yolov11_241.sh b1 b2
  bash tools/run_yolov11_241.sh a3 b3 d1
  bash tools/run_yolov11_241.sh configs/yolo11/enhance241/defect241.yaml b1
  bash tools/run_yolov11_241.sh configs/yolo11/enhance241/defect241.yaml b2
  bash tools/run_yolov11_241.sh configs/yolo11/enhance241/defect241.yaml b1 b2

Env:
  PYTHON_BIN=/path/to/python
  BASE_CONFIG=/path/to/base_config.yaml
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -gt 0 ]]; then
  if [[ "$1" == *.yml || "$1" == *.yaml ]]; then
    BASE_CONFIG="$1"
    shift
  fi
fi

if [[ ! -f "${BASE_CONFIG}" ]]; then
  echo "[error] base config not found: ${BASE_CONFIG}" >&2
  exit 2
fi

ensure_ultralytics() {
  local py="$1"
  "${py}" - <<'PY' >/dev/null 2>&1
import ultralytics  # noqa: F401
PY
}

if ! ensure_ultralytics "${PYTHON_BIN}"; then
  if [[ -x "/home/ubuntu/anaconda3/envs/yolo11/bin/python" ]] && ensure_ultralytics "/home/ubuntu/anaconda3/envs/yolo11/bin/python"; then
    PYTHON_BIN="/home/ubuntu/anaconda3/envs/yolo11/bin/python"
  else
    echo "[error] Failed to import ultralytics. Activate conda env 'yolo11' or set PYTHON_BIN." >&2
    exit 3
  fi
fi

SWITCHES=("$@")
if [[ ${#SWITCHES[@]} -eq 1 ]]; then
  if [[ "${SWITCHES[0]}" =~ [[:space:]] ]]; then
    read -r -a SWITCHES <<<"${SWITCHES[0]}"
  fi
fi
ENABLE_A3="false"
ENABLE_B1="false"
ENABLE_B2="false"
ENABLE_B3="false"
ENABLE_D1="false"
UNKNOWN=()

for sw in "${SWITCHES[@]}"; do
  key="$(echo "${sw}" | tr '[:upper:]' '[:lower:]')"
  case "${key}" in
    "" ) ;;
    a3 ) ENABLE_A3="true" ;;
    b1 ) ENABLE_B1="true" ;;
    b2 ) ENABLE_B2="true" ;;
    b3 ) ENABLE_B3="true" ;;
    d1 ) ENABLE_D1="true" ;;
    * ) UNKNOWN+=("${sw}") ;;
  esac
done

if [[ ${#UNKNOWN[@]} -gt 0 ]]; then
  echo "[error] Unsupported switches (only a3/b1/b2/b3/d1 are implemented): ${UNKNOWN[*]}" >&2
  exit 4
fi

CONFIG_TO_RUN="${BASE_CONFIG}"

if [[ "${ENABLE_A3}" == "true" || "${ENABLE_B1}" == "true" || "${ENABLE_B2}" == "true" || "${ENABLE_B3}" == "true" || "${ENABLE_D1}" == "true" ]]; then
  ENABLED_KEYS=()
  [[ "${ENABLE_A3}" == "true" ]] && ENABLED_KEYS+=("a3")
  [[ "${ENABLE_B1}" == "true" ]] && ENABLED_KEYS+=("b1")
  [[ "${ENABLE_B2}" == "true" ]] && ENABLED_KEYS+=("b2")
  [[ "${ENABLE_B3}" == "true" ]] && ENABLED_KEYS+=("b3")
  [[ "${ENABLE_D1}" == "true" ]] && ENABLED_KEYS+=("d1")

  if [[ ${#ENABLED_KEYS[@]} -eq 1 ]]; then
    case "${ENABLED_KEYS[0]}" in
      b1 )
        DERIVED_CONFIG="${ROOT_DIR}/configs/yolo11/enhance241/defect241b1.yaml"
        DERIVED_SUFFIX="__b1"
        ;;
      b2 )
        DERIVED_CONFIG="${ROOT_DIR}/configs/yolo11/enhance241/defect241b2.yaml"
        DERIVED_SUFFIX="__b2"
        ;;
      a3 )
        DERIVED_CONFIG="${ROOT_DIR}/configs/yolo11/enhance241/defect241a3.yaml"
        DERIVED_SUFFIX="__a3"
        ;;
      b3 )
        DERIVED_CONFIG="${ROOT_DIR}/configs/yolo11/enhance241/defect241b3.yaml"
        DERIVED_SUFFIX="__b3"
        ;;
      d1 )
        DERIVED_CONFIG="${ROOT_DIR}/configs/yolo11/enhance241/defect241d1.yaml"
        DERIVED_SUFFIX="__d1"
        ;;
    esac
  else
    JOINED="$(IFS=_; echo "${ENABLED_KEYS[*]}")"
    DERIVED_CONFIG="${ROOT_DIR}/configs/yolo11/enhance241/defect241${JOINED}.yaml"
    DERIVED_SUFFIX="$(printf "__%s" "${ENABLED_KEYS[@]}")"
  fi

  if [[ ! -f "${DERIVED_CONFIG}" ]]; then
    export _E241_BASE="${BASE_CONFIG}"
    export _E241_OUT="${DERIVED_CONFIG}"
    export _E241_KEYS="${ENABLED_KEYS[*]}"
    export _E241_SUFFIX="${DERIVED_SUFFIX}"
    "${PYTHON_BIN}" - <<'PY'
import os
from pathlib import Path

import yaml

base = Path(os.environ["_E241_BASE"]).resolve()
out = Path(os.environ["_E241_OUT"]).resolve()
keys = os.environ["_E241_KEYS"].split()
suffix = os.environ["_E241_SUFFIX"]

with base.open("r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}
if not isinstance(cfg, dict):
    raise SystemExit(f"Base YAML must be a mapping: {base}")

enh = cfg.setdefault("enhance241", {})
if not isinstance(enh, dict):
    raise SystemExit("enhance241 must be a mapping in base config.")
enh["a3"] = False
enh["b1"] = False
enh["b2"] = False
enh["b3"] = False
enh["d1"] = False
for k in keys:
    enh[k] = True
if enh.get("b1"):
    enh["b1_version"] = "v3"

exp = str(cfg.get("exp_name", "defect241"))
if not exp.endswith(suffix):
    cfg["exp_name"] = exp + suffix

out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
print(str(out))
PY
  fi

  CONFIG_TO_RUN="${DERIVED_CONFIG}"
fi

echo "[run] config=${CONFIG_TO_RUN} switches=${SWITCHES[*]:-(none)} python=${PYTHON_BIN}"
"${PYTHON_BIN}" "${ROOT_DIR}/src/train.py" --config "${CONFIG_TO_RUN}"
