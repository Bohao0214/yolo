#!/usr/bin/env bash
set -euo pipefail

export LANG=C.UTF-8
export LC_ALL=C.UTF-8

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

BASE_CONFIG_DEFAULT="${ROOT_DIR}/configs/yolo11/enhance241/defect241.yaml"
BASE_CONFIG="${BASE_CONFIG:-${BASE_CONFIG_DEFAULT}}"
PYTHON_BIN="${PYTHON_BIN:-python}"
E241_SAFE_BATCH="${E241_SAFE_BATCH:-6}"
E241_SAFE_WORKERS="${E241_SAFE_WORKERS:-4}"

usage() {
  cat <<'USAGE'
Usage:
  bash tools/run_yolov11_241.sh [base_config.yaml] [switches...]

Switches (implemented):
  a3    Enable 2.4.1 a3 (SPDConvDownsample P3 downsample)
  a5    Enable 2.4.1 a5 (P3-side residual-safe lightweight enhancement)
  a7    Enable 2.4.1 a7 (HorNet/C3HB-style residual-safe P3 generation enhancer)
  b1    Enable 2.4.1 b1 (ASFF-lite P4->P3 fuse)
  b2    Enable 2.4.1 b2-safe (residual P4->P3 fuse, baseline-safe init)
  b3    Enable 2.4.1 b3 (NASFPNLite P5->P4 + P4->P3)
  b5    Enable 2.4.1 b5 (GFPN-like CSPStage fusion refinement)
  b7    Enable 2.4.1 b7 (CARAFE residual-safe upsampling)
  c5    Enable 2.4.1 c5 (BRA residual-safe semantic enhancer)
  c7    Enable 2.4.1 c7 (MCBAM residual-safe gate)
  d5    Enable 2.4.1 d5 (Add P2 stride=4/160x160 detect head)
  d7    Enable 2.4.1 d7 (keep only small-target detect head, P3-only)
  d3    Enable 2.4.1 d3 (P3 logit temperature; minimal score-shaping)
  d1    Legacy alias of d3 (kept for backward compatibility)

Examples:
  bash tools/run_yolov11_241.sh
  bash tools/run_yolov11_241.sh a3
  bash tools/run_yolov11_241.sh a5
  bash tools/run_yolov11_241.sh a7
  bash tools/run_yolov11_241.sh b1
  bash tools/run_yolov11_241.sh b2
  bash tools/run_yolov11_241.sh b3
  bash tools/run_yolov11_241.sh b5
  bash tools/run_yolov11_241.sh b7
  bash tools/run_yolov11_241.sh c5
  bash tools/run_yolov11_241.sh c7
  bash tools/run_yolov11_241.sh d5
  bash tools/run_yolov11_241.sh d7
  bash tools/run_yolov11_241.sh d3
  bash tools/run_yolov11_241.sh d1
  bash tools/run_yolov11_241.sh b1 b2
  bash tools/run_yolov11_241.sh a3 b3 d3
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
ENABLE_A5="false"
ENABLE_A7="false"
ENABLE_B1="false"
ENABLE_B2="false"
ENABLE_B3="false"
ENABLE_B5="false"
ENABLE_B7="false"
ENABLE_C5="false"
ENABLE_C7="false"
ENABLE_D3="false"
ENABLE_D5="false"
ENABLE_D7="false"
ENABLE_D1_ALIAS="false"
UNKNOWN=()

for sw in "${SWITCHES[@]}"; do
  key="$(echo "${sw}" | tr '[:upper:]' '[:lower:]')"
  case "${key}" in
    "" ) ;;
    a3 ) ENABLE_A3="true" ;;
    a5 ) ENABLE_A5="true" ;;
    a7 ) ENABLE_A7="true" ;;
    b1 ) ENABLE_B1="true" ;;
    b2 ) ENABLE_B2="true" ;;
    b3 ) ENABLE_B3="true" ;;
    b5 ) ENABLE_B5="true" ;;
    b7 ) ENABLE_B7="true" ;;
    c5 ) ENABLE_C5="true" ;;
    c7 ) ENABLE_C7="true" ;;
    d5 ) ENABLE_D5="true" ;;
    d7 ) ENABLE_D7="true" ;;
    d3 ) ENABLE_D3="true" ;;
    d1 ) ENABLE_D3="true"; ENABLE_D1_ALIAS="true" ;;
    * ) UNKNOWN+=("${sw}") ;;
  esac
done

if [[ ${#UNKNOWN[@]} -gt 0 ]]; then
  echo "[error] Unsupported switches (a3/a5/a7/b1/b2/b3/b5/b7/c5/c7/d5/d7/d3 plus legacy d1 alias): ${UNKNOWN[*]}" >&2
  exit 4
fi

CONFIG_TO_RUN="${BASE_CONFIG}"

if [[ "${ENABLE_A3}" == "true" || "${ENABLE_A5}" == "true" || "${ENABLE_A7}" == "true" || "${ENABLE_B1}" == "true" || "${ENABLE_B2}" == "true" || "${ENABLE_B3}" == "true" || "${ENABLE_B5}" == "true" || "${ENABLE_B7}" == "true" || "${ENABLE_C5}" == "true" || "${ENABLE_C7}" == "true" || "${ENABLE_D5}" == "true" || "${ENABLE_D7}" == "true" || "${ENABLE_D3}" == "true" ]]; then
  ENABLED_KEYS=()
  [[ "${ENABLE_A3}" == "true" ]] && ENABLED_KEYS+=("a3")
  [[ "${ENABLE_A5}" == "true" ]] && ENABLED_KEYS+=("a5")
  [[ "${ENABLE_A7}" == "true" ]] && ENABLED_KEYS+=("a7")
  [[ "${ENABLE_B1}" == "true" ]] && ENABLED_KEYS+=("b1")
  [[ "${ENABLE_B2}" == "true" ]] && ENABLED_KEYS+=("b2")
  [[ "${ENABLE_B3}" == "true" ]] && ENABLED_KEYS+=("b3")
  [[ "${ENABLE_B5}" == "true" ]] && ENABLED_KEYS+=("b5")
  [[ "${ENABLE_B7}" == "true" ]] && ENABLED_KEYS+=("b7")
  [[ "${ENABLE_C5}" == "true" ]] && ENABLED_KEYS+=("c5")
  [[ "${ENABLE_C7}" == "true" ]] && ENABLED_KEYS+=("c7")
  [[ "${ENABLE_D5}" == "true" ]] && ENABLED_KEYS+=("d5")
  [[ "${ENABLE_D7}" == "true" ]] && ENABLED_KEYS+=("d7")
  if [[ "${ENABLE_D1_ALIAS}" == "true" ]]; then
    ENABLED_KEYS+=("d1")
  elif [[ "${ENABLE_D3}" == "true" ]]; then
    ENABLED_KEYS+=("d3")
  fi

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
      a5 )
        DERIVED_CONFIG="${ROOT_DIR}/configs/yolo11/enhance241/defect241a5.yaml"
        DERIVED_SUFFIX="__a5"
        ;;
      a7 )
        DERIVED_CONFIG="${ROOT_DIR}/configs/yolo11/enhance241/defect241a7.yaml"
        DERIVED_SUFFIX="__a7"
        ;;
      b3 )
        DERIVED_CONFIG="${ROOT_DIR}/configs/yolo11/enhance241/defect241b3.yaml"
        DERIVED_SUFFIX="__b3"
        ;;
      b5 )
        DERIVED_CONFIG="${ROOT_DIR}/configs/yolo11/enhance241/defect241b5.yaml"
        DERIVED_SUFFIX="__b5"
        ;;
      b7 )
        DERIVED_CONFIG="${ROOT_DIR}/configs/yolo11/enhance241/defect241b7.yaml"
        DERIVED_SUFFIX="__b7"
        ;;
      c5 )
        DERIVED_CONFIG="${ROOT_DIR}/configs/yolo11/enhance241/defect241c5.yaml"
        DERIVED_SUFFIX="__c5"
        ;;
      c7 )
        DERIVED_CONFIG="${ROOT_DIR}/configs/yolo11/enhance241/defect241c7.yaml"
        DERIVED_SUFFIX="__c7"
        ;;
      d5 )
        DERIVED_CONFIG="${ROOT_DIR}/configs/yolo11/enhance241/defect241d5.yaml"
        DERIVED_SUFFIX="__d5"
        ;;
      d7 )
        DERIVED_CONFIG="${ROOT_DIR}/configs/yolo11/enhance241/defect241d7.yaml"
        DERIVED_SUFFIX="__d7"
        ;;
      d1 )
        DERIVED_CONFIG="${ROOT_DIR}/configs/yolo11/enhance241/defect241d1.yaml"
        DERIVED_SUFFIX="__d1"
        ;;
      d3 )
        DERIVED_CONFIG="${ROOT_DIR}/configs/yolo11/enhance241/defect241d3.yaml"
        DERIVED_SUFFIX="__d3"
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
enh["a5"] = False
enh["a7"] = False
enh["b1"] = False
enh["b2"] = False
enh["b3"] = False
enh["b5"] = False
enh["b7"] = False
enh["c5"] = False
enh["c7"] = False
enh["d1"] = False
enh["d3"] = False
enh["d5"] = False
enh["d7"] = False
for k in keys:
    if k == "d1":
        enh["d1"] = True
        enh["d3"] = True
    elif k == "d3":
        enh["d3"] = True
    else:
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

# Safety guard for enhance runs on limited VRAM:
# baseline (all enhance241 flags false) stays untouched.
export _E241_RUN_CFG="${CONFIG_TO_RUN}"
export _E241_ENABLE_D3="${ENABLE_D3}"
export _E241_ENABLE_D5="${ENABLE_D5}"
export _E241_ENABLE_D7="${ENABLE_D7}"
export _E241_SAFE_BATCH="${E241_SAFE_BATCH}"
export _E241_SAFE_WORKERS="${E241_SAFE_WORKERS}"
"${PYTHON_BIN}" - <<'PY'
import os
from pathlib import Path

import yaml

cfg_path = Path(os.environ["_E241_RUN_CFG"]).resolve()
enable_d3_arg = str(os.environ.get("_E241_ENABLE_D3", "false")).lower() == "true"
enable_d5_arg = str(os.environ.get("_E241_ENABLE_D5", "false")).lower() == "true"
enable_d7_arg = str(os.environ.get("_E241_ENABLE_D7", "false")).lower() == "true"
safe_batch = int(os.environ.get("_E241_SAFE_BATCH", "6"))
safe_workers = int(os.environ.get("_E241_SAFE_WORKERS", "4"))

with cfg_path.open("r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}
if not isinstance(cfg, dict):
    raise SystemExit(f"Invalid yaml mapping: {cfg_path}")

enh = cfg.get("enhance241", {}) or {}
if not isinstance(enh, dict):
    enh = {}

enhance_enabled = any(
    bool(enh.get(k, False))
    for k in ("a3", "a5", "a7", "b1", "b2", "b3", "b5", "b7", "c5", "c7", "d1", "d3", "d5", "d7")
)
if not enhance_enabled:
    raise SystemExit(0)

enable_d3 = enable_d3_arg or bool(enh.get("d3", False)) or bool(enh.get("d1", False))
enable_d5 = enable_d5_arg or bool(enh.get("d5", False))
enable_d7 = enable_d7_arg or bool(enh.get("d7", False))
changed = False

try:
    batch = int(cfg.get("batch", safe_batch))
except Exception:
    batch = safe_batch
if batch >= 8:
    cfg["batch"] = int(safe_batch)
    changed = True

if enable_d3 or enable_d5 or enable_d7:
    try:
        workers = int(cfg.get("workers", safe_workers))
    except Exception:
        workers = safe_workers
    if workers > safe_workers:
        cfg["workers"] = int(safe_workers)
        changed = True

if changed:
    with cfg_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
PY

echo "[run] config=${CONFIG_TO_RUN} switches=${SWITCHES[*]:-(none)} python=${PYTHON_BIN}"
"${PYTHON_BIN}" "${ROOT_DIR}/src/train.py" --config "${CONFIG_TO_RUN}"
