#!/usr/bin/env bash
set -u -o pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_CONFIG_DEFAULT="${ROOT_DIR}/configs/yolo11/enhance241/defect241.yaml"

BASE_CONFIG="${BASE_CONFIG:-${BASE_CONFIG_DEFAULT}}"
MATRIX_TAG="${MATRIX_TAG:-matrix_$(date +%y%m%d%H%M%S)}"
EPOCHS_OVERRIDE="${EPOCHS_OVERRIDE:-10}"
FORCE_MODE="${FORCE_MODE:-train_test}"
SEED_OVERRIDE="${SEED_OVERRIDE:-}"
BATCH_OVERRIDE="${BATCH_OVERRIDE:-6}"
D1_WORKERS_OVERRIDE="${D1_WORKERS_OVERRIDE:-4}"
TMP_CFG_DIR="${TMP_CFG_DIR:-}"
LOG_ROOT="${LOG_ROOT:-}"
PYTHON_CFG_BIN="${PYTHON_CFG_BIN:-${PYTHON_BIN:-python}}"
DRY_RUN="false"
COMBOS_RAW="${COMBOS_RAW:-}"

usage() {
  cat <<'USAGE'
Usage:
  bash tools/run_yolov11_241_matrix.sh [base_config.yaml] [--epochs N] [--combos LIST] [--tag NAME] [--dry-run]

Runs S2 quick A/B matrix with identical seed/data/epochs:
  baseline, a3, b3, d3, a3+b3, a3+d3, b3+d3, a3+b3+d3
  plus newly added singles: a5, b5, c5, d5
  (d1 is legacy alias of d3)

Behavior:
  - Generates per-case temp configs under TMP_CFG_DIR
  - Uses original experiment layout: experiments/<yolo_version>/<exp_name>/exp_*
  - If one case fails, continues other cases
  - Writes a visible error.md into that failed case's experiment path

Options:
  --epoch N    Override epochs for all cases (same as --epochs)
  --epochs N   Override epochs for all cases (default from EPOCHS_OVERRIDE, default 10)
  --combos L   Comma-separated cases, e.g.:
               baseline,a3,b3,d3,a3+b3,a3+d3,b3+d3,a3+b3+d3,a5,b5,c5,d5
               Supports aliases a3_b3 / a3_b3_d1 and raw '+' expressions
  --tag NAME   Matrix tag (used for temp config/log folders)
  --dry-run    Generate configs and print commands only (no training)
  -h, --help   Show help

Env overrides:
  BASE_CONFIG, EPOCHS_OVERRIDE, COMBOS_RAW, FORCE_MODE, SEED_OVERRIDE, BATCH_OVERRIDE(default 6)
  D1_WORKERS_OVERRIDE(default 4)
  TMP_CFG_DIR, LOG_ROOT, PYTHON_CFG_BIN, PYTHON_BIN
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --epoch|--epochs)
      [[ $# -ge 2 ]] || { echo "[error] --epochs requires a value" >&2; exit 2; }
      EPOCHS_OVERRIDE="$2"
      shift 2
      ;;
    --combos)
      [[ $# -ge 2 ]] || { echo "[error] --combos requires a value" >&2; exit 2; }
      COMBOS_RAW="$2"
      shift 2
      ;;
    --tag)
      [[ $# -ge 2 ]] || { echo "[error] --tag requires a value" >&2; exit 2; }
      MATRIX_TAG="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN="true"
      shift
      ;;
    *.yml|*.yaml)
      BASE_CONFIG="$1"
      shift
      ;;
    *)
      echo "[error] Unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ ! -f "${BASE_CONFIG}" ]]; then
  echo "[error] base config not found: ${BASE_CONFIG}" >&2
  exit 2
fi

if [[ -z "${TMP_CFG_DIR}" ]]; then
  TMP_CFG_DIR="/tmp/yolo241_matrix/${MATRIX_TAG}"
fi
if [[ -z "${LOG_ROOT}" ]]; then
  LOG_ROOT="${ROOT_DIR}/experiments/yolo11/matrix_logs/${MATRIX_TAG}"
fi

if ! "${PYTHON_CFG_BIN}" - <<'PY' >/dev/null 2>&1
import yaml  # noqa: F401
PY
then
  if [[ -x "/home/ubuntu/anaconda3/envs/yolo11/bin/python" ]]; then
    PYTHON_CFG_BIN="/home/ubuntu/anaconda3/envs/yolo11/bin/python"
  fi
fi

if ! "${PYTHON_CFG_BIN}" - <<'PY' >/dev/null 2>&1
import yaml  # noqa: F401
PY
then
  echo "[error] python for config generation has no PyYAML: ${PYTHON_CFG_BIN}" >&2
  exit 3
fi

mkdir -p "${TMP_CFG_DIR}" "${LOG_ROOT}"

# 默认组合：原有 S2 主线 + 新增模块单开
DEFAULT_COMBOS="baseline,a3,b3,d3,a3+b3,a3+d3,b3+d3,a3+b3+d3,a5,b5,c5,d5"
if [[ -z "${COMBOS_RAW}" ]]; then
  COMBOS_RAW="${DEFAULT_COMBOS}"
fi

CASE_TAGS=()
CASE_SWITCHES=()
declare -A _CASE_TAG_SEEN=()

add_case() {
  local tag="$1"
  local switches="$2"
  if [[ -z "${tag}" ]]; then
    return
  fi
  if [[ -n "${_CASE_TAG_SEEN[$tag]+x}" ]]; then
    return
  fi
  _CASE_TAG_SEEN["$tag"]=1
  CASE_TAGS+=("$tag")
  CASE_SWITCHES+=("$switches")
}

normalize_and_add_case() {
  local raw="$1"
  local token
  token="$(echo "${raw}" | tr '[:upper:]' '[:lower:]')"
  token="${token//[[:space:]]/}"
  if [[ -z "${token}" ]]; then
    return
  fi

  case "${token}" in
    baseline|base|none)
      add_case "baseline" ""
      return
      ;;
    a3|a5|b1|b2|b3|b5|c5|d1|d3|d5)
      add_case "${token}" "${token}"
      return
      ;;
    a3_b3|a3+b3)
      add_case "a3_b3" "a3 b3"
      return
      ;;
    a3_d1|a3+d1)
      add_case "a3_d1" "a3 d1"
      return
      ;;
    b3_d1|b3+d1)
      add_case "b3_d1" "b3 d1"
      return
      ;;
    a3_b3_d1|a3+b3+d1)
      add_case "a3_b3_d1" "a3 b3 d1"
      return
      ;;
  esac

  # Generic parser: supports raw switch expression like a3+b3+d1 or a3_b3_d1.
  local expr="${token//_/+}"
  local part
  local -a parts=()
  local -a switches=()
  declare -A seen=()
  IFS='+' read -r -a parts <<< "${expr}"
  for part in "${parts[@]}"; do
    [[ -z "${part}" ]] && continue
    case "${part}" in
      baseline|base|none) continue ;;
      a3|a5|b1|b2|b3|b5|c5|d1|d3|d5) ;;
      *)
        echo "[error] Unsupported combo token '${part}' in '${raw}' (allowed: baseline,a3,a5,b1,b2,b3,b5,c5,d1,d3,d5)" >&2
        exit 2
        ;;
    esac
    if [[ -z "${seen[$part]+x}" ]]; then
      seen["$part"]=1
      switches+=("${part}")
    fi
  done

  if [[ ${#switches[@]} -eq 0 ]]; then
    add_case "baseline" ""
    return
  fi

  local tag
  local switch_str
  tag="$(IFS=_; echo "${switches[*]}")"
  switch_str="$(IFS=' '; echo "${switches[*]}")"
  add_case "${tag}" "${switch_str}"
}

IFS=',' read -r -a _combo_tokens <<< "${COMBOS_RAW}"
for _token in "${_combo_tokens[@]}"; do
  normalize_and_add_case "${_token}"
done

if [[ ${#CASE_TAGS[@]} -eq 0 ]]; then
  echo "[error] No valid cases resolved from --combos='${COMBOS_RAW}'" >&2
  exit 2
fi

list_exp_dirs() {
  local exp_root="$1"
  if [[ -d "${exp_root}" ]]; then
    find "${exp_root}" -mindepth 1 -maxdepth 1 -type d -name 'exp_*' -printf '%f\n' | sort
  fi
}

generate_case_cfg() {
  local out_cfg="$1"
  local switches="$2"
  _M_BASE_CONFIG="${BASE_CONFIG}" \
  _M_OUT_CFG="${out_cfg}" \
  _M_SWITCHES="${switches}" \
  _M_EPOCHS="${EPOCHS_OVERRIDE}" \
  _M_FORCE_MODE="${FORCE_MODE}" \
  _M_SEED_OVERRIDE="${SEED_OVERRIDE}" \
  _M_BATCH_OVERRIDE="${BATCH_OVERRIDE}" \
  _M_D1_WORKERS_OVERRIDE="${D1_WORKERS_OVERRIDE}" \
  "${PYTHON_CFG_BIN}" - <<'PY'
import os
from pathlib import Path

import yaml

base_cfg = Path(os.environ["_M_BASE_CONFIG"]).resolve()
out_cfg = Path(os.environ["_M_OUT_CFG"]).resolve()
switches = [s.strip() for s in os.environ["_M_SWITCHES"].split() if s.strip()]
epochs = os.environ["_M_EPOCHS"].strip()
force_mode = os.environ["_M_FORCE_MODE"].strip()
seed_override = os.environ["_M_SEED_OVERRIDE"].strip()
batch_override = os.environ["_M_BATCH_OVERRIDE"].strip()
d1_workers_override = os.environ["_M_D1_WORKERS_OVERRIDE"].strip()

with base_cfg.open("r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}

if not isinstance(cfg, dict):
    raise SystemExit(f"Base config must be mapping: {base_cfg}")

enh = cfg.setdefault("enhance241", {})
if not isinstance(enh, dict):
    raise SystemExit("enhance241 must be a mapping in base config.")

for key in ("a3", "a5", "b1", "b2", "b3", "b5", "c5", "d1", "d3", "d5"):
    enh[key] = False
for key in switches:
    if key not in {"a3", "a5", "b1", "b2", "b3", "b5", "c5", "d1", "d3", "d5"}:
        raise SystemExit(f"Unsupported matrix switch: {key}")
    if key == "d1":
        enh["d1"] = True
        enh["d3"] = True
    elif key == "d3":
        enh["d3"] = True
    else:
        enh[key] = True

exp_name = str(cfg.get("exp_name", "defect241"))
if switches:
    suffix = "__" + "__".join(switches)
    if not exp_name.endswith(suffix):
        exp_name = exp_name + suffix
cfg["exp_name"] = exp_name

if force_mode:
    cfg["mode"] = force_mode
if epochs:
    cfg["epochs"] = int(epochs)
if seed_override:
    cfg["seed"] = int(seed_override)
if batch_override:
    cfg["batch"] = int(batch_override)
if ("d1" in switches or "d3" in switches or "d5" in switches) and d1_workers_override:
    cfg["workers"] = int(d1_workers_override)

out_cfg.parent.mkdir(parents=True, exist_ok=True)
with out_cfg.open("w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

yolo_version = str(cfg.get("yolo_version", "yolo11"))
print(f"{exp_name}\t{yolo_version}")
PY
}

SUMMARY_TSV="${LOG_ROOT}/summary.tsv"
{
  echo -e "tag\tswitches\tstatus\texp_dir\tconfig\tlog"
} > "${SUMMARY_TSV}"

echo "[matrix] base_config=${BASE_CONFIG}"
echo "[matrix] epochs=${EPOCHS_OVERRIDE} mode=${FORCE_MODE} seed=${SEED_OVERRIDE:-<keep>} batch=${BATCH_OVERRIDE:-<keep>}"
echo "[matrix] combos_raw=${COMBOS_RAW}"
echo "[matrix] combos_resolved=${CASE_TAGS[*]}"
echo "[matrix] tmp_cfg_dir=${TMP_CFG_DIR}"
echo "[matrix] log_root=${LOG_ROOT}"

fail_count=0

for i in "${!CASE_TAGS[@]}"; do
  tag="${CASE_TAGS[$i]}"
  switches="${CASE_SWITCHES[$i]}"
  cfg_path="${TMP_CFG_DIR}/${tag}.yaml"
  log_path="${LOG_ROOT}/${tag}.log"

  meta="$(generate_case_cfg "${cfg_path}" "${switches}")"
  exp_name="$(echo "${meta}" | awk -F $'\t' '{print $1}')"
  yolo_version="$(echo "${meta}" | awk -F $'\t' '{print $2}')"
  exp_root="${ROOT_DIR}/experiments/${yolo_version}/${exp_name}"

  before_tmp="$(mktemp)"
  after_tmp="$(mktemp)"
  list_exp_dirs "${exp_root}" > "${before_tmp}"

  cmd=(bash "${ROOT_DIR}/tools/run_yolov11_241.sh" "${cfg_path}")
  echo "[matrix:${tag}] config=${cfg_path} switches=${switches:-baseline}"
  echo "[matrix:${tag}] cmd=${cmd[*]}"

  status=0
  if [[ "${DRY_RUN}" == "true" ]]; then
    {
      echo "[dry-run] ${cmd[*]}"
    } > "${log_path}"
  else
    "${cmd[@]}" > "${log_path}" 2>&1
    status=$?
  fi

  exp_dir=""
  if [[ "${DRY_RUN}" == "true" ]]; then
    exp_dir="<dry-run>"
  else
    list_exp_dirs "${exp_root}" > "${after_tmp}"
    new_name="$(comm -13 "${before_tmp}" "${after_tmp}" | tail -n 1)"
    if [[ -n "${new_name}" ]]; then
      exp_dir="${exp_root}/${new_name}"
    elif [[ -d "${exp_root}" ]]; then
      exp_dir="$(ls -1dt "${exp_root}"/exp_* 2>/dev/null | head -n 1 || true)"
    fi
  fi

  if [[ ${status} -ne 0 ]]; then
    fail_count=$((fail_count + 1))
    if [[ -z "${exp_dir}" ]]; then
      exp_dir="${exp_root}/exp_error_${MATRIX_TAG}_${tag}"
      mkdir -p "${exp_dir}"
    fi
    error_md="${exp_dir}/error.md"
    {
      echo "# ERROR"
      echo
      echo "- tag: \`${tag}\`"
      echo "- switches: \`${switches:-baseline}\`"
      echo "- status: \`${status}\`"
      echo "- config: \`${cfg_path}\`"
      echo "- log: \`${log_path}\`"
      echo
      echo "## Log Tail"
      echo '```text'
      tail -n 200 "${log_path}" || true
      echo '```'
    } > "${error_md}"
    echo "[matrix:${tag}] failed -> ${error_md}"
  else
    echo "[matrix:${tag}] success exp_dir=${exp_dir}"
  fi

  printf "%s\t%s\t%s\t%s\t%s\t%s\n" \
    "${tag}" "${switches:-baseline}" "${status}" "${exp_dir}" "${cfg_path}" "${log_path}" \
    >> "${SUMMARY_TSV}"

  rm -f "${before_tmp}" "${after_tmp}"
done

echo
echo "[matrix] finished. fail_count=${fail_count}"
echo "[matrix] summary=${SUMMARY_TSV}"
column -s $'\t' -t "${SUMMARY_TSV}" || cat "${SUMMARY_TSV}"

exit 0
 

# 默认8组，10 epoch
# bash tools/run_yolov11_241_matrix.sh

# 指定epoch
# bash tools/run_yolov11_241_matrix.sh --epoch 12

# 只跑你选的组合
# bash tools/run_yolov11_241_matrix.sh --epochs 8 --combos baseline,a3+b3,d1

# 组合也支持下划线别名
# bash tools/run_yolov11_241_matrix.sh --epochs 8 --combos baseline,a3_b3,a3_b3_d1

# 换基础配置
# bash tools/run_yolov11_241_matrix.sh configs/yolo11/enhance241/defect241.yaml --epochs 6 --combos a3,b3,d1
