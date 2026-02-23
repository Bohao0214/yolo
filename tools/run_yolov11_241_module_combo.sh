#!/usr/bin/env bash
set -u -o pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_CONFIG_DEFAULT="${ROOT_DIR}/configs/yolo11/enhance241/defect241.yaml"

BASE_CONFIG="${BASE_CONFIG:-${BASE_CONFIG_DEFAULT}}"
MATRIX_TAG="${MATRIX_TAG:-module_combo_$(date +%y%m%d%H%M%S)}"
EPOCHS_OVERRIDE="${EPOCHS_OVERRIDE:-10}"
FORCE_MODE="${FORCE_MODE:-train_test}"
SEED_OVERRIDE="${SEED_OVERRIDE:-}"
BATCH_OVERRIDE="${BATCH_OVERRIDE:-6}"
D1_WORKERS_OVERRIDE="${D1_WORKERS_OVERRIDE:-4}"
VRAM_GUARD_OVERRIDE="${VRAM_GUARD_OVERRIDE:-auto}" # auto|on|off
GUARD_MAX_GB_OVERRIDE="${GUARD_MAX_GB_OVERRIDE:-10}"
SAFE_BATCH_OVERRIDE="${SAFE_BATCH_OVERRIDE:-${E241_SAFE_BATCH:-6}}"
SAFE_WORKERS_OVERRIDE="${SAFE_WORKERS_OVERRIDE:-${E241_SAFE_WORKERS:-4}}"
TMP_CFG_DIR="${TMP_CFG_DIR:-}"
LOG_ROOT="${LOG_ROOT:-}"
PYTHON_CFG_BIN="${PYTHON_CFG_BIN:-${PYTHON_BIN:-python}}"
DRY_RUN="false"
COMBOS_RAW="${COMBOS_RAW:-}"

usage() {
  cat <<'USAGE'
Usage:
  bash tools/run_yolov11_241_module_combo.sh [base_config.yaml] [--epochs N] [--batch N] [--combos LIST] [--tag NAME] [--vram-guard auto|on|off] [--guard-max-gb N] [--safe-batch N] [--safe-workers N] [--dry-run]

Runs S2 quick A/B matrix with identical seed/data/epochs:
  baseline, a3, b3, d3, a3+b3, a3+d3, b3+d3, a3+b3+d3
  plus added singles: a5, a7, a9, b5, b7, b9, c5, c7, c9, d5, d7, d9
  (d1 is legacy alias of d3)

Behavior:
  - Generates per-case temp configs under TMP_CFG_DIR
  - Uses original experiment layout: experiments/<yolo_version>/<exp_name>/exp_*
  - If one case fails, continues other cases
  - Writes a visible error.md into that failed case's experiment path

Options:
  --epoch N    Override epochs for all cases (same as --epochs)
  --epochs N   Override epochs for all cases (default from EPOCHS_OVERRIDE, default 10)
  --batch N    Override batch size for all cases (same as --batch-size; default from BATCH_OVERRIDE, default 6)
  --batch-size N
               Override batch size for all cases
  --combos L   Comma-separated cases, e.g.:
               baseline,a3,b3,d3,a3+b3,a3+d3,b3+d3,a3+b3+d3,a5,a7,a9,b5,b7,b9,c5,c7,c9,d5,d7,d9
               Built-in aliases:
               hmc7/abcd7 => a7+b7+c7+d7
               pdd9/abcd9 => a9+b9+c9+d9
               Supports aliases a3_b3 / a3_b3_d1 and raw '+' expressions
  --tag NAME   Matrix tag (used for temp config/log folders)
  --vram-guard MODE
               Pass-through to run_yolov11_241.sh: auto|on|off (default auto)
  --guard-max-gb N
               Pass-through to run_yolov11_241.sh auto mode threshold (default 10)
  --safe-batch N
               Pass-through safe fallback batch when guard applies (default 6)
  --safe-workers N
               Pass-through worker cap for non-low-VRAM path (default 4)
  --dry-run    Generate configs and print commands only (no training)
  -h, --help   Show help

Env overrides:
  BASE_CONFIG, EPOCHS_OVERRIDE, COMBOS_RAW, FORCE_MODE, SEED_OVERRIDE, BATCH_OVERRIDE(default 6)
  D1_WORKERS_OVERRIDE(default 4), VRAM_GUARD_OVERRIDE(auto|on|off), GUARD_MAX_GB_OVERRIDE(default 10)
  SAFE_BATCH_OVERRIDE(default 6), SAFE_WORKERS_OVERRIDE(default 4)
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
    --batch|--batch-size)
      [[ $# -ge 2 ]] || { echo "[error] --batch requires a value" >&2; exit 2; }
      BATCH_OVERRIDE="$2"
      shift 2
      ;;
    --tag)
      [[ $# -ge 2 ]] || { echo "[error] --tag requires a value" >&2; exit 2; }
      MATRIX_TAG="$2"
      shift 2
      ;;
    --vram-guard)
      [[ $# -ge 2 ]] || { echo "[error] --vram-guard requires a value (auto|on|off)" >&2; exit 2; }
      VRAM_GUARD_OVERRIDE="$(echo "$2" | tr '[:upper:]' '[:lower:]')"
      shift 2
      ;;
    --guard-max-gb)
      [[ $# -ge 2 ]] || { echo "[error] --guard-max-gb requires a value" >&2; exit 2; }
      GUARD_MAX_GB_OVERRIDE="$2"
      shift 2
      ;;
    --safe-batch)
      [[ $# -ge 2 ]] || { echo "[error] --safe-batch requires a value" >&2; exit 2; }
      SAFE_BATCH_OVERRIDE="$2"
      shift 2
      ;;
    --safe-workers)
      [[ $# -ge 2 ]] || { echo "[error] --safe-workers requires a value" >&2; exit 2; }
      SAFE_WORKERS_OVERRIDE="$2"
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

if [[ "${VRAM_GUARD_OVERRIDE}" != "auto" && "${VRAM_GUARD_OVERRIDE}" != "on" && "${VRAM_GUARD_OVERRIDE}" != "off" ]]; then
  echo "[error] invalid --vram-guard='${VRAM_GUARD_OVERRIDE}', expected auto|on|off" >&2
  exit 2
fi

if [[ -z "${TMP_CFG_DIR}" ]]; then
  TMP_CFG_DIR="/tmp/yolo241_module_combo/${MATRIX_TAG}"
fi
if [[ -z "${LOG_ROOT}" ]]; then
  LOG_ROOT="${ROOT_DIR}/experiments/yolo11/module_combo_logs/${MATRIX_TAG}"
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
DEFAULT_COMBOS="baseline,hmc7,pdd9,a9,b9,c9,d9,a5,a7,b5,b7,c5,c7,d5,d7,a3,b3,d3,a3+b3,a3+d3,b3+d3,a3+b3+d3,b5+c5,c5+d5,b5+d5,b5+c5+d5"
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
    hmc7|abcd7|a7_b7_c7_d7|a7+b7+c7+d7)
      add_case "a7_b7_c7_d7" "a7 b7 c7 d7"
      return
      ;;
    pdd9|abcd9|a9_b9_c9_d9|a9+b9+c9+d9)
      add_case "a9_b9_c9_d9" "a9 b9 c9 d9"
      return
      ;;
    a3|a5|a7|a9|b1|b2|b3|b5|b7|b9|c5|c7|c9|d1|d3|d5|d7|d9)
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
      a3|a5|a7|a9|b1|b2|b3|b5|b7|b9|c5|c7|c9|d1|d3|d5|d7|d9) ;;
      *)
        echo "[error] Unsupported combo token '${part}' in '${raw}' (allowed: baseline,hmc7/pdd9,a3,a5,a7,a9,b1,b2,b3,b5,b7,b9,c5,c7,c9,d1,d3,d5,d7,d9)" >&2
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

for key in ("a3", "a5", "a7", "a9", "b1", "b2", "b3", "b5", "b7", "b9", "c5", "c7", "c9", "d1", "d3", "d5", "d7", "d9"):
    enh[key] = False
for key in switches:
    if key not in {"a3", "a5", "a7", "a9", "b1", "b2", "b3", "b5", "b7", "b9", "c5", "c7", "c9", "d1", "d3", "d5", "d7", "d9"}:
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
if ("d1" in switches or "d3" in switches or "d5" in switches or "d7" in switches or "d9" in switches) and d1_workers_override:
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

echo "[module-combo] base_config=${BASE_CONFIG}"
echo "[module-combo] epochs=${EPOCHS_OVERRIDE} mode=${FORCE_MODE} seed=${SEED_OVERRIDE:-<keep>} batch=${BATCH_OVERRIDE:-<keep>}"
echo "[module-combo] guard mode=${VRAM_GUARD_OVERRIDE} max_gb=${GUARD_MAX_GB_OVERRIDE} safe_batch=${SAFE_BATCH_OVERRIDE} safe_workers=${SAFE_WORKERS_OVERRIDE}"
echo "[module-combo] combos_raw=${COMBOS_RAW}"
echo "[module-combo] combos_resolved=${CASE_TAGS[*]}"
echo "[module-combo] tmp_cfg_dir=${TMP_CFG_DIR}"
echo "[module-combo] log_root=${LOG_ROOT}"

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

  cmd=(
    bash "${ROOT_DIR}/tools/run_yolov11_241.sh"
    --vram-guard "${VRAM_GUARD_OVERRIDE}"
    --guard-max-gb "${GUARD_MAX_GB_OVERRIDE}"
    --safe-batch "${SAFE_BATCH_OVERRIDE}"
    --safe-workers "${SAFE_WORKERS_OVERRIDE}"
    "${cfg_path}"
  )
  echo "[module-combo:${tag}] config=${cfg_path} switches=${switches:-baseline}"
  echo "[module-combo:${tag}] cmd=${cmd[*]}"

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
    echo "[module-combo:${tag}] failed -> ${error_md}"
  else
    echo "[module-combo:${tag}] success exp_dir=${exp_dir}"
  fi

  printf "%s\t%s\t%s\t%s\t%s\t%s\n" \
    "${tag}" "${switches:-baseline}" "${status}" "${exp_dir}" "${cfg_path}" "${log_path}" \
    >> "${SUMMARY_TSV}"

  rm -f "${before_tmp}" "${after_tmp}"
done

echo
echo "[module-combo] finished. fail_count=${fail_count}"
echo "[module-combo] summary=${SUMMARY_TSV}"
column -s $'\t' -t "${SUMMARY_TSV}" || cat "${SUMMARY_TSV}"

exit 0

: <<'EXAMPLES'
#
# 0) 最小启动：默认组合（含 baseline/hmc7/pdd9 等），10 epoch，batch=6
# 场景：先确认脚本可跑通。
bash tools/run_yolov11_241_module_combo.sh

# 1) 只改训练轮数，不改其它（默认仍会走 guard=auto）
# 场景：快速缩短或拉长实验。
bash tools/run_yolov11_241_module_combo.sh \
  --epochs 12

# 2) 显式指定 batch
# 场景：你想统一所有组合的 batch。
bash tools/run_yolov11_241_module_combo.sh \
  --epochs 12 \
  --batch 8

# 3) 只跑指定组合（用 + 表示同一次实验里同时开多个模块）
# 场景：小规模对比。
bash tools/run_yolov11_241_module_combo.sh \
  --epochs 8 \
  --batch 6 \
  --combos baseline,a3+b3,d1

# 4) 组合同样支持下划线别名（与上条等价）
bash tools/run_yolov11_241_module_combo.sh \
  --epochs 8 \
  --combos baseline,a3_b3,a3_b3_d1

# 5) 指定基础配置（例如用 defect.yaml 只跑纯 baseline）
# 场景：做“增强框架 vs 原始流程”的公平对比。
bash tools/run_yolov11_241_module_combo.sh \
  configs/yolo11/defect.yaml \
  --epochs 100 \
  --batch 10 \
  --combos baseline

# 6) 跑新模块单开全集（A/B/C/D 各模块单独验证）
bash tools/run_yolov11_241_module_combo.sh \
  --epochs 10 \
  --batch 6 \
  --combos a5,a7,a9,b5,b7,b9,c5,c7,c9,d5,d7,d9

# 7) 跑论文分组组合
# hmc7 = a7+b7+c7+d7
# pdd9 = a9+b9+c9+d9
bash tools/run_yolov11_241_module_combo.sh \
  --epochs 10 \
  --batch 6 \
  --combos baseline,hmc7,pdd9

# 8) 干跑检查（不训练，只生成配置和最终执行命令）
# 场景：先确认解析结果、路径、开关是否正确。
bash tools/run_yolov11_241_module_combo.sh \
  --epochs 10 \
  --batch 6 \
  --combos baseline,b5+c5 \
  --dry-run

# 9) 8GB 机器推荐：自动显存保护（auto）
# 规则：检测到显存 <= guard-max-gb 时，回落到 safe-batch；
#       safe-workers 主要用于非低显存路径（防 CPU/IO 瓶颈）。
bash tools/run_yolov11_241_module_combo.sh \
  --epochs 150 \
  --batch 12 \
  --vram-guard auto \
  --guard-max-gb 10 \
  --safe-batch 6 \
  --safe-workers 4 \
  --combos a5,a7,a9,b5,b7,b9,c5,c7,c9,d5,d7,d9

# 10) 24GB 机器推荐：关闭显存保护（off）
# 规则：严格尊重 --batch/配置里的 workers，不做自动降级。
bash tools/run_yolov11_241_module_combo.sh \
  --epochs 150 \
  --batch 12 \
  --vram-guard off \
  --combos a5,a7,a9,b5,b7,b9,c5,c7,c9,d5,d7,d9

# 11) 强制保护（on）
# 场景：即使是大显存机，也希望固定使用安全参数复现实验。
bash tools/run_yolov11_241_module_combo.sh \
  --epochs 50 \
  --batch 16 \
  --vram-guard on \
  --safe-batch 6 \
  --safe-workers 4 \
  --combos baseline,hmc7,pdd9

# 12) 自定义 tag（方便区分日志目录和 summary）
bash tools/run_yolov11_241_module_combo.sh \
  --tag ablation_$(date +%m%d_%H%M) \
  --epochs 20 \
  --batch 6 \
  --combos baseline,a3,b3,d3,a3+b3+d3

# 13) 环境变量等价写法（适合脚本/CI）
VRAM_GUARD_OVERRIDE=off \
BATCH_OVERRIDE=12 \
EPOCHS_OVERRIDE=120 \
COMBOS_RAW=baseline,hmc7,pdd9 \
bash tools/run_yolov11_241_module_combo.sh
EXAMPLES
