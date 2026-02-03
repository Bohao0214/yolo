#!/usr/bin/env bash
set -u
set -o pipefail

export LANG=C.UTF-8
export LC_ALL=C.UTF-8

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_CONFIG="${BASE_CONFIG:-${ROOT_DIR}/configs/yolo11/defect_231.yaml}"
AB_DIR="${ROOT_DIR}/configs/yolo11/ablation"
EXP_ROOT="${ROOT_DIR}/experiments/yolo11/defect"
RUN_LOG="${EXP_ROOT}/ablation_runs.log"
FAIL_LOG="${EXP_ROOT}/ablation_failures.log"
HEAD_TYPE="${HEAD_TYPE:-dyhead}"
PYTHON_BIN="${PYTHON_BIN:-python}"
TARGETS="${TARGETS:-}"

if [[ $# -gt 0 ]]; then
  if [[ "$1" == *.yml || "$1" == *.yaml ]]; then
    BASE_CONFIG="$1"
    shift
  fi
  if [[ -z "${TARGETS}" && $# -gt 0 ]]; then
    TARGETS="$*"
  fi
fi

mkdir -p "${AB_DIR}" "${EXP_ROOT}"

export BASE_CONFIG AB_DIR HEAD_TYPE

${PYTHON_BIN} - <<'PY'
import copy
import os
import yaml

base_path = os.environ["BASE_CONFIG"]
ab_dir = os.environ["AB_DIR"]
head_type = os.environ.get("HEAD_TYPE", "dyhead").lower()

with open(base_path, "r", encoding="utf-8") as f:
    base_cfg = yaml.safe_load(f) or {}

def write_cfg(name, updates):
    cfg = copy.deepcopy(base_cfg)
    enhance = cfg.setdefault("enhance", {})
    enhance.update({
        "enable_p2": False,
        "p2_source": "auto",
        "p2_backbone_select": "first",
        "p2_pan_fuse": True,
        "enable_bifpn": False,
        "bifpn_depth": 1,
        "bifpn_channels": 0,
        "enable_carafe": False,
        "carafe_replace_neck": False,
        "carafe_kernel": 5,
        "carafe_up_kernel": 3,
        "carafe_comp_channels": 64,
        "head_type": "base",
        "head_blocks": 1,
        "enable_dcn": False,
        "dcn_scales": [1],
        "dcn_kernel": 3,
    })
    enhance.update(updates)
    out_path = os.path.join(ab_dir, f"defect_{name}.yaml")
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
    print(out_path)

configs = []
configs.append(write_cfg("a0", {}))
configs.append(write_cfg("a1", {"enable_p2": True}))
configs.append(write_cfg("a2", {"enable_bifpn": True}))
configs.append(write_cfg("a3", {"enable_carafe": True, "carafe_replace_neck": True}))
configs.append(write_cfg("a4", {"head_type": head_type}))
configs.append(write_cfg("a5", {"enable_dcn": True, "dcn_scales": [1]}))
PY

if [[ -n "${TARGETS}" ]]; then
  CONFIGS=()
  for t in ${TARGETS}; do
    key="${t,,}"
    if [[ "${key}" =~ ^[0-5]$ ]]; then
      key="a${key}"
    fi
    if [[ ! "${key}" =~ ^a[0-5]$ ]]; then
      echo "[warn] skip invalid target: ${t}"
      continue
    fi
    cfg="${AB_DIR}/defect_${key}.yaml"
    if [[ -f "${cfg}" ]]; then
      CONFIGS+=("${cfg}")
    else
      echo "[warn] missing config: ${cfg}"
    fi
  done
else
  mapfile -t CONFIGS < <(ls -1 "${AB_DIR}"/defect_a*.yaml | sort)
fi

run_one() {
  local cfg="$1"
  local exp_name
  exp_name="$(basename "${cfg}")"

  mapfile -t before_list < <(ls -1d "${EXP_ROOT}"/exp_* 2>/dev/null | sort || true)

  echo "[run] ${exp_name} config=${cfg}"
  "${PYTHON_BIN}" "${ROOT_DIR}/src/train.py" --config "${cfg}"
  local code=$?

  mapfile -t after_list < <(ls -1d "${EXP_ROOT}"/exp_* 2>/dev/null | sort || true)
  local new_exp=""
  for item in "${after_list[@]}"; do
    local found="false"
    for b in "${before_list[@]}"; do
      if [[ "${item}" == "${b}" ]]; then
        found="true"
        break
      fi
    done
    if [[ "${found}" == "false" ]]; then
      new_exp="${item}"
    fi
  done
  if [[ -z "${new_exp}" ]]; then
    new_exp="$(ls -1dt "${EXP_ROOT}"/exp_* 2>/dev/null | head -n 1)"
  fi

  local stamp
  stamp="$(date "+%F %T")"
  if [[ ${code} -eq 0 ]]; then
    echo "[${stamp}] OK ${exp_name} exp=${new_exp} cfg=${cfg}" | tee -a "${RUN_LOG}"
  else
    echo "[${stamp}] FAIL ${exp_name} exp=${new_exp} cfg=${cfg} code=${code}" | tee -a "${RUN_LOG}" >> "${FAIL_LOG}"
  fi
  return ${code}
}

for cfg in "${CONFIGS[@]}"; do
  run_one "${cfg}" || true
done

# bash tools/run_yolov11_ablate.sh 1 2 3 4 5 
#用公用配置覆盖a1-a5，只有模块区别/configs/yolo11/defect_231.yaml

# 单独训练某个配置，参数个性化，如轮次batchsize、iou等 configs/yolo11/ablation/defect_a1.yaml
# python src/train.py --config configs/yolo11/ablation/defect_a1.yaml

# conda activate yolo11
# for e in e3 e4 e5; do
#   python src/train.py --config configs/yolo11/ablation/defect_${e}.yaml || true
# done
# for a in a1; do   
#   CUDA_VISIBLE_DEVICES=0 
#   python src/train.py --config configs/yolo11/ablation/defect_${a}.yaml || true; 
# done
