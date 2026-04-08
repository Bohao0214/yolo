#!/usr/bin/env bash
set -euo pipefail

: <<'DOC'
一次运行 4 个数据集，分别训练 batch=4/6/10，epochs=150。

默认行为：
1) 复用 tools/run_yolov11_241.sh（不改模型结构）
2) 自动为每个「数据集+batch」生成临时配置 yaml
3) 顺序执行 12 组训练，产物落到 experiments/
4) 默认跳过 post_eval 指标统计（只保留训练产物与权重），后续再统一评估

默认 4 个数据集根目录：
- /home/ubuntu/hpproject/yolo/experiments/gc10det_622_halves
- /home/ubuntu/hpproject/yolo/dataset/yolo/DeepPCB_standard
- /home/ubuntu/hpproject/yolo/dataset/yolo/kolektorsdd_622_halves
- /home/ubuntu/hpproject/yolo/dataset/yolo/neudet_622

可覆盖环境变量：
- ROOT=/home/ubuntu/hpproject/yolo
- TEMPLATE_CFG=$ROOT/configs/enhance/datasetm6c/defect241.yaml
- CFG_OUT_DIR=$ROOT/configs/enhance/multi4_train
- EPOCHS=150
- BATCH_LIST="4 6 10"
- DEVICE=0
- WORKERS=4
- SKIP_POST_EVAL=true
- RESUME_POLICY=auto

运行：
bash /home/ubuntu/hpproject/yolo/analyze/code/ch4tool/run_4datasets_bs4610_e150.sh
DOC

ROOT="${ROOT:-/home/ubuntu/hpproject/yolo}"
TEMPLATE_CFG="${TEMPLATE_CFG:-${ROOT}/configs/enhance/datasetm6c/defect241.yaml}"
CFG_OUT_DIR="${CFG_OUT_DIR:-${ROOT}/configs/enhance/multi4_train}"
EPOCHS="${EPOCHS:-150}"
BATCH_LIST="${BATCH_LIST:-4 6 10}"
DEVICE="${DEVICE:-0}"
WORKERS="${WORKERS:-4}"
SKIP_POST_EVAL="${SKIP_POST_EVAL:-true}"
RESUME_POLICY="${RESUME_POLICY:-auto}"

mkdir -p "${CFG_OUT_DIR}"

declare -a DATASET_ALIASES=(
  "gc10det_622_halves"
  "DeepPCB_standard"
  "kolektorsdd_622_halves"
  "neudet_622"
)

declare -A DATASET_ROOTS
DATASET_ROOTS["gc10det_622_halves"]="${ROOT}/experiments/gc10det_622_halves"
DATASET_ROOTS["DeepPCB_standard"]="${ROOT}/dataset/yolo/DeepPCB_standard"
DATASET_ROOTS["kolektorsdd_622_halves"]="${ROOT}/dataset/yolo/kolektorsdd_622_halves"
DATASET_ROOTS["neudet_622"]="${ROOT}/dataset/yolo/neudet_622"

pick_data_yaml() {
  local ds_root="$1"
  if [[ -f "${ds_root}/data.yaml" ]]; then
    echo "${ds_root}/data.yaml"
    return 0
  fi
  if [[ -f "${ds_root}/dataset.yaml" ]]; then
    echo "${ds_root}/dataset.yaml"
    return 0
  fi
  return 1
}

make_cfg() {
  local base_cfg="$1"
  local out_cfg="$2"
  local data_yaml="$3"
  local data_root="$4"
  local exp_name="$5"
  local epochs="$6"
  local batch="$7"
  local device="$8"
  local workers="$9"
  local skip_post="${10}"

  python - "${base_cfg}" "${out_cfg}" "${data_yaml}" "${data_root}" "${exp_name}" "${epochs}" "${batch}" "${device}" "${workers}" "${skip_post}" <<'PY'
from pathlib import Path
import sys
import yaml

base_cfg = Path(sys.argv[1]).resolve()
out_cfg = Path(sys.argv[2]).resolve()
data_yaml = Path(sys.argv[3]).resolve()
data_root = Path(sys.argv[4]).resolve()
exp_name = str(sys.argv[5])
epochs = int(sys.argv[6])
batch = int(sys.argv[7])
device = str(sys.argv[8])
workers = int(sys.argv[9])
skip_post = str(sys.argv[10]).strip().lower() in {"1", "true", "yes", "on"}

cfg = yaml.safe_load(base_cfg.read_text(encoding="utf-8"))
if not isinstance(cfg, dict):
    raise SystemExit(f"invalid yaml mapping: {base_cfg}")

cfg["data"] = str(data_yaml)
cfg["data_root"] = str(data_root)
cfg["exp_name"] = exp_name
cfg["run_name"] = ""
cfg["epochs"] = epochs
cfg["batch"] = batch
cfg["device"] = device
cfg["workers"] = workers
cfg["mode"] = "train_test"
cfg["skip_post_eval_metrics"] = skip_post

out_cfg.parent.mkdir(parents=True, exist_ok=True)
out_cfg.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
print(out_cfg)
PY
}

if [[ ! -f "${TEMPLATE_CFG}" ]]; then
  echo "[error] template config not found: ${TEMPLATE_CFG}" >&2
  exit 2
fi

echo "[plan] ROOT=${ROOT}"
echo "[plan] TEMPLATE_CFG=${TEMPLATE_CFG}"
echo "[plan] CFG_OUT_DIR=${CFG_OUT_DIR}"
echo "[plan] EPOCHS=${EPOCHS} BATCH_LIST=${BATCH_LIST} DEVICE=${DEVICE} WORKERS=${WORKERS}"
echo "[plan] SKIP_POST_EVAL=${SKIP_POST_EVAL} RESUME_POLICY=${RESUME_POLICY}"

for alias in "${DATASET_ALIASES[@]}"; do
  ds_root="${DATASET_ROOTS[${alias}]}"
  if ! data_yaml="$(pick_data_yaml "${ds_root}")"; then
    echo "[error] data.yaml/dataset.yaml not found for ${alias}: ${ds_root}" >&2
    exit 3
  fi

  for bs in ${BATCH_LIST}; do
    cfg_path="${CFG_OUT_DIR}/defect241_${alias}_bs${bs}_e${EPOCHS}.yaml"
    exp_name="smoke241/${alias}/bs${bs}_e${EPOCHS}"
    make_cfg "${TEMPLATE_CFG}" "${cfg_path}" "${data_yaml}" "${ds_root}" "${exp_name}" "${EPOCHS}" "${bs}" "${DEVICE}" "${WORKERS}" "${SKIP_POST_EVAL}" >/dev/null

    echo "[run] dataset=${alias} batch=${bs} epochs=${EPOCHS}"
    echo "[run] cfg=${cfg_path}"
    (
      cd "${ROOT}"
      env \
        XDG_CONFIG_HOME=/tmp/.config \
        MPLCONFIGDIR=/tmp/matplotlib \
        FC_CACHEDIR=/tmp/fontconfig \
        E241_RESUME_POLICY="${RESUME_POLICY}" \
        bash tools/run_yolov11_241.sh --base-config "${cfg_path}"
    )
  done
done

echo "[done] all dataset/batch jobs finished."
