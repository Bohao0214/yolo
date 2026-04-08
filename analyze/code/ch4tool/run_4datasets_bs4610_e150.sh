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

dataset_root_candidates() {
  local alias="$1"
  case "${alias}" in
    gc10det_622_halves)
      cat <<EOF
${ROOT}/experiments/gc10det_622_halves
${ROOT}/dataset/yolo/gc10det_622_halves
EOF
      ;;
    DeepPCB_standard)
      cat <<EOF
${ROOT}/dataset/yolo/DeepPCB_standard
${ROOT}/experiments/DeepPCB_standard
EOF
      ;;
    kolektorsdd_622_halves)
      cat <<EOF
${ROOT}/dataset/yolo/kolektorsdd_622_halves
${ROOT}/experiments/kolektorsdd_622_halves
EOF
      ;;
    neudet_622)
      cat <<EOF
${ROOT}/dataset/yolo/neudet_622
${ROOT}/experiments/neudet_622
EOF
      ;;
    *)
      cat <<EOF
${ROOT}/dataset/yolo/${alias}
${ROOT}/experiments/${alias}
EOF
      ;;
  esac
}

pick_dataset_root() {
  local alias="$1"
  local cand
  while IFS= read -r cand; do
    [[ -z "${cand}" ]] && continue
    if [[ -d "${cand}" ]]; then
      echo "${cand}"
      return 0
    fi
  done < <(dataset_root_candidates "${alias}")
  return 1
}

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

auto_make_data_yaml() {
  local alias="$1"
  local ds_root="$2"
  local out_yaml="$3"

  python - "${alias}" "${ds_root}" "${out_yaml}" <<'PY'
from pathlib import Path
import sys
import yaml

alias = str(sys.argv[1])
ds_root = Path(sys.argv[2]).resolve()
out_yaml = Path(sys.argv[3]).resolve()

images = ds_root / "images"
labels = ds_root / "labels"
if not images.exists():
    raise SystemExit(f"images dir not found: {images}")
if not labels.exists():
    raise SystemExit(f"labels dir not found: {labels}")

def choose_split(primary: str, fallback: str) -> str:
    p = images / primary
    if p.exists():
        return f"images/{primary}"
    f = images / fallback
    if f.exists():
        return f"images/{fallback}"
    return "images/train"

train_rel = choose_split("train", "val")
val_rel = choose_split("val", "test")
test_rel = choose_split("test", "val")

cls_ids = set()
for split in ("train", "val", "test"):
    d = labels / split
    if not d.exists():
        continue
    for txt in d.glob("*.txt"):
        try:
            for line in txt.read_text(encoding="utf-8", errors="ignore").splitlines():
                s = line.strip().split()
                if not s:
                    continue
                cls_ids.add(int(float(s[0])))
        except Exception:
            continue

if cls_ids:
    max_id = max(cls_ids)
    names = [f"class_{i}" for i in range(max_id + 1)]
else:
    names = ["defect"]

data = {
    "path": str(ds_root),
    "train": train_rel,
    "val": val_rel,
    "test": test_rel,
    "nc": len(names),
    "names": names,
}
out_yaml.parent.mkdir(parents=True, exist_ok=True)
out_yaml.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
print(out_yaml)
PY
}

resolve_data_yaml() {
  local alias="$1"
  local ds_root="$2"
  local data_yaml
  if data_yaml="$(pick_data_yaml "${ds_root}")"; then
    echo "${data_yaml}"
    return 0
  fi
  data_yaml="${CFG_OUT_DIR}/auto_data_${alias}.yaml"
  auto_make_data_yaml "${alias}" "${ds_root}" "${data_yaml}" >/dev/null
  echo "${data_yaml}"
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
  if ! ds_root="$(pick_dataset_root "${alias}")"; then
    echo "[error] dataset root not found for ${alias}" >&2
    echo "[hint] tried:" >&2
    dataset_root_candidates "${alias}" | sed 's/^/  - /' >&2
    exit 3
  fi
  data_yaml="$(resolve_data_yaml "${alias}" "${ds_root}")"
  echo "[info] dataset=${alias} root=${ds_root}"
  echo "[info] data_yaml=${data_yaml}"

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
