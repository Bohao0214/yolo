#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET_ROOT="${DATASET_ROOT:-${ROOT_DIR}/dataset/yolo/datasetm6c}"
DATASET_TAG_RAW="$(basename "${DATASET_ROOT}")"
DATASET_TAG="${DATASET_TAG:-$(echo "${DATASET_TAG_RAW}" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9]+/_/g; s/^_+//; s/_+$//')}"
METHOD_TAG="${METHOD_TAG:-yolo11-det-bench}"
DATE_TAG="${DATE_TAG:-$(date +%y%m%d%H%M)}"
OUT_ROOT="${OUT_ROOT:-${ROOT_DIR}/experiments/${METHOD_TAG}/${DATASET_TAG}/exp_${DATE_TAG}}"
ENV_NAME="${ENV_NAME:-yolo11}"
DEVICE="${DEVICE:-0}"
IMGSZ="${IMGSZ:-640}"

FRCNN_EPOCHS="${FRCNN_EPOCHS:-30}"
FRCNN_BATCH="${FRCNN_BATCH:-4}"
FRCNN_LR="${FRCNN_LR:-0.0025}"
RTDETR_EPOCHS="${RTDETR_EPOCHS:-100}"
RTDETR_BATCH="${RTDETR_BATCH:-8}"
RTDETR_MODEL="${RTDETR_MODEL:-rtdetr-l.pt}"
RTDETR_PRETRAINED="${RTDETR_PRETRAINED:-1}"
FRCNN_PRETRAINED_COCO="${FRCNN_PRETRAINED_COCO:-1}"
PRED_CONF="${PRED_CONF:-0.001}"
OBJ_SCORE_THR="${OBJ_SCORE_THR:-0.05}"
FRCNN_BATCH_TRY="${FRCNN_BATCH_TRY:-${FRCNN_BATCH} 2}"
RTDETR_BATCH_TRY="${RTDETR_BATCH_TRY:-${RTDETR_BATCH} 2}"

if [[ ! -d "${DATASET_ROOT}" ]]; then
  echo "[error] dataset not found: ${DATASET_ROOT}" >&2
  exit 2
fi

mkdir -p "${OUT_ROOT}/preds" "${OUT_ROOT}/metrics"

is_oom_log() {
  local log_file="$1"
  if [[ ! -f "${log_file}" ]]; then
    return 1
  fi
  grep -Eiq "out of memory|cuda.*memory|cuda error: out of memory|cudnn_status_alloc_failed|cuda_oom|oom" "${log_file}"
}

run_with_fallback() {
  local tag="$1"
  local out_subdir="$2"
  local candidates="$3"
  shift 3
  local -a cmd_base=("$@")
  local tried=()
  local bs
  local used=""
  for bs in ${candidates}; do
    if [[ -z "${bs}" ]]; then
      continue
    fi
    tried+=("${bs}")
    echo "[run] ${tag} try batch=${bs}" >&2
    rm -rf "${OUT_ROOT}/${out_subdir}"
    mkdir -p "${OUT_ROOT}/${out_subdir}"
    local log_file
    log_file="${OUT_ROOT}/${out_subdir}/train_try_bs${bs}.log"
    local -a cmd=()
    local tok
    for tok in "${cmd_base[@]}"; do
      if [[ "${tok}" == "__BATCH__" ]]; then
        cmd+=("${bs}")
      else
        cmd+=("${tok}")
      fi
    done
    (
      set +e
      "${cmd[@]}"
    ) >"${log_file}" 2>&1
    local rc=$?
    if [[ ${rc} -eq 0 ]]; then
      echo "[done] ${tag} batch=${bs}" >&2
      used="${bs}"
      break
    fi
    if is_oom_log "${log_file}"; then
      echo "[warn] ${tag} batch=${bs} failed with OOM, fallback to next candidate..." >&2
      continue
    fi
    echo "[error] ${tag} batch=${bs} failed (non-OOM). see ${log_file}" >&2
    cat "${log_file}" >&2
    return ${rc}
  done
  if [[ -z "${used}" ]]; then
    echo "[error] ${tag} all batch candidates failed: ${tried[*]}" >&2
    return 90
  fi
  echo "${used}"
}

echo "[cfg] DATASET_ROOT=${DATASET_ROOT}"
echo "[cfg] OUT_ROOT=${OUT_ROOT}"
echo "[cfg] DEVICE=${DEVICE} IMGSZ=${IMGSZ}"
echo "[cfg] FRCNN_EPOCHS=${FRCNN_EPOCHS} FRCNN_BATCH_TRY='${FRCNN_BATCH_TRY}' FRCNN_LR=${FRCNN_LR} FRCNN_PRETRAINED_COCO=${FRCNN_PRETRAINED_COCO}"
echo "[cfg] RTDETR_MODEL=${RTDETR_MODEL} RTDETR_EPOCHS=${RTDETR_EPOCHS} RTDETR_BATCH_TRY='${RTDETR_BATCH_TRY}' RTDETR_PRETRAINED=${RTDETR_PRETRAINED}"
echo "[cfg] PRED_CONF=${PRED_CONF} OBJ_SCORE_THR=${OBJ_SCORE_THR}"

echo "[run] train Faster R-CNN"
FRCNN_EXTRA_ARGS=()
if [[ "${FRCNN_PRETRAINED_COCO}" == "1" ]]; then
  FRCNN_EXTRA_ARGS+=(--pretrained-coco)
fi
FRCNN_BATCH_USED="$(run_with_fallback \
  "Faster R-CNN train" \
  "faster_rcnn" \
  "${FRCNN_BATCH_TRY}" \
  conda run -n "${ENV_NAME}" python "${ROOT_DIR}/third_party/faster_rcnn/train.py" \
    --dataset-root "${DATASET_ROOT}" \
    --output-dir "${OUT_ROOT}/faster_rcnn" \
    --epochs "${FRCNN_EPOCHS}" \
    --batch-size "__BATCH__" \
    --lr "${FRCNN_LR}" \
    --device "${DEVICE}" \
    "${FRCNN_EXTRA_ARGS[@]}")"
echo "[cfg] FRCNN_BATCH_USED=${FRCNN_BATCH_USED}"

echo "[run] infer Faster R-CNN"
conda run -n "${ENV_NAME}" python "${ROOT_DIR}/third_party/faster_rcnn/predict.py" \
  --weights "${OUT_ROOT}/faster_rcnn/best.pt" \
  --dataset-root "${DATASET_ROOT}" \
  --split test \
  --output-json "${OUT_ROOT}/preds/faster_rcnn_test.json" \
  --batch-size "${FRCNN_BATCH_USED}" \
  --conf "${PRED_CONF}" \
  --device "${DEVICE}"

echo "[run] train RT-DETR"
RTDETR_EXTRA_ARGS=()
if [[ "${RTDETR_PRETRAINED}" == "1" ]]; then
  RTDETR_EXTRA_ARGS+=(--pretrained)
fi
RTDETR_BATCH_USED="$(run_with_fallback \
  "RT-DETR train" \
  "rt_detr" \
  "${RTDETR_BATCH_TRY}" \
  conda run -n "${ENV_NAME}" python "${ROOT_DIR}/third_party/rt_detr/train.py" \
    --dataset-root "${DATASET_ROOT}" \
    --output-dir "${OUT_ROOT}/rt_detr" \
    --model "${RTDETR_MODEL}" \
    --epochs "${RTDETR_EPOCHS}" \
    --imgsz "${IMGSZ}" \
    --batch "__BATCH__" \
    --device "${DEVICE}" \
    "${RTDETR_EXTRA_ARGS[@]}")"
echo "[cfg] RTDETR_BATCH_USED=${RTDETR_BATCH_USED}"

echo "[run] infer RT-DETR"
conda run -n "${ENV_NAME}" python "${ROOT_DIR}/third_party/rt_detr/predict.py" \
  --weights "${OUT_ROOT}/rt_detr/best.pt" \
  --dataset-root "${DATASET_ROOT}" \
  --split test \
  --output-json "${OUT_ROOT}/preds/rt_detr_test.json" \
  --imgsz "${IMGSZ}" \
  --conf "${PRED_CONF}" \
  --batch "${RTDETR_BATCH_USED}" \
  --device "${DEVICE}"

echo "[run] unified evaluation"
conda run -n "${ENV_NAME}" python "${ROOT_DIR}/tools/eval_detection_benchmark.py" \
  --dataset-root "${DATASET_ROOT}" \
  --split test \
  --pred-json "faster_rcnn=${OUT_ROOT}/preds/faster_rcnn_test.json" \
  --pred-json "rt_detr=${OUT_ROOT}/preds/rt_detr_test.json" \
  --score-thr "${OBJ_SCORE_THR}" \
  --out-csv "${OUT_ROOT}/metrics/main_table.csv" \
  --out-json "${OUT_ROOT}/metrics/main_table.json"

echo "[done] main table:"
echo "  ${OUT_ROOT}/metrics/main_table.csv"
