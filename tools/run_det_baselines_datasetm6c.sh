#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET_ROOT="${DATASET_ROOT:-${ROOT_DIR}/dataset/yolo/datasetm6c}"
OUT_ROOT="${OUT_ROOT:-${ROOT_DIR}/experiments/det_bench/datasetm6c}"
ENV_NAME="${ENV_NAME:-yolo11}"
DEVICE="${DEVICE:-0}"
IMGSZ="${IMGSZ:-640}"

FRCNN_EPOCHS="${FRCNN_EPOCHS:-30}"
FRCNN_BATCH="${FRCNN_BATCH:-4}"
RTDETR_EPOCHS="${RTDETR_EPOCHS:-100}"
RTDETR_BATCH="${RTDETR_BATCH:-8}"
RTDETR_MODEL="${RTDETR_MODEL:-rtdetr-l.yaml}"
RTDETR_PRETRAINED="${RTDETR_PRETRAINED:-0}"
FRCNN_PRETRAINED_COCO="${FRCNN_PRETRAINED_COCO:-0}"
PRED_CONF="${PRED_CONF:-0.001}"
OBJ_SCORE_THR="${OBJ_SCORE_THR:-0.25}"

if [[ ! -d "${DATASET_ROOT}" ]]; then
  echo "[error] dataset not found: ${DATASET_ROOT}" >&2
  exit 2
fi

mkdir -p "${OUT_ROOT}/preds" "${OUT_ROOT}/metrics"

echo "[run] train Faster R-CNN"
FRCNN_EXTRA_ARGS=()
if [[ "${FRCNN_PRETRAINED_COCO}" == "1" ]]; then
  FRCNN_EXTRA_ARGS+=(--pretrained-coco)
fi
conda run -n "${ENV_NAME}" python "${ROOT_DIR}/third_party/faster_rcnn/train.py" \
  --dataset-root "${DATASET_ROOT}" \
  --output-dir "${OUT_ROOT}/faster_rcnn" \
  --epochs "${FRCNN_EPOCHS}" \
  --batch-size "${FRCNN_BATCH}" \
  --device "${DEVICE}" \
  "${FRCNN_EXTRA_ARGS[@]}"

echo "[run] infer Faster R-CNN"
conda run -n "${ENV_NAME}" python "${ROOT_DIR}/third_party/faster_rcnn/predict.py" \
  --weights "${OUT_ROOT}/faster_rcnn/best.pt" \
  --dataset-root "${DATASET_ROOT}" \
  --split test \
  --output-json "${OUT_ROOT}/preds/faster_rcnn_test.json" \
  --batch-size "${FRCNN_BATCH}" \
  --conf "${PRED_CONF}" \
  --device "${DEVICE}"

echo "[run] train RT-DETR"
RTDETR_EXTRA_ARGS=()
if [[ "${RTDETR_PRETRAINED}" == "1" ]]; then
  RTDETR_EXTRA_ARGS+=(--pretrained)
fi
conda run -n "${ENV_NAME}" python "${ROOT_DIR}/third_party/rt_detr/train.py" \
  --dataset-root "${DATASET_ROOT}" \
  --output-dir "${OUT_ROOT}/rt_detr" \
  --model "${RTDETR_MODEL}" \
  --epochs "${RTDETR_EPOCHS}" \
  --imgsz "${IMGSZ}" \
  --batch "${RTDETR_BATCH}" \
  --device "${DEVICE}" \
  "${RTDETR_EXTRA_ARGS[@]}"

echo "[run] infer RT-DETR"
conda run -n "${ENV_NAME}" python "${ROOT_DIR}/third_party/rt_detr/predict.py" \
  --weights "${OUT_ROOT}/rt_detr/best.pt" \
  --dataset-root "${DATASET_ROOT}" \
  --split test \
  --output-json "${OUT_ROOT}/preds/rt_detr_test.json" \
  --imgsz "${IMGSZ}" \
  --conf "${PRED_CONF}" \
  --batch "${RTDETR_BATCH}" \
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
