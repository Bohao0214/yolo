#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

BASE_CONFIG="${BASE_CONFIG:-${ROOT_DIR}/configs/baseline/datasetm6c.yaml}"
DATASET_ROOT="${DATASET_ROOT:-${ROOT_DIR}/dataset/yolo/datasetm6c}"
EPOCHS="${EPOCHS:-150}"
COMBOS="${COMBOS:-a3+b3+d3,a7+b7+c7+d7}"
BATCH_LIST="${BATCH_LIST:-6,10}"
PARALLEL="${PARALLEL:-1}"
VRAM_GUARD="${VRAM_GUARD:-off}" # off|auto|on

usage() {
  cat <<'USAGE'
Usage:
  bash tools/run_yolov11_241_datasetm6c_combo.sh

Default behavior:
  - dataset: /home/ubuntu/hpproject/yolo/dataset/yolo/datasetm6c
  - epochs: 150
  - combos: a3+b3+d3, a7+b7+c7+d7
  - batch: 6 and 10

Env overrides:
  BASE_CONFIG   baseline config path
  DATASET_ROOT  dataset root dir
  EPOCHS        epochs for all runs
  COMBOS        combo list for module runner
  BATCH_LIST    comma list, e.g. "6,10"
  PARALLEL      multi-dataset parallel jobs (default 1)
  VRAM_GUARD    off|auto|on (default off)

Examples:
  bash tools/run_yolov11_241_datasetm6c_combo.sh
  BATCH_LIST=6 bash tools/run_yolov11_241_datasetm6c_combo.sh
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ ! -f "${BASE_CONFIG}" ]]; then
  echo "[error] base config not found: ${BASE_CONFIG}" >&2
  exit 2
fi
if [[ ! -d "${DATASET_ROOT}" ]]; then
  echo "[error] dataset root not found: ${DATASET_ROOT}" >&2
  exit 2
fi

IFS=',' read -r -a _BATCHES <<< "${BATCH_LIST}"
if [[ ${#_BATCHES[@]} -eq 0 ]]; then
  echo "[error] BATCH_LIST is empty" >&2
  exit 2
fi

echo "[run] base_config=${BASE_CONFIG}"
echo "[run] dataset=${DATASET_ROOT}"
echo "[run] epochs=${EPOCHS} combos=${COMBOS} batches=${BATCH_LIST}"

for raw_bs in "${_BATCHES[@]}"; do
  bs="$(echo "${raw_bs}" | sed -E 's/^[[:space:]]+//; s/[[:space:]]+$//')"
  [[ -z "${bs}" ]] && continue
  if ! [[ "${bs}" =~ ^[0-9]+$ ]]; then
    echo "[error] invalid batch value: ${bs}" >&2
    exit 2
  fi

  tag="datasetm6c_combo_e${EPOCHS}_b${bs}"
  echo "[run] start batch=${bs} tag=${tag}"

  bash "${ROOT_DIR}/tools/run_yolov11_241_multi_dataset.sh" \
    --base-config "${BASE_CONFIG}" \
    --runner module_combo \
    --epochs "${EPOCHS}" \
    --batch "${bs}" \
    --parallel "${PARALLEL}" \
    --method-tag "yolo11-combo_b${bs}" \
    --combos "${COMBOS}" \
    --vram-guard "${VRAM_GUARD}" \
    --tag "${tag}" \
    --dataset "${DATASET_ROOT}"

  echo "[run] done batch=${bs}"
done

echo "[run] all finished"
