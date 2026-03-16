#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

BATCHES_RAW=""
EPOCHS_TARGET="150"
COMBOS_RAW="baseline,a4+b7+d6"
DATASETS_RAW=""
YOLO_VERSION="yolo11"
TAG_PREFIX="batch"
PARALLEL_HINT="1"

usage() {
  cat <<'USAGE'
Usage:
  bash tools/check_yolov11_241_multi_dataset_recovery.sh \
    --batches "6,10" \
    --epochs 150 \
    --combos "baseline,a4+b7+d6" \
    --datasets "/abs/ds1,/abs/ds2,..." \
    [--yolo-version yolo11] \
    [--tag-prefix batch] \
    [--parallel 1]

Purpose:
  Inspect historical multi-dataset batch runs and classify each (dataset, combo) as:
  - done: results.csv epoch reached target and checkpoint exists
  - resume: partial epochs with checkpoint
  - fresh: missing/invalid progress

Output:
  - Prints an aligned table.
  - Prints suggested commands:
    1) continue command (reuse latest batch tag for that batch size)
    2) fresh command (new timestamp tag)
USAGE
}

append_csv_items() {
  local raw="$1"
  local normalized item
  normalized="${raw//、/,}"
  IFS=',' read -r -a _parts <<< "${normalized}"
  for item in "${_parts[@]}"; do
    item="$(echo "${item}" | sed -E 's/^[[:space:]]+//; s/[[:space:]]+$//')"
    [[ -n "${item}" ]] && ITEMS+=("${item}")
  done
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --batches|--batch-list)
      [[ $# -ge 2 ]] || { echo "[error] $1 requires a value" >&2; exit 2; }
      BATCHES_RAW="$2"
      shift 2
      ;;
    --epochs)
      [[ $# -ge 2 ]] || { echo "[error] --epochs requires a value" >&2; exit 2; }
      EPOCHS_TARGET="$2"
      shift 2
      ;;
    --combos)
      [[ $# -ge 2 ]] || { echo "[error] --combos requires a value" >&2; exit 2; }
      COMBOS_RAW="$2"
      shift 2
      ;;
    --datasets)
      [[ $# -ge 2 ]] || { echo "[error] --datasets requires a value" >&2; exit 2; }
      DATASETS_RAW="$2"
      shift 2
      ;;
    --yolo-version)
      [[ $# -ge 2 ]] || { echo "[error] --yolo-version requires a value" >&2; exit 2; }
      YOLO_VERSION="$2"
      shift 2
      ;;
    --tag-prefix)
      [[ $# -ge 2 ]] || { echo "[error] --tag-prefix requires a value" >&2; exit 2; }
      TAG_PREFIX="$2"
      shift 2
      ;;
    --parallel)
      [[ $# -ge 2 ]] || { echo "[error] --parallel requires a value" >&2; exit 2; }
      PARALLEL_HINT="$2"
      shift 2
      ;;
    *)
      echo "[error] unknown arg: $1" >&2
      usage
      exit 2
      ;;
  esac
done

[[ -n "${BATCHES_RAW}" ]] || { echo "[error] --batches is required" >&2; exit 2; }
[[ -n "${DATASETS_RAW}" ]] || { echo "[error] --datasets is required" >&2; exit 2; }
[[ "${EPOCHS_TARGET}" =~ ^[0-9]+$ ]] || { echo "[error] --epochs must be integer" >&2; exit 2; }
[[ "${PARALLEL_HINT}" =~ ^[0-9]+$ ]] || { echo "[error] --parallel must be integer" >&2; exit 2; }

ITEMS=()
append_csv_items "${BATCHES_RAW}"
BATCHES=("${ITEMS[@]}")
[[ "${#BATCHES[@]}" -gt 0 ]] || { echo "[error] no valid batch values" >&2; exit 2; }
for b in "${BATCHES[@]}"; do
  [[ "${b}" =~ ^[0-9]+$ ]] || { echo "[error] invalid batch: ${b}" >&2; exit 2; }
done

ITEMS=()
append_csv_items "${DATASETS_RAW}"
DATASETS=("${ITEMS[@]}")
[[ "${#DATASETS[@]}" -gt 0 ]] || { echo "[error] no valid dataset paths" >&2; exit 2; }

RUN_ROOT="${ROOT_DIR}/experiments/${YOLO_VERSION}/batch_runs"
SUMMARY_TSV="/tmp/check_multi_dataset_recovery_$(date +%y%m%d%H%M%S).tsv"

case_lines="$("${PYTHON_BIN}" - <<'PY' "${COMBOS_RAW}"
import sys

raw = (sys.argv[1] if len(sys.argv) > 1 else "").strip()
tokens = [x.strip() for x in raw.split(",") if x.strip()]
if not tokens:
    tokens = ["baseline"]

cases = []
seen = set()

def add(tag, switches):
    if tag in seen:
        return
    seen.add(tag)
    suffix = ""
    if switches:
        suffix = "__" + "__".join(switches)
    cases.append((tag, suffix))

def normalize_and_add_case(token):
    t = token.lower().replace(" ", "")
    if not t:
        return
    if t in {"baseline", "base", "none"}:
        add("baseline", [])
        return
    if t in {"hmc7", "abcd7", "a7_b7_c7_d7", "a7+b7+c7+d7"}:
        add("a7_b7_c7_d7", ["a7", "b7", "c7", "d7"])
        return
    if t in {"pdd9", "abcd9", "a9_b9_c9_d9", "a9+b9+c9+d9"}:
        add("a9_b9_c9_d9", ["a9", "b9", "c9", "d9"])
        return
    if t in {"abcd6", "a6_b6_c6_d6", "a6+b6+c6+d6"}:
        add("a6_b6_c6_d6", ["a6", "b6", "c6", "d6"])
        return
    if t in {"abcd11", "a11_b11_c11_d11", "a11+b11+c11+d11"}:
        add("a11_b11_c11_d11", ["a11", "b11", "c11", "d11"])
        return
    if t in {"abcd21", "pack21", "a21_b21_c21_d21", "a21+b21+c21+d21"}:
        add("a21_b21_c21_d21", ["a21", "b21", "c21", "d21"])
        return
    if t == "b1237":
        add("b1", ["b1"])
        add("b2", ["b2"])
        add("b3", ["b3"])
        add("b7", ["b7"])
        return
    if t == "d1579":
        add("d1", ["d1"])
        add("d5", ["d5"])
        add("d7", ["d7"])
        add("d9", ["d9"])
        return

    supported = {
        "a3","a4","a5","a6","a7","a9","a11","a21",
        "b1","b2","b3","b5","b6","b7","b9","b11","b21",
        "c4","c5","c6","c7","c9","c11","c21",
        "d1","d3","d5","d6","d7","d9","d11","d21",
    }
    if t in supported:
        add(t, [t])
        return

    expr = t.replace("_", "+")
    parts = []
    seen_local = set()
    for p in expr.split("+"):
        if not p or p in {"baseline", "base", "none"}:
            continue
        if p not in supported:
            raise SystemExit(f"unsupported combo token '{p}' in '{token}'")
        if p not in seen_local:
            seen_local.add(p)
            parts.append(p)
    if not parts:
        add("baseline", [])
    else:
        add("_".join(parts), parts)

for tk in tokens:
    normalize_and_add_case(tk)

for tag, suffix in cases:
    print(f"{tag}\t{suffix}")
PY
)"

CASE_TAGS=()
CASE_SUFFIXES=()
while IFS=$'\t' read -r tag suffix; do
  [[ -z "${tag}" ]] && continue
  CASE_TAGS+=("${tag}")
  CASE_SUFFIXES+=("${suffix}")
done <<< "${case_lines}"
[[ "${#CASE_TAGS[@]}" -gt 0 ]] || { echo "[error] no cases resolved from --combos" >&2; exit 2; }

normalize_dataset_tag() {
  local raw="$1"
  local base
  base="$(basename "${raw}")"
  base="$(echo "${base}" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9]+/_/g; s/^_+//; s/_+$//')"
  [[ -n "${base}" ]] || base="dataset"
  echo "${base}"
}

find_latest_batch_tag() {
  local batch="$1"
  local prefix
  prefix="${TAG_PREFIX}_${batch}_"
  if [[ ! -d "${RUN_ROOT}" ]]; then
    return 0
  fi
  find "${RUN_ROOT}" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' 2>/dev/null \
    | awk -v p="${prefix}" 'index($0, p) == 1' \
    | sort \
    | tail -n 1 || true
}

read_max_epoch() {
  local results_csv="$1"
  [[ -f "${results_csv}" ]] || { echo ""; return 0; }
  awk -F',' '
    NR == 1 && $1 == "epoch" { next }
    {
      gsub(/^[ \t]+|[ \t]+$/, "", $1)
      if ($1 ~ /^-?[0-9]+([.][0-9]+)?$/) {
        v = int($1 + 0)
        if (m == "" || v > m) m = v
      }
    }
    END { print m }
  ' "${results_csv}" 2>/dev/null || true
}

{
  echo -e "batch\trun_tag\tdataset_tag\tcombo_tag\tstatus\taction\tmax_epoch\ttarget_epochs\tremaining_epochs\texp_dir\tresume_weight\tnote"
} > "${SUMMARY_TSV}"

for b in "${BATCHES[@]}"; do
  run_tag="$(find_latest_batch_tag "${b}")"
  for ds in "${DATASETS[@]}"; do
    ds_tag="$(normalize_dataset_tag "${ds}")"
    for i in "${!CASE_TAGS[@]}"; do
      combo_tag="${CASE_TAGS[$i]}"
      suffix="${CASE_SUFFIXES[$i]}"
      exp_leaf="${ds_tag}${suffix}"

      if [[ -z "${run_tag}" ]]; then
        printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
          "${b}" "" "${ds_tag}" "${combo_tag}" "fresh" "from_scratch" "" "${EPOCHS_TARGET}" "${EPOCHS_TARGET}" "" "" "no_previous_batch_tag" \
          >> "${SUMMARY_TSV}"
        continue
      fi

      exp_root="${RUN_ROOT}/${run_tag}/${exp_leaf}"
      latest_exp="$(ls -1dt "${exp_root}"/exp_* 2>/dev/null | head -n 1 || true)"
      if [[ -z "${latest_exp}" ]]; then
        printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
          "${b}" "${run_tag}" "${ds_tag}" "${combo_tag}" "fresh" "from_scratch" "" "${EPOCHS_TARGET}" "${EPOCHS_TARGET}" "" "" "missing_exp_dir" \
          >> "${SUMMARY_TSV}"
        continue
      fi

      best_pt="${latest_exp}/train/weights/best.pt"
      last_pt="${latest_exp}/train/weights/last.pt"
      has_best="0"
      has_last="0"
      [[ -f "${best_pt}" ]] && has_best="1"
      [[ -f "${last_pt}" ]] && has_last="1"
      resume_weight=""
      if [[ "${has_last}" == "1" ]]; then
        resume_weight="${last_pt}"
      elif [[ "${has_best}" == "1" ]]; then
        resume_weight="${best_pt}"
      fi

      max_epoch="$(read_max_epoch "${latest_exp}/train/results.csv")"
      status="fresh"
      action="from_scratch"
      remain="${EPOCHS_TARGET}"
      note=""

      if [[ -n "${max_epoch}" && -n "${resume_weight}" ]]; then
        if (( max_epoch >= EPOCHS_TARGET - 1 )); then
          status="done"
          action="validate_only"
          remain="0"
          note="complete_by_results"
        else
          status="resume"
          action="resume_finetune"
          remain="$(( EPOCHS_TARGET - (max_epoch + 1) ))"
          (( remain < 1 )) && remain=1
          note="partial_results_found"
        fi
      elif [[ -n "${max_epoch}" && -z "${resume_weight}" ]]; then
        status="fresh"
        action="from_scratch"
        note="no_checkpoint"
      else
        status="fresh"
        action="from_scratch"
        note="missing_or_invalid_results_csv"
      fi

      printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "${b}" "${run_tag}" "${ds_tag}" "${combo_tag}" "${status}" "${action}" "${max_epoch}" "${EPOCHS_TARGET}" "${remain}" "${latest_exp}" "${resume_weight}" "${note}" \
        >> "${SUMMARY_TSV}"
    done
  done
done

echo "[check] summary=${SUMMARY_TSV}"
if command -v column >/dev/null 2>&1; then
  awk -F $'\t' 'BEGIN{OFS=FS} NR==1{print; next} {for (i=1; i<=NF; i++) if ($i=="") $i="-"; print}' "${SUMMARY_TSV}" \
    | column -s $'\t' -t
else
  cat "${SUMMARY_TSV}"
fi
echo
echo "[check] suggested commands:"

for b in "${BATCHES[@]}"; do
  run_tag="$(find_latest_batch_tag "${b}")"
  if [[ -n "${run_tag}" ]]; then
    need_continue_count="$(awk -F $'\t' -v bb="${b}" 'NR>1 && $1==bb && $5!="done" {c++} END{print c+0}' "${SUMMARY_TSV}")"
    if [[ "${need_continue_count}" -gt 0 ]]; then
      echo "# batch=${b} continue (reuse latest tag)"
      echo "bash ${ROOT_DIR}/tools/run_yolov11_241_multi_dataset.sh --runner module_combo --parallel ${PARALLEL_HINT} --tag \"${run_tag}\" --batch ${b} --epochs ${EPOCHS_TARGET} --combos \"${COMBOS_RAW}\" --datasets \"${DATASETS_RAW}\""
    else
      echo "# batch=${b} all done under tag=${run_tag}"
    fi
  else
    new_tag="${TAG_PREFIX}_${b}_$(date +%y%m%d%H%M%S)"
    echo "# batch=${b} no history, fresh run"
    echo "bash ${ROOT_DIR}/tools/run_yolov11_241_multi_dataset.sh --runner module_combo --parallel ${PARALLEL_HINT} --tag \"${new_tag}\" --batch ${b} --epochs ${EPOCHS_TARGET} --combos \"${COMBOS_RAW}\" --datasets \"${DATASETS_RAW}\""
  fi
  echo
done
