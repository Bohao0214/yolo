#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

RUN_ROOTS_RAW=""
EPOCHS_TARGET="150"
EVAL_BATCH="1"
EVAL_WORKERS="2"
TRAIN_BATCH_OVERRIDE=""
MAX_PARALLEL="1"
EXECUTE="false"
INCLUDE_FROM_SCRATCH="false"

usage() {
  cat <<'USAGE'
Usage:
  bash tools/repair_yolov11_241_batch_runs.sh \
    --run-roots "/abs/run1,/abs/run2,/abs/run3" \
    --epochs 150 \
    --eval-batch 1 \
    --parallel 1 \
    [--train-batch N] \
    [--eval-workers 2] \
    [--include-from-scratch] \
    [--execute]

Purpose:
  Repair existing legacy batch run directories (typically under experiments/*/batch_runs):
  - Unfinished runs: resume training from last/best checkpoint.
  - Finished but post-eval incomplete (or error.md exists): run test-only with smaller eval batch.
  - Finished and metrics complete: skip.

Classification rules (per dataset/combo leaf dir):
  1) Locate latest exp_*.
  2) Read train/results.csv max epoch.
  3) Check train/weights/{best,last}.pt.
  4) Check metrics completeness:
     - metrics/eval_summary.txt exists and non-empty
     - metrics/eval_image_level.csv exists and non-empty
  5) If error.md exists in exp dir, treat as needing validate-only even if epoch complete.

Without --execute:
  - Only prints summary + suggested execute command.

With --execute:
  - Builds runtime configs under /tmp.
  - Runs `python src/train.py --config <runtime_cfg>` for actionable items.
  - validate-only jobs will force:
      mode=test
      batch=<eval-batch>
      eval_batch=<eval-batch>
      workers=<eval-workers>
      skip_post_eval_metrics=false

Action selection:
  default actionable actions:
    - resume_finetune
    - validate_only_low_batch
  add --include-from-scratch to also run:
    - from_scratch
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

is_integer() {
  [[ "$1" =~ ^[0-9]+$ ]]
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --run-roots)
      [[ $# -ge 2 ]] || { echo "[error] --run-roots requires a value" >&2; exit 2; }
      RUN_ROOTS_RAW="$2"
      shift 2
      ;;
    --epochs)
      [[ $# -ge 2 ]] || { echo "[error] --epochs requires a value" >&2; exit 2; }
      EPOCHS_TARGET="$2"
      shift 2
      ;;
    --eval-batch)
      [[ $# -ge 2 ]] || { echo "[error] --eval-batch requires a value" >&2; exit 2; }
      EVAL_BATCH="$2"
      shift 2
      ;;
    --eval-workers)
      [[ $# -ge 2 ]] || { echo "[error] --eval-workers requires a value" >&2; exit 2; }
      EVAL_WORKERS="$2"
      shift 2
      ;;
    --train-batch)
      [[ $# -ge 2 ]] || { echo "[error] --train-batch requires a value" >&2; exit 2; }
      TRAIN_BATCH_OVERRIDE="$2"
      shift 2
      ;;
    --parallel)
      [[ $# -ge 2 ]] || { echo "[error] --parallel requires a value" >&2; exit 2; }
      MAX_PARALLEL="$2"
      shift 2
      ;;
    --execute)
      EXECUTE="true"
      shift
      ;;
    --include-from-scratch)
      INCLUDE_FROM_SCRATCH="true"
      shift
      ;;
    *)
      echo "[error] unknown arg: $1" >&2
      usage
      exit 2
      ;;
  esac
done

[[ -n "${RUN_ROOTS_RAW}" ]] || { echo "[error] --run-roots is required" >&2; exit 2; }
is_integer "${EPOCHS_TARGET}" || { echo "[error] --epochs must be integer" >&2; exit 2; }
is_integer "${EVAL_BATCH}" || { echo "[error] --eval-batch must be integer" >&2; exit 2; }
is_integer "${EVAL_WORKERS}" || { echo "[error] --eval-workers must be integer" >&2; exit 2; }
is_integer "${MAX_PARALLEL}" || { echo "[error] --parallel must be integer" >&2; exit 2; }

if [[ -n "${TRAIN_BATCH_OVERRIDE}" ]]; then
  is_integer "${TRAIN_BATCH_OVERRIDE}" || { echo "[error] --train-batch must be integer" >&2; exit 2; }
fi

ITEMS=()
append_csv_items "${RUN_ROOTS_RAW}"
RUN_ROOTS=("${ITEMS[@]}")
[[ "${#RUN_ROOTS[@]}" -gt 0 ]] || { echo "[error] no valid run roots" >&2; exit 2; }

for rr in "${RUN_ROOTS[@]}"; do
  [[ -d "${rr}" ]] || { echo "[error] run root not found: ${rr}" >&2; exit 2; }
done

if ! "${PYTHON_BIN}" - <<'PY' >/dev/null 2>&1
import yaml  # noqa: F401
PY
then
  echo "[error] python has no PyYAML: ${PYTHON_BIN}" >&2
  exit 3
fi

RUN_TS="$(date +%y%m%d%H%M%S)"
TMP_ROOT="/tmp/yolo241_batch_repair_${RUN_TS}"
mkdir -p "${TMP_ROOT}"

SUMMARY_TSV="${TMP_ROOT}/summary.tsv"

RUN_ROOTS_JOINED="$(IFS=,; echo "${RUN_ROOTS[*]}")"

"${PYTHON_BIN}" - <<'PY' "${RUN_ROOTS_JOINED}" "${EPOCHS_TARGET}" > "${SUMMARY_TSV}"
import csv
import os
import sys
from pathlib import Path

run_roots = [x.strip() for x in (sys.argv[1] if len(sys.argv) > 1 else "").split(",") if x.strip()]
target_epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 150


def read_max_epoch(results_csv: Path):
    if not results_csv.exists():
        return None
    try:
        with results_csv.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames and "epoch" in [str(x).strip() for x in reader.fieldnames]:
                m = None
                for row in reader:
                    v = row.get("epoch", "")
                    try:
                        e = int(round(float(str(v).strip())))
                    except Exception:
                        continue
                    m = e if m is None else max(m, e)
                return m
    except Exception:
        pass

    try:
        with results_csv.open("r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            m = None
            for row in reader:
                if not row:
                    continue
                try:
                    e = int(round(float(str(row[0]).strip())))
                except Exception:
                    continue
                m = e if m is None else max(m, e)
            return m
    except Exception:
        return None


def latest_exp_dir(leaf_dir: Path):
    exps = [p for p in leaf_dir.glob("exp_*") if p.is_dir()]
    if not exps:
        return None
    exps.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return exps[0]


def norm_run_tag(run_root: Path):
    return run_root.name


print(
    "\t".join(
        [
            "run_tag",
            "run_root",
            "leaf_tag",
            "exp_dir",
            "action",
            "status",
            "max_epoch",
            "remaining_epochs",
            "weight",
            "config_src",
            "metrics_complete",
            "has_error_md",
            "note",
        ]
    )
)

for rr in run_roots:
    run_root = Path(rr).resolve()
    run_tag = norm_run_tag(run_root)
    leaf_dirs = [p for p in run_root.iterdir() if p.is_dir()]
    leaf_dirs.sort(key=lambda p: p.name)

    for leaf in leaf_dirs:
        latest = latest_exp_dir(leaf)
        if latest is None:
            print(
                "\t".join(
                    [
                        run_tag,
                        str(run_root),
                        leaf.name,
                        "",
                        "skip_missing_exp",
                        "missing_exp",
                        "",
                        "",
                        "",
                        "",
                        "0",
                        "0",
                        "no_exp_dir",
                    ]
                )
            )
            continue

        train_dir = latest / "train"
        metrics_dir = latest / "metrics"
        cfg_src = train_dir / "config.yaml"
        if not cfg_src.exists():
            alt = latest / "config.yaml"
            cfg_src = alt if alt.exists() else cfg_src

        best_pt = train_dir / "weights" / "best.pt"
        last_pt = train_dir / "weights" / "last.pt"
        has_best = best_pt.exists()
        has_last = last_pt.exists()
        has_ckpt = has_best or has_last

        max_epoch = read_max_epoch(train_dir / "results.csv")

        eval_summary = metrics_dir / "eval_summary.txt"
        eval_img = metrics_dir / "eval_image_level.csv"
        metrics_complete = int(eval_summary.exists() and eval_summary.stat().st_size > 0 and eval_img.exists() and eval_img.stat().st_size > 0)
        has_error = int((latest / "error.md").exists())

        resume_weight = ""
        eval_weight = ""
        if has_last:
            resume_weight = str(last_pt)
        elif has_best:
            resume_weight = str(best_pt)
        if has_best:
            eval_weight = str(best_pt)
        elif has_last:
            eval_weight = str(last_pt)
        weight = ""

        action = "skip_done"
        status = "done"
        remain = 0
        note = "complete_and_metrics_ready"

        if not cfg_src.exists():
            action = "skip_no_config"
            status = "blocked"
            note = "missing_train_config_yaml"
        elif has_ckpt and max_epoch is not None and max_epoch >= (target_epochs - 1):
            weight = eval_weight
            if metrics_complete and not has_error:
                action = "skip_done"
                status = "done"
                note = "complete_and_metrics_ready"
            else:
                action = "validate_only_low_batch"
                status = "repair_validate"
                note = "complete_but_metrics_missing_or_error_md"
        elif has_ckpt:
            weight = resume_weight
            seen = (max_epoch + 1) if max_epoch is not None else 0
            remain = target_epochs - seen
            if remain < 1:
                remain = 1
            action = "resume_finetune"
            status = "resume"
            note = "partial_training_found"
        elif metrics_complete and not has_error:
            action = "skip_done_metric_only"
            status = "done"
            remain = 0
            note = "metrics_complete_but_no_checkpoint"
        else:
            action = "from_scratch"
            status = "fresh"
            remain = target_epochs
            note = "no_checkpoint_found"

        print(
            "\t".join(
                [
                    run_tag,
                    str(run_root),
                    leaf.name,
                    str(latest),
                    action,
                    status,
                    "" if max_epoch is None else str(max_epoch),
                    str(remain),
                    weight,
                    str(cfg_src) if cfg_src.exists() else "",
                    str(metrics_complete),
                    str(has_error),
                    note,
                ]
            )
        )
PY

echo "[repair] summary=${SUMMARY_TSV}"
if command -v column >/dev/null 2>&1; then
  awk -F $'\t' 'BEGIN{OFS=FS} NR==1{print; next} {for (i=1; i<=NF; i++) if ($i=="") $i="-"; print}' "${SUMMARY_TSV}" | column -s $'\t' -t
else
  cat "${SUMMARY_TSV}"
fi

ACTIONABLE_TSV="${TMP_ROOT}/actionable.tsv"
if [[ "${INCLUDE_FROM_SCRATCH}" == "true" ]]; then
  awk -F $'\t' 'NR==1 || $5=="resume_finetune" || $5=="validate_only_low_batch" || $5=="from_scratch" {print}' "${SUMMARY_TSV}" > "${ACTIONABLE_TSV}"
else
  awk -F $'\t' 'NR==1 || $5=="resume_finetune" || $5=="validate_only_low_batch" {print}' "${SUMMARY_TSV}" > "${ACTIONABLE_TSV}"
fi

action_count="$(awk -F $'\t' 'NR>1 {c++} END{print c+0}' "${ACTIONABLE_TSV}")"
echo "[repair] actionable=${action_count}"
include_flag=""
if [[ "${INCLUDE_FROM_SCRATCH}" == "true" ]]; then
  include_flag=" --include-from-scratch"
fi

if [[ "${EXECUTE}" != "true" ]]; then
  echo
  echo "[repair] preview only. To execute:"
  echo "bash ${ROOT_DIR}/tools/repair_yolov11_241_batch_runs.sh --run-roots \"${RUN_ROOTS_RAW}\" --epochs ${EPOCHS_TARGET} --eval-batch ${EVAL_BATCH} --eval-workers ${EVAL_WORKERS} --parallel ${MAX_PARALLEL}${TRAIN_BATCH_OVERRIDE:+ --train-batch ${TRAIN_BATCH_OVERRIDE}}${include_flag} --execute"
  exit 0
fi

if [[ "${action_count}" -eq 0 ]]; then
  echo "[repair] nothing to run."
  exit 0
fi

JOBS_TSV="${TMP_ROOT}/jobs.tsv"
{
  echo -e "run_tag\tleaf_tag\taction\truntime_cfg\tlog_path\tcmd"
} > "${JOBS_TSV}"

while IFS=$'\t' read -r run_tag run_root leaf_tag exp_dir action status max_epoch remain weight cfg_src metrics_complete has_error note; do
  [[ "${run_tag}" == "run_tag" ]] && continue
  [[ -n "${cfg_src}" && -f "${cfg_src}" ]] || continue

  safe_run="$(echo "${run_tag}" | sed -E 's/[^A-Za-z0-9._-]+/_/g')"
  safe_leaf="$(echo "${leaf_tag}" | sed -E 's/[^A-Za-z0-9._-]+/_/g')"
  runtime_cfg="${TMP_ROOT}/${safe_run}__${safe_leaf}.yaml"
  run_log_dir="${run_root}/_repair_logs/${RUN_TS}"
  mkdir -p "${run_log_dir}"
  log_path="${run_log_dir}/${safe_leaf}.log"

  _R_CFG_SRC="${cfg_src}" \
  _R_CFG_OUT="${runtime_cfg}" \
  _R_ACTION="${action}" \
  _R_WEIGHT="${weight}" \
  _R_REMAIN="${remain}" \
  _R_EPOCHS_TARGET="${EPOCHS_TARGET}" \
  _R_EVAL_BATCH="${EVAL_BATCH}" \
  _R_EVAL_WORKERS="${EVAL_WORKERS}" \
  _R_TRAIN_BATCH="${TRAIN_BATCH_OVERRIDE}" \
  "${PYTHON_BIN}" - <<'PY'
import os
from pathlib import Path
import yaml

src = Path(os.environ["_R_CFG_SRC"]).resolve()
out = Path(os.environ["_R_CFG_OUT"]).resolve()
action = os.environ["_R_ACTION"].strip()
weight = os.environ["_R_WEIGHT"].strip()
remain = int(os.environ.get("_R_REMAIN", "0") or 0)
epochs_target = int(os.environ.get("_R_EPOCHS_TARGET", "150") or 150)
eval_batch = int(os.environ.get("_R_EVAL_BATCH", "1") or 1)
eval_workers = int(os.environ.get("_R_EVAL_WORKERS", "2") or 2)
train_batch_raw = os.environ.get("_R_TRAIN_BATCH", "").strip()

with src.open("r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}
if not isinstance(cfg, dict):
    raise SystemExit(f"invalid config mapping: {src}")

cfg["skip_post_eval_metrics"] = False
cfg["eval_batch"] = int(eval_batch)
cfg["workers"] = int(eval_workers)

if action == "validate_only_low_batch":
    cfg["mode"] = "test"
    if weight:
        cfg["weights"] = weight
    cfg["batch"] = int(eval_batch)
elif action == "resume_finetune":
    cfg["mode"] = "finetune_test"
    if weight:
        cfg["weights"] = weight
    cfg["epochs"] = int(remain if remain > 0 else 1)
    if train_batch_raw:
        cfg["batch"] = int(train_batch_raw)
elif action == "from_scratch":
    cfg["mode"] = "train_test"
    cfg["epochs"] = int(epochs_target)
    if train_batch_raw:
        cfg["batch"] = int(train_batch_raw)
else:
    raise SystemExit(f"unsupported action for runtime cfg: {action}")

out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
PY

  cmd="${PYTHON_BIN} ${ROOT_DIR}/src/train.py --config ${runtime_cfg}"
  printf "%s\t%s\t%s\t%s\t%s\t%s\n" \
    "${run_tag}" "${leaf_tag}" "${action}" "${runtime_cfg}" "${log_path}" "${cmd}" \
    >> "${JOBS_TSV}"
done < "${ACTIONABLE_TSV}"

job_count="$(awk -F $'\t' 'NR>1 {c++} END{print c+0}' "${JOBS_TSV}")"
echo "[repair] jobs=${job_count}"

if [[ "${job_count}" -eq 0 ]]; then
  echo "[repair] nothing executable (likely missing config.yaml in old exp dirs)."
  exit 0
fi

if command -v column >/dev/null 2>&1; then
  column -s $'\t' -t "${JOBS_TSV}" || cat "${JOBS_TSV}"
else
  cat "${JOBS_TSV}"
fi

mapfile -t JOB_LINES < <(awk -F $'\t' 'NR>1 {print $0}' "${JOBS_TSV}")
total="${#JOB_LINES[@]}"
fail_count=0

echo "[repair] starting execution (retry-safe mode, sequential)"
if [[ "${MAX_PARALLEL}" != "1" ]]; then
  echo "[repair] note: MAX_PARALLEL=${MAX_PARALLEL} requested, but retry-safe mode currently runs sequentially."
fi

apply_retry_profile() {
  local cfg_in="$1"
  local cfg_out="$2"
  local action="$3"
  local attempt="$4"
  _RR_CFG_IN="${cfg_in}" \
  _RR_CFG_OUT="${cfg_out}" \
  _RR_ACTION="${action}" \
  _RR_ATTEMPT="${attempt}" \
  "${PYTHON_BIN}" - <<'PY'
import os
from pathlib import Path
import yaml

cfg_in = Path(os.environ["_RR_CFG_IN"]).resolve()
cfg_out = Path(os.environ["_RR_CFG_OUT"]).resolve()
action = os.environ["_RR_ACTION"].strip()
attempt = int(os.environ["_RR_ATTEMPT"])

with cfg_in.open("r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}
if not isinstance(cfg, dict):
    raise SystemExit(f"invalid config mapping: {cfg_in}")

# Base safety
cfg["eval_batch"] = 1
cfg["workers"] = 0
cfg["amp"] = False
cfg["cache"] = False

def _to_int(v, d):
    try:
        return int(v)
    except Exception:
        return d

if action in {"resume_finetune", "from_scratch"}:
    cur_batch = _to_int(cfg.get("batch", 8), 8)
    if attempt == 2:
        cfg["batch"] = min(cur_batch, 6)
    else:
        cfg["batch"] = min(cur_batch, 4)
        try:
            imgsz = _to_int(cfg.get("imgsz", 640), 640)
            cfg["imgsz"] = min(imgsz, 512)
        except Exception:
            pass
elif action == "validate_only_low_batch":
    cfg["batch"] = 1
    if attempt >= 3:
        cfg["device"] = "cpu"

cfg_out.parent.mkdir(parents=True, exist_ok=True)
with cfg_out.open("w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
PY
}

for line in "${JOB_LINES[@]}"; do
  IFS=$'\t' read -r run_tag leaf_tag action runtime_cfg log_path cmd <<< "${line}"
  desc="${run_tag}/${leaf_tag}/${action}"
  echo "[repair:start] run=${run_tag} leaf=${leaf_tag} action=${action}"
  echo "[repair:start] cmd=${cmd}"

  base_cfg="${runtime_cfg%.yaml}__base.yaml"
  cp "${runtime_cfg}" "${base_cfg}"

  success="false"
  last_status=0
  last_try_log="${log_path}"

  for attempt in 1 2 3; do
    try_cfg="${runtime_cfg%.yaml}__try${attempt}.yaml"
    try_log="${log_path%.log}.try${attempt}.log"
    last_try_log="${try_log}"

    if [[ "${attempt}" == "1" ]]; then
      cp "${base_cfg}" "${try_cfg}"
    else
      apply_retry_profile "${base_cfg}" "${try_cfg}" "${action}" "${attempt}"
    fi

    echo "[repair:try] ${desc} attempt=${attempt} cfg=${try_cfg}"
    if PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      "${PYTHON_BIN}" "${ROOT_DIR}/src/train.py" --config "${try_cfg}" > "${try_log}" 2>&1; then
      success="true"
      cp "${try_log}" "${log_path}"
      echo "[repair:done] ${desc} attempt=${attempt}"
      break
    else
      last_status=$?
      echo "[repair:retry] ${desc} attempt=${attempt} status=${last_status}"
      cp "${try_log}" "${log_path}"
      if [[ "${attempt}" == "3" ]]; then
        break
      fi
    fi
  done

  if [[ "${success}" != "true" ]]; then
    fail_count=$((fail_count + 1))
    echo "[repair:fail] ${desc} status=${last_status} log=${last_try_log}"
  fi
done

echo "[repair] finished fail_count=${fail_count}"
if [[ "${fail_count}" -gt 0 ]]; then
  exit 1
fi
exit 0
