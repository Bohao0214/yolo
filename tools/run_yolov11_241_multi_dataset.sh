#!/usr/bin/env bash
set -u -o pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_CONFIG_DEFAULT="${ROOT_DIR}/configs/yolo11/enhance241/defect241.yaml"

BASE_CONFIG="${BASE_CONFIG:-${BASE_CONFIG_DEFAULT}}"
RUN_TAG="${RUN_TAG:-multi_dataset_$(date +%y%m%d%H%M%S)}"
RUNNER_MODE="${RUNNER_MODE:-baseline}" # baseline | module_combo
COMBOS_RAW="${COMBOS_RAW:-}"
BATCH_OVERRIDE="${BATCH_OVERRIDE:-}"
EPOCHS_OVERRIDE="${EPOCHS_OVERRIDE:-}"
TRAIN_MODE_OVERRIDE="${TRAIN_MODE_OVERRIDE:-}"
MAX_PARALLEL="${MAX_PARALLEL:-2}"
TMP_CFG_DIR="${TMP_CFG_DIR:-}"
LOG_ROOT="${LOG_ROOT:-}"
PYTHON_CFG_BIN="${PYTHON_CFG_BIN:-${PYTHON_BIN:-python}}"
VRAM_GUARD_OVERRIDE="${VRAM_GUARD_OVERRIDE:-auto}" # auto|on|off
GUARD_MAX_GB_OVERRIDE="${GUARD_MAX_GB_OVERRIDE:-10}"
SAFE_BATCH_OVERRIDE="${SAFE_BATCH_OVERRIDE:-${E241_SAFE_BATCH:-6}}"
SAFE_WORKERS_OVERRIDE="${SAFE_WORKERS_OVERRIDE:-${E241_SAFE_WORKERS:-4}}"
DRY_RUN="false"

DATASET_PATHS=()
RUNNING_PIDS=()

print_pid_snapshot() {
  local stage="${1:-snapshot}"
  local pgid
  pgid="$(ps -o pgid= -p "$$" 2>/dev/null | tr -d '[:space:]' || true)"
  if [[ -n "${RUNNING_PIDS[*]:-}" ]]; then
    echo "[multi-dataset][pid] stage=${stage} self=$$ ppid=${PPID:-NA} pgid=${pgid:-NA} children=${RUNNING_PIDS[*]}"
  else
    echo "[multi-dataset][pid] stage=${stage} self=$$ ppid=${PPID:-NA} pgid=${pgid:-NA} children=(none)"
  fi
}

collect_descendant_pids() {
  local parent_pid="$1"
  local child
  if ! command -v pgrep >/dev/null 2>&1; then
    return 0
  fi
  while read -r child; do
    [[ -z "${child}" ]] && continue
    collect_descendant_pids "${child}"
    echo "${child}"
  done < <(pgrep -P "${parent_pid}" 2>/dev/null || true)
}

kill_pid_tree() {
  local root_pid="$1"
  local descendants
  local p
  descendants="$(collect_descendant_pids "${root_pid}" || true)"
  for p in ${descendants}; do
    kill -TERM "${p}" 2>/dev/null || true
  done
  kill -TERM "${root_pid}" 2>/dev/null || true
}

prune_running_pid() {
  local target="$1"
  local p
  local -a kept=()
  for p in "${RUNNING_PIDS[@]:-}"; do
    [[ "${p}" != "${target}" ]] && kept+=("${p}")
  done
  RUNNING_PIDS=("${kept[@]}")
}

on_interrupt() {
  local sig="${1:-INT}"
  local p
  echo
  echo "[multi-dataset] received ${sig}, terminating running jobs..."
  print_pid_snapshot "before_interrupt_cleanup"
  for p in "${RUNNING_PIDS[@]:-}"; do
    kill_pid_tree "${p}"
  done
  sleep 1
  for p in "${RUNNING_PIDS[@]:-}"; do
    kill -KILL "${p}" 2>/dev/null || true
  done
  exit 130
}

trap 'on_interrupt INT' INT
trap 'on_interrupt TERM' TERM

usage() {
  cat <<'USAGE'
Usage:
  bash tools/run_yolov11_241_multi_dataset.sh [base_config.yaml] [dataset_dir ...] [options]

Purpose:
  Run one or more datasets in parallel using:
  - baseline runner (tools/run_yolov11_241.sh)
  - module combo runner (tools/run_yolov11_241_module_combo.sh)

Dataset inputs:
  --dataset DIR      Add one dataset dir (repeatable)
  --datasets LIST    Comma-separated dataset dirs. Also supports Chinese separator "、".
  Positional dirs are also accepted.

Main options:
  --runner MODE      baseline | module_combo (default baseline)
  --batch N          Override batch size for all datasets
  --epochs N         Override epochs for all datasets
  --train-mode MODE  Override cfg mode (test | train_test | finetune_test)
  --parallel N       Max datasets running at the same time (default 2)
  --tag NAME         Run tag for tmp/log folder naming
  --dry-run          Generate configs and command logs only

Module-combo options (only when --runner module_combo):
  --combos LIST      Pass-through combos list, e.g. baseline,a3,b3,a3+b3

Safety guard options (pass-through):
  --vram-guard MODE  auto|on|off (default auto)
  --guard-max-gb N   default 10
  --safe-batch N     default 6
  --safe-workers N   default 4

Examples:
  # Baseline: run two datasets in parallel
  bash tools/run_yolov11_241_multi_dataset.sh \
    --runner baseline \
    --batch 6 \
    --datasets "/home/ubuntu/hpproject/yolo/dataset/yolo/PKU-Market-PCB(Data enhanced version),/home/ubuntu/hpproject/yolo/dataset/yolo/DeepPCB"

  # Module-combo: run two datasets, each dataset runs the same combo list
  bash tools/run_yolov11_241_multi_dataset.sh \
    --runner module_combo \
    --batch 6 \
    --epochs 150 \
    --combos "baseline,a3,b3,a3+b3" \
    --dataset "/home/ubuntu/hpproject/yolo/dataset/yolo/PKU-Market-PCB(Data enhanced version)" \
    --dataset "/home/ubuntu/hpproject/yolo/dataset/yolo/DeepPCB"

Env overrides:
  BASE_CONFIG, RUN_TAG, RUNNER_MODE, COMBOS_RAW, BATCH_OVERRIDE, EPOCHS_OVERRIDE,
  TRAIN_MODE_OVERRIDE, MAX_PARALLEL, TMP_CFG_DIR, LOG_ROOT, PYTHON_CFG_BIN, PYTHON_BIN,
  VRAM_GUARD_OVERRIDE, GUARD_MAX_GB_OVERRIDE, SAFE_BATCH_OVERRIDE, SAFE_WORKERS_OVERRIDE
USAGE
}

append_datasets_from_csv() {
  local raw="$1"
  local normalized
  local item
  normalized="${raw//、/,}"
  IFS=',' read -r -a _items <<< "${normalized}"
  for item in "${_items[@]}"; do
    item="$(echo "${item}" | sed -E 's/^[[:space:]]+//; s/[[:space:]]+$//')"
    [[ -n "${item}" ]] && DATASET_PATHS+=("${item}")
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
    --dataset)
      [[ $# -ge 2 ]] || { echo "[error] --dataset requires a path" >&2; exit 2; }
      DATASET_PATHS+=("$2")
      shift 2
      ;;
    --datasets)
      [[ $# -ge 2 ]] || { echo "[error] --datasets requires a value" >&2; exit 2; }
      append_datasets_from_csv "$2"
      shift 2
      ;;
    --runner)
      [[ $# -ge 2 ]] || { echo "[error] --runner requires a value (baseline|module_combo)" >&2; exit 2; }
      RUNNER_MODE="$(echo "$2" | tr '[:upper:]' '[:lower:]')"
      shift 2
      ;;
    --batch|--batch-size)
      [[ $# -ge 2 ]] || { echo "[error] --batch requires a value" >&2; exit 2; }
      BATCH_OVERRIDE="$2"
      shift 2
      ;;
    --epoch|--epochs)
      [[ $# -ge 2 ]] || { echo "[error] --epochs requires a value" >&2; exit 2; }
      EPOCHS_OVERRIDE="$2"
      shift 2
      ;;
    --train-mode)
      [[ $# -ge 2 ]] || { echo "[error] --train-mode requires a value" >&2; exit 2; }
      TRAIN_MODE_OVERRIDE="$2"
      shift 2
      ;;
    --parallel)
      [[ $# -ge 2 ]] || { echo "[error] --parallel requires a value" >&2; exit 2; }
      MAX_PARALLEL="$2"
      shift 2
      ;;
    --tag)
      [[ $# -ge 2 ]] || { echo "[error] --tag requires a value" >&2; exit 2; }
      RUN_TAG="$2"
      shift 2
      ;;
    --combos)
      [[ $# -ge 2 ]] || { echo "[error] --combos requires a value" >&2; exit 2; }
      COMBOS_RAW="$2"
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
      DATASET_PATHS+=("$1")
      shift
      ;;
  esac
done

if [[ "${RUNNER_MODE}" != "baseline" && "${RUNNER_MODE}" != "module_combo" ]]; then
  echo "[error] invalid --runner='${RUNNER_MODE}', expected baseline|module_combo" >&2
  exit 2
fi

if [[ "${VRAM_GUARD_OVERRIDE}" != "auto" && "${VRAM_GUARD_OVERRIDE}" != "on" && "${VRAM_GUARD_OVERRIDE}" != "off" ]]; then
  echo "[error] invalid --vram-guard='${VRAM_GUARD_OVERRIDE}', expected auto|on|off" >&2
  exit 2
fi

if ! is_integer "${MAX_PARALLEL}" || [[ "${MAX_PARALLEL}" == "0" ]]; then
  echo "[error] --parallel must be a positive integer, got '${MAX_PARALLEL}'" >&2
  exit 2
fi

if [[ -n "${BATCH_OVERRIDE}" ]] && ! is_integer "${BATCH_OVERRIDE}"; then
  echo "[error] --batch must be an integer, got '${BATCH_OVERRIDE}'" >&2
  exit 2
fi

if [[ -n "${EPOCHS_OVERRIDE}" ]] && ! is_integer "${EPOCHS_OVERRIDE}"; then
  echo "[error] --epochs must be an integer, got '${EPOCHS_OVERRIDE}'" >&2
  exit 2
fi

if [[ ! -f "${BASE_CONFIG}" ]]; then
  echo "[error] base config not found: ${BASE_CONFIG}" >&2
  exit 2
fi

if [[ ${#DATASET_PATHS[@]} -eq 0 ]]; then
  echo "[error] No datasets provided. Use --dataset/--datasets or positional dataset dirs." >&2
  exit 2
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

RUN_TAG_SAFE="$(echo "${RUN_TAG}" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9]+/_/g; s/^_+//; s/_+$//')"
[[ -n "${RUN_TAG_SAFE}" ]] || RUN_TAG_SAFE="multi_dataset"

if [[ -z "${TMP_CFG_DIR}" ]]; then
  TMP_CFG_DIR="/tmp/yolo241_multi_dataset/${RUN_TAG_SAFE}"
fi
if [[ -z "${LOG_ROOT}" ]]; then
  LOG_ROOT="${ROOT_DIR}/experiments/yolo11/multi_dataset_logs/${RUN_TAG_SAFE}"
fi

mkdir -p "${TMP_CFG_DIR}" "${LOG_ROOT}"

BASE_META="$(
  _M_BASE_CONFIG="${BASE_CONFIG}" "${PYTHON_CFG_BIN}" - <<'PY'
import os
from pathlib import Path

import yaml

cfg_path = Path(os.environ["_M_BASE_CONFIG"]).resolve()
with cfg_path.open("r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}
if not isinstance(cfg, dict):
    raise SystemExit(f"Base config must be mapping: {cfg_path}")

epochs = int(cfg.get("epochs", 150))
batch = int(cfg.get("batch", 6))
mode = str(cfg.get("mode", "train_test"))
print(f"{epochs}\t{batch}\t{mode}")
PY
)"

BASE_EPOCHS="$(echo "${BASE_META}" | awk -F $'\t' '{print $1}')"
BASE_BATCH="$(echo "${BASE_META}" | awk -F $'\t' '{print $2}')"
BASE_TRAIN_MODE="$(echo "${BASE_META}" | awk -F $'\t' '{print $3}')"

RUN_EPOCHS="${EPOCHS_OVERRIDE:-${BASE_EPOCHS}}"
RUN_BATCH="${BATCH_OVERRIDE:-${BASE_BATCH}}"
RUN_TRAIN_MODE="${TRAIN_MODE_OVERRIDE:-${BASE_TRAIN_MODE}}"

normalize_tag() {
  local raw="$1"
  local base
  base="$(basename "${raw}")"
  base="$(echo "${base}" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9]+/_/g; s/^_+//; s/_+$//')"
  [[ -n "${base}" ]] || base="dataset"
  echo "${base}"
}

generate_dataset_bundle() {
  local dataset_root="$1"
  local dataset_tag="$2"
  local out_data_yaml="$3"
  local out_cfg_yaml="$4"

  _M_ROOT_DIR="${ROOT_DIR}" \
  _M_BASE_CONFIG="${BASE_CONFIG}" \
  _M_DATASET_ROOT="${dataset_root}" \
  _M_DATASET_TAG="${dataset_tag}" \
  _M_DATA_OUT="${out_data_yaml}" \
  _M_CFG_OUT="${out_cfg_yaml}" \
  _M_RUN_TAG="${RUN_TAG_SAFE}" \
  _M_BATCH_OVERRIDE="${BATCH_OVERRIDE}" \
  _M_EPOCHS_OVERRIDE="${EPOCHS_OVERRIDE}" \
  _M_TRAIN_MODE_OVERRIDE="${TRAIN_MODE_OVERRIDE}" \
  "${PYTHON_CFG_BIN}" - <<'PY'
import ast
import os
from pathlib import Path
from typing import List

import yaml


def load_yaml(path: Path):
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise SystemExit(f"YAML must be mapping: {path}")
    return data


def parse_names(path: Path) -> List[str]:
    text = path.read_text(encoding="utf-8", errors="ignore").strip()
    if not text:
        return []
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    parts: List[str] = []
    if len(lines) > 1:
        parts = lines
    else:
        line = lines[0]
        if line.startswith("[") and line.endswith("]"):
            try:
                obj = ast.literal_eval(line)
                if isinstance(obj, (list, tuple)):
                    return [str(x).strip() for x in obj if str(x).strip()]
            except Exception:
                pass
        if "," in line:
            parts = [x.strip() for x in line.split(",") if x.strip()]
        else:
            parts = [line]
    cleaned = [x.strip().strip("'\"") for x in parts if x.strip().strip("'\"")]
    return cleaned


def parse_names_obj(obj) -> List[str]:
    if isinstance(obj, (list, tuple)):
        return [str(x).strip() for x in obj if str(x).strip()]
    if isinstance(obj, dict):
        items = []
        for k, v in obj.items():
            key_s = str(k).strip()
            val_s = str(v).strip()
            if not val_s:
                continue
            try:
                key_i = int(float(key_s))
            except Exception:
                key_i = 10**9
            items.append((key_i, key_s, val_s))
        items.sort(key=lambda x: (x[0], x[1]))
        return [v for _, _, v in items]
    if isinstance(obj, str):
        s = obj.strip()
        if not s:
            return []
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, (list, tuple, dict)):
                return parse_names_obj(parsed)
        except Exception:
            pass
        if "," in s:
            return [x.strip().strip("'\"") for x in s.split(",") if x.strip().strip("'\"")]
        return [s.strip().strip("'\"")]
    return []


def pick_existing(root: Path, candidates: List[str]) -> str:
    for rel in candidates:
        if (root / rel).exists():
            return rel
    return ""


def detect_splits(root: Path):
    train_entry = pick_existing(root, ["train/images", "images/train", "train", "images"])
    if not train_entry:
        raise SystemExit(
            "Cannot detect train split. Expected one of "
            "train/images, images/train, train, images under dataset root."
        )

    if train_entry == "train/images":
        val_entry = pick_existing(root, ["valid/images", "val/images", "train/images"]) or "train/images"
        test_entry = pick_existing(root, ["test/images", "valid/images"])
    elif train_entry == "images/train":
        val_entry = pick_existing(root, ["images/val", "images/valid", "images/train"]) or "images/train"
        test_entry = pick_existing(root, ["images/test", "images/val"])
    elif train_entry == "train":
        val_entry = pick_existing(root, ["valid", "val", "train"]) or "train"
        test_entry = pick_existing(root, ["test", "valid"])
    else:
        val_entry = pick_existing(root, ["images/val", "val", "valid", "images"]) or "images"
        test_entry = pick_existing(root, ["images/test", "test"])

    return train_entry, val_entry, test_entry


def image_entry_to_label_dir(entry: str) -> Path:
    p = Path(entry)
    parts = list(p.parts)
    for i, part in enumerate(parts):
        if part.lower() in {"images", "image"}:
            parts[i] = "labels"
            return Path(*parts)
    return Path("labels") / p


def collect_label_dirs(root: Path, train_entry: str, val_entry: str, test_entry: str) -> List[Path]:
    rel_dirs = [
        image_entry_to_label_dir(train_entry),
        image_entry_to_label_dir(val_entry),
    ]
    if test_entry:
        rel_dirs.append(image_entry_to_label_dir(test_entry))
    rel_dirs.extend(
        [
            Path("labels"),
            Path("label"),
            Path("lable"),
            Path("train/labels"),
            Path("valid/labels"),
            Path("val/labels"),
            Path("test/labels"),
        ]
    )

    out: List[Path] = []
    seen = set()
    for rel in rel_dirs:
        abs_dir = (root / rel).resolve()
        key = str(abs_dir)
        if key in seen:
            continue
        seen.add(key)
        if abs_dir.is_dir():
            out.append(abs_dir)
    return out


def infer_nc_from_labels(root: Path, train_entry: str, val_entry: str, test_entry: str):
    label_dirs = collect_label_dirs(root, train_entry, val_entry, test_entry)
    max_cls_first = -1
    max_cls_last = -1
    first_hits = 0
    last_hits = 0
    fallback_max = -1

    def is_int_like(v: float) -> bool:
        return v >= 0 and abs(v - round(v)) < 1e-9

    for label_dir in label_dirs:
        for txt in label_dir.rglob("*.txt"):
            if txt.name.lower() in {"class.txt", "classes.txt", "names.txt", "labels.txt"}:
                continue
            try:
                lines = txt.read_text(encoding="utf-8", errors="ignore").splitlines()
            except Exception:
                continue
            for line in lines:
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                parts = s.split()
                if not parts:
                    continue
                vals = []
                try:
                    vals = [float(x) for x in parts]
                except Exception:
                    continue
                if not vals:
                    continue

                first = vals[0]
                last = vals[-1]
                first_ok = is_int_like(first)
                last_ok = is_int_like(last)
                if first_ok:
                    fcls = int(round(first))
                    if fcls > fallback_max:
                        fallback_max = fcls

                used = False
                if len(vals) >= 5:
                    first_style = first_ok and all(0.0 <= v <= 1.0 for v in vals[1:5])
                    last_style = last_ok and any(v > 1.0 for v in vals[:4])
                    if first_style:
                        first_hits += 1
                        if int(round(first)) > max_cls_first:
                            max_cls_first = int(round(first))
                        used = True
                    if last_style:
                        last_hits += 1
                        if int(round(last)) > max_cls_last:
                            max_cls_last = int(round(last))
                        used = True

                if not used:
                    if first_ok and not last_ok:
                        first_hits += 1
                        if int(round(first)) > max_cls_first:
                            max_cls_first = int(round(first))
                    elif last_ok and not first_ok:
                        last_hits += 1
                        if int(round(last)) > max_cls_last:
                            max_cls_last = int(round(last))

    if first_hits == 0 and last_hits == 0:
        max_cls = fallback_max
    elif last_hits > first_hits:
        max_cls = max_cls_last
    else:
        max_cls = max_cls_first
    return (max_cls + 1) if max_cls >= 0 else None


root_dir = Path(os.environ["_M_ROOT_DIR"]).resolve()
base_cfg_path = Path(os.environ["_M_BASE_CONFIG"]).resolve()
dataset_root = Path(os.environ["_M_DATASET_ROOT"]).resolve()
dataset_tag = os.environ["_M_DATASET_TAG"].strip()
out_data = Path(os.environ["_M_DATA_OUT"]).resolve()
out_cfg = Path(os.environ["_M_CFG_OUT"]).resolve()
run_tag = os.environ["_M_RUN_TAG"].strip()
batch_override = os.environ["_M_BATCH_OVERRIDE"].strip()
epochs_override = os.environ["_M_EPOCHS_OVERRIDE"].strip()
train_mode_override = os.environ["_M_TRAIN_MODE_OVERRIDE"].strip()

if not dataset_root.is_dir():
    raise SystemExit(f"Dataset root not found: {dataset_root}")

cfg = load_yaml(base_cfg_path)
data_ref = str(cfg.get("data", "configs/data/defect.yaml"))
data_path = Path(data_ref)
if not data_path.is_absolute():
    data_path = (root_dir / data_path).resolve()

dataset_meta_info = {}
for _cand in ("data.yaml", "data.yml", "dataset.yaml", "dataset.yml"):
    _cand_path = dataset_root / _cand
    if _cand_path.is_file():
        try:
            _doc = load_yaml(_cand_path)
            if isinstance(_doc, dict):
                dataset_meta_info = _doc
                break
        except Exception:
            pass

default_nc = 1
default_names: List[str] = ["defect"]
if dataset_meta_info:
    names_obj = dataset_meta_info.get("names", [])
    parsed_names = parse_names_obj(names_obj)
    if parsed_names:
        default_names = parsed_names
    if default_names:
        default_nc = len(default_names)
    try:
        default_nc = int(dataset_meta_info.get("nc", default_nc))
    except Exception:
        pass
elif data_path.exists():
    data_info = load_yaml(data_path)
    names_obj = data_info.get("names", [])
    if isinstance(names_obj, (list, tuple)):
        default_names = [str(x) for x in names_obj if str(x).strip()]
    if default_names:
        default_nc = len(default_names)
    try:
        default_nc = int(data_info.get("nc", default_nc))
    except Exception:
        pass

train_entry, val_entry, test_entry = detect_splits(dataset_root)
if dataset_meta_info:
    meta_train = str(dataset_meta_info.get("train", "")).strip()
    meta_val = str(dataset_meta_info.get("val", dataset_meta_info.get("valid", ""))).strip()
    meta_test = str(dataset_meta_info.get("test", "")).strip()

    def _entry_exists(entry: str) -> bool:
        if not entry:
            return False
        p = Path(entry)
        if p.is_absolute():
            return p.exists()
        return (dataset_root / p).exists()

    if _entry_exists(meta_train):
        train_entry = meta_train
    if _entry_exists(meta_val):
        val_entry = meta_val
    if _entry_exists(meta_test):
        test_entry = meta_test

inferred_nc = infer_nc_from_labels(dataset_root, train_entry, val_entry, test_entry)

names = []
names_from_file = False
for candidate in ("Class.txt", "classes.txt", "class.txt", "names.txt", "labels.txt"):
    cand_path = dataset_root / candidate
    if cand_path.exists():
        parsed = parse_names(cand_path)
        if parsed:
            names = parsed
            names_from_file = True
            break

if not names:
    if inferred_nc and inferred_nc > 0:
        use_default_names = (
            len(default_names) >= inferred_nc
            and not (len(default_names) == 1 and default_names[0].strip().lower() in {"defect", "object"} and inferred_nc > 1)
        )
        if use_default_names:
            names = list(default_names[:inferred_nc])
        else:
            names = [f"class_{i}" for i in range(inferred_nc)]
    else:
        names = list(default_names) if default_names else [f"class_{i}" for i in range(default_nc)]

if names_from_file:
    nc = max(int(default_nc), len(names))
elif inferred_nc and inferred_nc > 0:
    nc = max(int(default_nc), len(names), int(inferred_nc))
else:
    nc = max(int(default_nc), len(names))
if len(names) < nc:
    names = names + [f"class_{i}" for i in range(len(names), nc)]

data_doc = {
    "path": str(dataset_root),
    "train": train_entry,
    "val": val_entry,
    "nc": int(nc),
    "names": names,
}
if test_entry:
    data_doc["test"] = test_entry

out_data.parent.mkdir(parents=True, exist_ok=True)
with out_data.open("w", encoding="utf-8") as f:
    yaml.safe_dump(data_doc, f, sort_keys=False, allow_unicode=True)

exp_base = str(cfg.get("exp_name", "defect241"))
suffix = f"__{run_tag}__ds_{dataset_tag}"
if not exp_base.endswith(suffix):
    cfg["exp_name"] = exp_base + suffix

cfg["data"] = str(out_data)
cfg["data_root"] = str(dataset_root)

if batch_override:
    cfg["batch"] = int(batch_override)
if epochs_override:
    cfg["epochs"] = int(epochs_override)
if train_mode_override:
    cfg["mode"] = train_mode_override

out_cfg.parent.mkdir(parents=True, exist_ok=True)
with out_cfg.open("w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

print(f"{train_entry}\t{val_entry}\t{test_entry}\t{nc}\t{cfg['exp_name']}")
PY
}

build_cmd_for_index() {
  local idx="$1"
  CMD=()
  if [[ "${RUNNER_MODE}" == "baseline" ]]; then
    CMD=(
      bash "${ROOT_DIR}/tools/run_yolov11_241.sh"
      --vram-guard "${VRAM_GUARD_OVERRIDE}"
      --guard-max-gb "${GUARD_MAX_GB_OVERRIDE}"
      --safe-batch "${SAFE_BATCH_OVERRIDE}"
      --safe-workers "${SAFE_WORKERS_OVERRIDE}"
      "${CFG_PATHS[$idx]}"
    )
  else
    CMD=(
      bash "${ROOT_DIR}/tools/run_yolov11_241_module_combo.sh"
      "${CFG_PATHS[$idx]}"
      --epochs "${RUN_EPOCHS}"
      --batch "${RUN_BATCH}"
      --tag "${RUN_TAG_SAFE}__${DATASET_TAGS[$idx]}"
      --vram-guard "${VRAM_GUARD_OVERRIDE}"
      --guard-max-gb "${GUARD_MAX_GB_OVERRIDE}"
      --safe-batch "${SAFE_BATCH_OVERRIDE}"
      --safe-workers "${SAFE_WORKERS_OVERRIDE}"
    )
    [[ -n "${COMBOS_RAW}" ]] && CMD+=(--combos "${COMBOS_RAW}")
  fi
}

resolve_abs_dir() {
  local input="$1"
  (cd "${input}" >/dev/null 2>&1 && pwd) || return 1
}

DATASET_ROOTS=()
DATASET_TAGS=()
CFG_PATHS=()
DATA_YAMLS=()
TRAIN_ENTRIES=()
VAL_ENTRIES=()
TEST_ENTRIES=()
CLASS_COUNTS=()

declare -A TAG_SEEN=()

for ds in "${DATASET_PATHS[@]}"; do
  abs_ds="$(resolve_abs_dir "${ds}")" || {
    echo "[error] dataset dir not found or not accessible: ${ds}" >&2
    exit 2
  }
  base_tag="$(normalize_tag "${abs_ds}")"
  n="${TAG_SEEN[$base_tag]:-0}"
  TAG_SEEN["$base_tag"]=$((n + 1))
  if [[ "${n}" == "0" ]]; then
    ds_tag="${base_tag}"
  else
    ds_tag="${base_tag}_${n}"
  fi

  case_dir="${TMP_CFG_DIR}/${ds_tag}"
  data_yaml="${case_dir}/data.yaml"
  cfg_yaml="${case_dir}/config.yaml"

  meta="$(generate_dataset_bundle "${abs_ds}" "${ds_tag}" "${data_yaml}" "${cfg_yaml}")" || {
    echo "[error] failed to generate temp config for dataset: ${abs_ds}" >&2
    exit 3
  }
  train_entry="$(echo "${meta}" | awk -F $'\t' '{print $1}')"
  val_entry="$(echo "${meta}" | awk -F $'\t' '{print $2}')"
  test_entry="$(echo "${meta}" | awk -F $'\t' '{print $3}')"
  class_count="$(echo "${meta}" | awk -F $'\t' '{print $4}')"

  DATASET_ROOTS+=("${abs_ds}")
  DATASET_TAGS+=("${ds_tag}")
  CFG_PATHS+=("${cfg_yaml}")
  DATA_YAMLS+=("${data_yaml}")
  TRAIN_ENTRIES+=("${train_entry}")
  VAL_ENTRIES+=("${val_entry}")
  TEST_ENTRIES+=("${test_entry}")
  CLASS_COUNTS+=("${class_count}")
done

SUMMARY_TSV="${LOG_ROOT}/summary.tsv"
{
  echo -e "dataset_tag\tdataset_root\trunner\tstatus\ttrain\tval\ttest\tnc\tconfig\tdata_yaml\tlog"
} > "${SUMMARY_TSV}"

echo "[multi-dataset] base_config=${BASE_CONFIG}"
echo "[multi-dataset] runner=${RUNNER_MODE} dry_run=${DRY_RUN} parallel=${MAX_PARALLEL}"
echo "[multi-dataset] epochs=${RUN_EPOCHS} batch=${RUN_BATCH} train_mode=${RUN_TRAIN_MODE}"
echo "[multi-dataset] guard mode=${VRAM_GUARD_OVERRIDE} max_gb=${GUARD_MAX_GB_OVERRIDE} safe_batch=${SAFE_BATCH_OVERRIDE} safe_workers=${SAFE_WORKERS_OVERRIDE}"
echo "[multi-dataset] run_tag=${RUN_TAG_SAFE}"
echo "[multi-dataset] tmp_cfg_dir=${TMP_CFG_DIR}"
echo "[multi-dataset] log_root=${LOG_ROOT}"
echo "[multi-dataset] datasets=${#DATASET_ROOTS[@]}"

fail_count=0

if [[ "${DRY_RUN}" == "true" ]]; then
  for i in "${!DATASET_ROOTS[@]}"; do
    log_path="${LOG_ROOT}/${DATASET_TAGS[$i]}.log"
    build_cmd_for_index "${i}"
    {
      printf "[dry-run] dataset=%s\n" "${DATASET_ROOTS[$i]}"
      printf "[dry-run] config=%s\n" "${CFG_PATHS[$i]}"
      printf "[dry-run] data_yaml=%s\n" "${DATA_YAMLS[$i]}"
      printf "[dry-run] cmd:"
      printf " %q" "${CMD[@]}"
      printf "\n"
    } > "${log_path}"

    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
      "${DATASET_TAGS[$i]}" "${DATASET_ROOTS[$i]}" "${RUNNER_MODE}" "dry_run" \
      "${TRAIN_ENTRIES[$i]}" "${VAL_ENTRIES[$i]}" "${TEST_ENTRIES[$i]}" "${CLASS_COUNTS[$i]}" \
      "${CFG_PATHS[$i]}" "${DATA_YAMLS[$i]}" "${log_path}" \
      >> "${SUMMARY_TSV}"
  done
else
  total="${#DATASET_ROOTS[@]}"
  idx=0
  while [[ "${idx}" -lt "${total}" ]]; do
    chunk_start="${idx}"
    chunk_end=$((chunk_start + MAX_PARALLEL))
    [[ "${chunk_end}" -gt "${total}" ]] && chunk_end="${total}"

    pids=()
    declare -A PID_TO_INDEX=()

    for ((j=chunk_start; j<chunk_end; j++)); do
      build_cmd_for_index "${j}"
      log_path="${LOG_ROOT}/${DATASET_TAGS[$j]}.log"

      echo "[multi-dataset:${DATASET_TAGS[$j]}] dataset=${DATASET_ROOTS[$j]}"
      echo "[multi-dataset:${DATASET_TAGS[$j]}] cmd=${CMD[*]}"
      echo "[multi-dataset:${DATASET_TAGS[$j]}] data_yaml=${DATA_YAMLS[$j]} nc=${CLASS_COUNTS[$j]}"

      if [[ "${RUNNER_MODE}" == "module_combo" ]]; then
        FORCE_MODE="${RUN_TRAIN_MODE}" "${CMD[@]}" > "${log_path}" 2>&1 &
      else
        "${CMD[@]}" > "${log_path}" 2>&1 &
      fi
      pid=$!
      pids+=("${pid}")
      RUNNING_PIDS+=("${pid}")
      PID_TO_INDEX["${pid}"]="${j}"
      echo "[multi-dataset:${DATASET_TAGS[$j]}] started pid=${pid} log=${log_path}"
      echo "[multi-dataset][train-pid] dataset=${DATASET_TAGS[$j]} pid=${pid} runner=${RUNNER_MODE} log=${log_path}"
    done

    for pid in "${pids[@]}"; do
      status=0
      wait "${pid}" || status=$?
      prune_running_pid "${pid}"
      j="${PID_TO_INDEX[$pid]}"
      log_path="${LOG_ROOT}/${DATASET_TAGS[$j]}.log"

      if [[ "${status}" != "0" ]]; then
        fail_count=$((fail_count + 1))
        echo "[multi-dataset:${DATASET_TAGS[$j]}] failed status=${status} log=${log_path}"
      else
        echo "[multi-dataset:${DATASET_TAGS[$j]}] success log=${log_path}"
      fi

      printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "${DATASET_TAGS[$j]}" "${DATASET_ROOTS[$j]}" "${RUNNER_MODE}" "${status}" \
        "${TRAIN_ENTRIES[$j]}" "${VAL_ENTRIES[$j]}" "${TEST_ENTRIES[$j]}" "${CLASS_COUNTS[$j]}" \
        "${CFG_PATHS[$j]}" "${DATA_YAMLS[$j]}" "${log_path}" \
        >> "${SUMMARY_TSV}"
    done

    idx="${chunk_end}"
  done
fi

echo
echo "[multi-dataset] finished fail_count=${fail_count}"
echo "[multi-dataset] summary=${SUMMARY_TSV}"
column -s $'\t' -t "${SUMMARY_TSV}" || cat "${SUMMARY_TSV}"

if [[ "${DRY_RUN}" == "true" ]]; then
  exit 0
fi

if [[ "${fail_count}" -gt 0 ]]; then
  exit 1
fi
exit 0
