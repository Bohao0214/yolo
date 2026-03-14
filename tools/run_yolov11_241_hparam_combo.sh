#!/usr/bin/env bash
set -u -o pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_CONFIG_DEFAULT="${ROOT_DIR}/configs/yolo11/enhance241/defect241.yaml"

BASE_CONFIG="${BASE_CONFIG:-${BASE_CONFIG_DEFAULT}}"
SWEEP_TAG="${SWEEP_TAG:-hparam_combo_$(date +%y%m%d%H%M%S)}"
MODULES_RAW="${MODULES_RAW:-a3+c5}"
HPARAMS_RAW="${HPARAMS_RAW:-}"
GRID_RAW="${GRID_RAW:-}"
MAX_CASES_OVERRIDE="${MAX_CASES_OVERRIDE:-256}"
FORCE_MODE="${FORCE_MODE:-train_test}"
SEED_OVERRIDE="${SEED_OVERRIDE:-}"
D1_WORKERS_OVERRIDE="${D1_WORKERS_OVERRIDE:-4}"
VRAM_GUARD_OVERRIDE="${VRAM_GUARD_OVERRIDE:-auto}" # auto|on|off
GUARD_MAX_GB_OVERRIDE="${GUARD_MAX_GB_OVERRIDE:-10}"
SAFE_BATCH_OVERRIDE="${SAFE_BATCH_OVERRIDE:-${E241_SAFE_BATCH:-6}}"
SAFE_WORKERS_OVERRIDE="${SAFE_WORKERS_OVERRIDE:-${E241_SAFE_WORKERS:-4}}"
TMP_CFG_DIR="${TMP_CFG_DIR:-}"
LOG_ROOT="${LOG_ROOT:-}"
PYTHON_CFG_BIN="${PYTHON_CFG_BIN:-${PYTHON_BIN:-python}}"
DRY_RUN="false"
ACTIVE_CHILD_PID=""

print_pid_snapshot() {
  local stage="${1:-snapshot}"
  local pgid
  pgid="$(ps -o pgid= -p "$$" 2>/dev/null | tr -d '[:space:]' || true)"
  if [[ -n "${ACTIVE_CHILD_PID:-}" ]]; then
    echo "[hparam-combo][pid] stage=${stage} self=$$ ppid=${PPID:-NA} pgid=${pgid:-NA} child=${ACTIVE_CHILD_PID}"
  else
    echo "[hparam-combo][pid] stage=${stage} self=$$ ppid=${PPID:-NA} pgid=${pgid:-NA} child=(none)"
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

on_interrupt() {
  local sig="${1:-INT}"
  echo
  echo "[hparam-combo] received ${sig}, terminating current child..."
  print_pid_snapshot "before_interrupt_cleanup"
  if [[ -n "${ACTIVE_CHILD_PID:-}" ]]; then
    kill_pid_tree "${ACTIVE_CHILD_PID}"
    sleep 1
    kill -KILL "${ACTIVE_CHILD_PID}" 2>/dev/null || true
  fi
  exit 130
}

trap 'on_interrupt INT' INT
trap 'on_interrupt TERM' TERM

usage() {
  cat <<'USAGE'
Usage:
  bash tools/run_yolov11_241_hparam_combo.sh [base_config.yaml] [--modules EXPR] [--hparams CASES | --grid GRID] [--epochs N] [--seed N] [--tag NAME] [--vram-guard auto|on|off] [--guard-max-gb N] [--safe-batch N] [--safe-workers N] [--dry-run]

Purpose:
  Fix one module combination (for example a3+c5), then run multiple hyperparameter
  combinations for vertical batch experiments.

Hyperparameters supported in each case:
  epochs, patience, batch, grad_accum, lr0, lrf, warmup_epochs, best_select_metric

Input forms (choose one):
  1) --hparams CASES
     CASES format: case1|case2|...
     each case format: key=value,key=value,...

     Example:
       --hparams "epochs=150,patience=0,batch=6,grad_accum=1,lr0=0.012,lrf=0.12,warmup_epochs=0,best_select_metric=iFN|epochs=180,patience=20,batch=6,grad_accum=1,lr0=0.01,lrf=0.1,warmup_epochs=2,best_select_metric=iAUROC@fpr0.5"

  2) --grid GRID
     GRID format: key=v1,v2;key=v1,v2;...
     script expands cartesian product.

     Example:
       --grid "epochs=120,150;patience=0,20;batch=6;grad_accum=1,2;lr0=0.010,0.012;lrf=0.10,0.12;warmup_epochs=0,2;best_select_metric=iFN,iAUROC@fpr0.5"

Module expression:
  --modules supports baseline, hmc7/abcd7, pdd9/abcd9, and raw combinations like
  a3+c5, a3_c5, a3,c5 (comma is treated as plus).

Options:
  --modules EXPR  Fixed module combination (default a3+c5)
  --hparams CASES Explicit hyperparameter cases (mutually exclusive with --grid)
  --grid GRID     Hyperparameter grid, expanded to cases (mutually exclusive with --hparams)
  --max-cases N   Safety cap for generated case count (default 256)
  --epoch N       Shortcut: override epochs for all generated cases
  --epochs N      Same as --epoch
  --seed N        Override seed for all cases
  --tag NAME      Sweep tag (for tmp/log directory names)
  --d1-workers N  Worker override when D-branch modules are enabled (default 4)
  --vram-guard MODE
                  Pass-through to run_yolov11_241.sh: auto|on|off (default auto)
  --guard-max-gb N
                  Pass-through to run_yolov11_241.sh auto threshold (default 10)
  --safe-batch N
                  Pass-through safe fallback batch when guard applies (default 6)
  --safe-workers N
                  Pass-through worker cap for non-low-VRAM path (default 4)
  --dry-run       Generate configs and commands only (no training)
  -h, --help      Show help

Notes:
  - grad_accum is recorded in generated config and summary for traceability.
  - current src/train.py may not consume grad_accum directly.
  - output layout: experiments/<yolo_version>/<exp_group_hp>/<exp_group_hp__case_tag>/exp_*

Env overrides:
  BASE_CONFIG, SWEEP_TAG, MODULES_RAW, HPARAMS_RAW, GRID_RAW, MAX_CASES_OVERRIDE
  FORCE_MODE, SEED_OVERRIDE, D1_WORKERS_OVERRIDE
  VRAM_GUARD_OVERRIDE, GUARD_MAX_GB_OVERRIDE, SAFE_BATCH_OVERRIDE, SAFE_WORKERS_OVERRIDE
  TMP_CFG_DIR, LOG_ROOT, PYTHON_CFG_BIN, PYTHON_BIN
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --modules)
      [[ $# -ge 2 ]] || { echo "[error] --modules requires a value" >&2; exit 2; }
      MODULES_RAW="$2"
      shift 2
      ;;
    --hparams)
      [[ $# -ge 2 ]] || { echo "[error] --hparams requires a value" >&2; exit 2; }
      HPARAMS_RAW="$2"
      shift 2
      ;;
    --grid)
      [[ $# -ge 2 ]] || { echo "[error] --grid requires a value" >&2; exit 2; }
      GRID_RAW="$2"
      shift 2
      ;;
    --max-cases)
      [[ $# -ge 2 ]] || { echo "[error] --max-cases requires a value" >&2; exit 2; }
      MAX_CASES_OVERRIDE="$2"
      shift 2
      ;;
    --epoch|--epochs)
      [[ $# -ge 2 ]] || { echo "[error] --epochs requires a value" >&2; exit 2; }
      EPOCHS_OVERRIDE="$2"
      shift 2
      ;;
    --seed)
      [[ $# -ge 2 ]] || { echo "[error] --seed requires a value" >&2; exit 2; }
      SEED_OVERRIDE="$2"
      shift 2
      ;;
    --tag)
      [[ $# -ge 2 ]] || { echo "[error] --tag requires a value" >&2; exit 2; }
      SWEEP_TAG="$2"
      shift 2
      ;;
    --d1-workers)
      [[ $# -ge 2 ]] || { echo "[error] --d1-workers requires a value" >&2; exit 2; }
      D1_WORKERS_OVERRIDE="$2"
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

if [[ -n "${HPARAMS_RAW}" && -n "${GRID_RAW}" ]]; then
  echo "[error] --hparams and --grid are mutually exclusive" >&2
  exit 2
fi

if [[ ! -f "${BASE_CONFIG}" ]]; then
  echo "[error] base config not found: ${BASE_CONFIG}" >&2
  exit 2
fi

if [[ "${VRAM_GUARD_OVERRIDE}" != "auto" && "${VRAM_GUARD_OVERRIDE}" != "on" && "${VRAM_GUARD_OVERRIDE}" != "off" ]]; then
  echo "[error] invalid --vram-guard='${VRAM_GUARD_OVERRIDE}', expected auto|on|off" >&2
  exit 2
fi

if [[ -z "${TMP_CFG_DIR}" ]]; then
  TMP_CFG_DIR="/tmp/yolo241_hparam_combo/${SWEEP_TAG}"
fi
if [[ -z "${LOG_ROOT}" ]]; then
  LOG_ROOT="${ROOT_DIR}/experiments/yolo11/hparam_combo_logs/${SWEEP_TAG}"
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

MODULE_TAG=""
MODULE_SWITCHES=""
resolve_modules() {
  local raw="$1"
  local token
  token="$(echo "${raw}" | tr '[:upper:]' '[:lower:]')"
  token="${token//[[:space:]]/}"

  if [[ -z "${token}" ]]; then
    MODULE_TAG="baseline"
    MODULE_SWITCHES=""
    return
  fi

  case "${token}" in
    baseline|base|none)
      MODULE_TAG="baseline"
      MODULE_SWITCHES=""
      return
      ;;
    hmc7|abcd7|a7_b7_c7_d7|a7+b7+c7+d7)
      MODULE_TAG="a7_b7_c7_d7"
      MODULE_SWITCHES="a7 b7 c7 d7"
      return
      ;;
    pdd9|abcd9|a9_b9_c9_d9|a9+b9+c9+d9)
      MODULE_TAG="a9_b9_c9_d9"
      MODULE_SWITCHES="a9 b9 c9 d9"
      return
      ;;
  esac

  local expr="${token//_/+}"
  expr="${expr//,/+}"

  local part
  local -a parts=()
  local -a switches=()
  declare -A seen=()
  IFS='+' read -r -a parts <<< "${expr}"
  for part in "${parts[@]}"; do
    [[ -z "${part}" ]] && continue
    case "${part}" in
      baseline|base|none)
        continue
        ;;
      a3|a5|a7|a9|b1|b2|b3|b5|b7|b9|c5|c7|c9|d1|d3|d5|d7|d9)
        ;;
      *)
        echo "[error] Unsupported module token '${part}' in '${raw}'" >&2
        echo "        allowed: baseline,hmc7/pdd9,a3,a5,a7,a9,b1,b2,b3,b5,b7,b9,c5,c7,c9,d1,d3,d5,d7,d9" >&2
        exit 2
        ;;
    esac
    if [[ -z "${seen[$part]+x}" ]]; then
      seen["$part"]=1
      switches+=("${part}")
    fi
  done

  if [[ ${#switches[@]} -eq 0 ]]; then
    MODULE_TAG="baseline"
    MODULE_SWITCHES=""
    return
  fi

  MODULE_TAG="$(IFS=_; echo "${switches[*]}")"
  MODULE_SWITCHES="$(IFS=' '; echo "${switches[*]}")"
}

resolve_modules "${MODULES_RAW}"

build_hparam_cases() {
  _HP_BASE_CONFIG="${BASE_CONFIG}" \
  _HP_HPARAMS_RAW="${HPARAMS_RAW}" \
  _HP_GRID_RAW="${GRID_RAW}" \
  _HP_MAX_CASES="${MAX_CASES_OVERRIDE}" \
  _HP_EPOCHS_OVERRIDE="${EPOCHS_OVERRIDE:-}" \
  "${PYTHON_CFG_BIN}" - <<'PY'
import itertools
import json
import os
from pathlib import Path

import yaml

INT_KEYS = ("epochs", "patience", "batch", "grad_accum")
FLOAT_KEYS = ("lr0", "lrf", "warmup_epochs")
STR_KEYS = ("best_select_metric",)
ALL_KEYS = INT_KEYS + FLOAT_KEYS + STR_KEYS

base_cfg = Path(os.environ["_HP_BASE_CONFIG"]).resolve()
hparams_raw = os.environ.get("_HP_HPARAMS_RAW", "").strip()
grid_raw = os.environ.get("_HP_GRID_RAW", "").strip()
max_cases = int(os.environ.get("_HP_MAX_CASES", "256"))
epochs_override = os.environ.get("_HP_EPOCHS_OVERRIDE", "").strip()

with base_cfg.open("r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}

if not isinstance(cfg, dict):
    raise SystemExit(f"Base config must be mapping: {base_cfg}")

def parse_value(key: str, raw_value: str):
    v = raw_value.strip()
    if key in INT_KEYS:
        return int(v)
    if key in FLOAT_KEYS:
        return float(v)
    if key in STR_KEYS:
        vl = v.lower()
        if vl not in {"fitness", "map", "ifn", "iauroc@fpr0.5", "default", "默认"}:
            raise SystemExit(f"Unsupported best_select_metric '{v}', expected: fitness(default/默认) | iFN | iAUROC@fpr0.5")
        if vl in {"fitness", "map", "default", "默认"}:
            return "fitness"
        if vl == "ifn":
            return "iFN"
        return "iAUROC@fpr0.5"
    raise SystemExit(f"Unsupported key: {key}")

def parse_case(case_raw: str):
    out = {}
    tokens = [t.strip() for t in case_raw.split(",") if t.strip()]
    if not tokens:
        raise SystemExit(f"Empty case in --hparams: {case_raw!r}")
    for tok in tokens:
        if "=" not in tok:
            raise SystemExit(f"Invalid token '{tok}', expected key=value")
        key, value = tok.split("=", 1)
        key = key.strip().lower()
        if key not in ALL_KEYS:
            raise SystemExit(
                f"Unsupported hyperparameter '{key}' in '{tok}'. "
                f"Allowed: {', '.join(ALL_KEYS)}"
            )
        out[key] = parse_value(key, value)
    return out

def normalize_float(v: float) -> str:
    text = f"{v:g}"
    if text == "-0":
        return "0"
    return text

def slug_num(v) -> str:
    if isinstance(v, str):
        s = v.strip().lower()
        s = s.replace("@", "at").replace(".", "p").replace("/", "_").replace("-", "_")
        return s
    if isinstance(v, int):
        s = str(v)
    else:
        s = normalize_float(float(v))
    s = s.replace("-", "m").replace(".", "p")
    return s

base_defaults = {
    "epochs": int(cfg.get("epochs", 150)),
    "patience": int(cfg.get("patience", 30)),
    "batch": int(cfg.get("batch", 6)),
    "grad_accum": int(cfg.get("grad_accum", 1)),
    "lr0": float(cfg.get("lr0", 0.01)),
    "lrf": float(cfg.get("lrf", 0.1)),
    "warmup_epochs": float(cfg.get("warmup_epochs", 3.0)),
    "best_select_metric": parse_value("best_select_metric", str(cfg.get("best_select_metric", "fitness"))),
}

if epochs_override:
    base_defaults["epochs"] = int(epochs_override)

raw_cases = []
if hparams_raw:
    for chunk in hparams_raw.split("|"):
        chunk = chunk.strip()
        if not chunk:
            continue
        raw_cases.append(parse_case(chunk))
elif grid_raw:
    grid = {}
    parts = [p.strip() for p in grid_raw.split(";") if p.strip()]
    if not parts:
        raise SystemExit("--grid is empty")
    for p in parts:
        if "=" not in p:
            raise SystemExit(f"Invalid grid item '{p}', expected key=v1,v2")
        key, values = p.split("=", 1)
        key = key.strip().lower()
        if key not in ALL_KEYS:
            raise SystemExit(
                f"Unsupported grid key '{key}'. Allowed: {', '.join(ALL_KEYS)}"
            )
        vals = [x.strip() for x in values.split(",") if x.strip()]
        if not vals:
            raise SystemExit(f"Grid key '{key}' has no values")
        grid[key] = [parse_value(key, x) for x in vals]

    keys = list(grid.keys())
    for combo in itertools.product(*(grid[k] for k in keys)):
        case = {k: v for k, v in zip(keys, combo)}
        raw_cases.append(case)
else:
    raw_cases.append({})

resolved = []
seen = set()
for case in raw_cases:
    merged = dict(base_defaults)
    merged.update(case)

    if merged["epochs"] <= 0:
        raise SystemExit(f"epochs must be > 0, got {merged['epochs']}")
    if merged["patience"] < 0:
        raise SystemExit(f"patience must be >= 0, got {merged['patience']}")
    if merged["batch"] <= 0:
        raise SystemExit(f"batch must be > 0, got {merged['batch']}")
    if merged["grad_accum"] <= 0:
        raise SystemExit(f"grad_accum must be > 0, got {merged['grad_accum']}")
    if merged["lr0"] <= 0:
        raise SystemExit(f"lr0 must be > 0, got {merged['lr0']}")
    if merged["lrf"] < 0:
        raise SystemExit(f"lrf must be >= 0, got {merged['lrf']}")
    if merged["warmup_epochs"] < 0:
        raise SystemExit(f"warmup_epochs must be >= 0, got {merged['warmup_epochs']}")
    if str(merged["best_select_metric"]).lower() not in {"fitness", "map", "ifn", "iauroc@fpr0.5", "default", "默认"}:
        raise SystemExit(f"best_select_metric invalid: {merged['best_select_metric']}")

    sig = tuple(merged[k] for k in ALL_KEYS)
    if sig in seen:
        continue
    seen.add(sig)
    resolved.append(merged)

if not resolved:
    raise SystemExit("No valid hyperparameter cases resolved")
if len(resolved) > max_cases:
    raise SystemExit(
        f"Generated {len(resolved)} cases, exceeds --max-cases={max_cases}. "
        "Please narrow --grid/--hparams or raise --max-cases."
    )

for idx, case in enumerate(resolved, start=1):
    case_tag = (
        f"c{idx:03d}_"
        f"e{slug_num(case['epochs'])}_"
        f"p{slug_num(case['patience'])}_"
        f"b{slug_num(case['batch'])}_"
        f"ga{slug_num(case['grad_accum'])}_"
        f"lr{slug_num(case['lr0'])}_"
        f"lrf{slug_num(case['lrf'])}_"
        f"wu{slug_num(case['warmup_epochs'])}_"
        f"br{slug_num(case['best_select_metric'])}"
    )
    text = ",".join(
        [
            f"epochs={case['epochs']}",
            f"patience={case['patience']}",
            f"batch={case['batch']}",
            f"grad_accum={case['grad_accum']}",
            f"lr0={normalize_float(case['lr0'])}",
            f"lrf={normalize_float(case['lrf'])}",
            f"warmup_epochs={normalize_float(case['warmup_epochs'])}",
            f"best_select_metric={case['best_select_metric']}",
        ]
    )
    print(case_tag + "\t" + json.dumps(case, ensure_ascii=True, separators=(",", ":")) + "\t" + text)
PY
}

list_exp_dirs() {
  local exp_root="$1"
  if [[ -d "${exp_root}" ]]; then
    find "${exp_root}" -mindepth 1 -maxdepth 1 -type d -name 'exp_*' -printf '%f\n' | sort
  fi
}

generate_case_cfg() {
  local out_cfg="$1"
  local switches="$2"
  local case_tag="$3"
  local hparam_json="$4"

  _M_BASE_CONFIG="${BASE_CONFIG}" \
  _M_OUT_CFG="${out_cfg}" \
  _M_SWITCHES="${switches}" \
  _M_CASE_TAG="${case_tag}" \
  _M_HPARAM_JSON="${hparam_json}" \
  _M_FORCE_MODE="${FORCE_MODE}" \
  _M_SEED_OVERRIDE="${SEED_OVERRIDE}" \
  _M_D1_WORKERS_OVERRIDE="${D1_WORKERS_OVERRIDE}" \
  "${PYTHON_CFG_BIN}" - <<'PY'
import json
import os
from pathlib import Path

import yaml

base_cfg = Path(os.environ["_M_BASE_CONFIG"]).resolve()
out_cfg = Path(os.environ["_M_OUT_CFG"]).resolve()
switches = [s.strip() for s in os.environ["_M_SWITCHES"].split() if s.strip()]
case_tag = os.environ["_M_CASE_TAG"].strip()
hparams = json.loads(os.environ["_M_HPARAM_JSON"])
force_mode = os.environ["_M_FORCE_MODE"].strip()
seed_override = os.environ["_M_SEED_OVERRIDE"].strip()
d1_workers_override = os.environ["_M_D1_WORKERS_OVERRIDE"].strip()

allowed_switches = {
    "a3", "a5", "a7", "a9",
    "b1", "b2", "b3", "b5", "b7", "b9",
    "c5", "c7", "c9",
    "d1", "d3", "d5", "d7", "d9",
}

with base_cfg.open("r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}

if not isinstance(cfg, dict):
    raise SystemExit(f"Base config must be mapping: {base_cfg}")

enh = cfg.setdefault("enhance241", {})
if not isinstance(enh, dict):
    raise SystemExit("enhance241 must be a mapping in base config.")

for key in sorted(allowed_switches):
    enh[key] = False

for key in switches:
    if key not in allowed_switches:
        raise SystemExit(f"Unsupported module switch: {key}")
    if key == "d1":
        enh["d1"] = True
        enh["d3"] = True
    elif key == "d3":
        enh["d3"] = True
    else:
        enh[key] = True

exp_name = str(cfg.get("exp_name", "defect241"))
if switches:
    module_suffix = "__" + "__".join(switches)
    if not exp_name.endswith(module_suffix):
        exp_name = exp_name + module_suffix

# New layout:
#   experiments/<yolo_version>/<group_hp>/<group_hp__case_tag>/exp_*
# Example:
#   experiments/yolo11/defect241__a3__c5__hp/defect241__a3__c5__hp__c001_xxx/exp_*
exp_group_hp = exp_name if exp_name.endswith("__hp") else f"{exp_name}__hp"
case_exp_name = f"{exp_group_hp}__{case_tag}"
cfg["exp_name"] = f"{exp_group_hp}/{case_exp_name}"

cfg["epochs"] = int(hparams["epochs"])
cfg["patience"] = int(hparams["patience"])
cfg["batch"] = int(hparams["batch"])
cfg["grad_accum"] = int(hparams["grad_accum"])
cfg["lr0"] = float(hparams["lr0"])
cfg["lrf"] = float(hparams["lrf"])
cfg["warmup_epochs"] = float(hparams["warmup_epochs"])
cfg["best_select_metric"] = str(hparams.get("best_select_metric", cfg.get("best_select_metric", "fitness")))
if str(cfg["best_select_metric"]).lower() in {"ifn", "iauroc@fpr0.5"}:
    cfg["record_epoch_image_metrics"] = True

if force_mode:
    cfg["mode"] = force_mode
if seed_override:
    cfg["seed"] = int(seed_override)

if any(k in {"d1", "d3", "d5", "d7", "d9"} for k in switches) and d1_workers_override:
    cfg["workers"] = int(d1_workers_override)

yolo_version = str(cfg.get("yolo_version", "yolo11"))

out_cfg.parent.mkdir(parents=True, exist_ok=True)
with out_cfg.open("w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

hparam_text = (
    f"epochs={cfg['epochs']},"
    f"patience={cfg['patience']},"
    f"batch={cfg['batch']},"
    f"grad_accum={cfg['grad_accum']},"
    f"lr0={cfg['lr0']:g},"
    f"lrf={cfg['lrf']:g},"
    f"warmup_epochs={cfg['warmup_epochs']:g},"
    f"best_select_metric={cfg['best_select_metric']}"
)

print(f"{cfg['exp_name']}\t{yolo_version}\t{hparam_text}")
PY
}

CASE_TAGS=()
CASE_HPARAM_JSONS=()
CASE_HPARAM_TEXTS=()
while IFS=$'\t' read -r _case_tag _case_json _case_text; do
  [[ -n "${_case_tag}" ]] || continue
  CASE_TAGS+=("${_case_tag}")
  CASE_HPARAM_JSONS+=("${_case_json}")
  CASE_HPARAM_TEXTS+=("${_case_text}")
done < <(build_hparam_cases)

if [[ ${#CASE_TAGS[@]} -eq 0 ]]; then
  echo "[error] no valid hparam cases resolved" >&2
  exit 2
fi

SUMMARY_TSV="${LOG_ROOT}/summary.tsv"
{
  echo -e "case\tmodules\thparams\tstatus\texp_dir\tconfig\tlog"
} > "${SUMMARY_TSV}"

echo "[hparam-combo] base_config=${BASE_CONFIG}"
echo "[hparam-combo] modules_raw=${MODULES_RAW}"
echo "[hparam-combo] modules_resolved=${MODULE_SWITCHES:-baseline}"
echo "[hparam-combo] cases=${#CASE_TAGS[@]}"
echo "[hparam-combo] mode=${FORCE_MODE} seed=${SEED_OVERRIDE:-<keep>}"
echo "[hparam-combo] guard mode=${VRAM_GUARD_OVERRIDE} max_gb=${GUARD_MAX_GB_OVERRIDE} safe_batch=${SAFE_BATCH_OVERRIDE} safe_workers=${SAFE_WORKERS_OVERRIDE}"
echo "[hparam-combo] tmp_cfg_dir=${TMP_CFG_DIR}"
echo "[hparam-combo] log_root=${LOG_ROOT}"
print_pid_snapshot "startup"
echo "[hparam-combo] note: grad_accum is recorded in config/summary; src/train.py may not consume it directly."

fail_count=0

for i in "${!CASE_TAGS[@]}"; do
  case_tag="${CASE_TAGS[$i]}"
  case_json="${CASE_HPARAM_JSONS[$i]}"
  case_text="${CASE_HPARAM_TEXTS[$i]}"
  cfg_path="${TMP_CFG_DIR}/${MODULE_TAG}__${case_tag}.yaml"
  log_path="${LOG_ROOT}/${MODULE_TAG}__${case_tag}.log"

  meta="$(generate_case_cfg "${cfg_path}" "${MODULE_SWITCHES}" "${case_tag}" "${case_json}")"
  exp_name="$(echo "${meta}" | awk -F $'\t' '{print $1}')"
  yolo_version="$(echo "${meta}" | awk -F $'\t' '{print $2}')"
  hparam_text="$(echo "${meta}" | awk -F $'\t' '{print $3}')"
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

  echo "[hparam-combo:${case_tag}] modules=${MODULE_SWITCHES:-baseline}"
  echo "[hparam-combo:${case_tag}] hparams=${hparam_text}"
  echo "[hparam-combo:${case_tag}] cmd=${cmd[*]}"

  status=0
  if [[ "${DRY_RUN}" == "true" ]]; then
    {
      echo "[dry-run] ${cmd[*]}"
    } > "${log_path}"
  else
    "${cmd[@]}" > "${log_path}" 2>&1 &
    ACTIVE_CHILD_PID=$!
    print_pid_snapshot "after_spawn_${case_tag}"
    wait "${ACTIVE_CHILD_PID}" || status=$?
    ACTIVE_CHILD_PID=""
    print_pid_snapshot "after_wait_${case_tag}"
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
      exp_dir="${exp_root}/exp_error_${SWEEP_TAG}_${case_tag}"
      mkdir -p "${exp_dir}"
    fi
    error_md="${exp_dir}/error.md"
    {
      echo "# ERROR"
      echo
      echo "- case: \`${case_tag}\`"
      echo "- modules: \`${MODULE_SWITCHES:-baseline}\`"
      echo "- hparams: \`${hparam_text}\`"
      echo "- status: \`${status}\`"
      echo "- config: \`${cfg_path}\`"
      echo "- log: \`${log_path}\`"
      echo
      echo "## Log Tail"
      echo '```text'
      tail -n 200 "${log_path}" || true
      echo '```'
    } > "${error_md}"
    echo "[hparam-combo:${case_tag}] failed -> ${error_md}"
  else
    echo "[hparam-combo:${case_tag}] success exp_dir=${exp_dir}"
  fi

  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "${case_tag}" "${MODULE_SWITCHES:-baseline}" "${hparam_text}" "${status}" "${exp_dir}" "${cfg_path}" "${log_path}" \
    >> "${SUMMARY_TSV}"

  rm -f "${before_tmp}" "${after_tmp}"
done

echo
echo "[hparam-combo] finished. fail_count=${fail_count}"
echo "[hparam-combo] summary=${SUMMARY_TSV}"
column -s $'\t' -t "${SUMMARY_TSV}" || cat "${SUMMARY_TSV}"

exit 0

: <<'EXAMPLES'
# 0) default: modules=a3+c5, one case from base config
bash tools/run_yolov11_241_hparam_combo.sh

# 1) explicit cases on fixed a3+c5
bash tools/run_yolov11_241_hparam_combo.sh \
  --modules a3+c5 \
  --hparams "epochs=150,patience=0,batch=6,grad_accum=1,lr0=0.012,lrf=0.12,warmup_epochs=0|epochs=180,patience=20,batch=6,grad_accum=1,lr0=0.01,lrf=0.1,warmup_epochs=2"

# 2) grid expansion (cartesian product)
bash tools/run_yolov11_241_hparam_combo.sh \
  --modules a3+c5 \
  --grid "epochs=120,150;patience=0,20;batch=6;grad_accum=1,2;lr0=0.010,0.012;lrf=0.10,0.12;warmup_epochs=0,2" \
  --max-cases 128

# 3) dry-run only (check generated cfg and commands)
bash tools/run_yolov11_241_hparam_combo.sh \
  --modules a3+c5 \
  --hparams "epochs=150,patience=0,batch=6,grad_accum=1,lr0=0.012,lrf=0.12,warmup_epochs=0|epochs=150,patience=30,batch=6,grad_accum=1,lr0=0.008,lrf=0.1,warmup_epochs=2" \
  --dry-run

# 4) baseline module with explicit seed
bash tools/run_yolov11_241_hparam_combo.sh \
  configs/yolo11/defect.yaml \
  --modules baseline \
  --seed 42 \
  --hparams "epochs=120,patience=0,batch=8,grad_accum=1,lr0=0.008,lrf=0.12,warmup_epochs=2"
EXAMPLES
