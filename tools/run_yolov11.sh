#!/usr/bin/env bash
set -euo pipefail

export LANG=C.UTF-8
export LC_ALL=C.UTF-8

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${1:-${ROOT_DIR}/configs/yolo11/defect.yaml}"
ACTIVE_CHILD_PID=""

print_pid_snapshot() {
  local stage="${1:-snapshot}"
  local pgid
  pgid="$(ps -o pgid= -p "$$" 2>/dev/null | tr -d '[:space:]' || true)"
  if [[ -n "${ACTIVE_CHILD_PID:-}" ]]; then
    echo "[run-yolov11][pid] stage=${stage} self=$$ ppid=${PPID:-NA} pgid=${pgid:-NA} child=${ACTIVE_CHILD_PID}"
  else
    echo "[run-yolov11][pid] stage=${stage} self=$$ ppid=${PPID:-NA} pgid=${pgid:-NA} child=(none)"
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
  echo "[run-yolov11] received ${sig}, terminating current child..."
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

python "${ROOT_DIR}/src/train.py" --config "${CONFIG}" &
ACTIVE_CHILD_PID=$!
echo "[run-yolov11][train-pid] pid=${ACTIVE_CHILD_PID} config=${CONFIG}"
train_status=0
wait "${ACTIVE_CHILD_PID}" || train_status=$?
ACTIVE_CHILD_PID=""
exit "${train_status}"

# bash tools/run_yolov11.sh configs/yolo11/defect.yaml
