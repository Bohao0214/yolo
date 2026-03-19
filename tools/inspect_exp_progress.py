#!/usr/bin/env python3
"""Inspect latest experiment progress and optionally rewrite config for recovery."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Optional

import yaml


def _to_int(v: object) -> Optional[int]:
    try:
        s = str(v).strip()
        if not s:
            return None
        return int(round(float(s)))
    except Exception:
        return None


def _read_latest_epoch(results_csv: Path) -> Optional[int]:
    if not results_csv.exists():
        return None

    # Prefer headered csv with 'epoch' column.
    try:
        with results_csv.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames and "epoch" in [str(x).strip() for x in reader.fieldnames]:
                best = None
                for row in reader:
                    e = _to_int(row.get("epoch", ""))
                    if e is None:
                        continue
                    best = e if best is None else max(best, e)
                if best is not None:
                    return best
    except Exception:
        pass

    # Fallback: use first column as epoch.
    try:
        with results_csv.open("r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            best = None
            for row in reader:
                if not row:
                    continue
                e = _to_int(row[0])
                if e is None:
                    continue
                best = e if best is None else max(best, e)
            return best
    except Exception:
        return None


def _resolve_exp_root(cfg: Dict[str, object], root_dir: Path) -> Path:
    project_root = Path(str(cfg.get("project_root", root_dir))).resolve()
    exp_name = str(cfg.get("exp_name", "defect")).strip() or "defect"
    run_name = str(cfg.get("run_name", "")).strip()
    exp_root_new = project_root / "experiments" / exp_name
    if run_name:
        exp_root_new = exp_root_new / run_name
    if exp_root_new.exists():
        return exp_root_new

    # Backward-compatible fallback for old layout: experiments/<yolo_version>/<exp_name>
    yolo_version = str(cfg.get("yolo_version", "yolo11")).strip() or "yolo11"
    exp_root_old = project_root / "experiments" / yolo_version / exp_name
    if run_name:
        exp_root_old = exp_root_old / run_name
    return exp_root_old


def _latest_exp_dir(exp_root: Path) -> Optional[Path]:
    if not exp_root.exists():
        return None
    exp_dirs = [p for p in exp_root.glob("exp_*") if p.is_dir()]
    if not exp_dirs:
        return None
    exp_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return exp_dirs[0]


def _classify(
    cfg: Dict[str, object],
    root_dir: Path,
    ignore_mode: bool = False,
) -> Dict[str, object]:
    mode = str(cfg.get("mode", "train_test")).strip().lower()
    target_epochs = _to_int(cfg.get("epochs", 0)) or 0
    exp_root = _resolve_exp_root(cfg, root_dir)
    latest = _latest_exp_dir(exp_root)

    out: Dict[str, object] = {
        "action": "fresh",
        "reason": "",
        "exp_root": str(exp_root),
        "latest_exp_dir": str(latest) if latest else "",
        "mode": mode,
        "target_epochs": int(target_epochs),
        "max_epoch": "",
        "remaining_epochs": "",
        "resume_weight": "",
        "has_best": "0",
        "has_last": "0",
    }

    if not ignore_mode and mode not in {"train_test", "finetune_test"}:
        out["reason"] = f"mode={mode}"
        return out

    if target_epochs <= 0:
        out["reason"] = "invalid_target_epochs"
        return out

    if latest is None:
        out["reason"] = "no_previous_exp"
        return out

    best_pt = latest / "train" / "weights" / "best.pt"
    last_pt = latest / "train" / "weights" / "last.pt"
    has_best = best_pt.exists()
    has_last = last_pt.exists()
    out["has_best"] = "1" if has_best else "0"
    out["has_last"] = "1" if has_last else "0"

    resume_weight = last_pt if has_last else (best_pt if has_best else None)
    eval_weight = best_pt if has_best else (last_pt if has_last else None)
    out["resume_weight"] = str(resume_weight) if resume_weight else ""

    results_csv = latest / "train" / "results.csv"
    max_epoch = _read_latest_epoch(results_csv)
    out["max_epoch"] = "" if max_epoch is None else int(max_epoch)

    if max_epoch is None:
        out["reason"] = "missing_or_invalid_results_csv"
        return out

    # results.csv usually stores 0-based epoch index.
    is_complete = (max_epoch >= (target_epochs - 1)) and bool(eval_weight)
    if is_complete:
        out["action"] = "validate_only"
        out["remaining_epochs"] = 0
        out["resume_weight"] = str(eval_weight) if eval_weight else ""
        out["reason"] = "complete_by_results"
        return out

    if resume_weight and max_epoch >= 0:
        remaining = target_epochs - (max_epoch + 1)
        if remaining < 1:
            remaining = 1
        out["action"] = "resume_finetune"
        out["remaining_epochs"] = int(remaining)
        out["reason"] = "partial_results_found"
        return out

    out["reason"] = "no_resume_checkpoint"
    return out


def _apply_recovery(cfg_path: Path, cfg: Dict[str, object], state: Dict[str, object]) -> None:
    action = str(state.get("action", "fresh"))
    weight = str(state.get("resume_weight", "")).strip()
    if action == "validate_only":
        if weight:
            cfg["mode"] = "test"
            cfg["weights"] = weight
    elif action == "resume_finetune":
        if weight:
            cfg["mode"] = "finetune_test"
            cfg["weights"] = weight
            rem = _to_int(state.get("remaining_epochs", ""))
            if rem is not None and rem > 0:
                cfg["epochs"] = int(rem)
    else:
        return

    with cfg_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="Inspect latest exp progress and emit recovery action.")
    ap.add_argument("--config", required=True, help="Runtime YAML config path")
    ap.add_argument("--root-dir", default="", help="Project root fallback")
    ap.add_argument("--policy", default="auto", choices=["auto", "off"])
    ap.add_argument("--apply", action="store_true", help="Apply action by rewriting config.")
    ap.add_argument(
        "--ignore-mode",
        action="store_true",
        help="Ignore cfg.mode restriction when classifying (used for post-run salvage).",
    )
    args = ap.parse_args()

    cfg_path = Path(args.config).resolve()
    root_dir = Path(args.root_dir).resolve() if str(args.root_dir).strip() else Path.cwd().resolve()

    if args.policy == "off":
        print("fresh\t\t\t\t\tpolicy_off")
        return 0

    if not cfg_path.exists():
        print("fresh\t\t\t\t\tmissing_config")
        return 0

    try:
        with cfg_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        if not isinstance(cfg, dict):
            print("fresh\t\t\t\t\tinvalid_config_mapping")
            return 0
    except Exception:
        print("fresh\t\t\t\t\tinvalid_config_yaml")
        return 0

    state = _classify(cfg, root_dir=root_dir, ignore_mode=bool(args.ignore_mode))

    if args.apply and state.get("action") in {"validate_only", "resume_finetune"}:
        _apply_recovery(cfg_path, cfg, state)

    action = str(state.get("action", "fresh"))
    latest = str(state.get("latest_exp_dir", ""))
    max_epoch = str(state.get("max_epoch", ""))
    target = str(state.get("target_epochs", ""))
    remaining = str(state.get("remaining_epochs", ""))
    weight = str(state.get("resume_weight", ""))
    reason = str(state.get("reason", ""))
    print(f"{action}\t{latest}\t{max_epoch}\t{target}\t{remaining}\t{weight}\t{reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
