#!/usr/bin/env python3
"""Baseline grid runner with custom best-selection rules.

目标：
- 运行 baseline 2x2x2 网格（epochs x batch x best_select_metric）。
- 不保存权重/可视化/后评估曲线，只保留训练曲线与逐 epoch 指标表。
- 结果写入 analyze/result/report_*/。
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import yaml  # type: ignore


ROOT = Path("/home/ubuntu/hpproject/yolo").resolve()
DEFAULT_BASE_CFG = ROOT / "configs" / "yolo11" / "defect.yaml"
DEFAULT_OUT_ROOT = ROOT / "analyze" / "result"


@dataclass
class Case:
    case_id: int
    epochs: int
    batch: int
    best_select_metric: str

    @property
    def case_name(self) -> str:
        rule = self.best_select_metric.lower().replace("@", "at").replace(".", "p").replace("/", "_")
        return f"baseline_c{self.case_id:03d}_e{self.epochs}_b{self.batch}_{rule}"


def parse_list(raw: str, cast_fn) -> List[Any]:
    out: List[Any] = []
    for x in str(raw).split(","):
        t = x.strip()
        if not t:
            continue
        out.append(cast_fn(t))
    if not out:
        raise ValueError(f"Empty list from input: {raw!r}")
    return out


def normalize_best_rule(raw: str) -> str:
    token = str(raw).strip().lower()
    if token in {"fitness", "default", "默认", "map", "m_ap"}:
        return "fitness"
    if token in {"ifn"}:
        return "iFN"
    if token in {"iauroc@fpr0.5"}:
        return "iAUROC@fpr0.5"
    raise ValueError(f"Unsupported best rule: {raw}. Allowed: fitness(default/默认), iFN, iAUROC@fpr0.5")


def make_report_dir(out_root: Path) -> Path:
    out_root.mkdir(parents=True, exist_ok=True)
    ts = dt.datetime.now().strftime("report_%y%m%d%H%M")
    cand = out_root / ts
    if not cand.exists():
        cand.mkdir(parents=True, exist_ok=False)
        return cand
    i = 1
    while True:
        p = out_root / f"{ts}_{i:02d}"
        if not p.exists():
            p.mkdir(parents=True, exist_ok=False)
            return p
        i += 1


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML must be mapping: {path}")
    return data


def dump_yaml(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def list_exp_dirs(exp_root: Path) -> List[Path]:
    if not exp_root.exists():
        return []
    return sorted([p for p in exp_root.glob("exp_*") if p.is_dir()], key=lambda p: p.name)


def newest(paths: Sequence[Path]) -> Optional[Path]:
    if not paths:
        return None
    return max(paths, key=lambda p: p.stat().st_mtime)


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def to_epoch_key(v: Any) -> Optional[int]:
    try:
        return int(round(float(v)))
    except Exception:
        return None


def merge_epoch_rows(
    case: Case,
    exp_dir: Path,
    results_rows: List[Dict[str, str]],
    image_rows: List[Dict[str, str]],
) -> List[Dict[str, Any]]:
    results_map: Dict[int, Dict[str, str]] = {}
    image_map: Dict[int, Dict[str, str]] = {}

    for r in results_rows:
        k = to_epoch_key(r.get("epoch"))
        if k is not None:
            results_map[k] = r
    for r in image_rows:
        k = to_epoch_key(r.get("epoch"))
        if k is not None:
            image_map[k] = r

    epochs = sorted(set(results_map.keys()) | set(image_map.keys()))
    merged: List[Dict[str, Any]] = []
    for e in epochs:
        base: Dict[str, Any] = {
            "case_name": case.case_name,
            "epochs_cfg": case.epochs,
            "batch_cfg": case.batch,
            "best_select_metric_cfg": case.best_select_metric,
            "exp_dir": str(exp_dir),
            "epoch": e,
        }
        rr = results_map.get(e, {})
        ir = image_map.get(e, {})
        for k, v in rr.items():
            if k == "epoch":
                continue
            base[k] = v
        for k, v in ir.items():
            if k == "epoch":
                continue
            base[k] = v
        merged.append(base)
    return merged


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: List[str] = []
    for r in rows:
        for k in r.keys():
            if k not in keys:
                keys.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> None:
    parser = argparse.ArgumentParser(description="Baseline 2x2x2 grid runner with custom best rule.")
    parser.add_argument("--base-config", type=str, default=str(DEFAULT_BASE_CFG))
    parser.add_argument("--out-root", type=str, default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--epochs", type=str, default="100,150")
    parser.add_argument("--batches", type=str, default="6,10")
    parser.add_argument("--best-rules", type=str, default="fitness,iFN,iAUROC@fpr0.5")
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--lr0", type=float, default=0.012)
    parser.add_argument("--lrf", type=float, default=0.12)
    parser.add_argument("--warmup-epochs", type=float, default=0.0)
    parser.add_argument("--python-bin", type=str, default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    base_cfg_path = Path(args.base_config).resolve()
    if not base_cfg_path.exists():
        raise FileNotFoundError(f"base config not found: {base_cfg_path}")

    epochs_list = parse_list(args.epochs, int)
    batch_list = parse_list(args.batches, int)
    rules = [x.strip() for x in str(args.best_rules).split(",") if x.strip()]
    if not rules:
        raise ValueError("best-rules is empty")
    rules = [normalize_best_rule(x) for x in rules]

    report_dir = make_report_dir(Path(args.out_root).resolve())
    cfg_dir = report_dir / "tmp_cfgs"
    log_dir = report_dir / "logs"
    case_table_dir = report_dir / "case_tables"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    case_table_dir.mkdir(parents=True, exist_ok=True)

    base_cfg = load_yaml(base_cfg_path)
    yolo_version = str(base_cfg.get("yolo_version", "yolo11"))
    exp_name = str(base_cfg.get("exp_name", "defect"))
    exp_root = ROOT / "experiments" / yolo_version / exp_name

    cases: List[Case] = []
    cid = 1
    for ep in epochs_list:
        for bt in batch_list:
            for rule in rules:
                cases.append(Case(case_id=cid, epochs=int(ep), batch=int(bt), best_select_metric=rule))
                cid += 1

    write_csv(report_dir / "plan.csv", [case.__dict__ | {"case_name": case.case_name} for case in cases])
    (report_dir / "plan.json").write_text(
        json.dumps(
            {
                "base_config": str(base_cfg_path),
                "cases": [case.__dict__ | {"case_name": case.case_name} for case in cases],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    run_records: List[Dict[str, Any]] = []
    merged_all: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    fail_count = 0

    print(
        "[run] "
        f"report_dir={report_dir} cases={len(cases)} dry_run={args.dry_run} "
        f"base_config={base_cfg_path}"
    )

    for case in cases:
        cfg = dict(base_cfg)
        run_name = f"{case.case_name}_{dt.datetime.now().strftime('%y%m%d%H%M%S')}"
        cfg["run_name"] = run_name
        cfg["mode"] = "train_test"
        cfg["epochs"] = int(case.epochs)
        cfg["batch"] = int(case.batch)
        cfg["patience"] = int(args.patience)
        cfg["grad_accum"] = int(args.grad_accum)
        cfg["lr0"] = float(args.lr0)
        cfg["lrf"] = float(args.lrf)
        cfg["warmup_epochs"] = float(args.warmup_epochs)
        cfg["best_select_metric"] = str(case.best_select_metric)
        cfg["record_epoch_image_metrics"] = True
        cfg["save_weights"] = False
        cfg["save_val_pic"] = False
        cfg["save_test_pic"] = False
        cfg["skip_eval_visuals"] = True
        cfg["skip_post_eval_metrics"] = True

        cfg_path = cfg_dir / f"{case.case_name}.yaml"
        dump_yaml(cfg_path, cfg)

        before = {str(p) for p in list_exp_dirs(exp_root / run_name)}
        cmd = [str(args.python_bin), "src/train.py", "--config", str(cfg_path)]
        log_path = log_dir / f"{case.case_name}.log"
        started = dt.datetime.now().isoformat(timespec="seconds")
        status = 0
        if args.dry_run:
            log_path.write_text("[dry-run] " + " ".join(cmd) + "\n", encoding="utf-8")
        else:
            with log_path.open("w", encoding="utf-8") as f:
                proc = subprocess.run(cmd, cwd=str(ROOT), stdout=f, stderr=subprocess.STDOUT, check=False)
                status = int(proc.returncode)
        ended = dt.datetime.now().isoformat(timespec="seconds")

        after_paths = list_exp_dirs(exp_root / run_name)
        after = {str(p) for p in after_paths}
        created = sorted(after - before)
        exp_dir = Path(created[-1]).resolve() if created else (newest(after_paths) if after_paths else None)

        rec = {
            "case_name": case.case_name,
            "status": status,
            "dry_run": bool(args.dry_run),
            "started_at": started,
            "ended_at": ended,
            "run_name": run_name,
            "config_path": str(cfg_path),
            "log_path": str(log_path),
            "exp_dir": str(exp_dir) if exp_dir is not None else "",
            "command": " ".join(cmd),
        }
        run_records.append(rec)

        if status != 0 or exp_dir is None:
            if (not args.dry_run) and (status != 0 or exp_dir is None):
                fail_count += 1
            note = "run_failed_or_missing_exp_dir"
            if args.dry_run and status == 0 and exp_dir is None:
                note = "dry_run_no_training"
            summary_rows.append(
                {
                    "case_name": case.case_name,
                    "status": status,
                    "dry_run": bool(args.dry_run),
                    "best_select_metric_cfg": case.best_select_metric,
                    "exp_dir": str(exp_dir) if exp_dir is not None else "",
                    "note": note,
                }
            )
            continue

        results_csv = exp_dir / "train" / "results.csv"
        image_csv = exp_dir / "train" / "epoch_image_metrics.csv"
        results_rows = read_csv(results_csv)
        image_rows = read_csv(image_csv)
        merged_rows = merge_epoch_rows(case, exp_dir, results_rows, image_rows)
        merged_all.extend(merged_rows)
        write_csv(case_table_dir / f"{case.case_name}_epoch_metrics.csv", merged_rows)

        best_epochs = [to_epoch_key(r.get("epoch")) for r in image_rows if str(r.get("is_best_by_rule", "")) in {"1", "1.0"}]
        best_epochs = [x for x in best_epochs if x is not None]
        best_epoch = max(best_epochs) if best_epochs else None
        val_ifn_values = []
        val_iauroc_values = []
        for r in image_rows:
            try:
                val_ifn_values.append(float(r.get("val_iFN", "nan")))
            except Exception:
                pass
            try:
                val_iauroc_values.append(float(r.get("val_iAUROC_fpr0p5", "nan")))
            except Exception:
                pass
        val_ifn_values = [x for x in val_ifn_values if x == x]
        val_iauroc_values = [x for x in val_iauroc_values if x == x]

        summary_rows.append(
            {
                "case_name": case.case_name,
                "status": status,
                "dry_run": bool(args.dry_run),
                "best_select_metric_cfg": case.best_select_metric,
                "exp_dir": str(exp_dir),
                "epochs_cfg": case.epochs,
                "batch_cfg": case.batch,
                "epoch_rows": len(merged_rows),
                "best_epoch_by_rule": best_epoch if best_epoch is not None else "",
                "val_iFN_min": min(val_ifn_values) if val_ifn_values else "",
                "val_iAUROC_fpr0p5_max": max(val_iauroc_values) if val_iauroc_values else "",
            }
        )

    write_csv(report_dir / "run_records.csv", run_records)
    write_csv(report_dir / "epoch_metrics_all.csv", merged_all)
    write_csv(report_dir / "run_summary.csv", summary_rows)

    md_lines = [
        f"# Baseline Grid Report ({dt.datetime.now().isoformat(timespec='seconds')})",
        "",
        f"- report_dir: `{report_dir}`",
        f"- base_config: `{base_cfg_path}`",
        f"- cases: `{len(cases)}`",
        f"- dry_run: `{args.dry_run}`",
        f"- fail_count: `{fail_count}`",
        "- fixed_hparams:",
        f"  - patience={args.patience}",
        f"  - grad_accum={args.grad_accum}",
        f"  - lr0={args.lr0}",
        f"  - lrf={args.lrf}",
        f"  - warmup_epochs={args.warmup_epochs}",
        "",
        "## Outputs",
        f"- `{report_dir / 'plan.csv'}`",
        f"- `{report_dir / 'run_records.csv'}`",
        f"- `{report_dir / 'run_summary.csv'}`",
        f"- `{report_dir / 'epoch_metrics_all.csv'}`",
        f"- `{case_table_dir}`",
    ]
    (report_dir / "report.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    if (not args.dry_run) and fail_count > 0:
        print(f"[failed] report_dir={report_dir} fail_count={fail_count}")
        first_failed = next((r for r in run_records if int(r.get("status", 0)) != 0), None)
        if first_failed is not None:
            log_path = Path(str(first_failed.get("log_path", "")))
            print(f"[failed] first_failed_case={first_failed.get('case_name')} log={log_path}")
            if log_path.exists():
                try:
                    lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
                    tail = lines[-30:] if len(lines) > 30 else lines
                    print("[failed] ---- log tail ----")
                    for line in tail:
                        print(line)
                    print("[failed] ------------------")
                except Exception:
                    pass
        raise SystemExit(2)
    print(f"[ok] report_dir={report_dir} fail_count={fail_count}")


if __name__ == "__main__":
    main()
