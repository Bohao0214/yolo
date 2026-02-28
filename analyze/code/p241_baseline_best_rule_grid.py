#!/usr/bin/env python3
"""Baseline/module runner with per-epoch multi-rule best tracking.

目标：
- 运行 baseline 或 enhance241 模块组合网格（默认 epochs=150, batch=6/10, modules=baseline）。
- 单次训练同时记录 default/iFN/iAUROC@fpr0.5 三套 best 标记。
- 不保存权重/可视化/后评估曲线，只保留训练曲线与逐 epoch 指标表。
- 结果写入 analyze/result/report_*/。
"""

from __future__ import annotations

import argparse
import copy
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
DEFAULT_MODULE_BASE_CFG = ROOT / "configs" / "yolo11" / "enhance241" / "defect241.yaml"
DEFAULT_OUT_ROOT = ROOT / "analyze" / "result"

RULE_FLAG_COLUMNS = {
    "fitness": "is_best_default",
    "iFN": "is_best_ifn",
    "iAUROC@fpr0.5": "is_best_iauroc_fpr0p5",
}

ENHANCE241_FLAG_KEYS = [
    "a3",
    "a4",
    "a5",
    "a7",
    "a9",
    "a11",
    "a21",
    "b1",
    "b2",
    "b3",
    "b5",
    "b7",
    "b9",
    "b11",
    "b21",
    "c4",
    "c5",
    "c7",
    "c9",
    "c11",
    "c21",
    "d1",
    "d3",
    "d5",
    "d7",
    "d9",
    "d11",
    "d21",
]
MODULE_KEY_SET = set(ENHANCE241_FLAG_KEYS)


@dataclass
class ModuleSpec:
    raw: str
    tag: str
    keys: List[str]


@dataclass
class Case:
    case_id: int
    epochs: int
    batch: int
    module_tag: str
    module_raw: str
    module_keys: List[str]

    @property
    def case_name(self) -> str:
        return f"{self.module_tag}_c{self.case_id:03d}_e{self.epochs}_b{self.batch}"


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


def unique_keep_order(items: Sequence[str]) -> List[str]:
    out: List[str] = []
    for item in items:
        if item not in out:
            out.append(item)
    return out


def rule_slug(rule: str) -> str:
    if rule == "fitness":
        return "default"
    if rule == "iFN":
        return "ifn"
    return "iauroc_fpr0p5"


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


def flag_is_true(v: Any) -> bool:
    token = str(v).strip().lower()
    return token in {"1", "1.0", "true", "yes"}


def best_epoch_by_flag(rows: Sequence[Dict[str, Any]], flag_col: str) -> Optional[int]:
    epochs = [to_epoch_key(r.get("epoch")) for r in rows if flag_is_true(r.get(flag_col, ""))]
    epochs = [e for e in epochs if e is not None]
    return max(epochs) if epochs else None


def _parse_combo_keys(expr: str) -> List[str]:
    token = str(expr).strip().lower()
    if not token:
        return []
    sep = None
    if "+" in token:
        sep = "+"
    elif "_" in token:
        sep = "_"
    if not sep:
        if token in MODULE_KEY_SET:
            return [token]
        return []
    parts = [p.strip() for p in token.split(sep) if p.strip()]
    if not parts:
        return []
    if all(p in MODULE_KEY_SET for p in parts):
        return parts
    return []


def normalize_module_token(raw: str) -> List[ModuleSpec]:
    token = str(raw).strip().lower()
    token = token.replace(" ", "")
    if not token:
        return []

    if token in {"baseline", "base", "none", "default"}:
        return [ModuleSpec(raw="baseline", tag="baseline", keys=[])]
    if token in {"hmc7", "abcd7", "a7_b7_c7_d7", "a7+b7+c7+d7"}:
        return [ModuleSpec(raw=token, tag="a7__b7__c7__d7", keys=["a7", "b7", "c7", "d7"])]
    if token in {"pdd9", "abcd9", "a9_b9_c9_d9", "a9+b9+c9+d9"}:
        return [ModuleSpec(raw=token, tag="a9__b9__c9__d9", keys=["a9", "b9", "c9", "d9"])]
    if token in {"abcd11", "a11_b11_c11_d11", "a11+b11+c11+d11"}:
        return [ModuleSpec(raw=token, tag="a11__b11__c11__d11", keys=["a11", "b11", "c11", "d11"])]
    if token in {"abcd21", "pack21", "a21_b21_c21_d21", "a21+b21+c21+d21"}:
        return [ModuleSpec(raw=token, tag="a21__b21__c21__d21", keys=["a21", "b21", "c21", "d21"])]
    if token == "b1237":
        return [
            ModuleSpec(raw=token, tag="b1", keys=["b1"]),
            ModuleSpec(raw=token, tag="b2", keys=["b2"]),
            ModuleSpec(raw=token, tag="b3", keys=["b3"]),
            ModuleSpec(raw=token, tag="b7", keys=["b7"]),
        ]
    if token == "d1579":
        return [
            ModuleSpec(raw=token, tag="d1", keys=["d1"]),
            ModuleSpec(raw=token, tag="d5", keys=["d5"]),
            ModuleSpec(raw=token, tag="d7", keys=["d7"]),
            ModuleSpec(raw=token, tag="d9", keys=["d9"]),
        ]
    if token in MODULE_KEY_SET:
        return [ModuleSpec(raw=token, tag=token, keys=[token])]

    combo_keys = _parse_combo_keys(token)
    if combo_keys:
        return [ModuleSpec(raw=token, tag="__".join(combo_keys), keys=combo_keys)]

    raise ValueError(
        "Unsupported module token: "
        f"{raw}. Allowed: baseline, single module keys ({'/'.join(ENHANCE241_FLAG_KEYS)}), "
        "aliases hmc7/pdd9/abcd11/abcd21/pack21/b1237/d1579, or '+'/'_' combinations like a3+c5."
    )


def parse_module_specs(raw: str) -> List[ModuleSpec]:
    specs: List[ModuleSpec] = []
    for item in str(raw).split(","):
        part = item.strip()
        if not part:
            continue
        specs.extend(normalize_module_token(part))
    if not specs:
        raise ValueError("modules is empty")
    out: List[ModuleSpec] = []
    seen: set[Tuple[str, Tuple[str, ...]]] = set()
    for spec in specs:
        key = (spec.tag, tuple(spec.keys))
        if key in seen:
            continue
        seen.add(key)
        out.append(spec)
    return out


def choose_case_base_config(
    base_cfg_path: Path,
    module_base_cfg_path: Path,
    has_modules: bool,
) -> Path:
    if has_modules:
        return module_base_cfg_path
    return base_cfg_path


def build_case_config(
    template_cfg: Dict[str, Any],
    case: Case,
    *,
    patience: int,
    grad_accum: int,
    lr0: float,
    lrf: float,
    warmup_epochs: float,
) -> Dict[str, Any]:
    cfg = copy.deepcopy(template_cfg)
    cfg["run_name"] = f"{case.case_name}_{dt.datetime.now().strftime('%y%m%d%H%M%S')}"
    cfg["mode"] = "train_test"
    cfg["epochs"] = int(case.epochs)
    cfg["batch"] = int(case.batch)
    cfg["patience"] = int(patience)
    cfg["grad_accum"] = int(grad_accum)
    cfg["lr0"] = float(lr0)
    cfg["lrf"] = float(lrf)
    cfg["warmup_epochs"] = float(warmup_epochs)
    cfg["best_select_metric"] = "fitness"
    cfg["record_epoch_image_metrics"] = True
    cfg["save_weights"] = False
    cfg["save_val_pic"] = False
    cfg["save_test_pic"] = False
    cfg["skip_eval_visuals"] = True
    cfg["skip_post_eval_metrics"] = True

    enh = cfg.get("enhance241")
    if not isinstance(enh, dict):
        enh = {}
        cfg["enhance241"] = enh
    for key in ENHANCE241_FLAG_KEYS:
        enh[key] = False
    for key in case.module_keys:
        if key == "d1":
            enh["d1"] = True
            enh["d3"] = True
        elif key == "d3":
            enh["d3"] = True
        else:
            enh[key] = True
    if enh.get("b1"):
        enh["b1_version"] = "v3"
    return cfg


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
            "module_tag": case.module_tag,
            "module_raw": case.module_raw,
            "module_keys": " ".join(case.module_keys),
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


def collect_metric_extrema(rows: Sequence[Dict[str, str]]) -> Dict[str, Any]:
    val_ifn_values: List[float] = []
    val_iauroc_values: List[float] = []
    for r in rows:
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
    return {
        "val_iFN_min": min(val_ifn_values) if val_ifn_values else "",
        "val_iAUROC_fpr0p5_max": max(val_iauroc_values) if val_iauroc_values else "",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Baseline/module runner with shared per-epoch best-rule tracking.")
    parser.add_argument("--base-config", type=str, default=str(DEFAULT_BASE_CFG))
    parser.add_argument(
        "--module-base-config",
        type=str,
        default=str(DEFAULT_MODULE_BASE_CFG),
        help="Template config used when module cases are requested.",
    )
    parser.add_argument("--out-root", type=str, default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--epochs", type=str, default="150")
    parser.add_argument("--batches", type=str, default="6,10")
    parser.add_argument(
        "--modules",
        type=str,
        default="baseline",
        help="Comma-separated module cases: baseline, a3, a3+c5, hmc7, abcd11, b1237, etc.",
    )
    parser.add_argument(
        "--best-rules",
        type=str,
        default="default,iFN,iAUROC@fpr0.5",
        help="Tracked rules in reports only. Does not increase training run count.",
    )
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
    module_base_cfg_path = Path(args.module_base_config).resolve()
    if not module_base_cfg_path.exists():
        raise FileNotFoundError(f"module base config not found: {module_base_cfg_path}")

    epochs_list = parse_list(args.epochs, int)
    batch_list = parse_list(args.batches, int)
    module_specs = parse_module_specs(args.modules)
    tracked_rules_raw = [x.strip() for x in str(args.best_rules).split(",") if x.strip()]
    if not tracked_rules_raw:
        raise ValueError("best-rules is empty")
    tracked_rules = unique_keep_order([normalize_best_rule(x) for x in tracked_rules_raw])

    base_cfg_plain = load_yaml(base_cfg_path)
    base_cfg_module = load_yaml(module_base_cfg_path)

    report_dir = make_report_dir(Path(args.out_root).resolve())
    cfg_dir = report_dir / "tmp_cfgs"
    log_dir = report_dir / "logs"
    case_table_dir = report_dir / "case_tables"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    case_table_dir.mkdir(parents=True, exist_ok=True)

    cases: List[Case] = []
    cid = 1
    for mod in module_specs:
        for ep in epochs_list:
            for bt in batch_list:
                cases.append(
                    Case(
                        case_id=cid,
                        epochs=int(ep),
                        batch=int(bt),
                        module_tag=mod.tag,
                        module_raw=mod.raw,
                        module_keys=list(mod.keys),
                    )
                )
                cid += 1

    write_csv(
        report_dir / "plan.csv",
        [
            {
                **case.__dict__,
                "case_name": case.case_name,
                "module_keys": " ".join(case.module_keys),
            }
            for case in cases
        ],
    )
    (report_dir / "plan.json").write_text(
        json.dumps(
            {
                "base_config": str(base_cfg_path),
                "module_base_config": str(module_base_cfg_path),
                "tracked_rules": tracked_rules,
                "modules": [{"raw": m.raw, "tag": m.tag, "keys": m.keys} for m in module_specs],
                "cases": [
                    {
                        **case.__dict__,
                        "case_name": case.case_name,
                    }
                    for case in cases
                ],
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
        f"modules={','.join(m.tag for m in module_specs)} tracked_rules={','.join(tracked_rules)} "
        f"base_config={base_cfg_path}"
    )

    for case in cases:
        template_path = choose_case_base_config(
            base_cfg_path=base_cfg_path,
            module_base_cfg_path=module_base_cfg_path,
            has_modules=bool(case.module_keys),
        )
        template_cfg = base_cfg_module if template_path == module_base_cfg_path else base_cfg_plain
        cfg = build_case_config(
            template_cfg,
            case,
            patience=args.patience,
            grad_accum=args.grad_accum,
            lr0=args.lr0,
            lrf=args.lrf,
            warmup_epochs=args.warmup_epochs,
        )
        cfg_path = cfg_dir / f"{case.case_name}.yaml"
        dump_yaml(cfg_path, cfg)

        yolo_version = str(cfg.get("yolo_version", "yolo11"))
        exp_name = str(cfg.get("exp_name", "defect"))
        exp_root = ROOT / "experiments" / yolo_version / exp_name
        run_name = str(cfg["run_name"])

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
            "module_tag": case.module_tag,
            "module_raw": case.module_raw,
            "module_keys": " ".join(case.module_keys),
            "status": status,
            "dry_run": bool(args.dry_run),
            "tracked_rules": ",".join(tracked_rules),
            "template_config": str(template_path),
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
                    "module_tag": case.module_tag,
                    "module_raw": case.module_raw,
                    "module_keys": " ".join(case.module_keys),
                    "status": status,
                    "dry_run": bool(args.dry_run),
                    "tracked_rules": ",".join(tracked_rules),
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

        summary: Dict[str, Any] = {
            "case_name": case.case_name,
            "module_tag": case.module_tag,
            "module_raw": case.module_raw,
            "module_keys": " ".join(case.module_keys),
            "status": status,
            "dry_run": bool(args.dry_run),
            "tracked_rules": ",".join(tracked_rules),
            "exp_dir": str(exp_dir),
            "epochs_cfg": case.epochs,
            "batch_cfg": case.batch,
            "epoch_rows": len(merged_rows),
            "last_epoch_recorded": max([r["epoch"] for r in merged_rows], default=""),
        }
        summary.update(collect_metric_extrema(image_rows))
        for rule in tracked_rules:
            flag_col = RULE_FLAG_COLUMNS[rule]
            summary[f"best_epoch_{rule_slug(rule)}"] = best_epoch_by_flag(image_rows, flag_col) or ""
        summary_rows.append(summary)

    write_csv(report_dir / "run_records.csv", run_records)
    write_csv(report_dir / "epoch_metrics_all.csv", merged_all)
    write_csv(report_dir / "run_summary.csv", summary_rows)

    md_lines = [
        f"# Baseline Module Report ({dt.datetime.now().isoformat(timespec='seconds')})",
        "",
        f"- report_dir: `{report_dir}`",
        f"- base_config: `{base_cfg_path}`",
        f"- module_base_config: `{module_base_cfg_path}`",
        f"- cases: `{len(cases)}`",
        f"- modules: `{','.join(m.tag for m in module_specs)}`",
        f"- tracked_rules: `{','.join(tracked_rules)}`",
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
    """""
    python analyze/code/p241_baseline_best_rule_grid.py --modules baseline,a3+c5,abcd11
    python analyze/code/p241_baseline_best_rule_grid.py --epochs 150 --batches 6,1
    """""
    main()
