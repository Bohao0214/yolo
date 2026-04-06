#!/usr/bin/env python3
"""
用法说明（消融表生成）：

本脚本读取 compare_main.csv，自动归类并输出 ablation_main.csv / ablation_main.md。
默认报告根目录为脚本上一级：
/home/ubuntu/hpproject/yolo/analyze/code/ch4tool

示例：
conda run -n yolo11 python /home/ubuntu/hpproject/yolo/analyze/code/ch4tool/code/run_eval_ablation.py \
  --report-root /home/ubuntu/hpproject/yolo/analyze/code/ch4tool
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Tuple

SCRIPT_PATH = Path(__file__).resolve()
REPORT_ROOT = SCRIPT_PATH.parents[1]
PROJECT_ROOT = SCRIPT_PATH.parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPT_PATH.parent) not in sys.path:
    sys.path.insert(0, str(SCRIPT_PATH.parent))

from utils_common import markdown_table, write_csv


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build ablation table from compare_main.csv")
    p.add_argument("--report-root", type=Path, default=REPORT_ROOT)
    return p.parse_args()


def _read_csv(path: Path) -> List[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _groups_set(s: str) -> set:
    return {x.strip() for x in str(s).split("+") if x.strip()}


def _to_float(v: str) -> float:
    try:
        return float(v)
    except Exception:
        return -1.0


def _variant_key(row: dict) -> str:
    groups = _groups_set(row.get("module_groups", ""))
    if not groups or groups == {"baseline"}:
        return "baseline"
    full = {"feature_extraction", "feature_fusion", "classification_calibration"}
    if full.issubset(groups):
        return "all_modules"
    if groups == {"feature_extraction"}:
        return "feature_extraction"
    if groups == {"feature_fusion"}:
        return "feature_fusion"
    if groups == {"classification_calibration"}:
        return "classification_calibration"
    return "mixed"


def main() -> None:
    args = parse_args()
    report_root = args.report_root.expanduser().resolve()
    tables_dir = report_root / "tables"

    compare_path = tables_dir / "compare_main.csv"
    rows = _read_csv(compare_path)

    ok_rows = [r for r in rows if str(r.get("status", "")) == "ok"]
    by_variant: Dict[str, List[dict]] = {}
    for r in ok_rows:
        by_variant.setdefault(_variant_key(r), []).append(r)

    order: List[Tuple[str, str]] = [
        ("baseline", "baseline"),
        ("feature_extraction", "+ 特征提取增强"),
        ("feature_fusion", "+ 特征融合增强"),
        ("classification_calibration", "+ 分类校准"),
        ("all_modules", "全部模块"),
    ]

    out_rows: List[dict] = []
    for key, label in order:
        cands = by_variant.get(key, [])
        if not cands:
            out_rows.append(
                {
                    "variant_key": key,
                    "variant_name": label,
                    "model_name": "",
                    "exp_name": "",
                    "ablation_tags": "",
                    "status": "not_found",
                    "precision": "",
                    "recall": "",
                    "map50": "",
                    "map50_95": "",
                    "tp": "",
                    "fp": "",
                    "fn": "",
                    "selection_note": "no available model in this group",
                }
            )
            continue

        best = sorted(cands, key=lambda x: _to_float(x.get("map50", "")), reverse=True)[0]
        out_rows.append(
            {
                "variant_key": key,
                "variant_name": label,
                "model_name": best.get("model_name", ""),
                "exp_name": best.get("exp_name", ""),
                "ablation_tags": best.get("ablation_tags", ""),
                "status": "ok",
                "precision": best.get("precision", ""),
                "recall": best.get("recall", ""),
                "map50": best.get("map50", ""),
                "map50_95": best.get("map50_95", ""),
                "tp": best.get("tp", ""),
                "fp": best.get("fp", ""),
                "fn": best.get("fn", ""),
                "selection_note": "best mAP@0.5 within this group",
            }
        )

    # keep a mixed group for traceability (secondary)
    mixed_rows = by_variant.get("mixed", [])
    for r in sorted(mixed_rows, key=lambda x: _to_float(x.get("map50", "")), reverse=True):
        out_rows.append(
            {
                "variant_key": "mixed",
                "variant_name": "混合增强(自动归类)",
                "model_name": r.get("model_name", ""),
                "exp_name": r.get("exp_name", ""),
                "ablation_tags": r.get("ablation_tags", ""),
                "status": "ok",
                "precision": r.get("precision", ""),
                "recall": r.get("recall", ""),
                "map50": r.get("map50", ""),
                "map50_95": r.get("map50_95", ""),
                "tp": r.get("tp", ""),
                "fp": r.get("fp", ""),
                "fn": r.get("fn", ""),
                "selection_note": "mixed module groups (not strict single-module ablation)",
            }
        )

    fields = [
        "variant_key",
        "variant_name",
        "model_name",
        "exp_name",
        "ablation_tags",
        "status",
        "precision",
        "recall",
        "map50",
        "map50_95",
        "tp",
        "fp",
        "fn",
        "selection_note",
    ]
    out_csv = tables_dir / "ablation_main.csv"
    write_csv(out_csv, out_rows, fields)

    md_rows = []
    for r in out_rows:
        md_rows.append(
            {
                "variant": r["variant_name"],
                "model": r["model_name"],
                "status": r["status"],
                "P": r["precision"],
                "R": r["recall"],
                "mAP@0.5": r["map50"],
                "mAP@0.5:0.95": r["map50_95"],
            }
        )
    (tables_dir / "ablation_main.md").write_text(
        markdown_table(md_rows, ["variant", "model", "status", "P", "R", "mAP@0.5", "mAP@0.5:0.95"]),
        encoding="utf-8",
    )

    print(f"[done] ablation_main.csv -> {out_csv}")


if __name__ == "__main__":
    main()
