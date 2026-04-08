#!/usr/bin/env python3
"""
从已有 image_level/test_image_level.csv 回填 summary.csv（不做推理）。

用法：
python /home/ubuntu/hpproject/yolo/analyze/code/merge_image_level_from_existing.py \
  --summary-csv /home/ubuntu/hpproject/yolo/experiments/industrial_bs_sweep_260408_2114_01/summary.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge image-level stats from existing test_image_level.csv into summary.csv")
    p.add_argument("--summary-csv", type=str, required=True, help="Path to summary.csv")
    p.add_argument("--only-status", type=str, default="", help='Optional filter, e.g. "failed"')
    return p.parse_args()


def safe_ratio(n: int, d: int) -> float:
    return float(n) / float(d) if d > 0 else 0.0


def compute_from_image_level_csv(path: Path) -> Dict[str, Any]:
    tp = fn = fp = tn = 0
    obj_fn = obj_fp = 0
    with path.open("r", newline="", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for r in rd:
            o = str(r.get("outcome", "")).strip().upper()
            if o == "TP":
                tp += 1
            elif o == "FN":
                fn += 1
            elif o == "FP":
                fp += 1
            elif o == "TN":
                tn += 1
            try:
                obj_fn += int(float(str(r.get("obj_fn", "0") or "0")))
            except Exception:
                pass
            try:
                obj_fp += int(float(str(r.get("obj_fp", "0") or "0")))
            except Exception:
                pass

    return {
        "img_tp": f"{tp:.6f}",
        "img_fn": f"{fn:.6f}",
        "img_fp": f"{fp:.6f}",
        "img_tn": f"{tn:.6f}",
        "image_precision": f"{safe_ratio(tp, tp + fp):.6f}",
        "image_recall": f"{safe_ratio(tp, tp + fn):.6f}",
        "image_fpr": f"{safe_ratio(fp, fp + tn):.6f}",
        "obj_fn_total": f"{obj_fn:.6f}",
        "obj_fp_total": f"{obj_fp:.6f}",
    }


def main() -> None:
    args = parse_args()
    summary_csv = Path(args.summary_csv).expanduser().resolve()
    if not summary_csv.exists():
        raise FileNotFoundError(f"summary.csv not found: {summary_csv}")

    rows: List[Dict[str, Any]] = []
    with summary_csv.open("r", newline="", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        fieldnames = list(rd.fieldnames or [])
        rows = [dict(r) for r in rd]

    changed = 0
    scanned = 0
    for row in rows:
        if args.only_status and str(row.get("status", "")).strip() != args.only_status:
            continue
        scanned += 1
        run_dir = Path(str(row.get("train_run_dir", "")).strip())
        image_csv = run_dir / "image_level" / "test_image_level.csv"
        if not image_csv.exists():
            continue

        stats = compute_from_image_level_csv(image_csv)
        row.update(stats)
        if str(row.get("status", "")).strip() in ("failed", "partial"):
            row["status"] = "ok"
            row["error"] = ""
        changed += 1

    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        wt = csv.DictWriter(f, fieldnames=fieldnames)
        wt.writeheader()
        wt.writerows(rows)
    (summary_csv.with_suffix(".json")).write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] scanned={scanned} changed={changed}")
    print(f"[done] {summary_csv}")
    print(f"[done] {summary_csv.with_suffix('.json')}")


if __name__ == "__main__":
    main()
