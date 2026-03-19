from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
from pathlib import Path
from typing import Dict, List, Optional


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
METRIC_KEYS = ("fn_total", "fp_total", "val_fn", "val_fp", "test_fn", "test_fp")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Count image-level FN/FP files for each hparam case directory under "
            "experiments/<hp_root>/<case>/exp_xxx/{test_vis,val_vis}/image_{fn,fp}."
        )
    )
    p.add_argument(
        "--hp_root",
        type=str,
        default="/home/ubuntu/hpproject/yolo/experiments/a3c5/datasetm6c",
        help="HP root directory that contains case folders.",
    )
    p.add_argument(
        "--case_glob",
        type=str,
        default="defect241__a3__c5__hp__*",
        help="Case directory glob under hp_root.",
    )
    p.add_argument(
        "--exp_name",
        type=str,
        default="",
        help=(
            "Exact run directory name in each case (e.g. exp_2602240148). "
            "Leave empty to auto-select the latest exp_* in each case."
        ),
    )
    p.add_argument(
        "--fallback_latest",
        action="store_true",
        help="Only used when --exp_name is set: if missing, fallback to latest exp_* in that case.",
    )
    p.add_argument(
        "--out_root",
        type=str,
        default="/home/ubuntu/hpproject/yolo/analyze/result",
        help="Output root for report files.",
    )
    return p.parse_args()


def make_report_dir(out_root: Path) -> Path:
    ts = dt.datetime.now().strftime("report_%Y%m%d%H%M")
    base = out_root / ts
    if not base.exists():
        base.mkdir(parents=True, exist_ok=False)
        return base
    idx = 1
    while True:
        cand = out_root / f"{ts}_{idx:02d}"
        if not cand.exists():
            cand.mkdir(parents=True, exist_ok=False)
            return cand
        idx += 1


def count_images(path: Path) -> int:
    if not path.exists() or not path.is_dir():
        return 0
    return sum(1 for p in path.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS)


def pick_run_dir(case_dir: Path, exp_name: str, fallback_latest: bool) -> Optional[Path]:
    runs = [p for p in case_dir.glob("exp_*") if p.is_dir()]
    runs.sort(key=lambda p: p.stat().st_mtime)

    exp_name = str(exp_name).strip()
    if not exp_name:
        # Auto mode: each case may have different exp_*, choose latest.
        return runs[-1] if runs else None

    target = case_dir / exp_name
    if target.exists() and target.is_dir():
        return target

    if fallback_latest:
        return runs[-1] if runs else None
    return None


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    headers = [
        "experiment_name",
        "fn_total",
        "fp_total",
        "val_fn",
        "val_fp",
        "test_fn",
        "test_fp",
        "summary_type",
        "metric_name",
        "metric_value",
        "status",
        "run_name",
        "run_dir",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_minima_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    headers = ["experiment_name", "metric_name", "metric_value"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "experiment_name": row.get("experiment_name", ""),
                    "metric_name": row.get("metric_name", ""),
                    "metric_value": row.get("metric_value", ""),
                }
            )


def main() -> None:
    args = parse_args()

    hp_root = Path(args.hp_root).resolve()
    out_root = Path(args.out_root).resolve()

    if not hp_root.exists() or not hp_root.is_dir():
        raise FileNotFoundError(f"hp_root not found: {hp_root}")

    case_dirs = sorted([p for p in hp_root.glob(args.case_glob) if p.is_dir()], key=lambda p: p.name)

    rows: List[Dict[str, object]] = []
    for case_dir in case_dirs:
        run_dir = pick_run_dir(case_dir, args.exp_name, args.fallback_latest)

        if run_dir is None:
            rows.append(
                {
                    "experiment_name": case_dir.name,
                    "fn_total": 0,
                    "fp_total": 0,
                    "val_fn": 0,
                    "val_fp": 0,
                    "test_fn": 0,
                    "test_fp": 0,
                    "summary_type": "case",
                    "metric_name": "",
                    "metric_value": "",
                    "run_name": "",
                    "status": f"missing:{args.exp_name or 'exp_*'}",
                    "run_dir": "",
                }
            )
            continue

        test_fn = count_images(run_dir / "test_vis" / "image_fn")
        test_fp = count_images(run_dir / "test_vis" / "image_fp")
        val_fn = count_images(run_dir / "val_vis" / "image_fn")
        val_fp = count_images(run_dir / "val_vis" / "image_fp")
        fn_total = val_fn + test_fn
        fp_total = val_fp + test_fp

        rows.append(
            {
                "experiment_name": case_dir.name,
                "fn_total": fn_total,
                "fp_total": fp_total,
                "val_fn": val_fn,
                "val_fp": val_fp,
                "test_fn": test_fn,
                "test_fp": test_fp,
                "summary_type": "case",
                "metric_name": "",
                "metric_value": "",
                "run_name": run_dir.name,
                "status": "ok",
                "run_dir": str(run_dir),
            }
        )

    minima_rows: List[Dict[str, object]] = []
    ok_rows = [r for r in rows if r["status"] == "ok"]
    for metric in METRIC_KEYS:
        if not ok_rows:
            break
        winner = min(ok_rows, key=lambda r: (int(r[metric]), str(r["experiment_name"])))
        minima_rows.append(
            {
                "experiment_name": winner["experiment_name"],
                "fn_total": "",
                "fp_total": "",
                "val_fn": "",
                "val_fp": "",
                "test_fn": "",
                "test_fp": "",
                "summary_type": "min",
                "metric_name": metric,
                "metric_value": int(winner[metric]),
                "status": "ok",
                "run_name": winner.get("run_name", ""),
                "run_dir": winner.get("run_dir", ""),
            }
        )

    report_dir = make_report_dir(out_root)
    csv_path = report_dir / "hparam_case_fpfn_counts.csv"
    minima_csv_path = report_dir / "hparam_case_fpfn_minima.csv"
    json_path = report_dir / "hparam_case_fpfn_summary.json"
    md_path = report_dir / "hparam_case_fpfn_readme.md"

    write_csv(csv_path, rows + minima_rows)
    write_minima_csv(minima_csv_path, minima_rows)

    summary = {
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        "hp_root": str(hp_root),
        "case_glob": args.case_glob,
        "exp_name": args.exp_name or "<auto_latest>",
        "fallback_latest": bool(args.fallback_latest),
        "case_count": len(rows),
        "ok_count": len(ok_rows),
        "sum_test_fn": int(sum(int(r["test_fn"]) for r in ok_rows)),
        "sum_test_fp": int(sum(int(r["test_fp"]) for r in ok_rows)),
        "sum_val_fn": int(sum(int(r["val_fn"]) for r in ok_rows)),
        "sum_val_fp": int(sum(int(r["val_fp"]) for r in ok_rows)),
        "sum_fn_total": int(sum(int(r["fn_total"]) for r in ok_rows)),
        "sum_fp_total": int(sum(int(r["fp_total"]) for r in ok_rows)),
        "minima": [
            {
                "experiment_name": r["experiment_name"],
                "metric_name": r["metric_name"],
                "metric_value": int(r["metric_value"]),
            }
            for r in minima_rows
        ],
        "report_csv": str(csv_path),
        "minima_csv": str(minima_csv_path),
    }
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    md_lines = [
        "# Hparam Case Image FN/FP Counts",
        "",
        f"- hp_root: `{hp_root}`",
        f"- case_glob: `{args.case_glob}`",
        f"- exp_name: `{args.exp_name or '<auto_latest>'}`",
        f"- fallback_latest: `{args.fallback_latest}`",
        f"- case_count: `{len(rows)}`",
        f"- ok_count: `{len(ok_rows)}`",
        f"- csv: `{csv_path}`",
        f"- minima_csv: `{minima_csv_path}`",
        "",
        "## Columns",
        "",
        "- experiment_name",
        "- fn_total, fp_total",
        "- val_fn, val_fp",
        "- test_fn, test_fp",
        "- summary_type, metric_name, metric_value",
        "- status, run_name, run_dir",
        "",
        "## Summary Rows",
        "",
        "- summary_type=min rows are appended at the end of the main csv",
        "- each min row contains: experiment_name + metric_name + metric_value",
    ]
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"[DONE] report_dir: {report_dir}")
    print(f"[DONE] csv: {csv_path}")
    print(f"[DONE] minima_csv: {minima_csv_path}")
    print(f"[DONE] cases={len(rows)} ok={len(ok_rows)}")


if __name__ == "__main__":
    main()
