from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import shutil
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Move old hparam experiment dirs from experiments/yolo11/<prefix>__* "
            "to experiments/yolo11/<prefix>/<prefix>__*."
        )
    )
    p.add_argument(
        "--experiments_root",
        type=str,
        default="/home/ubuntu/hpproject/yolo/experiments/yolo11",
        help="Root directory that contains experiment folders.",
    )
    p.add_argument(
        "--prefix",
        type=str,
        default="defect241__a3__c5__hp",
        help="Experiment family prefix. Source dirs are matched as '<prefix>__*'.",
    )
    p.add_argument(
        "--out_root",
        type=str,
        default="/home/ubuntu/hpproject/yolo/analyze/result",
        help="Output root for migration report (csv/json/md).",
    )
    p.add_argument(
        "--dry_run",
        action="store_true",
        help="Only print planned moves, do not actually move directories.",
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


def write_csv(path: Path, rows: List[Dict[str, str]]) -> None:
    headers = ["src", "dst", "status", "note"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()

    root = Path(args.experiments_root).resolve()
    prefix = str(args.prefix).strip().strip("/")
    out_root = Path(args.out_root).resolve()

    if not root.exists():
        raise FileNotFoundError(f"experiments_root not found: {root}")

    dst_parent = root / prefix
    src_dirs = sorted(
        [
            p
            for p in root.glob(f"{prefix}__*")
            if p.is_dir() and p.name != prefix
        ],
        key=lambda p: p.name,
    )

    rows: List[Dict[str, str]] = []
    moved = 0
    skipped = 0

    if not args.dry_run:
        dst_parent.mkdir(parents=True, exist_ok=True)

    for src in src_dirs:
        dst = dst_parent / src.name
        if dst.exists():
            rows.append(
                {
                    "src": str(src),
                    "dst": str(dst),
                    "status": "skip_exists",
                    "note": "destination already exists",
                }
            )
            skipped += 1
            continue

        if args.dry_run:
            rows.append(
                {
                    "src": str(src),
                    "dst": str(dst),
                    "status": "dry_run",
                    "note": "not moved",
                }
            )
            continue

        shutil.move(str(src), str(dst))
        rows.append(
            {
                "src": str(src),
                "dst": str(dst),
                "status": "moved",
                "note": "",
            }
        )
        moved += 1

    if not rows:
        rows.append(
            {
                "src": "",
                "dst": str(dst_parent),
                "status": "empty",
                "note": f"no source dirs matched: {prefix}__*",
            }
        )

    report_dir = make_report_dir(out_root)
    csv_path = report_dir / "hparam_migrate.csv"
    json_path = report_dir / "hparam_migrate_summary.json"
    md_path = report_dir / "hparam_migrate_readme.md"

    write_csv(csv_path, rows)

    summary = {
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        "experiments_root": str(root),
        "prefix": prefix,
        "dest_parent": str(dst_parent),
        "dry_run": bool(args.dry_run),
        "matched": len(src_dirs),
        "moved": moved,
        "skipped": skipped,
        "report_csv": str(csv_path),
    }
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    md_lines = [
        "# Hparam Directory Migration",
        "",
        f"- experiments_root: `{root}`",
        f"- prefix: `{prefix}`",
        f"- dest_parent: `{dst_parent}`",
        f"- dry_run: `{args.dry_run}`",
        f"- matched: `{len(src_dirs)}`",
        f"- moved: `{moved}`",
        f"- skipped: `{skipped}`",
        f"- csv: `{csv_path}`",
    ]
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"[DONE] report_dir: {report_dir}")
    print(f"[DONE] csv: {csv_path}")
    print(f"[DONE] moved={moved} skipped={skipped} matched={len(src_dirs)}")


if __name__ == "__main__":
    main()
