#!/usr/bin/env python3
"""
仅回填图像级指标（不重新训练）。

用途：
1) 对已经有 summary.csv 的实验目录，逐行读取 best.pt 与 data.yaml。
2) 只做 test 集图像级统计（有框=有缺陷），写回 image_level/test_image_level.*。
3) 更新 summary.csv / summary.json 的图像级字段，保留原检测指标。

示例：
python /home/ubuntu/hpproject/yolo/analyze/code/backfill_image_level_from_summary.py \
  --summary-csv /home/ubuntu/hpproject/yolo/experiments/industrial_bs_sweep_260408_2114_01/summary.csv \
  --image-eval-device cpu \
  --image-eval-batch 1
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
from pathlib import Path
from typing import Any, Dict, List

import yaml
from ultralytics import YOLO

from src.eval import compute_image_level_results, save_image_level_report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Backfill image-level metrics from existing summary.csv")
    p.add_argument("--summary-csv", type=str, required=True, help="Path to summary.csv")
    p.add_argument("--image-conf-thr", type=float, default=0.25, help="Image-level defect decision threshold")
    p.add_argument("--image-iou-match", type=float, default=0.3, help="IoU for GT-pred match")
    p.add_argument("--metric-conf", type=float, default=0.01, help="Predict conf for proposal generation")
    p.add_argument("--image-eval-batch", type=int, default=1, help="Predict batch size for image-level eval")
    p.add_argument("--image-eval-device", type=str, default="cpu", help='Predict device for image-level eval, e.g. "cpu" or "0"')
    p.add_argument("--nms-iou", type=float, default=0.7, help="NMS IoU")
    p.add_argument("--max-det", type=int, default=300, help="Max detections")
    p.add_argument("--only-status", type=str, default="", help='Optional filter, e.g. "failed" to process only failed rows')
    p.add_argument(
        "--dataset-contains",
        type=str,
        default="",
        help='Optional substring filter for dataset name, e.g. "neudet" (case-insensitive)',
    )
    return p.parse_args()


def safe_ratio(n: int, d: int) -> float:
    return float(n) / float(d) if d > 0 else 0.0


def summarize_image_level(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    tp = sum(1 for x in items if str(x.get("outcome", "")) == "TP")
    fn = sum(1 for x in items if str(x.get("outcome", "")) == "FN")
    fp = sum(1 for x in items if str(x.get("outcome", "")) == "FP")
    tn = sum(1 for x in items if str(x.get("outcome", "")) == "TN")
    obj_fn = sum(int(x.get("obj_fn", 0) or 0) for x in items)
    obj_fp = sum(int(x.get("obj_fp", 0) or 0) for x in items)
    return {
        "img_tp": int(tp),
        "img_fn": int(fn),
        "img_fp": int(fp),
        "img_tn": int(tn),
        "image_precision": safe_ratio(tp, tp + fp),
        "image_recall": safe_ratio(tp, tp + fn),
        "image_fpr": safe_ratio(fp, fp + tn),
        "obj_fn_total": int(obj_fn),
        "obj_fp_total": int(obj_fp),
    }


def read_yaml(path: Path) -> Dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"yaml top-level not mapping: {path}")
    return data


def resolve_data_root(data_yaml: Path, info: Dict[str, Any]) -> Path:
    root_raw = str(info.get("path", "")).strip()
    if root_raw:
        p = Path(root_raw)
        if not p.is_absolute():
            p = (data_yaml.parent / p).resolve()
        return p
    return data_yaml.parent.resolve()


def resolve_split_sources(data_yaml: Path, split: str) -> List[Path]:
    info = read_yaml(data_yaml)
    root = resolve_data_root(data_yaml, info)
    v = info.get(split)
    if v is None:
        return []
    refs: List[str] = []
    if isinstance(v, str):
        refs = [x.strip() for x in v.split(",") if x.strip()]
    elif isinstance(v, (list, tuple)):
        refs = [str(x).strip() for x in v if str(x).strip()]
    else:
        refs = [str(v).strip()]
    out: List[Path] = []
    for r in refs:
        p = Path(r)
        if not p.is_absolute():
            p = (root / p).resolve()
        out.append(p)
    return out


def fmt_float(x: Any) -> Any:
    try:
        return f"{float(x):.6f}"
    except Exception:
        return x


def main() -> None:
    args = parse_args()
    summary_csv = Path(args.summary_csv).expanduser().resolve()
    if not summary_csv.exists():
        raise FileNotFoundError(f"summary.csv not found: {summary_csv}")

    rows: List[Dict[str, Any]] = []
    with summary_csv.open("r", newline="", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        fieldnames = list(rd.fieldnames or [])
        for r in rd:
            rows.append(dict(r))

    if not rows:
        print("[done] empty summary, nothing to do")
        return

    print(f"[info] rows={len(rows)} summary={summary_csv}")
    for i, row in enumerate(rows, 1):
        if args.only_status and str(row.get("status", "")).strip() != args.only_status:
            continue

        dataset = str(row.get("dataset", "")).strip() or f"row{i}"
        if args.dataset_contains and args.dataset_contains.lower() not in dataset.lower():
            continue
        bs = str(row.get("batch", "")).strip()
        best_pt = Path(str(row.get("best_pt", "")).strip())
        data_yaml = Path(str(row.get("data_yaml", "")).strip())
        run_dir = Path(str(row.get("train_run_dir", "")).strip())
        tag = f"{dataset} bs={bs}"
        print(f"[run] ({i}/{len(rows)}) {tag}")

        model = None
        try:
            if not best_pt.exists():
                raise FileNotFoundError(f"best.pt not found: {best_pt}")
            if not data_yaml.exists():
                raise FileNotFoundError(f"data.yaml not found: {data_yaml}")
            if not run_dir.exists():
                run_dir.mkdir(parents=True, exist_ok=True)

            model = YOLO(str(best_pt))
            sources = resolve_split_sources(data_yaml, "test")
            all_items: List[Dict[str, Any]] = []
            for src in sources:
                if not src.exists():
                    print(f"[warn] missing test source: {src}")
                    continue
                items = compute_image_level_results(
                    model=model,
                    source=src,
                    conf_threshold=float(args.image_conf_thr),
                    iou_match=float(args.image_iou_match),
                    metric_conf=float(args.metric_conf),
                    batch=max(1, int(args.image_eval_batch)),
                    device=str(args.image_eval_device),
                    nms_iou=float(args.nms_iou),
                    max_det=int(args.max_det),
                    split="test",
                    vis_root=None,
                    save_visuals=False,
                )
                all_items.extend(items)

            if not all_items:
                row["status"] = "partial"
                row["error"] = "image-level items empty (test split missing or no predictions)"
                print(f"[warn] {tag} no image items")
                continue

            stats = summarize_image_level(all_items)
            row.update({k: fmt_float(v) for k, v in stats.items()})
            if str(row.get("status", "")).strip() in ("failed", "partial"):
                row["status"] = "ok"
                row["error"] = ""

            meta = {
                "dataset": dataset,
                "data_yaml": str(data_yaml),
                "split": "test",
                "batch": int(bs) if bs else 0,
                "conf_threshold": float(args.image_conf_thr),
                "iou_match": float(args.image_iou_match),
                "metric_conf": float(args.metric_conf),
                "nms_iou": float(args.nms_iou),
                "max_det": int(args.max_det),
                "image_eval_batch": int(args.image_eval_batch),
                "image_eval_device": str(args.image_eval_device),
            }
            save_image_level_report(run_dir / "image_level", "test", all_items, meta)
            print(
                "[done] {tag} img_p={p} img_r={r}".format(
                    tag=tag,
                    p=row.get("image_precision", "0"),
                    r=row.get("image_recall", "0"),
                )
            )

        except Exception as exc:
            row["status"] = "partial"
            row["error"] = f"image_level_failed: {type(exc).__name__}: {exc}"
            print(f"[error] {tag}: {row['error']}")
        finally:
            try:
                if model is not None:
                    del model
            except Exception:
                pass
            gc.collect()

        with summary_csv.open("w", newline="", encoding="utf-8") as f:
            wt = csv.DictWriter(f, fieldnames=fieldnames)
            wt.writeheader()
            wt.writerows(rows)
        (summary_csv.with_suffix(".json")).write_text(
            json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    print(f"[all_done] {summary_csv}")
    print(f"[all_done] {summary_csv.with_suffix('.json')}")


if __name__ == "__main__":
    main()
