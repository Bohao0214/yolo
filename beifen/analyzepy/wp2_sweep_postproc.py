"""WP2: sweep inference-time postprocess knobs (conf / nms_iou / max_det) for deployment selection.

Writes a single CSV + one JSON into report_dir (either a new timestamp dir or an existing --report_dir).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from analyzepy.eval_core import ensure_ultralytics, evaluate_dataset, infer_label_dir, list_images, run_inference


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="WP2 sweep (conf, nms_iou, max_det).")
    p.add_argument("--weights", type=str, default="/home/ubuntu/project/deduibi/yolo/models/defect/exp_2601261106/best/best.pt")
    p.add_argument("--image_dir", type=str, default="/home/ubuntu/project/deduibi/yolo/dataset/yolo/datasetm6c/images/val")
    p.add_argument("--out_root", type=str, default="/home/ubuntu/project/deduibi/yolo/analysis")
    p.add_argument("--report_dir", type=str, default="", help="If set, write outputs into this existing directory.")

    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--tp_iou", type=float, default=0.5)

    p.add_argument("--conf_min", type=float, default=0.05)
    p.add_argument("--conf_max", type=float, default=0.50)
    p.add_argument("--conf_step", type=float, default=0.05)

    p.add_argument("--nms_min", type=float, default=0.30)
    p.add_argument("--nms_max", type=float, default=0.80)
    p.add_argument("--nms_step", type=float, default=0.10)

    p.add_argument("--max_det_list", type=str, default="100,300,1000", help="Comma-separated.")
    p.add_argument("--device", type=str, default="")
    p.add_argument("--infer_chunk", type=int, default=0, help="Inference chunk size to limit GPU memory (0 = all).")
    return p.parse_args()


def make_report_dir(out_root: Path, report_dir: Optional[str]) -> Path:
    if report_dir:
        rd = Path(report_dir)
        rd.mkdir(parents=True, exist_ok=True)
        return rd
    ts = dt.datetime.now().strftime("report_%y%m%d%H%M")
    rd = out_root / ts
    rd.mkdir(parents=True, exist_ok=False)
    return rd


def frange(a: float, b: float, step: float) -> List[float]:
    xs = []
    v = a
    while v <= b + 1e-9:
        xs.append(round(v, 6))
        v += step
    return xs


def safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b else 0.0


def main() -> None:
    args = parse_args()
    ensure_ultralytics()

    image_dir = Path(args.image_dir)
    out_root = Path(args.out_root)
    label_dir = infer_label_dir(image_dir)

    images = list_images(image_dir)
    if not images:
        raise RuntimeError(f"No images found under: {image_dir}")
    if not label_dir.exists():
        raise FileNotFoundError(f"Label directory not found: {label_dir}")

    report_dir = make_report_dir(out_root, args.report_dir if args.report_dir else None)

    confs = frange(float(args.conf_min), float(args.conf_max), float(args.conf_step))
    nms_ious = frange(float(args.nms_min), float(args.nms_max), float(args.nms_step))
    max_dets = [int(x.strip()) for x in str(args.max_det_list).split(",") if x.strip()]

    # reuse single model to reduce overhead
    from ultralytics import YOLO

    model = YOLO(str(args.weights))

    rows: List[dict] = []
    for conf in confs:
        for nms_iou in nms_ious:
            for max_det in max_dets:
                preds = run_inference(
                    Path(args.weights),
                    images,
                    int(args.imgsz),
                    float(conf),
                    float(nms_iou),
                    int(max_det),
                    int(args.batch),
                    str(args.device),
                    model=model,
                    chunk_size=int(args.infer_chunk),
                )
                _, _, _, totals = evaluate_dataset(
                    images=images,
                    label_dir=label_dir,
                    preds=preds,
                    imgsz=int(args.imgsz),
                    tp_iou=float(args.tp_iou),
                    near_iou_low=0.10,
                    near_iou_high=min(0.50, float(args.tp_iou) - 1e-6),
                )

                tp = int(totals.get("tp", 0))
                fp_s = int(totals.get("fp_strict", 0))
                fp_n = int(totals.get("fp_near", 0))
                fn = int(totals.get("fn", 0))
                n_img = int(totals.get("images", len(images)))

                recall = safe_div(tp, tp + fn)
                precision = safe_div(tp, tp + fp_s + fp_n)
                fp_per_img = safe_div(fp_s + fp_n, n_img)

                row = {
                    "conf": float(conf),
                    "nms_iou": float(nms_iou),
                    "max_det": int(max_det),
                    "tp": tp,
                    "fn": fn,
                    "fp_strict": fp_s,
                    "fp_near": fp_n,
                    "recall": recall,
                    "precision": precision,
                    "fp_per_image": fp_per_img,
                }
                rows.append(row)

    # Rank: maximize recall first, then minimize fp_per_image, then maximize precision
    def rank_key(r: dict) -> Tuple[float, float, float]:
        return (float(r["recall"]), -float(r["fp_per_image"]), float(r["precision"]))

    best = max(rows, key=rank_key) if rows else None

    import csv

    csv_path = report_dir / "sweep_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)

    meta = {
        "weights": str(args.weights),
        "image_dir": str(image_dir),
        "label_dir": str(label_dir),
        "imgsz": int(args.imgsz),
        "batch": int(args.batch),
        "tp_iou": float(args.tp_iou),
        "grid": {"conf": confs, "nms_iou": nms_ious, "max_det": max_dets},
        "best": best,
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
    }
    (report_dir / "sweep_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[WP2] report_dir = {report_dir}")
    print(f"[WP2] best = {best}")


if __name__ == "__main__":
    main()
"""""
python /home/ubuntu/project/deduibi/yolo/analyzepy/wp2_sweep_postproc.py \
  --weights /home/ubuntu/project/deduibi/yolo/models/defect/exp_2601261106/best/best.pt \
  --image_dir /home/ubuntu/project/deduibi/yolo/dataset/yolo/datasetm6c/images/val \
  --out_root /home/ubuntu/project/deduibi/yolo/analysis \
  --report_dir /home/ubuntu/project/deduibi/yolo/analysis/report_2601282104 \
  --imgsz 640 --batch 4 --tp_iou 0.3 \
  --infer_chunk 16

"""""
