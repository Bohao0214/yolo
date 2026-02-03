"""Compare TP@0.3 vs TP@0.5 on the same inference outputs.

Outputs:
- tp03_not_tp05_stats.csv
- tp03_not_tp05_samples.csv
- run_summary.json
- (optional) topk_tp03_not_tp05.png

Example:
python /home/ubuntu/project/deduibi/yolo/analyzepy/compare0305.py \
  --weights /home/ubuntu/project/deduibi/yolo/models/defect/exp_2601261106/best/best.pt \
  --image_dir /home/ubuntu/project/deduibi/yolo/dataset/yolo/datasetm6c/images/val \
  --out_root /home/ubuntu/project/deduibi/yolo/analysis \
  --report_dir /home/ubuntu/project/deduibi/yolo/analysis/report_2601282104 \
  --imgsz 640 --batch 4 --infer_chunk 8

"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from analyzepy.eval_core import (
    BoxRecord,
    bucket_name_from_min_side,
    ensure_ultralytics,
    infer_label_dir,
    letterbox_transform_xyxy,
    list_images,
    load_labels_for_image,
    run_inference,
    greedy_match,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare TP@0.3 vs TP@0.5 on same predictions.")
    p.add_argument("--weights", type=str, default="/home/ubuntu/project/deduibi/yolo/models/defect/exp_2601261106/best/best.pt")
    p.add_argument("--image_dir", type=str, default="/home/ubuntu/project/deduibi/yolo/dataset/yolo/datasetm6c/images/val")
    p.add_argument("--out_root", type=str, default="/home/ubuntu/project/deduibi/yolo/analysis")
    p.add_argument("--report_dir", type=str, default="")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--conf", type=float, default=0.2)
    p.add_argument("--nms_iou", type=float, default=0.5)
    p.add_argument("--max_det", type=int, default=300)
    p.add_argument("--device", type=str, default="")
    p.add_argument("--infer_chunk", type=int, default=16, help="Inference chunk size to limit GPU memory (0 = all).")
    p.add_argument("--topk", type=int, default=20)
    p.add_argument("--highlight_pctl", type=float, default=85.0)
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


def bbox_to_str(box: Tuple[float, float, float, float]) -> str:
    return ",".join(f"{v:.2f}" for v in box)


def compute_highlight_flag(img_bgr: np.ndarray, box: Tuple[float, float, float, float], pctl: float) -> bool:
    x1, y1, x2, y2 = map(int, box)
    x1 = max(0, min(img_bgr.shape[1] - 1, x1))
    x2 = max(0, min(img_bgr.shape[1], x2))
    y1 = max(0, min(img_bgr.shape[0] - 1, y1))
    y2 = max(0, min(img_bgr.shape[0], y2))
    if x2 <= x1 or y2 <= y1:
        return False
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    crop = gray[y1:y2, x1:x2]
    if crop.size == 0:
        return False
    thresh = float(np.percentile(gray, pctl))
    return float(crop.mean()) > thresh


def draw_overlay(
    img: np.ndarray,
    gt_box: Tuple[float, float, float, float],
    pred_box: Tuple[float, float, float, float],
    score: float,
    iou: float,
    title: str,
) -> np.ndarray:
    out = img.copy()
    x1, y1, x2, y2 = map(int, gt_box)
    cv2.rectangle(out, (x1, y1), (x2, y2), (0, 200, 0), 2)
    x1, y1, x2, y2 = map(int, pred_box)
    cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 220), 2)
    label = f"score={score:.2f} iou={iou:.2f}"
    cv2.putText(out, label, (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 220), 1)
    cv2.putText(out, title, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    cv2.putText(out, title, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (10, 10, 10), 1)
    return out


def make_mosaic(images: List[np.ndarray], cols: int = 5, pad: int = 2) -> np.ndarray:
    if not images:
        return np.zeros((10, 10, 3), dtype=np.uint8)
    h = max(im.shape[0] for im in images)
    w = max(im.shape[1] for im in images)
    rows = (len(images) + cols - 1) // cols
    canvas = np.zeros((rows * h + pad * (rows - 1), cols * w + pad * (cols - 1), 3), dtype=np.uint8)
    canvas[:] = 0
    for idx, im in enumerate(images):
        r = idx // cols
        c = idx % cols
        y = r * (h + pad)
        x = c * (w + pad)
        resized = cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA)
        canvas[y : y + h, x : x + w] = resized
    return canvas


def main() -> None:
    args = parse_args()
    ensure_ultralytics()

    weights = Path(args.weights)
    image_dir = Path(args.image_dir)
    out_root = Path(args.out_root)
    if not weights.exists():
        raise FileNotFoundError(f"Weights not found: {weights}")
    if not image_dir.exists():
        raise FileNotFoundError(f"Image dir not found: {image_dir}")

    label_dir = infer_label_dir(image_dir)
    if not label_dir.exists():
        raise FileNotFoundError(f"Label dir not found: {label_dir}")

    images = list_images(image_dir)
    if not images:
        raise RuntimeError(f"No images found under: {image_dir}")

    report_dir = make_report_dir(out_root, args.report_dir if args.report_dir else None)

    preds_by_img = run_inference(
        weights,
        images,
        int(args.imgsz),
        float(args.conf),
        float(args.nms_iou),
        int(args.max_det),
        int(args.batch),
        str(args.device),
        chunk_size=int(args.infer_chunk),
    )
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    samples: List[dict] = []
    bucket_stats: Dict[Tuple[str, int], dict] = {}
    matched_iou_values: List[float] = []
    total_tp03 = 0
    total_tp03_not_tp05 = 0
    pred_area_ratios: List[float] = []
    tp03_not_tp05_area_ratios: List[float] = []

    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            raise FileNotFoundError(f"Failed to read image: {img_path}")
        H, W = img.shape[:2]
        image_id = img_path.stem
        label_path = label_dir / f"{image_id}.txt"
        gts = load_labels_for_image(label_path, W, H, image_id, img_path)
        preds = preds_by_img.get(str(img_path), [])

        matches03, _ = greedy_match(gts, preds, 0.3, 0.10, 0.30)
        matches05, _ = greedy_match(gts, preds, 0.5, 0.10, 0.50 - 1e-6)

        tp_pred_03 = {m.pred_index: m for m in matches03 if m.match_type == "TP"}
        tp_pred_05 = {m.pred_index: m for m in matches05 if m.match_type == "TP"}
        total_tp03 += len(tp_pred_03)

        # Pred area ratios (all preds for this image)
        for pred in preds:
            x1, y1, x2, y2 = pred.bbox_xyxy
            area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            pred_area_ratios.append(area / float(W * H))

        for pred_idx, m03 in tp_pred_03.items():
            if pred_idx in tp_pred_05:
                continue
            pred = preds[pred_idx]
            total_tp03_not_tp05 += 1
            gt_idx = m03.gt_index if m03.gt_index is not None else m03.best_gt_index
            if gt_idx is None:
                continue
            gt = gts[int(gt_idx)]
            matched_iou_values.append(float(m03.best_iou))
            # area ratio for TP@0.3 not TP@0.5 (use GT box)
            x1, y1, x2, y2 = gt.bbox_xyxy
            area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            tp03_not_tp05_area_ratios.append(area / float(W * H))

            lb = letterbox_transform_xyxy(gt.bbox_xyxy, W, H, int(args.imgsz))
            gw = max(0.0, lb[2] - lb[0])
            gh = max(0.0, lb[3] - lb[1])
            size_bin = bucket_name_from_min_side(min(gw, gh))
            hl_flag = compute_highlight_flag(img, gt.bbox_xyxy, float(args.highlight_pctl))

            samples.append(
                {
                    "img_id": image_id,
                    "img_path": str(img_path),
                    "gt_xyxy": bbox_to_str(gt.bbox_xyxy),
                    "pred_xyxy": bbox_to_str(pred.bbox_xyxy),
                    "best_iou": float(m03.best_iou),
                    "score": float(pred.score or 0.0),
                    "size_bin": size_bin,
                    "highlight_flag": int(hl_flag),
                }
            )

            key = (size_bin, int(hl_flag))
            if key not in bucket_stats:
                bucket_stats[key] = {
                    "size_bin": size_bin,
                    "highlight_flag": int(hl_flag),
                    "count": 0,
                    "sum_iou": 0.0,
                    "sum_score": 0.0,
                }
            bucket_stats[key]["count"] += 1
            bucket_stats[key]["sum_iou"] += float(m03.best_iou)
            bucket_stats[key]["sum_score"] += float(pred.score or 0.0)

    stats_rows: List[dict] = []
    total_count = sum(v["count"] for v in bucket_stats.values())
    total_tp03_safe = max(1, total_tp03)
    for key, stat in bucket_stats.items():
        count = stat["count"]
        stats_rows.append(
            {
                "size_bin": stat["size_bin"],
                "highlight_flag": stat["highlight_flag"],
                "count": count,
                "ratio": float(count) / float(total_count) if total_count else 0.0,
                "mean_iou": stat["sum_iou"] / count if count else 0.0,
                "mean_score": stat["sum_score"] / count if count else 0.0,
            }
        )

    stats_rows.insert(
        0,
        {
            "size_bin": "ALL",
            "highlight_flag": -1,
            "count": total_tp03_not_tp05,
            "ratio": float(total_tp03_not_tp05) / float(total_tp03_safe),
            "mean_iou": float(np.mean(matched_iou_values)) if matched_iou_values else 0.0,
            "mean_score": float(np.mean([s["score"] for s in samples])) if samples else 0.0,
        },
    )

    stats_path = report_dir / "tp03_not_tp05_stats.csv"
    with stats_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["size_bin", "highlight_flag", "count", "ratio", "mean_iou", "mean_score"])
        w.writeheader()
        for row in stats_rows:
            w.writerow(row)

    samples_path = report_dir / "tp03_not_tp05_samples.csv"
    with samples_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["img_id", "img_path", "gt_xyxy", "pred_xyxy", "best_iou", "score", "size_bin", "highlight_flag"],
        )
        w.writeheader()
        for row in samples:
            w.writerow(row)

    # Optional mosaic
    if samples:
        topk = sorted(samples, key=lambda r: (-float(r["best_iou"]), -float(r["score"])))[0 : int(args.topk)]
        mosaics: List[np.ndarray] = []
        for item in topk:
            img = cv2.imread(item["img_path"])
            if img is None:
                continue
            gt = tuple(float(x) for x in item["gt_xyxy"].split(","))
            pred = tuple(float(x) for x in item["pred_xyxy"].split(","))
            mosaics.append(
                draw_overlay(
                    img,
                    gt,
                    pred,
                    float(item["score"]),
                    float(item["best_iou"]),
                    "TP@0.3 not TP@0.5",
                )
            )
        mosaic = make_mosaic(mosaics, cols=5)
        cv2.imwrite(str(report_dir / "topk_tp03_not_tp05.png"), mosaic)

    def pct(vals: List[float], q: float) -> float:
        if not vals:
            return 0.0
        return float(np.percentile(np.array(vals, dtype=np.float32), q))

    run_summary = {
        "weights": str(weights),
        "image_dir": str(image_dir),
        "label_dir": str(label_dir),
        "imgsz": int(args.imgsz),
        "batch": int(args.batch),
        "conf": float(args.conf),
        "nms_iou": float(args.nms_iou),
        "max_det": int(args.max_det),
        "tp_iou_list": [0.3, 0.5],
        "highlight_pctl": float(args.highlight_pctl),
        "bbox_area_ratio_summary": {
            "pred": {
                "p10": pct(pred_area_ratios, 10),
                "p50": pct(pred_area_ratios, 50),
                "p90": pct(pred_area_ratios, 90),
            },
            "tp03_not_tp05": {
                "p10": pct(tp03_not_tp05_area_ratios, 10),
                "p50": pct(tp03_not_tp05_area_ratios, 50),
                "p90": pct(tp03_not_tp05_area_ratios, 90),
            },
        },
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "report_dir": str(report_dir),
    }
    (report_dir / "run_summary.json").write_text(json.dumps(run_summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[compare0305] report_dir = {report_dir}")
    print(f"[compare0305] samples = {len(samples)}")


if __name__ == "__main__":
    main()
