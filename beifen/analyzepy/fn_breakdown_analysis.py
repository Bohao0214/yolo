"""FN breakdown and near-miss error shape analysis (offline).

Inputs (from report_dir):
- eval_boxes.csv (wp1_box_detail)
- image_summary.csv (wp1_per_image, optional)
- sweep_results.csv (wp2_sweep_summary, optional)
- run_summary.json or meta.json (run_config)

Outputs (to report_dir):
- fn_breakdown.csv
- near_miss_error_shape.csv

Example:
python /home/ubuntu/project/deduibi/yolo/analyzepy/fn_breakdown_analysis.py \
  --report_dir /home/ubuntu/project/deduibi/yolo/analysis/report_2601282104 \
  --image_dir /home/ubuntu/project/deduibi/yolo/dataset/yolo/datasetm6c/images/val
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from analyzepy.eval_core import bucket_name_from_min_side, infer_label_dir, letterbox_transform_xyxy, load_labels_for_image, compute_iou_matrix


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FN breakdown + near-miss error shape analysis.")
    p.add_argument("--report_dir", type=str, required=True)
    p.add_argument("--image_dir", type=str, required=True)
    p.add_argument("--tp_iou", type=float, default=0.5, help="Fallback TP IoU if not found in run_summary/meta.")
    return p.parse_args()


def read_csv(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def parse_bbox(s: str) -> Optional[Tuple[float, float, float, float]]:
    if not s:
        return None
    parts = s.split(",")
    if len(parts) != 4:
        return None
    try:
        vals = tuple(float(x) for x in parts)
        return vals  # type: ignore
    except ValueError:
        return None


def safe_float(v: str, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def load_run_config(report_dir: Path, fallback_tp_iou: float) -> dict:
    run_summary = report_dir / "run_summary.json"
    meta = report_dir / "meta.json"
    if run_summary.exists():
        return json.loads(run_summary.read_text(encoding="utf-8"))
    if meta.exists():
        return json.loads(meta.read_text(encoding="utf-8"))
    return {"tp_iou": fallback_tp_iou}


def build_pred_maps(rows: List[dict]) -> Dict[Tuple[str, int], dict]:
    pred_map: Dict[Tuple[str, int], dict] = {}
    for r in rows:
        pred_bbox = r.get("pred_bbox_xyxy", "")
        if not pred_bbox:
            continue
        pred_idx_str = r.get("pred_index", "")
        try:
            pred_idx = int(pred_idx_str)
        except Exception:
            continue
        pred_map[(r["image_id"], pred_idx)] = r
    return pred_map


def get_image_size_map(image_dir: Path) -> Dict[str, Tuple[int, int]]:
    size_map: Dict[str, Tuple[int, int]] = {}
    for img_path in image_dir.rglob("*"):
        if not img_path.is_file():
            continue
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        size_map[img_path.stem] = (w, h)
    return size_map


def box_center(box: Tuple[float, float, float, float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def main() -> None:
    args = parse_args()
    report_dir = Path(args.report_dir)
    image_dir = Path(args.image_dir)

    box_detail = report_dir / "eval_boxes.csv"
    if not box_detail.exists():
        raise FileNotFoundError(f"Missing eval_boxes.csv: {box_detail}")

    rows = read_csv(box_detail)

    cfg = load_run_config(report_dir, args.tp_iou)
    tp_iou = float(cfg.get("tp_iou", cfg.get("tp_iou_list", [args.tp_iou])[0] if isinstance(cfg.get("tp_iou_list"), list) else args.tp_iou))
    imgsz = int(cfg.get("imgsz", 640))

    label_dir = infer_label_dir(image_dir)
    if not label_dir.exists():
        raise FileNotFoundError(f"Label dir not found: {label_dir}")

    # Build per-image pred list from eval_boxes.csv (pred rows only)
    preds_by_image: Dict[str, List[Tuple[Tuple[float, float, float, float], float]]] = {}
    for r in rows:
        pred_box = parse_bbox(r.get("pred_bbox_xyxy", ""))
        if pred_box is None:
            continue
        score = safe_float(r.get("pred_score", "0"), 0.0)
        preds_by_image.setdefault(r["image_id"], []).append((pred_box, score))

    # Counters
    total_fn = 0
    near_miss = 0
    no_resp = 0
    postproc = 0

    bucket_stats: Dict[Tuple[str], dict] = {}

    # Near-miss error shape samples
    nm_samples: List[dict] = []

    # Build FN candidates directly from labels + preds (no new inference).
    for img_path in image_dir.rglob("*"):
        if not img_path.is_file() or img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            continue
        image_id = img_path.stem
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        label_path = label_dir / f"{image_id}.txt"
        gts = load_labels_for_image(label_path, w, h, image_id, img_path)
        if not gts:
            continue
        pred_list = preds_by_image.get(image_id, [])
        pred_boxes = np.array([b for b, _ in pred_list], dtype=np.float32) if pred_list else np.zeros((0, 4), dtype=np.float32)
        pred_scores = [s for _, s in pred_list]

        gt_boxes = np.array([g.bbox_xyxy for g in gts], dtype=np.float32)
        iou_mat = compute_iou_matrix(gt_boxes, pred_boxes)

        for gi, gt in enumerate(gts):
            best_iou = 0.0
            best_pred_score = 0.0
            pred_box = None
            if pred_boxes.size > 0:
                ious = iou_mat[gi]
                best_pi = int(np.argmax(ious))
                best_iou = float(ious[best_pi])
                best_pred_score = float(pred_scores[best_pi]) if pred_scores else 0.0
                pred_box = tuple(float(x) for x in pred_boxes[best_pi])

            # FN if not reaching TP IoU
            if best_iou >= tp_iou:
                continue

            total_fn += 1
            is_near = 0.3 <= best_iou < 0.5
            is_no_response = (best_iou < 0.1) and (best_pred_score < 0.05)
            is_postproc = (not is_no_response) and (not is_near)

            if is_near:
                near_miss += 1
            elif is_no_response:
                no_resp += 1
            else:
                postproc += 1

            lb = letterbox_transform_xyxy(gt.bbox_xyxy, w, h, imgsz)
            gw = max(0.0, lb[2] - lb[0])
            gh = max(0.0, lb[3] - lb[1])
            size_bin = bucket_name_from_min_side(min(gw, gh))

            if size_bin not in bucket_stats:
                bucket_stats[size_bin] = {"total_fn": 0, "near": 0, "no_resp": 0, "postproc": 0}
            bucket_stats[size_bin]["total_fn"] += 1
            if is_near:
                bucket_stats[size_bin]["near"] += 1
            elif is_no_response:
                bucket_stats[size_bin]["no_resp"] += 1
            else:
                bucket_stats[size_bin]["postproc"] += 1

            if is_near and pred_box is not None:
                gx, gy = box_center(gt.bbox_xyxy)
                px, py = box_center(pred_box)
                off_x = px - gx
                off_y = py - gy
                gw = max(1e-6, gt.bbox_xyxy[2] - gt.bbox_xyxy[0])
                gh = max(1e-6, gt.bbox_xyxy[3] - gt.bbox_xyxy[1])
                gd = max(1e-6, (gw ** 2 + gh ** 2) ** 0.5)
                pw = max(1e-6, pred_box[2] - pred_box[0])
                ph = max(1e-6, pred_box[3] - pred_box[1])
                nm_samples.append(
                    {
                        "center_offset_x": off_x,
                        "center_offset_y": off_y,
                        "center_offset_norm": (off_x ** 2 + off_y ** 2) ** 0.5 / gd,
                        "size_ratio_w": pw / gw,
                        "size_ratio_h": ph / gh,
                        "aspect_ratio_error": (pw / ph) / (gw / gh),
                    }
                )

    # fn_breakdown.csv
    fn_breakdown_path = report_dir / "fn_breakdown.csv"
    with fn_breakdown_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "scope",
            "size_bin",
            "total_fn",
            "near_miss_count",
            "near_miss_ratio",
            "no_response_count",
            "no_response_ratio",
            "postproc_killed_count",
            "postproc_killed_ratio",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        total_fn_safe = max(1, total_fn)
        w.writerow(
            {
                "scope": "ALL",
                "size_bin": "ALL",
                "total_fn": total_fn,
                "near_miss_count": near_miss,
                "near_miss_ratio": near_miss / total_fn_safe,
                "no_response_count": no_resp,
                "no_response_ratio": no_resp / total_fn_safe,
                "postproc_killed_count": postproc,
                "postproc_killed_ratio": postproc / total_fn_safe,
            }
        )
        for size_bin, stat in bucket_stats.items():
            total_b = max(1, stat["total_fn"])
            w.writerow(
                {
                    "scope": "SIZE_BIN",
                    "size_bin": size_bin,
                    "total_fn": stat["total_fn"],
                    "near_miss_count": stat["near"],
                    "near_miss_ratio": stat["near"] / total_b,
                    "no_response_count": stat["no_resp"],
                    "no_response_ratio": stat["no_resp"] / total_b,
                    "postproc_killed_count": stat["postproc"],
                    "postproc_killed_ratio": stat["postproc"] / total_b,
                }
            )

    # near_miss_error_shape.csv (summary stats)
    near_miss_path = report_dir / "near_miss_error_shape.csv"
    def pct(vals: List[float], q: float) -> float:
        if not vals:
            return 0.0
        return float(np.percentile(np.array(vals, dtype=np.float32), q))

    def mean(vals: List[float]) -> float:
        return float(np.mean(vals)) if vals else 0.0

    if nm_samples:
        off_x = [s["center_offset_x"] for s in nm_samples]
        off_y = [s["center_offset_y"] for s in nm_samples]
        off_n = [s["center_offset_norm"] for s in nm_samples]
        rw = [s["size_ratio_w"] for s in nm_samples]
        rh = [s["size_ratio_h"] for s in nm_samples]
        ar = [s["aspect_ratio_error"] for s in nm_samples]
        pos_x = sum(1 for v in off_x if v > 0) / max(1, len(off_x))
        pos_y = sum(1 for v in off_y if v > 0) / max(1, len(off_y))
    else:
        off_x = off_y = off_n = rw = rh = ar = []
        pos_x = pos_y = 0.0

    with near_miss_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["metric", "mean", "median", "p90", "pos_ratio"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerow({"metric": "center_offset_x", "mean": mean(off_x), "median": pct(off_x, 50), "p90": pct(off_x, 90), "pos_ratio": pos_x})
        w.writerow({"metric": "center_offset_y", "mean": mean(off_y), "median": pct(off_y, 50), "p90": pct(off_y, 90), "pos_ratio": pos_y})
        w.writerow({"metric": "center_offset_norm", "mean": mean(off_n), "median": pct(off_n, 50), "p90": pct(off_n, 90), "pos_ratio": ""})
        w.writerow({"metric": "size_ratio_w", "mean": mean(rw), "median": pct(rw, 50), "p90": pct(rw, 90), "pos_ratio": ""})
        w.writerow({"metric": "size_ratio_h", "mean": mean(rh), "median": pct(rh, 50), "p90": pct(rh, 90), "pos_ratio": ""})
        w.writerow({"metric": "aspect_ratio_error", "mean": mean(ar), "median": pct(ar, 50), "p90": pct(ar, 90), "pos_ratio": ""})

    # Print conclusion
    ratios = {
        "near-miss": near_miss / max(1, total_fn),
        "no-response": no_resp / max(1, total_fn),
        "postproc-killed": postproc / max(1, total_fn),
    }
    main_issue = max(ratios.items(), key=lambda kv: kv[1])[0] if total_fn else "none"
    if main_issue == "near-miss":
        suggestion = "主矛盾=near-miss：优先优化定位（分辨率/检测头/回归损失），再考虑轻微后处理放宽。"
    elif main_issue == "no-response":
        suggestion = "主矛盾=no-response：优先数据/召回（增广、难例、特征表征），再调后处理。"
    elif main_issue == "postproc-killed":
        suggestion = "主矛盾=postproc-killed：优先放宽后处理（conf/NMS/max_det）并结合扫参上限。"
    else:
        suggestion = "主矛盾=none：未检测到 FN 或统计不足。"

    print(suggestion)


if __name__ == "__main__":
    main()
