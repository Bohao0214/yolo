"""Shared evaluation utilities for WP1/WP2 error analysis.

Example:
  from tools.eval_core import evaluate_dataset, run_inference
"""

from __future__ import annotations

import math
from dataclasses import dataclass
import gc
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except Exception as exc:  # pragma: no cover - surfaced in caller
    YOLO = None  # type: ignore
    ULTRALYTICS_IMPORT_ERROR = exc
else:
    ULTRALYTICS_IMPORT_ERROR = None

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
BUCKETS: Sequence[Tuple[str, float, float]] = (
    ("<8", 0.0, 8.0),
    ("8-16", 8.0, 16.0),
    ("16-32", 16.0, 32.0),
    ("32-64", 32.0, 64.0),
    (">64", 64.0, math.inf),
)


@dataclass(frozen=True)
class BoxRecord:
    image_id: str
    img_path: str
    index: int
    cls: int
    bbox_xyxy: Tuple[float, float, float, float]
    score: Optional[float] = None


@dataclass(frozen=True)
class MatchResult:
    pred_index: int
    gt_index: Optional[int]
    best_gt_index: Optional[int]
    best_iou: float
    match_iou: float
    match_type: str


def ensure_ultralytics() -> None:
    if YOLO is None:
        msg = (
            "Failed to import ultralytics. Install with one of:\n"
            "  pip install ultralytics\n"
            "  conda install -c conda-forge ultralytics\n"
            f"Original error: {ULTRALYTICS_IMPORT_ERROR}"
        )
        raise ImportError(msg)


def list_images(image_dir: Path) -> List[Path]:
    images = [p for p in image_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]
    return sorted(images)


def infer_label_dir(image_dir: Path) -> Path:
    parts = list(image_dir.parts)
    if "images" in parts:
        idx = parts.index("images")
        return Path(*parts[:idx], "labels", *parts[idx + 1 :])
    if image_dir.name in {"train", "val", "test"}:
        return image_dir.parent.parent / "labels" / image_dir.name
    return image_dir.parent / "labels"


def xywhn_to_xyxy(xc: float, yc: float, w: float, h: float, img_w: int, img_h: int) -> Tuple[float, float, float, float]:
    x1 = (xc - w / 2.0) * img_w
    y1 = (yc - h / 2.0) * img_h
    x2 = (xc + w / 2.0) * img_w
    y2 = (yc + h / 2.0) * img_h
    return (float(x1), float(y1), float(x2), float(y2))


def load_labels_for_image(label_path: Path, img_w: int, img_h: int, image_id: str, img_path: Path) -> List[BoxRecord]:
    if not label_path.exists():
        return []
    records: List[BoxRecord] = []
    with label_path.open("r", encoding="utf-8", errors="ignore") as f:
        for idx, line in enumerate(f):
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            try:
                cls = int(float(parts[0]))
                xc, yc, w, h = map(float, parts[1:5])
            except ValueError:
                continue
            bbox = xywhn_to_xyxy(xc, yc, w, h, img_w, img_h)
            records.append(BoxRecord(image_id=image_id, img_path=str(img_path), index=idx, cls=cls, bbox_xyxy=bbox))
    return records


def read_image_size(img_path: Path) -> Tuple[int, int]:
    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {img_path}")
    h, w = img.shape[:2]
    return w, h


def letterbox_transform_xyxy(box: Tuple[float, float, float, float], img_w: int, img_h: int, imgsz: int) -> Tuple[float, float, float, float]:
    scale = min(imgsz / img_w, imgsz / img_h)
    new_w = img_w * scale
    new_h = img_h * scale
    pad_w = (imgsz - new_w) / 2.0
    pad_h = (imgsz - new_h) / 2.0
    x1, y1, x2, y2 = box
    return (
        float(x1 * scale + pad_w),
        float(y1 * scale + pad_h),
        float(x2 * scale + pad_w),
        float(y2 * scale + pad_h),
    )


def box_wh(box: Tuple[float, float, float, float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1), max(0.0, y2 - y1)


def bucket_name_from_min_side(min_side: float) -> str:
    for name, lo, hi in BUCKETS:
        if lo <= min_side < hi:
            return name
    return BUCKETS[-1][0]


def compute_iou_matrix(gt_boxes: np.ndarray, pred_boxes: np.ndarray) -> np.ndarray:
    if gt_boxes.size == 0 or pred_boxes.size == 0:
        return np.zeros((gt_boxes.shape[0], pred_boxes.shape[0]), dtype=np.float32)
    ix1 = np.maximum(gt_boxes[:, None, 0], pred_boxes[None, :, 0])
    iy1 = np.maximum(gt_boxes[:, None, 1], pred_boxes[None, :, 1])
    ix2 = np.minimum(gt_boxes[:, None, 2], pred_boxes[None, :, 2])
    iy2 = np.minimum(gt_boxes[:, None, 3], pred_boxes[None, :, 3])
    iw = np.maximum(0.0, ix2 - ix1)
    ih = np.maximum(0.0, iy2 - iy1)
    inter = iw * ih
    gt_area = np.maximum(0.0, (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1]))
    pred_area = np.maximum(0.0, (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1]))
    union = gt_area[:, None] + pred_area[None, :] - inter
    return np.where(union > 0, inter / union, 0.0).astype(np.float32)


def greedy_match(
    gt_records: Sequence[BoxRecord],
    pred_records: Sequence[BoxRecord],
    tp_iou: float,
    near_iou_low: float,
    near_iou_high: float,
) -> Tuple[List[MatchResult], List[int]]:
    gt_boxes = np.array([r.bbox_xyxy for r in gt_records], dtype=np.float32)
    pred_boxes = np.array([r.bbox_xyxy for r in pred_records], dtype=np.float32)
    iou_mat = compute_iou_matrix(gt_boxes, pred_boxes)

    pred_order = sorted(range(len(pred_records)), key=lambda i: (pred_records[i].score or 0.0), reverse=True)
    gt_used = np.zeros(len(gt_records), dtype=bool)
    matches: List[MatchResult] = []

    for pi in pred_order:
        if len(gt_records) == 0:
            matches.append(
                MatchResult(
                    pred_index=pi,
                    gt_index=None,
                    best_gt_index=None,
                    best_iou=0.0,
                    match_iou=0.0,
                    match_type="FP_strict",
                )
            )
            continue

        ious = iou_mat[:, pi]
        best_gt_idx = int(np.argmax(ious))
        best_iou = float(ious[best_gt_idx])
        assign_gt_idx: Optional[int] = None
        match_type = "FP_strict"
        match_iou = best_iou

        # Assign to best unmatched GT if possible.
        candidate_indices = np.argsort(-ious)
        for gi in candidate_indices:
            gi_int = int(gi)
            if gt_used[gi_int]:
                continue
            assign_gt_idx = gi_int
            match_iou = float(ious[gi_int])
            break

        if assign_gt_idx is not None and match_iou >= tp_iou:
            gt_used[assign_gt_idx] = True
            match_type = "TP"
        elif near_iou_low <= best_iou < near_iou_high:
            match_type = "FP_near"
        else:
            match_type = "FP_strict"

        matches.append(
            MatchResult(
                pred_index=pi,
                gt_index=assign_gt_idx if match_type == "TP" else None,
                best_gt_index=best_gt_idx,
                best_iou=best_iou,
                match_iou=match_iou,
                match_type=match_type,
            )
        )

    fn_indices = [i for i, used in enumerate(gt_used) if not used]
    return matches, fn_indices


def run_inference(
    weights: Path,
    images: Sequence[Path],
    imgsz: int,
    conf: float,
    nms_iou: float,
    max_det: int,
    batch: int,
    device: str,
    model: Optional["YOLO"] = None,
    chunk_size: int = 0,
) -> Dict[str, List[BoxRecord]]:
    ensure_ultralytics()
    model_obj = model if model is not None else YOLO(str(weights))
    preds: Dict[str, List[BoxRecord]] = {}
    if chunk_size <= 0:
        chunk_size = len(images)

    try:
        import torch
    except Exception:
        torch = None  # type: ignore

    for start in range(0, len(images), chunk_size):
        chunk = images[start : start + chunk_size]
        results = model_obj.predict(
            source=[str(p) for p in chunk],
            imgsz=int(imgsz),
            conf=float(conf),
            iou=float(nms_iou),
            max_det=int(max_det),
            save=False,
            verbose=False,
            batch=int(batch),
            device=device if device else None,
            stream=True,
        )
        for res in results:
            img_path = Path(res.path)
            image_id = img_path.stem
            boxes = res.boxes
            if boxes is None or boxes.xyxy is None:
                preds[str(img_path)] = []
                continue
            xyxy = boxes.xyxy.detach().cpu().numpy()
            confs = boxes.conf.detach().cpu().numpy() if boxes.conf is not None else np.zeros(len(xyxy), dtype=np.float32)
            clss = boxes.cls.detach().cpu().numpy() if boxes.cls is not None else np.zeros(len(xyxy), dtype=np.float32)
            records: List[BoxRecord] = []
            for i, (box, score, cls) in enumerate(zip(xyxy, confs, clss)):
                records.append(
                    BoxRecord(
                        image_id=image_id,
                        img_path=str(img_path),
                        index=i,
                        cls=int(cls),
                        bbox_xyxy=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                        score=float(score),
                    )
                )
            preds[str(img_path)] = records

        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    return preds


def evaluate_dataset(
    images: Sequence[Path],
    label_dir: Path,
    preds: Dict[str, List[BoxRecord]],
    imgsz: int,
    tp_iou: float,
    near_iou_low: float,
    near_iou_high: float,
) -> Tuple[List[dict], List[dict], List[dict], dict]:
    image_rows: List[dict] = []
    box_rows: List[dict] = []

    bucket_stats = {
        name: {
            "n_gt": 0,
            "tp_gt": 0,
            "fp": 0,
            "scores": [],
            "near_ious": [],
            "tp_ious": [],
        }
        for name, _, _ in BUCKETS
    }

    totals = {
        "images": 0,
        "gt": 0,
        "pred": 0,
        "tp": 0,
        "fp_strict": 0,
        "fp_near": 0,
        "fn": 0,
    }

    for img_path in images:
        img_w, img_h = read_image_size(img_path)
        image_id = img_path.stem
        label_path = label_dir / f"{image_id}.txt"
        gt_records = load_labels_for_image(label_path, img_w, img_h, image_id, img_path)
        pred_records = preds.get(str(img_path), [])

        matches, fn_indices = greedy_match(gt_records, pred_records, tp_iou, near_iou_low, near_iou_high)

        gt_boxes = np.array([r.bbox_xyxy for r in gt_records], dtype=np.float32)
        pred_boxes = np.array([r.bbox_xyxy for r in pred_records], dtype=np.float32)
        iou_mat = compute_iou_matrix(gt_boxes, pred_boxes)

        tp_gt_flags = np.zeros(len(gt_records), dtype=bool)
        for m in matches:
            if m.match_type == "TP" and m.gt_index is not None:
                tp_gt_flags[m.gt_index] = True

        # Bucket stats for GT (recall by size bucket).
        for gi, gt in enumerate(gt_records):
            lb_box = letterbox_transform_xyxy(gt.bbox_xyxy, img_w, img_h, imgsz)
            gw, gh = box_wh(lb_box)
            bucket = bucket_name_from_min_side(min(gw, gh))
            bucket_stats[bucket]["n_gt"] += 1
            if tp_gt_flags[gi]:
                bucket_stats[bucket]["tp_gt"] += 1

        # Per-pred rows and bucket stats.
        for m in matches:
            pred = pred_records[m.pred_index]
            best_gt_idx = m.best_gt_index
            best_iou = m.best_iou
            assigned_gt_idx = m.gt_index

            gt_bbox = ""
            if assigned_gt_idx is not None:
                gt_bbox = ",".join(f"{v:.2f}" for v in gt_records[assigned_gt_idx].bbox_xyxy)
            elif best_gt_idx is not None and len(gt_records) > 0:
                gt_bbox = ",".join(f"{v:.2f}" for v in gt_records[best_gt_idx].bbox_xyxy)

            box_rows.append(
                {
                    "image_id": image_id,
                    "img_path": str(img_path),
                    "gt_bbox_xyxy": gt_bbox,
                    "pred_bbox_xyxy": ",".join(f"{v:.2f}" for v in pred.bbox_xyxy),
                    "pred_score": pred.score,
                    "match_iou": best_iou,
                    "match_type": m.match_type,
                    "best_match_id": (
                        f"gt:{assigned_gt_idx}" if assigned_gt_idx is not None else (f"gt:{best_gt_idx}" if best_gt_idx is not None else "")
                    ),
                    "pred_index": pred.index,
                }
            )

            lb_box = letterbox_transform_xyxy(pred.bbox_xyxy, img_w, img_h, imgsz)
            pw, ph = box_wh(lb_box)
            bucket = bucket_name_from_min_side(min(pw, ph))
            if m.match_type != "TP":
                bucket_stats[bucket]["fp"] += 1
            bucket_stats[bucket]["scores"].append(float(pred.score or 0.0))
            if m.match_type == "FP_near":
                bucket_stats[bucket]["near_ious"].append(best_iou)
            if m.match_type == "TP":
                bucket_stats[bucket]["tp_ious"].append(best_iou)

        # FN rows (record best IoU to any pred if available).
        for gi in fn_indices:
            gt = gt_records[gi]
            best_pred_idx: Optional[int] = None
            best_pred_iou = float("nan")
            if pred_boxes.size > 0:
                ious = iou_mat[gi]
                best_pred_idx = int(np.argmax(ious))
                best_pred_iou = float(ious[best_pred_idx])
            box_rows.append(
                {
                    "image_id": image_id,
                    "img_path": str(img_path),
                    "gt_bbox_xyxy": ",".join(f"{v:.2f}" for v in gt.bbox_xyxy),
                    "pred_bbox_xyxy": "",
                    "pred_score": np.nan,
                    "match_iou": best_pred_iou,
                    "match_type": "FN",
                    "best_match_id": f"pred:{best_pred_idx}" if best_pred_idx is not None else "",
                    "pred_index": "",
                }
            )

        n_tp = sum(1 for m in matches if m.match_type == "TP")
        n_fp_strict = sum(1 for m in matches if m.match_type == "FP_strict")
        n_fp_near = sum(1 for m in matches if m.match_type == "FP_near")
        n_fn = len(fn_indices)

        image_rows.append(
            {
                "image_id": image_id,
                "img_path": str(img_path),
                "n_gt": len(gt_records),
                "n_pred": len(pred_records),
                "n_tp": n_tp,
                "n_fp_strict": n_fp_strict,
                "n_fp_near": n_fp_near,
                "n_fn": n_fn,
            }
        )

        totals["images"] += 1
        totals["gt"] += len(gt_records)
        totals["pred"] += len(pred_records)
        totals["tp"] += n_tp
        totals["fp_strict"] += n_fp_strict
        totals["fp_near"] += n_fp_near
        totals["fn"] += n_fn

    bucket_rows: List[dict] = []
    n_images = max(1, totals["images"])
    for name, _, _ in BUCKETS:
        stats = bucket_stats[name]
        n_gt = stats["n_gt"]
        recall = (stats["tp_gt"] / n_gt) if n_gt > 0 else 0.0
        fp_per_image = stats["fp"] / n_images
        scores = stats["scores"]
        near_ious = stats["near_ious"]
        tp_ious = stats["tp_ious"]
        bucket_rows.append(
            {
                "bucket_name": name,
                "n_gt": n_gt,
                "recall": recall,
                "fp_per_image": fp_per_image,
                "mean_pred_score": float(np.mean(scores)) if scores else 0.0,
                "mean_iou_near": float(np.mean(near_ious)) if near_ious else 0.0,
                "mean_iou_tp": float(np.mean(tp_ious)) if tp_ious else 0.0,
            }
        )

    return image_rows, box_rows, bucket_rows, totals


def parse_float_list(csv_text: str) -> List[float]:
    return [float(x.strip()) for x in csv_text.split(",") if x.strip()]


def parse_int_list(csv_text: str) -> List[int]:
    return [int(x.strip()) for x in csv_text.split(",") if x.strip()]


def crop_with_padding(image: np.ndarray, center_xy: Tuple[float, float], crop_size: int) -> np.ndarray:
    cx, cy = center_xy
    half = crop_size // 2
    x1 = int(round(cx - half))
    y1 = int(round(cy - half))
    x2 = x1 + crop_size
    y2 = y1 + crop_size

    h, w = image.shape[:2]
    pad_left = max(0, -x1)
    pad_top = max(0, -y1)
    pad_right = max(0, x2 - w)
    pad_bottom = max(0, y2 - h)

    if pad_left or pad_top or pad_right or pad_bottom:
        image = cv2.copyMakeBorder(image, pad_top, pad_bottom, pad_left, pad_right, borderType=cv2.BORDER_CONSTANT, value=0)
        x1 += pad_left
        y1 += pad_top
        x2 += pad_left
        y2 += pad_top

    return image[y1:y2, x1:x2].copy()


def box_center_xy(box: Tuple[float, float, float, float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
