"""P2.3.2 FP + pred_dup typing analysis (YOLO).

Example:
python /home/ubuntu/project/deduibi/yolo/analyze/code/wp1_fp_dup_typing.py \
  --weights /home/ubuntu/project/deduibi/yolo/models/defect/exp_260202_base/best/best.pt \
  --image_dir /home/ubuntu/project/deduibi/yolo/dataset/yolo/datasetm6c/images/val \
  --image_dir /home/ubuntu/project/deduibi/yolo/dataset/yolo/datasetm6c/images/test \
  --out_root /home/ubuntu/project/deduibi/yolo/analyze/result \
  --batch 4 --infer_chunk 16
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import gc
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except Exception as exc:  # pragma: no cover
    YOLO = None  # type: ignore
    ULTRALYTICS_IMPORT_ERROR = exc
else:
    ULTRALYTICS_IMPORT_ERROR = None


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
TYPE_NAMES = ["highlight", "edge_bg", "texture_boundary", "speckle", "other"]
DUP_STATES = ["unmatched", "pred_dup", "both", "other"]
WHITE_THRESH = 250
SPATIAL_BINS: Sequence[Tuple[str, float, float]] = (
    ("<32", 0.0, 32.0),
    ("32-64", 32.0, 64.0),
    (">64", 64.0, math.inf),
)


@dataclass(frozen=True)
class PredRecord:
    bbox_xyxy: Tuple[float, float, float, float]
    score: float
    cls: int


def ensure_ultralytics() -> None:
    if YOLO is None:
        raise ImportError(
            "Failed to import ultralytics. Please install ultralytics in the environment. "
            f"Original error: {ULTRALYTICS_IMPORT_ERROR}"
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="P2.3.2 FP + pred_dup typing analysis.")
    p.add_argument("--weights", type=str, required=True)
    p.add_argument(
        "--image_dir",
        type=str,
        required=True,
        action="append",
        help="Dataset image directory (val/test). Can be provided multiple times or comma-separated.",
    )
    p.add_argument(
        "--label_dir",
        type=str,
        default=None,
        action="append",
        help="Optional label directory override (can be repeated or comma-separated).",
    )
    p.add_argument("--out_root", type=str, required=True)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--infer_chunk", type=int, default=16)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", type=str, default="")

    # postprocess params
    p.add_argument("--conf", type=float, default=0.3)
    p.add_argument("--tp_iou", type=float, default=0.2)
    p.add_argument("--nms_iou", type=float, default=0.6)
    p.add_argument("--max_det", type=int, default=20)

    # typing thresholds
    p.add_argument("--edge_white_frac", type=float, default=0.33)
    p.add_argument("--hl_bright_percentile", type=float, default=95.0)
    p.add_argument("--hl_grad_percentile", type=float, default=90.0)
    p.add_argument("--hl_frac", type=float, default=0.08)
    p.add_argument("--texture_frac", type=float, default=0.10)
    p.add_argument("--speckle_frac", type=float, default=0.10)
    p.add_argument("--speckle_area_ratio_max", type=float, default=0.01)

    p.add_argument(
        "--bucket_edges",
        type=str,
        default="16,32,64",
        help="Comma-separated bucket edges for short side in pixels. Default: 16,32,64 -> <16,16-32,32-64,>64",
    )
    p.add_argument("--sample_per_type", type=int, default=5)
    return p.parse_args()


def normalize_path_list(raw: Optional[Sequence[str]]) -> List[Path]:
    if not raw:
        return []
    out: List[Path] = []
    for item in raw:
        if not item:
            continue
        for part in str(item).split(","):
            p = part.strip()
            if not p:
                continue
            out.append(Path(p))
    seen = set()
    uniq: List[Path] = []
    for p in out:
        key = str(p)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(p)
    return uniq


def list_images(image_dir: Path) -> List[Path]:
    return sorted([p for p in image_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS])


def _label_dir_candidates(image_dir: Path) -> List[Path]:
    parts = list(image_dir.parts)
    candidates: List[Path] = []
    if "images" in parts:
        idx = parts.index("images")
        for name in ("labels", "label", "lable"):
            candidates.append(Path(*parts[:idx], name, *parts[idx + 1 :]))
    if "image" in parts:
        idx = parts.index("image")
        for name in ("labels", "label", "lable"):
            candidates.append(Path(*parts[:idx], name, *parts[idx + 1 :]))
    if image_dir.name in {"train", "val", "test"}:
        parent = image_dir.parent
        if parent.name in {"images", "image"}:
            for name in ("labels", "label", "lable"):
                candidates.append(parent.parent / name / image_dir.name)
        for name in ("labels", "label", "lable"):
            candidates.append(image_dir.parent / name)
    for name in ("labels", "label", "lable"):
        candidates.append(image_dir.parent / name)
    seen = set()
    uniq: List[Path] = []
    for c in candidates:
        if str(c) not in seen:
            uniq.append(c)
            seen.add(str(c))
    return uniq


def infer_label_dir(image_dir: Path) -> Path:
    candidates = _label_dir_candidates(image_dir)
    for cand in candidates:
        if cand.exists():
            return cand
    return candidates[0] if candidates else image_dir.parent / "labels"


def make_report_dir(out_root: Path) -> Path:
    ts = dt.datetime.now().strftime("report_%y%m%d%H%M")
    base = out_root / ts
    if not base.exists():
        base.mkdir(parents=True, exist_ok=False)
        return base
    suffix = 1
    while True:
        cand = out_root / f"{ts}_{suffix:02d}"
        if not cand.exists():
            cand.mkdir(parents=True, exist_ok=False)
            return cand
        suffix += 1


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


def xywhn_to_xyxy(xc: float, yc: float, w: float, h: float, img_w: int, img_h: int) -> Tuple[float, float, float, float]:
    x1 = (xc - w / 2.0) * img_w
    y1 = (yc - h / 2.0) * img_h
    x2 = (xc + w / 2.0) * img_w
    y2 = (yc + h / 2.0) * img_h
    return (float(x1), float(y1), float(x2), float(y2))


def load_labels(label_path: Path, img_w: int, img_h: int) -> List[Tuple[float, float, float, float]]:
    if not label_path.exists():
        return []
    out: List[Tuple[float, float, float, float]] = []
    with label_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            try:
                xc, yc, w, h = map(float, parts[1:5])
            except ValueError:
                continue
            out.append(xywhn_to_xyxy(xc, yc, w, h, img_w, img_h))
    return out


def mask_ratio_for_box(integral: np.ndarray, box: Tuple[float, float, float, float], w: int, h: int) -> float:
    x1, y1, x2, y2 = box
    x1i = max(0, min(w, int(math.floor(x1))))
    x2i = max(0, min(w, int(math.ceil(x2))))
    y1i = max(0, min(h, int(math.floor(y1))))
    y2i = max(0, min(h, int(math.ceil(y2))))
    if x2i <= x1i or y2i <= y1i:
        return 0.0
    area = float((x2i - x1i) * (y2i - y1i))
    if area <= 0:
        return 0.0
    s = integral[y2i, x2i] - integral[y1i, x2i] - integral[y2i, x1i] + integral[y1i, x1i]
    return float(s) / area


def compute_integral(mask: np.ndarray) -> np.ndarray:
    return cv2.integral(mask.astype(np.uint8))


def build_grad_and_masks(
    img_bgr: np.ndarray, bright_p: float, grad_p: float, valid_mask: Optional[np.ndarray]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(gx, gy)
    if valid_mask is not None and np.any(valid_mask):
        gray_vals = gray[valid_mask]
        grad_vals = grad[valid_mask]
    else:
        gray_vals = gray.reshape(-1)
        grad_vals = grad.reshape(-1)
    bright_thresh = float(np.percentile(gray_vals, bright_p))
    grad_thresh = float(np.percentile(grad_vals, grad_p))
    bright_mask = gray >= bright_thresh
    grad_mask = grad >= grad_thresh
    if valid_mask is not None and np.any(valid_mask):
        bright_mask = bright_mask & valid_mask
        grad_mask = grad_mask & valid_mask
    highlight_mask = bright_mask & grad_mask
    texture_mask = (~bright_mask) & grad_mask
    return highlight_mask.astype(np.uint8), texture_mask.astype(np.uint8), grad_mask.astype(np.uint8), bright_thresh, grad_thresh


def speckle_ratio(grad_mask: np.ndarray, box: Tuple[float, float, float, float], w: int, h: int, area_ratio_max: float) -> float:
    x1, y1, x2, y2 = box
    x1i = max(0, min(w, int(math.floor(x1))))
    x2i = max(0, min(w, int(math.ceil(x2))))
    y1i = max(0, min(h, int(math.floor(y1))))
    y2i = max(0, min(h, int(math.ceil(y2))))
    if x2i <= x1i or y2i <= y1i:
        return 0.0
    roi = grad_mask[y1i:y2i, x1i:x2i].astype(np.uint8)
    area = (x2i - x1i) * (y2i - y1i)
    if area <= 0 or roi.sum() == 0:
        return 0.0
    num, labels, stats, _ = cv2.connectedComponentsWithStats(roi, connectivity=8)
    if num <= 1:
        return 0.0
    max_area = max(10, int(area * area_ratio_max))
    small_pixels = 0
    for i in range(1, num):
        comp_area = int(stats[i, cv2.CC_STAT_AREA])
        if comp_area <= max_area:
            small_pixels += comp_area
    return float(small_pixels) / float(area)


def short_side(box: Tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = box
    return float(min(max(0.0, x2 - x1), max(0.0, y2 - y1)))


def build_buckets(edges: List[float]) -> List[Tuple[str, float, float]]:
    edges_sorted = sorted(edges)
    buckets: List[Tuple[str, float, float]] = []
    prev = 0.0
    for e in edges_sorted:
        buckets.append((f"<{int(e)}" if prev == 0.0 else f"{int(prev)}-{int(e)}", prev, e))
        prev = e
    buckets.append((f">{int(prev)}", prev, math.inf))
    return buckets


def bucket_name_from_value(value: float, buckets: Sequence[Tuple[str, float, float]]) -> str:
    for name, lo, hi in buckets:
        if lo <= value < hi:
            return name
    return buckets[-1][0]


def dup_state(is_unmatched: bool, is_dup: bool) -> str:
    if is_unmatched and is_dup:
        return "both"
    if is_unmatched:
        return "unmatched"
    if is_dup:
        return "pred_dup"
    return "other"


def safe_ratio(a: float, b: float) -> float:
    return float(a) / float(b) if b else 0.0


def stats_summary(values: List[float]) -> Tuple[float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0
    arr = np.asarray(values, dtype=np.float32)
    return float(arr.mean()), float(np.median(arr)), float(np.percentile(arr, 90))


def init_stats(buckets: Sequence[Tuple[str, float, float]]) -> dict:
    return {
        "fp_total": 0,
        "dup_state": {s: 0 for s in DUP_STATES},
        "gt_dup": 0,
        "type_counts": {t: 0 for t in TYPE_NAMES},
        "type_scores": {t: [] for t in TYPE_NAMES},
        "type_max_ious": {t: [] for t in TYPE_NAMES},
        "type_dup_state": {t: {s: 0 for s in DUP_STATES} for t in TYPE_NAMES},
        "bucket_fp": {name: 0 for name, _, _ in buckets},
        "bucket_dup_state": {name: {s: 0 for s in DUP_STATES} for name, _, _ in buckets},
    }


def update_stats(
    stats: dict,
    fp_type: str,
    bucket: str,
    dup_state_name: str,
    score: float,
    max_iou: float,
) -> None:
    stats["fp_total"] += 1
    stats["dup_state"][dup_state_name] += 1
    stats["type_counts"][fp_type] += 1
    stats["type_scores"][fp_type].append(score)
    stats["type_max_ious"][fp_type].append(max_iou)
    stats["type_dup_state"][fp_type][dup_state_name] += 1
    stats["bucket_fp"][bucket] += 1
    stats["bucket_dup_state"][bucket][dup_state_name] += 1


def update_sample(sample_map: Dict[str, Dict[str, Tuple[float, str]]], fp_type: str, filename: str, score: float, source: str) -> None:
    by_type = sample_map.setdefault(fp_type, {})
    current = by_type.get(filename)
    if current is None or score > current[0]:
        by_type[filename] = (score, source)


def finalize_samples(sample_map: Dict[str, Dict[str, Tuple[float, str]]], topk: int) -> Dict[str, List[Tuple[str, float, str]]]:
    out: Dict[str, List[Tuple[str, float, str]]] = {}
    for t in TYPE_NAMES:
        items = [(fname, score, source) for fname, (score, source) in sample_map.get(t, {}).items()]
        items.sort(key=lambda x: x[1], reverse=True)
        out[t] = items[:topk]
    return out


def write_csv(path: Path, rows: List[dict], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def center_edge_distance(box: Tuple[float, float, float, float], w: int, h: int) -> float:
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    return float(min(cx, cy, max(0.0, w - cx), max(0.0, h - cy)))


def spatial_bin_name(dist: float) -> str:
    for name, lo, hi in SPATIAL_BINS:
        if lo <= dist < hi:
            return name
    return SPATIAL_BINS[-1][0]


def choose_fp_type(
    white_ratio: float,
    highlight_ratio: float,
    texture_ratio: float,
    speckle_ratio_val: float,
    args: argparse.Namespace,
) -> str:
    if highlight_ratio >= float(args.hl_frac):
        return "highlight"
    if white_ratio >= float(args.edge_white_frac):
        return "edge_bg"
    if texture_ratio >= float(args.texture_frac):
        return "texture_boundary"
    if speckle_ratio_val >= float(args.speckle_frac):
        return "speckle"
    return "other"


def main_logic(args: argparse.Namespace) -> Path:
    ensure_ultralytics()

    if args.batch >= 8:
        raise ValueError("batch 必须 < 8")
    if args.infer_chunk > 64:
        raise ValueError("infer_chunk 过大，请控制在 64 以内")

    image_dirs = normalize_path_list(args.image_dir)
    if not image_dirs:
        raise ValueError("image_dir 为空")

    label_dirs_input = normalize_path_list(args.label_dir)
    if label_dirs_input and len(label_dirs_input) not in (1, len(image_dirs)):
        raise ValueError("label_dir 数量必须为 1 或与 image_dir 数量一致")

    label_dirs: List[Path] = []
    if label_dirs_input:
        if len(label_dirs_input) == 1:
            label_dirs = [label_dirs_input[0] for _ in image_dirs]
        else:
            label_dirs = label_dirs_input
        # If label_dir points to a root containing val/test, try to align automatically.
        aligned: List[Path] = []
        for img_dir, lbl_dir in zip(image_dirs, label_dirs):
            cand = lbl_dir / img_dir.name
            aligned.append(cand if cand.exists() else lbl_dir)
        label_dirs = aligned
    else:
        for d in image_dirs:
            label_dirs.append(infer_label_dir(d))

    for d in image_dirs:
        if not d.exists():
            raise FileNotFoundError(f"image_dir not found: {d}")
    for d in label_dirs:
        if not d.exists():
            raise FileNotFoundError(f"label_dir not found: {d}")

    buckets = build_buckets([float(x) for x in args.bucket_edges.split(",") if x.strip()])
    report_dir = make_report_dir(Path(args.out_root))

    stats_all = init_stats(buckets)
    stats_by_source: Dict[str, dict] = {}
    sample_by_type: Dict[str, Dict[str, Tuple[float, str]]] = {t: {} for t in TYPE_NAMES}
    dup_mult_counts: Dict[int, int] = {}
    spatial_counts: Dict[str, int] = {name: 0 for name, _, _ in SPATIAL_BINS}

    items: List[Tuple[Path, Path, str]] = []
    for img_dir, lbl_dir in zip(image_dirs, label_dirs):
        source_name = img_dir.name
        stats_by_source[source_name] = init_stats(buckets)
        for img_path in list_images(img_dir):
            items.append((img_path, lbl_dir, source_name))

    model = YOLO(str(args.weights))
    total_images = len(items)
    if total_images == 0:
        raise RuntimeError("No images found.")

    try:
        import torch
    except Exception:
        torch = None  # type: ignore

    for start in range(0, total_images, int(args.infer_chunk)):
        chunk = items[start : start + int(args.infer_chunk)]
        sources = [str(p[0]) for p in chunk]
        results = model.predict(
            source=sources,
            imgsz=int(args.imgsz),
            conf=float(args.conf),
            iou=float(args.nms_iou),
            max_det=int(args.max_det),
            save=False,
            verbose=False,
            batch=int(args.batch),
            device=str(args.device) if args.device else None,
            stream=True,
        )
        result_map = {}
        for res in results:
            result_map[str(res.path)] = res

        for img_path, label_dir, source_name in chunk:
            res = result_map.get(str(img_path))
            if res is None:
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                raise FileNotFoundError(f"Failed to read image: {img_path}")
            h, w = img.shape[:2]
            label_path = label_dir / f"{img_path.stem}.txt"
            gt_boxes = load_labels(label_path, w, h)

            boxes = res.boxes
            preds: List[PredRecord] = []
            if boxes is not None and boxes.xyxy is not None:
                xyxy = boxes.xyxy.detach().cpu().numpy()
                confs = boxes.conf.detach().cpu().numpy() if boxes.conf is not None else np.zeros(len(xyxy), dtype=np.float32)
                clss = boxes.cls.detach().cpu().numpy() if boxes.cls is not None else np.zeros(len(xyxy), dtype=np.float32)
                for box, score, cls in zip(xyxy, confs, clss):
                    preds.append(
                        PredRecord(
                            bbox_xyxy=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                            score=float(score),
                            cls=int(cls),
                        )
                    )

            gt_arr = np.array(gt_boxes, dtype=np.float32) if gt_boxes else np.zeros((0, 4), dtype=np.float32)
            pred_arr = np.array([p.bbox_xyxy for p in preds], dtype=np.float32) if preds else np.zeros((0, 4), dtype=np.float32)
            iou_mat = compute_iou_matrix(gt_arr, pred_arr)

            assigned_pred_for_gt: Dict[int, int] = {}
            assigned_gt_for_pred: Dict[int, int] = {}
            if iou_mat.size > 0:
                pairs: List[Tuple[float, int, int]] = []
                for gi in range(iou_mat.shape[0]):
                    for pi in range(iou_mat.shape[1]):
                        v = float(iou_mat[gi, pi])
                        if v >= float(args.tp_iou):
                            pairs.append((v, gi, pi))
                pairs.sort(reverse=True, key=lambda x: x[0])
                for v, gi, pi in pairs:
                    if gi in assigned_pred_for_gt or pi in assigned_gt_for_pred:
                        continue
                    assigned_pred_for_gt[gi] = pi
                    assigned_gt_for_pred[pi] = gi

            pred_dup: Dict[int, int] = {}
            for gi in range(len(gt_boxes)):
                if iou_mat.size == 0:
                    continue
                cand = np.where(iou_mat[gi] >= float(args.tp_iou))[0].tolist()
                if len(cand) <= 1:
                    continue
                def key(pi: int) -> Tuple[float, float]:
                    return (float(iou_mat[gi, pi]), float(preds[pi].score if pi < len(preds) else 0.0))
                best_pi = max(cand, key=key)
                for pi in cand:
                    if pi != best_pi:
                        pred_dup[pi] = gi

            # dup multiplicity distribution: count preds per GT (>= tp_iou)
            if iou_mat.size > 0:
                for gi in range(iou_mat.shape[0]):
                    dup_count = int(np.sum(iou_mat[gi] >= float(args.tp_iou)))
                    if dup_count >= 2:
                        dup_mult_counts[dup_count] = dup_mult_counts.get(dup_count, 0) + 1

            gt_dup = set()
            for gi in range(len(gt_boxes)):
                if gi in assigned_pred_for_gt:
                    continue
                if iou_mat.size == 0:
                    continue
                cand = np.where(iou_mat[gi] >= float(args.tp_iou))[0].tolist()
                for pi in cand:
                    if pi in assigned_gt_for_pred:
                        gt_dup.add(gi)
                        break

            white_mask = (
                (img[:, :, 0] >= WHITE_THRESH)
                & (img[:, :, 1] >= WHITE_THRESH)
                & (img[:, :, 2] >= WHITE_THRESH)
            )
            white_integral = compute_integral(white_mask)
            valid_mask = ~white_mask
            highlight_mask, texture_mask, grad_mask, bright_th, grad_th = build_grad_and_masks(
                img, float(args.hl_bright_percentile), float(args.hl_grad_percentile), valid_mask=valid_mask
            )
            hl_integral = compute_integral(highlight_mask)
            texture_integral = compute_integral(texture_mask)

            for pi, pred in enumerate(preds):
                if pi in assigned_gt_for_pred:
                    continue
                max_iou = float(iou_mat[:, pi].max()) if iou_mat.size > 0 else 0.0
                is_unmatched = max_iou < float(args.tp_iou)
                is_dup = pi in pred_dup
                box = pred.bbox_xyxy
                bucket = bucket_name_from_value(short_side(box), buckets)

                white_ratio = mask_ratio_for_box(white_integral, box, w, h)
                highlight_ratio = mask_ratio_for_box(hl_integral, box, w, h)
                texture_ratio = mask_ratio_for_box(texture_integral, box, w, h)
                speckle_ratio_val = speckle_ratio(grad_mask, box, w, h, float(args.speckle_area_ratio_max))

                fp_type = choose_fp_type(white_ratio, highlight_ratio, texture_ratio, speckle_ratio_val, args)
                state = dup_state(is_unmatched, is_dup)
                score = float(pred.score)
                update_stats(stats_all, fp_type, bucket, state, score, max_iou)
                update_stats(stats_by_source[source_name], fp_type, bucket, state, score, max_iou)

                update_sample(sample_by_type, fp_type, img_path.name, score, source_name)

                dist = center_edge_distance(box, w, h)
                spatial_counts[spatial_bin_name(dist)] += 1

            stats_all["gt_dup"] += len(gt_dup)
            stats_by_source[source_name]["gt_dup"] += len(gt_dup)

            del img, gt_arr, pred_arr, iou_mat, white_mask, white_integral, highlight_mask, texture_mask, grad_mask, hl_integral, texture_integral

        del result_map, results, sources, chunk
        gc.collect()

        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()

    samples_top = finalize_samples(sample_by_type, int(args.sample_per_type))
    pred_dup_total = stats_all["dup_state"]["pred_dup"] + stats_all["dup_state"]["both"]
    unmatched_total = stats_all["dup_state"]["unmatched"] + stats_all["dup_state"]["both"]
    both_total = stats_all["dup_state"]["both"]
    other_total = stats_all["dup_state"]["other"]

    type_metrics_rows: List[dict] = []
    type_dup_rows: List[dict] = []
    for t in TYPE_NAMES:
        cnt = stats_all["type_counts"][t]
        score_mean, score_med, score_p90 = stats_summary(stats_all["type_scores"][t])
        iou_mean, iou_med, iou_p90 = stats_summary(stats_all["type_max_ious"][t])
        type_metrics_rows.append(
            {
                "type": t,
                "count": cnt,
                "ratio": safe_ratio(cnt, stats_all["fp_total"]),
                "score_mean": score_mean,
                "score_median": score_med,
                "score_p90": score_p90,
                "max_iou_mean": iou_mean,
                "max_iou_median": iou_med,
                "max_iou_p90": iou_p90,
            }
        )
        dup_counts = stats_all["type_dup_state"][t]
        type_dup_rows.append(
            {
                "type": t,
                "unmatched": dup_counts["unmatched"],
                "pred_dup": dup_counts["pred_dup"],
                "both": dup_counts["both"],
                "other": dup_counts["other"],
                "fp_total": cnt,
            }
        )

    bucket_dup_rows: List[dict] = []
    for name, _, _ in buckets:
        fp = stats_all["bucket_fp"][name]
        dup_counts = stats_all["bucket_dup_state"][name]
        dup_ratio = safe_ratio(dup_counts["pred_dup"] + dup_counts["both"], fp)
        bucket_dup_rows.append(
            {
                "bucket": name,
                "fp_total": fp,
                "unmatched": dup_counts["unmatched"],
                "pred_dup": dup_counts["pred_dup"],
                "both": dup_counts["both"],
                "other": dup_counts["other"],
                "dup_ratio": dup_ratio,
            }
        )

    dup_mult_rows: List[dict] = []
    total_dup_gt = sum(dup_mult_counts.values())
    for k in sorted(dup_mult_counts.keys()):
        dup_mult_rows.append(
            {
                "dup_count": k,
                "gt_count": dup_mult_counts[k],
                "ratio": safe_ratio(dup_mult_counts[k], total_dup_gt),
            }
        )

    # write optional CSVs (<=5)
    fp_summary_rows = [
        {"key": "created_at", "value": dt.datetime.now().isoformat(timespec="seconds")},
        {"key": "weights", "value": str(args.weights)},
        {"key": "image_dir", "value": ", ".join(str(p) for p in image_dirs)},
        {"key": "label_dir", "value": ", ".join(str(p) for p in label_dirs)},
        {"key": "conf", "value": args.conf},
        {"key": "nms_iou", "value": args.nms_iou},
        {"key": "max_det", "value": args.max_det},
        {"key": "tp_iou", "value": args.tp_iou},
        {"key": "batch", "value": args.batch},
        {"key": "infer_chunk", "value": args.infer_chunk},
        {"key": "bucket_edges", "value": args.bucket_edges},
        {"key": "white_thresh", "value": WHITE_THRESH},
        {"key": "fp_total", "value": stats_all["fp_total"]},
        {"key": "pred_dup_total", "value": pred_dup_total},
        {"key": "unmatched_total", "value": unmatched_total},
        {"key": "both_total", "value": both_total},
        {"key": "other_total", "value": other_total},
        {"key": "gt_dup", "value": stats_all["gt_dup"]},
        {"key": "pred_dup_ratio", "value": safe_ratio(pred_dup_total, stats_all["fp_total"])},
        {"key": "unmatched_ratio", "value": safe_ratio(unmatched_total, stats_all["fp_total"])},
        {"key": "both_ratio", "value": safe_ratio(both_total, stats_all["fp_total"])},
    ]
    write_csv(report_dir / "fp_summary.csv", fp_summary_rows, ["key", "value"])
    write_csv(report_dir / "fp_type_metrics.csv", type_metrics_rows, list(type_metrics_rows[0].keys()) if type_metrics_rows else ["type"])
    write_csv(report_dir / "fp_type_x_dup.csv", type_dup_rows, list(type_dup_rows[0].keys()) if type_dup_rows else ["type"])
    write_csv(report_dir / "fp_bucket_x_dup.csv", bucket_dup_rows, list(bucket_dup_rows[0].keys()) if bucket_dup_rows else ["bucket"])
    write_csv(report_dir / "dup_multiplicity.csv", dup_mult_rows, list(dup_mult_rows[0].keys()) if dup_mult_rows else ["dup_count"])

    # write report
    report_path = report_dir / "P2_3_2_fp_dup_typing.md"
    lines: List[str] = []
    lines.append(f"# P2.3.2 误报与 pred_dup 类型化归因")
    lines.append("")
    lines.append(f"- created_at: {dt.datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"- weights: {args.weights}")
    lines.append(f"- image_dir: {', '.join(str(p) for p in image_dirs)}")
    if args.label_dir:
        lines.append(f"- label_dir(override): {', '.join(str(p) for p in label_dirs)}")
    lines.append(f"- out_root: {args.out_root}")
    lines.append("")
    lines.append("## 名词与缩写")
    lines.append("- GT = ground_truth（标注框）")
    lines.append("- FP = false positive（误报）")
    lines.append("- dup = 重复覆盖（同一 GT 被多个预测框覆盖；或单个预测框覆盖多个 GT）")
    lines.append("- NMS = 非极大值抑制")
    lines.append("")
    lines.append("## 口径参数")
    lines.append(f"- conf: {args.conf}")
    lines.append(f"- nms_iou: {args.nms_iou}")
    lines.append(f"- max_det: {args.max_det}")
    lines.append(f"- tp_iou: {args.tp_iou}")
    lines.append(f"- batch: {args.batch}")
    lines.append(f"- infer_chunk: {args.infer_chunk}")
    lines.append("- 推理后处理顺序：conf → NMS(nms_iou) → max_det")
    lines.append("- 坐标口径：预测框为 Ultralytics 原图坐标；GT 为原图像素坐标")
    lines.append(f"- 尺度桶边界(short side): {args.bucket_edges}")
    lines.append("")
    lines.append("## 分类规则（可复核）")
    lines.append("- 高光/强反光类：预测框内 (亮度>=p_bright 且 梯度>=p_grad) 的像素占比 >= hl_frac")
    lines.append("- 边缘/背景侵入类：预测框内白色背景像素占比 >= edge_white_frac（白色阈值：RGB>=250）")
    lines.append("- 纹理过渡/阴影边界类：预测框内 (梯度>=p_grad 且 亮度<p_bright) 的像素占比 >= texture_frac")
    lines.append("- 背景污点/颗粒类：预测框内高梯度小连通域像素占比 >= speckle_frac")
    lines.append("- 其他/未归类：不满足以上规则")
    lines.append("- 分类优先级：highlight → edge_bg → texture_boundary → speckle → other")
    lines.append("- 亮度/梯度分位数阈值在非白背景像素上计算（~white_mask）")
    lines.append("")
    lines.append("阈值说明：")
    lines.append(f"- p_bright(percentile): {args.hl_bright_percentile}")
    lines.append(f"- p_grad(percentile): {args.hl_grad_percentile}")
    lines.append(f"- hl_frac: {args.hl_frac}")
    lines.append(f"- edge_white_frac: {args.edge_white_frac}")
    lines.append(f"- texture_frac: {args.texture_frac}")
    lines.append(f"- speckle_frac: {args.speckle_frac}")
    lines.append(f"- speckle_area_ratio_max: {args.speckle_area_ratio_max}")
    lines.append("")

    lines.append("## 总览（合并）")
    lines.append("- unmatched：与任意 GT 的 IoU 均小于 tp_iou")
    lines.append("- pred_dup：同一 GT 有多个预测框，仅保留一个匹配，其余记为 FP")
    lines.append("- both：同时满足 unmatched 与 pred_dup（若出现，显式统计）")
    lines.append(f"- 目标级 FP 总数: {stats_all['fp_total']}")
    lines.append(
        f"- pred_dup(含both) 数量: {pred_dup_total} (占比 {safe_ratio(pred_dup_total, stats_all['fp_total']):.4f})"
    )
    lines.append(
        f"- unmatched(含both) 数量: {unmatched_total} (占比 {safe_ratio(unmatched_total, stats_all['fp_total']):.4f})"
    )
    lines.append(f"- both 数量: {both_total} (占比 {safe_ratio(both_total, stats_all['fp_total']):.4f})")
    if other_total > 0:
        lines.append(f"- other 数量: {other_total} (占比 {safe_ratio(other_total, stats_all['fp_total']):.4f})")
    lines.append(f"- gt_dup 数量(仅统计): {stats_all['gt_dup']}")
    lines.append("")

    lines.append("## 按来源(val/test)统计")
    lines.append("| source | fp_total | pred_dup(含both) | unmatched(含both) | both | gt_dup |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for source, st in stats_by_source.items():
        pd_total = st["dup_state"]["pred_dup"] + st["dup_state"]["both"]
        um_total = st["dup_state"]["unmatched"] + st["dup_state"]["both"]
        lines.append(
            "| {src} | {fp} | {pd} | {unm} | {both} | {gd} |".format(
                src=source,
                fp=st["fp_total"],
                pd=pd_total,
                unm=um_total,
                both=st["dup_state"]["both"],
                gd=st["gt_dup"],
            )
        )
    lines.append("")

    lines.append("## 表A：FP 类型指标（计数/占比/分数/IoU）")
    lines.append("| type | count | ratio | score_mean | score_median | score_p90 | max_iou_mean | max_iou_median | max_iou_p90 |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for r in type_metrics_rows:
        lines.append(
            "| {type} | {count} | {ratio:.4f} | {score_mean:.4f} | {score_median:.4f} | {score_p90:.4f} | {max_iou_mean:.4f} | {max_iou_median:.4f} | {max_iou_p90:.4f} |".format(
                **r
            )
        )
    lines.append("")

    lines.append("## 表B：FP 类型 × unmatched/pred_dup/both")
    lines.append("| type | unmatched | pred_dup | both | other | fp_total |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for r in type_dup_rows:
        lines.append(
            "| {type} | {unmatched} | {pred_dup} | {both} | {other} | {fp_total} |".format(
                **r
            )
        )
    lines.append("")

    lines.append("## 表C：尺度桶 × unmatched/pred_dup/both（含 dup 占比）")
    lines.append("| bucket | fp_total | unmatched | pred_dup | both | other | dup_ratio |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for r in bucket_dup_rows:
        lines.append(
            "| {bucket} | {fp_total} | {unmatched} | {pred_dup} | {both} | {other} | {dup_ratio:.4f} |".format(
                **r
            )
        )
    lines.append("")

    lines.append("## 表D：dup 倍数分布（每个 GT 被覆盖的预测框数量）")
    lines.append("| dup_count | gt_count | ratio |")
    lines.append("| --- | --- | --- |")
    if dup_mult_rows:
        for r in dup_mult_rows:
            lines.append("| {dup_count} | {gt_count} | {ratio:.4f} |".format(**r))
    else:
        lines.append("| - | 0 | 0.0000 |")
    lines.append("")

    lines.append("## 可选表E：预测框中心距边界分桶（空间分布）")
    lines.append("| bin | fp_count | ratio |")
    lines.append("| --- | --- | --- |")
    fp_total = stats_all["fp_total"]
    for name, _, _ in SPATIAL_BINS:
        cnt = spatial_counts.get(name, 0)
        lines.append(f"| {name} | {cnt} | {safe_ratio(cnt, fp_total):.4f} |")
    lines.append("")

    lines.append("## 样例索引（TopK by score，每类去重）")
    lines.append("| type | samples |")
    lines.append("| --- | --- |")
    for t in TYPE_NAMES:
        samples = samples_top.get(t, [])
        if samples:
            sample_text = ", ".join([f"{src}/{fname} (score={score:.3f})" for fname, score, src in samples])
        else:
            sample_text = "-"
        lines.append(f"| {t} | {sample_text} |")
    lines.append("")

    lines.append("## 关键结论（请根据实际结果复核）")
    lines.append("- 误报的主要类型、dup 占比与尺度桶关系请参考表A/表B/表C。")
    lines.append("- 若某尺度桶内 pred_dup 占比显著升高，优先检查该桶的重复覆盖原因。")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[P2.3.2] report_dir = {report_dir}")
    return report_dir


def main() -> None:
    args = parse_args()
    main_logic(args)


if __name__ == "__main__":
    main()
