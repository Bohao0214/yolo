"""P2.3.3 目标级漏检（FN）机制拆解（YOLO）。

P2.3.0 评估口径冻结：
- 置信度阈值（conf）过滤 -> NMS（非极大值抑制，重叠框去重）-> max_det
- 匹配阈值 tp_iou（IoU=交并比），目标级一对一匹配

术语（首次出现给出中文解释）：
- GT = ground_truth（标注框）
- FP/TP/FN = 误报/命中/漏检
- IoU = 交并比
- NMS = 非极大值抑制（重叠框去重）

FN 诊断类型（diag_type）：
1) no_response：标注框附近几乎无有效响应（重叠候选极少或最高分极低）
2) low_score：有响应但最高分 < conf（被置信度阈值挡住）
3) regression_poor：有较高分候选，但 IoU 始终 < tp_iou（定位/尺寸不稳）
4) postproc_suppressed：conf 后曾出现可达 tp_iou 的候选，但被 NMS 或 max_det 抑制

python /home/ubuntu/project/deduibi/yolo/analyze/code/p23_3_fn_diagnose.py \
  --weights /home/ubuntu/project/deduibi/yolo/models/defect/exp_260202_base/best/best.pt \
  --image_dir /home/ubuntu/project/deduibi/yolo/dataset/yolo/datasetm6c/images/val \
  --image_dir /home/ubuntu/project/deduibi/yolo/dataset/yolo/datasetm6c/images/test \
  --out_root /home/ubuntu/project/deduibi/yolo/analyze/result \
  --batch 4 --infer_chunk 16 \
  --conf 0.3 --tp_iou 0.24 --nms_iou 0.67 --max_det 100

  
python /home/ubuntu/project/deduibi/yolo/analyze/code/p23_3_fn_diagnose.py \
  --reuse_report_dir /home/ubuntu/project/deduibi/yolo/analyze/result/report_2602031613 \
  --out_root /home/ubuntu/project/deduibi/yolo/analyze/result

"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import gc
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import cv2
except Exception as exc:  # pragma: no cover
    cv2 = None  # type: ignore
    CV2_IMPORT_ERROR = exc
else:
    CV2_IMPORT_ERROR = None

try:
    from ultralytics import YOLO
except Exception as exc:  # pragma: no cover
    YOLO = None  # type: ignore
    ULTRALYTICS_IMPORT_ERROR = exc
else:
    ULTRALYTICS_IMPORT_ERROR = None


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SCALE_BUCKETS: Sequence[Tuple[str, float, float]] = (
    ("<16", 0.0, 16.0),
    ("16-32", 16.0, 32.0),
    ("32-64", 32.0, 64.0),
    (">64", 64.0, float("inf")),
)
DIAG_TYPES = ["no_response", "low_score", "regression_poor", "postproc_suppressed"]


def ensure_ultralytics() -> None:
    if YOLO is None:
        raise ImportError(
            "Failed to import ultralytics. Please install ultralytics in the environment. "
            f"Original error: {ULTRALYTICS_IMPORT_ERROR}"
        )


def ensure_cv2() -> None:
    if cv2 is None:
        raise ImportError(
            "Failed to import cv2. Please install opencv-python in the environment. "
            f"Original error: {CV2_IMPORT_ERROR}"
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="P2.3.3 FN 机制拆解诊断脚本。")
    p.add_argument("--weights", type=str, default="/home/ubuntu/project/deduibi/yolo/models/defect/exp_260202_base/best/best.pt")
    p.add_argument(
        "--image_dir",
        type=str,
        action="append",
        default=[],
        help="Dataset image directory (val/test). Can be provided multiple times or comma-separated.",
    )
    p.add_argument("--out_root", type=str, default="/home/ubuntu/project/deduibi/yolo/analyze/result")
    p.add_argument("--reuse_report_dir", type=str, default="", help="Reuse report_dir that may contain infer_export.csv.")
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--infer_chunk", type=int, default=8)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", type=str, default="")

    # P2.3.0 postprocess
    p.add_argument("--conf", type=float, default=0.3)
    p.add_argument("--tp_iou", type=float, default=0.2)
    p.add_argument("--nms_iou", type=float, default=0.6)
    p.add_argument("--max_det", type=int, default=20)

    # diag thresholds
    p.add_argument("--no_resp_score", type=float, default=0.001)
    p.add_argument("--raw_max_det", type=int, default=3000, help="Max raw boxes per image when running inference.")
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
    ts = dt.datetime.now().strftime("report_%Y%m%d%H%M")
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


def compute_iou_vec(gt_box: np.ndarray, pred_boxes: np.ndarray) -> np.ndarray:
    if pred_boxes.size == 0:
        return np.zeros((0,), dtype=np.float32)
    ix1 = np.maximum(gt_box[0], pred_boxes[:, 0])
    iy1 = np.maximum(gt_box[1], pred_boxes[:, 1])
    ix2 = np.minimum(gt_box[2], pred_boxes[:, 2])
    iy2 = np.minimum(gt_box[3], pred_boxes[:, 3])
    iw = np.maximum(0.0, ix2 - ix1)
    ih = np.maximum(0.0, iy2 - iy1)
    inter = iw * ih
    gt_area = max(0.0, (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1]))
    pred_area = np.maximum(0.0, (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1]))
    union = gt_area + pred_area - inter
    return np.where(union > 0, inter / union, 0.0).astype(np.float32)


def nms_numpy(boxes: np.ndarray, scores: np.ndarray, iou_thres: float) -> List[int]:
    if boxes.size == 0:
        return []
    idxs = scores.argsort()[::-1].tolist()
    keep: List[int] = []
    while idxs:
        i = idxs.pop(0)
        keep.append(i)
        if not idxs:
            break
        rest = np.array(idxs, dtype=np.int64)
        ious = compute_iou_vec(boxes[i], boxes[rest])
        idxs = [idxs[j] for j in range(len(idxs)) if ious[j] <= iou_thres]
    return keep


def safe_ratio(num: int, den: int) -> float:
    return float(num) / float(den) if den > 0 else 0.0


def short_side(box: Tuple[float, float, float, float]) -> float:
    return max(0.0, min(box[2] - box[0], box[3] - box[1]))


def bucket_name(val: float) -> str:
    for name, lo, hi in SCALE_BUCKETS:
        if lo <= val < hi:
            return name
    return ">64"


def best_score_with_overlap(gt_box: np.ndarray, boxes: np.ndarray, scores: np.ndarray) -> float:
    if boxes.size == 0:
        return 0.0
    ious = compute_iou_vec(gt_box, boxes)
    mask = ious > 0.0
    if not np.any(mask):
        return 0.0
    return float(np.max(scores[mask]))


def best_iou(gt_box: np.ndarray, boxes: np.ndarray) -> float:
    if boxes.size == 0:
        return 0.0
    return float(np.max(compute_iou_vec(gt_box, boxes)))


def format_box(box: Tuple[float, float, float, float]) -> str:
    return f"{box[0]:.1f},{box[1]:.1f},{box[2]:.1f},{box[3]:.1f}"


def write_csv(path: Path, rows: List[dict], headers: List[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_infer_export(path: Path, image_index: Dict[str, Dict[str, Path]], warnings: List[str]) -> Dict[str, List[Tuple[Tuple[float, float, float, float], float]]]:
    pred_map: Dict[str, List[Tuple[Tuple[float, float, float, float], float]]] = {}
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_path_raw = (row.get("image_path") or "").strip()
            img_path: Optional[Path] = None
            if img_path_raw:
                cand = Path(img_path_raw)
                if cand.exists():
                    img_path = cand
            if img_path is None:
                source = (row.get("source_name") or "").strip()
                image_id = (row.get("image_id") or "").strip()
                if not image_id and img_path_raw:
                    image_id = Path(img_path_raw).stem
                if source and image_id and source in image_index:
                    img_path = image_index[source].get(image_id)
            if img_path is None:
                warnings.append(f"- [提醒] infer_export 无法解析 image_path: {img_path_raw}")
                continue
            if "pred_box" in row and row["pred_box"]:
                parts = [p.strip() for p in row["pred_box"].split(",")]
                if len(parts) != 4:
                    continue
                try:
                    box = (float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]))
                except ValueError:
                    continue
            else:
                keys = ("pred_x1", "pred_y1", "pred_x2", "pred_y2")
                if not all(k in row for k in keys):
                    continue
                try:
                    box = (float(row["pred_x1"]), float(row["pred_y1"]), float(row["pred_x2"]), float(row["pred_y2"]))
                except ValueError:
                    continue
            score_raw = row.get("score") or row.get("pred_score") or row.get("conf") or ""
            try:
                score = float(score_raw)
            except ValueError:
                score = 0.0
            pred_map.setdefault(str(img_path), []).append((box, score))
    return pred_map


def build_image_index(image_dirs: List[Path]) -> Dict[str, Dict[str, Path]]:
    index: Dict[str, Dict[str, Path]] = {}
    for d in image_dirs:
        source = d.name
        index[source] = {}
        for p in list_images(d):
            if p.stem not in index[source]:
                index[source][p.stem] = p
    return index


def process_image(
    img_path: Path,
    label_dir: Path,
    source_name: str,
    preds_raw: List[Tuple[Tuple[float, float, float, float], float]],
    args: argparse.Namespace,
    fn_rows: List[dict],
    diag_counts: Dict[str, int],
    stats: Dict[str, int],
) -> None:
    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {img_path}")
    h, w = img.shape[:2]
    del img

    label_path = label_dir / f"{img_path.stem}.txt"
    gt_boxes = load_labels(label_path, w, h)

    gt_arr = np.array(gt_boxes, dtype=np.float32) if gt_boxes else np.zeros((0, 4), dtype=np.float32)
    pred_boxes = np.array([p[0] for p in preds_raw], dtype=np.float32) if preds_raw else np.zeros((0, 4), dtype=np.float32)
    pred_scores = np.array([p[1] for p in preds_raw], dtype=np.float32) if preds_raw else np.zeros((0,), dtype=np.float32)

    stats["gt_total"] += len(gt_boxes)

    # stage: after conf
    conf_mask = pred_scores >= float(args.conf)
    pred_conf_boxes = pred_boxes[conf_mask] if pred_boxes.size > 0 else pred_boxes
    pred_conf_scores = pred_scores[conf_mask] if pred_scores.size > 0 else pred_scores

    # stage: after NMS
    keep_nms = nms_numpy(pred_conf_boxes, pred_conf_scores, float(args.nms_iou))
    pred_nms_boxes = pred_conf_boxes[keep_nms] if keep_nms else np.zeros((0, 4), dtype=np.float32)
    pred_nms_scores = pred_conf_scores[keep_nms] if keep_nms else np.zeros((0,), dtype=np.float32)

    # stage: after max_det
    if pred_nms_scores.size > 0:
        order = pred_nms_scores.argsort()[::-1]
        order = order[: int(args.max_det)]
        pred_max_boxes = pred_nms_boxes[order]
        pred_max_scores = pred_nms_scores[order]
    else:
        pred_max_boxes = np.zeros((0, 4), dtype=np.float32)
        pred_max_scores = np.zeros((0,), dtype=np.float32)

    stats["pred_total"] += int(pred_max_boxes.shape[0])

    iou_mat = compute_iou_matrix(gt_arr, pred_max_boxes)
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

    tp = len(assigned_pred_for_gt)
    fn = len(gt_boxes) - tp
    fp = int(pred_max_boxes.shape[0]) - tp
    stats["tp"] += tp
    stats["fn"] += fn
    stats["fp"] += fp

    # pred_dup stats
    if iou_mat.size > 0:
        for gi in range(iou_mat.shape[0]):
            dup_cnt = int(np.sum(iou_mat[gi] >= float(args.tp_iou)))
            if dup_cnt >= 2:
                stats["dup_gt_count"] += 1
                stats["pred_dup_extra"] += dup_cnt - 1

    # image quad
    has_gt = len(gt_boxes) > 0
    has_pred = pred_max_boxes.shape[0] > 0
    if has_gt and has_pred:
        stats["img_gt1_pred1"] += 1
    elif has_gt and not has_pred:
        stats["img_gt1_pred0"] += 1
    elif not has_gt and has_pred:
        stats["img_gt0_pred1"] += 1
    else:
        stats["img_gt0_pred0"] += 1

    if fn <= 0:
        return

    for gi, gt in enumerate(gt_boxes):
        if gi in assigned_pred_for_gt:
            continue
        gt_box = np.array(gt, dtype=np.float32)
        best_score_raw = best_score_with_overlap(gt_box, pred_boxes, pred_scores)
        best_score_after_conf = best_score_with_overlap(gt_box, pred_conf_boxes, pred_conf_scores)
        best_iou_after_conf = best_iou(gt_box, pred_conf_boxes)
        best_iou_after_nms = best_iou(gt_box, pred_nms_boxes)
        best_iou_after_maxdet = best_iou(gt_box, pred_max_boxes)

        diag_type = "no_response"
        suppressed_stage = "none"
        if best_iou_after_conf >= float(args.tp_iou):
            if best_iou_after_nms < float(args.tp_iou):
                diag_type = "postproc_suppressed"
                suppressed_stage = "NMS"
            elif best_iou_after_maxdet < float(args.tp_iou):
                diag_type = "postproc_suppressed"
                suppressed_stage = "max_det"
            else:
                diag_type = "regression_poor"
        elif best_score_raw >= float(args.conf):
            if best_iou_after_conf > 0.0:
                diag_type = "regression_poor"
            else:
                diag_type = "no_response"
        elif best_score_raw > float(args.no_resp_score):
            diag_type = "low_score"
            suppressed_stage = "conf"
        else:
            diag_type = "no_response"

        diag_counts[diag_type] += 1

        fn_rows.append(
            {
                "image_path": str(img_path),
                "source_name": source_name,
                "gt_id": gi,
                "gt_xyxy": format_box(gt),
                "gt_short_side_px": f"{short_side(gt):.1f}",
                "gt_count_in_image": len(gt_boxes),
                "best_score_raw": f"{best_score_raw:.6f}",
                "best_score_after_conf": f"{best_score_after_conf:.6f}",
                "best_iou_after_conf": f"{best_iou_after_conf:.6f}",
                "best_iou_after_nms": f"{best_iou_after_nms:.6f}",
                "best_iou_after_maxdet": f"{best_iou_after_maxdet:.6f}",
                "diag_type": diag_type,
                "suppressed_stage": suppressed_stage,
            }
        )


def build_diag_matrix(diag_bucket_counts_by_source: Dict[str, Dict[str, Dict[str, int]]]) -> List[dict]:
    rows: List[dict] = []
    for source_name, bucket_counts in diag_bucket_counts_by_source.items():
        for bucket, diag_counts in bucket_counts.items():
            total = sum(diag_counts.values())
            for diag in DIAG_TYPES:
                rows.append(
                    {
                        "source_name": source_name,
                        "scale_bucket": bucket,
                        "diag_type": diag,
                        "count": diag_counts[diag],
                        "ratio_within_bucket": f"{safe_ratio(diag_counts[diag], total):.6f}",
                    }
                )
    return rows


def write_readme(path: Path, args: argparse.Namespace) -> None:
    lines: List[str] = []
    lines.append("# P2.3.3 目标级漏检（FN）机制拆解")
    lines.append("")
    lines.append("## 口径")
    lines.append("- conf -> NMS -> max_det；一对一匹配 @ tp_iou。")
    lines.append("- IoU=交并比；NMS=非极大值抑制（重叠框去重）；GT=标注框；FP/TP/FN=误报/命中/漏检。")
    lines.append("")
    lines.append("## 诊断类型")
    lines.append("- no_response：标注框附近几乎无有效响应（重叠候选极少或最高分极低）。")
    lines.append("- low_score：有响应但最高分 < conf（被置信度阈值挡住）。")
    lines.append("- regression_poor：有较高分候选，但 IoU 始终 < tp_iou。")
    lines.append("- postproc_suppressed：conf 后曾出现可达 tp_iou 的候选，但被 NMS 或 max_det 抑制。")
    lines.append("")
    lines.append("## 运行示例")
    lines.append(
        "python /home/ubuntu/project/deduibi/yolo/analyze/code/p23_3_fn_diagnose.py \\\n"
        "  --weights /home/ubuntu/project/deduibi/yolo/models/defect/exp_260202_base/best/best.pt \\\n"
        "  --image_dir /home/ubuntu/project/deduibi/yolo/dataset/yolo/datasetm6c/images/val \\\n"
        "  --image_dir /home/ubuntu/project/deduibi/yolo/dataset/yolo/datasetm6c/images/test \\\n"
        "  --out_root /home/ubuntu/project/deduibi/yolo/analyze/result \\\n"
        "  --batch 4 --infer_chunk 8 \\\n"
        "  --conf 0.3 --tp_iou 0.2 --nms_iou 0.6 --max_det 20"
    )
    if args.reuse_report_dir:
        lines.append("")
        lines.append("## 复用导出")
        lines.append("若 reuse_report_dir 中存在 infer_export.csv，将优先读取预测框；缺失时回退到重新推理。")
        lines.append("若 infer_export.csv 已经过 conf 过滤，low_score/no_response 可能偏小。")
    with path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    ensure_cv2()
    if int(args.batch) >= 8:
        raise ValueError("--batch 必须 < 8")
    if int(args.infer_chunk) > 64:
        raise ValueError("--infer_chunk 不要太高（建议 <= 16）")

    reuse_args: Dict[str, object] = {}
    if args.reuse_report_dir:
        run_args_path = Path(args.reuse_report_dir) / "run_args.json"
        if run_args_path.exists():
            with run_args_path.open("r", encoding="utf-8") as f:
                reuse_args = json.load(f)

    image_dirs = normalize_path_list(args.image_dir)
    if not image_dirs:
        reuse_dirs = reuse_args.get("image_dir", [])
        image_dirs = [Path(p) for p in reuse_dirs] if reuse_dirs else []
    if not image_dirs:
        raise ValueError("未提供 image_dir，且 reuse_report_dir 中未找到 image_dir。")

    for d in image_dirs:
        if not d.exists():
            raise FileNotFoundError(f"image_dir not found: {d}")

    label_dirs = [infer_label_dir(d) for d in image_dirs]
    warnings: List[str] = []
    for d in label_dirs:
        if not d.exists():
            warnings.append(f"- [提醒] label_dir 不存在: {d}")

    report_dir = make_report_dir(Path(args.out_root))

    run_args = {
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "weights": str(args.weights),
        "image_dir": [str(p) for p in image_dirs],
        "label_dir": [str(p) for p in label_dirs],
        "out_root": str(args.out_root),
        "report_dir": str(report_dir),
        "reuse_report_dir": str(args.reuse_report_dir) if args.reuse_report_dir else "",
        "conf": float(args.conf),
        "nms_iou": float(args.nms_iou),
        "max_det": int(args.max_det),
        "tp_iou": float(args.tp_iou),
        "batch": int(args.batch),
        "infer_chunk": int(args.infer_chunk),
        "imgsz": int(args.imgsz),
        "device": str(args.device),
        "no_resp_score": float(args.no_resp_score),
        "raw_max_det": int(args.raw_max_det),
        "eval_pipeline": "conf -> NMS -> max_det; one-to-one match @ tp_iou",
        "raw_stage": "推理阶段使用 conf=0.0, iou=1.0, max_det=raw_max_det 近似保留原始候选，再由脚本执行 conf/NMS/max_det。",
        "no_response_rule": "best_score_raw <= no_resp_score 时判为 no_response；best_score_raw 在 (no_resp_score, conf) 判为 low_score。",
        "diag_definitions": {
            "no_response": "标注框附近几乎无有效响应（重叠候选极少或最高分极低）。",
            "low_score": "有响应但最高分 < conf（被置信度阈值挡住）。",
            "regression_poor": "有较高分候选，但 IoU 始终 < tp_iou。",
            "postproc_suppressed": "conf 后曾出现可达 tp_iou 的候选，但被 NMS 或 max_det 抑制。",
        },
        "fields": {
            "fn_cases": [
                "image_path",
                "source_name",
                "gt_id",
                "gt_xyxy(像素)",
                "gt_short_side_px",
                "gt_count_in_image",
                "best_score_raw",
                "best_score_after_conf",
                "best_iou_after_conf",
                "best_iou_after_nms",
                "best_iou_after_maxdet",
                "diag_type",
                "suppressed_stage",
            ],
            "fn_diag_matrix": [
                "source_name",
                "scale_bucket(<16,16-32,32-64,>64)",
                "diag_type",
                "count",
                "ratio_within_bucket",
            ],
        },
        "abbr": {
            "IoU": "交并比",
            "NMS": "非极大值抑制（重叠框去重）",
            "GT": "标注框",
            "FP/TP/FN": "误报/命中/漏检",
        },
    }
    with (report_dir / "run_args.json").open("w", encoding="utf-8") as f:
        json.dump(run_args, f, ensure_ascii=False, indent=2)

    # build items
    items: List[Tuple[Path, Path, str]] = []
    for img_dir, lbl_dir in zip(image_dirs, label_dirs):
        source_name = img_dir.name
        for img_path in list_images(img_dir):
            items.append((img_path, lbl_dir, source_name))
    if not items:
        raise RuntimeError("未找到图像。")

    stats = {
        "gt_total": 0,
        "pred_total": 0,
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "pred_dup_extra": 0,
        "dup_gt_count": 0,
        "img_gt1_pred1": 0,
        "img_gt1_pred0": 0,
        "img_gt0_pred1": 0,
        "img_gt0_pred0": 0,
    }
    diag_counts: Dict[str, int] = {k: 0 for k in DIAG_TYPES}
    fn_rows: List[dict] = []

    # reuse export if available
    pred_map: Dict[str, List[Tuple[Tuple[float, float, float, float], float]]] = {}
    use_export = False
    if args.reuse_report_dir:
        export_path = Path(args.reuse_report_dir) / "infer_export.csv"
        if export_path.exists():
            image_index = build_image_index(image_dirs)
            pred_map = parse_infer_export(export_path, image_index, warnings)
            if pred_map:
                use_export = True
            else:
                warnings.append("- [提醒] infer_export.csv 解析为空，将回退到重新推理。")
        else:
            warnings.append("- [提醒] reuse_report_dir 未找到 infer_export.csv，将回退到重新推理。")

    if use_export:
        for img_path, lbl_dir, source_name in items:
            preds_raw = pred_map.get(str(img_path), [])
            process_image(
                img_path,
                lbl_dir,
                source_name,
                preds_raw,
                args,
                fn_rows,
                diag_counts,
                stats,
            )
    else:
        ensure_ultralytics()
        model = YOLO(str(args.weights))
        try:
            import torch
        except Exception:
            torch = None  # type: ignore

        for start in range(0, len(items), int(args.infer_chunk)):
            chunk = items[start : start + int(args.infer_chunk)]
            sources = [str(p[0]) for p in chunk]
            results = model.predict(
                source=sources,
                imgsz=int(args.imgsz),
                conf=0.0,
                iou=1.0,
                max_det=int(args.raw_max_det),
                save=False,
                verbose=False,
                batch=int(args.batch),
                device=str(args.device) if args.device else None,
                stream=True,
            )
            result_map = {str(res.path): res for res in results}

            for img_path, lbl_dir, source_name in chunk:
                res = result_map.get(str(img_path))
                if res is None:
                    warnings.append(f"- [提醒] 推理结果缺失: {img_path}")
                    preds_raw: List[Tuple[Tuple[float, float, float, float], float]] = []
                else:
                    if getattr(res, "orig_shape", None):
                        h, w = int(res.orig_shape[0]), int(res.orig_shape[1])
                    else:
                        img_tmp = cv2.imread(str(img_path))
                        if img_tmp is None:
                            raise FileNotFoundError(f"Failed to read image: {img_path}")
                        h, w = img_tmp.shape[:2]
                        del img_tmp

                    boxes = res.boxes
                    preds_raw = []
                    if boxes is not None:
                        if getattr(boxes, "xyxyn", None) is not None:
                            xyxyn = boxes.xyxyn.detach().cpu().numpy()
                            xyxy = np.zeros_like(xyxyn)
                            xyxy[:, [0, 2]] = xyxyn[:, [0, 2]] * float(w)
                            xyxy[:, [1, 3]] = xyxyn[:, [1, 3]] * float(h)
                        else:
                            xyxy = boxes.xyxy.detach().cpu().numpy() if boxes.xyxy is not None else np.zeros((0, 4))
                        confs = boxes.conf.detach().cpu().numpy() if getattr(boxes, "conf", None) is not None else np.zeros(len(xyxy))
                        for box, score in zip(xyxy, confs):
                            preds_raw.append(
                                (
                                    (float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                                    float(score),
                                )
                            )

                process_image(
                    img_path,
                    lbl_dir,
                    source_name,
                    preds_raw,
                    args,
                    fn_rows,
                    diag_counts,
                    stats,
                )

            del result_map, results, sources, chunk
            gc.collect()
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()

    # rebuild aggregated bucket counts from fn_rows to ensure correctness
    diag_bucket_counts_by_source_all = {name: {k: 0 for k in DIAG_TYPES} for name, _, _ in SCALE_BUCKETS}
    diag_bucket_counts_by_source_map: Dict[str, Dict[str, Dict[str, int]]] = {}
    for img_dir in image_dirs:
        diag_bucket_counts_by_source_map[img_dir.name] = {name: {k: 0 for k in DIAG_TYPES} for name, _, _ in SCALE_BUCKETS}

    for row in fn_rows:
        source = row["source_name"]
        gt_short = float(row["gt_short_side_px"])
        bucket = bucket_name(gt_short)
        diag = row["diag_type"]
        diag_bucket_counts_by_source_map[source][bucket][diag] += 1
        diag_bucket_counts_by_source_all[bucket][diag] += 1

    diag_bucket_counts_by_source_map["all"] = diag_bucket_counts_by_source_all

    # write outputs
    fn_cases_path = report_dir / "fn_cases.csv"
    write_csv(
        fn_cases_path,
        fn_rows,
        [
            "image_path",
            "source_name",
            "gt_id",
            "gt_xyxy",
            "gt_short_side_px",
            "gt_count_in_image",
            "best_score_raw",
            "best_score_after_conf",
            "best_iou_after_conf",
            "best_iou_after_nms",
            "best_iou_after_maxdet",
            "diag_type",
            "suppressed_stage",
        ],
    )

    diag_matrix_rows = build_diag_matrix(diag_bucket_counts_by_source_map)
    diag_matrix_path = report_dir / "fn_diag_matrix.csv"
    write_csv(
        diag_matrix_path,
        diag_matrix_rows,
        ["source_name", "scale_bucket", "diag_type", "count", "ratio_within_bucket"],
    )

    fn_total = stats["fn"]
    fn_diag_ratios = {k: safe_ratio(diag_counts[k], fn_total) for k in DIAG_TYPES}

    summary = {
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "counts": {
            "gt_total": stats["gt_total"],
            "pred_total": stats["pred_total"],
            "tp": stats["tp"],
            "fp": stats["fp"],
            "fn": stats["fn"],
        },
        "image_quad": {
            "gt1_pred1": stats["img_gt1_pred1"],
            "gt1_pred0": stats["img_gt1_pred0"],
            "gt0_pred1": stats["img_gt0_pred1"],
            "gt0_pred0": stats["img_gt0_pred0"],
        },
        "dup": {
            "pred_dup_extra": stats["pred_dup_extra"],
            "dup_gt_count": stats["dup_gt_count"],
        },
        "fn_diag": {
            "counts": diag_counts,
            "ratios": fn_diag_ratios,
        },
        "sanity": {
            "gt_total_gt0": stats["gt_total"] > 0,
        },
        "warnings": warnings,
    }
    summary_path = report_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    readme_path = report_dir / "p23_3_readme.md"
    write_readme(readme_path, args)

    if not summary["sanity"]["gt_total_gt0"]:
        raise RuntimeError("GT_total=0，已写入 summary.json 的 sanity，但需检查数据集标签。")

    if warnings:
        print("[WARN] 统计提醒：")
        for w in warnings:
            print(w)

    print(f"[DONE] report_dir: {report_dir}")
    print(f"[DONE] fn_cases: {fn_cases_path}")
    print(f"[DONE] fn_diag_matrix: {diag_matrix_path}")
    print(f"[DONE] summary: {summary_path}")
    print(f"[DONE] readme: {readme_path}")


if __name__ == "__main__":
    main()
