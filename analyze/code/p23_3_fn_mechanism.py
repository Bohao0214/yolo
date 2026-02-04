"""P2.3.3（整合版）图像级 + 目标级 FN 机制拆解（YOLO）。

P2.3.0 评估口径冻结：
- 置信度阈值（conf）过滤 -> NMS（非极大值抑制，重叠框去重）-> max_det
- 匹配阈值 tp_iou（IoU=交并比），目标级一对一匹配（匈牙利算法最大化 IoU 总和）

术语（首次出现给出中文解释）：
- IoU = 交并比
- NMS = 非极大值抑制（重叠框去重）
- GT = 标注框
- FP/TP/FN = 误报/命中/漏检

python /home/ubuntu/project/deduibi/yolo/analyze/code/p23_3_fn_mechanism.py \
  --weights /home/ubuntu/project/deduibi/yolo/models/defect/exp_260202_base/best/best.pt \
  --image_dir /home/ubuntu/project/deduibi/yolo/dataset/yolo/datasetm6c/images/val \
  --image_dir /home/ubuntu/project/deduibi/yolo/dataset/yolo/datasetm6c/images/test \
  --out_root /home/ubuntu/project/deduibi/yolo/analyze/result \
  --conf 0.3 --tp_iou 0.2 --nms_iou 0.6 --max_det 20 \
  --batch 4 --infer_chunk 16

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
    p = argparse.ArgumentParser(description="P2.3.3 FN 机制拆解（图像级 + 目标级）。")
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
    p.add_argument("--infer_chunk", type=int, default=16)
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

    # density / highlight
    p.add_argument("--dense_threshold", type=int, default=4)
    p.add_argument("--enable_highlight", type=int, default=1)
    p.add_argument("--white_thresh", type=int, default=250)
    p.add_argument("--hl_bright_percentile", type=float, default=95.0)
    p.add_argument("--hl_grad_percentile", type=float, default=90.0)
    p.add_argument("--highlight_frac", type=float, default=0.05)
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


def hungarian_assign(cost: np.ndarray) -> List[int]:
    """Hungarian algorithm (min-cost). Returns assignment for each row -> column index."""
    n, m = cost.shape
    if n == 0:
        return []
    if n > m:
        pad = np.full((n, n - m), 1.0, dtype=cost.dtype)
        cost = np.hstack([cost, pad])
        m = n
    u = np.zeros(n + 1, dtype=np.float32)
    v = np.zeros(m + 1, dtype=np.float32)
    p = np.zeros(m + 1, dtype=np.int64)
    way = np.zeros(m + 1, dtype=np.int64)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(m + 1, np.inf, dtype=np.float32)
        used = np.zeros(m + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = np.inf
            j1 = 0
            for j in range(1, m + 1):
                if used[j]:
                    continue
                cur = cost[i0 - 1, j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    assignment = [-1] * n
    for j in range(1, m + 1):
        if p[j] != 0:
            assignment[p[j] - 1] = j - 1
    return assignment


def safe_ratio(num: int, den: int) -> float:
    return float(num) / float(den) if den > 0 else 0.0


def short_side(box: Tuple[float, float, float, float]) -> float:
    return max(0.0, min(box[2] - box[0], box[3] - box[1]))


def bucket_name(val: float) -> str:
    for name, lo, hi in SCALE_BUCKETS:
        if lo <= val < hi:
            return name
    return ">64"


def format_box(box: Tuple[float, float, float, float]) -> str:
    return f"{box[0]:.1f},{box[1]:.1f},{box[2]:.1f},{box[3]:.1f}"


def write_csv(path: Path, rows: List[dict], headers: List[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_image_index(image_dirs: List[Path]) -> Dict[str, Dict[str, Path]]:
    index: Dict[str, Dict[str, Path]] = {}
    for d in image_dirs:
        source = d.name
        index[source] = {}
        for p in list_images(d):
            if p.stem not in index[source]:
                index[source][p.stem] = p
    return index


def parse_infer_export(
    path: Path, image_index: Dict[str, Dict[str, Path]], warnings: List[str]
) -> Dict[str, List[Tuple[Tuple[float, float, float, float], float]]]:
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


def compute_integral(mask: np.ndarray) -> np.ndarray:
    return cv2.integral(mask.astype(np.uint8))


def mask_ratio_for_box(integral: np.ndarray, box: Tuple[float, float, float, float], w: int, h: int) -> float:
    x1, y1, x2, y2 = box
    x1i = max(0, min(w, int(np.floor(x1))))
    x2i = max(0, min(w, int(np.ceil(x2))))
    y1i = max(0, min(h, int(np.floor(y1))))
    y2i = max(0, min(h, int(np.ceil(y2))))
    if x2i <= x1i or y2i <= y1i:
        return 0.0
    area = float((x2i - x1i) * (y2i - y1i))
    if area <= 0:
        return 0.0
    s = integral[y2i, x2i] - integral[y1i, x2i] - integral[y2i, x1i] + integral[y1i, x1i]
    return float(s) / area


def build_highlight_mask(
    img_bgr: np.ndarray,
    white_thresh: int,
    bright_percentile: float,
    grad_percentile: float,
) -> np.ndarray:
    white_mask = (
        (img_bgr[:, :, 0] >= white_thresh)
        & (img_bgr[:, :, 1] >= white_thresh)
        & (img_bgr[:, :, 2] >= white_thresh)
    )
    valid_mask = ~white_mask
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(gx, gy)
    if np.any(valid_mask):
        gray_vals = gray[valid_mask]
        grad_vals = grad[valid_mask]
    else:
        gray_vals = gray.reshape(-1)
        grad_vals = grad.reshape(-1)
    bright_thresh = float(np.percentile(gray_vals, bright_percentile))
    grad_thresh = float(np.percentile(grad_vals, grad_percentile))
    bright_mask = gray >= bright_thresh
    grad_mask = grad >= grad_thresh
    if np.any(valid_mask):
        bright_mask = bright_mask & valid_mask
        grad_mask = grad_mask & valid_mask
    highlight_mask = bright_mask & grad_mask
    return highlight_mask.astype(np.uint8)


def main() -> None:
    args = parse_args()
    ensure_cv2()
    if int(args.batch) >= 8:
        raise ValueError("--batch 必须 < 8")
    if int(args.infer_chunk) > 16:
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

    config = {
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
        "dense_threshold": int(args.dense_threshold),
        "enable_highlight": bool(int(args.enable_highlight)),
        "highlight_params": {
            "white_thresh": int(args.white_thresh),
            "hl_bright_percentile": float(args.hl_bright_percentile),
            "hl_grad_percentile": float(args.hl_grad_percentile),
            "highlight_frac": float(args.highlight_frac),
        },
        "eval_pipeline": "conf -> NMS -> max_det; one-to-one match @ tp_iou (Hungarian maximize IoU)",
        "diag_definitions": {
            "no_response": "标注框附近几乎无有效响应（重叠候选极少或最高分极低）。",
            "low_score": "存在候选但最高分 < conf（被置信度阈值挡住）。",
            "regression_poor": "最高分 >= conf，但最大 IoU 始终 < tp_iou；或匹配冲突未分配到候选。",
            "postproc_suppressed": "conf 后出现可达 tp_iou 的候选，但在 NMS 或 max_det 阶段被移除。",
        },
        "fn_diag_summary_stat_type": {
            "overall": "FN 四类总体占比",
            "scale": "按 GT 短边尺度分桶",
            "density": "按 gt_count_in_image 密集度分桶",
            "highlight": "按高光/非高光分桶（如启用）",
            "image_fn_gt_scale": "图像级 FN 图中的 GT 尺度分布",
            "image_fn_gt_density": "图像级 FN 图的密集度分布",
        },
        "abbr": {
            "IoU": "交并比",
            "NMS": "非极大值抑制（重叠框去重）",
            "GT": "标注框",
            "FP/TP/FN": "误报/命中/漏检",
        },
    }
    with (report_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    items: List[Tuple[Path, Path, str]] = []
    for img_dir, lbl_dir in zip(image_dirs, label_dirs):
        source_name = img_dir.name
        for img_path in list_images(img_dir):
            items.append((img_path, lbl_dir, source_name))
    if not items:
        raise RuntimeError("未找到图像。")

    # image-level summary
    img_stats: Dict[str, Dict[str, int]] = {}
    for d in image_dirs:
        img_stats[d.name] = {
            "img_total": 0,
            "gt1_pred1": 0,
            "gt1_pred0": 0,
            "gt0_pred1": 0,
            "gt0_pred0": 0,
        }
    img_stats["all"] = {k: 0 for k in img_stats[image_dirs[0].name].keys()}

    image_fn_cases: List[dict] = []
    image_fn_gt_scale: Dict[str, Dict[str, int]] = {name: {b[0]: 0 for b in SCALE_BUCKETS} for name in list(img_stats.keys())}
    image_fn_density: Dict[str, Dict[str, int]] = {name: {"dense": 0, "sparse": 0} for name in list(img_stats.keys())}

    fn_rows: List[dict] = []

    diag_counts_by_source: Dict[str, Dict[str, int]] = {name: {k: 0 for k in DIAG_TYPES} for name in img_stats.keys()}
    diag_scale_by_source: Dict[str, Dict[str, Dict[str, int]]] = {
        name: {b[0]: {k: 0 for k in DIAG_TYPES} for b in SCALE_BUCKETS} for name in img_stats.keys()
    }
    diag_density_by_source: Dict[str, Dict[str, Dict[str, int]]] = {
        name: {"dense": {k: 0 for k in DIAG_TYPES}, "sparse": {k: 0 for k in DIAG_TYPES}} for name in img_stats.keys()
    }
    diag_highlight_by_source: Dict[str, Dict[str, Dict[str, int]]] = {
        name: {"highlight": {k: 0 for k in DIAG_TYPES}, "non_highlight": {k: 0 for k in DIAG_TYPES}}
        for name in img_stats.keys()
    }

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

    if not use_export:
        ensure_ultralytics()
        model = YOLO(str(args.weights))
        try:
            import torch
        except Exception:
            torch = None  # type: ignore
    else:
        model = None  # type: ignore
        torch = None  # type: ignore

    for start in range(0, len(items), int(args.infer_chunk)):
        chunk = items[start : start + int(args.infer_chunk)]
        result_map: Dict[str, object] = {}
        if model is not None:
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
            # load image size
            h, w = 0, 0
            if model is not None:
                res = result_map.get(str(img_path))
                if res is not None and getattr(res, "orig_shape", None):
                    h, w = int(res.orig_shape[0]), int(res.orig_shape[1])
            if h == 0 or w == 0:
                img_tmp = cv2.imread(str(img_path))
                if img_tmp is None:
                    raise FileNotFoundError(f"Failed to read image: {img_path}")
                h, w = img_tmp.shape[:2]
                del img_tmp

            label_path = lbl_dir / f"{img_path.stem}.txt"
            gt_boxes = load_labels(label_path, w, h)
            gt_arr = np.array(gt_boxes, dtype=np.float32) if gt_boxes else np.zeros((0, 4), dtype=np.float32)

            # raw preds
            preds_raw: List[Tuple[Tuple[float, float, float, float], float]] = []
            if use_export:
                preds_raw = pred_map.get(str(img_path), [])
            else:
                res = result_map.get(str(img_path))
                if res is not None:
                    boxes = res.boxes
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

            pred_boxes = np.array([p[0] for p in preds_raw], dtype=np.float32) if preds_raw else np.zeros((0, 4), dtype=np.float32)
            pred_scores = np.array([p[1] for p in preds_raw], dtype=np.float32) if preds_raw else np.zeros((0,), dtype=np.float32)

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
                order = pred_nms_scores.argsort()[::-1][: int(args.max_det)]
                pred_final_boxes = pred_nms_boxes[order]
                pred_final_scores = pred_nms_scores[order]
            else:
                pred_final_boxes = np.zeros((0, 4), dtype=np.float32)
                pred_final_scores = np.zeros((0,), dtype=np.float32)

            # image-level stats
            img_stats[source_name]["img_total"] += 1
            img_stats["all"]["img_total"] += 1
            has_gt = len(gt_boxes) > 0
            has_pred = pred_final_boxes.shape[0] > 0
            if has_gt and has_pred:
                img_stats[source_name]["gt1_pred1"] += 1
                img_stats["all"]["gt1_pred1"] += 1
            elif has_gt and not has_pred:
                img_stats[source_name]["gt1_pred0"] += 1
                img_stats["all"]["gt1_pred0"] += 1
            elif (not has_gt) and has_pred:
                img_stats[source_name]["gt0_pred1"] += 1
                img_stats["all"]["gt0_pred1"] += 1
            else:
                img_stats[source_name]["gt0_pred0"] += 1
                img_stats["all"]["gt0_pred0"] += 1

            if has_gt and not has_pred:
                image_fn_cases.append(
                    {
                        "image_path": str(img_path),
                        "source_name": source_name,
                        "gt_count": len(gt_boxes),
                    }
                )
                density_bucket = "dense" if len(gt_boxes) >= int(args.dense_threshold) else "sparse"
                image_fn_density[source_name][density_bucket] += 1
                image_fn_density["all"][density_bucket] += 1
                for gt in gt_boxes:
                    bucket = bucket_name(short_side(gt))
                    image_fn_gt_scale[source_name][bucket] += 1
                    image_fn_gt_scale["all"][bucket] += 1

            # target-level matching (Hungarian maximize IoU)
            if gt_arr.size > 0 and pred_final_boxes.size > 0:
                iou_mat = compute_iou_matrix(gt_arr, pred_final_boxes)
            else:
                iou_mat = np.zeros((gt_arr.shape[0], pred_final_boxes.shape[0]), dtype=np.float32)

            if iou_mat.size > 0:
                cost = np.where(iou_mat >= float(args.tp_iou), 1.0 - iou_mat, 1.0)
                assignment = hungarian_assign(cost)
            else:
                assignment = [-1] * gt_arr.shape[0]

            matched_pred = set()
            matched_gt = set()
            for gi, pj in enumerate(assignment):
                if pj is None or pj < 0:
                    continue
                if pj >= pred_final_boxes.shape[0]:
                    continue
                if iou_mat.size == 0:
                    continue
                if iou_mat[gi, pj] >= float(args.tp_iou):
                    matched_gt.add(gi)
                    matched_pred.add(pj)

            tp = len(matched_gt)
            fn = len(gt_boxes) - tp
            fp = int(pred_final_boxes.shape[0]) - len(matched_pred)

            # pred_dup stats
            if iou_mat.size > 0:
                for gi in range(iou_mat.shape[0]):
                    dup_cnt = int(np.sum(iou_mat[gi] >= float(args.tp_iou)))
                    if dup_cnt >= 2:
                        # gt 被多个预测覆盖
                        pass

            # target-level FN diagnosis
            if fn > 0:
                # prepare highlight mask if enabled
                highlight_integral = None
                if int(args.enable_highlight) == 1:
                    img = cv2.imread(str(img_path))
                    if img is None:
                        raise FileNotFoundError(f"Failed to read image: {img_path}")
                    hl_mask = build_highlight_mask(
                        img,
                        int(args.white_thresh),
                        float(args.hl_bright_percentile),
                        float(args.hl_grad_percentile),
                    )
                    highlight_integral = compute_integral(hl_mask)
                    del img, hl_mask

                for gi, gt in enumerate(gt_boxes):
                    if gi in matched_gt:
                        continue
                    gt_box = np.array(gt, dtype=np.float32)

                    best_score_raw = float(np.max(pred_scores[compute_iou_vec(gt_box, pred_boxes) > 0.0])) if pred_scores.size > 0 and np.any(compute_iou_vec(gt_box, pred_boxes) > 0.0) else 0.0
                    best_score_after_conf = float(np.max(pred_conf_scores[compute_iou_vec(gt_box, pred_conf_boxes) > 0.0])) if pred_conf_scores.size > 0 and np.any(compute_iou_vec(gt_box, pred_conf_boxes) > 0.0) else 0.0
                    best_iou_raw = float(np.max(compute_iou_vec(gt_box, pred_boxes))) if pred_boxes.size > 0 else 0.0
                    best_iou_after_conf = float(np.max(compute_iou_vec(gt_box, pred_conf_boxes))) if pred_conf_boxes.size > 0 else 0.0
                    best_iou_after_nms = float(np.max(compute_iou_vec(gt_box, pred_nms_boxes))) if pred_nms_boxes.size > 0 else 0.0
                    best_iou_after_maxdet = float(np.max(compute_iou_vec(gt_box, pred_final_boxes))) if pred_final_boxes.size > 0 else 0.0

                    diag_type = "no_response"
                    suppressed_stage = "none"
                    if best_score_raw <= float(args.no_resp_score) or best_iou_raw == 0.0:
                        diag_type = "no_response"
                    elif best_score_raw < float(args.conf):
                        diag_type = "low_score"
                        suppressed_stage = "conf"
                    elif best_iou_after_conf < float(args.tp_iou):
                        diag_type = "regression_poor"
                    else:
                        if best_iou_after_nms < float(args.tp_iou):
                            diag_type = "postproc_suppressed"
                            suppressed_stage = "NMS"
                        elif best_iou_after_maxdet < float(args.tp_iou):
                            diag_type = "postproc_suppressed"
                            suppressed_stage = "max_det"
                        else:
                            diag_type = "regression_poor"

                    diag_counts_by_source[source_name][diag_type] += 1
                    diag_counts_by_source["all"][diag_type] += 1

                    scale_bucket = bucket_name(short_side(gt))
                    diag_scale_by_source[source_name][scale_bucket][diag_type] += 1
                    diag_scale_by_source["all"][scale_bucket][diag_type] += 1

                    density_bucket = "dense" if len(gt_boxes) >= int(args.dense_threshold) else "sparse"
                    diag_density_by_source[source_name][density_bucket][diag_type] += 1
                    diag_density_by_source["all"][density_bucket][diag_type] += 1

                    highlight_flag = "non_highlight"
                    if highlight_integral is not None:
                        ratio = mask_ratio_for_box(highlight_integral, gt, w, h)
                        if ratio >= float(args.highlight_frac):
                            highlight_flag = "highlight"
                    diag_highlight_by_source[source_name][highlight_flag][diag_type] += 1
                    diag_highlight_by_source["all"][highlight_flag][diag_type] += 1

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
                            "scale_bucket": scale_bucket,
                            "density_bucket": density_bucket,
                            "highlight_flag": highlight_flag if int(args.enable_highlight) == 1 else "not_enabled",
                        }
                    )

                if highlight_integral is not None:
                    del highlight_integral

            del gt_arr, pred_boxes, pred_scores, pred_conf_boxes, pred_conf_scores, pred_nms_boxes, pred_nms_scores, pred_final_boxes, pred_final_scores

        if model is not None:
            del result_map
            gc.collect()
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()

    # write image-level summary
    image_level_rows: List[dict] = []
    for source_name, s in img_stats.items():
        recall = safe_ratio(s["gt1_pred1"], s["gt1_pred1"] + s["gt1_pred0"])
        fp_rate = safe_ratio(s["gt0_pred1"], s["gt0_pred1"] + s["gt0_pred0"])
        precision = safe_ratio(s["gt1_pred1"], s["gt1_pred1"] + s["gt0_pred1"])
        image_level_rows.append(
            {
                "source_name": source_name,
                "image_total": s["img_total"],
                "img_gt1_pred1": s["gt1_pred1"],
                "img_gt1_pred0": s["gt1_pred0"],
                "img_gt0_pred1": s["gt0_pred1"],
                "img_gt0_pred0": s["gt0_pred0"],
                "image_recall": f"{recall:.6f}",
                "image_fp_rate": f"{fp_rate:.6f}",
                "image_precision": f"{precision:.6f}",
            }
        )
    write_csv(
        report_dir / "image_level.csv",
        image_level_rows,
        [
            "source_name",
            "image_total",
            "img_gt1_pred1",
            "img_gt1_pred0",
            "img_gt0_pred1",
            "img_gt0_pred0",
            "image_recall",
            "image_fp_rate",
            "image_precision",
        ],
    )

    write_csv(
        report_dir / "image_fn_cases.csv",
        image_fn_cases,
        ["image_path", "source_name", "gt_count"],
    )

    # write fn_cases
    write_csv(
        report_dir / "fn_cases.csv",
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
            "scale_bucket",
            "density_bucket",
            "highlight_flag",
        ],
    )

    # build fn_diag_summary
    fn_diag_rows: List[dict] = []
    for source_name in diag_counts_by_source.keys():
        total = sum(diag_counts_by_source[source_name].values())
        for diag in DIAG_TYPES:
            fn_diag_rows.append(
                {
                    "source_name": source_name,
                    "stat_type": "overall",
                    "diag_type": diag,
                    "bucket": "all",
                    "count": diag_counts_by_source[source_name][diag],
                    "ratio_within_group": f"{safe_ratio(diag_counts_by_source[source_name][diag], total):.6f}",
                }
            )
        for bucket in diag_scale_by_source[source_name]:
            b_total = sum(diag_scale_by_source[source_name][bucket].values())
            for diag in DIAG_TYPES:
                fn_diag_rows.append(
                    {
                        "source_name": source_name,
                        "stat_type": "scale",
                        "diag_type": diag,
                        "bucket": bucket,
                        "count": diag_scale_by_source[source_name][bucket][diag],
                        "ratio_within_group": f"{safe_ratio(diag_scale_by_source[source_name][bucket][diag], b_total):.6f}",
                    }
                )
        for bucket in diag_density_by_source[source_name]:
            b_total = sum(diag_density_by_source[source_name][bucket].values())
            for diag in DIAG_TYPES:
                fn_diag_rows.append(
                    {
                        "source_name": source_name,
                        "stat_type": "density",
                        "diag_type": diag,
                        "bucket": bucket,
                        "count": diag_density_by_source[source_name][bucket][diag],
                        "ratio_within_group": f"{safe_ratio(diag_density_by_source[source_name][bucket][diag], b_total):.6f}",
                    }
                )
        if int(args.enable_highlight) == 1:
            for bucket in diag_highlight_by_source[source_name]:
                b_total = sum(diag_highlight_by_source[source_name][bucket].values())
                for diag in DIAG_TYPES:
                    fn_diag_rows.append(
                        {
                            "source_name": source_name,
                            "stat_type": "highlight",
                            "diag_type": diag,
                            "bucket": bucket,
                            "count": diag_highlight_by_source[source_name][bucket][diag],
                            "ratio_within_group": f"{safe_ratio(diag_highlight_by_source[source_name][bucket][diag], b_total):.6f}",
                        }
                    )

        # image-level FN distributions (gt scale & density)
        for bucket in image_fn_gt_scale[source_name]:
            total_scale = sum(image_fn_gt_scale[source_name].values())
            fn_diag_rows.append(
                {
                    "source_name": source_name,
                    "stat_type": "image_fn_gt_scale",
                    "diag_type": "all",
                    "bucket": bucket,
                    "count": image_fn_gt_scale[source_name][bucket],
                    "ratio_within_group": f"{safe_ratio(image_fn_gt_scale[source_name][bucket], total_scale):.6f}",
                }
            )
        for bucket in image_fn_density[source_name]:
            total_dense = sum(image_fn_density[source_name].values())
            fn_diag_rows.append(
                {
                    "source_name": source_name,
                    "stat_type": "image_fn_gt_density",
                    "diag_type": "all",
                    "bucket": bucket,
                    "count": image_fn_density[source_name][bucket],
                    "ratio_within_group": f"{safe_ratio(image_fn_density[source_name][bucket], total_dense):.6f}",
                }
            )

    write_csv(
        report_dir / "fn_diag_summary.csv",
        fn_diag_rows,
        ["source_name", "stat_type", "diag_type", "bucket", "count", "ratio_within_group"],
    )

    # write README
    readme_lines: List[str] = []
    readme_lines.append("# P2.3.3 目标级/图像级 FN 机制拆解")
    readme_lines.append("")
    readme_lines.append("口径：conf -> NMS -> max_det；一对一匹配使用匈牙利算法（最大化 IoU 总和），仅 IoU>=tp_iou 参与有效匹配。")
    readme_lines.append("输出文件：config.json, image_level.csv, image_fn_cases.csv, fn_cases.csv, fn_diag_summary.csv, README.md。")
    readme_lines.append("FN 四类：no_response / low_score / regression_poor / postproc_suppressed（suppressed_stage=conf/NMS/max_det/none）。")
    readme_lines.append("fn_diag_summary.csv 的 stat_type 包含 overall/scale/density/highlight 以及 image_fn_gt_scale/image_fn_gt_density。")
    if int(args.enable_highlight) != 1:
        readme_lines.append("本轮未纳入高光交叉统计（enable_highlight=0）。")
    if use_export:
        readme_lines.append("注意：infer_export.csv 若已过 conf 过滤，low_score/no_response 可能偏小。")
    readme_lines.append("")
    readme_lines.append("运行示例：")
    readme_lines.append(
        "python /home/ubuntu/project/deduibi/yolo/analyze/code/p23_3_fn_mechanism.py \\\n"
        "  --weights /home/ubuntu/project/deduibi/yolo/models/defect/exp_260202_base/best/best.pt \\\n"
        "  --image_dir /home/ubuntu/project/deduibi/yolo/dataset/yolo/datasetm6c/images/val \\\n"
        "  --image_dir /home/ubuntu/project/deduibi/yolo/dataset/yolo/datasetm6c/images/test \\\n"
        "  --out_root /home/ubuntu/project/deduibi/yolo/analyze/result \\\n"
        "  --conf 0.3 --tp_iou 0.2 --nms_iou 0.6 --max_det 20 \\\n"
        "  --batch 4 --infer_chunk 16"
    )
    with (report_dir / "README.md").open("w", encoding="utf-8") as f:
        f.write("\n".join(readme_lines) + "\n")

    if warnings:
        print("[WARN] 统计提醒：")
        for w in warnings:
            print(w)

    print(f"[DONE] report_dir: {report_dir}")


if __name__ == "__main__":
    main()
