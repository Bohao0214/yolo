"""P2.3.2b 目标级误报（FP）类型化归因与结构交叉（YOLO）。

P2.3.0 评估口径冻结：
- 置信度阈值（conf）过滤 -> NMS（非极大值抑制，重叠框去重）-> max_det
- 匹配阈值 tp_iou（IoU=交并比），目标级一对一匹配

术语（首次出现给出中文解释）：
- GT = ground_truth（标注框）
- FP/TP/FN = 误报/命中/漏检
- IoU = 交并比
- NMS = 非极大值抑制（重叠框去重）

类型规则优先级（先命中高优先级不再下探）：
1) highlight（高光/强反光）
2) edge（边缘/白底）
3) texture_boundary（纹理过渡/阴影边界）
4) particle（颗粒/小噪点，可选，默认不实现）
5) other

模式：
A) --report_dir：复用 P2.3.2a 的 FP 样本明细（p2_3_2a_fp_samples.csv）

python /home/ubuntu/project/deduibi/yolo/analyze/code/p2_3_2b_fp_type.py \
  --report_dir /home/ubuntu/project/deduibi/yolo/analyze/result/report_2602022226 \
  --out_root /home/ubuntu/project/deduibi/yolo/analyze/result

B) --weights + --image_dir：从头推理并统计

python /home/ubuntu/project/deduibi/yolo/analyze/code/p2_3_2b_fp_type.py \
  --weights /home/ubuntu/project/deduibi/yolo/models/defect/exp_260202_base/best/best.pt \
  --image_dir /home/ubuntu/project/deduibi/yolo/dataset/yolo/datasetm6c/images/val \
  --image_dir /home/ubuntu/project/deduibi/yolo/dataset/yolo/datasetm6c/images/test \
  --out_root /home/ubuntu/project/deduibi/yolo/analyze/result \
  --conf 0.3 --tp_iou 0.2 --nms_iou 0.6 --max_det 100 \
  --batch 4 --infer_chunk 16

"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import gc
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

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
TYPE_NAMES = ["highlight", "edge", "texture_boundary", "particle", "other"]
FP_TAGS = ["unmatched", "pred_dup", "both"]


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
    p = argparse.ArgumentParser(description="P2.3.2b FP 类型化归因与结构交叉。")
    p.add_argument("--report_dir", type=str, default="", help="Reuse report_dir from P2.3.2a (preferred).")
    p.add_argument("--weights", type=str, default="")
    p.add_argument(
        "--image_dir",
        type=str,
        action="append",
        default=[],
        help="Dataset image directory (val/test). Can be provided multiple times or comma-separated.",
    )
    p.add_argument("--out_root", type=str, required=True)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--infer_chunk", type=int, default=16)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", type=str, default="")

    # P2.3.0 postprocess
    p.add_argument("--conf", type=float, default=0.3)
    p.add_argument("--tp_iou", type=float, default=0.2)
    p.add_argument("--nms_iou", type=float, default=0.6)
    p.add_argument("--max_det", type=int, default=20)

    # type rules
    p.add_argument("--white_thresh", type=int, default=250)
    p.add_argument("--edge_white_frac", type=float, default=0.33)
    p.add_argument("--hl_bright_percentile", type=float, default=95.0)
    p.add_argument("--hl_grad_percentile", type=float, default=90.0)
    p.add_argument("--highlight_frac", type=float, default=0.05)
    p.add_argument("--texture_frac", type=float, default=0.10)
    p.add_argument("--enable_particle", type=int, default=0, help="Enable particle rule (experimental).")
    p.add_argument("--particle_area_max", type=float, default=256.0, help="Max box area in pixels for particle rule.")
    p.add_argument("--particle_texture_frac", type=float, default=0.20)
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


def safe_ratio(num: int, den: int) -> float:
    return float(num) / float(den) if den > 0 else 0.0


def write_csv(path: Path, rows: List[dict], headers: List[str], comment: Optional[str] = None) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        if comment:
            f.write(comment.rstrip() + "\n")
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


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


def build_masks(
    img_bgr: np.ndarray,
    white_thresh: int,
    hl_bright_percentile: float,
    hl_grad_percentile: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
    bright_thresh = float(np.percentile(gray_vals, hl_bright_percentile))
    grad_thresh = float(np.percentile(grad_vals, hl_grad_percentile))
    bright_mask = gray >= bright_thresh
    grad_mask = grad >= grad_thresh
    if np.any(valid_mask):
        bright_mask = bright_mask & valid_mask
        grad_mask = grad_mask & valid_mask
    highlight_mask = bright_mask & grad_mask
    texture_mask = (~bright_mask) & grad_mask & valid_mask
    return white_mask.astype(np.uint8), highlight_mask.astype(np.uint8), texture_mask.astype(np.uint8), grad_mask.astype(np.uint8)


def classify_fp_type(
    box: Tuple[float, float, float, float],
    w: int,
    h: int,
    white_integral: np.ndarray,
    highlight_integral: np.ndarray,
    texture_integral: np.ndarray,
    args: argparse.Namespace,
) -> str:
    highlight_ratio = mask_ratio_for_box(highlight_integral, box, w, h)
    if 0.0 < highlight_ratio <= float(args.highlight_frac):
        return "highlight"

    white_ratio = mask_ratio_for_box(white_integral, box, w, h)
    if white_ratio >= float(args.edge_white_frac):
        return "edge"

    texture_ratio = mask_ratio_for_box(texture_integral, box, w, h)
    if texture_ratio >= float(args.texture_frac):
        return "texture_boundary"

    if int(args.enable_particle) == 1:
        x1, y1, x2, y2 = box
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        if area <= float(args.particle_area_max) and texture_ratio >= float(args.particle_texture_frac):
            return "particle"

    return "other"


def parse_fp_samples(sample_path: Path) -> List[dict]:
    rows: List[dict] = []
    with sample_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def parse_pred_box(raw: str) -> Tuple[float, float, float, float]:
    parts = [p.strip() for p in raw.split(",")]
    if len(parts) != 4:
        raise ValueError(f"Invalid pred_box: {raw}")
    return (float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]))


def build_rules_comment(args: argparse.Namespace) -> str:
    rule = {
        "priority": TYPE_NAMES,
        "highlight": {
            "hl_bright_percentile": float(args.hl_bright_percentile),
            "hl_grad_percentile": float(args.hl_grad_percentile),
            "highlight_frac": float(args.highlight_frac),
        },
        "edge": {"white_thresh": int(args.white_thresh), "edge_white_frac": float(args.edge_white_frac)},
        "texture_boundary": {"texture_frac": float(args.texture_frac)},
        "particle": {
            "enabled": bool(int(args.enable_particle)),
            "particle_area_max": float(args.particle_area_max),
            "particle_texture_frac": float(args.particle_texture_frac),
        },
        "note": "particle 默认不实现，未命中规则归入 other。",
    }
    return "# rules: " + json.dumps(rule, ensure_ascii=False)


def init_type_stats() -> Dict[str, int]:
    stats = {"fp_total": 0}
    for t in TYPE_NAMES:
        stats[f"type_{t}"] = 0
    return stats


def init_cross_stats() -> Dict[str, Dict[str, int]]:
    return {tag: {t: 0 for t in TYPE_NAMES} for tag in FP_TAGS}


def update_stats(stats: Dict[str, int], fp_type: str) -> None:
    stats["fp_total"] += 1
    stats[f"type_{fp_type}"] += 1


def update_cross(cross: Dict[str, Dict[str, int]], fp_tag: str, fp_type: str) -> None:
    if fp_tag not in cross:
        return
    cross[fp_tag][fp_type] += 1


def process_fp_boxes(
    image_path: Path,
    boxes: List[Tuple[float, float, float, float]],
    tags: List[str],
    source_name: str,
    label_path: Optional[Path],
    warnings: List[str],
    stats_all: Dict[str, int],
    stats_by_source: Dict[str, Dict[str, int]],
    cross_all: Dict[str, Dict[str, int]],
    cross_by_source: Dict[str, Dict[str, Dict[str, int]]],
    args: argparse.Namespace,
) -> None:
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")
    h, w = img.shape[:2]
    if label_path is not None:
        if not label_path.exists():
            warnings.append(f"- [提醒] 缺失标签文件: {label_path}")
        else:
            _ = load_labels(label_path, w, h)
    white_mask, highlight_mask, texture_mask, _ = build_masks(
        img,
        int(args.white_thresh),
        float(args.hl_bright_percentile),
        float(args.hl_grad_percentile),
    )
    white_integral = compute_integral(white_mask)
    hl_integral = compute_integral(highlight_mask)
    texture_integral = compute_integral(texture_mask)

    for box, fp_tag in zip(boxes, tags):
        fp_type = classify_fp_type(
            box,
            w,
            h,
            white_integral,
            hl_integral,
            texture_integral,
            args,
        )
        update_stats(stats_all, fp_type)
        update_stats(stats_by_source[source_name], fp_type)
        update_cross(cross_all, fp_tag, fp_type)
        update_cross(cross_by_source[source_name], fp_tag, fp_type)

    del img, white_mask, highlight_mask, texture_mask, white_integral, hl_integral, texture_integral


def run_with_report_dir(args: argparse.Namespace) -> Tuple[Path, List[str], Dict[str, int], Dict[str, Dict[str, int]], Dict[str, Dict[str, int]], Dict[str, Dict[str, Dict[str, int]]], List[str]]:
    report_dir = Path(args.report_dir)
    if not report_dir.exists():
        raise FileNotFoundError(f"report_dir not found: {report_dir}")

    run_args_path = report_dir / "run_args.json"
    if not run_args_path.exists():
        raise FileNotFoundError("report_dir 缺少 run_args.json，无法定位 image_dir。")

    with run_args_path.open("r", encoding="utf-8") as f:
        old_args = json.load(f)

    image_dirs = [Path(p) for p in old_args.get("image_dir", [])]
    if not image_dirs:
        raise ValueError("run_args.json 未包含 image_dir，无法定位图像。")

    label_dirs = [infer_label_dir(d) for d in image_dirs]

    sample_path = report_dir / "p2_3_2a_fp_samples.csv"
    if not sample_path.exists():
        raise FileNotFoundError(
            "report_dir 缺少 p2_3_2a_fp_samples.csv，请先用 2.3.2a 生成 FP 样本明细（可设置 --sample_max）。"
        )

    rows = parse_fp_samples(sample_path)
    if not rows:
        raise RuntimeError("p2_3_2a_fp_samples.csv 为空，无法进行类型统计。")

    sample_counts: Dict[str, int] = {}
    for r in rows:
        src = r.get("source_name", "")
        if not src:
            continue
        sample_counts[src] = sample_counts.get(src, 0) + 1

    source_to_dir: Dict[str, Path] = {}
    source_to_label: Dict[str, Path] = {}
    source_order: List[str] = []
    for p, lbl in zip(image_dirs, label_dirs):
        key = p.name
        if key in source_to_dir:
            print(f"[WARN] source_name 重复: {key}，将使用第一个路径。")
            continue
        source_to_dir[key] = p
        source_to_label[key] = lbl
        source_order.append(key)

    needed_stems_by_source: Dict[str, set] = {}
    for r in rows:
        src = r.get("source_name", "")
        image_id = r.get("image_id", "")
        if not src or not image_id:
            continue
        needed_stems_by_source.setdefault(src, set()).add(image_id)

    stem_path_map: Dict[Tuple[str, str], Path] = {}
    for source_name, stems in needed_stems_by_source.items():
        img_dir = source_to_dir.get(source_name)
        if img_dir is None:
            print(f"[WARN] source_name 无对应 image_dir: {source_name}")
            continue
        for img_path in img_dir.rglob("*"):
            if not img_path.is_file() or img_path.suffix.lower() not in IMG_EXTS:
                continue
            if img_path.stem in stems and (source_name, img_path.stem) not in stem_path_map:
                stem_path_map[(source_name, img_path.stem)] = img_path

    stats_all = init_type_stats()
    stats_by_source: Dict[str, Dict[str, int]] = {name: init_type_stats() for name in source_order}
    cross_all = init_cross_stats()
    cross_by_source: Dict[str, Dict[str, Dict[str, int]]] = {name: init_cross_stats() for name in source_order}
    warnings: List[str] = []
    for p, lbl in zip(image_dirs, label_dirs):
        if not p.exists():
            warnings.append(f"- [提醒] image_dir 不存在: {p}")
        if not lbl.exists():
            warnings.append(f"- [提醒] label_dir 不存在: {lbl}")

    summary_path = report_dir / "p2_3_2a_fp_split_summary.csv"
    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                src = row.get("source_name", "")
                if not src or src == "all":
                    continue
                try:
                    fp_total = int(float(row.get("FP_total", "0")))
                except ValueError:
                    fp_total = 0
                if fp_total > 0:
                    sample_cnt = sample_counts.get(src, 0)
                    if sample_cnt < fp_total:
                        warnings.append(
                            f"- [提醒] {src} 的 FP 样本数={sample_cnt} < FP_total={fp_total}，统计可能偏小。"
                        )

    grouped: Dict[Path, Tuple[List[Tuple[float, float, float, float]], List[str], str]] = {}
    for r in rows:
        src = r.get("source_name", "")
        image_id = r.get("image_id", "")
        pred_box = r.get("pred_box", "")
        fp_tag = r.get("fp_tag", "")
        if fp_tag not in FP_TAGS:
            fp_tag = "unmatched"
        if not src or not image_id or not pred_box:
            continue
        img_path = stem_path_map.get((src, image_id))
        if img_path is None:
            warnings.append(f"- [提醒] 未找到图像: source={src}, image_id={image_id}")
            continue
        if src not in stats_by_source:
            stats_by_source[src] = init_type_stats()
            cross_by_source[src] = init_cross_stats()
            source_order.append(src)
        box = parse_pred_box(pred_box)
        if img_path not in grouped:
            grouped[img_path] = ([], [], src)
        grouped[img_path][0].append(box)
        grouped[img_path][1].append(fp_tag)

    ensure_cv2()
    for img_path, (boxes, tags, src) in grouped.items():
        label_dir = source_to_label.get(src)
        label_path = label_dir / f"{img_path.stem}.txt" if label_dir is not None else None
        process_fp_boxes(
            img_path,
            boxes,
            tags,
            src,
            label_path,
            warnings,
            stats_all,
            stats_by_source,
            cross_all,
            cross_by_source,
            args,
        )

    return report_dir, source_order, stats_all, stats_by_source, cross_all, cross_by_source, warnings


def run_from_scratch(args: argparse.Namespace) -> Tuple[Path, List[str], Dict[str, int], Dict[str, Dict[str, int]], Dict[str, Dict[str, int]], Dict[str, Dict[str, Dict[str, int]]], List[str]]:
    ensure_ultralytics()
    ensure_cv2()
    if not args.weights:
        raise ValueError("未提供 --weights")
    image_dirs = normalize_path_list(args.image_dir)
    if not image_dirs:
        raise ValueError("未提供 --image_dir")
    for d in image_dirs:
        if not d.exists():
            raise FileNotFoundError(f"image_dir not found: {d}")

    report_dir = make_report_dir(Path(args.out_root))
    label_dirs = [infer_label_dir(d) for d in image_dirs]

    run_args = {
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "weights": str(args.weights),
        "image_dir": [str(p) for p in image_dirs],
        "label_dir": [str(p) for p in label_dirs],
        "out_root": str(args.out_root),
        "report_dir": str(report_dir),
        "conf": float(args.conf),
        "nms_iou": float(args.nms_iou),
        "max_det": int(args.max_det),
        "tp_iou": float(args.tp_iou),
        "batch": int(args.batch),
        "infer_chunk": int(args.infer_chunk),
        "imgsz": int(args.imgsz),
        "device": str(args.device),
        "eval_pipeline": "conf -> NMS -> max_det; one-to-one match @ tp_iou",
        "type_rules": {
            "priority": TYPE_NAMES,
            "highlight": {
                "hl_bright_percentile": float(args.hl_bright_percentile),
                "hl_grad_percentile": float(args.hl_grad_percentile),
                "highlight_frac": float(args.highlight_frac),
            },
            "edge": {"white_thresh": int(args.white_thresh), "edge_white_frac": float(args.edge_white_frac)},
            "texture_boundary": {"texture_frac": float(args.texture_frac)},
            "particle": {
                "enabled": bool(int(args.enable_particle)),
                "particle_area_max": float(args.particle_area_max),
                "particle_texture_frac": float(args.particle_texture_frac),
            },
            "note": "particle 默认不实现，未命中规则归入 other。",
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

    stats_all = init_type_stats()
    stats_by_source: Dict[str, Dict[str, int]] = {}
    cross_all = init_cross_stats()
    cross_by_source: Dict[str, Dict[str, Dict[str, int]]] = {}
    source_order: List[str] = []
    warnings: List[str] = []

    model = YOLO(str(args.weights))

    try:
        import torch
    except Exception:
        torch = None  # type: ignore

    for img_dir, lbl_dir in zip(image_dirs, label_dirs):
        source_name = img_dir.name
        if source_name not in stats_by_source:
            stats_by_source[source_name] = init_type_stats()
            cross_by_source[source_name] = init_cross_stats()
            source_order.append(source_name)
        img_paths = list_images(img_dir)
        if not img_paths:
            warnings.append(f"- [提醒] image_dir 为空: {img_dir}")
            continue
        for start in range(0, len(img_paths), int(args.infer_chunk)):
            chunk = img_paths[start : start + int(args.infer_chunk)]
            sources = [str(p) for p in chunk]
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
            result_map = {str(res.path): res for res in results}

            for img_path in chunk:
                res = result_map.get(str(img_path))
                if res is None:
                    warnings.append(f"- [提醒] 推理结果缺失: {img_path}")
                    continue
                if getattr(res, "orig_shape", None):
                    h, w = int(res.orig_shape[0]), int(res.orig_shape[1])
                else:
                    img_tmp = cv2.imread(str(img_path))
                    if img_tmp is None:
                        raise FileNotFoundError(f"Failed to read image: {img_path}")
                    h, w = img_tmp.shape[:2]
                    del img_tmp

                label_path = lbl_dir / f"{img_path.stem}.txt"
                gt_boxes = load_labels(label_path, w, h)

                boxes = res.boxes
                preds_xyxy: List[Tuple[float, float, float, float]] = []
                scores: List[float] = []
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
                        preds_xyxy.append((float(box[0]), float(box[1]), float(box[2]), float(box[3])))
                        scores.append(float(score))

                gt_arr = np.array(gt_boxes, dtype=np.float32) if gt_boxes else np.zeros((0, 4), dtype=np.float32)
                pred_arr = np.array(preds_xyxy, dtype=np.float32) if preds_xyxy else np.zeros((0, 4), dtype=np.float32)
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

                pred_dup: set[int] = set()
                if iou_mat.size > 0:
                    for gi in range(iou_mat.shape[0]):
                        cand = np.where(iou_mat[gi] >= float(args.tp_iou))[0].tolist()
                        if len(cand) < 2:
                            continue
                        if gi in assigned_pred_for_gt:
                            keep_pi = assigned_pred_for_gt[gi]
                        else:
                            keep_pi = max(
                                cand,
                                key=lambda pi: (float(iou_mat[gi, pi]), float(scores[pi] if pi < len(scores) else 0.0)),
                            )
                        for pi in cand:
                            if pi != keep_pi:
                                pred_dup.add(pi)

                fp_indices = [pi for pi in range(len(preds_xyxy)) if pi not in assigned_gt_for_pred]
                if not fp_indices:
                    del gt_arr, pred_arr, iou_mat
                    continue

                # prepare masks once per image (仅在存在 FP 时计算)
                img = cv2.imread(str(img_path))
                if img is None:
                    raise FileNotFoundError(f"Failed to read image: {img_path}")
                white_mask, highlight_mask, texture_mask, _ = build_masks(
                    img,
                    int(args.white_thresh),
                    float(args.hl_bright_percentile),
                    float(args.hl_grad_percentile),
                )
                white_integral = compute_integral(white_mask)
                hl_integral = compute_integral(highlight_mask)
                texture_integral = compute_integral(texture_mask)

                for pi in fp_indices:
                    pred = preds_xyxy[pi]
                    max_iou = float(iou_mat[:, pi].max()) if iou_mat.size > 0 else 0.0
                    is_unmatched = max_iou < float(args.tp_iou)
                    is_pred_dup = pi in pred_dup
                    if is_unmatched and is_pred_dup:
                        tag = "both"
                    elif is_unmatched:
                        tag = "unmatched"
                    elif is_pred_dup:
                        tag = "pred_dup"
                    else:
                        tag = "unmatched"

                    fp_type = classify_fp_type(
                        pred,
                        w,
                        h,
                        white_integral,
                        hl_integral,
                        texture_integral,
                        args,
                    )
                    update_stats(stats_all, fp_type)
                    update_stats(stats_by_source[source_name], fp_type)
                    update_cross(cross_all, tag, fp_type)
                    update_cross(cross_by_source[source_name], tag, fp_type)

                del img, white_mask, highlight_mask, texture_mask, white_integral, hl_integral, texture_integral
                del gt_arr, pred_arr, iou_mat

            del result_map, results, sources, chunk
            gc.collect()
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()

        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()

    return report_dir, source_order, stats_all, stats_by_source, cross_all, cross_by_source, warnings


def build_type_table_rows(stats_all: Dict[str, int], stats_by_source: Dict[str, Dict[str, int]], source_order: List[str]) -> List[dict]:
    rows: List[dict] = []
    for source in ["all"] + source_order:
        stats = stats_all if source == "all" else stats_by_source[source]
        fp_total = stats["fp_total"]
        row = {"source_name": source, "fp_total": fp_total}
        for t in TYPE_NAMES:
            cnt = stats[f"type_{t}"]
            row[f"type_{t}"] = cnt
            row[f"ratio_{t}"] = f"{safe_ratio(cnt, fp_total):.6f}"
        rows.append(row)
    return rows


def build_cross_rows(
    cross_all: Dict[str, Dict[str, int]],
    cross_by_source: Dict[str, Dict[str, Dict[str, int]]],
    source_order: List[str],
) -> List[dict]:
    rows: List[dict] = []
    for source in ["all"] + source_order:
        cross = cross_all if source == "all" else cross_by_source[source]
        for tag in FP_TAGS:
            total = sum(cross[tag].values())
            for t in TYPE_NAMES:
                cnt = cross[tag][t]
                rows.append(
                    {
                        "source_name": source,
                        "fp_tag": tag,
                        "type_name": t,
                        "count": cnt,
                        "ratio_within_tag": f"{safe_ratio(cnt, total):.6f}",
                    }
                )
    return rows


def write_notes(
    path: Path,
    report_dir: Path,
    stats_all: Dict[str, int],
    stats_by_source: Dict[str, Dict[str, int]],
    source_order: List[str],
    warnings: Iterable[str],
    args: argparse.Namespace,
    report_dir_mode: bool,
) -> None:
    lines: List[str] = []
    lines.append("# P2.3.2b 目标级误报（FP）类型化归因")
    lines.append("")
    lines.append(f"- created_at: {dt.datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"- report_dir: {report_dir}")
    lines.append("")
    lines.append("## 结果解读")
    lines.append("本报告在 P2.3.0 口径下统计 FP，并按类型规则进行归因；类型优先级：highlight > edge > texture_boundary > particle > other。")
    lines.append("type_* 为 FP 类型计数，ratio_* 为其占 FP 总量的比例；交叉表提供 unmatched/pred_dup/both 内的类型分布。")
    lines.append("unmatched 表示预测框与任意 GT 的最大 IoU < tp_iou；pred_dup 表示同一 GT 被多个预测框重复覆盖（仅 1 个可为 TP）。")
    if int(args.enable_particle) != 1:
        lines.append("particle 规则默认未启用，未命中规则的样本归入 other。")
    if report_dir_mode:
        lines.append("本次运行基于 2.3.2a 的 FP 样本明细进行类型统计，若样本非全量，统计可能存在偏差。")
    if warnings:
        lines.append("")
        lines.append("## 提醒")
        for w in warnings:
            lines.append(str(w))
    with path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    if int(args.batch) >= 8:
        raise ValueError("--batch 必须 < 8")
    if int(args.infer_chunk) > 64:
        raise ValueError("--infer_chunk 不能太高（建议 <= 64）")

    report_dir_mode = bool(args.report_dir)
    if report_dir_mode:
        report_dir, source_order, stats_all, stats_by_source, cross_all, cross_by_source, warnings = run_with_report_dir(args)
    else:
        report_dir, source_order, stats_all, stats_by_source, cross_all, cross_by_source, warnings = run_from_scratch(args)

    rules_comment = build_rules_comment(args)

    type_table_rows = build_type_table_rows(stats_all, stats_by_source, source_order)
    type_table_path = report_dir / "p2_3_2b_fp_type_table.csv"
    write_csv(
        type_table_path,
        type_table_rows,
        ["source_name", "fp_total"]
        + [f"type_{t}" for t in TYPE_NAMES]
        + [f"ratio_{t}" for t in TYPE_NAMES],
        comment=rules_comment,
    )

    cross_rows = build_cross_rows(cross_all, cross_by_source, source_order)
    cross_path = report_dir / "p2_3_2b_fp_type_cross.csv"
    write_csv(
        cross_path,
        cross_rows,
        ["source_name", "fp_tag", "type_name", "count", "ratio_within_tag"],
        comment=rules_comment,
    )

    notes_path = report_dir / "p2_3_2b_fp_type_notes.md"
    write_notes(notes_path, report_dir, stats_all, stats_by_source, source_order, warnings, args, report_dir_mode)

    if warnings:
        print("[WARN] 统计提醒：")
        for w in warnings:
            print(w)

    print(f"[DONE] report_dir: {report_dir}")
    print(f"[DONE] type_table: {type_table_path}")
    print(f"[DONE] cross_table: {cross_path}")
    print(f"[DONE] notes: {notes_path}")


if __name__ == "__main__":
    main()
