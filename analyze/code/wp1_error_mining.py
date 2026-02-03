"""WP1 error evidence closed-loop analysis for YOLO defect detection.

Outputs (per run):
- report_YYMMDDhhmm/01_scale_bucket_metrics.md
- report_YYMMDDhhmm/02_edge_impact.md
- report_YYMMDDhhmm/03_highlight_impact.md
- report_YYMMDDhhmm/04_interaction_scale_x_edge.md
- report_YYMMDDhhmm/05_interaction_scale_x_highlight.md
- report_YYMMDDhhmm/summary_metrics.csv
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
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
BUCKETS: Sequence[Tuple[str, float, float]] = (
    ("<8", 0.0, 8.0),
    ("8-16", 8.0, 16.0),
    ("16-32", 16.0, 32.0),
    ("32-64", 32.0, 64.0),
    (">64", 64.0, math.inf),
)
BUCKET_NAMES = [b[0] for b in BUCKETS]

EXAMPLE_CMD = (
    "python /home/ubuntu/project/deduibi/yolo/analyze/code/wp1_error_mining.py \\\n"
    "  --weights /home/ubuntu/project/deduibi/yolo/models/defect/exp_2601292207/best/best.pt \\\n"
    "  --image_dir /home/ubuntu/project/deduibi/yolo/dataset/yolo/datasetm6c/images/val \\\n"
    "  --out_root /home/ubuntu/project/deduibi/yolo/analyze/result \\\n"
    "  --batch 4 --infer_chunk 16"
)


@dataclass(frozen=True)
class BoxRecord:
    image_id: str
    img_path: str
    index: int
    cls: int
    bbox_xyxy: Tuple[float, float, float, float]
    score: Optional[float] = None


def ensure_ultralytics() -> None:
    if YOLO is None:
        raise ImportError(
            "Failed to import ultralytics. Please install ultralytics in the environment. "
            f"Original error: {ULTRALYTICS_IMPORT_ERROR}"
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="WP1 error evidence closed-loop analysis.")
    p.add_argument(
        "--weights",
        type=str,
        default="/home/ubuntu/project/deduibi/yolo/models/defect/exp_2601292207/best/best.pt",
    )
    p.add_argument(
        "--image_dir",
        type=str,
        required=True,
        action="append",
        help="Dataset image directory (val or test). Can be provided multiple times or comma-separated.",
    )
    p.add_argument(
        "--out_root",
        type=str,
        default="/home/ubuntu/project/deduibi/yolo/analyze/result",
    )
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--infer_chunk", type=int, default=16)

    # inference postprocess (P2.3.0)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--nms_iou", type=float, default=0.70)
    p.add_argument("--max_det", type=int, default=20)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", type=str, default="")

    # evaluation
    p.add_argument("--tp_iou", type=float, default=0.50)

    # edge / highlight rules
    p.add_argument("--edge_white_frac", type=float, default=0.33)
    p.add_argument("--hl_bright_percentile", type=float, default=95.0)
    p.add_argument("--hl_grad_percentile", type=float, default=90.0)
    p.add_argument("--hl_frac", type=float, default=0.05)

    return p.parse_args()


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
        for name in ("label", "labels", "lable"):
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


def normalize_image_dirs(raw: Sequence[str]) -> List[Path]:
    dirs: List[Path] = []
    for item in raw:
        if not item:
            continue
        for part in item.split(","):
            p = part.strip()
            if not p:
                continue
            dirs.append(Path(p))
    seen = set()
    uniq: List[Path] = []
    for d in dirs:
        key = str(d)
        if key in seen:
            continue
        uniq.append(d)
        seen.add(key)
    return uniq


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


def run_inference(
    weights: Path,
    images: Sequence[Path],
    imgsz: int,
    conf: float,
    nms_iou: float,
    max_det: int,
    batch: int,
    device: str,
    infer_chunk: int,
) -> Dict[str, List[BoxRecord]]:
    ensure_ultralytics()
    model = YOLO(str(weights))
    preds: Dict[str, List[BoxRecord]] = {}
    if infer_chunk <= 0:
        infer_chunk = len(images)

    try:
        import torch
    except Exception:
        torch = None  # type: ignore

    for start in range(0, len(images), infer_chunk):
        chunk = images[start : start + infer_chunk]
        results = model.predict(
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
    return preds


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


def _clip_box(box: Tuple[float, float, float, float], w: int, h: int) -> Optional[Tuple[int, int, int, int]]:
    x1, y1, x2, y2 = box
    x1i = int(math.floor(x1))
    y1i = int(math.floor(y1))
    x2i = int(math.ceil(x2))
    y2i = int(math.ceil(y2))
    x1i = max(0, min(w, x1i))
    x2i = max(0, min(w, x2i))
    y1i = max(0, min(h, y1i))
    y2i = max(0, min(h, y2i))
    if x2i <= x1i or y2i <= y1i:
        return None
    return x1i, y1i, x2i, y2i


def _box_sum(integral: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> float:
    return float(integral[y2, x2] - integral[y1, x2] - integral[y2, x1] + integral[y1, x1])


def mask_ratio_for_box(integral: np.ndarray, box: Tuple[float, float, float, float], w: int, h: int) -> float:
    clipped = _clip_box(box, w, h)
    if clipped is None:
        return 0.0
    x1, y1, x2, y2 = clipped
    area = max(1, (x2 - x1) * (y2 - y1))
    return _box_sum(integral, x1, y1, x2, y2) / float(area)


def compute_integral(mask: np.ndarray) -> np.ndarray:
    return cv2.integral(mask.astype(np.uint8))


def build_background_mask(white_mask: np.ndarray) -> np.ndarray:
    mask = white_mask.astype(np.uint8)
    kernel = np.ones((5, 5), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num <= 1:
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    if areas.size == 0:
        return mask
    largest_idx = 1 + int(np.argmax(areas))
    return (labels == largest_idx).astype(np.uint8)


def build_highlight_mask(
    img_bgr: np.ndarray, bright_p: float, grad_p: float, bg_mask: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, float, float]:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(gx, gy)
    valid = None
    if bg_mask is not None:
        valid = bg_mask == 0
    if valid is not None and np.any(valid):
        gray_vals = gray[valid]
        grad_vals = grad[valid]
    else:
        gray_vals = gray.reshape(-1)
        grad_vals = grad.reshape(-1)
    bright_thresh = float(np.percentile(gray_vals, bright_p))
    grad_thresh = float(np.percentile(grad_vals, grad_p))
    mask = (gray >= bright_thresh) & (grad >= grad_thresh)
    if valid is not None and np.any(valid):
        mask = mask & valid
    return mask.astype(np.uint8), bright_thresh, grad_thresh


def min_side(box: Tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = box
    return float(min(max(0.0, x2 - x1), max(0.0, y2 - y1)))


def box_to_int(box: Tuple[float, float, float, float], w: int, h: int) -> Optional[Tuple[int, int, int, int]]:
    return _clip_box(box, w, h)


def draw_overlay(
    img: np.ndarray,
    gts: Sequence[Tuple[float, float, float, float]],
    preds: Sequence[Tuple[Tuple[float, float, float, float], float]],
    highlight_gt: Optional[Tuple[float, float, float, float]] = None,
    highlight_pred: Optional[Tuple[float, float, float, float]] = None,
    title: str = "",
) -> np.ndarray:
    out = img.copy()
    h, w = out.shape[:2]

    for box in gts:
        clipped = box_to_int(box, w, h)
        if clipped is None:
            continue
        x1, y1, x2, y2 = clipped
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 200, 0), 2)

    for box, score in preds:
        clipped = box_to_int(box, w, h)
        if clipped is None:
            continue
        x1, y1, x2, y2 = clipped
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 220), 2)
        cv2.putText(
            out,
            f"{score:.2f}",
            (x1, max(12, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 220),
            1,
        )

    if highlight_gt is not None:
        clipped = box_to_int(highlight_gt, w, h)
        if clipped is not None:
            x1, y1, x2, y2 = clipped
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 255), 3)

    if highlight_pred is not None:
        clipped = box_to_int(highlight_pred, w, h)
        if clipped is not None:
            x1, y1, x2, y2 = clipped
            cv2.rectangle(out, (x1, y1), (x2, y2), (255, 200, 0), 3)

    if title:
        cv2.putText(out, title, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(out, title, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (10, 10, 10), 1)
    return out


def push_topk(items: List[dict], item: dict, key_fn, k: int = 20) -> None:
    items.append(item)
    if len(items) > k:
        items.sort(key=key_fn, reverse=True)
        del items[k:]


def save_vis_candidates(cands: Sequence[dict], out_dir: Path, prefix: str, limit: Optional[int] = 20) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if limit is not None:
        ordered = sorted(cands, key=lambda x: x.get("priority", 0), reverse=True)
    else:
        ordered = list(cands)
    take = ordered if limit is None else ordered[:limit]
    for idx, c in enumerate(take, 1):
        img = cv2.imread(c["img_path"])
        if img is None:
            continue
        overlay = draw_overlay(
            img,
            c.get("gts", []),
            c.get("preds", []),
            highlight_gt=c.get("highlight_gt"),
            highlight_pred=c.get("highlight_pred"),
            title=c.get("title", ""),
        )
        image_id = c.get("image_id", f"img{idx:02d}")
        out_path = out_dir / f"{prefix}_{idx:02d}_{image_id}.jpg"
        cv2.imwrite(str(out_path), overlay)


def safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b else 0.0


def init_bucket_stats() -> Dict[str, dict]:
    stats: Dict[str, dict] = {}
    for name in BUCKET_NAMES:
        stats[name] = {
            "n_gt": 0,
            "n_gt_tp": 0,
            "n_gt_fn": 0,
            "n_gt_dup": 0,
            "n_pred": 0,
            "n_pred_tp": 0,
            "n_pred_fp": 0,
            "n_pred_dup": 0,
            "tp_edge": 0,
            "fp_edge": 0,
            "fn_edge": 0,
            "tp_hl": 0,
            "fp_hl": 0,
            "fn_hl": 0,
            "tp": 0,
            "fp": 0,
            "fn": 0,
        }
    return stats


def init_bucket_image_stats() -> Dict[str, dict]:
    stats: Dict[str, dict] = {}
    for name in BUCKET_NAMES:
        stats[name] = {
            "pos_images": 0,
            "neg_images": 0,
            "tp_images": 0,
            "fn_images": 0,
            "fp_images": 0,
            "tn_images": 0,
            "fp_edge_images": 0,
            "fp_hl_images": 0,
        }
    return stats


def init_bucket_interaction() -> Dict[str, dict]:
    stats: Dict[str, dict] = {}
    for name in BUCKET_NAMES:
        stats[name] = {
            0: {"tp": 0, "fp": 0, "fn": 0},
            1: {"tp": 0, "fp": 0, "fn": 0},
        }
    return stats


def write_markdown(path: Path, title: str, lines: Sequence[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _image_dir_list(args: argparse.Namespace) -> List[str]:
    v = getattr(args, "image_dir", None)
    if v is None:
        return []
    if isinstance(v, (list, tuple)):
        return [str(x) for x in v]
    return [str(v)]


def format_args(args: argparse.Namespace) -> List[str]:
    rows: List[str] = []
    for k, v in sorted(vars(args).items()):
        if k == "image_dir":
            rows.append(f"- {k}: {', '.join(_image_dir_list(args))}")
        else:
            rows.append(f"- {k}: {v}")
    return rows


def format_thresholds(args: argparse.Namespace, hl_strategy: str) -> List[str]:
    return [
        f"- conf: {args.conf}",
        f"- nms_iou: {args.nms_iou}",
        f"- max_det: {args.max_det}",
        f"- tp_iou: {args.tp_iou}",
        f"- edge_white_frac: {args.edge_white_frac}",
        f"- highlight_strategy: {hl_strategy}",
        f"- highlight_frac: {args.hl_frac}",
    ]


def build_header_block(args: argparse.Namespace, report_dir: Path, created_at: str, hl_strategy: str) -> List[str]:
    image_dirs = _image_dir_list(args)
    image_dirs_joined = ", ".join(image_dirs) if image_dirs else ""
    cmd_parts = [
        "python",
        "/home/ubuntu/project/deduibi/yolo/analyze/code/wp1_error_mining.py",
        f"--weights {args.weights}",
    ]
    for d in image_dirs:
        cmd_parts.append(f"--image_dir {d}")
    cmd_parts.extend(
        [
            f"--out_root {args.out_root}",
            f"--batch {args.batch}",
            f"--infer_chunk {args.infer_chunk}",
            f"--conf {args.conf}",
            f"--nms_iou {args.nms_iou}",
            f"--max_det {args.max_det}",
            f"--imgsz {args.imgsz}",
            f"--tp_iou {args.tp_iou}",
        ]
    )
    lines = [
        f"# {report_dir.name}",
        "",
        f"- created_at: {created_at}",
        f"- weights: {args.weights}",
        f"- image_dir: {image_dirs_joined}",
        f"- out_root: {args.out_root}",
        "",
        "## CLI (requested example)",
        "```bash",
        EXAMPLE_CMD,
        "```",
        "",
        "## CLI (actual run)",
        "```bash",
        " ".join(cmd_parts),
        "```",
        "",
        "## Parameters",
        *format_args(args),
        "",
        "## Thresholds",
        *format_thresholds(args, hl_strategy),
        "",
    ]
    return lines


def analyze(args: argparse.Namespace) -> Path:
    ensure_ultralytics()

    weights = Path(args.weights)
    image_dirs = normalize_image_dirs(args.image_dir)
    if not image_dirs:
        raise ValueError("No image_dir provided after normalization.")
    out_root = Path(args.out_root)

    if not weights.exists():
        raise FileNotFoundError(f"Weights not found: {weights}")
    for d in image_dirs:
        if not d.exists():
            raise FileNotFoundError(f"Image dir not found: {d}")

    images: List[Path] = []
    label_dir_by_image: Dict[str, Path] = {}
    label_dirs: List[Path] = []
    for d in image_dirs:
        label_dir = infer_label_dir(d)
        if not label_dir.exists():
            tried = [str(p) for p in _label_dir_candidates(d)]
            raise FileNotFoundError(f"Label dir not found for {d}. Tried: {tried}")
        label_dirs.append(label_dir)
        for img in list_images(d):
            key = str(img)
            if key in label_dir_by_image:
                continue
            label_dir_by_image[key] = label_dir
            images.append(img)

    if not images:
        raise RuntimeError(f"No images found under: {', '.join(str(d) for d in image_dirs)}")

    args.image_dir = [str(d) for d in image_dirs]

    report_dir = make_report_dir(out_root)

    preds_by_img = run_inference(
        weights=weights,
        images=images,
        imgsz=int(args.imgsz),
        conf=float(args.conf),
        nms_iou=float(args.nms_iou),
        max_det=int(args.max_det),
        batch=int(args.batch),
        device=str(args.device),
        infer_chunk=int(args.infer_chunk),
    )

    fn_conf = 0.1
    preds_for_fn = preds_by_img
    if float(args.conf) > fn_conf:
        preds_for_fn = run_inference(
            weights=weights,
            images=images,
            imgsz=int(args.imgsz),
            conf=float(fn_conf),
            nms_iou=float(args.nms_iou),
            max_det=int(args.max_det),
            batch=int(args.batch),
            device=str(args.device),
            infer_chunk=int(args.infer_chunk),
        )

    image_fp_vis: List[dict] = []
    image_fn_vis: List[dict] = []
    obj_fp_vis: List[dict] = []
    obj_fn_vis: List[dict] = []

    bucket_stats = init_bucket_stats()
    bucket_image_stats = init_bucket_image_stats()
    bucket_edge_interaction = init_bucket_interaction()
    bucket_hl_interaction = init_bucket_interaction()

    total_tp = total_fp = total_fn = 0
    total_tp_edge = total_fp_edge = total_fn_edge = 0
    total_tp_hl = total_fp_hl = total_fn_hl = 0

    total_pred_dup = 0
    total_gt_dup = 0

    image_level = {"tp": 0, "fn": 0, "fp": 0, "tn": 0}

    created_at = dt.datetime.now().isoformat(timespec="seconds")
    hl_strategy = (
        f"exclude white background (255,255,255) via morph(open+close,k=5)+largest-cc; "
        f"per-image percentiles on non-bg: bright_p={args.hl_bright_percentile}, "
        f"grad_p={args.hl_grad_percentile}, mask=(gray>=p_bright)&(grad>=p_grad)"
    )

    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            raise FileNotFoundError(f"Failed to read image: {img_path}")
        h, w = img.shape[:2]
        image_id = img_path.stem

        label_dir = label_dir_by_image.get(str(img_path))
        if label_dir is None:
            raise RuntimeError(f"Missing label_dir mapping for image: {img_path}")
        label_path = label_dir / f"{image_id}.txt"
        gts = load_labels_for_image(label_path, w, h, image_id, img_path)
        preds = preds_by_img.get(str(img_path), [])

        # masks for edge/highlight
        white_mask = (
            (img[:, :, 0] == 255)
            & (img[:, :, 1] == 255)
            & (img[:, :, 2] == 255)
        )
        white_integral = compute_integral(white_mask)
        bg_mask = build_background_mask(white_mask)
        hl_mask, _bright_thresh, _grad_thresh = build_highlight_mask(
            img, float(args.hl_bright_percentile), float(args.hl_grad_percentile), bg_mask=bg_mask
        )
        hl_integral = compute_integral(hl_mask)

        gts_vis = [g.bbox_xyxy for g in gts]
        preds_vis = [(p.bbox_xyxy, float(p.score or 0.0)) for p in preds]

        preds_fn_all = preds_for_fn.get(str(img_path), [])
        preds_fn = [p for p in preds_fn_all if float(p.score or 0.0) >= fn_conf]
        pred_boxes_fn = np.array([p.bbox_xyxy for p in preds_fn], dtype=np.float32) if preds_fn else np.zeros((0, 4), dtype=np.float32)
        pred_scores_fn = np.array([float(p.score or 0.0) for p in preds_fn], dtype=np.float32) if preds_fn else np.zeros((0,), dtype=np.float32)

        gt_boxes = np.array([g.bbox_xyxy for g in gts], dtype=np.float32)
        pred_boxes = np.array([p.bbox_xyxy for p in preds], dtype=np.float32)

        if gt_boxes.size > 0 and pred_boxes_fn.size > 0:
            iou_fn = compute_iou_matrix(gt_boxes, pred_boxes_fn)
            best_fn_pred_idx = iou_fn.argmax(axis=1)
            best_fn_iou = iou_fn.max(axis=1)
        else:
            iou_fn = np.zeros((gt_boxes.shape[0], pred_boxes_fn.shape[0]), dtype=np.float32)
            best_fn_pred_idx = np.full((gt_boxes.shape[0],), -1, dtype=np.int32)
            best_fn_iou = np.zeros((gt_boxes.shape[0],), dtype=np.float32)

        # image-level stats (overall)
        img_has_gt = len(gts) > 0
        img_has_pred = len(preds) > 0
        if img_has_gt and img_has_pred:
            image_level["tp"] += 1
        elif img_has_gt and not img_has_pred:
            image_level["fn"] += 1
        elif (not img_has_gt) and img_has_pred:
            image_level["fp"] += 1
        else:
            image_level["tn"] += 1

        if (not img_has_gt) and img_has_pred:
            best_pred = max(preds, key=lambda p: float(p.score or 0.0))
            best_score = float(best_pred.score or 0.0)
            image_fp_vis.append(
                {
                    "priority": best_score,
                    "img_path": str(img_path),
                    "image_id": image_id,
                    "gts": gts_vis,
                    "preds": preds_vis,
                    "highlight_gt": None,
                    "highlight_pred": best_pred.bbox_xyxy,
                    "title": f"image_fp score={best_score:.2f}",
                }
            )

        if img_has_gt and (not img_has_pred):
            highlight_gt = gts[0].bbox_xyxy if gts else None
            highlight_pred = None
            title = "image_fn"
            if best_fn_pred_idx.size > 0:
                gi = int(np.argmax(best_fn_iou)) if best_fn_iou.size > 0 else -1
                if gi >= 0:
                    highlight_gt = gts[gi].bbox_xyxy if gi < len(gts) else highlight_gt
                    pi = int(best_fn_pred_idx[gi])
                    if pi >= 0 and pi < len(preds_fn):
                        highlight_pred = preds_fn[pi].bbox_xyxy
                        best_score = float(pred_scores_fn[pi])
                        title = f"image_fn bestIoU={float(best_fn_iou[gi]):.2f} score={best_score:.2f}"
            image_fn_vis.append(
                {
                    "priority": 0.0,
                    "img_path": str(img_path),
                    "image_id": image_id,
                    "gts": gts_vis,
                    "preds": preds_vis,
                    "highlight_gt": highlight_gt,
                    "highlight_pred": highlight_pred,
                    "title": title,
                }
            )

        # bucket presence per image (image-level definition uses empty/non-empty)
        gt_buckets = set()
        pred_buckets = set()
        pred_edge_buckets = set()
        pred_hl_buckets = set()

        for gt in gts:
            b = bucket_name_from_min_side(min_side(gt.bbox_xyxy))
            gt_buckets.add(b)

        for pred in preds:
            pb = bucket_name_from_min_side(min_side(pred.bbox_xyxy))
            pred_buckets.add(pb)
            pred_edge = mask_ratio_for_box(white_integral, pred.bbox_xyxy, w, h) >= float(args.edge_white_frac)
            pred_hl = mask_ratio_for_box(hl_integral, pred.bbox_xyxy, w, h) >= float(args.hl_frac)
            if pred_edge:
                pred_edge_buckets.add(pb)
            if pred_hl:
                pred_hl_buckets.add(pb)

        for b in BUCKET_NAMES:
            if b in gt_buckets:
                bucket_image_stats[b]["pos_images"] += 1
                if b in pred_buckets:
                    bucket_image_stats[b]["tp_images"] += 1
                else:
                    bucket_image_stats[b]["fn_images"] += 1
            else:
                bucket_image_stats[b]["neg_images"] += 1
                if b in pred_buckets:
                    bucket_image_stats[b]["fp_images"] += 1
                    if b in pred_edge_buckets:
                        bucket_image_stats[b]["fp_edge_images"] += 1
                    if b in pred_hl_buckets:
                        bucket_image_stats[b]["fp_hl_images"] += 1
                else:
                    bucket_image_stats[b]["tn_images"] += 1

        # object-level matching
        iou_mat = compute_iou_matrix(gt_boxes, pred_boxes)

        assigned_gt_for_pred: Dict[int, int] = {}
        assigned_pred_for_gt: Dict[int, int] = {}
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

        # pred-dup: extra preds overlapping matched GT
        pred_dup: Dict[int, int] = {}
        for gi, pi in assigned_pred_for_gt.items():
            if iou_mat.size == 0:
                continue
            cand = np.where(iou_mat[gi] >= float(args.tp_iou))[0].tolist()
            if len(cand) <= 1:
                continue
            # keep assigned pred, others are dup
            for pidx in cand:
                if pidx != pi:
                    pred_dup[pidx] = gi

        # gt-dup: unmatched GT overlapped by pred assigned to another GT
        gt_dup = set()
        for gi in range(len(gts)):
            if gi in assigned_pred_for_gt:
                continue
            if iou_mat.size == 0:
                continue
            cand = np.where(iou_mat[gi] >= float(args.tp_iou))[0].tolist()
            for pidx in cand:
                if pidx in assigned_gt_for_pred:
                    gt_dup.add(gi)
                    break

        # per-pred stats
        for pi, pred in enumerate(preds):
            pred_bucket = bucket_name_from_min_side(min_side(pred.bbox_xyxy))
            pred_edge = mask_ratio_for_box(white_integral, pred.bbox_xyxy, w, h) >= float(args.edge_white_frac)
            pred_hl = mask_ratio_for_box(hl_integral, pred.bbox_xyxy, w, h) >= float(args.hl_frac)

            bucket_stats[pred_bucket]["n_pred"] += 1

            if pi in assigned_gt_for_pred:
                gi = assigned_gt_for_pred[pi]
                gt_bucket = bucket_name_from_min_side(min_side(gts[gi].bbox_xyxy))
                # TP counted in GT bucket for category stats
                bucket_stats[gt_bucket]["tp"] += 1
                bucket_stats[gt_bucket]["tp_edge"] += int(pred_edge)
                bucket_stats[gt_bucket]["tp_hl"] += int(pred_hl)
                total_tp += 1
                total_tp_edge += int(pred_edge)
                total_tp_hl += int(pred_hl)

                bucket_edge_interaction[gt_bucket][int(pred_edge)]["tp"] += 1
                bucket_hl_interaction[gt_bucket][int(pred_hl)]["tp"] += 1

                # pred-based precision bucket
                bucket_stats[pred_bucket]["n_pred_tp"] += 1
            else:
                # FP
                bucket_stats[pred_bucket]["fp"] += 1
                bucket_stats[pred_bucket]["fp_edge"] += int(pred_edge)
                bucket_stats[pred_bucket]["fp_hl"] += int(pred_hl)
                total_fp += 1
                total_fp_edge += int(pred_edge)
                total_fp_hl += int(pred_hl)

                bucket_edge_interaction[pred_bucket][int(pred_edge)]["fp"] += 1
                bucket_hl_interaction[pred_bucket][int(pred_hl)]["fp"] += 1

                bucket_stats[pred_bucket]["n_pred_fp"] += 1
                if pi in pred_dup:
                    bucket_stats[pred_bucket]["n_pred_dup"] += 1
                    total_pred_dup += 1

                fp_score = float(pred.score or 0.0)
                push_topk(
                    obj_fp_vis,
                    {
                        "priority": fp_score,
                        "img_path": str(img_path),
                        "image_id": image_id,
                        "gts": gts_vis,
                        "preds": preds_vis,
                        "highlight_gt": None,
                        "highlight_pred": pred.bbox_xyxy,
                        "title": f"obj_fp score={fp_score:.2f}",
                    },
                    key_fn=lambda x: x["priority"],
                    k=20,
                )

        # per-GT stats
        for gi, gt in enumerate(gts):
            gt_bucket = bucket_name_from_min_side(min_side(gt.bbox_xyxy))
            gt_edge = mask_ratio_for_box(white_integral, gt.bbox_xyxy, w, h) >= float(args.edge_white_frac)
            gt_hl = mask_ratio_for_box(hl_integral, gt.bbox_xyxy, w, h) >= float(args.hl_frac)

            bucket_stats[gt_bucket]["n_gt"] += 1
            if gi in assigned_pred_for_gt:
                bucket_stats[gt_bucket]["n_gt_tp"] += 1
            else:
                bucket_stats[gt_bucket]["n_gt_fn"] += 1
                bucket_stats[gt_bucket]["fn"] += 1
                bucket_stats[gt_bucket]["fn_edge"] += int(gt_edge)
                bucket_stats[gt_bucket]["fn_hl"] += int(gt_hl)
                total_fn += 1
                total_fn_edge += int(gt_edge)
                total_fn_hl += int(gt_hl)

                bucket_edge_interaction[gt_bucket][int(gt_edge)]["fn"] += 1
                bucket_hl_interaction[gt_bucket][int(gt_hl)]["fn"] += 1

                if gi in gt_dup:
                    bucket_stats[gt_bucket]["n_gt_dup"] += 1
                    total_gt_dup += 1

                if gi < best_fn_pred_idx.size:
                    pi = int(best_fn_pred_idx[gi])
                else:
                    pi = -1
                if pi >= 0 and pi < len(preds_fn):
                    best_score = float(pred_scores_fn[pi])
                    best_iou = float(best_fn_iou[gi]) if gi < best_fn_iou.size else 0.0
                    push_topk(
                        obj_fn_vis,
                        {
                            "priority": (best_iou, best_score),
                            "img_path": str(img_path),
                            "image_id": image_id,
                            "gts": gts_vis,
                            "preds": preds_vis,
                            "highlight_gt": gt.bbox_xyxy,
                            "highlight_pred": preds_fn[pi].bbox_xyxy,
                            "title": f"obj_fn bestIoU={best_iou:.2f} score={best_score:.2f}",
                        },
                        key_fn=lambda x: x["priority"],
                        k=20,
                    )

    # derived metrics per bucket
    summary_rows: List[dict] = []
    for b in BUCKET_NAMES:
        bs = bucket_stats[b]
        bi = bucket_image_stats[b]
        image_recall = safe_div(bi["tp_images"], bi["tp_images"] + bi["fn_images"])
        image_fp_rate = safe_div(bi["fp_images"], bi["fp_images"] + bi["tn_images"])
        obj_recall = safe_div(bs["n_gt_tp"], bs["n_gt"])
        obj_precision = safe_div(bs["n_pred_tp"], bs["n_pred_tp"] + bs["n_pred_fp"])
        edge_tp_rate = safe_div(bs["tp_edge"], bs["tp"])
        edge_fp_rate = safe_div(bs["fp_edge"], bs["fp"])
        edge_fn_rate = safe_div(bs["fn_edge"], bs["fn"])
        hl_tp_rate = safe_div(bs["tp_hl"], bs["tp"])
        hl_fp_rate = safe_div(bs["fp_hl"], bs["fp"])
        hl_fn_rate = safe_div(bs["fn_hl"], bs["fn"])

        summary_rows.append(
            {
                "bucket": b,
                "n_gt": bs["n_gt"],
                "n_gt_tp": bs["n_gt_tp"],
                "n_gt_fn": bs["n_gt_fn"],
                "n_gt_dup": bs["n_gt_dup"],
                "n_pred": bs["n_pred"],
                "n_pred_tp": bs["n_pred_tp"],
                "n_pred_fp": bs["n_pred_fp"],
                "n_pred_dup": bs["n_pred_dup"],
                "image_pos": bi["pos_images"],
                "image_neg": bi["neg_images"],
                "image_tp": bi["tp_images"],
                "image_fn": bi["fn_images"],
                "image_fp": bi["fp_images"],
                "image_tn": bi["tn_images"],
                "image_recall": image_recall,
                "image_fp_rate": image_fp_rate,
                "obj_recall": obj_recall,
                "obj_precision": obj_precision,
                "edge_tp_rate": edge_tp_rate,
                "edge_fp_rate": edge_fp_rate,
                "edge_fn_rate": edge_fn_rate,
                "highlight_tp_rate": hl_tp_rate,
                "highlight_fp_rate": hl_fp_rate,
                "highlight_fn_rate": hl_fn_rate,
                "fp_edge_image_rate": safe_div(bi["fp_edge_images"], bi["fp_images"]),
                "fp_highlight_image_rate": safe_div(bi["fp_hl_images"], bi["fp_images"]),
            }
        )

    # write summary CSV
    summary_path = report_dir / "summary_metrics.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(summary_rows[0].keys()) if summary_rows else ["bucket"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in summary_rows:
            w.writerow(r)

    # 01 scale bucket metrics
    lines = build_header_block(args, report_dir, created_at, hl_strategy)
    lines += [
        "## Notes",
        "- Image-level metrics follow the definition: GT present + preds non-empty => hit; GT present + preds empty => miss.",
        "- Object-level recall uses GT buckets by GT size; object-level precision uses pred buckets by pred size.",
        "",
        "## Bucket Metrics",
        "| bucket | n_gt | n_pred | image_recall | image_fp_rate | obj_recall | obj_precision | image_pos | image_neg |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in summary_rows:
        lines.append(
            "| {bucket} | {n_gt} | {n_pred} | {image_recall:.4f} | {image_fp_rate:.4f} | {obj_recall:.4f} | {obj_precision:.4f} | {image_pos} | {image_neg} |".format(
                **r
            )
        )
    write_markdown(report_dir / "01_scale_bucket_metrics.md", "scale", lines)

    # 02 edge impact
    lines = build_header_block(args, report_dir, created_at, hl_strategy)
    lines += [
        "## Overall Edge Rates (object-level)",
        f"- TP edge rate: {safe_div(total_tp_edge, total_tp):.4f}",
        f"- FP edge rate: {safe_div(total_fp_edge, total_fp):.4f}",
        f"- FN edge rate: {safe_div(total_fn_edge, total_fn):.4f}",
        "",
        "## Bucket Edge Rates (object-level)",
        "| bucket | tp_edge_rate | fp_edge_rate | fn_edge_rate | tp | fp | fn |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for b in BUCKET_NAMES:
        bs = bucket_stats[b]
        lines.append(
            "| {bucket} | {tp_edge_rate:.4f} | {fp_edge_rate:.4f} | {fn_edge_rate:.4f} | {tp} | {fp} | {fn} |".format(
                bucket=b,
                tp_edge_rate=safe_div(bs["tp_edge"], bs["tp"]),
                fp_edge_rate=safe_div(bs["fp_edge"], bs["fp"]),
                fn_edge_rate=safe_div(bs["fn_edge"], bs["fn"]),
                tp=bs["tp"],
                fp=bs["fp"],
                fn=bs["fn"],
            )
        )
    write_markdown(report_dir / "02_edge_impact.md", "edge", lines)

    # 03 highlight impact
    lines = build_header_block(args, report_dir, created_at, hl_strategy)
    lines += [
        "## Overall Highlight Rates (object-level)",
        f"- TP highlight rate: {safe_div(total_tp_hl, total_tp):.4f}",
        f"- FP highlight rate: {safe_div(total_fp_hl, total_fp):.4f}",
        f"- FN highlight rate: {safe_div(total_fn_hl, total_fn):.4f}",
        "",
        "## Bucket Highlight Rates (object-level)",
        "| bucket | tp_hl_rate | fp_hl_rate | fn_hl_rate | tp | fp | fn |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for b in BUCKET_NAMES:
        bs = bucket_stats[b]
        lines.append(
            "| {bucket} | {tp_hl_rate:.4f} | {fp_hl_rate:.4f} | {fn_hl_rate:.4f} | {tp} | {fp} | {fn} |".format(
                bucket=b,
                tp_hl_rate=safe_div(bs["tp_hl"], bs["tp"]),
                fp_hl_rate=safe_div(bs["fp_hl"], bs["fp"]),
                fn_hl_rate=safe_div(bs["fn_hl"], bs["fn"]),
                tp=bs["tp"],
                fp=bs["fp"],
                fn=bs["fn"],
            )
        )
    write_markdown(report_dir / "03_highlight_impact.md", "highlight", lines)

    # 04 interaction: scale x edge
    lines = build_header_block(args, report_dir, created_at, hl_strategy)
    lines += [
        "## Scale x Edge (object-level counts)",
        "| bucket | tp_edge | tp_non_edge | fp_edge | fp_non_edge | fn_edge | fn_non_edge | fp_edge_image_rate |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for b in BUCKET_NAMES:
        edge = bucket_edge_interaction[b][1]
        non_edge = bucket_edge_interaction[b][0]
        fp_edge_image_rate = safe_div(bucket_image_stats[b]["fp_edge_images"], bucket_image_stats[b]["fp_images"])
        lines.append(
            "| {bucket} | {tp_edge} | {tp_non} | {fp_edge} | {fp_non} | {fn_edge} | {fn_non} | {fp_edge_image_rate:.4f} |".format(
                bucket=b,
                tp_edge=edge["tp"],
                tp_non=non_edge["tp"],
                fp_edge=edge["fp"],
                fp_non=non_edge["fp"],
                fn_edge=edge["fn"],
                fn_non=non_edge["fn"],
                fp_edge_image_rate=fp_edge_image_rate,
            )
        )
    write_markdown(report_dir / "04_interaction_scale_x_edge.md", "interaction_edge", lines)

    # 05 interaction: scale x highlight
    lines = build_header_block(args, report_dir, created_at, hl_strategy)
    lines += [
        "## Scale x Highlight (object-level counts)",
        "| bucket | tp_hl | tp_non_hl | fp_hl | fp_non_hl | fn_hl | fn_non_hl | fp_highlight_image_rate |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for b in BUCKET_NAMES:
        hl = bucket_hl_interaction[b][1]
        non_hl = bucket_hl_interaction[b][0]
        fp_hl_image_rate = safe_div(bucket_image_stats[b]["fp_hl_images"], bucket_image_stats[b]["fp_images"])
        lines.append(
            "| {bucket} | {tp_hl} | {tp_non} | {fp_hl} | {fp_non} | {fn_hl} | {fn_non} | {fp_hl_image_rate:.4f} |".format(
                bucket=b,
                tp_hl=hl["tp"],
                tp_non=non_hl["tp"],
                fp_hl=hl["fp"],
                fp_non=non_hl["fp"],
                fn_hl=hl["fn"],
                fn_non=non_hl["fn"],
                fp_hl_image_rate=fp_hl_image_rate,
            )
        )
    write_markdown(report_dir / "05_interaction_scale_x_highlight.md", "interaction_hl", lines)

    vis_dir = report_dir / "vis"
    save_vis_candidates(image_fp_vis, vis_dir / "image_fp", "image_fp", limit=None)
    save_vis_candidates(image_fn_vis, vis_dir / "image_fn", "image_fn", limit=None)
    save_vis_candidates(obj_fp_vis, vis_dir / "obj_fp", "obj_fp", limit=20)
    save_vis_candidates(obj_fn_vis, vis_dir / "obj_fn", "obj_fn", limit=20)

    # minimal meta
    meta = {
        "created_at": created_at,
        "weights": str(weights),
        "image_dirs": [str(d) for d in image_dirs],
        "label_dirs": [str(d) for d in label_dirs],
        "image_dir": ", ".join(str(d) for d in image_dirs),
        "out_root": str(out_root),
        "report_dir": str(report_dir),
        "conf": float(args.conf),
        "nms_iou": float(args.nms_iou),
        "max_det": int(args.max_det),
        "tp_iou": float(args.tp_iou),
        "edge_white_frac": float(args.edge_white_frac),
        "highlight_strategy": hl_strategy,
        "highlight_frac": float(args.hl_frac),
        "imgsz": int(args.imgsz),
        "batch": int(args.batch),
        "infer_chunk": int(args.infer_chunk),
        "image_level": image_level,
        "dup": {"pred_dup": total_pred_dup, "gt_dup": total_gt_dup},
    }
    (report_dir / "run_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[WP1] report_dir = {report_dir}")
    return report_dir


def main() -> None:
    args = parse_args()
    analyze(args)


if __name__ == "__main__":
    main()
"""""
python /home/ubuntu/project/deduibi/yolo/analyze/code/wp1_error_mining.py \
  --weights /home/ubuntu/project/deduibi/yolo/models/defect/exp_2601292207/best/best.pt \
  --image_dir /home/ubuntu/project/deduibi/yolo/dataset/yolo/datasetm6c/images/val \
  --image_dir /home/ubuntu/project/deduibi/yolo/dataset/yolo/datasetm6c/images/test \
  --out_root /home/ubuntu/project/deduibi/yolo/analyze/result \
  --batch 4 --infer_chunk 16
"""""
