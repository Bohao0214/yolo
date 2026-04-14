#!/usr/bin/env python3
"""
Multi-model inference + overlay + "ours advantage" analysis.

Edit USER_EDIT_CONFIG only, then run:
  python /home/ubuntu/hpproject/yolo/analyze/code/girdfig.py
"""

from __future__ import annotations

import argparse
import copy
import csv
import datetime as dt
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import yaml

REPO_ROOT = Path("/home/ubuntu/hpproject/yolo")
os.environ.setdefault("YOLO_AUTOINSTALL", "False")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
try:
    import third_party  # type: ignore # noqa: F401
except Exception:
    pass

from ultralytics import YOLO

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


# 你可以只修改这里，不改其它代码
USER_EDIT_CONFIG: Dict[str, Any] = {
    "models": [
        # {"name": "baseline", "path": "/abs/path/to/best.pt"},
        {"name": "bsd", "path": "/home/ubuntu/hpproject/yolo/experiments/a3b3d3/datasetm6c/exp_2604050042/train/weights/best.pt"},
        {"name": "baseline", "path": "/home/ubuntu/hpproject/yolo/experiments/baseline/datasetm6c/exp_2603040206/train/weights/best.pt"},
        {"name": "our", "path": "/home/ubuntu/hpproject/yolo/experiments/a4b7d6/datasetm6c/defect241__a4__b7__d6/exp_2603060404/train/weights/best.pt"},
    ],
    "data_yaml": "/home/ubuntu/hpproject/yolo/configs/enhance/datasetm6c/defect241.yaml",
    "dataset_root": "/home/ubuntu/hpproject/yolo/dataset/yolo/datasetm6c",
    "split": "val+test",
    "infer_params": {
        "imgsz": 640,
        "conf": 0.25,
        "iou": 0.7,
        "max_det": 100,
        "device": "0",
        "batch": 4,
        "tp_iou": 0.3,
        "match_metric": "iou",  # iou | ios
        "score_floor": 0.01,
        "raw_max_det": 3000,
    },
    "out_root": "/home/ubuntu/hpproject/yolo/analyze/result",
    "report_prefix": "report_",
    "ours_model_name": "our",
    "ours_name_patterns": ["our", "ours", "本文", "sd-yolo11"],
    "visual": {
        "save_overlays": True,
        "draw_conf": 0.25,
        "jpeg_quality": 92,
        "max_images_per_split": 0,  # 0 = all
        "draw_gt": True,
    },
    "advantage": {
        "topk_samples_per_other": 300,
    },
}


@dataclass(frozen=True)
class Det:
    box: Tuple[float, float, float, float]
    score: float
    label: int


@dataclass(frozen=True)
class ImageItem:
    image_path: Path
    label_path: Path
    split: str
    rel_key: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multi-model grid/overlay + ours-advantage analysis.")
    p.add_argument("--config_json", type=str, default="", help="JSON override for USER_EDIT_CONFIG.")
    p.add_argument("--dry_run", action="store_true", help="Only resolve config and dataset; do not run inference.")
    return p.parse_args()


def deep_merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def resolve_path(path_like: str, base: Path = REPO_ROOT) -> Path:
    p = Path(str(path_like).strip()).expanduser()
    if p.is_absolute():
        return p.resolve()
    return (base / p).resolve()


def make_report_dir(out_root: Path, prefix: str) -> Path:
    ts = dt.datetime.now().strftime("%y%m%d%H%M")
    root = out_root / f"{prefix}{ts}"
    if not root.exists():
        root.mkdir(parents=True, exist_ok=False)
        return root
    idx = 1
    while True:
        c = out_root / f"{prefix}{ts}_{idx:02d}"
        if not c.exists():
            c.mkdir(parents=True, exist_ok=False)
            return c
        idx += 1


def load_yaml(path: Path) -> Dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML is not mapping: {path}")
    return data


def parse_split_spec(spec: str) -> List[str]:
    s = (spec or "test").strip()
    parts = [x.strip() for x in s.replace(",", "+").split("+") if x.strip()]
    out: List[str] = []
    seen = set()
    for p in parts:
        if p not in seen:
            out.append(p)
            seen.add(p)
    return out if out else ["test"]


def resolve_data_yaml_and_root(cfg: Dict[str, Any]) -> Tuple[Path, Path]:
    data_yaml = cfg.get("data_yaml")
    dataset_root = cfg.get("dataset_root")
    d_yaml: Optional[Path] = None
    d_root: Optional[Path] = None

    if isinstance(data_yaml, str) and data_yaml.strip():
        d_yaml = resolve_path(data_yaml)
    if isinstance(dataset_root, str) and dataset_root.strip():
        d_root = resolve_path(dataset_root)

    if d_yaml is None:
        if d_root is None:
            raise RuntimeError("need at least one of data_yaml / dataset_root")
        for n in ("data.yaml", "dataset.yaml", "data.yml", "dataset.yml"):
            cand = (d_root / n).resolve()
            if cand.exists():
                d_yaml = cand
                break
        if d_yaml is None:
            raise FileNotFoundError(f"data yaml not found under dataset_root: {d_root}")

    if d_root is None:
        d_obj = load_yaml(d_yaml)
        p = d_obj.get("path")
        if isinstance(p, str) and p.strip():
            d_root = resolve_path(p, base=d_yaml.parent)
        else:
            d_root = d_yaml.parent.resolve()

    return d_yaml.resolve(), d_root.resolve()


def split_dirs(dataset_root: Path, split: str) -> Tuple[Path, Path]:
    image_dir = (dataset_root / "images" / split).resolve()
    label_dir = (dataset_root / "labels" / split).resolve()
    if image_dir.exists() and label_dir.exists():
        return image_dir, label_dir
    if split == "bal":
        alt = "val"
    elif split == "val":
        alt = "bal"
    else:
        alt = ""
    if alt:
        image_dir = (dataset_root / "images" / alt).resolve()
        label_dir = (dataset_root / "labels" / alt).resolve()
        if image_dir.exists() and label_dir.exists():
            return image_dir, label_dir
    raise FileNotFoundError(f"split dirs not found: {split} under {dataset_root}")


def list_images(image_dir: Path) -> List[Path]:
    return sorted([p for p in image_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS])


def read_image_size(image_path: Path) -> Tuple[int, int]:
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"failed read image: {image_path}")
    h, w = img.shape[:2]
    return int(w), int(h)


def xywhn_to_xyxy(xc: float, yc: float, w: float, h: float, img_w: int, img_h: int) -> Tuple[float, float, float, float]:
    x1 = (xc - w / 2.0) * img_w
    y1 = (yc - h / 2.0) * img_h
    x2 = (xc + w / 2.0) * img_w
    y2 = (yc + h / 2.0) * img_h
    return float(x1), float(y1), float(x2), float(y2)


def seg_norm_to_xyxy(coords: Sequence[float], img_w: int, img_h: int) -> Tuple[float, float, float, float]:
    xs = [float(v) * img_w for v in coords[0::2]]
    ys = [float(v) * img_h for v in coords[1::2]]
    return float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))


def clip_xyxy(box: Sequence[float], img_w: int, img_h: int) -> Optional[Tuple[float, float, float, float]]:
    x1, y1, x2, y2 = [float(v) for v in box[:4]]
    x1 = max(0.0, min(float(img_w), x1))
    x2 = max(0.0, min(float(img_w), x2))
    y1 = max(0.0, min(float(img_h), y1))
    y2 = max(0.0, min(float(img_h), y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def parse_yolo_label_file(label_path: Path, img_w: int, img_h: int) -> List[Det]:
    out: List[Det] = []
    if not label_path.exists():
        return out
    for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        try:
            cls_id = int(float(parts[0]))
            nums = [float(x) for x in parts[1:]]
        except Exception:
            continue
        if len(nums) > 4 and len(nums) % 2 == 0:
            box = seg_norm_to_xyxy(nums, img_w, img_h)
        else:
            if len(nums) < 4:
                continue
            box = xywhn_to_xyxy(nums[0], nums[1], nums[2], nums[3], img_w, img_h)
        c = clip_xyxy(box, img_w, img_h)
        if c is None:
            continue
        out.append(Det(box=c, score=1.0, label=cls_id))
    return out


def collect_images_and_gt(dataset_root: Path, splits: Sequence[str], max_images_per_split: int) -> Tuple[List[ImageItem], Dict[str, List[Det]], List[int]]:
    items: List[ImageItem] = []
    gt_map: Dict[str, List[Det]] = {}
    classes = set()

    for sp in splits:
        image_dir, label_dir = split_dirs(dataset_root, sp)
        imgs = list_images(image_dir)
        if max_images_per_split > 0:
            imgs = imgs[: int(max_images_per_split)]
        for p in imgs:
            rel = p.relative_to(image_dir)
            lp = (label_dir / rel).with_suffix(".txt")
            w, h = read_image_size(p)
            gts = parse_yolo_label_file(lp, w, h)
            key = str(p.resolve())
            gt_map[key] = gts
            for g in gts:
                classes.add(int(g.label))
            items.append(ImageItem(image_path=p.resolve(), label_path=lp.resolve(), split=sp, rel_key=str(rel)))
    items.sort(key=lambda x: str(x.image_path))
    return items, gt_map, sorted(classes)


def iou_one_to_many(box: Sequence[float], boxes: np.ndarray) -> np.ndarray:
    if boxes.size == 0:
        return np.zeros((0,), dtype=np.float32)
    x1 = np.maximum(float(box[0]), boxes[:, 0])
    y1 = np.maximum(float(box[1]), boxes[:, 1])
    x2 = np.minimum(float(box[2]), boxes[:, 2])
    y2 = np.minimum(float(box[3]), boxes[:, 3])
    inter = np.clip(x2 - x1, 0.0, None) * np.clip(y2 - y1, 0.0, None)
    area_a = max(float(box[2] - box[0]), 0.0) * max(float(box[3] - box[1]), 0.0)
    area_b = np.clip(boxes[:, 2] - boxes[:, 0], 0.0, None) * np.clip(boxes[:, 3] - boxes[:, 1], 0.0, None)
    union = area_a + area_b - inter
    return np.divide(inter, np.maximum(union, 1e-12), out=np.zeros_like(inter), where=union > 0).astype(np.float32)


def ios_one_to_many(box: Sequence[float], boxes: np.ndarray) -> np.ndarray:
    if boxes.size == 0:
        return np.zeros((0,), dtype=np.float32)
    x1 = np.maximum(float(box[0]), boxes[:, 0])
    y1 = np.maximum(float(box[1]), boxes[:, 1])
    x2 = np.minimum(float(box[2]), boxes[:, 2])
    y2 = np.minimum(float(box[3]), boxes[:, 3])
    inter = np.clip(x2 - x1, 0.0, None) * np.clip(y2 - y1, 0.0, None)
    area_a = max(float(box[2] - box[0]), 0.0) * max(float(box[3] - box[1]), 0.0)
    area_b = np.clip(boxes[:, 2] - boxes[:, 0], 0.0, None) * np.clip(boxes[:, 3] - boxes[:, 1], 0.0, None)
    smaller = np.minimum(area_a, area_b)
    return np.divide(inter, np.maximum(smaller, 1e-12), out=np.zeros_like(inter), where=smaller > 0).astype(np.float32)


def chunks(items: Sequence[ImageItem], n: int) -> Iterable[Sequence[ImageItem]]:
    n = max(1, int(n))
    for i in range(0, len(items), n):
        yield items[i : i + n]


def run_inference_one_model(
    name: str,
    weight_path: Path,
    items: Sequence[ImageItem],
    infer_cfg: Dict[str, Any],
    raw_preds_dir: Path,
) -> Dict[str, List[Det]]:
    model = YOLO(str(weight_path))
    pred_map: Dict[str, List[Det]] = {str(i.image_path): [] for i in items}

    by_split: Dict[str, List[ImageItem]] = {}
    for it in items:
        by_split.setdefault(it.split, []).append(it)

    score_floor = float(infer_cfg.get("score_floor", 0.01))
    iou = float(infer_cfg.get("iou", 0.7))
    imgsz = int(infer_cfg.get("imgsz", 640))
    batch = int(infer_cfg.get("batch", 4))
    max_det = int(infer_cfg.get("raw_max_det", infer_cfg.get("max_det", 3000)))
    device = str(infer_cfg.get("device", ""))

    for sp, split_items in by_split.items():
        rows = []
        for chunk in chunks(split_items, batch):
            kwargs = dict(
                source=[str(it.image_path) for it in chunk],
                conf=score_floor,
                iou=iou,
                imgsz=imgsz,
                max_det=max_det,
                batch=batch,
                save=False,
                verbose=False,
                stream=True,
            )
            if device:
                kwargs["device"] = device
            stream = model.predict(**kwargs)
            tmp: Dict[str, List[Det]] = {}
            for res in stream:
                ip = str(Path(res.path).resolve())
                dets: List[Det] = []
                if res.boxes is not None and res.boxes.xyxy is not None:
                    boxes = res.boxes.xyxy.detach().cpu().numpy().tolist()
                    scores = res.boxes.conf.detach().cpu().numpy().tolist() if res.boxes.conf is not None else [1.0] * len(boxes)
                    labels = res.boxes.cls.detach().cpu().numpy().tolist() if res.boxes.cls is not None else [0] * len(boxes)
                    for b, s, c in zip(boxes, scores, labels):
                        dets.append(Det(box=(float(b[0]), float(b[1]), float(b[2]), float(b[3])), score=float(s), label=int(c)))
                tmp[ip] = dets
            for it in chunk:
                key = str(it.image_path)
                dets = tmp.get(key, [])
                pred_map[key] = dets
                rows.append(
                    {
                        "image_id": it.image_path.stem,
                        "image_path": key,
                        "boxes": [[float(v) for v in d.box] for d in dets],
                        "scores": [float(d.score) for d in dets],
                        "labels": [int(d.label) for d in dets],
                    }
                )

        out_json = raw_preds_dir / name / f"{sp}.json"
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(
            json.dumps(
                {
                    "format": "det_preds_v1",
                    "model": name,
                    "weights": str(weight_path),
                    "split": sp,
                    "predictions": rows,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    return pred_map


def _auto_style(h: int, w: int) -> Tuple[int, float, int]:
    base = min(h, w)
    thickness = max(1, int(round(base / 380)))
    font_scale = max(0.5, base / 980.0)
    pad = max(2, int(round(base / 260)))
    return thickness, font_scale, pad


def _draw_label_box(img: np.ndarray, x: int, y: int, text: str, color: Tuple[int, int, int], font_scale: float, pad: int) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), bl = cv2.getTextSize(text, font, font_scale, 1)
    x1 = max(0, x)
    y1 = max(0, y - th - 2 * pad - bl)
    x2 = min(img.shape[1] - 1, x1 + tw + 2 * pad)
    y2 = min(img.shape[0] - 1, y1 + th + 2 * pad + bl)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, -1, lineType=cv2.LINE_AA)
    cv2.putText(img, text, (x1 + pad, y2 - pad - bl), font, font_scale, (20, 20, 20), 1, cv2.LINE_AA)


def draw_overlay(
    item: ImageItem,
    gt_dets: Sequence[Det],
    pred_dets: Sequence[Det],
    draw_conf: float,
    out_path: Path,
    model_name: str,
    draw_gt: bool,
    jpeg_quality: int,
) -> None:
    img = cv2.imread(str(item.image_path))
    if img is None:
        return
    h, w = img.shape[:2]
    thick, fscale, pad = _auto_style(h, w)

    if draw_gt:
        for g in gt_dets:
            x1, y1, x2, y2 = [int(round(v)) for v in g.box]
            cv2.rectangle(img, (x1, y1), (x2, y2), (70, 210, 70), thickness=thick + 1, lineType=cv2.LINE_AA)
        if gt_dets:
            _draw_label_box(img, 8, 26 + int(20 * fscale), f"GT: {len(gt_dets)}", (70, 210, 70), fscale, pad)

    keep = [d for d in pred_dets if d.score >= float(draw_conf)]
    keep = sorted(keep, key=lambda d: d.score, reverse=True)
    for d in keep:
        x1, y1, x2, y2 = [int(round(v)) for v in d.box]
        color = (55, 130, 255)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness=thick, lineType=cv2.LINE_AA)
        _draw_label_box(img, x1, y1, f"c{int(d.label)} {d.score:.2f}", color, fscale, pad)

    _draw_label_box(
        img,
        8,
        h - 8,
        f"{model_name} | pred>={draw_conf:.2f}: {len(keep)} | split={item.split}",
        (210, 200, 85),
        max(0.45, fscale * 0.9),
        max(2, pad - 1),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])


def match_details_for_image(
    gt_dets: Sequence[Det],
    pred_dets: Sequence[Det],
    score_thr: float,
    iou_thr: float,
    metric: str = "iou",
) -> Dict[str, Any]:
    keep = [(i, p) for i, p in enumerate(pred_dets) if p.score >= float(score_thr)]
    gt_cls: Dict[int, List[int]] = {}
    pred_cls: Dict[int, List[int]] = {}
    for gi, g in enumerate(gt_dets):
        gt_cls.setdefault(int(g.label), []).append(gi)
    for pi, p in keep:
        pred_cls.setdefault(int(p.label), []).append(pi)

    matched_gt: set[int] = set()
    tp_pred: set[int] = set()

    for cls_id in sorted(set(gt_cls.keys()) | set(pred_cls.keys())):
        gidx = gt_cls.get(cls_id, [])
        pidx = pred_cls.get(cls_id, [])
        if not pidx:
            continue
        gboxes = (
            np.asarray([gt_dets[i].box for i in gidx], dtype=np.float32).reshape(-1, 4)
            if gidx
            else np.zeros((0, 4), dtype=np.float32)
        )
        used = np.zeros((len(gidx),), dtype=bool)
        pidx = sorted(pidx, key=lambda idx: pred_dets[idx].score, reverse=True)
        for pi in pidx:
            if gboxes.shape[0] == 0:
                continue
            if metric == "ios":
                ov = ios_one_to_many(pred_dets[pi].box, gboxes)
            else:
                ov = iou_one_to_many(pred_dets[pi].box, gboxes)
            best = int(np.argmax(ov))
            if ov[best] >= float(iou_thr) and not used[best]:
                used[best] = True
                matched_gt.add(gidx[best])
                tp_pred.add(pi)

    tp = len(tp_pred)
    fp = len(keep) - tp
    fn = len(gt_dets) - len(matched_gt)
    return {
        "matched_gt_idx": matched_gt,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "keep_count": len(keep),
        "has_gt": len(gt_dets) > 0,
        "has_pred": len(keep) > 0,
    }


def collect_classes(gt_map: Dict[str, List[Det]], pred_map: Dict[str, List[Det]]) -> List[int]:
    cls = set()
    for dets in gt_map.values():
        for d in dets:
            cls.add(int(d.label))
    for dets in pred_map.values():
        for d in dets:
            cls.add(int(d.label))
    return sorted(cls)


def compute_ap_101(gt_map: Dict[str, List[Det]], pred_map: Dict[str, List[Det]], cls_id: int, iou_thr: float) -> float:
    gt_by_img = {}
    npos = 0
    for k, ds in gt_map.items():
        arr = np.asarray([d.box for d in ds if d.label == cls_id], dtype=np.float32).reshape(-1, 4) if ds else np.zeros((0, 4), dtype=np.float32)
        gt_by_img[k] = arr
        npos += int(arr.shape[0])
    if npos == 0:
        return float("nan")

    pred_rows = []
    for k, ds in pred_map.items():
        for d in ds:
            if d.label == cls_id:
                pred_rows.append((k, float(d.score), d.box))
    pred_rows.sort(key=lambda x: x[1], reverse=True)
    if not pred_rows:
        return 0.0

    matched = {k: np.zeros((v.shape[0],), dtype=bool) for k, v in gt_by_img.items()}
    tp = np.zeros((len(pred_rows),), dtype=np.float32)
    fp = np.zeros((len(pred_rows),), dtype=np.float32)
    for i, (k, _, box) in enumerate(pred_rows):
        g = gt_by_img.get(k, np.zeros((0, 4), dtype=np.float32))
        if g.shape[0] == 0:
            fp[i] = 1.0
            continue
        ov = iou_one_to_many(box, g)
        best = int(np.argmax(ov))
        if ov[best] >= float(iou_thr) and not matched[k][best]:
            tp[i] = 1.0
            matched[k][best] = True
        else:
            fp[i] = 1.0

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    rec = tp_cum / max(float(npos), 1e-12)
    pre = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)
    levels = np.linspace(0.0, 1.0, 101, dtype=np.float32)
    vals = []
    for r in levels:
        p = pre[rec >= r]
        vals.append(float(np.max(p)) if p.size > 0 else 0.0)
    return float(np.mean(vals))


def compute_main_metrics(
    gt_map: Dict[str, List[Det]],
    pred_map: Dict[str, List[Det]],
    score_thr: float,
    tp_iou: float,
    match_metric: str,
) -> Dict[str, Any]:
    details = {}
    tp = fp = fn = 0
    hit = miss = fp_img = tn = 0
    for k in sorted(gt_map.keys()):
        d = match_details_for_image(gt_map[k], pred_map.get(k, []), score_thr, tp_iou, metric=match_metric)
        details[k] = d
        tp += int(d["tp"])
        fp += int(d["fp"])
        fn += int(d["fn"])
        has_gt = bool(d["has_gt"])
        has_pred = bool(d["has_pred"])
        if has_gt and has_pred:
            hit += 1
        elif has_gt and (not has_pred):
            miss += 1
        elif (not has_gt) and has_pred:
            fp_img += 1
        else:
            tn += 1

    p_obj = float(tp) / float(tp + fp) if (tp + fp) > 0 else 0.0
    r_obj = float(tp) / float(tp + fn) if (tp + fn) > 0 else 0.0
    p_img = float(hit) / float(hit + fp_img) if (hit + fp_img) > 0 else 0.0
    r_img = float(hit) / float(hit + miss) if (hit + miss) > 0 else 0.0
    fpr_img = float(fp_img) / float(fp_img + tn) if (fp_img + tn) > 0 else 0.0

    classes = collect_classes(gt_map, pred_map)
    iou_thresholds = [0.5 + 0.05 * i for i in range(10)]
    map_by_iou = []
    for thr in iou_thresholds:
        aps = []
        for cls_id in classes:
            ap = compute_ap_101(gt_map, pred_map, int(cls_id), thr)
            if not np.isnan(ap):
                aps.append(ap)
        map_by_iou.append(float(np.mean(aps)) if aps else 0.0)

    return {
        "map50": map_by_iou[0] if map_by_iou else 0.0,
        "map50_95": float(np.mean(map_by_iou)) if map_by_iou else 0.0,
        "obj_precision": p_obj,
        "obj_recall": r_obj,
        "img_precision": p_img,
        "img_recall": r_img,
        "img_fpr": fpr_img,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "hit_img": hit,
        "miss_img": miss,
        "fp_img": fp_img,
        "tn_img": tn,
        "details": details,
    }


def infer_ours_name(model_names: Sequence[str], cfg: Dict[str, Any]) -> Optional[str]:
    explicit = str(cfg.get("ours_model_name", "")).strip()
    if explicit and explicit in model_names:
        return explicit
    pats = [str(x).lower() for x in cfg.get("ours_name_patterns", []) if str(x).strip()]
    for n in model_names:
        low = n.lower()
        if any(p in low for p in pats):
            return n
    return None


def analyze_ours_advantage(
    ours_name: str,
    model_metrics: Dict[str, Dict[str, Any]],
    gt_map: Dict[str, List[Det]],
    image_index: Dict[str, ImageItem],
    topk: int,
) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {
        "object_pairwise": [],
        "object_pairwise_samples": [],
        "object_strict": [],
        "object_strict_samples": [],
        "image_pairwise": [],
        "image_pairwise_samples": [],
        "image_strict": [],
        "image_strict_samples": [],
        "fp_box_suppress_pairwise": [],
    }
    all_models = sorted(model_metrics.keys())
    others = [m for m in all_models if m != ours_name]
    if not others:
        return out

    ours_det = model_metrics[ours_name]["details"]

    # pairwise object gain: ours TP, other FN (same GT object)
    for other in others:
        o_det = model_metrics[other]["details"]
        gain_count = 0
        gain_img = 0
        samples = []
        total_gt = 0
        for k, gts in gt_map.items():
            total_gt += len(gts)
            ours_hit = set(ours_det[k]["matched_gt_idx"])
            oth_hit = set(o_det[k]["matched_gt_idx"])
            gain = sorted(list(ours_hit - oth_hit))
            if gain:
                gain_img += 1
                gain_count += len(gain)
                if len(samples) < topk:
                    it = image_index[k]
                    samples.append(
                        {
                            "other_model": other,
                            "split": it.split,
                            "image_path": k,
                            "gain_gt_indices": ",".join(str(x) for x in gain),
                            "gain_count": len(gain),
                        }
                    )
        out["object_pairwise"].append(
            {
                "ours_model": ours_name,
                "other_model": other,
                "gain_tp_objects": gain_count,
                "gain_images": gain_img,
                "total_gt_objects": total_gt,
                "gain_ratio_vs_gt": float(gain_count) / float(total_gt) if total_gt > 0 else 0.0,
            }
        )
        out["object_pairwise_samples"].extend(samples)

    # strict object gain: ours TP and all others FN
    strict_count = 0
    strict_img = 0
    strict_samples = []
    total_gt = sum(len(v) for v in gt_map.values())
    for k, gts in gt_map.items():
        ours_hit = set(ours_det[k]["matched_gt_idx"])
        other_union = set()
        for o in others:
            other_union |= set(model_metrics[o]["details"][k]["matched_gt_idx"])
        gain = sorted(list(ours_hit - other_union))
        if gain:
            strict_img += 1
            strict_count += len(gain)
            if len(strict_samples) < topk:
                it = image_index[k]
                strict_samples.append(
                    {
                        "split": it.split,
                        "image_path": k,
                        "gain_gt_indices": ",".join(str(x) for x in gain),
                        "gain_count": len(gain),
                    }
                )
    out["object_strict"].append(
        {
            "ours_model": ours_name,
            "strict_gain_tp_objects": strict_count,
            "strict_gain_images": strict_img,
            "total_gt_objects": total_gt,
            "strict_gain_ratio_vs_gt": float(strict_count) / float(total_gt) if total_gt > 0 else 0.0,
        }
    )
    out["object_strict_samples"] = strict_samples

    # image-level: ours TP while other FN (positive images)
    for other in others:
        o_det = model_metrics[other]["details"]
        gain_tp_img = 0
        gain_tn_img = 0
        fp_boxes = 0
        samples = []
        for k, gts in gt_map.items():
            has_gt = len(gts) > 0
            ours_has_pred = bool(ours_det[k]["has_pred"])
            other_has_pred = bool(o_det[k]["has_pred"])
            if has_gt and ours_has_pred and (not other_has_pred):
                gain_tp_img += 1
                if len(samples) < topk:
                    it = image_index[k]
                    samples.append(
                        {
                            "other_model": other,
                            "type": "ours_tp_other_fn",
                            "split": it.split,
                            "image_path": k,
                            "other_fp_boxes": 0,
                        }
                    )
            if (not has_gt) and (not ours_has_pred) and other_has_pred:
                gain_tn_img += 1
                fp_boxes += int(o_det[k]["fp"])
                if len(samples) < topk:
                    it = image_index[k]
                    samples.append(
                        {
                            "other_model": other,
                            "type": "ours_tn_other_fp",
                            "split": it.split,
                            "image_path": k,
                            "other_fp_boxes": int(o_det[k]["fp"]),
                        }
                    )
        out["image_pairwise"].append(
            {
                "ours_model": ours_name,
                "other_model": other,
                "ours_tp_other_fn_images": gain_tp_img,
                "ours_tn_other_fp_images": gain_tn_img,
            }
        )
        out["fp_box_suppress_pairwise"].append(
            {
                "ours_model": ours_name,
                "other_model": other,
                "suppressed_fp_boxes_on_clean_images": fp_boxes,
            }
        )
        out["image_pairwise_samples"].extend(samples)

    # strict image-level
    strict_tp = 0
    strict_tn = 0
    strict_samples = []
    for k, gts in gt_map.items():
        has_gt = len(gts) > 0
        ours_has_pred = bool(ours_det[k]["has_pred"])
        others_has_pred_any = any(bool(model_metrics[o]["details"][k]["has_pred"]) for o in others)
        others_has_pred_all = all(bool(model_metrics[o]["details"][k]["has_pred"]) for o in others)

        if has_gt and ours_has_pred and (not others_has_pred_any):
            strict_tp += 1
            if len(strict_samples) < topk:
                it = image_index[k]
                strict_samples.append({"type": "strict_ours_tp_all_others_fn", "split": it.split, "image_path": k})
        if (not has_gt) and (not ours_has_pred) and others_has_pred_all:
            strict_tn += 1
            if len(strict_samples) < topk:
                it = image_index[k]
                strict_samples.append({"type": "strict_ours_tn_all_others_fp", "split": it.split, "image_path": k})
    out["image_strict"].append(
        {
            "ours_model": ours_name,
            "strict_ours_tp_all_others_fn_images": strict_tp,
            "strict_ours_tn_all_others_fp_images": strict_tn,
        }
    )
    out["image_strict_samples"] = strict_samples
    return out


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def save_report_md(
    report_path: Path,
    cfg: Dict[str, Any],
    report_dir: Path,
    main_rows: List[Dict[str, Any]],
    ours_name: Optional[str],
    adv: Dict[str, List[Dict[str, Any]]],
    failures: List[Dict[str, Any]],
) -> None:
    lines = [
        "# GridFig Report",
        "",
        f"- report_dir: `{report_dir}`",
        f"- split: `{cfg.get('split')}`",
        f"- ours_model: `{ours_name or 'N/A'}`",
        "",
        "## Main Metrics",
        "",
        "| model | mAP@0.5 | mAP@0.5:0.95 | P_obj | R_obj | R_img | FPR_img |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in main_rows:
        lines.append(
            f"| {r['model']} | {r['mAP@0.5']} | {r['mAP@0.5:0.95']} | {r['P_obj']} | {r['R_obj']} | {r['R_img']} | {r['FPR_img']} |"
        )
    lines += [
        "",
        "## Ours Advantage (Pairwise Object/Image)",
        "",
        f"- object_pairwise rows: {len(adv.get('object_pairwise', []))}",
        f"- image_pairwise rows: {len(adv.get('image_pairwise', []))}",
        f"- fp_box_suppress_pairwise rows: {len(adv.get('fp_box_suppress_pairwise', []))}",
    ]
    if failures:
        lines += ["", "## Failures", ""]
        for e in failures:
            lines.append(f"- {json.dumps(e, ensure_ascii=False)}")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    cfg = copy.deepcopy(USER_EDIT_CONFIG)
    if args.config_json:
        cfg = deep_merge(cfg, json.loads(args.config_json))

    data_yaml, dataset_root = resolve_data_yaml_and_root(cfg)
    splits = parse_split_spec(str(cfg.get("split", "test")))
    infer_cfg = cfg.get("infer_params", {}) if isinstance(cfg.get("infer_params"), dict) else {}
    vis_cfg = cfg.get("visual", {}) if isinstance(cfg.get("visual"), dict) else {}
    topk = int((cfg.get("advantage", {}) or {}).get("topk_samples_per_other", 300))

    max_images_per_split = int(vis_cfg.get("max_images_per_split", 0))
    items, gt_map, gt_classes = collect_images_and_gt(dataset_root, splits, max_images_per_split=max_images_per_split)
    image_index = {str(it.image_path): it for it in items}

    out_root = resolve_path(str(cfg.get("out_root", "/home/ubuntu/hpproject/yolo/analyze/result")))
    report_dir = make_report_dir(out_root, str(cfg.get("report_prefix", "report_")))
    raw_preds_dir = report_dir / "raw_preds"
    fig_dir = report_dir / "figures" / "overlays"
    tables_dir = report_dir / "tables"
    logs_dir = report_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    runtime_info = {
        "time": dt.datetime.now().isoformat(timespec="seconds"),
        "data_yaml": str(data_yaml),
        "dataset_root": str(dataset_root),
        "split": splits,
        "num_images": len(items),
        "infer_params": infer_cfg,
    }
    (report_dir / "config_used.json").write_text(json.dumps({"config": cfg, "runtime": runtime_info}, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.dry_run:
        print(f"[dry-run] report_dir={report_dir}")
        print(f"[dry-run] dataset_root={dataset_root}")
        print(f"[dry-run] splits={splits}, images={len(items)}")
        return

    models = cfg.get("models", [])
    if not isinstance(models, list) or not models:
        raise RuntimeError("USER_EDIT_CONFIG.models is empty.")

    failures: List[Dict[str, Any]] = []
    model_metrics: Dict[str, Dict[str, Any]] = {}
    main_rows: List[Dict[str, Any]] = []

    score_thr = float(infer_cfg.get("conf", 0.25))
    tp_iou = float(infer_cfg.get("tp_iou", 0.3))
    match_metric = str(infer_cfg.get("match_metric", "iou")).lower().strip() or "iou"
    draw_conf = float(vis_cfg.get("draw_conf", score_thr))
    jpeg_quality = int(vis_cfg.get("jpeg_quality", 92))
    save_overlays = bool(vis_cfg.get("save_overlays", True))
    draw_gt = bool(vis_cfg.get("draw_gt", True))

    for m in models:
        if not isinstance(m, dict):
            failures.append({"model": str(m), "error": "model config item not dict"})
            continue
        name = str(m.get("name", "")).strip()
        path_raw = str(m.get("path", "")).strip()
        if not name or not path_raw:
            failures.append({"model": name or "<empty>", "error": "missing name/path"})
            continue
        w = resolve_path(path_raw)
        if not w.exists():
            failures.append({"model": name, "error": "weight not found", "weight": str(w)})
            continue
        try:
            pred_map = run_inference_one_model(name, w, items, infer_cfg, raw_preds_dir)
            mm = compute_main_metrics(gt_map, pred_map, score_thr=score_thr, tp_iou=tp_iou, match_metric=match_metric)
            model_metrics[name] = mm
            main_rows.append(
                {
                    "model": name,
                    "mAP@0.5": f"{mm['map50']:.6f}",
                    "mAP@0.5:0.95": f"{mm['map50_95']:.6f}",
                    "P_obj": f"{mm['obj_precision']:.6f}",
                    "R_obj": f"{mm['obj_recall']:.6f}",
                    "R_img": f"{mm['img_recall']:.6f}",
                    "FPR_img": f"{mm['img_fpr']:.6f}",
                    "img_precision": f"{mm['img_precision']:.6f}",
                    "TP": int(mm["tp"]),
                    "FP": int(mm["fp"]),
                    "FN": int(mm["fn"]),
                }
            )
            if save_overlays:
                for it in items:
                    out_img = fig_dir / name / it.split / Path(it.rel_key).with_suffix(".jpg")
                    draw_overlay(
                        item=it,
                        gt_dets=gt_map.get(str(it.image_path), []),
                        pred_dets=pred_map.get(str(it.image_path), []),
                        draw_conf=draw_conf,
                        out_path=out_img,
                        model_name=name,
                        draw_gt=draw_gt,
                        jpeg_quality=jpeg_quality,
                    )
        except Exception as ex:
            failures.append({"model": name, "weight": str(w), "error": str(ex)})

    main_rows = sorted(main_rows, key=lambda x: x["model"])
    write_csv(tables_dir / "main_metrics.csv", main_rows)

    ours_name = infer_ours_name(sorted(model_metrics.keys()), cfg)
    if ours_name is None:
        failures.append({"error": "cannot infer ours model from config/model names"})
        adv = {
            "object_pairwise": [],
            "object_pairwise_samples": [],
            "object_strict": [],
            "object_strict_samples": [],
            "image_pairwise": [],
            "image_pairwise_samples": [],
            "image_strict": [],
            "image_strict_samples": [],
            "fp_box_suppress_pairwise": [],
        }
    else:
        adv = analyze_ours_advantage(ours_name, model_metrics, gt_map, image_index, topk=topk)

    write_csv(tables_dir / "ours_adv_object_pairwise.csv", adv["object_pairwise"])
    write_csv(tables_dir / "ours_adv_object_pairwise_samples.csv", adv["object_pairwise_samples"])
    write_csv(tables_dir / "ours_adv_object_strict.csv", adv["object_strict"])
    write_csv(tables_dir / "ours_adv_object_strict_samples.csv", adv["object_strict_samples"])
    write_csv(tables_dir / "ours_adv_image_pairwise.csv", adv["image_pairwise"])
    write_csv(tables_dir / "ours_adv_image_pairwise_samples.csv", adv["image_pairwise_samples"])
    write_csv(tables_dir / "ours_adv_image_strict.csv", adv["image_strict"])
    write_csv(tables_dir / "ours_adv_image_strict_samples.csv", adv["image_strict_samples"])
    write_csv(tables_dir / "ours_adv_fp_box_suppress_pairwise.csv", adv["fp_box_suppress_pairwise"])

    if failures:
        (report_dir / "failures.json").write_text(json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8")

    save_report_md(
        report_path=report_dir / "report.md",
        cfg=cfg,
        report_dir=report_dir,
        main_rows=main_rows,
        ours_name=ours_name,
        adv=adv,
        failures=failures,
    )

    print(f"[done] report_dir: {report_dir}")
    print(f"[done] main metrics: {tables_dir / 'main_metrics.csv'}")
    print(f"[done] advantage table: {tables_dir / 'ours_adv_object_pairwise.csv'}")
    print(f"[done] overlays dir: {fig_dir}")
    if failures:
        print(f"[warn] failures: {report_dir / 'failures.json'}")


if __name__ == "__main__":
    main()
