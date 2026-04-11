#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
MASK_EXTS = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".npy"]


@dataclass(frozen=True)
class Det:
    box: Tuple[float, float, float, float]
    score: float
    label: int


def list_images(image_dir: Path) -> List[Path]:
    return sorted([p for p in image_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS])


def read_image_size(image_path: Path) -> Tuple[int, int]:
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"failed to read image: {image_path}")
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


def parse_yolo_gt_file(label_path: Path, img_w: int, img_h: int) -> List[Det]:
    out: List[Det] = []
    if not label_path.exists():
        return out
    for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        try:
            cls_id = int(float(parts[0]))
            coords = [float(x) for x in parts[1:]]
        except ValueError:
            continue
        if len(coords) > 4 and len(coords) % 2 == 0:
            box = seg_norm_to_xyxy(coords, img_w, img_h)
        else:
            if len(coords) < 4:
                continue
            box = xywhn_to_xyxy(coords[0], coords[1], coords[2], coords[3], img_w, img_h)
        clipped = clip_xyxy(box, img_w, img_h)
        if clipped is None:
            continue
        out.append(Det(box=clipped, score=1.0, label=cls_id))
    return out


def load_ground_truth(dataset_root: Path, split: str) -> Tuple[List[Path], Dict[str, List[Det]], List[int]]:
    image_dir = dataset_root / "images" / split
    label_dir = dataset_root / "labels" / split
    if not image_dir.exists():
        raise FileNotFoundError(f"image dir not found: {image_dir}")
    if not label_dir.exists():
        raise FileNotFoundError(f"label dir not found: {label_dir}")

    image_paths = list_images(image_dir)
    gt: Dict[str, List[Det]] = {}
    classes = set()
    for image_path in image_paths:
        img_w, img_h = read_image_size(image_path)
        label_path = label_dir / f"{image_path.stem}.txt"
        dets = parse_yolo_gt_file(label_path, img_w, img_h)
        gt[str(image_path.resolve())] = dets
        for d in dets:
            classes.add(int(d.label))
    return image_paths, gt, sorted(classes)


def parse_named_path(spec: str, flag_name: str) -> Tuple[str, Path]:
    if "=" not in spec:
        raise ValueError(f"{flag_name} must be in name=path format, got: {spec}")
    name, path = spec.split("=", 1)
    name = name.strip()
    path = path.strip()
    if not name or not path:
        raise ValueError(f"{flag_name} must be in name=path format, got: {spec}")
    return name, Path(path).expanduser().resolve()


def parse_boxes(raw) -> List[Tuple[float, float, float, float]]:
    out: List[Tuple[float, float, float, float]] = []
    if not isinstance(raw, list):
        return out
    for item in raw:
        if isinstance(item, dict):
            vals = [item.get("x1"), item.get("y1"), item.get("x2"), item.get("y2")]
            if any(v is None for v in vals):
                continue
            box = (float(vals[0]), float(vals[1]), float(vals[2]), float(vals[3]))
            out.append(box)
        elif isinstance(item, (list, tuple)) and len(item) >= 4:
            out.append((float(item[0]), float(item[1]), float(item[2]), float(item[3])))
    return out


def load_pred_json(
    pred_json: Path,
    known_image_paths: Sequence[Path],
) -> Dict[str, List[Det]]:
    known_abs = [str(p.resolve()) for p in known_image_paths]
    known_set = set(known_abs)
    by_name: Dict[str, str] = {}
    by_stem: Dict[str, str] = {}
    for p in known_image_paths:
        abs_p = str(p.resolve())
        by_name.setdefault(p.name, abs_p)
        by_stem.setdefault(p.stem, abs_p)

    raw = json.loads(pred_json.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        if isinstance(raw.get("predictions"), list):
            rows = raw["predictions"]
        elif isinstance(raw.get("results"), list):
            rows = raw["results"]
        else:
            rows = []
    elif isinstance(raw, list):
        rows = raw
    else:
        rows = []

    out: Dict[str, List[Det]] = {str(p.resolve()): [] for p in known_image_paths}
    unresolved = 0
    for row in rows:
        if not isinstance(row, dict):
            continue
        key = (
            row.get("image_path")
            or row.get("img_path")
            or row.get("file_name")
            or row.get("image")
            or row.get("image_id")
        )
        if key is None:
            unresolved += 1
            continue
        key_s = str(key)
        resolved_path: Optional[str] = None
        key_path = Path(key_s)
        if key_s in known_set:
            resolved_path = key_s
        elif key_path.is_absolute() and str(key_path.resolve()) in known_set:
            resolved_path = str(key_path.resolve())
        elif key_path.name in by_name:
            resolved_path = by_name[key_path.name]
        elif key_path.stem in by_stem:
            resolved_path = by_stem[key_path.stem]

        if resolved_path is None:
            unresolved += 1
            continue

        boxes = parse_boxes(row.get("boxes", []))
        scores_raw = row.get("scores", [])
        labels_raw = row.get("labels", [])
        if not isinstance(scores_raw, list):
            scores_raw = []
        if not isinstance(labels_raw, list):
            labels_raw = []

        dets: List[Det] = []
        for i, box in enumerate(boxes):
            score = float(scores_raw[i]) if i < len(scores_raw) else 1.0
            label = int(labels_raw[i]) if i < len(labels_raw) else 0
            dets.append(Det(box=box, score=score, label=label))
        out[resolved_path] = dets

    if unresolved > 0:
        print(f"[warn] {pred_json} has {unresolved} rows that could not be matched to split images.")
    return out


def resolve_mask_path(mask_dir: Path, image_path: Path, split: str) -> Optional[Path]:
    stem = image_path.stem
    candidates = []
    candidates.append(mask_dir / image_path.name)
    candidates.append(mask_dir / split / image_path.name)
    for ext in MASK_EXTS:
        candidates.append(mask_dir / f"{stem}{ext}")
        candidates.append(mask_dir / split / f"{stem}{ext}")
    for c in candidates:
        if c.exists() and c.is_file():
            return c
    return None


def mask_to_binary(mask: np.ndarray, thr: float) -> np.ndarray:
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    if mask.dtype == np.bool_:
        return mask.astype(np.uint8)
    mask_f = mask.astype(np.float32)
    max_v = float(mask_f.max()) if mask_f.size > 0 else 0.0
    if max_v <= 1.0:
        binary = mask_f >= float(thr)
    else:
        binary = mask_f >= float(thr) * 255.0
    return binary.astype(np.uint8)


def mask_to_dets(mask: np.ndarray, min_area: int, score: float, label: int) -> List[Det]:
    binary = (mask > 0).astype(np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    dets: List[Det] = []
    for i in range(1, int(n)):
        x, y, w, h, area = stats[i]
        if int(area) < int(min_area):
            continue
        dets.append(Det(box=(float(x), float(y), float(x + w), float(y + h)), score=float(score), label=int(label)))
    return dets


def load_mask_preds(
    mask_dir: Path,
    image_paths: Sequence[Path],
    split: str,
    mask_thr: float,
    min_mask_area: int,
    score: float,
    label: int,
) -> Dict[str, List[Det]]:
    out: Dict[str, List[Det]] = {}
    miss = 0
    for image_path in image_paths:
        image_key = str(image_path.resolve())
        mask_path = resolve_mask_path(mask_dir, image_path, split)
        if mask_path is None:
            out[image_key] = []
            miss += 1
            continue
        if mask_path.suffix.lower() == ".npy":
            mask = np.load(mask_path)
        else:
            mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            out[image_key] = []
            continue
        binary = mask_to_binary(mask, mask_thr)
        out[image_key] = mask_to_dets(binary, min_area=min_mask_area, score=score, label=label)
    if miss > 0:
        print(f"[warn] {mask_dir} missing masks for {miss} images in split.")
    return out


def iou_one_to_many(box: Tuple[float, float, float, float], boxes: np.ndarray) -> np.ndarray:
    if boxes.size == 0:
        return np.zeros((0,), dtype=np.float32)
    b = np.asarray(box, dtype=np.float32)
    ix1 = np.maximum(b[0], boxes[:, 0])
    iy1 = np.maximum(b[1], boxes[:, 1])
    ix2 = np.minimum(b[2], boxes[:, 2])
    iy2 = np.minimum(b[3], boxes[:, 3])
    iw = np.maximum(0.0, ix2 - ix1)
    ih = np.maximum(0.0, iy2 - iy1)
    inter = iw * ih
    area_b = max(0.0, (b[2] - b[0]) * (b[3] - b[1]))
    area_boxes = np.maximum(0.0, (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]))
    union = area_b + area_boxes - inter
    return np.where(union > 0, inter / union, 0.0).astype(np.float32)


def collect_classes(maps: Iterable[Dict[str, List[Det]]]) -> List[int]:
    cls = set()
    for mp in maps:
        for dets in mp.values():
            for d in dets:
                cls.add(int(d.label))
    return sorted(cls)


def match_counts_for_obj_pr(
    gt_map: Dict[str, List[Det]],
    pred_map: Dict[str, List[Det]],
    classes: Sequence[int],
    iou_thr: float,
    score_thr: float,
) -> Tuple[int, int, int]:
    tp = 0
    fp = 0
    fn = 0

    all_images = sorted(set(gt_map.keys()) | set(pred_map.keys()))
    for image_key in all_images:
        gt_dets = gt_map.get(image_key, [])
        pred_dets = [p for p in pred_map.get(image_key, []) if p.score >= score_thr]
        for cls in classes:
            g_cls = [d for d in gt_dets if d.label == cls]
            p_cls = sorted([d for d in pred_dets if d.label == cls], key=lambda d: d.score, reverse=True)
            gt_boxes = np.asarray([d.box for d in g_cls], dtype=np.float32).reshape(-1, 4) if g_cls else np.zeros((0, 4), dtype=np.float32)
            matched = np.zeros((gt_boxes.shape[0],), dtype=bool)

            for pred in p_cls:
                if gt_boxes.shape[0] == 0:
                    fp += 1
                    continue
                ious = iou_one_to_many(pred.box, gt_boxes)
                best = int(np.argmax(ious))
                if ious[best] >= iou_thr and not matched[best]:
                    matched[best] = True
                    tp += 1
                else:
                    fp += 1
            fn += int((~matched).sum())

    return tp, fp, fn


def compute_ap_101(
    gt_map: Dict[str, List[Det]],
    pred_map: Dict[str, List[Det]],
    cls_id: int,
    iou_thr: float,
) -> float:
    gt_by_img = {}
    npos = 0
    for image_key, dets in gt_map.items():
        boxes = [d.box for d in dets if d.label == cls_id]
        arr = np.asarray(boxes, dtype=np.float32).reshape(-1, 4) if boxes else np.zeros((0, 4), dtype=np.float32)
        gt_by_img[image_key] = arr
        npos += int(arr.shape[0])
    if npos == 0:
        return float("nan")

    pred_rows = []
    for image_key, dets in pred_map.items():
        for d in dets:
            if d.label == cls_id:
                pred_rows.append((image_key, float(d.score), d.box))
    pred_rows.sort(key=lambda x: x[1], reverse=True)
    if len(pred_rows) == 0:
        return 0.0

    matched = {k: np.zeros((v.shape[0],), dtype=bool) for k, v in gt_by_img.items()}
    tp = np.zeros((len(pred_rows),), dtype=np.float32)
    fp = np.zeros((len(pred_rows),), dtype=np.float32)

    for i, (image_key, _, box) in enumerate(pred_rows):
        gt_boxes = gt_by_img.get(image_key, np.zeros((0, 4), dtype=np.float32))
        if gt_boxes.shape[0] == 0:
            fp[i] = 1.0
            continue
        ious = iou_one_to_many(box, gt_boxes)
        best = int(np.argmax(ious))
        if ious[best] >= iou_thr and not matched[image_key][best]:
            tp[i] = 1.0
            matched[image_key][best] = True
        else:
            fp[i] = 1.0

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recall = tp_cum / max(float(npos), 1e-12)
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)

    recall_levels = np.linspace(0.0, 1.0, 101, dtype=np.float32)
    p_interp = []
    for r in recall_levels:
        p = precision[recall >= r]
        p_interp.append(float(np.max(p)) if p.size > 0 else 0.0)
    return float(np.mean(p_interp))


def compute_metrics(
    gt_map: Dict[str, List[Det]],
    pred_map: Dict[str, List[Det]],
    classes: Sequence[int],
    score_thr: float,
    obj_iou: float,
) -> Dict[str, float]:
    tp, fp, fn = match_counts_for_obj_pr(
        gt_map=gt_map,
        pred_map=pred_map,
        classes=classes,
        iou_thr=float(obj_iou),
        score_thr=float(score_thr),
    )
    precision = float(tp) / float(tp + fp) if (tp + fp) > 0 else 0.0
    recall = float(tp) / float(tp + fn) if (tp + fn) > 0 else 0.0

    iou_thresholds = [0.50 + 0.05 * i for i in range(10)]
    map_by_iou = []
    for thr in iou_thresholds:
        aps = []
        for cls_id in classes:
            ap = compute_ap_101(gt_map=gt_map, pred_map=pred_map, cls_id=int(cls_id), iou_thr=float(thr))
            if not np.isnan(ap):
                aps.append(ap)
        map_by_iou.append(float(np.mean(aps)) if aps else 0.0)

    return {
        "object_precision": precision,
        "object_recall": recall,
        "map50": map_by_iou[0] if map_by_iou else 0.0,
        "map50_95": float(np.mean(map_by_iou)) if map_by_iou else 0.0,
    }


def compute_image_level_metrics(
    gt_map: Dict[str, List[Det]],
    pred_map: Dict[str, List[Det]],
    score_thr: float,
) -> Dict[str, float]:
    hit_img = 0
    miss_img = 0
    fp_img = 0
    tn_img = 0

    all_images = sorted(set(gt_map.keys()) | set(pred_map.keys()))
    for image_key in all_images:
        gt_dets = gt_map.get(image_key, [])
        pred_dets = [p for p in pred_map.get(image_key, []) if p.score >= float(score_thr)]
        has_gt = len(gt_dets) > 0
        has_pred = len(pred_dets) > 0

        if has_gt and has_pred:
            hit_img += 1
        elif has_gt and (not has_pred):
            miss_img += 1
        elif (not has_gt) and has_pred:
            fp_img += 1
        else:
            tn_img += 1

    image_precision = float(hit_img) / float(hit_img + fp_img) if (hit_img + fp_img) > 0 else 0.0
    image_recall = float(hit_img) / float(hit_img + miss_img) if (hit_img + miss_img) > 0 else 0.0

    return {
        "image_precision": float(image_precision),
        "image_recall": float(image_recall),
    }


def print_table(rows: List[dict]) -> None:
    header = [
        "model",
        "mAP@0.5",
        "mAP@0.5:0.95",
        "obj_precision",
        "obj_recall",
        "img_precision",
        "img_recall",
    ]
    print(",".join(header))
    for r in rows:
        print(
            ",".join(
                [
                    str(r["model"]),
                    f"{r['map50']:.6f}",
                    f"{r['map50_95']:.6f}",
                    f"{r['object_precision']:.6f}",
                    f"{r['object_recall']:.6f}",
                    f"{r['image_precision']:.6f}",
                    f"{r['image_recall']:.6f}",
                ]
            )
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Unified detection benchmark (boxes + mask->box).")
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    p.add_argument("--pred-json", type=str, action="append", default=[], help="name=prediction_json, repeatable.")
    p.add_argument("--mask-dir", type=str, action="append", default=[], help="name=mask_dir, repeatable.")
    p.add_argument("--mask-thr", type=float, default=0.5, help="Binary threshold for mask predictions.")
    p.add_argument("--min-mask-area", type=int, default=10, help="Min connected-component area in mask.")
    p.add_argument("--mask-score", type=float, default=1.0, help="Score assigned to each mask-derived box.")
    p.add_argument("--mask-label", type=int, default=0, help="Class id assigned to mask-derived boxes.")
    p.add_argument("--obj-iou", type=float, default=0.5, help="IoU threshold for object precision/recall.")
    p.add_argument("--score-thr", type=float, default=0.25, help="Score threshold for object precision/recall.")
    p.add_argument(
        "--score-thr-list",
        type=float,
        nargs="+",
        default=[],
        help="Run a sweep on multiple score thresholds, e.g. --score-thr-list 0.2 0.3 0.4 0.5",
    )
    p.add_argument("--print-only", action="store_true", help="Only print results; do not save CSV/JSON.")
    p.add_argument("--out-csv", type=Path, default=None)
    p.add_argument("--out-json", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    image_paths, gt_map, gt_classes = load_ground_truth(dataset_root=dataset_root, split=args.split)

    models: List[Tuple[str, Dict[str, List[Det]]]] = []
    for spec in args.pred_json:
        name, path = parse_named_path(spec, "--pred-json")
        pred_map = load_pred_json(path, known_image_paths=image_paths)
        models.append((name, pred_map))
    for spec in args.mask_dir:
        name, path = parse_named_path(spec, "--mask-dir")
        pred_map = load_mask_preds(
            mask_dir=path,
            image_paths=image_paths,
            split=args.split,
            mask_thr=float(args.mask_thr),
            min_mask_area=int(args.min_mask_area),
            score=float(args.mask_score),
            label=int(args.mask_label),
        )
        models.append((name, pred_map))

    if not models:
        raise RuntimeError("No predictors provided. Use --pred-json and/or --mask-dir.")

    thr_list = [float(x) for x in args.score_thr_list] if args.score_thr_list else [float(args.score_thr)]

    for idx, thr in enumerate(thr_list):
        rows = []
        for model_name, pred_map in models:
            classes = gt_classes if gt_classes else collect_classes([pred_map])
            metrics = compute_metrics(
                gt_map=gt_map,
                pred_map=pred_map,
                classes=classes,
                score_thr=float(thr),
                obj_iou=float(args.obj_iou),
            )
            img_metrics = compute_image_level_metrics(
                gt_map=gt_map,
                pred_map=pred_map,
                score_thr=float(thr),
            )
            row = {"model": model_name, **metrics, **img_metrics}
            rows.append(row)

        if len(thr_list) > 1:
            print(f"[score-thr={thr}]")
        print_table(rows)
        print("note: 掩膜转最小外接框后得到的是检测能力对比，不是分割精度对比。")
        if idx != len(thr_list) - 1:
            print()

        if args.print_only:
            continue

        if args.out_csv is not None:
            out_csv = args.out_csv.resolve()
            if len(thr_list) > 1:
                out_csv = out_csv.with_name(f"{out_csv.stem}_thr{thr:g}{out_csv.suffix}")
            out_csv.parent.mkdir(parents=True, exist_ok=True)
            with out_csv.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "model",
                        "map50",
                        "map50_95",
                        "object_precision",
                        "object_recall",
                        "image_precision",
                        "image_recall",
                    ],
                )
                writer.writeheader()
                for r in rows:
                    writer.writerow(r)
            print(f"[done] csv -> {out_csv}")

        if args.out_json is not None:
            out_json = args.out_json.resolve()
            if len(thr_list) > 1:
                out_json = out_json.with_name(f"{out_json.stem}_thr{thr:g}{out_json.suffix}")
            out_json.parent.mkdir(parents=True, exist_ok=True)
            out_json.write_text(
                json.dumps(
                    {
                        "dataset_root": str(dataset_root),
                        "split": args.split,
                        "obj_iou": float(args.obj_iou),
                        "score_thr": float(thr),
                        "rows": rows,
                        "note": "mask->bbox comparison measures detection ability, not segmentation quality",
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            print(f"[done] json -> {out_json}")


if __name__ == "__main__":
    main()
