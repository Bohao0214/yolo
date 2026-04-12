#!/usr/bin/env python3
"""
Run 3 ablation experiments from a reference checkpoint configuration:
  - a4 only
  - b7 only
  - a4 + b7

It will:
1) parse ref args.yaml from ref weight path
2) generate 3 train configs with same core parameters
3) train 3 runs via src/train.py
4) evaluate and export summary table with metrics:
   mAP@0.5, mAP@0.5:0.95, P_obj, R_obj, R_img, FPR_img

Usage:
python /home/ubuntu/hpproject/yolo/analyze/code/ch4tool/run_ablation_a4_b7_from_ref.py \
  --ref-args /home/ubuntu/hpproject/yolo/experiments/a4b7d6/datasetm6c/defect241__a4__b7__d6/exp_2603060404/train/args.yaml \
  --split test \
  --device 0
"""

from __future__ import annotations

import argparse
import copy
import csv
import datetime as dt
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import yaml
from ultralytics import YOLO

REPO_ROOT = Path("/home/ubuntu/hpproject/yolo")
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

ENHANCE_BOOL_KEYS = [
    "a3",
    "a4",
    "a5",
    "a6",
    "a7",
    "a9",
    "a11",
    "a21",
    "b1",
    "b2",
    "b3",
    "b5",
    "b6",
    "b7",
    "b9",
    "b11",
    "b21",
    "c4",
    "c5",
    "c6",
    "c7",
    "c9",
    "c11",
    "c21",
    "d1",
    "d3",
    "d5",
    "d6",
    "d7",
    "d9",
    "d11",
    "d21",
]

VARIANTS = [
    ("a4_only", ["a4"], "SPD only"),
    ("b7_only", ["b7"], "CARAFE only"),
    ("a4_b7", ["a4", "b7"], "SPD+CARAFE"),
]


@dataclass(frozen=True)
class Det:
    box: Tuple[float, float, float, float]
    score: float
    label: int


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run a4/b7 ablation from reference weight config.")
    p.add_argument(
        "--ref-weight",
        type=Path,
        default=None,
        help="Reference best.pt path (optional if --ref-args is provided).",
    )
    p.add_argument(
        "--ref-args",
        type=Path,
        default=None,
        help="Reference args.yaml path. If set, it has higher priority than --ref-weight.",
    )
    p.add_argument("--a4-weight", type=Path, default=None, help="Existing a4-only weight for eval-only mode.")
    p.add_argument("--b7-weight", type=Path, default=None, help="Existing b7-only weight for eval-only mode.")
    p.add_argument("--a4b7-weight", type=Path, default=None, help="Existing a4+b7 weight for eval-only mode.")
    p.add_argument("--split", type=str, default="test", help="test / val / train / val+test.")
    p.add_argument("--device", type=str, default="", help="Override device, e.g. 0")
    p.add_argument("--epochs", type=int, default=0, help="Override epochs if >0.")
    p.add_argument("--batch", type=int, default=0, help="Override batch if >0.")
    p.add_argument("--workers", type=int, default=0, help="Override workers if >0.")
    p.add_argument("--out-root", type=Path, default=None, help="Output root; default under experiments/ablation_a4_b7.")
    p.add_argument("--skip-train", action="store_true", help="Skip train, only evaluate existing latest runs.")
    return p.parse_args()


def load_yaml(path: Path) -> Dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def dump_yaml(path: Path, data: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")


def resolve_path_maybe_repo(path_like: str, base: Path = REPO_ROOT) -> Path:
    p = Path(path_like).expanduser()
    if p.is_absolute():
        return p.resolve()
    return (base / p).resolve()


def find_args_yaml_from_weight(weight: Path) -> Path:
    w = weight.resolve()
    cands = []
    if w.parent.name == "weights":
        cands.append(w.parent.parent / "args.yaml")
    for i in range(1, 8):
        cands.append(w.parents[i - 1] / "args.yaml")
    for c in cands:
        if c.exists():
            return c
    raise FileNotFoundError(f"args.yaml not found near weight: {weight}")


def infer_data_yaml_and_root(args_cfg: Dict, args_yaml_path: Path) -> Tuple[Path, Path]:
    data_ref = str(args_cfg.get("data", "")).strip()
    if not data_ref:
        raise RuntimeError(f"`data` missing in args yaml: {args_yaml_path}")
    p = Path(data_ref).expanduser()
    if not p.is_absolute():
        p = (args_yaml_path.parent / p).resolve()
    else:
        p = p.resolve()

    if p.is_file():
        data_yaml = p
        data_cfg = load_yaml(data_yaml)
        root = data_cfg.get("path")
        if isinstance(root, str) and root.strip():
            root_p = Path(root).expanduser()
            if not root_p.is_absolute():
                root_p = (data_yaml.parent / root_p).resolve()
            else:
                root_p = root_p.resolve()
            dataset_root = root_p
        else:
            dataset_root = data_yaml.parent
    else:
        dataset_root = p
        data_yaml = dataset_root / "data.yaml"
        if not data_yaml.exists():
            data_yaml = dataset_root / "dataset.yaml"
        if not data_yaml.exists():
            raise FileNotFoundError(f"data yaml not found under dataset root: {dataset_root}")

    return data_yaml.resolve(), dataset_root.resolve()


def parse_split_spec(spec: str) -> List[str]:
    s = (spec or "test").strip()
    parts = [x.strip() for x in s.replace(",", "+").split("+") if x.strip()]
    out = []
    seen = set()
    for x in parts:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out if out else ["test"]


def list_images(image_dir: Path) -> List[Path]:
    if not image_dir.exists():
        return []
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


def load_ground_truth(dataset_root: Path, splits: Sequence[str]) -> Tuple[List[Path], Dict[str, List[Det]], List[int]]:
    all_images: List[Path] = []
    gt: Dict[str, List[Det]] = {}
    classes = set()
    for sp in splits:
        image_dir = dataset_root / "images" / sp
        label_dir = dataset_root / "labels" / sp
        imgs = list_images(image_dir)
        for image_path in imgs:
            img_w, img_h = read_image_size(image_path)
            label_path = label_dir / f"{image_path.stem}.txt"
            dets = parse_yolo_gt_file(label_path, img_w, img_h)
            key = str(image_path.resolve())
            gt[key] = dets
            all_images.append(image_path.resolve())
            for d in dets:
                classes.add(int(d.label))
    uniq_images = sorted({p.resolve() for p in all_images})
    return uniq_images, gt, sorted(classes)


def iou_one_to_many(box: Sequence[float], boxes: np.ndarray) -> np.ndarray:
    if boxes.size == 0:
        return np.zeros((0,), dtype=np.float32)
    x1 = np.maximum(float(box[0]), boxes[:, 0])
    y1 = np.maximum(float(box[1]), boxes[:, 1])
    x2 = np.minimum(float(box[2]), boxes[:, 2])
    y2 = np.minimum(float(box[3]), boxes[:, 3])
    inter_w = np.clip(x2 - x1, a_min=0.0, a_max=None)
    inter_h = np.clip(y2 - y1, a_min=0.0, a_max=None)
    inter = inter_w * inter_h
    area_a = max(float(box[2] - box[0]), 0.0) * max(float(box[3] - box[1]), 0.0)
    area_b = np.clip(boxes[:, 2] - boxes[:, 0], a_min=0.0, a_max=None) * np.clip(
        boxes[:, 3] - boxes[:, 1], a_min=0.0, a_max=None
    )
    union = area_a + area_b - inter
    iou = np.divide(inter, np.maximum(union, 1e-12), out=np.zeros_like(inter), where=union > 0)
    return iou.astype(np.float32)


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
            gt_boxes = (
                np.asarray([d.box for d in g_cls], dtype=np.float32).reshape(-1, 4)
                if g_cls
                else np.zeros((0, 4), dtype=np.float32)
            )
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


def compute_ap_101(gt_map: Dict[str, List[Det]], pred_map: Dict[str, List[Det]], cls_id: int, iou_thr: float) -> float:
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


def compute_obj_map_metrics(
    gt_map: Dict[str, List[Det]],
    pred_map: Dict[str, List[Det]],
    classes: Sequence[int],
    score_thr: float,
    obj_iou: float,
) -> Dict[str, float]:
    tp, fp, fn = match_counts_for_obj_pr(gt_map, pred_map, classes, obj_iou, score_thr)
    p = float(tp) / float(tp + fp) if (tp + fp) > 0 else 0.0
    r = float(tp) / float(tp + fn) if (tp + fn) > 0 else 0.0

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
        "obj_precision": p,
        "obj_recall": r,
        "map50": map_by_iou[0] if map_by_iou else 0.0,
        "map50_95": float(np.mean(map_by_iou)) if map_by_iou else 0.0,
    }


def compute_image_recall_fpr(gt_map: Dict[str, List[Det]], pred_map: Dict[str, List[Det]], score_thr: float) -> Dict[str, float]:
    hit = miss = fp = tn = 0
    all_images = sorted(set(gt_map.keys()) | set(pred_map.keys()))
    for k in all_images:
        has_gt = len(gt_map.get(k, [])) > 0
        has_pred = any(d.score >= float(score_thr) for d in pred_map.get(k, []))
        if has_gt and has_pred:
            hit += 1
        elif has_gt and (not has_pred):
            miss += 1
        elif (not has_gt) and has_pred:
            fp += 1
        else:
            tn += 1
    recall = float(hit) / float(hit + miss) if (hit + miss) > 0 else 0.0
    fpr = float(fp) / float(fp + tn) if (fp + tn) > 0 else 0.0
    return {
        "img_recall": recall,
        "img_fpr": fpr,
        "hit_img": hit,
        "miss_img": miss,
        "fp_img": fp,
        "tn_img": tn,
    }


def chunks(items: Sequence[Path], n: int) -> Iterable[Sequence[Path]]:
    n = max(1, int(n))
    for i in range(0, len(items), n):
        yield items[i : i + n]


def infer_pred_map(
    weight_path: Path,
    image_paths: Sequence[Path],
    *,
    imgsz: int,
    nms_iou: float,
    score_floor: float,
    raw_max_det: int,
    batch: int,
    device: str,
) -> Dict[str, List[Det]]:
    model = YOLO(str(weight_path))
    pred_map: Dict[str, List[Det]] = {str(p.resolve()): [] for p in image_paths}
    for chunk_paths in chunks(list(image_paths), batch):
        kwargs = dict(
            source=[str(p) for p in chunk_paths],
            imgsz=int(imgsz),
            conf=float(score_floor),
            iou=float(nms_iou),
            max_det=int(raw_max_det),
            batch=int(batch),
            save=False,
            verbose=False,
            stream=True,
        )
        if device:
            kwargs["device"] = str(device)
        stream = model.predict(**kwargs)
        for res in stream:
            img_path = str(Path(res.path).resolve())
            boxes_obj = res.boxes
            dets: List[Det] = []
            if boxes_obj is not None and boxes_obj.xyxy is not None:
                boxes = boxes_obj.xyxy.detach().cpu().numpy().tolist()
                scores = boxes_obj.conf.detach().cpu().numpy().tolist() if boxes_obj.conf is not None else [1.0] * len(boxes)
                labels = boxes_obj.cls.detach().cpu().numpy().tolist() if boxes_obj.cls is not None else [0] * len(boxes)
                for b, s, c in zip(boxes, scores, labels):
                    dets.append(Det(box=(float(b[0]), float(b[1]), float(b[2]), float(b[3])), score=float(s), label=int(c)))
            pred_map[img_path] = dets
    return pred_map


def disable_all_enhance_flags(cfg: Dict) -> Dict:
    enh = copy.deepcopy(cfg.get("enhance241", {}))
    if not isinstance(enh, dict):
        enh = {}
    for k in ENHANCE_BOOL_KEYS:
        enh[k] = False
    return enh


def prepare_variant_config(
    ref_cfg: Dict,
    *,
    variant_keys: Sequence[str],
    variant_tag: str,
    out_cfg_path: Path,
    dataset_tag: str,
    args: argparse.Namespace,
) -> Dict:
    cfg = copy.deepcopy(ref_cfg)
    cfg["project_root"] = str(REPO_ROOT)
    cfg["run_name"] = ""
    cfg["resume"] = False
    cfg["weights"] = ""
    cfg["skip_post_eval_metrics"] = True
    cfg["save_test_pic"] = False
    cfg["save_val_pic"] = False
    cfg["save_train_pic"] = False
    cfg["exp_name"] = f"ablation_a4b7/{dataset_tag}/{variant_tag}"

    if args.epochs > 0:
        cfg["epochs"] = int(args.epochs)
    if args.batch > 0:
        cfg["batch"] = int(args.batch)
        cfg["eval_batch"] = int(args.batch)
    if args.workers > 0:
        cfg["workers"] = int(args.workers)
    if args.device.strip():
        cfg["device"] = str(args.device).strip()
        cfg["eval_device"] = str(args.device).strip()

    enh = disable_all_enhance_flags(cfg)
    for k in variant_keys:
        enh[k] = True
    if enh.get("d1", False):
        enh["d3"] = True
    cfg["enhance241"] = enh

    dump_yaml(out_cfg_path, cfg)
    return cfg


def resolve_latest_exp_dir(exp_name: str) -> Path:
    exp_root = (REPO_ROOT / "experiments" / Path(exp_name)).resolve()
    if not exp_root.exists():
        raise FileNotFoundError(f"exp_root not found: {exp_root}")
    cand = sorted([p for p in exp_root.glob("exp_*") if p.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True)
    if not cand:
        raise FileNotFoundError(f"No exp_* dirs under: {exp_root}")
    return cand[0].resolve()


def run_train(cfg_path: Path, log_path: Path) -> None:
    cmd = [sys.executable, str(REPO_ROOT / "src/train.py"), "--config", str(cfg_path)]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as f:
        proc = subprocess.run(cmd, cwd=str(REPO_ROOT), stdout=f, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"train failed: cfg={cfg_path}, log={log_path}, code={proc.returncode}")


def format_pct(x: float) -> str:
    return f"{x * 100.0:.2f}"


def resolve_optional_weight(p: Optional[Path]) -> Optional[Path]:
    if p is None:
        return None
    return resolve_path_maybe_repo(str(p))


def main() -> None:
    args = parse_args()
    ref_weight: Optional[Path] = None
    if args.ref_args is not None:
        args_yaml_path = resolve_path_maybe_repo(str(args.ref_args))
        if not args_yaml_path.exists():
            raise FileNotFoundError(f"ref args yaml not found: {args_yaml_path}")
        if args_yaml_path.is_dir():
            args_yaml_path = (args_yaml_path / "args.yaml").resolve()
        if not args_yaml_path.exists():
            raise FileNotFoundError(f"args.yaml not found: {args_yaml_path}")
    else:
        if args.ref_weight is None:
            raise RuntimeError("require one of: --ref-args or --ref-weight")
        ref_weight = resolve_path_maybe_repo(str(args.ref_weight))
        if not ref_weight.exists():
            raise FileNotFoundError(f"ref weight not found: {ref_weight}")
        args_yaml_path = find_args_yaml_from_weight(ref_weight)

    ref_args_cfg = load_yaml(args_yaml_path)
    data_yaml, dataset_root = infer_data_yaml_and_root(ref_args_cfg, args_yaml_path)
    dataset_tag = dataset_root.name
    split_list = parse_split_spec(args.split)

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M")
    out_root = (
        args.out_root.expanduser().resolve()
        if args.out_root is not None
        else (REPO_ROOT / "experiments" / "ablation_a4_b7" / dataset_tag / f"run_{ts}").resolve()
    )
    cfg_dir = out_root / "configs"
    log_dir = out_root / "logs"
    table_dir = out_root / "tables"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    # eval params (match your existing evaluation口径)
    conf = float(ref_args_cfg.get("metric_conf", ref_args_cfg.get("conf", 0.25)))
    nms_iou = float(ref_args_cfg.get("nms_iou", ref_args_cfg.get("iou", 0.7)))
    tp_iou = float(ref_args_cfg.get("tp_iou", 0.3))
    score_floor = float(ref_args_cfg.get("score_floor", 0.01))
    raw_max_det = int(ref_args_cfg.get("raw_max_det", 3000))
    imgsz = int(ref_args_cfg.get("imgsz", 640))
    eval_batch = int(ref_args_cfg.get("eval_batch", ref_args_cfg.get("batch", 4)))
    eval_device = str(args.device).strip() or str(ref_args_cfg.get("eval_device", ref_args_cfg.get("device", "0")))

    image_paths, gt_map, gt_classes = load_ground_truth(dataset_root=dataset_root, splits=split_list)
    if not gt_classes:
        gt_classes = [0]

    summary_rows: List[Dict] = []
    run_manifest: Dict = {
        "timestamp": ts,
        "ref_weight": str(ref_weight) if ref_weight is not None else "",
        "args_yaml": str(args_yaml_path),
        "data_yaml": str(data_yaml),
        "dataset_root": str(dataset_root),
        "split": "+".join(split_list),
        "eval_params": {
            "conf": conf,
            "nms_iou": nms_iou,
            "tp_iou": tp_iou,
            "score_floor": score_floor,
            "raw_max_det": raw_max_det,
            "imgsz": imgsz,
            "eval_batch": eval_batch,
            "device": eval_device,
        },
        "variants": [],
    }

    a4_w = resolve_optional_weight(args.a4_weight)
    b7_w = resolve_optional_weight(args.b7_weight)
    a4b7_w = resolve_optional_weight(args.a4b7_weight)
    eval_only_mode = any(x is not None for x in [a4_w, b7_w, a4b7_w])
    if eval_only_mode and not (a4_w and b7_w and a4b7_w):
        raise RuntimeError("eval-only mode requires all three: --a4-weight --b7-weight --a4b7-weight")

    explicit_variant_weights = {
        "a4_only": a4_w,
        "b7_only": b7_w,
        "a4_b7": a4b7_w,
    }

    for variant_tag, keys, method_name in VARIANTS:
        cfg_path = cfg_dir / f"{variant_tag}.yaml"
        log_path = log_dir / f"{variant_tag}.log"
        exp_dir: Optional[Path] = None

        if eval_only_mode:
            use_w = explicit_variant_weights[variant_tag]
            if use_w is None:
                raise RuntimeError(f"missing weight for variant: {variant_tag}")
            if not use_w.exists():
                raise FileNotFoundError(f"variant weight not found: {use_w}")
            cfg = copy.deepcopy(ref_args_cfg)
        else:
            cfg = prepare_variant_config(
                ref_cfg=ref_args_cfg,
                variant_keys=keys,
                variant_tag=variant_tag,
                out_cfg_path=cfg_path,
                dataset_tag=dataset_tag,
                args=args,
            )
            if not args.skip_train:
                run_train(cfg_path=cfg_path, log_path=log_path)

            exp_dir = resolve_latest_exp_dir(str(cfg["exp_name"]))
            best_w = exp_dir / "train" / "weights" / "best.pt"
            last_w = exp_dir / "train" / "weights" / "last.pt"
            use_w = best_w if best_w.exists() else last_w
            if not use_w.exists():
                raise FileNotFoundError(f"no best/last weight found in {exp_dir}")

        pred_map = infer_pred_map(
            weight_path=use_w,
            image_paths=image_paths,
            imgsz=imgsz,
            nms_iou=nms_iou,
            score_floor=score_floor,
            raw_max_det=raw_max_det,
            batch=eval_batch,
            device=eval_device,
        )
        obj_map = compute_obj_map_metrics(
            gt_map=gt_map,
            pred_map=pred_map,
            classes=gt_classes,
            score_thr=conf,
            obj_iou=tp_iou,
        )
        img_m = compute_image_recall_fpr(gt_map=gt_map, pred_map=pred_map, score_thr=conf)

        row = {
            "method": method_name,
            "variant": variant_tag,
            "mAP@0.5": f"{obj_map['map50']:.6f}",
            "mAP@0.5:0.95": f"{obj_map['map50_95']:.6f}",
            "P_obj": f"{obj_map['obj_precision']:.6f}",
            "R_obj": f"{obj_map['obj_recall']:.6f}",
            "R_img": f"{img_m['img_recall']:.6f}",
            "FPR_img": f"{img_m['img_fpr']:.6f}",
            "mAP@0.5(%)": format_pct(obj_map["map50"]),
            "mAP@0.5:0.95(%)": format_pct(obj_map["map50_95"]),
            "P_obj(%)": format_pct(obj_map["obj_precision"]),
            "R_obj(%)": format_pct(obj_map["obj_recall"]),
            "R_img(%)": format_pct(img_m["img_recall"]),
            "FPR_img(%)": format_pct(img_m["img_fpr"]),
            "weight": str(use_w),
            "exp_dir": str(exp_dir) if exp_dir is not None else "",
        }
        summary_rows.append(row)
        run_manifest["variants"].append(
            {
                "variant": variant_tag,
                "keys": keys,
                "method_name": method_name,
                "config": str(cfg_path),
                "log": str(log_path),
                "exp_dir": str(exp_dir) if exp_dir is not None else "",
                "weight": str(use_w),
                "eval_only": bool(eval_only_mode),
            }
        )

    # outputs
    csv_path = table_dir / "ablation_a4_b7_summary.csv"
    md_path = table_dir / "ablation_a4_b7_summary.md"
    tex_path = table_dir / "ablation_a4_b7_summary.tex"
    manifest_path = out_root / "run_manifest.json"

    fieldnames = [
        "method",
        "variant",
        "mAP@0.5",
        "mAP@0.5:0.95",
        "P_obj",
        "R_obj",
        "R_img",
        "FPR_img",
        "mAP@0.5(%)",
        "mAP@0.5:0.95(%)",
        "P_obj(%)",
        "R_obj(%)",
        "R_img(%)",
        "FPR_img(%)",
        "weight",
        "exp_dir",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in summary_rows:
            writer.writerow(r)

    md_lines = [
        "# Ablation Summary (a4 / b7 / a4+b7)",
        "",
        "| 方法 | mAP@0.5 | mAP@0.5:0.95 | P_obj/% | R_obj/% | R_img/% | FPR_img/% |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in summary_rows:
        md_lines.append(
            f"| {r['method']} | {r['mAP@0.5(%)']} | {r['mAP@0.5:0.95(%)']} | "
            f"{r['P_obj(%)']} | {r['R_obj(%)']} | {r['R_img(%)']} | {r['FPR_img(%)']} |"
        )
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    tex_lines = [
        "\\begin{tabular}{lcccccc}",
        "\\toprule",
        "方法 & $mAP@0.5$ & $mAP@0.5:0.95$ & $P_{\\mathrm{obj}}$/\\% & $R_{\\mathrm{obj}}$/\\% & $R_{\\mathrm{img}}$/\\% & $FPR_{\\mathrm{img}}$/\\% \\\\",
        "\\midrule",
    ]
    for r in summary_rows:
        tex_lines.append(
            f"{r['method']} & {r['mAP@0.5(%)']} & {r['mAP@0.5:0.95(%)']} & "
            f"{r['P_obj(%)']} & {r['R_obj(%)']} & {r['R_img(%)']} & {r['FPR_img(%)']} \\\\"
        )
    tex_lines += ["\\bottomrule", "\\end{tabular}", ""]
    tex_path.write_text("\n".join(tex_lines), encoding="utf-8")

    manifest_path.write_text(json.dumps(run_manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[done] out_root: {out_root}")
    print(f"[done] csv: {csv_path}")
    print(f"[done] md : {md_path}")
    print(f"[done] tex: {tex_path}")
    print(f"[done] manifest: {manifest_path}")


if __name__ == "__main__":
    main()
