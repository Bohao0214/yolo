#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gc
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np
import torch
import yaml
from ultralytics import YOLO

from src.eval import list_source_images


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Evaluate one YOLO weight and export metrics + FP/FN visuals")
    p.add_argument("--weights", type=str, required=True, help="Path to best.pt")
    p.add_argument("--data-yaml", type=str, required=True, help="Path to data.yaml")
    p.add_argument("--splits", type=str, default="test,val", help="Comma-separated splits, e.g. test,val")
    p.add_argument("--conf", type=float, default=0.25, help="Final confidence threshold")
    p.add_argument("--metric-conf", type=float, default=0.01, help="Raw predict conf before final threshold")
    p.add_argument("--nms-iou", type=float, default=0.7, help="NMS IoU in predict/val")
    p.add_argument("--tp-iou", type=float, default=0.3, help="GT-Pred IoU threshold for TP")
    p.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    p.add_argument("--batch", type=int, default=2, help="Inference/val batch")
    p.add_argument("--device", type=str, default="0", help="Device, e.g. 0 or cpu")
    p.add_argument("--max-det", type=int, default=100, help="max_det")
    p.add_argument("--save-fn-fp-only", action="store_true", help="Only save image-level FN/FP overlays")
    return p.parse_args()


def safe_div(a: int, b: int) -> float:
    return float(a) / float(b) if b > 0 else 0.0


def resolve_data_entry(entry: str, data_yaml_path: Path, data_cfg: Dict) -> Path:
    p = Path(str(entry))
    if p.is_absolute():
        return p
    base = Path(str(data_cfg.get("path", "")).strip()) if str(data_cfg.get("path", "")).strip() else data_yaml_path.parent
    return (base / p).resolve()


def infer_label_path_from_image(image_path: Path) -> Path:
    parts = list(image_path.parts)
    if "images" in parts:
        idx = parts.index("images")
        parts[idx] = "labels"
        return Path(*parts).with_suffix(".txt")
    if "image" in parts:
        idx = parts.index("image")
        parts[idx] = "labels"
        return Path(*parts).with_suffix(".txt")
    return image_path.with_suffix(".txt")


def count_gt_by_class_from_source(source: Path, n_cls: int) -> np.ndarray:
    cnt = np.zeros((n_cls,), dtype=np.int64)
    images = list_source_images(source)
    for im in images:
        lp = infer_label_path_from_image(im)
        if not lp.exists():
            continue
        for line in lp.read_text(encoding="utf-8", errors="ignore").splitlines():
            ss = line.strip().split()
            if len(ss) < 1:
                continue
            try:
                cid = int(float(ss[0]))
            except Exception:
                continue
            if 0 <= cid < n_cls:
                cnt[cid] += 1
    return cnt


def yolo_to_xyxy(xc: float, yc: float, w: float, h: float, im_w: int, im_h: int) -> List[float]:
    x1 = (xc - w / 2.0) * im_w
    y1 = (yc - h / 2.0) * im_h
    x2 = (xc + w / 2.0) * im_w
    y2 = (yc + h / 2.0) * im_h
    return [x1, y1, x2, y2]


def load_gt_xyxy_cls(image_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    label_path = image_path.with_suffix(".txt")
    parts = list(image_path.parts)
    if "images" in parts:
        idx = parts.index("images")
        parts[idx] = "labels"
        label_path = Path(*parts).with_suffix(".txt")
    if "image" in parts and not label_path.exists():
        idx = parts.index("image")
        parts[idx] = "labels"
        label_path = Path(*parts).with_suffix(".txt")

    if not label_path.exists():
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int32)

    img = cv2.imread(str(image_path))
    if img is None:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int32)
    h, w = img.shape[:2]

    boxes: List[List[float]] = []
    clss: List[int] = []
    for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        ss = line.strip().split()
        if len(ss) < 5:
            continue
        try:
            cls_id = int(float(ss[0]))
            xc, yc, bw, bh = map(float, ss[1:5])
        except Exception:
            continue
        boxes.append(yolo_to_xyxy(xc, yc, bw, bh, w, h))
        clss.append(cls_id)
    if not boxes:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int32)
    return np.asarray(boxes, dtype=np.float32), np.asarray(clss, dtype=np.int32)


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)
    ax1, ay1, ax2, ay2 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]

    ix1 = np.maximum(ax1[:, None], bx1[None, :])
    iy1 = np.maximum(ay1[:, None], by1[None, :])
    ix2 = np.minimum(ax2[:, None], bx2[None, :])
    iy2 = np.minimum(ay2[:, None], by2[None, :])
    iw = np.maximum(0.0, ix2 - ix1)
    ih = np.maximum(0.0, iy2 - iy1)
    inter = iw * ih

    area_a = np.maximum(0.0, (ax2 - ax1) * (ay2 - ay1))
    area_b = np.maximum(0.0, (bx2 - bx1) * (by2 - by1))
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / union, 0.0).astype(np.float32)


def match_one_to_one_class_aware(
    gt_boxes: np.ndarray,
    gt_cls: np.ndarray,
    pred_boxes: np.ndarray,
    pred_cls: np.ndarray,
    pred_scores: np.ndarray,
    tp_iou: float,
) -> Tuple[int, int, int]:
    if gt_boxes.shape[0] == 0:
        return 0, int(pred_boxes.shape[0]), 0
    if pred_boxes.shape[0] == 0:
        return 0, 0, int(gt_boxes.shape[0])

    order = np.argsort(-pred_scores)
    pred_boxes = pred_boxes[order]
    pred_cls = pred_cls[order]

    iou = iou_matrix(gt_boxes, pred_boxes)
    matched_gt = set()
    matched_pred = set()

    for pi in range(pred_boxes.shape[0]):
        cls = int(pred_cls[pi])
        candidates = [gi for gi in range(gt_boxes.shape[0]) if gi not in matched_gt and int(gt_cls[gi]) == cls]
        if not candidates:
            continue
        best_g = -1
        best_i = -1.0
        for gi in candidates:
            v = float(iou[gi, pi])
            if v > best_i:
                best_i = v
                best_g = gi
        if best_g >= 0 and best_i >= float(tp_iou):
            matched_gt.add(best_g)
            matched_pred.add(pi)

    tp = len(matched_gt)
    fp = pred_boxes.shape[0] - len(matched_pred)
    fn = gt_boxes.shape[0] - len(matched_gt)
    return int(tp), int(fp), int(fn)


def draw_overlay(
    img_bgr: np.ndarray,
    gt_boxes: np.ndarray,
    gt_cls: np.ndarray,
    pred_boxes: np.ndarray,
    pred_cls: np.ndarray,
    pred_scores: np.ndarray,
) -> np.ndarray:
    out = img_bgr.copy()
    h, w = out.shape[:2]

    def clip_box(b: Sequence[float]) -> Tuple[int, int, int, int]:
        x1 = int(max(0, min(w - 1, round(float(b[0])))))
        y1 = int(max(0, min(h - 1, round(float(b[1])))))
        x2 = int(max(0, min(w - 1, round(float(b[2])))))
        y2 = int(max(0, min(h - 1, round(float(b[3])))))
        return x1, y1, x2, y2

    # GT: green
    for i in range(gt_boxes.shape[0]):
        x1, y1, x2, y2 = clip_box(gt_boxes[i])
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 220, 0), 2)
        txt = f"GT:c{int(gt_cls[i])}"
        cv2.putText(out, txt, (x1, max(14, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 0), 1, cv2.LINE_AA)

    # Pred: red
    for i in range(pred_boxes.shape[0]):
        x1, y1, x2, y2 = clip_box(pred_boxes[i])
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 70, 255), 2)
        score = float(pred_scores[i]) if i < pred_scores.shape[0] else 0.0
        txt = f"P:c{int(pred_cls[i])} {score:.2f}"
        cv2.putText(out, txt, (x1, min(h - 4, y2 + 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 70, 255), 1, cv2.LINE_AA)
    return out


def extract_per_class_map(result_obj, n_cls: int) -> Tuple[np.ndarray, np.ndarray]:
    box = result_obj.box
    m95 = np.array(getattr(box, "maps", []), dtype=float).reshape(-1)
    all_ap = getattr(box, "all_ap", None)
    m50 = np.array([], dtype=float)
    if all_ap is not None:
        arr = np.array(all_ap, dtype=float)
        if arr.ndim == 2 and arr.shape[0] > 0:
            m50 = arr[:, 0]
            if m95.size == 0:
                m95 = np.nanmean(arr, axis=1)
    if m50.size == 0:
        m50 = np.full((n_cls,), np.nan, dtype=float)
    if m95.size == 0:
        m95 = np.full((n_cls,), np.nan, dtype=float)
    if m50.size < n_cls:
        m50 = np.pad(m50, (0, n_cls - m50.size), constant_values=np.nan)
    if m95.size < n_cls:
        m95 = np.pad(m95, (0, n_cls - m95.size), constant_values=np.nan)
    return m50[:n_cls], m95[:n_cls]


def get_names(data_cfg: Dict, result_obj, n_cls: int) -> List[str]:
    names_obj = getattr(result_obj, "names", None)
    if names_obj is None:
        names_obj = data_cfg.get("names", None)
    if isinstance(names_obj, dict):
        out = []
        for i in range(n_cls):
            out.append(str(names_obj.get(i, names_obj.get(str(i), f"class_{i}"))))
        return out
    if isinstance(names_obj, (list, tuple)):
        out = [str(x) for x in names_obj]
        if len(out) < n_cls:
            out.extend([f"class_{i}" for i in range(len(out), n_cls)])
        return out[:n_cls]
    return [f"class_{i}" for i in range(n_cls)]


def run_predict_with_auto_fallback(
    model: YOLO,
    image_paths: List[Path],
    metric_conf: float,
    nms_iou: float,
    imgsz: int,
    batch: int,
    device: str,
    max_det: int,
):
    cur_batch = max(1, int(batch))
    cur_device = str(device)
    last_err = None
    while True:
        try:
            return model.predict(
                source=[str(p) for p in image_paths],
                conf=float(metric_conf),
                iou=float(nms_iou),
                imgsz=int(imgsz),
                max_det=int(max_det),
                save=False,
                verbose=False,
                batch=int(cur_batch),
                device=cur_device,
            ), cur_batch, cur_device
        except RuntimeError as exc:
            msg = str(exc).lower()
            last_err = exc
            if "out of memory" not in msg and "cuda" not in msg:
                raise
            try:
                model.predictor = None
            except Exception:
                pass
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            gc.collect()
            if cur_batch > 1:
                cur_batch = max(1, cur_batch // 2)
                print(f"[warn] OOM in predict, fallback batch -> {cur_batch}")
                continue
            if cur_device != "cpu":
                cur_device = "cpu"
                print("[warn] OOM in predict at batch=1, fallback device -> cpu")
                continue
            raise last_err


def release_cuda_memory(model: YOLO | None = None) -> None:
    try:
        if model is not None:
            model.predictor = None
    except Exception:
        pass
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    gc.collect()


def main() -> None:
    args = parse_args()
    weights = Path(args.weights).resolve()
    data_yaml = Path(args.data_yaml).resolve()
    if not weights.exists():
        raise FileNotFoundError(f"weights not found: {weights}")
    if not data_yaml.exists():
        raise FileNotFoundError(f"data yaml not found: {data_yaml}")

    with data_yaml.open("r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f) or {}
    if not isinstance(data_cfg, dict):
        raise ValueError(f"invalid yaml dict: {data_yaml}")

    req_splits = [x.strip() for x in str(args.splits).split(",") if x.strip()]
    split_sources: Dict[str, Path] = {}
    missing_splits: List[str] = []
    for sp in req_splits:
        if sp in data_cfg and str(data_cfg.get(sp, "")).strip():
            p = resolve_data_entry(str(data_cfg[sp]), data_yaml, data_cfg)
            if p.exists():
                split_sources[sp] = p
            else:
                missing_splits.append(f"{sp}(path_not_found:{p})")
        else:
            missing_splits.append(f"{sp}(missing_in_data_yaml)")
    if not split_sources:
        raise RuntimeError(f"no valid split source found for {req_splits} in {data_yaml}")
    if missing_splits:
        raise RuntimeError(
            "requested splits are incomplete in data yaml. "
            f"missing={missing_splits}. "
            f"data_yaml={data_yaml}. "
            "请在 data.yaml 中补齐 test/val 路径后重跑。"
        )

    exp_dir = weights.parents[2]
    vis_dir = exp_dir / "vis"
    vis_dir.mkdir(parents=True, exist_ok=True)
    fn_dir = vis_dir / "image_level_fn"
    fp_dir = vis_dir / "image_level_fp"
    fn_dir.mkdir(parents=True, exist_ok=True)
    fp_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(weights))

    # -------- mAP by split / by class --------
    map_split_rows: List[Dict[str, object]] = []
    map_class_rows: List[Dict[str, object]] = []
    class_total_acc: Dict[int, Dict[str, float]] = {}
    map_class_total_rows: Dict[int, Dict[str, float]] = {}
    names_ref: List[str] = []
    total_w = 0.0
    total_m50 = 0.0
    total_m95 = 0.0

    for sp, src in split_sources.items():
        try:
            r = model.val(
                data=str(data_yaml),
                split=sp,
                conf=float(args.conf),
                iou=float(args.nms_iou),
                imgsz=int(args.imgsz),
                batch=int(args.batch),
                device=str(args.device),
                max_det=int(args.max_det),
                plots=False,
                save_json=False,
                verbose=False,
            )
        except RuntimeError as exc:
            # val OOM fallback
            msg = str(exc).lower()
            if "out of memory" in msg or "cuda" in msg:
                print("[warn] OOM in model.val, fallback batch=1")
                r = model.val(
                    data=str(data_yaml),
                    split=sp,
                    conf=float(args.conf),
                    iou=float(args.nms_iou),
                    imgsz=int(args.imgsz),
                    batch=1,
                    device=str(args.device),
                    max_det=int(args.max_det),
                    plots=False,
                    save_json=False,
                    verbose=False,
                )
            else:
                raise
        release_cuda_memory(model)

        box = r.box
        split_m50 = float(getattr(box, "map50", np.nan))
        split_m95 = float(getattr(box, "map", np.nan))
        split_p = float(getattr(box, "mp", np.nan))
        split_r = float(getattr(box, "mr", np.nan))
        n_cls = int(getattr(box, "nc", 0))
        if n_cls <= 0:
            names_candidate = data_cfg.get("names", [])
            if isinstance(names_candidate, (list, tuple)):
                n_cls = len(names_candidate)
        if n_cls <= 0:
            n_cls = 1

        ap50_cls, ap95_cls = extract_per_class_map(r, n_cls)
        cls_names = get_names(data_cfg, r, n_cls)
        names_ref = cls_names

        nt_arr = count_gt_by_class_from_source(src, n_cls).astype(float)
        inst = int(np.sum(nt_arr))

        map_split_rows.append(
            {
                "split": sp,
                "instances": inst,
                "mAP50": split_m50,
                "mAP50_95": split_m95,
                "precision": split_p,
                "recall": split_r,
            }
        )

        w = float(inst if inst > 0 else 1.0)
        if np.isfinite(split_m50):
            total_w += w
            total_m50 += split_m50 * w
            total_m95 += split_m95 * w

        for i in range(n_cls):
            row = {
                "split": sp,
                "class_id": i,
                "class_name": cls_names[i] if i < len(cls_names) else f"class_{i}",
                "instances": int(nt_arr[i]) if i < nt_arr.size else 0,
                "mAP50": float(ap50_cls[i]),
                "mAP50_95": float(ap95_cls[i]),
            }
            map_class_rows.append(row)

            if i not in class_total_acc:
                class_total_acc[i] = {"w": 0.0, "m50": 0.0, "m95": 0.0}
            wi = float(nt_arr[i]) if i < nt_arr.size else 0.0
            if wi > 0 and np.isfinite(float(ap50_cls[i])) and np.isfinite(float(ap95_cls[i])):
                class_total_acc[i]["w"] += wi
                class_total_acc[i]["m50"] += float(ap50_cls[i]) * wi
                class_total_acc[i]["m95"] += float(ap95_cls[i]) * wi

    for cid, acc in sorted(class_total_acc.items()):
        if acc["w"] <= 0:
            continue
        row = {
            "split": "val+test",
            "class_id": int(cid),
            "class_name": names_ref[cid] if cid < len(names_ref) else f"class_{cid}",
            "instances": int(acc["w"]),
            "mAP50": float(acc["m50"] / acc["w"]),
            "mAP50_95": float(acc["m95"] / acc["w"]),
        }
        map_class_rows.append(row)
        map_class_total_rows[int(cid)] = {"mAP50": float(row["mAP50"]), "mAP50_95": float(row["mAP50_95"])}

    mAP50_total = float(total_m50 / total_w) if total_w > 0 else float("nan")
    mAP50_95_total = float(total_m95 / total_w) if total_w > 0 else float("nan")
    map_split_rows.append(
        {
            "split": "val+test",
            "instances": int(sum(int(r["instances"]) for r in map_split_rows)),
            "mAP50": float(mAP50_total),
            "mAP50_95": float(mAP50_95_total),
            "precision": float("nan"),
            "recall": float("nan"),
        }
    )
    release_cuda_memory(model)

    # -------- image-level + object-level --------
    image_rows: List[Dict[str, object]] = []
    obj_tp = 0
    obj_fp = 0
    obj_fn = 0
    img_tp = 0
    img_fp = 0
    img_fn = 0
    img_tn = 0

    num_classes = len(names_ref)
    if num_classes <= 0:
        names_cfg = data_cfg.get("names", [])
        if isinstance(names_cfg, (list, tuple)):
            num_classes = len(names_cfg)
    if num_classes <= 0:
        num_classes = 1
        names_ref = [f"class_{i}" for i in range(num_classes)]
    elif len(names_ref) < num_classes:
        names_ref.extend([f"class_{i}" for i in range(len(names_ref), num_classes)])

    def _new_counter() -> Dict[str, int]:
        return {
            "obj_tp": 0,
            "obj_fp": 0,
            "obj_fn": 0,
            "img_tp": 0,
            "img_fp": 0,
            "img_fn": 0,
            "img_tn": 0,
        }

    split_keys = list(split_sources.keys()) + ["val+test"]
    split_counter: Dict[str, Dict[str, int]] = {k: _new_counter() for k in split_keys}
    class_counter: Dict[str, Dict[int, Dict[str, int]]] = {
        k: {cid: _new_counter() for cid in range(num_classes)} for k in split_keys
    }

    for sp, src in split_sources.items():
        release_cuda_memory(model)
        image_paths = list_source_images(src)
        if not image_paths:
            continue
        preds, used_batch, used_device = run_predict_with_auto_fallback(
            model=model,
            image_paths=image_paths,
            metric_conf=float(args.metric_conf),
            nms_iou=float(args.nms_iou),
            imgsz=int(args.imgsz),
            batch=int(args.batch),
            device=str(args.device),
            max_det=int(args.max_det),
        )
        print(f"[info] split={sp} predict_batch={used_batch} predict_device={used_device} images={len(image_paths)}")

        for im_path, res in zip(image_paths, preds):
            gt_boxes, gt_cls = load_gt_xyxy_cls(im_path)
            has_gt = gt_boxes.shape[0] > 0

            p_boxes = np.zeros((0, 4), dtype=np.float32)
            p_scores = np.zeros((0,), dtype=np.float32)
            p_cls = np.zeros((0,), dtype=np.int32)
            if res.boxes is not None and res.boxes.conf is not None and len(res.boxes.conf) > 0:
                p_boxes = res.boxes.xyxy.detach().cpu().numpy().astype(np.float32)
                p_scores = res.boxes.conf.detach().cpu().numpy().astype(np.float32)
                p_cls = res.boxes.cls.detach().cpu().numpy().astype(np.int32)

            keep = p_scores >= float(args.conf)
            p_boxes_f = p_boxes[keep]
            p_scores_f = p_scores[keep]
            p_cls_f = p_cls[keep]
            has_pred = p_boxes_f.shape[0] > 0

            # image-level
            if has_gt and has_pred:
                img_tp += 1
                outcome = "TP"
            elif has_gt and (not has_pred):
                img_fn += 1
                outcome = "FN"
            elif (not has_gt) and has_pred:
                img_fp += 1
                outcome = "FP"
            else:
                img_tn += 1
                outcome = "TN"

            # object-level (class-aware)
            tp_i, fp_i, fn_i = match_one_to_one_class_aware(
                gt_boxes=gt_boxes,
                gt_cls=gt_cls,
                pred_boxes=p_boxes_f,
                pred_cls=p_cls_f,
                pred_scores=p_scores_f,
                tp_iou=float(args.tp_iou),
            )
            obj_tp += int(tp_i)
            obj_fp += int(fp_i)
            obj_fn += int(fn_i)

            for split_name in (sp, "val+test"):
                split_counter[split_name]["obj_tp"] += int(tp_i)
                split_counter[split_name]["obj_fp"] += int(fp_i)
                split_counter[split_name]["obj_fn"] += int(fn_i)
                if outcome == "TP":
                    split_counter[split_name]["img_tp"] += 1
                elif outcome == "FN":
                    split_counter[split_name]["img_fn"] += 1
                elif outcome == "FP":
                    split_counter[split_name]["img_fp"] += 1
                else:
                    split_counter[split_name]["img_tn"] += 1

            for cid in range(num_classes):
                gt_mask = (gt_cls == cid) if gt_cls.size else np.zeros((0,), dtype=bool)
                pred_mask = (p_cls_f == cid) if p_cls_f.size else np.zeros((0,), dtype=bool)
                gt_c = gt_boxes[gt_mask] if gt_boxes.size else np.zeros((0, 4), dtype=np.float32)
                pred_c = p_boxes_f[pred_mask] if p_boxes_f.size else np.zeros((0, 4), dtype=np.float32)
                score_c = p_scores_f[pred_mask] if p_scores_f.size else np.zeros((0,), dtype=np.float32)
                has_gt_c = gt_c.shape[0] > 0
                has_pred_c = pred_c.shape[0] > 0

                tp_c, fp_c, fn_c = match_one_to_one_class_aware(
                    gt_boxes=gt_c,
                    gt_cls=np.zeros((gt_c.shape[0],), dtype=np.int32),
                    pred_boxes=pred_c,
                    pred_cls=np.zeros((pred_c.shape[0],), dtype=np.int32),
                    pred_scores=score_c,
                    tp_iou=float(args.tp_iou),
                )

                for split_name in (sp, "val+test"):
                    cc = class_counter[split_name][cid]
                    cc["obj_tp"] += int(tp_c)
                    cc["obj_fp"] += int(fp_c)
                    cc["obj_fn"] += int(fn_c)

                    if has_gt_c and has_pred_c:
                        cc["img_tp"] += 1
                    elif has_gt_c and (not has_pred_c):
                        cc["img_fn"] += 1
                    elif (not has_gt_c) and has_pred_c:
                        cc["img_fp"] += 1
                    else:
                        cc["img_tn"] += 1

            image_rows.append(
                {
                    "image": str(im_path),
                    "split": sp,
                    "outcome": outcome,
                    "gt_count": int(gt_boxes.shape[0]),
                    "pred_count": int(p_boxes_f.shape[0]),
                    "obj_tp_img": int(tp_i),
                    "obj_fp_img": int(fp_i),
                    "obj_fn_img": int(fn_i),
                }
            )

            if (not args.save_fn_fp_only) or outcome in {"FN", "FP"}:
                if outcome in {"FN", "FP"}:
                    img = cv2.imread(str(im_path))
                    if img is not None:
                        vis = draw_overlay(
                            img_bgr=img,
                            gt_boxes=gt_boxes,
                            gt_cls=gt_cls,
                            pred_boxes=p_boxes_f,
                            pred_cls=p_cls_f,
                            pred_scores=p_scores_f,
                        )
                        dst_root = fn_dir if outcome == "FN" else fp_dir
                        dst = dst_root / f"{sp}__{im_path.stem}.png"
                        cv2.imwrite(str(dst), vis)
        release_cuda_memory(model)

    obj_precision = safe_div(obj_tp, obj_tp + obj_fp)
    obj_recall = safe_div(obj_tp, obj_tp + obj_fn)
    img_precision = safe_div(img_tp, img_tp + img_fp)
    img_recall = safe_div(img_tp, img_tp + img_fn)
    img_fpr = safe_div(img_fp, img_fp + img_tn)

    map_split_lookup = {
        str(r.get("split", "")): {
            "mAP50": r.get("mAP50", np.nan),
            "mAP50_95": r.get("mAP50_95", np.nan),
        }
        for r in map_split_rows
    }
    map_class_lookup = {
        (str(r.get("split", "")), int(r.get("class_id", -1))): {
            "mAP50": r.get("mAP50", np.nan),
            "mAP50_95": r.get("mAP50_95", np.nan),
        }
        for r in map_class_rows
        if "class_id" in r
    }

    overall_by_split_rows: List[Dict[str, object]] = []
    for sp in split_keys:
        c = split_counter[sp]
        obj_p = safe_div(c["obj_tp"], c["obj_tp"] + c["obj_fp"])
        obj_r = safe_div(c["obj_tp"], c["obj_tp"] + c["obj_fn"])
        img_p = safe_div(c["img_tp"], c["img_tp"] + c["img_fp"])
        img_r = safe_div(c["img_tp"], c["img_tp"] + c["img_fn"])
        img_fpr_sp = safe_div(c["img_fp"], c["img_fp"] + c["img_tn"])
        mm = map_split_lookup.get(sp, {"mAP50": np.nan, "mAP50_95": np.nan})
        overall_by_split_rows.append(
            {
                "split": sp,
                "mAP50": mm.get("mAP50", np.nan),
                "mAP50_95": mm.get("mAP50_95", np.nan),
                "obj_precision": obj_p,
                "obj_recall": obj_r,
                "img_precision": img_p,
                "img_recall": img_r,
                "img_fpr": img_fpr_sp,
                "obj_tp": c["obj_tp"],
                "obj_fp": c["obj_fp"],
                "obj_fn": c["obj_fn"],
                "img_tp": c["img_tp"],
                "img_fp": c["img_fp"],
                "img_fn": c["img_fn"],
                "img_tn": c["img_tn"],
            }
        )

    class_prf_rows: List[Dict[str, object]] = []
    for sp in split_keys:
        for cid in range(num_classes):
            c = class_counter[sp][cid]
            obj_p = safe_div(c["obj_tp"], c["obj_tp"] + c["obj_fp"])
            obj_r = safe_div(c["obj_tp"], c["obj_tp"] + c["obj_fn"])
            img_p = safe_div(c["img_tp"], c["img_tp"] + c["img_fp"])
            img_r = safe_div(c["img_tp"], c["img_tp"] + c["img_fn"])
            img_fpr_sp = safe_div(c["img_fp"], c["img_fp"] + c["img_tn"])
            if sp == "val+test":
                mmc = map_class_total_rows.get(cid, {"mAP50": np.nan, "mAP50_95": np.nan})
            else:
                mmc = map_class_lookup.get((sp, cid), {"mAP50": np.nan, "mAP50_95": np.nan})
            class_prf_rows.append(
                {
                    "split": sp,
                    "class_id": cid,
                    "class_name": names_ref[cid] if cid < len(names_ref) else f"class_{cid}",
                    "mAP50": mmc.get("mAP50", np.nan),
                    "mAP50_95": mmc.get("mAP50_95", np.nan),
                    "obj_precision": obj_p,
                    "obj_recall": obj_r,
                    "img_precision": img_p,
                    "img_recall": img_r,
                    "img_fpr": img_fpr_sp,
                    "obj_tp": c["obj_tp"],
                    "obj_fp": c["obj_fp"],
                    "obj_fn": c["obj_fn"],
                    "img_tp": c["img_tp"],
                    "img_fp": c["img_fp"],
                    "img_fn": c["img_fn"],
                    "img_tn": c["img_tn"],
                }
            )

    total_row = {
        "conf": float(args.conf),
        "metric_conf": float(args.metric_conf),
        "nms_iou": float(args.nms_iou),
        "tp_iou": float(args.tp_iou),
        "imgsz": int(args.imgsz),
        "batch": int(args.batch),
        "device": str(args.device),
        "splits_used": ",".join(split_sources.keys()),
        "mAP50_total": float(mAP50_total),
        "mAP50_95_total": float(mAP50_95_total),
        "obj_precision": float(obj_precision),
        "obj_recall": float(obj_recall),
        "img_precision": float(img_precision),
        "img_recall": float(img_recall),
        "img_fpr": float(img_fpr),
        "obj_tp": int(obj_tp),
        "obj_fp": int(obj_fp),
        "obj_fn": int(obj_fn),
        "img_tp": int(img_tp),
        "img_fp": int(img_fp),
        "img_fn": int(img_fn),
        "img_tn": int(img_tn),
    }

    def write_csv(path: Path, rows: List[Dict[str, object]], headers: Sequence[str]) -> None:
        with path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(headers))
            w.writeheader()
            for r in rows:
                w.writerow(r)

    write_csv(
        vis_dir / "map_by_split_conf0.25.csv",
        map_split_rows,
        ["split", "instances", "mAP50", "mAP50_95", "precision", "recall"],
    )
    write_csv(
        vis_dir / "map_by_class_conf0.25.csv",
        map_class_rows,
        ["split", "class_id", "class_name", "instances", "mAP50", "mAP50_95"],
    )
    write_csv(
        vis_dir / "overall_by_split_conf0.25.csv",
        overall_by_split_rows,
        [
            "split",
            "mAP50",
            "mAP50_95",
            "obj_precision",
            "obj_recall",
            "img_precision",
            "img_recall",
            "img_fpr",
            "obj_tp",
            "obj_fp",
            "obj_fn",
            "img_tp",
            "img_fp",
            "img_fn",
            "img_tn",
        ],
    )
    write_csv(
        vis_dir / "class_prf_by_split_conf0.25.csv",
        class_prf_rows,
        [
            "split",
            "class_id",
            "class_name",
            "mAP50",
            "mAP50_95",
            "obj_precision",
            "obj_recall",
            "img_precision",
            "img_recall",
            "img_fpr",
            "obj_tp",
            "obj_fp",
            "obj_fn",
            "img_tp",
            "img_fp",
            "img_fn",
            "img_tn",
        ],
    )
    write_csv(
        vis_dir / "metrics_total_conf0.25.csv",
        [total_row],
        [
            "conf",
            "metric_conf",
            "nms_iou",
            "tp_iou",
            "imgsz",
            "batch",
            "device",
            "splits_used",
            "mAP50_total",
            "mAP50_95_total",
            "obj_precision",
            "obj_recall",
            "img_precision",
            "img_recall",
            "img_fpr",
            "obj_tp",
            "obj_fp",
            "obj_fn",
            "img_tp",
            "img_fp",
            "img_fn",
            "img_tn",
        ],
    )
    write_csv(
        vis_dir / "image_level_details_conf0.25.csv",
        image_rows,
        ["image", "split", "outcome", "gt_count", "pred_count", "obj_tp_img", "obj_fp_img", "obj_fn_img"],
    )
    (vis_dir / "metrics_total_conf0.25.json").write_text(
        json.dumps(total_row, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"[done] vis_dir: {vis_dir}")
    print(f"[done] {vis_dir / 'map_by_split_conf0.25.csv'}")
    print(f"[done] {vis_dir / 'map_by_class_conf0.25.csv'}")
    print(f"[done] {vis_dir / 'overall_by_split_conf0.25.csv'}")
    print(f"[done] {vis_dir / 'class_prf_by_split_conf0.25.csv'}")
    print(f"[done] {vis_dir / 'metrics_total_conf0.25.csv'}")
    print(f"[done] {vis_dir / 'image_level_details_conf0.25.csv'}")
    print(f"[done] FN images: {fn_dir}")
    print(f"[done] FP images: {fp_dir}")


if __name__ == "__main__":
    main()
