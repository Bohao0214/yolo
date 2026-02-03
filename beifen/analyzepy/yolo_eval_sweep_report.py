import argparse
import csv
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_images(image_dir: Path) -> List[Path]:
    return sorted([p for p in image_dir.rglob("*") if p.suffix.lower() in IMG_EXTS and p.is_file()])


def infer_label_dir(image_dir: Path) -> Path:
    parts = list(image_dir.parts)
    if "images" in parts:
        idx = parts.index("images")
        return Path(*parts[:idx], "labels", *parts[idx + 1 :])
    if image_dir.name in {"train", "val", "test"}:
        return image_dir.parent.parent / "labels" / image_dir.name
    return image_dir.parent / "labels"


def load_yolo_labels(label_path: Path, img_w: int, img_h: int) -> List[Tuple[int, float, float, float, float]]:
    if not label_path.exists():
        return []
    boxes = []
    with open(label_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            cls, xc, yc, bw, bh = parts[:5]
            try:
                boxes.append((int(cls), float(xc), float(yc), float(bw), float(bh)))
            except ValueError:
                continue
    return boxes


def xywhn_to_xyxy(xc: float, yc: float, w: float, h: float, img_w: int, img_h: int) -> List[float]:
    x1 = (xc - w / 2) * img_w
    y1 = (yc - h / 2) * img_h
    x2 = (xc + w / 2) * img_w
    y2 = (yc + h / 2) * img_h
    return [x1, y1, x2, y2]


def compute_iou_matrix(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    if gt.size == 0 or pred.size == 0:
        return np.zeros((gt.shape[0], pred.shape[0]), dtype=np.float32)
    ix1 = np.maximum(gt[:, None, 0], pred[None, :, 0])
    iy1 = np.maximum(gt[:, None, 1], pred[None, :, 1])
    ix2 = np.minimum(gt[:, None, 2], pred[None, :, 2])
    iy2 = np.minimum(gt[:, None, 3], pred[None, :, 3])
    iw = np.maximum(0, ix2 - ix1)
    ih = np.maximum(0, iy2 - iy1)
    inter = iw * ih
    gt_area = (gt[:, 2] - gt[:, 0]) * (gt[:, 3] - gt[:, 1])
    pred_area = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
    union = gt_area[:, None] + pred_area[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


def hungarian_assign(cost: np.ndarray) -> np.ndarray:
    n0, m0 = cost.shape
    transposed = False
    if n0 > m0:
        cost = cost.T
        transposed = True
    n, m = cost.shape
    u = np.zeros(n + 1, dtype=np.float32)
    v = np.zeros(m + 1, dtype=np.float32)
    p = np.zeros(m + 1, dtype=int)
    way = np.zeros(m + 1, dtype=int)
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
                if not used[j]:
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
    assignment = -np.ones(n, dtype=int)
    for j in range(1, m + 1):
        if p[j] > 0:
            assignment[p[j] - 1] = j - 1
    if not transposed:
        return assignment
    assignment_orig = -np.ones(n0, dtype=int)
    for row_t, col_t in enumerate(assignment):
        if col_t != -1:
            orig_row = col_t
            orig_col = row_t
            assignment_orig[orig_row] = orig_col
    return assignment_orig


def hungarian_match(gt: np.ndarray, pred: np.ndarray, iou_candidate: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    iou_mat = compute_iou_matrix(gt, pred)
    if gt.size == 0 or pred.size == 0:
        return -np.ones(len(gt), dtype=int), -np.ones(len(pred), dtype=int), iou_mat
    cost = 1.0 - iou_mat
    cost[iou_mat < iou_candidate] = 1e6
    assignment = hungarian_assign(cost)
    gt_match = -np.ones(len(gt), dtype=int)
    pred_match = -np.ones(len(pred), dtype=int)
    for gi, pi in enumerate(assignment):
        if pi != -1 and iou_mat[gi, pi] >= iou_candidate:
            gt_match[gi] = pi
            pred_match[pi] = gi
    return gt_match, pred_match, iou_mat


def box_center(box: List[float]) -> Tuple[float, float]:
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def center_distance(box1: List[float], box2: List[float]) -> float:
    c1x, c1y = box_center(box1)
    c2x, c2y = box_center(box2)
    return float(np.hypot(c1x - c2x, c1y - c2y))


def build_clusters(iou_mat: np.ndarray, thr: float) -> List[List[int]]:
    n = iou_mat.shape[0]
    visited = [False] * n
    clusters = []
    for i in range(n):
        if visited[i]:
            continue
        stack = [i]
        visited[i] = True
        cluster = [i]
        while stack:
            u = stack.pop()
            for v in range(n):
                if not visited[v] and u != v and iou_mat[u, v] > thr:
                    visited[v] = True
                    stack.append(v)
                    cluster.append(v)
        clusters.append(cluster)
    return clusters


def percentile(values: List[float], q: float) -> float:
    if not values:
        return float("nan")
    return float(np.percentile(np.array(values, dtype=np.float32), q))


def run_inference(
    model: YOLO,
    images: List[Path],
    imgsz: int,
    conf: float,
    nms_iou: float,
    batch: int,
    device: str,
    half: bool,
) -> Dict[str, List[Dict]]:
    try:
        import torch
    except Exception:
        torch = None  # type: ignore

    def _run(run_device: str, run_half: bool, run_batch: int) -> Dict[str, List[Dict]]:
        if torch is not None and run_device != "cpu" and torch.cuda.is_available():
            torch.cuda.empty_cache()
        results = model.predict(
            source=[str(p) for p in images],
            imgsz=int(imgsz),
            conf=float(conf),
            iou=float(nms_iou),
            save=False,
            verbose=False,
            batch=int(run_batch),
            device=run_device if run_device else None,
            half=run_half,
            stream=True,
        )
        preds: Dict[str, List[Dict]] = {}
        for res in results:
            img_path = Path(res.path)
            items = []
            if res.boxes is not None:
                xyxy = res.boxes.xyxy.cpu().numpy()
                confs = res.boxes.conf.cpu().numpy()
                clss = res.boxes.cls.cpu().numpy() if res.boxes.cls is not None else np.zeros(len(confs))
                for b, c, cl in zip(xyxy, confs, clss):
                    items.append({"xyxy": b.tolist(), "conf": float(c), "cls": int(cl)})
            preds[str(img_path)] = items
        if hasattr(model, "predictor"):
            model.predictor = None
        if torch is not None and run_device != "cpu" and torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
        return preds

    try:
        return _run(device, half, batch)
    except Exception as exc:
        if torch is None or device == "cpu":
            raise
        if "out of memory" not in str(exc).lower():
            raise
        print("CUDA OOM in inference, retrying on CPU with batch=1.")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return _run("cpu", False, 1)


def clip_box(xyxy: List[float], w: int, h: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = xyxy
    x1 = max(0, min(int(round(x1)), w - 1))
    y1 = max(0, min(int(round(y1)), h - 1))
    x2 = max(0, min(int(round(x2)), w))
    y2 = max(0, min(int(round(y2)), h))
    if x2 <= x1:
        x2 = min(w, x1 + 1)
    if y2 <= y1:
        y2 = min(h, y1 + 1)
    return x1, y1, x2, y2


def draw_boxes(img: np.ndarray, boxes: List[List[float]], color: Tuple[int, int, int], labels: List[str] = None) -> None:
    labels = labels or ["" for _ in boxes]
    for box, label in zip(boxes, labels):
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        if label:
            cv2.putText(
                img,
                label,
                (x1, max(0, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )


def build_collage(images: List[np.ndarray], out_path: Path, title: str, tile_size: int, cols: int = 5) -> None:
    if not images:
        canvas = np.zeros((tile_size + 30, tile_size, 3), dtype=np.uint8)
        cv2.putText(canvas, title, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(canvas, "No samples", (5, 20 + tile_size // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), canvas)
        return
    rows = int(np.ceil(len(images) / cols))
    canvas = np.zeros((rows * tile_size + 30, cols * tile_size, 3), dtype=np.uint8)
    cv2.putText(canvas, title, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    for idx, img in enumerate(images):
        r = idx // cols
        c = idx % cols
        y0 = r * tile_size + 30
        x0 = c * tile_size
        resized = cv2.resize(img, (tile_size, tile_size))
        canvas[y0 : y0 + tile_size, x0 : x0 + tile_size] = resized
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), canvas)


def save_crop(
    img: np.ndarray,
    xyxy: List[float],
    out_path: Path,
    draw_gt: List[List[float]] = None,
    draw_pred: List[List[float]] = None,
) -> None:
    h, w = img.shape[:2]
    x1, y1, x2, y2 = clip_box(xyxy, w, h)
    crop = img[y1:y2, x1:x2].copy()
    if draw_gt:
        for b in draw_gt:
            bx1, by1, bx2, by2 = [int(v) - x1 for v in b]
            cv2.rectangle(crop, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
    if draw_pred:
        for b in draw_pred:
            bx1, by1, bx2, by2 = [int(v) - x1 for v in b]
            cv2.rectangle(crop, (bx1, by1), (bx2, by2), (0, 0, 255), 2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), crop)


def fp_bucket(max_iou: float, gt_count: int, iou_match: float, iou_candidate: float) -> str:
    if gt_count == 0:
        return "FP_image"
    if max_iou < iou_candidate:
        return "FP_strict"
    if max_iou < iou_match:
        return "FP_near"
    return "FP_near"


def evaluate_predictions(
    images: List[Path],
    preds: Dict[str, List[Dict]],
    gt_map: Dict[str, List[List[float]]],
    shape_map: Dict[str, Tuple[int, int]],
    iou_match: float,
    iou_candidate: float,
) -> Tuple[Dict[str, float], Dict[str, List[Dict]]]:
    image_tp = 0
    image_fn = 0
    image_fp = 0
    images_with_gt = 0

    obj_tp = 0
    obj_fp = 0
    obj_fn = 0
    total_gt = 0
    total_preds = 0

    matched_ious: List[float] = []
    center_dists: List[float] = []

    fp_strict_total = 0
    fp_near_total = 0
    fp_image_total = 0

    dup_gt_total = 0
    dup_cluster_total = 0
    dup_candidate_gt_total = 0
    dup_candidate_cluster_total = 0

    image_fn_items: List[Dict] = []
    fp_items: List[Dict] = []
    loc_bad_items: List[Dict] = []
    dup_items: List[Dict] = []
    dup_candidate_items: List[Dict] = []

    for img_path in images:
        img_key = str(img_path)
        gt_boxes = gt_map.get(img_key, [])
        pred_items = preds.get(img_key, [])
        pred_boxes = [p["xyxy"] for p in pred_items]
        pred_confs = [p["conf"] for p in pred_items]

        gt_count = len(gt_boxes)
        pred_count = len(pred_items)

        total_gt += gt_count
        total_preds += pred_count

        has_gt = gt_count > 0
        has_pred = pred_count > 0
        if has_gt:
            images_with_gt += 1

        if has_gt and has_pred:
            gt_arr = np.array(gt_boxes, dtype=np.float32)
            pred_arr = np.array(pred_boxes, dtype=np.float32)
            gt_match, pred_match, iou_mat = hungarian_match(gt_arr, pred_arr, iou_candidate)

            assigned_pairs = [(gi, pi, float(iou_mat[gi, pi])) for gi, pi in enumerate(gt_match) if pi != -1]
            tp_pairs = [p for p in assigned_pairs if p[2] >= iou_match]
            tp_pred_ids = {pi for _, pi, _ in tp_pairs}

            if tp_pairs:
                image_tp += 1
            else:
                image_fn += 1
                image_fn_items.append(
                    {
                        "image": img_key,
                        "gt_boxes": gt_boxes,
                        "pred_boxes": pred_boxes,
                        "gt_count": gt_count,
                        "pred_count": pred_count,
                    }
                )

            obj_tp += len(tp_pairs)
            obj_fn += gt_count - len(tp_pairs)
            obj_fp += pred_count - len(tp_pairs)

            for gi, pi, iou in tp_pairs:
                matched_ious.append(iou)
                center_dists.append(center_distance(gt_boxes[gi], pred_boxes[pi]))

            # per pred
            for pi in range(pred_count):
                max_iou = float(iou_mat[:, pi].max()) if iou_mat.size else 0.0
                nearest_gt_id = int(np.argmax(iou_mat[:, pi])) if iou_mat.size else -1
                assigned_gt_id = int(pred_match[pi]) if pred_match.size else -1
                status = "TP" if pi in tp_pred_ids else "FP"
                if status == "FP":
                    fptype = fp_bucket(max_iou, gt_count, iou_match, iou_candidate)
                    if fptype == "FP_strict":
                        fp_strict_total += 1
                    elif fptype == "FP_near":
                        fp_near_total += 1
                    fp_items.append(
                        {
                            "image": img_key,
                            "pred_box": pred_boxes[pi],
                            "conf": float(pred_confs[pi]),
                            "max_iou": max_iou,
                            "nearest_gt_id": nearest_gt_id,
                            "assigned_gt_id": assigned_gt_id,
                            "status": status,
                            "fp_type": fptype,
                            "gt_boxes": gt_boxes,
                            "pred_boxes": pred_boxes,
                            "gt_count": gt_count,
                            "pred_count": pred_count,
                        }
                    )

            # duplicates strict
            overlaps = iou_mat >= iou_match
            gt_hit_counts = overlaps.sum(axis=1).tolist() if overlaps.size else []
            dup_gt_ids = np.where(overlaps.sum(axis=1) > 1)[0].tolist() if overlaps.size else []
            dup_gt_based = len(dup_gt_ids)

            # candidate duplicates (lower threshold)
            overlaps_cand = iou_mat >= 0.3
            gt_hit_counts_cand = overlaps_cand.sum(axis=1).tolist() if overlaps_cand.size else []
            dup_gt_ids_cand = np.where(overlaps_cand.sum(axis=1) > 1)[0].tolist() if overlaps_cand.size else []
            dup_gt_candidate = len(dup_gt_ids_cand)

            dup_cluster = 0
            pred_clusters = []
            if pred_arr.size >= 2:
                iou_pp = compute_iou_matrix(pred_arr, pred_arr)
                clusters = build_clusters(iou_pp, 0.7)
                pred_clusters = [c for c in clusters if len(c) >= 2]
                dup_cluster = len(pred_clusters)

            if dup_gt_based > 0 or dup_cluster > 0:
                dup_items.append(
                    {
                        "image": img_key,
                        "dup_count": dup_gt_based + dup_cluster,
                        "dup_gt_based": dup_gt_based,
                        "dup_cluster": dup_cluster,
                        "gt_boxes": gt_boxes,
                        "pred_items": pred_items,
                        "gt_hit_counts": gt_hit_counts,
                        "dup_gt_ids": dup_gt_ids,
                        "pred_clusters": pred_clusters,
                    }
                )

            if dup_gt_candidate > 0 or dup_cluster > 0:
                dup_candidate_items.append(
                    {
                        "image": img_key,
                        "dup_count": dup_gt_candidate + dup_cluster,
                        "dup_gt_candidate": dup_gt_candidate,
                        "dup_cluster": dup_cluster,
                        "gt_boxes": gt_boxes,
                        "pred_items": pred_items,
                        "gt_hit_counts": gt_hit_counts_cand,
                        "dup_gt_ids": dup_gt_ids_cand,
                        "pred_clusters": pred_clusters,
                    }
                )

            dup_gt_total += dup_gt_based
            dup_cluster_total += dup_cluster
            dup_candidate_gt_total += dup_gt_candidate
            dup_candidate_cluster_total += dup_cluster

            # loc bad from TP pairs only
            if tp_pairs:
                worst = sorted(tp_pairs, key=lambda x: (x[2], -center_distance(gt_boxes[x[0]], pred_boxes[x[1]])))[0]
                gi, pi, iou = worst
                dist = center_distance(gt_boxes[gi], pred_boxes[pi])
                loc_bad_items.append(
                    {
                        "image": img_key,
                        "gt_box": gt_boxes[gi],
                        "pred_box": pred_boxes[pi],
                        "iou": float(iou),
                        "center_dist": float(dist),
                        "gt_boxes": gt_boxes,
                        "pred_boxes": pred_boxes,
                        "assigned_gt_id": gi,
                        "pred_id": pi,
                        "gt_count": gt_count,
                        "pred_count": pred_count,
                    }
                )
        elif has_gt and not has_pred:
            image_fn += 1
            obj_fn += gt_count
            image_fn_items.append(
                {
                    "image": img_key,
                    "gt_boxes": gt_boxes,
                    "pred_boxes": [],
                    "gt_count": gt_count,
                    "pred_count": 0,
                }
            )
        elif has_pred and not has_gt:
            image_fp += 1
            obj_fp += pred_count
            fp_image_total += pred_count
            for pi in range(pred_count):
                fp_items.append(
                    {
                        "image": img_key,
                        "pred_box": pred_boxes[pi],
                        "conf": float(pred_confs[pi]),
                        "max_iou": 0.0,
                        "nearest_gt_id": -1,
                        "assigned_gt_id": -1,
                        "status": "FP",
                        "fp_type": "FP_image",
                        "gt_boxes": [],
                        "pred_boxes": pred_boxes,
                        "gt_count": 0,
                        "pred_count": pred_count,
                    }
                )
        else:
            pass

    num_images = len(images)
    image_level_recall = image_tp / images_with_gt if images_with_gt > 0 else 0.0
    image_fp_rate = image_fp / num_images if num_images > 0 else 0.0
    avg_pred_per_img = total_preds / num_images if num_images > 0 else 0.0

    obj_recall = obj_tp / total_gt if total_gt > 0 else 0.0
    obj_precision = obj_tp / (obj_tp + obj_fp) if (obj_tp + obj_fp) > 0 else 0.0

    metrics = {
        "image_level_recall": float(image_level_recall),
        "image_fp_rate": float(image_fp_rate),
        "avg_pred_per_img": float(avg_pred_per_img),
        "image_tp_count": int(image_tp),
        "image_fn_count": int(image_fn),
        "image_fp_count": int(image_fp),
        "num_images": int(num_images),
        "images_with_gt": int(images_with_gt),
        "obj_recall": float(obj_recall),
        "obj_precision": float(obj_precision),
        "obj_tp_count": int(obj_tp),
        "obj_fp_count": int(obj_fp),
        "obj_fn_count": int(obj_fn),
        "total_gt": int(total_gt),
        "fp_per_image": float(obj_fp / num_images) if num_images > 0 else 0.0,
        "fp_strict_per_img": float(fp_strict_total / num_images) if num_images > 0 else 0.0,
        "fp_near_per_img": float(fp_near_total / num_images) if num_images > 0 else 0.0,
        "fp_image_per_img": float(fp_image_total / num_images) if num_images > 0 else 0.0,
        "dup_per_image": float((dup_gt_total + dup_cluster_total) / num_images) if num_images > 0 else 0.0,
        "dup_gt_based_per_img": float(dup_gt_total / num_images) if num_images > 0 else 0.0,
        "dup_cluster_per_img": float(dup_cluster_total / num_images) if num_images > 0 else 0.0,
        "dup_candidate_per_img": float((dup_candidate_gt_total + dup_candidate_cluster_total) / num_images) if num_images > 0 else 0.0,
        "dup_candidate_gt_per_img": float(dup_candidate_gt_total / num_images) if num_images > 0 else 0.0,
        "dup_candidate_cluster_per_img": float(dup_candidate_cluster_total / num_images) if num_images > 0 else 0.0,
        "tp_iou_p10": percentile(matched_ious, 10),
        "tp_iou_p50": percentile(matched_ious, 50),
        "tp_iou_p90": percentile(matched_ious, 90),
        "center_dist_p10": percentile(center_dists, 10),
        "center_dist_p50": percentile(center_dists, 50),
        "center_dist_p90": percentile(center_dists, 90),
    }

    items = {
        "image_fn": image_fn_items,
        "fp": fp_items,
        "loc_bad": loc_bad_items,
        "dup": dup_items,
        "dup_candidate": dup_candidate_items,
    }
    return metrics, items


def write_base_metrics(out_path: Path, metrics: Dict[str, float]) -> None:
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "img_recall",
                "obj_recall",
                "obj_precision",
                "image_fp_rate",
                "avg_pred_per_img",
                "fp_per_img",
                "fp_image_per_img",
                "fp_strict_per_img",
                "fp_near_per_img",
                "dup_per_img",
                "dup_gt_based_per_img",
                "dup_cluster_per_img",
                "dup_candidate_per_img",
                "dup_candidate_gt_per_img",
                "dup_candidate_cluster_per_img",
                "tp_iou_p10",
                "tp_iou_p50",
                "tp_iou_p90",
                "center_dist_p10",
                "center_dist_p50",
                "center_dist_p90",
                "obj_tp_count",
                "obj_fp_count",
                "obj_fn_count",
                "image_tp_count",
                "image_fn_count",
                "image_fp_count",
            ]
        )
        writer.writerow(
            [
                f"{metrics['image_level_recall']:.6f}",
                f"{metrics['obj_recall']:.6f}",
                f"{metrics['obj_precision']:.6f}",
                f"{metrics['image_fp_rate']:.6f}",
                f"{metrics['avg_pred_per_img']:.6f}",
                f"{metrics['fp_per_image']:.6f}",
                f"{metrics['fp_image_per_img']:.6f}",
                f"{metrics['fp_strict_per_img']:.6f}",
                f"{metrics['fp_near_per_img']:.6f}",
                f"{metrics['dup_per_image']:.6f}",
                f"{metrics['dup_gt_based_per_img']:.6f}",
                f"{metrics['dup_cluster_per_img']:.6f}",
                f"{metrics['dup_candidate_per_img']:.6f}",
                f"{metrics['dup_candidate_gt_per_img']:.6f}",
                f"{metrics['dup_candidate_cluster_per_img']:.6f}",
                f"{metrics['tp_iou_p10']:.6f}" if not np.isnan(metrics["tp_iou_p10"]) else "",
                f"{metrics['tp_iou_p50']:.6f}" if not np.isnan(metrics["tp_iou_p50"]) else "",
                f"{metrics['tp_iou_p90']:.6f}" if not np.isnan(metrics["tp_iou_p90"]) else "",
                f"{metrics['center_dist_p10']:.6f}" if not np.isnan(metrics["center_dist_p10"]) else "",
                f"{metrics['center_dist_p50']:.6f}" if not np.isnan(metrics["center_dist_p50"]) else "",
                f"{metrics['center_dist_p90']:.6f}" if not np.isnan(metrics["center_dist_p90"]) else "",
                metrics["obj_tp_count"],
                metrics["obj_fp_count"],
                metrics["obj_fn_count"],
                metrics["image_tp_count"],
                metrics["image_fn_count"],
                metrics["image_fp_count"],
            ]
        )


def write_image_metrics(out_path: Path, metrics: Dict[str, float]) -> None:
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "image_level_recall",
                "image_fp_rate",
                "avg_pred_per_img",
                "image_tp_count",
                "image_fn_count",
                "image_fp_count",
                "num_images",
                "images_with_gt",
            ]
        )
        writer.writerow(
            [
                f"{metrics['image_level_recall']:.6f}",
                f"{metrics['image_fp_rate']:.6f}",
                f"{metrics['avg_pred_per_img']:.6f}",
                metrics["image_tp_count"],
                metrics["image_fn_count"],
                metrics["image_fp_count"],
                metrics["num_images"],
                metrics["images_with_gt"],
            ]
        )


def write_localization_metrics(out_path: Path, metrics: Dict[str, float]) -> None:
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "tp_iou_p10",
                "tp_iou_p50",
                "tp_iou_p90",
                "center_dist_p10",
                "center_dist_p50",
                "center_dist_p90",
                "dup_per_image",
                "dup_gt_based_per_img",
                "dup_cluster_per_img",
                "dup_candidate_per_img",
            ]
        )
        writer.writerow(
            [
                f"{metrics['tp_iou_p10']:.6f}" if not np.isnan(metrics["tp_iou_p10"]) else "",
                f"{metrics['tp_iou_p50']:.6f}" if not np.isnan(metrics["tp_iou_p50"]) else "",
                f"{metrics['tp_iou_p90']:.6f}" if not np.isnan(metrics["tp_iou_p90"]) else "",
                f"{metrics['center_dist_p10']:.6f}" if not np.isnan(metrics["center_dist_p10"]) else "",
                f"{metrics['center_dist_p50']:.6f}" if not np.isnan(metrics["center_dist_p50"]) else "",
                f"{metrics['center_dist_p90']:.6f}" if not np.isnan(metrics["center_dist_p90"]) else "",
                f"{metrics['dup_per_image']:.6f}",
                f"{metrics['dup_gt_based_per_img']:.6f}",
                f"{metrics['dup_cluster_per_img']:.6f}",
                f"{metrics['dup_candidate_per_img']:.6f}",
            ]
        )


def write_sweep_metrics(out_path: Path, rows: List[Dict]) -> None:
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "conf",
                "nms_iou",
                "img_recall",
                "obj_recall",
                "obj_precision",
                "image_fp_rate",
                "avg_pred_per_img",
                "fp_image_per_img",
                "fp_strict_per_img",
                "fp_near_per_img",
                "fp_per_img",
                "dup_per_img",
                "tp_iou_p50",
                "center_dist_p50",
            ]
        )
        for r in rows:
            writer.writerow(
                [
                    f"{r['conf']:.3f}",
                    f"{r['nms_iou']:.3f}",
                    f"{r['image_level_recall']:.6f}",
                    f"{r['obj_recall']:.6f}",
                    f"{r['obj_precision']:.6f}",
                    f"{r['image_fp_rate']:.6f}",
                    f"{r['avg_pred_per_img']:.6f}",
                    f"{r['fp_image_per_img']:.6f}",
                    f"{r['fp_strict_per_img']:.6f}",
                    f"{r['fp_near_per_img']:.6f}",
                    f"{r['fp_per_image']:.6f}",
                    f"{r['dup_per_image']:.6f}",
                    f"{r['tp_iou_p50']:.6f}" if not np.isnan(r["tp_iou_p50"]) else "",
                    f"{r['center_dist_p50']:.6f}" if not np.isnan(r["center_dist_p50"]) else "",
                ]
            )


def build_topk_outputs(
    items: Dict[str, List[Dict]],
    topk: int,
    out_dir: Path,
) -> Dict[str, int]:
    topk_dir = out_dir / "topk"
    crops_dir = out_dir / "crops"
    topk_dir.mkdir(parents=True, exist_ok=True)
    (crops_dir / "image_fn").mkdir(parents=True, exist_ok=True)
    (crops_dir / "image_fp").mkdir(parents=True, exist_ok=True)
    (crops_dir / "loc_bad").mkdir(parents=True, exist_ok=True)
    (crops_dir / "dup").mkdir(parents=True, exist_ok=True)

    fn_top = sorted(items["image_fn"], key=lambda x: (x["pred_count"], x["image"]))[:topk]
    fp_all = items["fp"]
    fp_image = [x for x in fp_all if x["fp_type"] == "FP_image"]
    fp_strict = [x for x in fp_all if x["fp_type"] == "FP_strict"]
    fp_near = [x for x in fp_all if x["fp_type"] == "FP_near"]

    fp_image_top = sorted(fp_image, key=lambda x: -x["conf"])[:topk]
    fp_strict_top = sorted(fp_strict, key=lambda x: -x["conf"])[:topk]
    fp_near_top = sorted(fp_near, key=lambda x: -x["conf"])[:topk]

    loc_top = sorted(items["loc_bad"], key=lambda x: (x["iou"], -x["center_dist"]))[:topk]
    dup_top = sorted(items["dup"], key=lambda x: (-x["dup_count"], x["image"]))[:topk]
    dup_cand_top = sorted(items["dup_candidate"], key=lambda x: (-x["dup_count"], x["image"]))[:topk]

    fn_imgs = []
    for idx, item in enumerate(fn_top):
        img = cv2.imread(item["image"])
        if img is None:
            continue
        vis = img.copy()
        draw_boxes(vis, item["gt_boxes"], (0, 255, 0))
        if item["pred_boxes"]:
            draw_boxes(vis, item["pred_boxes"], (0, 0, 255))
        cv2.putText(
            vis,
            f"GT={item['gt_count']} Pred={item['pred_count']} FN",
            (5, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )
        fn_imgs.append(vis)
        crop_path = crops_dir / "image_fn" / f"{idx:03d}_{Path(item['image']).stem}.png"
        cv2.imwrite(str(crop_path), vis)

    def build_fp_collage(fp_items: List[Dict], out_name: str, title: str) -> None:
        imgs = []
        for idx, item in enumerate(fp_items):
            img = cv2.imread(item["image"])
            if img is None:
                continue
            vis = img.copy()
            if item["gt_boxes"]:
                draw_boxes(vis, item["gt_boxes"], (0, 255, 0))
            if item.get("pred_boxes"):
                draw_boxes(vis, item["pred_boxes"], (0, 0, 255))
            draw_boxes(
                vis,
                [item["pred_box"]],
                (0, 0, 255),
                [
                    f"{item['conf']:.2f} iou{item['max_iou']:.2f} {item['fp_type']} ag{item['assigned_gt_id']}"
                ],
            )
            cv2.putText(
                vis,
                f"GT={item['gt_count']} Pred={item['pred_count']}",
                (5, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )
            imgs.append(vis)
            crop_path = crops_dir / "image_fp" / f"{Path(out_name).stem}_{idx:03d}_{Path(item['image']).stem}.png"
            save_crop(img, item["pred_box"], crop_path, draw_gt=item["gt_boxes"], draw_pred=[item["pred_box"]])
        build_collage(imgs, topk_dir / out_name, title, 640)

    build_fp_collage(fp_image_top, "topk_fp_image.png", "FP image TopK")
    build_fp_collage(fp_strict_top, "topk_fp_strict.png", "FP strict TopK")
    build_fp_collage(fp_near_top, "topk_fp_near.png", "FP near TopK")

    loc_imgs = []
    for idx, item in enumerate(loc_top):
        img = cv2.imread(item["image"])
        if img is None:
            continue
        vis = img.copy()
        draw_boxes(vis, item["gt_boxes"], (0, 255, 0))
        draw_boxes(vis, item["pred_boxes"], (0, 0, 255))
        label = (
            f"IoU {item['iou']:.2f} d {item['center_dist']:.1f} "
            f"GT={item['gt_count']} Pred={item['pred_count']} ag{item['assigned_gt_id']}"
        )
        cv2.putText(vis, label, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        loc_imgs.append(vis)
        crop_box = [
            min(item["gt_box"][0], item["pred_box"][0]),
            min(item["gt_box"][1], item["pred_box"][1]),
            max(item["gt_box"][2], item["pred_box"][2]),
            max(item["gt_box"][3], item["pred_box"][3]),
        ]
        crop_path = crops_dir / "loc_bad" / f"{idx:03d}_{Path(item['image']).stem}.png"
        save_crop(img, crop_box, crop_path, draw_gt=[item["gt_box"]], draw_pred=[item["pred_box"]])

    def build_dup_collage(dup_items: List[Dict], out_name: str, title: str) -> None:
        imgs = []
        for idx, item in enumerate(dup_items):
            img = cv2.imread(item["image"])
            if img is None:
                continue
            vis = img.copy()
            draw_boxes(vis, item["gt_boxes"], (0, 255, 0))
            pred_boxes = [p["xyxy"] for p in item["pred_items"]]
            draw_boxes(vis, pred_boxes, (0, 0, 255))
            for gi, gt_box in enumerate(item["gt_boxes"]):
                hit = item["gt_hit_counts"][gi] if gi < len(item["gt_hit_counts"]) else 0
                draw_boxes(vis, [gt_box], (0, 255, 0), [f"k={hit}"])
            cv2.putText(
                vis,
                f"dup_gt={item.get('dup_gt_based', item.get('dup_gt_candidate', 0))} dup_cluster={item['dup_cluster']}",
                (5, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )
            imgs.append(vis)
            if item.get("dup_gt_ids"):
                gi = item["dup_gt_ids"][0]
                crop_box = item["gt_boxes"][gi]
            elif item.get("pred_clusters"):
                cluster = item["pred_clusters"][0]
                xs = [pred_boxes[i][0] for i in cluster] + [pred_boxes[i][2] for i in cluster]
                ys = [pred_boxes[i][1] for i in cluster] + [pred_boxes[i][3] for i in cluster]
                crop_box = [min(xs), min(ys), max(xs), max(ys)]
            else:
                crop_box = [0, 0, img.shape[1], img.shape[0]]
            crop_path = crops_dir / "dup" / f"{Path(out_name).stem}_{idx:03d}_{Path(item['image']).stem}.png"
            save_crop(img, crop_box, crop_path, draw_gt=item["gt_boxes"], draw_pred=pred_boxes)
        build_collage(imgs, topk_dir / out_name, title, 640)

    build_collage(fn_imgs, topk_dir / "topk_image_fn.png", "Image-FN TopK", 640)
    build_collage(loc_imgs, topk_dir / "topk_loc_bad.png", "Loc-bad TopK", 640)
    build_dup_collage(dup_top, "topk_dup.png", "Duplicate TopK")
    build_dup_collage(dup_cand_top, "topk_dup_candidate.png", "Dup candidate TopK")

    with open(topk_dir / "image_fn_topk.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image_path", "boxes", "gt_count", "pred_count"])
        for item in fn_top:
            boxes = [{"type": "gt", "xyxy": b} for b in item["gt_boxes"]]
            boxes += [{"type": "pred", "xyxy": b} for b in item["pred_boxes"]]
            w.writerow([item["image"], json.dumps(boxes), item["gt_count"], item["pred_count"]])

    with open(topk_dir / "fp_topk.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "image_path",
                "boxes",
                "conf",
                "max_iou",
                "nearest_gt_id",
                "assigned_gt_id",
                "status",
                "fp_type",
                "gt_count",
                "pred_count",
            ]
        )
        for item in fp_image_top + fp_strict_top + fp_near_top:
            boxes = [{"type": "pred", "xyxy": item["pred_box"], "conf": item["conf"]}]
            boxes += [{"type": "gt", "xyxy": b} for b in item["gt_boxes"]]
            w.writerow(
                [
                    item["image"],
                    json.dumps(boxes),
                    f"{item['conf']:.6f}",
                    f"{item['max_iou']:.6f}",
                    item["nearest_gt_id"],
                    item["assigned_gt_id"],
                    item["status"],
                    item["fp_type"],
                    item["gt_count"],
                    item["pred_count"],
                ]
            )

    def write_fp_bucket_csv(path: Path, items_list: List[Dict]) -> None:
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["image_path", "boxes", "conf", "max_iou", "assigned_gt_id", "fp_type"])
            for item in items_list:
                boxes = [{"type": "pred", "xyxy": item["pred_box"], "conf": item["conf"]}]
                boxes += [{"type": "gt", "xyxy": b} for b in item["gt_boxes"]]
                w.writerow(
                    [
                        item["image"],
                        json.dumps(boxes),
                        f"{item['conf']:.6f}",
                        f"{item['max_iou']:.6f}",
                        item["assigned_gt_id"],
                        item["fp_type"],
                    ]
                )

    write_fp_bucket_csv(topk_dir / "fp_image_topk.csv", fp_image_top)
    write_fp_bucket_csv(topk_dir / "fp_strict_topk.csv", fp_strict_top)
    write_fp_bucket_csv(topk_dir / "fp_near_topk.csv", fp_near_top)

    with open(topk_dir / "loc_bad_topk.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image_path", "boxes", "iou", "center_dist", "assigned_gt_id", "pred_id"])
        for item in loc_top:
            boxes = [
                {"type": "gt", "xyxy": item["gt_box"]},
                {"type": "pred", "xyxy": item["pred_box"]},
            ]
            w.writerow(
                [
                    item["image"],
                    json.dumps(boxes),
                    f"{item['iou']:.6f}",
                    f"{item['center_dist']:.6f}",
                    item["assigned_gt_id"],
                    item["pred_id"],
                ]
            )

    with open(topk_dir / "dup_topk.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image_path", "boxes", "dup_gt_based", "dup_cluster", "gt_hit_counts"])
        for item in dup_top:
            boxes = [{"type": "gt", "xyxy": b} for b in item["gt_boxes"]]
            boxes += [{"type": "pred", "xyxy": p["xyxy"], "conf": p["conf"]} for p in item["pred_items"]]
            w.writerow(
                [
                    item["image"],
                    json.dumps(boxes),
                    item["dup_gt_based"],
                    item["dup_cluster"],
                    json.dumps(item["gt_hit_counts"]),
                ]
            )

    with open(topk_dir / "dup_candidate_topk.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image_path", "boxes", "dup_candidate_gt", "dup_cluster", "gt_hit_counts"])
        for item in dup_cand_top:
            boxes = [{"type": "gt", "xyxy": b} for b in item["gt_boxes"]]
            boxes += [{"type": "pred", "xyxy": p["xyxy"], "conf": p["conf"]} for p in item["pred_items"]]
            w.writerow(
                [
                    item["image"],
                    json.dumps(boxes),
                    item["dup_gt_candidate"],
                    item["dup_cluster"],
                    json.dumps(item["gt_hit_counts"]),
                ]
            )

    return {
        "fn_count": len(fn_top),
        "fp_image_count": len(fp_image_top),
        "fp_strict_count": len(fp_strict_top),
        "fp_near_count": len(fp_near_top),
        "loc_count": len(loc_top),
        "dup_count": len(dup_top),
        "dup_candidate_count": len(dup_cand_top),
    }


def get_git_commit(repo_root: Path) -> str:
    try:
        res = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except Exception:
        return ""


def get_torch_info() -> Dict[str, str]:
    try:
        import torch

        return {
            "torch_version": torch.__version__,
            "cuda_available": str(torch.cuda.is_available()),
            "cuda_version": str(torch.version.cuda),
            "device_count": str(torch.cuda.device_count()),
        }
    except Exception:
        return {"torch_version": "", "cuda_available": "", "cuda_version": "", "device_count": ""}


def write_report(
    report_path: Path,
    metrics: Dict[str, float],
    sweep_rows: List[Dict],
    topk_stats: Dict[str, int],
    conf: float,
    nms_iou: float,
) -> None:
    if sweep_rows:
        best = max(
            sweep_rows,
            key=lambda r: (r["image_level_recall"], -r["image_fp_rate"], -r["avg_pred_per_img"]),
        )
    else:
        best = {"conf": conf, "nms_iou": nms_iou, **metrics}

    lines = []
    lines.append("# YOLO Eval Report\n")
    lines.append("## 结论\n")
    lines.append(
        f"- 主 KPI（image-level）：image_recall={metrics['image_level_recall']:.3f}, "
        f"image_fp_rate={metrics['image_fp_rate']:.3f}, avg_pred/img={metrics['avg_pred_per_img']:.2f} "
        f"(conf={conf:.2f}, nms_iou={nms_iou:.2f})\n"
    )
    lines.append(
        f"- 辅 KPI（object-level）：obj_recall={metrics['obj_recall']:.3f}, obj_precision={metrics['obj_precision']:.3f}, "
        f"fp_per_img={metrics['fp_per_image']:.3f}, dup_per_img={metrics['dup_per_image']:.3f}\n"
    )
    lines.append(
        f"- 扫参推荐部署点：conf={best['conf']:.2f}, nms_iou={best['nms_iou']:.2f} "
        f"(image_recall={best['image_level_recall']:.3f}, image_fp_rate={best['image_fp_rate']:.3f}, "
        f"avg_pred/img={best['avg_pred_per_img']:.2f})\n"
    )
    lines.append("\n## TopK 主要模式\n")
    lines.append("- image-FN：topk/topk_image_fn.png\n")
    lines.append("- fp_image：topk/topk_fp_image.png\n")
    lines.append("- fp_strict：topk/topk_fp_strict.png\n")
    lines.append("- fp_near：topk/topk_fp_near.png\n")
    lines.append("- loc-bad：topk/topk_loc_bad.png\n")
    lines.append("- duplicate：topk/topk_dup.png\n")
    lines.append("- dup_candidate：topk/topk_dup_candidate.png\n")
    if topk_stats.get("dup_count", 0) == 0:
        lines.append("- dup 为空：可能由于 NMS 较强或阈值过高，建议确认统计路径。\n")
    lines.append("\n## Hook 证据（Work2）\n")
    lines.append("- 待生成 hook/hook_summary.csv 与热图拼图后补充结论。\n")

    report_path.write_text("".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="/home/ubuntu/project/deduibi/yolo/models/defect/exp_2601261106/best/best.pt")
    parser.add_argument("--image_dir", type=str, default="/home/ubuntu/project/deduibi/yolo/dataset/yolo/datasetm6c/images/val")
    parser.add_argument("--out_root", type=str, default="/home/ubuntu/project/deduibi/yolo/analysis")
    parser.add_argument("--report_dir", type=str, default="")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--conf", type=float, default=0.2)
    parser.add_argument("--nms_iou", type=float, default=0.5)
    parser.add_argument("--iou_match", type=float, default=0.5)
    parser.add_argument("--iou_candidate", type=float, default=0.1)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--half", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_dir = Path(args.image_dir)
    label_dir = infer_label_dir(image_dir)
    if not image_dir.exists():
        raise FileNotFoundError(f"image_dir not found: {image_dir}")
    if not label_dir.exists():
        raise FileNotFoundError(f"label_dir not found: {label_dir}")

    out_root = Path(args.out_root)
    if args.report_dir:
        report_dir = Path(args.report_dir)
        if not report_dir.name.startswith("report_"):
            raise ValueError("--report_dir must be a report_YYMMDDHHMM directory name.")
        timestamp = report_dir.name.replace("report_", "")
    else:
        timestamp = dt.datetime.now().strftime("%y%m%d%H%M")
        report_dir = out_root / f"report_{timestamp}"
    metrics_dir = report_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    images = list_images(image_dir)
    if not images:
        raise RuntimeError(f"No images found in {image_dir}")

    gt_map: Dict[str, List[List[float]]] = {}
    shape_map: Dict[str, Tuple[int, int]] = {}
    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        shape_map[str(img_path)] = (w, h)
        rel = img_path.relative_to(image_dir)
        label_path = (label_dir / rel).with_suffix(".txt")
        yolo_labels = load_yolo_labels(label_path, w, h)
        boxes = [xywhn_to_xyxy(xc, yc, bw, bh, w, h) for _, xc, yc, bw, bh in yolo_labels]
        gt_map[str(img_path)] = boxes

    model = YOLO(args.weights)

    base_preds = run_inference(
        model,
        images,
        args.imgsz,
        args.conf,
        args.nms_iou,
        args.batch,
        args.device,
        args.half,
    )

    metrics, items = evaluate_predictions(
        images,
        base_preds,
        gt_map,
        shape_map,
        args.iou_match,
        args.iou_candidate,
    )

    write_base_metrics(metrics_dir / "base_metrics.csv", metrics)
    write_image_metrics(metrics_dir / "image_metrics.csv", metrics)
    write_localization_metrics(metrics_dir / "localization_metrics.csv", metrics)

    topk_stats = build_topk_outputs(items, args.topk, report_dir)

    sweep_confs = [0.15, 0.2, 0.25]
    sweep_ious = [0.4, 0.5, 0.6]
    sweep_rows = []
    for conf in sweep_confs:
        for nms_iou in sweep_ious:
            preds = run_inference(model, images, args.imgsz, conf, nms_iou, args.batch, args.device, args.half)
            s_metrics, _ = evaluate_predictions(
                images,
                preds,
                gt_map,
                shape_map,
                args.iou_match,
                args.iou_candidate,
            )
            sweep_rows.append({"conf": float(conf), "nms_iou": float(nms_iou), **s_metrics})
    write_sweep_metrics(metrics_dir / "sweep_metrics.csv", sweep_rows)

    meta = {
        "timestamp": timestamp,
        "weights": args.weights,
        "image_dir": str(image_dir),
        "label_dir": str(label_dir),
        "out_root": str(out_root),
        "imgsz": args.imgsz,
        "batch": args.batch,
        "conf": args.conf,
        "nms_iou": args.nms_iou,
        "iou_match": args.iou_match,
        "iou_candidate": args.iou_candidate,
        "topk": args.topk,
        "sweep": {"conf": sweep_confs, "nms_iou": sweep_ious},
        "ultralytics_version": getattr(__import__("ultralytics"), "__version__", ""),
        "git_commit": get_git_commit(Path(__file__).resolve().parents[1]),
        "command": " ".join(sys.argv),
        **get_torch_info(),
    }
    with open(report_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    write_report(report_dir / "report.md", metrics, sweep_rows, topk_stats, args.conf, args.nms_iou)

    print(f"Report saved to: {report_dir}")


if __name__ == "__main__":
    main()
