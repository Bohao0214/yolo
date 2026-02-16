from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np


def list_source_images(source: Path) -> List[Path]:
    if source.is_file() and source.suffix.lower() == ".txt":
        with open(source, "r", encoding="utf-8") as f:
            return [Path(line.strip()) for line in f if line.strip()]
    if source.is_dir():
        return sorted([p for p in source.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}])
    return []


def infer_label_path(image_path: Path) -> Path:
    """Infer label txt path from an image path.

    Supports common dataset variants:
    - images/<split>/xxx.jpg  -> labels/<split>/xxx.txt
    - image/<split>/xxx.jpg   -> labels/<split>/xxx.txt
    And label dir names: labels / label / lable
    """

    parts = list(image_path.parts)
    img_keys = ("images", "image")
    lbl_keys = ("labels", "label", "lable")

    for img_key in img_keys:
        if img_key in parts:
            idx = parts.index(img_key)
            for lbl_key in lbl_keys:
                parts2 = parts.copy()
                parts2[idx] = lbl_key
                cand = Path(*parts2).with_suffix(".txt")
                if cand.exists():
                    return cand
            # return default candidate even if missing
            parts[idx] = "labels"
            return Path(*parts).with_suffix(".txt")

    # fallback: sibling labels folder next to images folder
    for lbl_key in lbl_keys:
        cand = image_path.with_suffix(".txt")
        # if image path already ends with /<split>/xxx.ext, try ../labels/<split>/xxx.txt
        try:
            split = image_path.parent.name
            cand2 = image_path.parent.parent / lbl_key / split / f"{image_path.stem}.txt"
            if cand2.exists():
                return cand2
        except Exception:
            pass
        if cand.exists():
            return cand

    return image_path.with_suffix(".txt")


def image_has_label(image_path: Path) -> bool:
    label_path = infer_label_path(image_path)
    if not label_path.exists():
        return False
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                return True
    return False


def load_yolo_labels(image_path: Path) -> List[Tuple[float, float, float, float]]:
    label_path = infer_label_path(image_path)
    if not label_path.exists():
        return []
    labels: List[Tuple[float, float, float, float]] = []
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            try:
                xc, yc, w, h = map(float, parts[1:5])
            except ValueError:
                continue
            labels.append((xc, yc, w, h))
    return labels


def yolo_to_xyxy(xc: float, yc: float, w: float, h: float, im_w: int, im_h: int) -> List[float]:
    x1 = (xc - w / 2) * im_w
    y1 = (yc - h / 2) * im_h
    x2 = (xc + w / 2) * im_w
    y2 = (yc + h / 2) * im_h
    return [x1, y1, x2, y2]


def compute_iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)
    ax1, ay1, ax2, ay2 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]

    inter_x1 = np.maximum(ax1[:, None], bx1[None, :])
    inter_y1 = np.maximum(ay1[:, None], by1[None, :])
    inter_x2 = np.minimum(ax2[:, None], bx2[None, :])
    inter_y2 = np.minimum(ay2[:, None], by2[None, :])

    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h

    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / union, 0.0).astype(np.float32)


def compute_image_scores(
    model,
    source: Path,
    conf: float,
    batch: int = 1,
    device: str = "",
    nms_iou: float = 0.7,
    max_det: int = 300,
) -> Tuple[np.ndarray, np.ndarray]:
    images = list_source_images(source)
    if not images:
        return np.array([]), np.array([])

    results = model.predict(
        source=[str(p) for p in images],
        conf=float(conf),
        iou=float(nms_iou),
        max_det=int(max_det),
        save=False,
        verbose=False,
        batch=int(batch),
        device=device,
    )
    scores = []
    labels = []
    for img_path, res in zip(images, results):
        labels.append(1 if image_has_label(img_path) else 0)
        if res.boxes is None or res.boxes.conf is None or len(res.boxes.conf) == 0:
            scores.append(0.0)
        else:
            scores.append(float(res.boxes.conf.max().item()))
    return np.array(labels, dtype=np.int32), np.array(scores, dtype=np.float32)


def compute_image_scores_iou(
    model,
    source: Path,
    conf: float,
    iou_match: float,
    batch: int = 1,
    device: str = "",
    nms_iou: float = 0.7,
    max_det: int = 300,
) -> Tuple[np.ndarray, np.ndarray]:
    images = list_source_images(source)
    if not images:
        return np.array([]), np.array([])

    results = model.predict(
        source=[str(p) for p in images],
        conf=float(conf),
        iou=float(nms_iou),
        max_det=int(max_det),
        save=False,
        verbose=False,
        batch=int(batch),
        device=device,
    )

    scores = []
    labels = []
    for img_path, res in zip(images, results):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        gt = load_yolo_labels(img_path)
        labels.append(1 if len(gt) > 0 else 0)

        pred_xyxy = np.zeros((0, 4), dtype=np.float32)
        pred_conf = np.zeros((0,), dtype=np.float32)
        if res.boxes is not None and res.boxes.conf is not None and len(res.boxes.conf) > 0:
            pred_xyxy = res.boxes.xyxy.cpu().numpy().astype(np.float32)
            pred_conf = res.boxes.conf.cpu().numpy().astype(np.float32)

        if len(gt) == 0:
            scores.append(float(np.max(pred_conf)) if pred_conf.size else 0.0)
            continue

        gt_xyxy = np.array([yolo_to_xyxy(xc, yc, bw, bh, w, h) for xc, yc, bw, bh in gt], dtype=np.float32)
        if pred_conf.size == 0:
            scores.append(0.0)
            continue

        iou_mat = compute_iou_matrix(gt_xyxy, pred_xyxy)
        match_mask = iou_mat >= float(iou_match)
        if not match_mask.any():
            scores.append(0.0)
            continue
        matched_pred_idx = np.where(match_mask.any(axis=0))[0]
        scores.append(float(np.max(pred_conf[matched_pred_idx])) if matched_pred_idx.size else 0.0)

    return np.array(labels, dtype=np.int32), np.array(scores, dtype=np.float32)


def _safe_name(path: Path) -> str:
    tail = "__".join(path.parts[-4:]) if len(path.parts) >= 4 else str(path)
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", tail)
    return name


def _draw_boxes(
    img: np.ndarray,
    boxes: np.ndarray,
    color: Tuple[int, int, int],
    labels: Optional[List[str]] = None,
    thickness: int = 2,
) -> None:
    if boxes.size == 0:
        return
    for i, (x1, y1, x2, y2) in enumerate(boxes.tolist()):
        p1 = (int(round(x1)), int(round(y1)))
        p2 = (int(round(x2)), int(round(y2)))
        cv2.rectangle(img, p1, p2, color, thickness)
        if labels and i < len(labels) and labels[i]:
            cv2.putText(
                img,
                labels[i],
                (p1[0], max(0, p1[1] - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )


def compute_image_level_results(
    model,
    source: Path,
    conf_threshold: float,
    iou_match: float,
    metric_conf: float,
    batch: int = 1,
    device: str = "",
    nms_iou: float = 0.7,
    max_det: int = 300,
    split: str = "",
    vis_root: Optional[Path] = None,
    save_visuals: bool = True,
) -> List[Dict[str, object]]:
    images = list_source_images(source)
    if not images:
        return []

    results = model.predict(
        source=[str(p) for p in images],
        conf=float(metric_conf),
        iou=float(nms_iou),
        max_det=int(max_det),
        save=False,
        verbose=False,
        batch=int(batch),
        device=device,
    )

    try:
        from ultralytics.utils import nms as ul_nms  # type: ignore

        nms_stats = getattr(ul_nms, "_nms_stats", None)
    except Exception:
        nms_stats = None

    # Prepare visualization dirs
    vis_dirs = {}
    if vis_root is not None and save_visuals:
        vis_dirs = {
            "image_fn": vis_root / "image_fn",
            "image_fp": vis_root / "image_fp",
            "object_fn": vis_root / "object_fn",
            "object_fp": vis_root / "object_fp",
        }
        for d in vis_dirs.values():
            d.mkdir(parents=True, exist_ok=True)

    items: List[Dict[str, object]] = []
    for img_path, res in zip(images, results):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        gt = load_yolo_labels(img_path)
        has_gt = len(gt) > 0

        gt_xyxy = np.array([yolo_to_xyxy(xc, yc, bw, bh, w, h) for xc, yc, bw, bh in gt], dtype=np.float32)

        pred_xyxy = np.zeros((0, 4), dtype=np.float32)
        pred_conf = np.zeros((0,), dtype=np.float32)
        if res.boxes is not None and res.boxes.conf is not None and len(res.boxes.conf) > 0:
            pred_xyxy = res.boxes.xyxy.cpu().numpy().astype(np.float32)
            pred_conf = res.boxes.conf.cpu().numpy().astype(np.float32)

        num_preds = int(pred_conf.size)

        # Image-level score: max conf among preds matched to any GT (or max conf if no GT)
        num_matched = 0
        best_iou = 0.0
        if has_gt and pred_conf.size:
            iou_mat_all = compute_iou_matrix(gt_xyxy, pred_xyxy)
            if iou_mat_all.size:
                best_iou = float(iou_mat_all.max())
            match_mask_all = iou_mat_all >= float(iou_match)
            matched_pred_idx = np.where(match_mask_all.any(axis=0))[0]
            num_matched = int(matched_pred_idx.size)
            max_conf = float(np.max(pred_conf[matched_pred_idx])) if matched_pred_idx.size else 0.0
        else:
            max_conf = float(np.max(pred_conf)) if pred_conf.size else 0.0

        pred_positive = bool(max_conf >= float(conf_threshold))
        if has_gt:
            outcome = "TP" if pred_positive else "FN"
        else:
            outcome = "FP" if pred_positive else "TN"

        # Object-level errors at conf_threshold
        keep = pred_conf >= float(conf_threshold)
        pred_xyxy_thr = pred_xyxy[keep]
        pred_conf_thr = pred_conf[keep]
        pred_count_thr = int(pred_conf_thr.size)

        obj_fn = 0
        obj_fp = 0
        gt_fn_mask = np.zeros((gt_xyxy.shape[0],), dtype=bool)
        pred_fp_mask = np.zeros((pred_xyxy_thr.shape[0],), dtype=bool)
        pred_match_mask = np.zeros((pred_xyxy_thr.shape[0],), dtype=bool)

        if has_gt:
            if pred_xyxy_thr.size:
                iou_mat_thr = compute_iou_matrix(gt_xyxy, pred_xyxy_thr)
                hit = iou_mat_thr >= float(iou_match)
                gt_matched = hit.any(axis=1)
                pred_matched = hit.any(axis=0)
                gt_fn_mask = ~gt_matched
                pred_match_mask = pred_matched
                pred_fp_mask = ~pred_matched
                obj_fn = int(gt_fn_mask.sum())
                obj_fp = int(pred_fp_mask.sum())
            else:
                gt_fn_mask = np.ones((gt_xyxy.shape[0],), dtype=bool)
                obj_fn = int(gt_fn_mask.sum())
                obj_fp = 0
        else:
            obj_fn = 0
            obj_fp = int(pred_count_thr)
            pred_fp_mask = np.ones((pred_xyxy_thr.shape[0],), dtype=bool) if pred_xyxy_thr.size else pred_fp_mask

        item = {
            "image": str(img_path),
            "split": str(split) if split else "",
            "has_gt": bool(has_gt),
            "gt_count": int(len(gt)),
            "max_conf": float(max_conf),
            "num_preds": num_preds,
            "pred_count_thr": int(pred_count_thr),
            "num_matched": int(num_matched),
            "best_iou": float(best_iou),
            "pred_positive": bool(pred_positive),
            "outcome": str(outcome),
            "obj_fn": int(obj_fn),
            "obj_fp": int(obj_fp),
            "n_pre_nms": None,
            "n_after_topk": None,
            "n_post_nms": None,
            "n_final": None,
            "nms_timeout": None,
        }
        items.append(item)

        # Save visuals
        if vis_dirs:
            # base overlay: GT (green), FN-GT (yellow), matched preds (blue), FP preds (red)
            base = img.copy()

            if gt_xyxy.size:
                gt_labels = ["GT" for _ in range(gt_xyxy.shape[0])]
                _draw_boxes(base, gt_xyxy, (0, 255, 0), labels=gt_labels, thickness=2)
                if gt_fn_mask.size and gt_fn_mask.any():
                    _draw_boxes(base, gt_xyxy[gt_fn_mask], (0, 255, 255), labels=None, thickness=3)

            if pred_xyxy_thr.size:
                pred_labels = [f"{c:.2f}" for c in pred_conf_thr.tolist()]
                # matched preds
                if pred_match_mask.size and pred_match_mask.any():
                    _draw_boxes(base, pred_xyxy_thr[pred_match_mask], (255, 0, 0), labels=None, thickness=2)
                # fp preds
                if pred_fp_mask.size and pred_fp_mask.any():
                    _draw_boxes(base, pred_xyxy_thr[pred_fp_mask], (0, 0, 255), labels=None, thickness=2)
                # draw conf labels for all preds
                _draw_boxes(base, pred_xyxy_thr, (255, 255, 255), labels=pred_labels, thickness=1)

            header = f"{outcome} split={split} conf={conf_threshold:.2f} iou={iou_match:.2f} nms={nms_iou:.2f}"
            cv2.putText(base, header, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            name = _safe_name(img_path)
            if outcome == "FN":
                cv2.imwrite(str(vis_dirs["image_fn"] / name), base)
            if outcome == "FP":
                cv2.imwrite(str(vis_dirs["image_fp"] / name), base)
            if obj_fn > 0:
                cv2.imwrite(str(vis_dirs["object_fn"] / name), base)
            if obj_fp > 0:
                cv2.imwrite(str(vis_dirs["object_fp"] / name), base)

    if nms_stats and len(nms_stats) == len(items):
        for item, stat in zip(items, nms_stats):
            item["n_pre_nms"] = stat.get("n_pre_nms")
            item["n_after_topk"] = stat.get("n_after_topk")
            item["n_post_nms"] = stat.get("n_post_nms")
            item["n_final"] = stat.get("n_final")
            item["nms_timeout"] = stat.get("nms_timeout")

    return items


def save_image_level_report(
    metrics_dir: Path,
    tag: str,
    items: List[Dict[str, object]],
    meta: Dict[str, object],
) -> None:
    import csv
    import json

    metrics_dir.mkdir(parents=True, exist_ok=True)
    data = {"meta": meta, "items": items}
    json_path = metrics_dir / f"{tag}_image_level.json"
    json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    csv_path = metrics_dir / f"{tag}_image_level.csv"
    fieldnames = [
        "image",
        "split",
        "outcome",
        "has_gt",
        "gt_count",
        "pred_positive",
        "max_conf",
        "num_preds",
        "pred_count_thr",
        "num_matched",
        "best_iou",
        "obj_fn",
        "obj_fp",
        "n_pre_nms",
        "n_after_topk",
        "n_post_nms",
        "n_final",
        "nms_timeout",
        "conf_threshold",
        "iou_match",
        "nms_iou",
        "max_det",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in items:
            row = dict(item)
            row.update(
                {
                    "conf_threshold": meta.get("conf_threshold"),
                    "iou_match": meta.get("iou_match"),
                    "nms_iou": meta.get("nms_iou"),
                    "max_det": meta.get("max_det"),
                }
            )
            writer.writerow(row)


def compute_threshold_metrics(
    labels: np.ndarray, scores: np.ndarray, thresholds: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    positives = labels.sum()
    negatives = len(labels) - positives
    recall = []
    precision = []
    fpr = []
    for thr in thresholds:
        preds = (scores >= thr).astype(np.int32)
        tp = int(((preds == 1) & (labels == 1)).sum())
        fp = int(((preds == 1) & (labels == 0)).sum())
        fn = int(((preds == 0) & (labels == 1)).sum())
        rec = tp / positives if positives > 0 else 0.0
        pre = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        fp_rate = fp / negatives if negatives > 0 else 0.0
        recall.append(rec)
        precision.append(pre)
        fpr.append(fp_rate)
    return np.array(recall), np.array(precision), np.array(fpr)


def compute_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    if recall.size == 0:
        return 0.0
    order = np.argsort(recall)
    r = recall[order]
    p = precision[order]
    mrec = np.concatenate(([0.0], r, [1.0]))
    mpre = np.concatenate(([0.0], p, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    return float(np.trapz(mpre, mrec))


def compute_auc(x: np.ndarray, y: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    order = np.argsort(x)
    return float(np.trapz(y[order], x[order]))


def save_metric_plots(
    out_dir: Path,
    tag: str,
    thresholds: np.ndarray,
    recall: np.ndarray,
    precision: np.ndarray,
    fpr: np.ndarray,
    plot_cfg: Optional[Dict[str, object]] = None,
) -> None:
    def _apply_axis_cfg(axis_cfg: Optional[Dict[str, object]]) -> None:
        if not isinstance(axis_cfg, dict):
            return

        def _float_pair(v: object) -> Optional[Tuple[float, float]]:
            if not isinstance(v, (list, tuple)) or len(v) != 2:
                return None
            try:
                return float(v[0]), float(v[1])
            except Exception:
                return None

        xlim_pair = _float_pair(axis_cfg.get("xlim"))
        ylim_pair = _float_pair(axis_cfg.get("ylim"))
        if xlim_pair is not None:
            plt.xlim(*xlim_pair)
        if ylim_pair is not None:
            plt.ylim(*ylim_pair)

        x_tick_step = axis_cfg.get("xtick_step")
        y_tick_step = axis_cfg.get("ytick_step")
        if x_tick_step is not None:
            try:
                step = float(x_tick_step)
                if step > 0:
                    start, end = xlim_pair if xlim_pair is not None else tuple(float(v) for v in plt.xlim())
                    plt.xticks(np.round(np.arange(start, end + 1e-9, step), 4))
            except Exception:
                pass
        if y_tick_step is not None:
            try:
                step = float(y_tick_step)
                if step > 0:
                    start, end = ylim_pair if ylim_pair is not None else tuple(float(v) for v in plt.ylim())
                    plt.yticks(np.round(np.arange(start, end + 1e-9, step), 4))
            except Exception:
                pass

    roc_axis_cfg = None
    pr_axis_cfg = None
    threshold_axis_cfg = None
    if isinstance(plot_cfg, dict):
        maybe_roc = plot_cfg.get("roc")
        if isinstance(maybe_roc, dict):
            roc_axis_cfg = maybe_roc
        maybe_pr = plot_cfg.get("pr", plot_cfg.get("curve"))
        if isinstance(maybe_pr, dict):
            pr_axis_cfg = maybe_pr
        maybe_thr = plot_cfg.get("threshold", plot_cfg.get("curve"))
        if isinstance(maybe_thr, dict):
            threshold_axis_cfg = maybe_thr

    out_dir.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.plot(fpr, recall, color="tab:blue", linewidth=0.8)
    plt.xlabel("FPR")
    plt.ylabel("TPR/Recall")
    plt.title(f"{tag} ROC")
    _apply_axis_cfg(roc_axis_cfg)
    plt.grid(True, alpha=0.3)
    plt.savefig(out_dir / f"{tag}_roc.png", dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(recall, precision, color="tab:green", linewidth=0.8)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"{tag} PR")
    _apply_axis_cfg(pr_axis_cfg)
    plt.grid(True, alpha=0.3)
    plt.savefig(out_dir / f"{tag}_pr.png", dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(thresholds, recall, label="recall", color="tab:orange", linewidth=0.8)
    plt.plot(thresholds, fpr, label="fpr", color="tab:red", linewidth=0.8)
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title(f"{tag} Recall/FPR vs Threshold")
    _apply_axis_cfg(threshold_axis_cfg)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(out_dir / f"{tag}_threshold_curve.png", dpi=150, bbox_inches="tight")
    plt.close()


def save_multi_curve(
    out_dir: Path,
    filename: str,
    xvals: np.ndarray,
    curves: List[Tuple[str, np.ndarray]],
    xlabel: str,
    ylabel: str,
    title: str,
    axis_cfg: Optional[Dict[str, object]] = None,
) -> None:
    def _apply_axis_cfg(cfg: Optional[Dict[str, object]]) -> None:
        if not isinstance(cfg, dict):
            return

        def _float_pair(v: object) -> Optional[Tuple[float, float]]:
            if not isinstance(v, (list, tuple)) or len(v) != 2:
                return None
            try:
                return float(v[0]), float(v[1])
            except Exception:
                return None

        xlim_pair = _float_pair(cfg.get("xlim"))
        ylim_pair = _float_pair(cfg.get("ylim"))
        if xlim_pair is not None:
            plt.xlim(*xlim_pair)
        if ylim_pair is not None:
            plt.ylim(*ylim_pair)

        x_tick_step = cfg.get("xtick_step")
        y_tick_step = cfg.get("ytick_step")
        if x_tick_step is not None:
            try:
                step = float(x_tick_step)
                if step > 0:
                    start, end = xlim_pair if xlim_pair is not None else tuple(float(v) for v in plt.xlim())
                    plt.xticks(np.round(np.arange(start, end + 1e-9, step), 4))
            except Exception:
                pass
        if y_tick_step is not None:
            try:
                step = float(y_tick_step)
                if step > 0:
                    start, end = ylim_pair if ylim_pair is not None else tuple(float(v) for v in plt.ylim())
                    plt.yticks(np.round(np.arange(start, end + 1e-9, step), 4))
            except Exception:
                pass

    out_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    for label, y in curves:
        plt.plot(xvals, y, linewidth=0.8, label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    _apply_axis_cfg(axis_cfg)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / filename, dpi=150, bbox_inches="tight")
    plt.close()


def save_multi_xy_curves(
    out_dir: Path,
    filename: str,
    curves: List[Tuple[str, np.ndarray, np.ndarray]],
    xlabel: str,
    ylabel: str,
    title: str,
    axis_cfg: Optional[Dict[str, object]] = None,
) -> None:
    def _apply_axis_cfg(cfg: Optional[Dict[str, object]]) -> None:
        if not isinstance(cfg, dict):
            return

        def _float_pair(v: object) -> Optional[Tuple[float, float]]:
            if not isinstance(v, (list, tuple)) or len(v) != 2:
                return None
            try:
                return float(v[0]), float(v[1])
            except Exception:
                return None

        xlim_pair = _float_pair(cfg.get("xlim"))
        ylim_pair = _float_pair(cfg.get("ylim"))
        if xlim_pair is not None:
            plt.xlim(*xlim_pair)
        if ylim_pair is not None:
            plt.ylim(*ylim_pair)

        x_tick_step = cfg.get("xtick_step")
        y_tick_step = cfg.get("ytick_step")
        if x_tick_step is not None:
            try:
                step = float(x_tick_step)
                if step > 0:
                    start, end = xlim_pair if xlim_pair is not None else tuple(float(v) for v in plt.xlim())
                    plt.xticks(np.round(np.arange(start, end + 1e-9, step), 4))
            except Exception:
                pass
        if y_tick_step is not None:
            try:
                step = float(y_tick_step)
                if step > 0:
                    start, end = ylim_pair if ylim_pair is not None else tuple(float(v) for v in plt.ylim())
                    plt.yticks(np.round(np.arange(start, end + 1e-9, step), 4))
            except Exception:
                pass

    out_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    for label, x, y in curves:
        plt.plot(x, y, linewidth=0.8, label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    _apply_axis_cfg(axis_cfg)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / filename, dpi=150, bbox_inches="tight")
    plt.close()


def save_multi_curve_grouped(
    out_dir: Path,
    base_filename: str,
    xvals: np.ndarray,
    curves: List[Tuple[str, np.ndarray]],
    xlabel: str,
    ylabel: str,
    title: str,
    group_size: int,
    group_label: str,
    axis_cfg: Optional[Dict[str, object]] = None,
) -> None:
    for i in range(0, len(curves), group_size):
        group = curves[i : i + group_size]
        if not group:
            continue
        idx = i // group_size + 1
        filename = f"{base_filename}_{group_label}{idx}.png"
        save_multi_curve(out_dir, filename, xvals, group, xlabel, ylabel, title, axis_cfg=axis_cfg)


def save_multi_xy_curves_grouped(
    out_dir: Path,
    base_filename: str,
    curves: List[Tuple[str, np.ndarray, np.ndarray]],
    xlabel: str,
    ylabel: str,
    title: str,
    group_size: int,
    group_label: str,
    axis_cfg: Optional[Dict[str, object]] = None,
) -> None:
    for i in range(0, len(curves), group_size):
        group = curves[i : i + group_size]
        if not group:
            continue
        idx = i // group_size + 1
        filename = f"{base_filename}_{group_label}{idx}.png"
        save_multi_xy_curves(out_dir, filename, group, xlabel, ylabel, title, axis_cfg=axis_cfg)


def save_threshold_table_multi(out_dir: Path, tag: str, rows: List[dict]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    header = ["fixed_var", "fixed_value", "threshold", "recall", "fpr"]
    lines = [",".join(header)]
    for r in rows:
        lines.append(
            f"{r['fixed_var']},{r['fixed_value']:.2f},{r['threshold']:.2f},{r['recall']:.6f},{r['fpr']:.6f}"
        )
    (out_dir / f"{tag}_thresholds_multi.csv").write_text("\n".join(lines), encoding="utf-8")


def save_threshold_table(out_dir: Path, tag: str, labels: np.ndarray, scores: np.ndarray, thresholds: np.ndarray) -> None:
    positives = labels.sum()
    negatives = len(labels) - positives
    rows = ["threshold,recall,fpr"]
    for thr in thresholds:
        preds = (scores >= thr).astype(np.int32)
        tp = int(((preds == 1) & (labels == 1)).sum())
        fp = int(((preds == 1) & (labels == 0)).sum())
        recall = tp / positives if positives > 0 else 0.0
        fpr = fp / negatives if negatives > 0 else 0.0
        rows.append(f"{thr:.2f},{recall:.6f},{fpr:.6f}")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{tag}_thresholds.csv", "w", encoding="utf-8") as f:
        f.write("\n".join(rows))


def annotate_fp_fn_images(vis_dir: Path, items: List[Dict[str, object]]) -> None:
    """Deprecated in favor of compute_image_level_results(vis_root=...)."""

    if not vis_dir.exists():
        return

    def find_vis_file(image_path: Path) -> Path:
        direct = vis_dir / image_path.name
        if direct.exists():
            return direct
        for ext in (".jpg", ".jpeg", ".png", ".bmp"):
            cand = vis_dir / f"{image_path.stem}{ext}"
            if cand.exists():
                return cand
        return Path()

    for item in items:
        outcome = item.get("outcome")
        if outcome not in {"FP", "FN"}:
            continue
        img_path = Path(str(item.get("image", "")))
        vis_path = find_vis_file(img_path)
        if not vis_path:
            continue
        img = cv2.imread(str(vis_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        color = (0, 0, 255) if outcome == "FN" else (0, 165, 255)
        thickness = max(2, int(min(h, w) * 0.01))
        cv2.rectangle(img, (0, 0), (w - 1, h - 1), color, thickness)
        cv2.putText(img, str(outcome), (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        cv2.imwrite(str(vis_path), img)
