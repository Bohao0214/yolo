from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import cv2


def list_source_images(source: Path) -> List[Path]:
    if source.is_file() and source.suffix.lower() == ".txt":
        with open(source, "r", encoding="utf-8") as f:
            return [Path(line.strip()) for line in f if line.strip()]
    if source.is_dir():
        return sorted([p for p in source.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}])
    return []


def image_has_label(image_path: Path) -> bool:
    parts = list(image_path.parts)
    if "images" in parts:
        idx = parts.index("images")
        parts[idx] = "labels"
    label_path = Path(*parts).with_suffix(".txt")
    if not label_path.exists():
        return False
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                return True
    return False


def load_yolo_labels(image_path: Path) -> List[Tuple[float, float, float, float]]:
    parts = list(image_path.parts)
    if "images" in parts:
        idx = parts.index("images")
        parts[idx] = "labels"
    label_path = Path(*parts).with_suffix(".txt")
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
        conf=conf,
        iou=float(nms_iou),
        max_det=int(max_det),
        save=False,
        verbose=False,
        batch=batch,
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
        batch=batch,
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

        pred_xyxy = []
        pred_conf = []
        if res.boxes is not None and res.boxes.conf is not None and len(res.boxes.conf) > 0:
            pred_xyxy = res.boxes.xyxy.cpu().numpy()
            pred_conf = res.boxes.conf.cpu().numpy()

        if len(gt) == 0:
            scores.append(float(np.max(pred_conf)) if len(pred_conf) else 0.0)
            continue

        gt_xyxy = np.array([yolo_to_xyxy(xc, yc, bw, bh, w, h) for xc, yc, bw, bh in gt], dtype=np.float32)
        if len(pred_conf) == 0:
            scores.append(0.0)
            continue

        iou_mat = compute_iou_matrix(gt_xyxy, np.array(pred_xyxy, dtype=np.float32))
        match_mask = iou_mat >= float(iou_match)
        if not match_mask.any():
            scores.append(0.0)
            continue
        matched_pred_idx = np.where(match_mask.any(axis=0))[0]
        if matched_pred_idx.size == 0:
            scores.append(0.0)
        else:
            scores.append(float(np.max(pred_conf[matched_pred_idx])))

    return np.array(labels, dtype=np.int32), np.array(scores, dtype=np.float32)


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
        batch=batch,
        device=device,
    )
    try:
        from ultralytics.utils import nms as ul_nms  # type: ignore

        nms_stats = getattr(ul_nms, "_nms_stats", None)
    except Exception:
        nms_stats = None

    items: List[Dict[str, object]] = []
    for img_path, res in zip(images, results):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        gt = load_yolo_labels(img_path)
        has_gt = len(gt) > 0

        pred_xyxy = []
        pred_conf = []
        if res.boxes is not None and res.boxes.conf is not None and len(res.boxes.conf) > 0:
            pred_xyxy = res.boxes.xyxy.cpu().numpy()
            pred_conf = res.boxes.conf.cpu().numpy()

        num_preds = int(len(pred_conf))
        num_matched = 0
        best_iou = 0.0
        if has_gt:
            if len(pred_conf) == 0:
                max_conf = 0.0
            else:
                gt_xyxy = np.array(
                    [yolo_to_xyxy(xc, yc, bw, bh, w, h) for xc, yc, bw, bh in gt], dtype=np.float32
                )
                iou_mat = compute_iou_matrix(gt_xyxy, np.array(pred_xyxy, dtype=np.float32))
                if iou_mat.size:
                    best_iou = float(iou_mat.max())
                match_mask = iou_mat >= float(iou_match)
                matched_pred_idx = np.where(match_mask.any(axis=0))[0]
                num_matched = int(matched_pred_idx.size)
                if matched_pred_idx.size == 0:
                    max_conf = 0.0
                else:
                    max_conf = float(np.max(pred_conf[matched_pred_idx]))
        else:
            max_conf = float(np.max(pred_conf)) if len(pred_conf) else 0.0

        pred_positive = bool(max_conf >= float(conf_threshold))
        if has_gt:
            outcome = "TP" if pred_positive else "FN"
        else:
            outcome = "FP" if pred_positive else "TN"

        items.append(
            {
                "image": str(img_path),
                "has_gt": bool(has_gt),
                "max_conf": float(max_conf),
                "num_preds": num_preds,
                "num_matched": num_matched,
                "best_iou": float(best_iou),
                "pred_positive": bool(pred_positive),
                "outcome": outcome,
                "n_pre_nms": None,
                "n_after_topk": None,
                "n_post_nms": None,
                "n_final": None,
                "nms_timeout": None,
            }
        )

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
    import json
    import csv

    metrics_dir.mkdir(parents=True, exist_ok=True)
    data = {"meta": meta, "items": items}
    json_path = metrics_dir / f"{tag}_image_level.json"
    json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    csv_path = metrics_dir / f"{tag}_image_level.csv"
    fieldnames = [
        "image",
        "outcome",
        "has_gt",
        "pred_positive",
        "max_conf",
        "num_preds",
        "num_matched",
        "best_iou",
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


def annotate_fp_fn_images(vis_dir: Path, items: List[Dict[str, object]]) -> None:
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
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.plot(fpr, recall, color="tab:blue", linewidth=1.0)
    plt.xlabel("FPR")
    plt.ylabel("TPR/Recall")
    plt.title(f"{tag} ROC")
    plt.grid(True, alpha=0.3)
    plt.savefig(out_dir / f"{tag}_roc.png", dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(recall, precision, color="tab:green", linewidth=1.0)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"{tag} PR")
    plt.grid(True, alpha=0.3)
    plt.savefig(out_dir / f"{tag}_pr.png", dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(thresholds, recall, label="recall", color="tab:orange", linewidth=1.0)
    plt.plot(thresholds, fpr, label="fpr", color="tab:red", linewidth=1.0)
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title(f"{tag} Recall/FPR vs Threshold")
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
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    for label, y in curves:
        plt.plot(xvals, y, marker="o", markersize=3, linewidth=1.0, label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / filename, dpi=150, bbox_inches="tight")
    plt.close()


def save_threshold_table_multi(out_dir: Path, tag: str, rows: List[dict]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    header = ["fixed_var", "fixed_value", "threshold", "recall", "fpr"]
    lines = [",".join(header)]
    for r in rows:
        lines.append(
            f"{r['fixed_var']},{r['fixed_value']:.2f},{r['threshold']:.2f},{r['recall']:.6f},{r['fpr']:.6f}"
        )
    (out_dir / f"{tag}_thresholds_multi.csv").write_text("\n".join(lines), encoding="utf-8")


def save_multi_xy_curves(
    out_dir: Path,
    filename: str,
    curves: List[Tuple[str, np.ndarray, np.ndarray]],
    xlabel: str,
    ylabel: str,
    title: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    for label, x, y in curves:
        plt.plot(x, y, marker="o", markersize=3, linewidth=1.0, label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / filename, dpi=150, bbox_inches="tight")
    plt.close()


def save_threshold_table(
    out_dir: Path, tag: str, labels: np.ndarray, scores: np.ndarray, thresholds: np.ndarray
) -> None:
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
