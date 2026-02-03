"""WP1: error evidence closed loop for YOLO defect detection.

This script is designed to be a *single source of truth* for:
- per-image summary (TP/FP/FN/DUP counts)
- per-pred box details (strict / near / dup)
- size+scene bucket error profile (adds edge/highlight dimensions)
- small, readable TopK mosaics (no per-image spam)

Default output file count is small (<= 1 meta + 3 csv + 4 png).

Run:
python /home/ubuntu/project/deduibi/yolo/tools/wp1_error_mining.py \
  --weights /home/ubuntu/project/deduibi/yolo/models/defect/exp_2601261106/best/best.pt \
  --image_dir /home/ubuntu/project/deduibi/yolo/dataset/yolo/datasetm6c/images/val \
  --out_root /home/ubuntu/project/deduibi/yolo/analysis \
  --batch 4 --imgsz 640 --conf 0.2 --nms_iou 0.5 --max_det 300 \
  --tp_iou 0.5 --topk 20

Tip:
- If you want WP1 + WP2 to write into the *same* report directory, first run WP1 to create it,
  then pass --report_dir <that_dir> to both scripts.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from tools.eval_core import (
    BUCKETS,
    BoxRecord,
    box_center_xy,
    box_wh,
    bucket_name_from_min_side,
    compute_iou_matrix,
    crop_with_padding,
    ensure_ultralytics,
    infer_label_dir,
    letterbox_transform_xyxy,
    list_images,
    load_labels_for_image,
    run_inference,
)


# -------------------------
# IO helpers
# -------------------------
def write_csv(path: Path, rows: Sequence[dict], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def bbox_to_str(box: Tuple[float, float, float, float]) -> str:
    return ",".join(f"{v:.2f}" for v in box)


def make_report_dir(out_root: Path, report_dir: Optional[str] = None) -> Path:
    if report_dir:
        rd = Path(report_dir)
        rd.mkdir(parents=True, exist_ok=True)
        return rd
    ts = dt.datetime.now().strftime("report_%y%m%d%H%M")
    rd = out_root / ts
    rd.mkdir(parents=True, exist_ok=False)
    return rd


def safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b else 0.0


# -------------------------
# Scene heuristics
# -------------------------
def is_edge_box_xyxy(box: Tuple[float, float, float, float], w: int, h: int, edge_px: int) -> bool:
    x1, y1, x2, y2 = box
    return (x1 < edge_px) or (y1 < edge_px) or ((w - x2) < edge_px) or ((h - y2) < edge_px)


def highlight_flag(
    img_bgr: np.ndarray,
    box: Tuple[float, float, float, float],
    v_thresh: int,
    s_thresh: int,
    frac_thresh: float,
) -> bool:
    """
    Specular highlight proxy: fraction of pixels satisfying (V high) & (S low) inside the bbox.
    """
    x1, y1, x2, y2 = map(int, box)
    x1 = max(0, min(img_bgr.shape[1] - 1, x1))
    x2 = max(0, min(img_bgr.shape[1], x2))
    y1 = max(0, min(img_bgr.shape[0] - 1, y1))
    y2 = max(0, min(img_bgr.shape[0], y2))
    if x2 <= x1 or y2 <= y1:
        return False
    crop = img_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return False
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    s = hsv[..., 1]
    v = hsv[..., 2]
    mask = (v >= v_thresh) & (s <= s_thresh)
    frac = float(mask.mean())
    return frac >= frac_thresh


# -------------------------
# Matching & duplicates
# -------------------------
def greedy_match_with_types(
    gts: Sequence[BoxRecord],
    preds: Sequence[BoxRecord],
    tp_iou: float,
    near_iou_low: float,
    near_iou_high: float,
) -> Tuple[List[dict], np.ndarray]:
    """
    Local matcher with explicit semantics.

    Returns:
      match_recs: per pred dicts:
        {pred_index, match_type( TP / FP_near / FP_strict ), best_iou, best_gt_index, assigned_gt_index}
      iou_mat: (G,P) IoU matrix (float32)
    """
    if len(preds) == 0:
        return [], np.zeros((len(gts), 0), dtype=np.float32)

    if len(gts) == 0:
        recs = [
            {"pred_index": i, "match_type": "FP_strict", "best_iou": 0.0, "best_gt_index": None, "assigned_gt_index": None}
            for i in range(len(preds))
        ]
        return recs, np.zeros((0, len(preds)), dtype=np.float32)

    gt_boxes = np.array([g.bbox_xyxy for g in gts], dtype=np.float32)
    pred_boxes = np.array([p.bbox_xyxy for p in preds], dtype=np.float32)
    iou = compute_iou_matrix(gt_boxes, pred_boxes)

    best_gt = iou.argmax(axis=0)
    best_iou = iou.max(axis=0)

    # One-to-one TP assignment by global IoU descending
    pairs: List[Tuple[float, int, int]] = []
    for gi in range(len(gts)):
        for pi in range(len(preds)):
            v = float(iou[gi, pi])
            if v >= tp_iou:
                pairs.append((v, gi, pi))
    pairs.sort(reverse=True, key=lambda x: x[0])

    used_g = set()
    used_p = set()
    assigned_gt_for_pred: Dict[int, int] = {}
    for v, gi, pi in pairs:
        if gi in used_g or pi in used_p:
            continue
        used_g.add(gi)
        used_p.add(pi)
        assigned_gt_for_pred[pi] = gi

    recs: List[dict] = []
    for pi in range(len(preds)):
        if pi in assigned_gt_for_pred:
            gi = assigned_gt_for_pred[pi]
            recs.append(
                {
                    "pred_index": pi,
                    "match_type": "TP",
                    "best_iou": float(iou[gi, pi]),
                    "best_gt_index": int(best_gt[pi]),
                    "assigned_gt_index": int(gi),
                }
            )
        else:
            bi = float(best_iou[pi])
            bgi = int(best_gt[pi])
            mt = "FP_near" if (near_iou_low <= bi < near_iou_high) else "FP_strict"
            recs.append(
                {
                    "pred_index": pi,
                    "match_type": mt,
                    "best_iou": bi,
                    "best_gt_index": bgi,
                    "assigned_gt_index": None,
                }
            )

    return recs, iou.astype(np.float32)


def dup_map_by_gt(
    gts: Sequence[BoxRecord],
    preds: Sequence[BoxRecord],
    iou_mat: np.ndarray,
    tp_iou: float,
) -> Dict[int, int]:
    """
    pred_index -> gt_index
    If a GT is overlapped by >=2 preds at IoU>=tp_iou, keep the best pred (max IoU, tie by score),
    mark the rest as duplicates of that GT.
    """
    dup: Dict[int, int] = {}
    if len(gts) == 0 or len(preds) == 0:
        return dup

    for gi in range(len(gts)):
        cand = np.where(iou_mat[gi] >= tp_iou)[0].tolist()
        if len(cand) <= 1:
            continue

        def key(pi: int) -> Tuple[float, float]:
            return (float(iou_mat[gi, pi]), float(getattr(preds[pi], "score", 0.0) or 0.0))

        best_pi = max(cand, key=key)
        for pi in cand:
            if pi != best_pi:
                dup[pi] = gi
    return dup


# -------------------------
# Visualization: mosaic
# -------------------------
def draw_overlay(
    img: np.ndarray,
    gts: Sequence[BoxRecord],
    preds: Sequence[BoxRecord],
    pred_meta: Dict[int, dict],
    header: str,
    highlight: Optional[Tuple[Tuple[float, float, float, float], Tuple[int, int, int]]] = None,
) -> np.ndarray:
    out = img.copy()

    # GT in green
    for gt in gts:
        x1, y1, x2, y2 = map(int, gt.bbox_xyxy)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 200, 0), 2)

    # Pred in red + text
    for pred in preds:
        x1, y1, x2, y2 = map(int, pred.bbox_xyxy)
        info = pred_meta.get(pred.index, {})
        mt = info.get("match_type", "P")
        mx = info.get("max_iou", 0.0)
        text = f"{mt} {float(pred.score or 0.0):.2f} IoU={float(mx):.2f}"
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 220), 2)
        cv2.putText(out, text, (x1, min(out.shape[0] - 4, y2 + 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 220), 1)

    if highlight is not None:
        box, color = highlight
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 4)

    cv2.putText(out, header, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2)
    cv2.putText(out, header, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (10, 10, 10), 1)
    return out


def make_mosaic(images: Sequence[np.ndarray], tile_w: int = 640, pad: int = 4, cols: int = 4) -> np.ndarray:
    if not images:
        return np.zeros((64, 64, 3), dtype=np.uint8)

    resized = []
    for im in images:
        h, w = im.shape[:2]
        if w <= 0 or h <= 0:
            continue
        scale = tile_w / float(w)
        new_h = max(1, int(round(h * scale)))
        resized.append(cv2.resize(im, (tile_w, new_h), interpolation=cv2.INTER_AREA))
    if not resized:
        return np.zeros((64, 64, 3), dtype=np.uint8)

    rows = int(math.ceil(len(resized) / cols))
    tile_h = max(im.shape[0] for im in resized)

    canvas_h = rows * tile_h + (rows + 1) * pad
    canvas_w = cols * tile_w + (cols + 1) * pad
    canvas = np.full((canvas_h, canvas_w, 3), 20, dtype=np.uint8)

    for idx, im in enumerate(resized):
        r = idx // cols
        c = idx % cols
        y0 = pad + r * (tile_h + pad)
        x0 = pad + c * (tile_w + pad)
        canvas[y0:y0 + im.shape[0], x0:x0 + im.shape[1]] = im
    return canvas


# -------------------------
# CLI
# -------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="WP1 error mining for YOLO detection.")
    p.add_argument("--weights", type=str, default="/home/ubuntu/project/deduibi/yolo/models/defect/exp_2601261106/best/best.pt")
    p.add_argument("--image_dir", type=str, default="/home/ubuntu/project/deduibi/yolo/dataset/yolo/datasetm6c/images/val")
    p.add_argument("--out_root", type=str, default="/home/ubuntu/project/deduibi/yolo/analysis")
    p.add_argument("--report_dir", type=str, default="", help="If set, write outputs into this existing directory (no new timestamp dir).")

    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.20)
    p.add_argument("--nms_iou", type=float, default=0.50)
    p.add_argument("--max_det", type=int, default=300)
    p.add_argument("--tp_iou", type=float, default=0.50)

    p.add_argument("--near_iou_low", type=float, default=0.10)
    p.add_argument("--near_iou_high", type=float, default=0.50)

    p.add_argument("--topk", type=int, default=20)

    # scene rules
    p.add_argument("--edge_px", type=int, default=20)
    p.add_argument("--hl_v_thresh", type=int, default=220)
    p.add_argument("--hl_s_thresh", type=int, default=60)
    p.add_argument("--hl_frac", type=float, default=0.05)

    # optional crops (still capped by topk)
    p.add_argument("--write_crops", action="store_true")
    p.add_argument("--crop_size", type=int, default=256)

    p.add_argument("--device", type=str, default="")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ensure_ultralytics()

    weights = Path(args.weights)
    image_dir = Path(args.image_dir)
    out_root = Path(args.out_root)
    label_dir = infer_label_dir(image_dir)
    if not label_dir.exists():
        raise FileNotFoundError(f"Label directory not found: {label_dir}")

    images = list_images(image_dir)
    if not images:
        raise RuntimeError(f"No images found under: {image_dir}")

    report_dir = make_report_dir(out_root, args.report_dir if args.report_dir else None)

    near_iou_high = float(args.near_iou_high)
    if near_iou_high >= float(args.tp_iou):
        near_iou_high = float(args.tp_iou) - 1e-6
    if float(args.near_iou_low) >= near_iou_high:
        raise ValueError("near_iou_low must be < near_iou_high")

    preds_by_img = run_inference(
        weights,
        images,
        args.imgsz,
        args.conf,
        args.nms_iou,
        args.max_det,
        args.batch,
        args.device,
    )

    # Core rows
    image_rows: List[dict] = []
    box_rows: List[dict] = []

    # bucket+scene stats (single CSV)
    scene_stats: Dict[Tuple[str, int, int], dict] = {}

    def get_stat(bucket: str, is_edge: bool, is_hl: bool) -> dict:
        key = (bucket, int(is_edge), int(is_hl))
        if key not in scene_stats:
            scene_stats[key] = {
                "bucket": bucket,
                "is_edge": int(is_edge),
                "is_highlight": int(is_hl),
                "n_images": 0,  # fill later
                "n_gt": 0,
                "n_gt_tp": 0,
                "n_fp_strict": 0,
                "n_fp_near": 0,
                "n_dup": 0,
                "sum_fp_score": 0.0,
                "cnt_fp_score": 0,
                "sum_near_iou": 0.0,
                "cnt_near_iou": 0,
            }
        return scene_stats[key]

    totals = {"images": 0, "gt": 0, "pred": 0, "tp": 0, "fp_strict": 0, "fp_near": 0, "fn": 0, "dup": 0}
    images_with_any_tp = 0

    # TopK items (store minimal metadata + overlay)
    fn_items: List[dict] = []
    fp_strict_items: List[dict] = []
    fp_near_items: List[dict] = []
    dup_items: List[dict] = []

    # crops
    crops_root = report_dir / "crops"
    if args.write_crops:
        for sub in ["FN", "FP_strict", "FP_near", "DUP"]:
            (crops_root / sub).mkdir(parents=True, exist_ok=True)
        crops_meta_f = (report_dir / "crops_meta.jsonl").open("w", encoding="utf-8")
    else:
        crops_meta_f = None

    try:
        for img_path in images:
            img = cv2.imread(str(img_path))
            if img is None:
                raise FileNotFoundError(f"Failed to read image: {img_path}")
            H, W = img.shape[:2]
            image_id = img_path.stem

            label_path = label_dir / f"{image_id}.txt"
            gts = load_labels_for_image(label_path, W, H, image_id, img_path)
            preds = preds_by_img.get(str(img_path), [])

            totals["images"] += 1
            totals["gt"] += len(gts)
            totals["pred"] += len(preds)

            match_recs, iou_mat = greedy_match_with_types(gts, preds, float(args.tp_iou), float(args.near_iou_low), near_iou_high)
            dup_map = dup_map_by_gt(gts, preds, iou_mat, float(args.tp_iou))

            # per-pred meta (indexed by pred.index for stable drawing)
            pred_meta: Dict[int, dict] = {}
            for rec in match_recs:
                pi = int(rec["pred_index"])
                mt = str(rec["match_type"])
                if pi in dup_map:
                    mt = "DUP"
                pred_meta[preds[pi].index] = {
                    "match_type": mt,
                    "max_iou": float(rec["best_iou"]),
                    "best_gt_index": rec["best_gt_index"],
                    "assigned_gt_index": rec["assigned_gt_index"] if mt == "TP" else None,
                    "dup_group": int(dup_map[pi]) if pi in dup_map else -1,
                }

            # GT TP flags
            gt_tp = np.zeros(len(gts), dtype=bool)
            for rec in match_recs:
                pi = int(rec["pred_index"])
                if pi in dup_map:
                    continue
                if rec["match_type"] == "TP":
                    gt_tp[int(rec["assigned_gt_index"])] = True

            n_tp = int(gt_tp.sum())
            n_fn = int(len(gts) - n_tp)
            n_fp_strict = 0
            n_fp_near = 0
            n_dup = 0

            # GT bucket stats
            for gi, gt in enumerate(gts):
                lb = letterbox_transform_xyxy(gt.bbox_xyxy, W, H, int(args.imgsz))
                gw, gh = box_wh(lb)
                bucket = bucket_name_from_min_side(min(gw, gh))
                edge = is_edge_box_xyxy(gt.bbox_xyxy, W, H, int(args.edge_px))
                hl = highlight_flag(img, gt.bbox_xyxy, int(args.hl_v_thresh), int(args.hl_s_thresh), float(args.hl_frac))
                st = get_stat(bucket, edge, hl)
                st["n_gt"] += 1
                st["n_gt_tp"] += int(bool(gt_tp[gi]))

            # preds: bucket stats + eval_boxes rows
            for rec in match_recs:
                pi = int(rec["pred_index"])
                pred = preds[pi]
                info = pred_meta.get(pred.index, {})
                mt = str(info.get("match_type", rec["match_type"]))
                biou = float(rec["best_iou"])
                bgi = rec["best_gt_index"]
                agi = info.get("assigned_gt_index", None)

                lbp = letterbox_transform_xyxy(pred.bbox_xyxy, W, H, int(args.imgsz))
                pw, ph = box_wh(lbp)
                bucket = bucket_name_from_min_side(min(pw, ph))
                edge = is_edge_box_xyxy(pred.bbox_xyxy, W, H, int(args.edge_px))
                hl = highlight_flag(img, pred.bbox_xyxy, int(args.hl_v_thresh), int(args.hl_s_thresh), float(args.hl_frac))
                st = get_stat(bucket, edge, hl)

                if mt == "FP_strict":
                    n_fp_strict += 1
                    st["n_fp_strict"] += 1
                    st["sum_fp_score"] += float(pred.score or 0.0)
                    st["cnt_fp_score"] += 1
                elif mt == "FP_near":
                    n_fp_near += 1
                    st["n_fp_near"] += 1
                    st["sum_fp_score"] += float(pred.score or 0.0)
                    st["cnt_fp_score"] += 1
                    st["sum_near_iou"] += biou
                    st["cnt_near_iou"] += 1
                elif mt == "DUP":
                    n_dup += 1
                    st["n_dup"] += 1
                    st["sum_fp_score"] += float(pred.score or 0.0)
                    st["cnt_fp_score"] += 1

                # best GT bbox string for debugging
                gt_bbox_str = ""
                best_match_id = ""
                if agi is not None and agi >= 0:
                    gt_bbox_str = bbox_to_str(gts[int(agi)].bbox_xyxy)
                    best_match_id = f"gt:{int(agi)}"
                elif bgi is not None and len(gts) > 0:
                    gt_bbox_str = bbox_to_str(gts[int(bgi)].bbox_xyxy)
                    best_match_id = f"gt:{int(bgi)}"

                pred_min_side_lb = float(min(pw, ph))
                pred_area_lb = float(pw * ph)
                box_rows.append(
                    {
                        "image_id": image_id,
                        "img_path": str(img_path),
                        "gt_bbox_xyxy": gt_bbox_str,
                        "pred_bbox_xyxy": bbox_to_str(pred.bbox_xyxy),
                        "pred_score": float(pred.score or 0.0),
                        "match_iou": biou,
                        "match_type": mt,
                        "best_match_id": best_match_id,
                        "pred_min_side_lb": pred_min_side_lb,
                        "pred_area_lb": pred_area_lb,
                        "is_edge": int(edge),
                        "is_highlight": int(hl),
                        "dup_group": int(info.get("dup_group", -1)),
                    }
                )

            totals["tp"] += n_tp
            totals["fn"] += n_fn
            totals["fp_strict"] += n_fp_strict
            totals["fp_near"] += n_fp_near
            totals["dup"] += n_dup

            if n_tp > 0:
                images_with_any_tp += 1

            image_rows.append(
                {
                    "image_id": image_id,
                    "img_path": str(img_path),
                    "n_gt": len(gts),
                    "n_pred": len(preds),
                    "n_tp": n_tp,
                    "n_fn": n_fn,
                    "n_fp_strict": n_fp_strict,
                    "n_fp_near": n_fp_near,
                    "n_dup": n_dup,
                }
            )

            # TopK evidence overlays
            # FN: each missed GT is one item; prioritize small in letterbox space
            for gi, gt in enumerate(gts):
                if gt_tp[gi]:
                    continue
                lbg = letterbox_transform_xyxy(gt.bbox_xyxy, W, H, int(args.imgsz))
                gw, gh = box_wh(lbg)
                min_side = float(min(gw, gh))
                header = f"FN img={image_id} gt#{gi} minSideLB={min_side:.1f}"
                ov = draw_overlay(img, gts, preds, pred_meta, header, highlight=(gt.bbox_xyxy, (0, 255, 255)))
                fn_items.append({"priority": min_side, "img": ov, "img_path": str(img_path), "box": gt.bbox_xyxy, "kind": "FN"})

            # FP strict / near / dup: by score desc
            for rec in match_recs:
                pi = int(rec["pred_index"])
                pred = preds[pi]
                mt = str(pred_meta.get(pred.index, {}).get("match_type", rec["match_type"]))
                score = float(pred.score or 0.0)
                if mt == "FP_strict":
                    header = f"FP_strict img={image_id} score={score:.2f}"
                    ov = draw_overlay(img, gts, preds, pred_meta, header, highlight=(pred.bbox_xyxy, (0, 255, 255)))
                    fp_strict_items.append({"priority": score, "img": ov, "img_path": str(img_path), "box": pred.bbox_xyxy, "kind": "FP_strict"})
                elif mt == "FP_near":
                    header = f"FP_near img={image_id} score={score:.2f} bestIoU={float(rec['best_iou']):.2f}"
                    ov = draw_overlay(img, gts, preds, pred_meta, header, highlight=(pred.bbox_xyxy, (0, 255, 255)))
                    fp_near_items.append({"priority": score, "img": ov, "img_path": str(img_path), "box": pred.bbox_xyxy, "kind": "FP_near"})
                elif mt == "DUP":
                    grp = int(pred_meta.get(pred.index, {}).get("dup_group", -1))
                    header = f"DUP img={image_id} score={score:.2f} dup_of_gt={grp}"
                    ov = draw_overlay(img, gts, preds, pred_meta, header, highlight=(pred.bbox_xyxy, (0, 255, 255)))
                    dup_items.append({"priority": score, "img": ov, "img_path": str(img_path), "box": pred.bbox_xyxy, "kind": "DUP"})

        # Fill n_images
        for st in scene_stats.values():
            st["n_images"] = totals["images"]

        # Write CSVs
        write_csv(
            report_dir / "image_summary.csv",
            image_rows,
            ["image_id", "img_path", "n_gt", "n_pred", "n_tp", "n_fn", "n_fp_strict", "n_fp_near", "n_dup"],
        )
        write_csv(
            report_dir / "eval_boxes.csv",
            box_rows,
            [
                "image_id",
                "img_path",
                "gt_bbox_xyxy",
                "pred_bbox_xyxy",
                "pred_score",
                "match_iou",
                "match_type",
                "best_match_id",
                "pred_min_side_lb",
                "pred_area_lb",
                "is_edge",
                "is_highlight",
                "dup_group",
            ],
        )

        bucket_rows: List[dict] = []

        def bucket_sort_key(k: Tuple[str, int, int]) -> Tuple[int, int, int]:
            b, e, h = k
            return (BUCKETS.index(b) if b in BUCKETS else 999, e, h)

        for (bucket, is_edge, is_hl), st in sorted(scene_stats.items(), key=lambda x: bucket_sort_key(x[0])):
            recall = safe_div(st["n_gt_tp"], st["n_gt"])
            fp_total = st["n_fp_strict"] + st["n_fp_near"] + st["n_dup"]
            fp_per_image = safe_div(fp_total, st["n_images"])
            mean_score = safe_div(st["sum_fp_score"], st["cnt_fp_score"])
            mean_near_iou = safe_div(st["sum_near_iou"], st["cnt_near_iou"])
            bucket_rows.append(
                {
                    "bucket": bucket,
                    "is_edge": is_edge,
                    "is_highlight": is_hl,
                    "n_images": st["n_images"],
                    "n_gt": st["n_gt"],
                    "n_gt_tp": st["n_gt_tp"],
                    "recall": recall,
                    "n_fp_strict": st["n_fp_strict"],
                    "n_fp_near": st["n_fp_near"],
                    "n_dup": st["n_dup"],
                    "fp_per_image": fp_per_image,
                    "mean_fp_score": mean_score,
                    "mean_near_iou": mean_near_iou,
                }
            )

        write_csv(
            report_dir / "bucket_report.csv",
            bucket_rows,
            [
                "bucket",
                "is_edge",
                "is_highlight",
                "n_images",
                "n_gt",
                "n_gt_tp",
                "recall",
                "n_fp_strict",
                "n_fp_near",
                "n_dup",
                "fp_per_image",
                "mean_fp_score",
                "mean_near_iou",
            ],
        )

        # Mosaics (cap to TopK)
        topk = int(args.topk)

        fn_sel = sorted(fn_items, key=lambda d: float(d["priority"]))[:topk]  # smallest first
        fp_s_sel = sorted(fp_strict_items, key=lambda d: float(d["priority"]), reverse=True)[:topk]
        fp_n_sel = sorted(fp_near_items, key=lambda d: float(d["priority"]), reverse=True)[:topk]
        dup_sel = sorted(dup_items, key=lambda d: float(d["priority"]), reverse=True)[:topk]

        vis_dir = report_dir / "vis"
        vis_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(vis_dir / "topk_fn.png"), make_mosaic([d["img"] for d in fn_sel]))
        cv2.imwrite(str(vis_dir / "topk_fp_strict.png"), make_mosaic([d["img"] for d in fp_s_sel]))
        cv2.imwrite(str(vis_dir / "topk_fp_near.png"), make_mosaic([d["img"] for d in fp_n_sel]))
        cv2.imwrite(str(vis_dir / "topk_dup.png"), make_mosaic([d["img"] for d in dup_sel]))

        # Optional crops (only selected TopK items; no per-image spam)
        if args.write_crops and crops_meta_f is not None:

            def export_one(idx: int, item: dict) -> None:
                src = cv2.imread(item["img_path"])
                if src is None:
                    return
                center = box_center_xy(item["box"])
                crop = crop_with_padding(src, center, int(args.crop_size))
                out_path = crops_root / item["kind"] / f"{item['kind']}_{idx:02d}.png"
                cv2.imwrite(str(out_path), crop)
                crops_meta_f.write(
                    json.dumps(
                        {"kind": item["kind"], "img_path": item["img_path"], "box_xyxy": item["box"], "out_path": str(out_path)},
                        ensure_ascii=False,
                    )
                    + "\n"
                )

            for i, it in enumerate(fn_sel):
                export_one(i, it)
            for i, it in enumerate(fp_s_sel):
                export_one(i, it)
            for i, it in enumerate(fp_n_sel):
                export_one(i, it)
            for i, it in enumerate(dup_sel):
                export_one(i, it)

        meta = {
            "weights": str(weights),
            "image_dir": str(image_dir),
            "label_dir": str(label_dir),
            "imgsz": int(args.imgsz),
            "batch": int(args.batch),
            "conf": float(args.conf),
            "nms_iou": float(args.nms_iou),
            "max_det": int(args.max_det),
            "tp_iou": float(args.tp_iou),
            "near_iou_low": float(args.near_iou_low),
            "near_iou_high": float(near_iou_high),
            "topk": int(args.topk),
            "scene_rule": {
                "edge_px": int(args.edge_px),
                "hl_v_thresh": int(args.hl_v_thresh),
                "hl_s_thresh": int(args.hl_s_thresh),
                "hl_frac": float(args.hl_frac),
            },
            "totals": totals,
            "image_level_recall": safe_div(images_with_any_tp, totals["images"]),
            "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        }
        (report_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

        print(f"[WP1] report_dir = {report_dir}")

    finally:
        if crops_meta_f is not None:
            crops_meta_f.close()


if __name__ == "__main__":
    main()
