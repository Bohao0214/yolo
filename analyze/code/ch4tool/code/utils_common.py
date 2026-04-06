#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import os
import re
import traceback
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

try:
    import yaml
except Exception:
    yaml = None

# We reuse the project's unified detection metric implementation.
from tools.eval_detection_benchmark import (
    Det,
    compute_metrics,
    iou_one_to_many,
    load_ground_truth,
)


SCALE_BUCKETS = [
    ("s<16", 0.0, 16.0),
    ("16<=s<32", 16.0, 32.0),
    ("32<=s<64", 32.0, 64.0),
    ("s>=64", 64.0, float("inf")),
]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def write_csv(path: Path, rows: List[dict], fieldnames: Optional[List[str]] = None) -> None:
    ensure_dir(path.parent)
    if not rows and fieldnames is None:
        fieldnames = []
    if fieldnames is None:
        keys = set()
        for r in rows:
            keys.update(r.keys())
        fieldnames = sorted(keys)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def write_json(path: Path, obj: object) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def markdown_table(rows: List[dict], columns: List[str]) -> str:
    lines = []
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for r in rows:
        vals = []
        for c in columns:
            v = r.get(c, "")
            vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines) + "\n"


@dataclass
class ModelSpec:
    model_path: str
    model_name: str
    exp_name: str
    ablation_tags: List[str]
    module_groups: List[str]
    status: str = "pending"
    error: str = ""
    config_path: str = ""
    data_yaml: str = ""


def parse_model_paths_txt(path: Path) -> List[str]:
    model_paths: List[str] = []
    if not path.exists():
        return model_paths
    for ln in safe_read_text(path).splitlines():
        s = ln.strip()
        if not s:
            continue
        if s.startswith("#"):
            continue
        if s.startswith("-"):
            s = s[1:].strip()
        model_paths.append(s)
    return model_paths


def infer_ablation_tags(path_text: str) -> List[str]:
    # Parse tags from path fragments such as a7b7c7d7 or defect241__a4__b7__d6.
    tags = re.findall(r"([abcd]\d+)", path_text.lower())
    uniq = []
    for t in tags:
        if t not in uniq:
            uniq.append(t)
    return uniq


def infer_module_groups(tags: Sequence[str]) -> List[str]:
    groups = []
    if any(t.startswith("a") for t in tags):
        groups.append("feature_extraction")
    if any(t.startswith("b") for t in tags):
        groups.append("feature_fusion")
    if any(t.startswith("c") or t.startswith("d") for t in tags):
        groups.append("classification_calibration")
    if not groups:
        groups.append("baseline")
    return groups


def infer_model_name(path_text: str) -> str:
    p = Path(path_text)
    stem = p.stem.lower()
    if stem in {"best", "last"}:
        if p.parent.name == "weights" and len(p.parents) > 2:
            # .../<exp_name>/train/weights/best.pt -> use <exp_name> instead of "train"
            stem = p.parents[2].name.lower()
        else:
            stem = p.parent.name.lower()
    tags = infer_ablation_tags(path_text)
    if tags:
        tag = "+".join(tags)
        return f"yolo_{tag}"
    # fallback to directory name
    return re.sub(r"[^a-z0-9_+-]+", "_", stem)


def infer_exp_name(path_text: str) -> str:
    p = Path(path_text)
    parts = p.parts
    if "experiments" in parts:
        idx = parts.index("experiments")
        keep = parts[idx + 1 : -3] if len(parts) > idx + 4 else parts[idx + 1 :]
        return "/".join(keep)
    return "/".join(p.parts[-6:-1])


def find_nearby_config(weight_path: Path) -> Optional[Path]:
    cands = []
    cur = weight_path.resolve()
    for _ in range(8):
        cur = cur.parent
        cands.extend(
            [
                cur / "args.yaml",
                cur / "train_args.yaml",
                cur / "config.yaml",
                cur / "cfg.yaml",
                cur / "run_args.json",
                cur / "train_meta.json",
            ]
        )
    for c in cands:
        if c.exists() and c.is_file():
            return c
    return None


def _yaml_load(path: Path) -> dict:
    if not path.exists():
        return {}
    txt = safe_read_text(path)
    if path.suffix.lower() == ".json":
        try:
            obj = json.loads(txt)
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}
    if yaml is None:
        return {}
    try:
        obj = yaml.safe_load(txt)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def auto_find_data_yaml_from_cfg(cfg_path: Optional[Path]) -> Optional[Path]:
    if cfg_path is None:
        return None
    cfg = _yaml_load(cfg_path)
    data_val = cfg.get("data")
    if isinstance(data_val, str) and data_val:
        p = Path(data_val)
        if not p.is_absolute():
            p = (cfg_path.parent / p).resolve()
        if p.exists() and p.is_file() and p.suffix.lower() in {".yaml", ".yml"}:
            return p

    # try sibling data.yaml generated by training wrappers
    for cand in [cfg_path.parent / "data.yaml", cfg_path.parent.parent / "data.yaml"]:
        if cand.exists() and cand.is_file():
            return cand.resolve()
    return None


def parse_eval_params_from_cfg(cfg_path: Optional[Path]) -> dict:
    out = {
        "imgsz": None,
        "conf": None,
        "iou": None,
        "max_det": None,
        "batch": None,
        "device": None,
        "score_thr": None,
        "obj_iou": None,
    }
    if cfg_path is None:
        return out
    cfg = _yaml_load(cfg_path)
    keymap = {
        "imgsz": ["imgsz", "img_size"],
        "conf": ["conf", "metric_conf"],
        "iou": ["iou", "nms_iou"],
        "max_det": ["max_det"],
        "batch": ["batch", "eval_batch"],
        "device": ["device", "eval_device"],
        # Prefer explicit score_thr, then conf (for object PR), metric_conf as last fallback.
        "score_thr": ["score_thr", "conf", "metric_conf"],
        # Historical names in previous scripts/configs.
        "obj_iou": ["obj_iou", "match_iou", "tp_iou"],
    }
    for k, keys in keymap.items():
        for kk in keys:
            if kk in cfg and cfg.get(kk) not in [None, ""]:
                out[k] = cfg.get(kk)
                break
    return out


def choose_unified_eval_params(param_dicts: List[dict], overrides: dict) -> Tuple[dict, List[str]]:
    # Historical P2.3.x defaults (frozen pipeline): conf -> NMS(iou) -> max_det.
    # Keeping these as fallback avoids drifting away from previous experiments.
    defaults = {
        "imgsz": 640,
        "conf": 0.3,
        "iou": 0.6,
        "max_det": 20,
        "batch": 4,
        "device": "0",
        "score_thr": 0.3,
        "obj_iou": 0.2,
    }
    pending = []
    out = dict(defaults)

    for key in ["imgsz", "conf", "iou", "max_det", "batch", "device", "score_thr", "obj_iou"]:
        vals = [d.get(key) for d in param_dicts if d.get(key) not in [None, ""]]
        if vals:
            out[key] = vals[0]
        else:
            pending.append(key)

    for k, v in overrides.items():
        if v is not None and v != "":
            out[k] = v
            if k in pending:
                pending.remove(k)

    # If score threshold is still missing, keep object-level threshold aligned with conf.
    if out.get("score_thr") in [None, ""]:
        out["score_thr"] = out.get("conf", 0.3)
        if "score_thr" in pending:
            pending.remove("score_thr")

    # normalize numeric types
    out["imgsz"] = int(float(out["imgsz"]))
    out["conf"] = float(out["conf"])
    out["iou"] = float(out["iou"])
    out["max_det"] = int(float(out["max_det"]))
    out["batch"] = int(float(out["batch"]))
    out["score_thr"] = float(out["score_thr"])
    out["obj_iou"] = float(out["obj_iou"])
    out["device"] = str(out["device"]) if out["device"] is not None else "0"
    return out, pending


def parse_dataset_root(data_yaml: Optional[Path], fallback_root: Optional[Path]) -> Tuple[Optional[Path], List[str]]:
    pending = []
    if fallback_root is not None:
        p = Path(fallback_root).resolve()
        if p.exists():
            return p, pending
    if data_yaml is None:
        pending.append("dataset_root")
        return None, pending
    obj = _yaml_load(data_yaml)
    path_val = obj.get("path") if isinstance(obj, dict) else None
    if isinstance(path_val, str) and path_val:
        p = Path(path_val)
        if not p.is_absolute():
            p = (data_yaml.parent / p).resolve()
        if p.exists():
            return p, pending
    pending.append("dataset_root")
    return None, pending


def build_model_specs(model_paths: List[str]) -> List[ModelSpec]:
    specs = []
    for mp in model_paths:
        tags = infer_ablation_tags(mp)
        specs.append(
            ModelSpec(
                model_path=mp,
                model_name=infer_model_name(mp),
                exp_name=infer_exp_name(mp),
                ablation_tags=tags,
                module_groups=infer_module_groups(tags),
            )
        )
    return specs


def _ultra_load_yolo(model_path: str):
    from ultralytics import YOLO

    return YOLO(model_path)


def run_yolo_predictions(
    model_path: str,
    image_paths: Sequence[Path],
    imgsz: int,
    conf: float,
    iou: float,
    max_det: int,
    batch: int,
    device: str,
) -> Dict[str, List[Det]]:
    model = _ultra_load_yolo(model_path)
    preds: Dict[str, List[Det]] = {str(p.resolve()): [] for p in image_paths}

    source = [str(p.resolve()) for p in image_paths]
    kwargs = {
        "source": source,
        "imgsz": int(imgsz),
        "conf": float(conf),
        "iou": float(iou),
        "max_det": int(max_det),
        "batch": int(batch),
        "save": False,
        "verbose": False,
        "stream": True,
    }
    if device:
        kwargs["device"] = device

    for res in model.predict(**kwargs):
        img_path = str(Path(res.path).resolve())
        boxes_obj = res.boxes
        dets: List[Det] = []
        if boxes_obj is not None and boxes_obj.xyxy is not None:
            boxes = boxes_obj.xyxy.detach().cpu().numpy()
            confs = boxes_obj.conf.detach().cpu().numpy() if boxes_obj.conf is not None else np.ones((len(boxes),), dtype=np.float32)
            clss = boxes_obj.cls.detach().cpu().numpy() if boxes_obj.cls is not None else np.zeros((len(boxes),), dtype=np.float32)
            for b, s, c in zip(boxes, confs, clss):
                dets.append(
                    Det(
                        box=(float(b[0]), float(b[1]), float(b[2]), float(b[3])),
                        score=float(s),
                        label=int(c),
                    )
                )
        preds[img_path] = dets
    return preds


def save_pred_json(path: Path, model_name: str, model_path: str, dataset_root: Path, split: str, pred_map: Dict[str, List[Det]]) -> None:
    rows = []
    for img_path, dets in pred_map.items():
        rows.append(
            {
                "image_id": Path(img_path).stem,
                "image_path": img_path,
                "boxes": [[float(x) for x in d.box] for d in dets],
                "scores": [float(d.score) for d in dets],
                "labels": [int(d.label) for d in dets],
            }
        )
    payload = {
        "format": "det_preds_v1",
        "model": model_name,
        "weights": str(Path(model_path).resolve()),
        "dataset_root": str(dataset_root.resolve()),
        "split": split,
        "predictions": rows,
    }
    write_json(path, payload)


def _bucket_name(min_side: float) -> str:
    for name, lo, hi in SCALE_BUCKETS:
        if lo <= min_side < hi:
            return name
    return SCALE_BUCKETS[-1][0]


def _iou_matrix_xyxy(gt_boxes: np.ndarray, pred_boxes: np.ndarray) -> np.ndarray:
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


def _hungarian_assign(cost: np.ndarray) -> List[int]:
    """Hungarian algorithm (min-cost), returns row->col assignment."""
    n, m = cost.shape
    if n == 0:
        return []
    if n > m:
        pad = np.full((n, n - m), 1.0, dtype=cost.dtype)
        cost = np.hstack([cost, pad])
        m = n
    u = np.zeros(n + 1, dtype=np.float32)
    v = np.zeros(m + 1, dtype=np.float32)
    p = np.zeros(m + 1, dtype=np.int64)
    way = np.zeros(m + 1, dtype=np.int64)
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
                if used[j]:
                    continue
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
    assignment = [-1] * n
    for j in range(1, m + 1):
        if p[j] != 0:
            assignment[p[j] - 1] = j - 1
    return assignment


def _match_one_to_one_for_image(
    gt_dets: List[Det],
    pred_dets: List[Det],
    score_thr: float,
    iou_thr: float,
    class_aware: bool = True,
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    # P2.3.x frozen style: one-to-one matching by maximizing total IoU (Hungarian),
    # with IoU threshold gating (IoU >= iou_thr).
    matches: List[Tuple[int, int]] = []
    gt_unmatched = set(range(len(gt_dets)))
    pred_unmatched = set(i for i, d in enumerate(pred_dets) if d.score >= score_thr)

    if not class_aware:
        classes = [0]
    else:
        classes = sorted(set([d.label for d in gt_dets] + [d.label for d in pred_dets]))
    for cls in classes:
        if class_aware:
            gi = [i for i, d in enumerate(gt_dets) if d.label == cls]
            pi = [i for i, d in enumerate(pred_dets) if d.label == cls and d.score >= score_thr]
        else:
            gi = [i for i, _ in enumerate(gt_dets)]
            pi = [i for i, d in enumerate(pred_dets) if d.score >= score_thr]

        if not gi or not pi:
            continue

        gt_boxes = np.asarray([gt_dets[i].box for i in gi], dtype=np.float32).reshape(-1, 4)
        pred_boxes = np.asarray([pred_dets[i].box for i in pi], dtype=np.float32).reshape(-1, 4)
        iou_mat = _iou_matrix_xyxy(gt_boxes, pred_boxes)
        cost = np.where(iou_mat >= float(iou_thr), 1.0 - iou_mat, 1.0)
        assign = _hungarian_assign(cost)
        for local_gi, local_pj in enumerate(assign):
            if local_pj < 0 or local_pj >= len(pi):
                continue
            if iou_mat[local_gi, local_pj] < float(iou_thr):
                continue
            gidx = gi[local_gi]
            pidx = pi[local_pj]
            matches.append((gidx, pidx))
            gt_unmatched.discard(gidx)
            pred_unmatched.discard(pidx)

    return matches, sorted(gt_unmatched), sorted(pred_unmatched)


def _greedy_match_for_image(
    gt_dets: List[Det],
    pred_dets: List[Det],
    score_thr: float,
    iou_thr: float,
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    # Backward-compatible alias. Internally we use one-to-one Hungarian matching
    # to keep consistency with historical P2.3.x experiments.
    return _match_one_to_one_for_image(
        gt_dets=gt_dets,
        pred_dets=pred_dets,
        score_thr=score_thr,
        iou_thr=iou_thr,
        class_aware=True,
    )


def compute_obj_pr_counts(
    gt_map: Dict[str, List[Det]],
    pred_map: Dict[str, List[Det]],
    score_thr: float,
    iou_thr: float,
) -> Dict[str, float]:
    tp = 0
    fp = 0
    fn = 0
    for img_path in sorted(set(gt_map.keys()) | set(pred_map.keys())):
        gt_dets = gt_map.get(img_path, [])
        pred_dets = pred_map.get(img_path, [])
        matches, gt_unmatched, pred_unmatched = _match_one_to_one_for_image(
            gt_dets=gt_dets,
            pred_dets=pred_dets,
            score_thr=score_thr,
            iou_thr=iou_thr,
            class_aware=True,
        )
        tp += len(matches)
        fn += len(gt_unmatched)
        fp += len(pred_unmatched)

    precision = float(tp) / float(tp + fp) if (tp + fp) > 0 else 0.0
    recall = float(tp) / float(tp + fn) if (tp + fn) > 0 else 0.0
    return {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "object_precision": precision,
        "object_recall": recall,
    }


def compute_scale_recall_and_image_fp(
    gt_map: Dict[str, List[Det]],
    pred_map: Dict[str, List[Det]],
    score_thr: float,
    iou_thr: float,
) -> Tuple[List[dict], dict]:
    # Scale recall
    bucket_cnt = {name: {"gt": 0, "tp": 0, "fn": 0} for name, _, _ in SCALE_BUCKETS}

    img_neg = 0
    img_fp = 0

    for img_path, gt_dets in gt_map.items():
        preds = pred_map.get(img_path, [])
        matches, gt_unmatched, _ = _match_one_to_one_for_image(gt_dets, preds, score_thr=score_thr, iou_thr=iou_thr)
        matched_gt = {g for g, _ in matches}

        for i, g in enumerate(gt_dets):
            w = max(0.0, g.box[2] - g.box[0])
            h = max(0.0, g.box[3] - g.box[1])
            b = _bucket_name(min(w, h))
            bucket_cnt[b]["gt"] += 1
            if i in matched_gt:
                bucket_cnt[b]["tp"] += 1
            else:
                bucket_cnt[b]["fn"] += 1

        has_gt = len(gt_dets) > 0
        has_pred = any(p.score >= score_thr for p in preds)
        if not has_gt:
            img_neg += 1
            if has_pred:
                img_fp += 1

    rows = []
    for name, _, _ in SCALE_BUCKETS:
        g = bucket_cnt[name]["gt"]
        tp = bucket_cnt[name]["tp"]
        fn = bucket_cnt[name]["fn"]
        rec = (tp / g) if g > 0 else 0.0
        rows.append(
            {
                "scale_bucket": name,
                "GT": g,
                "TP": tp,
                "FN": fn,
                "recall": f"{rec:.6f}",
            }
        )

    image_stats = {
        "image_negatives": img_neg,
        "image_fp": img_fp,
        "image_fp_rate": float(img_fp / img_neg) if img_neg > 0 else 0.0,
    }
    return rows, image_stats


def _crop_img(img: np.ndarray, box: Tuple[float, float, float, float]) -> np.ndarray:
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box
    xi1 = max(0, min(w - 1, int(math.floor(x1))))
    yi1 = max(0, min(h - 1, int(math.floor(y1))))
    xi2 = max(0, min(w, int(math.ceil(x2))))
    yi2 = max(0, min(h, int(math.ceil(y2))))
    if xi2 <= xi1 or yi2 <= yi1:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    return img[yi1:yi2, xi1:xi2]


def _fp_texture_type(img: np.ndarray, box: Tuple[float, float, float, float]) -> str:
    crop = _crop_img(img, box)
    if crop.size == 0:
        return "unknown"
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    mean_v = float(np.mean(gray))
    std_v = float(np.std(gray))
    edges = cv2.Canny(gray, 80, 160)
    edge_ratio = float(np.mean(edges > 0))

    if mean_v > 210 and std_v < 35:
        return "highlight"
    if edge_ratio > 0.18:
        return "texture_shadow"
    return "other"


def compute_fn_fp_breakdown(
    gt_map: Dict[str, List[Det]],
    pred_map: Dict[str, List[Det]],
    score_thr: float,
    iou_thr: float,
) -> Tuple[List[dict], List[dict], dict]:
    fn_counter = {
        "no_response": 0,
        "low_score": 0,
        "regression_poor": 0,
        "postproc_suppressed": 0,
    }
    fp_counter = {
        "FP_unmatched": 0,
        "FP_pred_dup": 0,
        "FP_both": 0,
        "FP_near": 0,
        "FP_other": 0,
        "FP_highlight": 0,
        "FP_texture_shadow": 0,
    }

    fn_rows: List[dict] = []
    fp_rows: List[dict] = []

    for img_path, gt_dets in gt_map.items():
        img = cv2.imread(img_path)
        if img is None:
            img = np.zeros((10, 10, 3), dtype=np.uint8)
        pred_dets = pred_map.get(img_path, [])
        pred_valid = [p for p in pred_dets if p.score >= score_thr]
        matches, gt_unmatched, pred_unmatched_valid = _match_one_to_one_for_image(
            gt_dets, pred_valid, score_thr=score_thr, iou_thr=iou_thr
        )
        pred_unmatched = pred_unmatched_valid

        # FN mechanisms
        for gi in gt_unmatched:
            g = gt_dets[gi]
            same_cls = [p for p in pred_valid if p.label == g.label]
            if not same_cls:
                mtype = "no_response"
                best_iou = 0.0
                best_score = 0.0
            else:
                boxes = np.asarray([p.box for p in same_cls], dtype=np.float32)
                ious = iou_one_to_many(g.box, boxes)
                k = int(np.argmax(ious)) if ious.size else 0
                best_iou = float(ious[k]) if ious.size else 0.0
                best_score = float(same_cls[k].score)

                if best_iou < 0.1 and best_score < 0.05:
                    mtype = "no_response"
                elif best_score < score_thr:
                    mtype = "low_score"
                elif best_iou < iou_thr:
                    mtype = "regression_poor"
                else:
                    mtype = "postproc_suppressed"

            fn_counter[mtype] += 1
            fn_rows.append(
                {
                    "image_path": img_path,
                    "gt_index": gi,
                    "diag_type": mtype,
                    "best_iou": f"{best_iou:.6f}",
                    "best_score": f"{best_score:.6f}",
                    "gt_box": ",".join([f"{x:.2f}" for x in g.box]),
                }
            )

        # FP structure + texture/highlight
        gt_boxes_all = np.asarray([g.box for g in gt_dets], dtype=np.float32) if gt_dets else np.zeros((0, 4), dtype=np.float32)
        # pred-dup condition: one GT overlapped by >=2 predicted boxes (IoU >= iou_thr).
        pred_dup_idx = set()
        if gt_boxes_all.shape[0] > 0 and len(pred_valid) > 1:
            pred_boxes_arr = np.asarray([p.box for p in pred_valid], dtype=np.float32).reshape(-1, 4)
            iou_mat = _iou_matrix_xyxy(gt_boxes_all, pred_boxes_arr)
            for gi in range(iou_mat.shape[0]):
                cand = np.where(iou_mat[gi] >= float(iou_thr))[0].tolist()
                if len(cand) >= 2:
                    for pi in cand:
                        pred_dup_idx.add(int(pi))
        for pi in pred_unmatched:
            p = pred_valid[pi]
            best_iou = 0.0
            if gt_boxes_all.shape[0] > 0:
                ious = iou_one_to_many(p.box, gt_boxes_all)
                best_iou = float(np.max(ious)) if ious.size else 0.0
            is_unmatched = best_iou < float(iou_thr)
            is_pred_dup = pi in pred_dup_idx
            if is_unmatched and is_pred_dup:
                fptype = "FP_both"
            elif is_unmatched:
                fptype = "FP_unmatched"
            elif is_pred_dup:
                fptype = "FP_pred_dup"
            else:
                fptype = "FP_other"

            fp_counter[fptype] = fp_counter.get(fptype, 0) + 1
            tex_type = _fp_texture_type(img, p.box)
            if tex_type == "highlight":
                fp_counter["FP_highlight"] += 1
            elif tex_type == "texture_shadow":
                fp_counter["FP_texture_shadow"] += 1

            fp_rows.append(
                {
                    "image_path": img_path,
                    "pred_index": pi,
                    "fp_type": fptype,
                    "texture_type": tex_type,
                    "best_iou": f"{best_iou:.6f}",
                    "pred_score": f"{p.score:.6f}",
                    "pred_box": ",".join([f"{x:.2f}" for x in p.box]),
                }
            )

    fn_table = [
        {
            "diag_type": k,
            "count": int(v),
        }
        for k, v in fn_counter.items()
    ]

    fp_table = [
        {
            "metric": k,
            "count": int(v),
        }
        for k, v in fp_counter.items()
    ]

    return fn_table, fp_table, {"fn_cases": fn_rows, "fp_cases": fp_rows}


def color_for_model(idx: int) -> Tuple[int, int, int]:
    palette = [
        (46, 204, 113),
        (52, 152, 219),
        (231, 76, 60),
        (241, 196, 15),
        (155, 89, 182),
        (26, 188, 156),
    ]
    return palette[idx % len(palette)]


def draw_boxes(
    image: np.ndarray,
    dets: List[Det],
    score_thr: float,
    color: Tuple[int, int, int],
    label_prefix: str,
) -> np.ndarray:
    out = image.copy()
    for d in dets:
        if d.score < score_thr:
            continue
        x1, y1, x2, y2 = [int(round(v)) for v in d.box]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        txt = f"{label_prefix}:{d.score:.2f}"
        cv2.putText(out, txt, (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    return out


def draw_gt(image: np.ndarray, gts: List[Det]) -> np.ndarray:
    out = image.copy()
    for g in gts:
        x1, y1, x2, y2 = [int(round(v)) for v in g.box]
        cv2.rectangle(out, (x1, y1), (x2, y2), (255, 255, 255), 2)
    cv2.putText(out, "GT", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def stack_h(images: List[np.ndarray]) -> np.ndarray:
    h = max(img.shape[0] for img in images)
    fixed = []
    for img in images:
        if img.shape[0] != h:
            w = int(round(img.shape[1] * (h / img.shape[0])))
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        fixed.append(img)
    return np.concatenate(fixed, axis=1)


def image_highlight_ratio(img: np.ndarray) -> float:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray >= 230))


def select_baseline_and_best(compare_rows: List[dict], model_specs: List[ModelSpec]) -> Tuple[Optional[str], Optional[str], str]:
    ok_rows = [r for r in compare_rows if str(r.get("status", "")) == "ok"]
    if not ok_rows:
        return None, None, "no model evaluated successfully"

    spec_map = {m.model_name: m for m in model_specs}

    baseline_candidates = [r for r in ok_rows if "baseline" in str(r.get("module_groups", ""))]
    if not baseline_candidates:
        # fallback: min module tag count
        ok_rows_sorted = sorted(ok_rows, key=lambda r: int(r.get("n_tags", 99)))
        baseline = ok_rows_sorted[0]["model_name"]
        note = "baseline not explicitly found; used model with minimum ablation tags"
    else:
        baseline = baseline_candidates[0]["model_name"]
        note = "baseline selected by module_groups contains baseline"

    best = sorted(ok_rows, key=lambda r: float(r.get("map50", 0.0)), reverse=True)[0]["model_name"]
    return baseline, best, note


def short_exc(exc: BaseException) -> str:
    s = f"{type(exc).__name__}: {exc}"
    return s[:500]


def now_str() -> str:
    from datetime import datetime

    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def append_log(log_path: Path, text: str) -> None:
    ensure_dir(log_path.parent)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"[{now_str()}] {text}\n")
