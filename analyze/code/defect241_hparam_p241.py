#!/usr/bin/env python3
"""P2.4.1 hyper-parameter validation runner and analyzer.

This script is designed for the current YOLO project layout and does not modify
training code. It orchestrates baseline vs a3+c5 runs through existing shell
entrypoints, then analyzes best/last checkpoints with image-level KPI.

Outputs are written to:
  /home/ubuntu/hpproject/yolo/analyze/result/report_yymmddHHMM/

Main KPI:
  ImageRecall@FPR<=0.40 (partial ROC area on image-level scores).
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import gc
import json
import math
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import yaml  # type: ignore
except Exception as exc:  # pragma: no cover
    yaml = None  # type: ignore
    YAML_IMPORT_ERROR = exc
else:
    YAML_IMPORT_ERROR = None

try:
    from ultralytics import YOLO  # type: ignore
except Exception as exc:  # pragma: no cover
    YOLO = None  # type: ignore
    ULTRALYTICS_IMPORT_ERROR = exc
else:
    ULTRALYTICS_IMPORT_ERROR = None

try:
    from scipy.optimize import linear_sum_assignment  # type: ignore
except Exception:
    linear_sum_assignment = None  # type: ignore


ROOT = Path("/home/ubuntu/hpproject/yolo").resolve()
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class CaseSpec:
    variant: str  # baseline | a3c5
    step: str  # step0 | step1 | step2
    case_name: str
    batch: int
    lr0: float
    lrf: float
    warmup_epochs: float
    patience: int
    epochs: int
    seed: int
    grad_accum: int
    lr_mode: str  # fixed | scaled | align
    repeat_id: int


@dataclass
class RunRecord:
    case_name: str
    variant: str
    step: str
    status: int
    started_at: str
    ended_at: str
    duration_sec: float
    run_name: str
    config_path: str
    exp_dir: str
    log_path: str
    command: str


def ensure_deps() -> None:
    if yaml is None:
        raise ImportError(
            "Failed to import PyYAML. Please install pyyaml first. "
            f"Original error: {YAML_IMPORT_ERROR}"
        )


def ensure_ultralytics() -> None:
    if YOLO is None:
        raise ImportError(
            "Failed to import ultralytics. Activate yolo11 env first. "
            f"Original error: {ULTRALYTICS_IMPORT_ERROR}"
        )


def load_yaml(path: Path) -> Dict[str, Any]:
    ensure_deps()
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML must be mapping: {path}")
    return data


def dump_yaml(path: Path, data: Dict[str, Any]) -> None:
    ensure_deps()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def make_report_dir(out_root: Path) -> Path:
    out_root.mkdir(parents=True, exist_ok=True)
    ts = dt.datetime.now().strftime("report_%y%m%d%H%M")
    report_dir = out_root / ts
    if not report_dir.exists():
        report_dir.mkdir(parents=True, exist_ok=False)
        return report_dir
    i = 1
    while True:
        cand = out_root / f"{ts}_{i:02d}"
        if not cand.exists():
            cand.mkdir(parents=True, exist_ok=False)
            return cand
        i += 1


def normalize_steps(raw: Sequence[str]) -> List[str]:
    out: List[str] = []
    for x in raw:
        s = str(x).strip().lower()
        if s in {"0", "step0"}:
            out.append("step0")
        elif s in {"1", "step1"}:
            out.append("step1")
        elif s in {"2", "step2"}:
            out.append("step2")
        else:
            raise ValueError(f"Unsupported step: {x}")
    uniq = []
    for s in out:
        if s not in uniq:
            uniq.append(s)
    return uniq


def normalize_variants(raw: Sequence[str]) -> List[str]:
    out: List[str] = []
    for x in raw:
        s = str(x).strip().lower()
        if s in {"baseline", "base"}:
            out.append("baseline")
        elif s in {"a3c5", "a3+c5", "a3_c5"}:
            out.append("a3c5")
        else:
            raise ValueError(f"Unsupported variant: {x}")
    uniq = []
    for s in out:
        if s not in uniq:
            uniq.append(s)
    return uniq


def build_cases(args: argparse.Namespace) -> List[CaseSpec]:
    cases: List[CaseSpec] = []
    lr_ref = float(args.lr_ref)
    for variant in args.variants:
        for repeat_id in range(int(args.repeats)):
            if "step0" in args.steps:
                cases.append(
                    CaseSpec(
                        variant=variant,
                        step="step0",
                        case_name=f"{variant}_step0_r{repeat_id+1}",
                        batch=10,
                        lr0=lr_ref,
                        lrf=float(args.lrf),
                        warmup_epochs=float(args.warmup_epochs),
                        patience=int(args.patience),
                        epochs=int(args.epochs_step0),
                        seed=int(args.seed_step0),
                        grad_accum=1,
                        lr_mode="fixed",
                        repeat_id=repeat_id + 1,
                    )
                )
            if "step1" in args.steps:
                for batch in (6, 10, 12):
                    lr_fixed = lr_ref
                    lr_scaled = lr_ref * batch / 10.0
                    cases.append(
                        CaseSpec(
                            variant=variant,
                            step="step1",
                            case_name=f"{variant}_step1_b{batch}_fixed_r{repeat_id+1}",
                            batch=batch,
                            lr0=lr_fixed,
                            lrf=float(args.lrf),
                            warmup_epochs=float(args.warmup_epochs),
                            patience=int(args.patience),
                            epochs=int(args.epochs_step1),
                            seed=int(args.seed_step1),
                            grad_accum=1,
                            lr_mode="fixed",
                            repeat_id=repeat_id + 1,
                        )
                    )
                    cases.append(
                        CaseSpec(
                            variant=variant,
                            step="step1",
                            case_name=f"{variant}_step1_b{batch}_scaled_r{repeat_id+1}",
                            batch=batch,
                            lr0=lr_scaled,
                            lrf=float(args.lrf),
                            warmup_epochs=float(args.warmup_epochs),
                            patience=int(args.patience),
                            epochs=int(args.epochs_step1),
                            seed=int(args.seed_step1),
                            grad_accum=1,
                            lr_mode="scaled",
                            repeat_id=repeat_id + 1,
                        )
                    )
            if "step2" in args.steps:
                # Current train.py does not expose explicit grad_accum argument.
                # We still record intended accum setting for diagnosis.
                cases.append(
                    CaseSpec(
                        variant=variant,
                        step="step2",
                        case_name=f"{variant}_step2_b6_acc2_r{repeat_id+1}",
                        batch=6,
                        lr0=lr_ref * 6.0 / 10.0,
                        lrf=float(args.lrf),
                        warmup_epochs=float(args.warmup_epochs),
                        patience=int(args.patience),
                        epochs=int(args.epochs_step2),
                        seed=int(args.seed_step2),
                        grad_accum=2,
                        lr_mode="align",
                        repeat_id=repeat_id + 1,
                    )
                )
                cases.append(
                    CaseSpec(
                        variant=variant,
                        step="step2",
                        case_name=f"{variant}_step2_b10_acc1_r{repeat_id+1}",
                        batch=10,
                        lr0=lr_ref,
                        lrf=float(args.lrf),
                        warmup_epochs=float(args.warmup_epochs),
                        patience=int(args.patience),
                        epochs=int(args.epochs_step2),
                        seed=int(args.seed_step2),
                        grad_accum=1,
                        lr_mode="align",
                        repeat_id=repeat_id + 1,
                    )
                )
    for c in cases:
        if c.epochs > 200:
            raise ValueError(f"Epochs must be <=200 by constraint: {c.case_name} -> {c.epochs}")
    return cases


def patch_cfg_for_case(base_cfg: Dict[str, Any], case: CaseSpec, run_name: str, args: argparse.Namespace) -> Dict[str, Any]:
    cfg = dict(base_cfg)
    cfg["epochs"] = int(case.epochs)
    cfg["batch"] = int(case.batch)
    cfg["lr0"] = float(case.lr0)
    cfg["lrf"] = float(case.lrf)
    cfg["warmup_epochs"] = float(case.warmup_epochs)
    cfg["patience"] = int(case.patience)
    cfg["seed"] = int(case.seed)
    cfg["mode"] = "train_test"
    cfg["run_name"] = run_name
    cfg["save_val_pic"] = True
    cfg["save_test_pic"] = True

    # Keep P2.3.0 style postprocess defaults for downstream analysis.
    cfg["conf"] = float(args.conf_op)
    cfg["match_iou"] = float(args.match_iou)
    cfg["nms_iou"] = float(args.nms_iou)
    cfg["max_det"] = int(args.max_det)
    cfg["metric_conf"] = float(args.metric_conf)
    cfg["eval_batch"] = int(args.eval_batch)
    cfg["eval_device"] = str(args.eval_device)
    return cfg


def glob_exp_dirs_by_run_name(run_name: str) -> List[Path]:
    pattern = f"*/{run_name}/exp_*"
    roots = sorted((ROOT / "experiments" / "yolo11").glob(pattern))
    return [p.resolve() for p in roots if p.is_dir()]


def newest_path(paths: Sequence[Path]) -> Optional[Path]:
    if not paths:
        return None
    return max(paths, key=lambda p: p.stat().st_mtime)


def run_case(case: CaseSpec, args: argparse.Namespace, report_dir: Path, case_cfg_path: Path) -> RunRecord:
    run_name = f"{args.run_name_prefix}_{case.case_name}"
    before = set(str(p) for p in glob_exp_dirs_by_run_name(run_name))

    log_dir = report_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{case.case_name}.log"

    if case.variant == "baseline":
        cmd = ["bash", "tools/run_yolov11.sh", str(case_cfg_path)]
    else:
        cmd = [
            "bash",
            "tools/run_yolov11_241.sh",
            "--vram-guard",
            str(args.vram_guard),
            "--guard-max-gb",
            str(args.guard_max_gb),
            "--safe-batch",
            str(args.safe_batch),
            "--safe-workers",
            str(args.safe_workers),
            str(case_cfg_path),
            "a3",
            "c5",
        ]

    started_at = dt.datetime.now().isoformat(timespec="seconds")
    t0 = time.time()
    status = 0
    if args.dry_run:
        log_path.write_text("[dry-run] " + " ".join(cmd) + "\n", encoding="utf-8")
    else:
        with log_path.open("w", encoding="utf-8") as lf:
            proc = subprocess.run(cmd, cwd=str(ROOT), stdout=lf, stderr=subprocess.STDOUT, check=False)
            status = int(proc.returncode)
    ended_at = dt.datetime.now().isoformat(timespec="seconds")
    duration_sec = float(time.time() - t0)

    after = set(str(p) for p in glob_exp_dirs_by_run_name(run_name))
    created = sorted(after - before)
    if created:
        exp_dir = created[-1]
    else:
        latest = newest_path([Path(p) for p in after])
        exp_dir = str(latest) if latest is not None else ""

    return RunRecord(
        case_name=case.case_name,
        variant=case.variant,
        step=case.step,
        status=status,
        started_at=started_at,
        ended_at=ended_at,
        duration_sec=duration_sec,
        run_name=run_name,
        config_path=str(case_cfg_path),
        exp_dir=exp_dir,
        log_path=str(log_path),
        command=" ".join(cmd),
    )


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def list_images(source: Path) -> List[Path]:
    if source.is_file() and source.suffix.lower() == ".txt":
        out: List[Path] = []
        with source.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                p = Path(line.strip())
                if p.suffix.lower() in IMG_EXTS:
                    out.append(p)
        return out
    if not source.is_dir():
        return []
    return sorted([p for p in source.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS])


def infer_label_path(image_path: Path) -> Path:
    parts = list(image_path.parts)
    if "images" in parts:
        idx = parts.index("images")
        parts[idx] = "labels"
        return Path(*parts).with_suffix(".txt")
    if "image" in parts:
        idx = parts.index("image")
        parts[idx] = "labels"
        return Path(*parts).with_suffix(".txt")
    return image_path.parent.parent / "labels" / f"{image_path.stem}.txt"


def image_has_gt(image_path: Path) -> bool:
    lp = infer_label_path(image_path)
    if not lp.exists():
        return False
    with lp.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.strip():
                return True
    return False


def load_gt_boxes_xyxy(image_path: Path, im_w: int, im_h: int) -> np.ndarray:
    lp = infer_label_path(image_path)
    if not lp.exists():
        return np.zeros((0, 4), dtype=np.float32)
    rows: List[List[float]] = []
    with lp.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            try:
                xc, yc, w, h = map(float, parts[1:5])
            except ValueError:
                continue
            x1 = (xc - w / 2.0) * im_w
            y1 = (yc - h / 2.0) * im_h
            x2 = (xc + w / 2.0) * im_w
            y2 = (yc + h / 2.0) * im_h
            rows.append([x1, y1, x2, y2])
    if not rows:
        return np.zeros((0, 4), dtype=np.float32)
    return np.array(rows, dtype=np.float32)


def compute_iou_matrix(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    if gt.size == 0 or pred.size == 0:
        return np.zeros((gt.shape[0], pred.shape[0]), dtype=np.float32)
    ix1 = np.maximum(gt[:, None, 0], pred[None, :, 0])
    iy1 = np.maximum(gt[:, None, 1], pred[None, :, 1])
    ix2 = np.minimum(gt[:, None, 2], pred[None, :, 2])
    iy2 = np.minimum(gt[:, None, 3], pred[None, :, 3])
    iw = np.maximum(0.0, ix2 - ix1)
    ih = np.maximum(0.0, iy2 - iy1)
    inter = iw * ih
    area_gt = np.maximum(0.0, (gt[:, 2] - gt[:, 0]) * (gt[:, 3] - gt[:, 1]))
    area_pred = np.maximum(0.0, (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1]))
    union = area_gt[:, None] + area_pred[None, :] - inter
    return np.where(union > 0, inter / union, 0.0).astype(np.float32)


def one_to_one_match(iou_mat: np.ndarray, thr: float) -> Tuple[List[int], List[int]]:
    if iou_mat.size == 0:
        return [], []
    if linear_sum_assignment is not None:
        cost = (1.0 - iou_mat).astype(np.float64, copy=False)
        rows, cols = linear_sum_assignment(cost)
        keep = iou_mat[rows, cols] >= float(thr)
        return rows[keep].tolist(), cols[keep].tolist()

    pairs: List[Tuple[float, int, int]] = []
    for gi in range(iou_mat.shape[0]):
        for pi in range(iou_mat.shape[1]):
            v = float(iou_mat[gi, pi])
            if v >= float(thr):
                pairs.append((v, gi, pi))
    pairs.sort(reverse=True, key=lambda x: x[0])
    used_g: set[int] = set()
    used_p: set[int] = set()
    gsel: List[int] = []
    psel: List[int] = []
    for _, gi, pi in pairs:
        if gi in used_g or pi in used_p:
            continue
        used_g.add(gi)
        used_p.add(pi)
        gsel.append(gi)
        psel.append(pi)
    return gsel, psel


def compute_curve_metrics(labels: np.ndarray, scores: np.ndarray, fpr_cap: float) -> Dict[str, float]:
    if labels.size == 0:
        return {
            "kpi_partial_auc_raw": 0.0,
            "kpi_partial_auc_norm": 0.0,
            "best_recall_under_fpr_cap": 0.0,
            "best_threshold_under_fpr_cap": 0.0,
        }

    thresholds = np.unique(np.concatenate([np.array([0.0], dtype=np.float32), scores.astype(np.float32), np.array([1.0], dtype=np.float32)]))
    thresholds.sort()
    thresholds = thresholds[::-1]
    pos = int(labels.sum())
    neg = int(labels.size - pos)
    recall: List[float] = []
    fpr: List[float] = []
    thr_keep: List[float] = []
    best_rec = 0.0
    best_thr = 0.0
    for thr in thresholds.tolist():
        pred = (scores >= float(thr)).astype(np.int32)
        tp = int(((pred == 1) & (labels == 1)).sum())
        fp = int(((pred == 1) & (labels == 0)).sum())
        rec = float(tp / pos) if pos > 0 else 0.0
        fpr_v = float(fp / neg) if neg > 0 else 0.0
        recall.append(rec)
        fpr.append(fpr_v)
        thr_keep.append(float(thr))
        if fpr_v <= float(fpr_cap) and rec >= best_rec:
            best_rec = rec
            best_thr = float(thr)

    recall_np = np.array(recall, dtype=np.float32)
    fpr_np = np.array(fpr, dtype=np.float32)
    order = np.argsort(fpr_np)
    x = fpr_np[order]
    y = recall_np[order]
    x2 = np.clip(x, 0.0, float(fpr_cap))
    cap = float(fpr_cap)
    if x2.size == 0:
        auc_raw = 0.0
    else:
        # Keep only points <= cap, add edge point at cap.
        mask = x <= cap
        x_sub = x[mask]
        y_sub = y[mask]
        if x_sub.size == 0:
            auc_raw = 0.0
        else:
            if float(x_sub[-1]) < cap:
                x_sub = np.concatenate([x_sub, np.array([cap], dtype=np.float32)])
                y_sub = np.concatenate([y_sub, np.array([float(y_sub[-1])], dtype=np.float32)])
            auc_raw = float(np.trapz(y_sub, x_sub))
    auc_norm = float(auc_raw / cap) if cap > 0 else 0.0
    return {
        "kpi_partial_auc_raw": float(auc_raw),
        "kpi_partial_auc_norm": float(auc_norm),
        "best_recall_under_fpr_cap": float(best_rec),
        "best_threshold_under_fpr_cap": float(best_thr),
    }


def resolve_data_sources(config_yaml: Path) -> Tuple[List[Path], List[Path]]:
    cfg = load_yaml(config_yaml)
    data_yaml = Path(str(cfg.get("data", ""))).resolve()
    if not data_yaml.exists():
        return [], []
    data_info = load_yaml(data_yaml)
    data_root = str(cfg.get("data_root", "")).strip()
    if data_root:
        roots = [Path(data_root).resolve()]
    else:
        root_val = str(data_info.get("path", "")).strip()
        roots = [Path(root_val).resolve()] if root_val else [data_yaml.parent.resolve()]

    val_entry = str(data_info.get("val", "images/val"))
    test_entry = str(data_info.get("test", "")).strip()
    val_sources: List[Path] = []
    test_sources: List[Path] = []
    for dr in roots:
        v = (dr / val_entry).resolve() if not Path(val_entry).is_absolute() else Path(val_entry).resolve()
        if v.exists():
            val_sources.append(v)
        if test_entry:
            t = (dr / test_entry).resolve() if not Path(test_entry).is_absolute() else Path(test_entry).resolve()
            if t.exists():
                test_sources.append(t)
    return val_sources, test_sources


def _predict_chunk(
    model: Any,
    image_paths: Sequence[Path],
    conf: float,
    nms_iou: float,
    max_det: int,
    batch: int,
    device: str,
) -> Dict[str, Dict[str, Any]]:
    if not image_paths:
        return {}
    results = model.predict(
        source=[str(p) for p in image_paths],
        conf=float(conf),
        iou=float(nms_iou),
        max_det=int(max_det),
        save=False,
        verbose=False,
        batch=int(batch),
        device=str(device) if device else None,
    )
    out: Dict[str, Dict[str, Any]] = {}
    for p, res in zip(image_paths, results):
        item: Dict[str, Any] = {
            "score_max": 0.0,
            "xyxy": np.zeros((0, 4), dtype=np.float32),
            "conf": np.zeros((0,), dtype=np.float32),
            "orig_shape": getattr(res, "orig_shape", None),
        }
        boxes = getattr(res, "boxes", None)
        if boxes is not None and getattr(boxes, "conf", None) is not None and len(boxes.conf) > 0:
            confs = boxes.conf.detach().cpu().numpy().astype(np.float32)
            if getattr(boxes, "xyxy", None) is not None:
                xyxy = boxes.xyxy.detach().cpu().numpy().astype(np.float32)
            else:
                xyxy = np.zeros((len(confs), 4), dtype=np.float32)
            item["score_max"] = float(np.max(confs))
            item["xyxy"] = xyxy
            item["conf"] = confs
        out[str(p.resolve())] = item
    del results
    gc.collect()
    return out


def evaluate_weight(
    weight_path: Path,
    split_sources: Dict[str, List[Path]],
    conf_op: float,
    match_iou: float,
    nms_iou: float,
    max_det: int,
    metric_conf: float,
    eval_batch: int,
    eval_device: str,
    fpr_cap: float,
    small_thr: float,
) -> Dict[str, Dict[str, Any]]:
    ensure_ultralytics()
    model = YOLO(str(weight_path))
    split_result: Dict[str, Dict[str, Any]] = {}
    for split, sources in split_sources.items():
        image_paths: List[Path] = []
        for src in sources:
            image_paths.extend(list_images(src))
        image_paths = sorted({str(p.resolve()): p for p in image_paths}.values(), key=lambda p: str(p))

        if not image_paths:
            split_result[split] = {
                "images_total": 0,
                "img_hit": 0,
                "img_miss": 0,
                "img_false_alarm": 0,
                "img_true_negative": 0,
                "obj_tp": 0,
                "obj_fp": 0,
                "obj_fn": 0,
                "small_gt": 0,
                "small_tp": 0,
                "small_recall": 0.0,
                "kpi_partial_auc_raw": 0.0,
                "kpi_partial_auc_norm": 0.0,
                "best_recall_under_fpr_cap": 0.0,
                "best_threshold_under_fpr_cap": 0.0,
                "fpr_cap": float(fpr_cap),
            }
            continue

        pred_map: Dict[str, Dict[str, Any]] = {}
        chunk = 128
        for i in range(0, len(image_paths), chunk):
            pred_map.update(
                _predict_chunk(
                    model=model,
                    image_paths=image_paths[i : i + chunk],
                    conf=float(metric_conf),
                    nms_iou=float(nms_iou),
                    max_det=int(max_det),
                    batch=int(eval_batch),
                    device=str(eval_device),
                )
            )

        labels: List[int] = []
        scores: List[float] = []
        img_hit = img_miss = img_false_alarm = img_true_negative = 0
        obj_tp = obj_fp = obj_fn = 0
        small_gt = small_tp = 0

        for img_path in image_paths:
            key = str(img_path.resolve())
            pred = pred_map.get(key, {})
            score_max = float(pred.get("score_max", 0.0))
            xyxy = pred.get("xyxy", np.zeros((0, 4), dtype=np.float32))
            confs = pred.get("conf", np.zeros((0,), dtype=np.float32))
            if not isinstance(xyxy, np.ndarray):
                xyxy = np.zeros((0, 4), dtype=np.float32)
            if not isinstance(confs, np.ndarray):
                confs = np.zeros((0,), dtype=np.float32)

            has_gt = image_has_gt(img_path)
            labels.append(1 if has_gt else 0)
            scores.append(score_max)

            pred_pos = score_max >= float(conf_op)
            if has_gt and pred_pos:
                img_hit += 1
            elif has_gt and not pred_pos:
                img_miss += 1
            elif (not has_gt) and pred_pos:
                img_false_alarm += 1
            else:
                img_true_negative += 1

            # Object-level diagnosis at operating threshold.
            if pred_pos and confs.size > 0:
                mask = confs >= float(conf_op)
                pred_boxes = xyxy[mask]
            else:
                pred_boxes = np.zeros((0, 4), dtype=np.float32)

            orig_shape = pred.get("orig_shape", None)
            if isinstance(orig_shape, (tuple, list)) and len(orig_shape) >= 2:
                h, w = int(orig_shape[0]), int(orig_shape[1])
            else:
                # Fallback by image decode only when metadata is missing.
                try:
                    import cv2  # type: ignore

                    img = cv2.imread(str(img_path))
                    if img is not None:
                        h, w = int(img.shape[0]), int(img.shape[1])
                    else:
                        h, w = 1, 1
                except Exception:
                    h, w = 1, 1

            gt_boxes = load_gt_boxes_xyxy(img_path, w, h)
            iou_mat = compute_iou_matrix(gt_boxes, pred_boxes)
            m_gt, m_pred = one_to_one_match(iou_mat, float(match_iou))
            tp = len(m_gt)
            fp = int(pred_boxes.shape[0]) - tp
            fn = int(gt_boxes.shape[0]) - tp
            obj_tp += int(tp)
            obj_fp += int(fp)
            obj_fn += int(fn)

            if gt_boxes.shape[0] > 0:
                gt_w = np.maximum(0.0, gt_boxes[:, 2] - gt_boxes[:, 0])
                gt_h = np.maximum(0.0, gt_boxes[:, 3] - gt_boxes[:, 1])
                gt_short = np.minimum(gt_w, gt_h)
                small_mask = gt_short <= float(small_thr)
                small_idx = np.where(small_mask)[0].tolist()
                small_gt += len(small_idx)
                if m_gt:
                    matched_small = len(set(small_idx) & set(m_gt))
                    small_tp += int(matched_small)

        labels_np = np.array(labels, dtype=np.int32)
        scores_np = np.array(scores, dtype=np.float32)
        curve_metrics = compute_curve_metrics(labels_np, scores_np, float(fpr_cap))
        split_result[split] = {
            "images_total": int(len(image_paths)),
            "img_hit": int(img_hit),
            "img_miss": int(img_miss),
            "img_false_alarm": int(img_false_alarm),
            "img_true_negative": int(img_true_negative),
            "obj_tp": int(obj_tp),
            "obj_fp": int(obj_fp),
            "obj_fn": int(obj_fn),
            "small_gt": int(small_gt),
            "small_tp": int(small_tp),
            "small_recall": float(small_tp / small_gt) if small_gt > 0 else 0.0,
            "kpi_partial_auc_raw": float(curve_metrics["kpi_partial_auc_raw"]),
            "kpi_partial_auc_norm": float(curve_metrics["kpi_partial_auc_norm"]),
            "best_recall_under_fpr_cap": float(curve_metrics["best_recall_under_fpr_cap"]),
            "best_threshold_under_fpr_cap": float(curve_metrics["best_threshold_under_fpr_cap"]),
            "fpr_cap": float(fpr_cap),
        }

    # merge "all" by summing confusion/object counts, recompute ratios
    merge_keys = [
        "images_total",
        "img_hit",
        "img_miss",
        "img_false_alarm",
        "img_true_negative",
        "obj_tp",
        "obj_fp",
        "obj_fn",
        "small_gt",
        "small_tp",
    ]
    all_row = {k: 0 for k in merge_keys}
    auc_norm_vals: List[float] = []
    best_rec_vals: List[float] = []
    for sp in split_result.values():
        for k in merge_keys:
            all_row[k] += int(sp.get(k, 0))
        auc_norm_vals.append(float(sp.get("kpi_partial_auc_norm", 0.0)))
        best_rec_vals.append(float(sp.get("best_recall_under_fpr_cap", 0.0)))
    all_row["small_recall"] = float(all_row["small_tp"] / all_row["small_gt"]) if all_row["small_gt"] > 0 else 0.0
    all_row["kpi_partial_auc_raw"] = float(np.mean(auc_norm_vals) * fpr_cap) if auc_norm_vals else 0.0
    all_row["kpi_partial_auc_norm"] = float(np.mean(auc_norm_vals)) if auc_norm_vals else 0.0
    all_row["best_recall_under_fpr_cap"] = float(np.mean(best_rec_vals)) if best_rec_vals else 0.0
    all_row["best_threshold_under_fpr_cap"] = 0.0
    all_row["fpr_cap"] = float(fpr_cap)
    split_result["all"] = all_row

    try:
        del model
    except Exception:
        pass
    gc.collect()
    return split_result


def moving_average(values: np.ndarray, win: int = 3) -> np.ndarray:
    if values.size == 0:
        return values
    win = max(1, int(win))
    if win == 1:
        return values.copy()
    out = np.convolve(values, np.ones(win, dtype=np.float32) / float(win), mode="same")
    return out.astype(np.float32)


def infer_progress_diagnosis(results_rows: List[Dict[str, str]]) -> Dict[str, Any]:
    if not results_rows:
        return {"reason": "train_results_missing"}

    def col(name: str) -> np.ndarray:
        vals: List[float] = []
        for r in results_rows:
            try:
                vals.append(float(r.get(name, "nan")))
            except Exception:
                vals.append(float("nan"))
        arr = np.array(vals, dtype=np.float32)
        return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    epochs = col("epoch")
    map5095 = col("metrics/mAP50-95(B)")
    if map5095.size == 0:
        return {"reason": "map5095_missing"}

    best_idx = int(np.argmax(map5095))
    best_epoch = int(epochs[best_idx]) if epochs.size else best_idx + 1
    smooth = moving_average(map5095, 3)
    best_smooth_idx = int(np.argmax(smooth))
    best_smooth_epoch = int(epochs[best_smooth_idx]) if epochs.size else best_smooth_idx + 1

    tail = map5095[-10:] if map5095.size >= 10 else map5095
    slope = float(np.polyfit(np.arange(tail.size), tail, 1)[0]) if tail.size >= 2 else 0.0
    final_epoch = int(epochs[-1]) if epochs.size else int(map5095.size)

    return {
        "best_epoch_proxy": int(best_epoch),
        "best_epoch_proxy_smooth3": int(best_smooth_epoch),
        "final_epoch": int(final_epoch),
        "tail_slope_map5095": float(slope),
        "best_near_end": bool(final_epoch - best_epoch <= 10),
        "smooth_shift_large": bool(abs(best_smooth_epoch - best_epoch) >= 10),
    }


def analyze_records(records: List[RunRecord], args: argparse.Namespace, report_dir: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    kpi_rows: List[Dict[str, Any]] = []
    run_summary_rows: List[Dict[str, Any]] = []
    for rr in records:
        run_summary: Dict[str, Any] = {
            "case_name": rr.case_name,
            "variant": rr.variant,
            "step": rr.step,
            "status": rr.status,
            "exp_dir": rr.exp_dir,
            "reasoning": "",
        }
        if rr.status != 0 or not rr.exp_dir:
            run_summary["reasoning"] = "run_failed_or_missing_exp_dir"
            run_summary_rows.append(run_summary)
            continue
        exp_dir = Path(rr.exp_dir)
        cfg_path = exp_dir / "config.yaml"
        results_csv = exp_dir / "train" / "results.csv"
        best_w = exp_dir / "train" / "weights" / "best.pt"
        last_w = exp_dir / "train" / "weights" / "last.pt"

        if not cfg_path.exists():
            run_summary["reasoning"] = "missing_config_yaml"
            run_summary_rows.append(run_summary)
            continue

        val_sources, test_sources = resolve_data_sources(cfg_path)
        split_sources = {"val": val_sources, "test": test_sources}

        weight_items: List[Tuple[str, Path]] = []
        if best_w.exists():
            weight_items.append(("best", best_w))
        if last_w.exists():
            weight_items.append(("last", last_w))
        if not weight_items:
            run_summary["reasoning"] = "no_best_last_weight"
            run_summary_rows.append(run_summary)
            continue

        progress_diag = infer_progress_diagnosis(read_csv_rows(results_csv))
        run_summary.update(progress_diag)

        per_kind_metrics: Dict[str, Dict[str, Any]] = {}
        for kind, wp in weight_items:
            try:
                split_metrics = evaluate_weight(
                    weight_path=wp,
                    split_sources=split_sources,
                    conf_op=float(args.conf_op),
                    match_iou=float(args.match_iou),
                    nms_iou=float(args.nms_iou),
                    max_det=int(args.max_det),
                    metric_conf=float(args.metric_conf),
                    eval_batch=int(args.eval_batch),
                    eval_device=str(args.eval_device),
                    fpr_cap=float(args.kpi_fpr_cap),
                    small_thr=float(args.small_thr),
                )
            except Exception as exc:
                run_summary[f"{kind}_eval_error"] = str(exc)
                continue
            per_kind_metrics[kind] = split_metrics
            for split, row in split_metrics.items():
                out = {
                    "case_name": rr.case_name,
                    "variant": rr.variant,
                    "step": rr.step,
                    "weight_kind": kind,
                    "split": split,
                    "exp_dir": rr.exp_dir,
                    "weight_path": str(wp),
                }
                out.update(row)
                kpi_rows.append(out)

        best_all = per_kind_metrics.get("best", {}).get("all", {})
        last_all = per_kind_metrics.get("last", {}).get("all", {})
        if best_all and last_all:
            delta = float(last_all.get("kpi_partial_auc_norm", 0.0)) - float(best_all.get("kpi_partial_auc_norm", 0.0))
            run_summary["delta_last_minus_best_kpi"] = float(delta)
            if delta > float(args.kpi_noise_eps):
                reason = "best_selection_mismatch_or_noise"
            elif bool(run_summary.get("best_near_end", False)) and float(run_summary.get("tail_slope_map5095", 0.0)) > 0.0:
                reason = "likely_under_converged"
            elif bool(run_summary.get("smooth_shift_large", False)):
                reason = "likely_metric_noise"
            else:
                reason = "stable_or_unclear"
            run_summary["reasoning"] = reason
        else:
            run_summary["reasoning"] = "incomplete_best_last_eval"

        run_summary_rows.append(run_summary)
    return kpi_rows, run_summary_rows


def summarize_step1_robustness(kpi_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # Compare variance across batch for fixed vs scaled within each variant.
    rows = [r for r in kpi_rows if r.get("step") == "step1" and r.get("split") == "all" and r.get("weight_kind") == "best"]
    out: List[Dict[str, Any]] = []
    for variant in sorted({str(r.get("variant")) for r in rows}):
        fixed_vals: List[float] = []
        scaled_vals: List[float] = []
        for r in rows:
            if str(r.get("variant")) != variant:
                continue
            case_name = str(r.get("case_name", ""))
            val = float(r.get("kpi_partial_auc_norm", 0.0))
            if "_fixed_" in case_name:
                fixed_vals.append(val)
            elif "_scaled_" in case_name:
                scaled_vals.append(val)
        if not fixed_vals and not scaled_vals:
            continue
        fixed_std = float(np.std(np.array(fixed_vals, dtype=np.float32))) if fixed_vals else 0.0
        scaled_std = float(np.std(np.array(scaled_vals, dtype=np.float32))) if scaled_vals else 0.0
        out.append(
            {
                "variant": variant,
                "fixed_mean_kpi": float(np.mean(fixed_vals)) if fixed_vals else 0.0,
                "fixed_std_kpi": fixed_std,
                "scaled_mean_kpi": float(np.mean(scaled_vals)) if scaled_vals else 0.0,
                "scaled_std_kpi": scaled_std,
                "scaled_more_robust": bool(scaled_std < fixed_std) if (fixed_vals and scaled_vals) else False,
            }
        )
    return out


def write_markdown_report(
    report_dir: Path,
    args: argparse.Namespace,
    records: List[RunRecord],
    run_summary_rows: List[Dict[str, Any]],
    step1_rows: List[Dict[str, Any]],
) -> None:
    lines: List[str] = []
    lines.append("# P2.4.1 Hyper-Parameter Validation Report")
    lines.append("")
    lines.append(f"- created_at: {dt.datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"- report_dir: {report_dir}")
    lines.append(f"- dry_run: {args.dry_run}")
    lines.append("")
    lines.append("## Scope")
    lines.append("- baseline is run via `bash tools/run_yolov11.sh`")
    lines.append("- a3+c5 is run via `bash tools/run_yolov11_241.sh <cfg> a3 c5`")
    lines.append("- training code is not modified")
    lines.append("")
    lines.append("## Main KPI")
    lines.append(f"- KPI: ImageRecall@FPR<={args.kpi_fpr_cap} (partial ROC area)")
    lines.append(f"- operating conf for image-level FP/FN: {args.conf_op}")
    lines.append(f"- object match IoU: {args.match_iou}")
    lines.append("")
    lines.append("## Run Status")
    total = len(records)
    ok = sum(1 for r in records if int(r.status) == 0)
    fail = total - ok
    lines.append(f"- total_cases: {total}")
    lines.append(f"- success: {ok}")
    lines.append(f"- fail: {fail}")
    lines.append("")
    lines.append("## Step1 Robustness (batch->lr scaling)")
    if step1_rows:
        for r in step1_rows:
            lines.append(
                f"- variant={r['variant']}, fixed_std={r['fixed_std_kpi']:.6f}, "
                f"scaled_std={r['scaled_std_kpi']:.6f}, scaled_more_robust={r['scaled_more_robust']}"
            )
    else:
        lines.append("- no step1 rows available")
    lines.append("")
    lines.append("## Best vs Last Diagnosis")
    if run_summary_rows:
        for r in run_summary_rows:
            lines.append(
                f"- {r.get('case_name')}: status={r.get('status')}, reason={r.get('reasoning')}, "
                f"delta_last_minus_best_kpi={r.get('delta_last_minus_best_kpi', 'NA')}"
            )
    else:
        lines.append("- no run summary rows")
    lines.append("")
    lines.append("## Important Constraint Notes")
    lines.append("- current `src/train.py` does not expose explicit `grad_accum` argument")
    lines.append("- Step2 records intended `grad_accum` for traceability; strict accum control requires train-entry support")
    lines.append("- if a run crashes before train end (e.g. CUDA OOM), only `train/` may exist and no `metrics/val_vis/test_vis` are produced")
    lines.append("")
    lines.append("## Commands")
    lines.append("```bash")
    lines.append(
        "python /home/ubuntu/hpproject/yolo/analyze/code/defect241_hparam_p241.py "
        "--variants baseline a3c5 --steps 0 1 2 --repeats 1"
    )
    lines.append("```")
    (report_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="P2.4.1 hyper-parameter validation runner/analyzer.")
    p.add_argument("--out_root", type=str, default="/home/ubuntu/hpproject/yolo/analyze/result")
    p.add_argument("--baseline_config", type=str, default="/home/ubuntu/hpproject/yolo/configs/baseline/datasetm6c.yaml")
    p.add_argument("--enhanced_base_config", type=str, default="/home/ubuntu/hpproject/yolo/configs/enhance/datasetm6c/defect241.yaml")
    p.add_argument("--run_name_prefix", type=str, default="p241")
    p.add_argument("--variants", nargs="+", default=["baseline", "a3c5"])
    p.add_argument("--steps", nargs="+", default=["0", "1", "2"])
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--execute", action="store_true", help="Execute training runs. Without this flag, only plan/analyze.")
    p.add_argument("--analyze_only", action="store_true", help="Skip training and only analyze matching existing runs.")

    p.add_argument("--epochs_step0", type=int, default=150)
    p.add_argument("--epochs_step1", type=int, default=150)
    p.add_argument("--epochs_step2", type=int, default=150)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--warmup_epochs", type=float, default=2.0)
    p.add_argument("--lrf", type=float, default=0.12)
    p.add_argument("--lr_ref", type=float, default=0.008)

    p.add_argument("--seed_step0", type=int, default=0)
    p.add_argument("--seed_step1", type=int, default=0)
    p.add_argument("--seed_step2", type=int, default=0)

    p.add_argument("--conf_op", type=float, default=0.3)
    p.add_argument("--match_iou", type=float, default=0.2)
    p.add_argument("--nms_iou", type=float, default=0.6)
    p.add_argument("--max_det", type=int, default=20)
    p.add_argument("--metric_conf", type=float, default=0.001)
    p.add_argument("--eval_batch", type=int, default=1)
    p.add_argument("--eval_device", type=str, default="")
    p.add_argument("--kpi_fpr_cap", type=float, default=0.4)
    p.add_argument("--small_thr", type=float, default=32.0)
    p.add_argument("--kpi_noise_eps", type=float, default=0.01)

    p.add_argument("--vram_guard", type=str, default="auto", choices=["auto", "on", "off"])
    p.add_argument("--guard_max_gb", type=float, default=10.0)
    p.add_argument("--safe_batch", type=int, default=6)
    p.add_argument("--safe_workers", type=int, default=4)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.steps = normalize_steps(args.steps)
    args.variants = normalize_variants(args.variants)
    ensure_deps()

    report_dir = make_report_dir(Path(args.out_root).resolve())
    cfg_dir = report_dir / "tmp_cfgs"
    cfg_dir.mkdir(parents=True, exist_ok=True)

    base_cfg_baseline = load_yaml(Path(args.baseline_config).resolve())
    base_cfg_enhanced = load_yaml(Path(args.enhanced_base_config).resolve())
    cases = build_cases(args)
    if not cases:
        raise RuntimeError("No cases generated")

    (report_dir / "plan.json").write_text(
        json.dumps(
            {
                "created_at": dt.datetime.now().isoformat(timespec="seconds"),
                "args": vars(args),
                "case_count": len(cases),
                "cases": [asdict(c) for c in cases],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    write_csv(report_dir / "plan.csv", [asdict(c) for c in cases])

    records: List[RunRecord] = []
    if args.execute and not args.analyze_only:
        for case in cases:
            run_name = f"{args.run_name_prefix}_{case.case_name}"
            base_cfg = base_cfg_baseline if case.variant == "baseline" else base_cfg_enhanced
            cfg = patch_cfg_for_case(base_cfg, case, run_name=run_name, args=args)
            cfg_path = cfg_dir / f"{case.case_name}.yaml"
            dump_yaml(cfg_path, cfg)
            rr = run_case(case=case, args=args, report_dir=report_dir, case_cfg_path=cfg_path)
            records.append(rr)
            print(
                f"[run] case={case.case_name} variant={case.variant} step={case.step} "
                f"status={rr.status} exp_dir={rr.exp_dir or '<none>'}"
            )
    else:
        # Analyze-only path: recover runs by run_name prefix from existing experiments.
        for case in cases:
            run_name = f"{args.run_name_prefix}_{case.case_name}"
            exp_dirs = glob_exp_dirs_by_run_name(run_name)
            exp_dir = str(newest_path(exp_dirs) or "")
            records.append(
                RunRecord(
                    case_name=case.case_name,
                    variant=case.variant,
                    step=case.step,
                    status=0 if exp_dir else 999,
                    started_at="",
                    ended_at="",
                    duration_sec=0.0,
                    run_name=run_name,
                    config_path="",
                    exp_dir=exp_dir,
                    log_path="",
                    command="",
                )
            )

    write_csv(report_dir / "run_records.csv", [asdict(r) for r in records])

    if not args.dry_run:
        kpi_rows, run_summary_rows = analyze_records(records, args, report_dir)
    else:
        kpi_rows, run_summary_rows = [], []

    write_csv(report_dir / "kpi_best_last.csv", kpi_rows)
    write_csv(report_dir / "run_summary.csv", run_summary_rows)
    step1_rows = summarize_step1_robustness(kpi_rows)
    write_csv(report_dir / "step1_robustness.csv", step1_rows)
    write_markdown_report(report_dir, args, records, run_summary_rows, step1_rows)

    print(f"[ok] report_dir={report_dir}")
    print(f"[ok] cases={len(cases)} records={len(records)} kpi_rows={len(kpi_rows)}")


if __name__ == "__main__":
    """""
    用法说明（主脚本）:
    
    1) 先只看实验计划（不训练）:
       /home/ubuntu/anaconda3/envs/yolo11/bin/python \
         /home/ubuntu/hpproject/yolo/analyze/code/defect241_hparam_p241.py \
         --dry_run --variants baseline a3c5 --steps 0 1 2 --repeats 1
    
    2) 按 Step0/1/2 执行训练+分析（baseline 与 a3+c5）:
       /home/ubuntu/anaconda3/envs/yolo11/bin/python \
         /home/ubuntu/hpproject/yolo/analyze/code/defect241_hparam_p241.py \
         --execute --variants baseline a3c5 --steps 0 1 2 --repeats 1 \
         --run_name_prefix p241_v1
    
    3) 仅分析已有实验（按 run_name_prefix 回收历史 exp_*）:
       /home/ubuntu/anaconda3/envs/yolo11/bin/python \
         /home/ubuntu/hpproject/yolo/analyze/code/defect241_hparam_p241.py \
         --analyze_only --variants baseline a3c5 --steps 0 1 2 \
         --run_name_prefix p241_v1
    
    关键可调参数:
    - 数据/配置: --baseline_config --enhanced_base_config --out_root
    - 实验结构: --variants --steps --repeats --run_name_prefix
    - 训练超参: --epochs_step0 --epochs_step1 --epochs_step2 --patience
              --warmup_epochs --lr_ref --lrf --seed_step0/1/2
    - 评估口径: --conf_op --match_iou --nms_iou --max_det
              --metric_conf --eval_batch --eval_device
    - KPI 定义: --kpi_fpr_cap --small_thr --kpi_noise_eps
    - 显存保护: --vram_guard --guard_max_gb --safe_batch --safe_workers
    """""
    main()
