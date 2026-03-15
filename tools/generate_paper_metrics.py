#!/usr/bin/env python3
"""Generate unified experiment metrics snapshot for one experiment directory."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import yaml


ANSI_RE = re.compile(r"\x1B\[[0-9;]*[A-Za-z]")
PARAMS_GFLOPS_RE = re.compile(
    r"summary:\s*.*?,\s*([0-9][0-9,]*)\s+parameters.*?([0-9]+(?:\.[0-9]+)?)\s+GFLOPs",
    re.IGNORECASE,
)
SPEED_RE = re.compile(
    r"Speed:\s*([0-9.]+)ms preprocess,\s*([0-9.]+)ms inference,\s*([0-9.]+)ms loss,\s*([0-9.]+)ms postprocess",
    re.IGNORECASE,
)

EPOCH_IMAGE_METRIC_FIELDS = [
    "epoch",
    "best_rule",
    "is_best_by_rule",
    "best_ckpt",
    "train_fitness",
    "is_best_default",
    "is_best_ifn",
    "is_best_iauroc_fpr0p5",
    "metrics/mAP50(B)",
    "metrics/mAP50-95(B)",
    "metrics/precision(B)",
    "metrics/recall(B)",
    "train/box_loss",
    "train/cls_loss",
    "train/dfl_loss",
    "val_iTP",
    "val_iFP",
    "val_iFN",
    "val_iTN",
    "val_iAUROC_fpr0p5",
    "test_iTP",
    "test_iFP",
    "test_iFN",
    "test_iTN",
    "test_iAUROC_fpr0p5",
]

FLOAT_DIGITS = {
    "gflops": 3,
    "speed_pre_ms": 3,
    "speed_inf_ms": 3,
    "speed_post_ms": 3,
    "speed_total_ms": 3,
    "fps": 3,
    "eval_conf_threshold": 4,
    "eval_iou_match": 4,
    "eval_nms_iou": 4,
    "recall_at_fpr0p05_iou0p3": 6,
    "thr_at_fpr0p05_iou0p3": 6,
    "recall_at_fpr0p10_iou0p3": 6,
    "thr_at_fpr0p10_iou0p3": 6,
}


def _to_float(v: object) -> Optional[float]:
    try:
        s = str(v).strip()
        if not s:
            return None
        out = float(s)
        if not math.isfinite(out):
            return None
        return out
    except Exception:
        return None


def _to_int(v: object) -> Optional[int]:
    fv = _to_float(v)
    if fv is None:
        return None
    try:
        return int(round(fv))
    except Exception:
        return None


def _fmt_value(key: str, value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        digits = FLOAT_DIGITS.get(key, 6)
        return f"{value:.{digits}f}"
    return str(value)


def _derive_binary_metrics(tp: Optional[int], fp: Optional[int], fn: Optional[int], tn: Optional[int]) -> Dict[str, Optional[float]]:
    if tp is None or fp is None or fn is None or tn is None:
        return {
            "precision": None,
            "recall": None,
            "f1": None,
            "accuracy": None,
            "fpr": None,
        }
    p = None
    r = None
    f1 = None
    acc = None
    fpr = None
    if (tp + fp) > 0:
        p = float(tp) / float(tp + fp)
    if (tp + fn) > 0:
        r = float(tp) / float(tp + fn)
    if p is not None and r is not None and (p + r) > 0:
        f1 = 2.0 * p * r / (p + r)
    total = tp + fp + fn + tn
    if total > 0:
        acc = float(tp + tn) / float(total)
    if (fp + tn) > 0:
        fpr = float(fp) / float(fp + tn)
    return {
        "precision": p,
        "recall": r,
        "f1": f1,
        "accuracy": acc,
        "fpr": fpr,
    }


def _read_results_metrics(results_csv: Path) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {
        "best_epoch": None,
        "precision": None,
        "recall": None,
        "map50": None,
        "map50_95": None,
        "f1": None,
    }
    if not results_csv.exists():
        return out

    with results_csv.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return out

    metric_keys = [
        "metrics/mAP50-95(B)",
        "metrics/mAP50(B)",
        "metrics/recall(B)",
        "metrics/precision(B)",
    ]

    epoch_rows: List[Tuple[int, Dict[str, str]]] = []
    for row in rows:
        epoch_v = _to_float(row.get("epoch", ""))
        if epoch_v is None:
            continue
        epoch_rows.append((int(round(epoch_v)), row))
    if not epoch_rows:
        return out

    best_epoch: Optional[int] = None
    for epoch_i, row in epoch_rows:
        note = str(row.get("best_epoch_note", "")).strip().upper()
        if note.startswith("BEST_SUMMARY"):
            best_epoch = epoch_i
            break

    if best_epoch is None:
        best_score = -float("inf")
        for epoch_i, row in epoch_rows:
            score = None
            for k in metric_keys:
                score = _to_float(row.get(k, ""))
                if score is not None:
                    break
            if score is None:
                continue
            if score > best_score:
                best_score = score
                best_epoch = epoch_i

    best_row = None
    if best_epoch is not None:
        for epoch_i, row in epoch_rows:
            if epoch_i == best_epoch:
                best_row = row
                break
    if best_row is None:
        best_row = epoch_rows[-1][1]
        best_epoch = epoch_rows[-1][0]

    p = _to_float(best_row.get("metrics/precision(B)", ""))
    r = _to_float(best_row.get("metrics/recall(B)", ""))
    m50 = _to_float(best_row.get("metrics/mAP50(B)", ""))
    m5095 = _to_float(best_row.get("metrics/mAP50-95(B)", ""))
    f1 = None
    if p is not None and r is not None and (p + r) > 0:
        f1 = 2.0 * p * r / (p + r)

    out["best_epoch"] = float(best_epoch) if best_epoch is not None else None
    out["precision"] = p
    out["recall"] = r
    out["map50"] = m50
    out["map50_95"] = m5095
    out["f1"] = f1
    return out


def _read_eval_summary(eval_summary_txt: Path) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {
        "image_auroc": None,
        "image_ap": None,
    }
    if not eval_summary_txt.exists():
        return out
    try:
        for line in eval_summary_txt.read_text(encoding="utf-8", errors="ignore").splitlines():
            if ":" not in line:
                continue
            k, v = line.split(":", 1)
            k = k.strip().lower()
            val = _to_float(v.strip())
            if k == "image_auroc":
                out["image_auroc"] = val
            elif k == "image_ap":
                out["image_ap"] = val
    except Exception:
        pass
    return out


def _read_eval_image_level(eval_csv: Path) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {
        "image_tp": None,
        "image_fp": None,
        "image_fn": None,
        "image_tn": None,
        "image_precision": None,
        "image_recall": None,
        "image_f1": None,
        "image_accuracy": None,
        "image_fpr": None,
        "val_image_tp": None,
        "val_image_fp": None,
        "val_image_fn": None,
        "val_image_tn": None,
        "val_image_precision": None,
        "val_image_recall": None,
        "val_image_f1": None,
        "val_image_accuracy": None,
        "val_image_fpr": None,
        "test_image_tp": None,
        "test_image_fp": None,
        "test_image_fn": None,
        "test_image_tn": None,
        "test_image_precision": None,
        "test_image_recall": None,
        "test_image_f1": None,
        "test_image_accuracy": None,
        "test_image_fpr": None,
        "eval_conf_threshold": None,
        "eval_iou_match": None,
        "eval_nms_iou": None,
        "eval_max_det": None,
    }
    if not eval_csv.exists():
        return out

    counts = {"TP": 0, "FP": 0, "FN": 0, "TN": 0}
    split_counts: Dict[str, Dict[str, int]] = {}

    try:
        with eval_csv.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                outcome = str(row.get("outcome", "")).strip().upper()
                split = str(row.get("split", "")).strip().lower()
                if outcome in counts:
                    counts[outcome] += 1
                    if split:
                        split_counts.setdefault(split, {"TP": 0, "FP": 0, "FN": 0, "TN": 0})
                        split_counts[split][outcome] += 1

                if out["eval_conf_threshold"] is None:
                    out["eval_conf_threshold"] = _to_float(row.get("conf_threshold", ""))
                if out["eval_iou_match"] is None:
                    out["eval_iou_match"] = _to_float(row.get("iou_match", ""))
                if out["eval_nms_iou"] is None:
                    out["eval_nms_iou"] = _to_float(row.get("nms_iou", ""))
                if out["eval_max_det"] is None:
                    out["eval_max_det"] = _to_int(row.get("max_det", ""))
    except Exception:
        return out

    out["image_tp"] = int(counts["TP"])
    out["image_fp"] = int(counts["FP"])
    out["image_fn"] = int(counts["FN"])
    out["image_tn"] = int(counts["TN"])
    derived = _derive_binary_metrics(out["image_tp"], out["image_fp"], out["image_fn"], out["image_tn"])
    out["image_precision"] = derived["precision"]
    out["image_recall"] = derived["recall"]
    out["image_f1"] = derived["f1"]
    out["image_accuracy"] = derived["accuracy"]
    out["image_fpr"] = derived["fpr"]

    for split_name in ("val", "test"):
        c = split_counts.get(split_name, {})
        tp = int(c.get("TP", 0))
        fp = int(c.get("FP", 0))
        fn = int(c.get("FN", 0))
        tn = int(c.get("TN", 0))
        out[f"{split_name}_image_tp"] = tp
        out[f"{split_name}_image_fp"] = fp
        out[f"{split_name}_image_fn"] = fn
        out[f"{split_name}_image_tn"] = tn
        d = _derive_binary_metrics(tp, fp, fn, tn)
        out[f"{split_name}_image_precision"] = d["precision"]
        out[f"{split_name}_image_recall"] = d["recall"]
        out[f"{split_name}_image_f1"] = d["f1"]
        out[f"{split_name}_image_accuracy"] = d["accuracy"]
        out[f"{split_name}_image_fpr"] = d["fpr"]
    return out


def _read_eval_thresholds_multi(th_csv: Path) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {
        "recall_at_fpr0p05_iou0p3": None,
        "thr_at_fpr0p05_iou0p3": None,
        "recall_at_fpr0p10_iou0p3": None,
        "thr_at_fpr0p10_iou0p3": None,
    }
    if not th_csv.exists():
        return out

    rows: List[Tuple[float, float, float, float]] = []
    try:
        with th_csv.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if str(row.get("fixed_var", "")).strip().lower() != "iou":
                    continue
                iou = _to_float(row.get("fixed_value", ""))
                thr = _to_float(row.get("threshold", ""))
                rec = _to_float(row.get("recall", ""))
                fpr = _to_float(row.get("fpr", ""))
                if None in (iou, thr, rec, fpr):
                    continue
                rows.append((float(iou), float(thr), float(rec), float(fpr)))
    except Exception:
        return out

    if not rows:
        return out

    target_iou = 0.3
    nearest_iou = min(rows, key=lambda x: abs(x[0] - target_iou))[0]
    iou_rows = [r for r in rows if abs(r[0] - nearest_iou) < 1e-12]
    if not iou_rows:
        return out

    for target_fpr, rec_key, thr_key in (
        (0.05, "recall_at_fpr0p05_iou0p3", "thr_at_fpr0p05_iou0p3"),
        (0.10, "recall_at_fpr0p10_iou0p3", "thr_at_fpr0p10_iou0p3"),
    ):
        best = min(iou_rows, key=lambda x: abs(x[3] - target_fpr))
        out[rec_key] = best[2]
        out[thr_key] = best[1]
    return out


def _read_epoch_image_metrics(csv_path: Path) -> Dict[str, object]:
    out: Dict[str, object] = {
        "epoch_metric_epoch": None,
        "best_rule": None,
        "train_fitness_epoch": None,
        "val_iTP_epoch": None,
        "val_iFP_epoch": None,
        "val_iFN_epoch": None,
        "val_iTN_epoch": None,
        "val_iPrecision_epoch": None,
        "val_iRecall_epoch": None,
        "val_iF1_epoch": None,
        "val_iAUROC_fpr0p5_epoch": None,
        "test_iTP_epoch": None,
        "test_iFP_epoch": None,
        "test_iFN_epoch": None,
        "test_iTN_epoch": None,
        "test_iPrecision_epoch": None,
        "test_iRecall_epoch": None,
        "test_iF1_epoch": None,
        "test_iAUROC_fpr0p5_epoch": None,
    }
    if not csv_path.exists():
        return out

    first_nonempty = ""
    try:
        with csv_path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                s = line.strip()
                if s:
                    first_nonempty = s
                    break
    except Exception:
        return out
    if not first_nonempty:
        return out

    has_header = first_nonempty.lower().startswith("epoch,")
    rows: List[Dict[str, str]] = []
    try:
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            if has_header:
                reader = csv.DictReader(f)
            else:
                reader = csv.DictReader(f, fieldnames=EPOCH_IMAGE_METRIC_FIELDS)
            for row in reader:
                ep = str(row.get("epoch", "")).strip().lower()
                if ep in {"", "epoch"}:
                    continue
                if _to_int(ep) is None:
                    continue
                rows.append(row)
    except Exception:
        return out
    if not rows:
        return out

    best_rows = [r for r in rows if (_to_int(r.get("is_best_by_rule", "")) or 0) == 1]
    if best_rows:
        selected = max(best_rows, key=lambda r: _to_int(r.get("epoch", "")) or -1)
    else:
        def _rank_key(r: Dict[str, str]) -> Tuple[int, int, float, int]:
            val_fn = _to_int(r.get("val_iFN", ""))
            val_auc = _to_float(r.get("val_iAUROC_fpr0p5", ""))
            epoch = _to_int(r.get("epoch", "")) or -1
            miss = 1 if val_fn is None else 0
            fn_key = int(1e9 if val_fn is None else val_fn)
            auc_key = float("inf") if val_auc is None else -float(val_auc)
            return (miss, fn_key, auc_key, -epoch)

        selected = min(rows, key=_rank_key)

    val_tp = _to_int(selected.get("val_iTP", ""))
    val_fp = _to_int(selected.get("val_iFP", ""))
    val_fn = _to_int(selected.get("val_iFN", ""))
    val_tn = _to_int(selected.get("val_iTN", ""))
    test_tp = _to_int(selected.get("test_iTP", ""))
    test_fp = _to_int(selected.get("test_iFP", ""))
    test_fn = _to_int(selected.get("test_iFN", ""))
    test_tn = _to_int(selected.get("test_iTN", ""))

    val_d = _derive_binary_metrics(val_tp, val_fp, val_fn, val_tn)
    test_d = _derive_binary_metrics(test_tp, test_fp, test_fn, test_tn)

    out["epoch_metric_epoch"] = _to_int(selected.get("epoch", ""))
    out["best_rule"] = str(selected.get("best_rule", "")).strip() or None
    out["train_fitness_epoch"] = _to_float(selected.get("train_fitness", ""))
    out["val_iTP_epoch"] = val_tp
    out["val_iFP_epoch"] = val_fp
    out["val_iFN_epoch"] = val_fn
    out["val_iTN_epoch"] = val_tn
    out["val_iPrecision_epoch"] = val_d["precision"]
    out["val_iRecall_epoch"] = val_d["recall"]
    out["val_iF1_epoch"] = val_d["f1"]
    out["val_iAUROC_fpr0p5_epoch"] = _to_float(selected.get("val_iAUROC_fpr0p5", ""))
    out["test_iTP_epoch"] = test_tp
    out["test_iFP_epoch"] = test_fp
    out["test_iFN_epoch"] = test_fn
    out["test_iTN_epoch"] = test_tn
    out["test_iPrecision_epoch"] = test_d["precision"]
    out["test_iRecall_epoch"] = test_d["recall"]
    out["test_iF1_epoch"] = test_d["f1"]
    out["test_iAUROC_fpr0p5_epoch"] = _to_float(selected.get("test_iAUROC_fpr0p5", ""))
    return out


def _iter_clean_lines(path: Path) -> Iterable[str]:
    if not path.exists():
        return []
    lines: List[str] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            lines.append(ANSI_RE.sub("", raw.rstrip("\n")))
    return lines


def _read_log_hardware_metrics(log_path: Optional[Path]) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {
        "params": None,
        "gflops": None,
        "speed_pre_ms": None,
        "speed_inf_ms": None,
        "speed_post_ms": None,
        "speed_total_ms": None,
        "fps": None,
    }
    if log_path is None or not log_path.exists():
        return out

    for line in _iter_clean_lines(log_path):
        m = PARAMS_GFLOPS_RE.search(line)
        if m:
            params_s = m.group(1).replace(",", "")
            out["params"] = _to_float(params_s)
            out["gflops"] = _to_float(m.group(2))
        m2 = SPEED_RE.search(line)
        if m2:
            pre = _to_float(m2.group(1))
            inf = _to_float(m2.group(2))
            post = _to_float(m2.group(4))
            out["speed_pre_ms"] = pre
            out["speed_inf_ms"] = inf
            out["speed_post_ms"] = post
            if pre is not None and inf is not None and post is not None:
                total = pre + inf + post
                out["speed_total_ms"] = total
            if inf is not None and inf > 0:
                out["fps"] = 1000.0 / inf
    return out


def _read_train_args(args_yaml: Path) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {
        "train_batch": None,
        "train_imgsz": None,
        "train_epochs": None,
        "train_lr0": None,
        "train_lrf": None,
        "train_patience": None,
    }
    if not args_yaml.exists():
        return out
    try:
        with args_yaml.open("r", encoding="utf-8") as f:
            d = yaml.safe_load(f) or {}
        if not isinstance(d, dict):
            return out
        out["train_batch"] = _to_float(d.get("batch", ""))
        imgsz = d.get("imgsz", "")
        if isinstance(imgsz, (list, tuple)) and imgsz:
            out["train_imgsz"] = _to_float(imgsz[0])
        else:
            out["train_imgsz"] = _to_float(imgsz)
        out["train_epochs"] = _to_float(d.get("epochs", ""))
        out["train_lr0"] = _to_float(d.get("lr0", ""))
        out["train_lrf"] = _to_float(d.get("lrf", ""))
        out["train_patience"] = _to_float(d.get("patience", ""))
    except Exception:
        return out
    return out


def _write_csv(path: Path, row: Dict[str, object], keys: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerow({k: _fmt_value(k, row.get(k)) for k in keys})


def _write_json(path: Path, row: Dict[str, object], keys: List[str]) -> None:
    payload = {k: row.get(k) for k in keys}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_markdown(path: Path, row: Dict[str, object]) -> None:
    def g(key: str) -> str:
        return _fmt_value(key, row.get(key))

    lines = [
        "# Experiment Metrics Snapshot",
        "",
        f"- generated_at: `{datetime.now().isoformat(timespec='seconds')}`",
        f"- exp_dir: `{g('exp_dir')}`",
        f"- tag: `{g('tag')}`",
        "",
        "## Detection",
        "",
        f"- best_epoch: `{g('best_epoch')}`",
        f"- precision: `{g('precision')}`",
        f"- recall: `{g('recall')}`",
        f"- mAP50: `{g('map50')}`",
        f"- mAP50-95: `{g('map50_95')}`",
        f"- F1: `{g('f1')}`",
        "",
        "## Image-Level",
        "",
        f"- image_auroc: `{g('image_auroc')}`",
        f"- image_ap: `{g('image_ap')}`",
        f"- image_tp/fp/fn/tn: `{g('image_tp')}/{g('image_fp')}/{g('image_fn')}/{g('image_tn')}`",
        f"- image_precision/recall/f1: `{g('image_precision')}/{g('image_recall')}/{g('image_f1')}`",
        f"- image_accuracy/fpr: `{g('image_accuracy')}/{g('image_fpr')}`",
        f"- val_image_f1: `{g('val_image_f1')}`",
        f"- test_image_f1: `{g('test_image_f1')}`",
        "",
        "## Threshold Slice",
        "",
        f"- recall@FPR0.05(IOU~0.3): `{g('recall_at_fpr0p05_iou0p3')}`",
        f"- threshold@FPR0.05(IOU~0.3): `{g('thr_at_fpr0p05_iou0p3')}`",
        f"- recall@FPR0.10(IOU~0.3): `{g('recall_at_fpr0p10_iou0p3')}`",
        f"- threshold@FPR0.10(IOU~0.3): `{g('thr_at_fpr0p10_iou0p3')}`",
        "",
        "## Epoch-Level Best",
        "",
        f"- best_rule: `{g('best_rule')}`",
        f"- epoch_metric_epoch: `{g('epoch_metric_epoch')}`",
        f"- val_iFN_epoch: `{g('val_iFN_epoch')}`",
        f"- val_iAUROC_fpr0p5_epoch: `{g('val_iAUROC_fpr0p5_epoch')}`",
        f"- test_iFN_epoch: `{g('test_iFN_epoch')}`",
        f"- test_iAUROC_fpr0p5_epoch: `{g('test_iAUROC_fpr0p5_epoch')}`",
        "",
        "## Runtime/Model",
        "",
        f"- params: `{g('params')}`",
        f"- gflops: `{g('gflops')}`",
        f"- speed_pre_ms: `{g('speed_pre_ms')}`",
        f"- speed_inf_ms: `{g('speed_inf_ms')}`",
        f"- speed_post_ms: `{g('speed_post_ms')}`",
        f"- speed_total_ms: `{g('speed_total_ms')}`",
        f"- fps: `{g('fps')}`",
        "",
        "## Train Args",
        "",
        f"- train_batch: `{g('train_batch')}`",
        f"- train_imgsz: `{g('train_imgsz')}`",
        f"- train_epochs: `{g('train_epochs')}`",
        f"- train_lr0/lrf: `{g('train_lr0')}/{g('train_lrf')}`",
        f"- train_patience: `{g('train_patience')}`",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Generate experiment_metrics.{csv,md,json} for one experiment."
    )
    ap.add_argument("--exp-dir", required=True, help="Experiment dir, e.g. .../exp_260315xxxx")
    ap.add_argument("--log", default="", help="Optional train log path for parsing params/GFLOPs/speed.")
    ap.add_argument("--tag", default="", help="Optional case tag.")
    args = ap.parse_args()

    exp_dir = Path(args.exp_dir).resolve()
    if not exp_dir.is_dir():
        print(f"warn: exp_dir not found: {exp_dir}")
        return 1
    log_path = Path(args.log).resolve() if str(args.log).strip() else None

    metrics_dir = exp_dir / "metrics"
    train_dir = exp_dir / "train"

    results = _read_results_metrics(train_dir / "results.csv")
    eval_s = _read_eval_summary(metrics_dir / "eval_summary.txt")
    eval_img = _read_eval_image_level(metrics_dir / "eval_image_level.csv")
    eval_th = _read_eval_thresholds_multi(metrics_dir / "eval_thresholds_multi.csv")
    epoch_s = _read_epoch_image_metrics(train_dir / "epoch_image_metrics.csv")
    run_s = _read_log_hardware_metrics(log_path)
    args_s = _read_train_args(train_dir / "args.yaml")

    row: Dict[str, object] = {
        "exp_dir": str(exp_dir),
        "tag": str(args.tag).strip(),
        **results,
        **eval_s,
        **eval_img,
        **eval_th,
        **epoch_s,
        **run_s,
        **args_s,
    }

    if row.get("params") is not None:
        row["params"] = _to_int(row["params"])
    if row.get("best_epoch") is not None:
        row["best_epoch"] = _to_int(row["best_epoch"])
    for k in ("train_batch", "train_imgsz", "train_epochs", "train_patience"):
        if row.get(k) is not None:
            row[k] = _to_int(row[k])

    keys = [
        "exp_dir",
        "tag",
        "best_epoch",
        "precision",
        "recall",
        "map50",
        "map50_95",
        "f1",
        "image_auroc",
        "image_ap",
        "image_tp",
        "image_fp",
        "image_fn",
        "image_tn",
        "image_precision",
        "image_recall",
        "image_f1",
        "image_accuracy",
        "image_fpr",
        "val_image_tp",
        "val_image_fp",
        "val_image_fn",
        "val_image_tn",
        "val_image_precision",
        "val_image_recall",
        "val_image_f1",
        "val_image_accuracy",
        "val_image_fpr",
        "test_image_tp",
        "test_image_fp",
        "test_image_fn",
        "test_image_tn",
        "test_image_precision",
        "test_image_recall",
        "test_image_f1",
        "test_image_accuracy",
        "test_image_fpr",
        "recall_at_fpr0p05_iou0p3",
        "thr_at_fpr0p05_iou0p3",
        "recall_at_fpr0p10_iou0p3",
        "thr_at_fpr0p10_iou0p3",
        "best_rule",
        "epoch_metric_epoch",
        "train_fitness_epoch",
        "val_iTP_epoch",
        "val_iFP_epoch",
        "val_iFN_epoch",
        "val_iTN_epoch",
        "val_iPrecision_epoch",
        "val_iRecall_epoch",
        "val_iF1_epoch",
        "val_iAUROC_fpr0p5_epoch",
        "test_iTP_epoch",
        "test_iFP_epoch",
        "test_iFN_epoch",
        "test_iTN_epoch",
        "test_iPrecision_epoch",
        "test_iRecall_epoch",
        "test_iF1_epoch",
        "test_iAUROC_fpr0p5_epoch",
        "eval_conf_threshold",
        "eval_iou_match",
        "eval_nms_iou",
        "eval_max_det",
        "params",
        "gflops",
        "speed_pre_ms",
        "speed_inf_ms",
        "speed_post_ms",
        "speed_total_ms",
        "fps",
        "train_batch",
        "train_imgsz",
        "train_epochs",
        "train_lr0",
        "train_lrf",
        "train_patience",
    ]

    csv_path = metrics_dir / "experiment_metrics.csv"
    md_path = metrics_dir / "experiment_metrics.md"
    json_path = metrics_dir / "experiment_metrics.json"

    # Compatibility for old consumers.
    legacy_csv_path = metrics_dir / "paper_metrics.csv"
    legacy_md_path = metrics_dir / "paper_metrics.md"

    _write_csv(csv_path, row, keys)
    _write_markdown(md_path, row)
    _write_json(json_path, row, keys)
    _write_csv(legacy_csv_path, row, keys)
    _write_markdown(legacy_md_path, row)

    print(
        f"experiment_metrics_csv={csv_path} "
        f"experiment_metrics_md={md_path} "
        f"experiment_metrics_json={json_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
