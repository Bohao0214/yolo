from __future__ import annotations

import argparse
import csv
import datetime as dt
import gc
import json
import locale
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


def _ensure_utf8_locale() -> None:
    try:
        enc = (locale.getpreferredencoding(False) or "").lower()
        if enc in {"ascii", "ansi_x3.4-1968"}:
            os.environ.setdefault("LANG", "C.UTF-8")
            os.environ.setdefault("LC_ALL", "C.UTF-8")
            locale.setlocale(locale.LC_ALL, "")
    except Exception:
        pass


_ensure_utf8_locale()

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import yaml  # type: ignore
from ultralytics import YOLO  # type: ignore

from src.eval import compute_image_level_results, save_image_level_report


DEFAULT_WEIGHTS = (
    "/home/ubuntu/hpproject/yolo/experiments/yolo11/defect241__a3__b7__c11__d11/"
    "exp_2602280003/train/weights/best.pt"
)
DEFAULT_CFG = "/home/ubuntu/hpproject/yolo/configs/yolo11/enhance241/defect241.yaml"
DEFAULT_OUT_ROOT = "/home/ubuntu/hpproject/yolo/analyze/result"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run split-wise validation with train/val/test visuals and CSV summaries.")
    p.add_argument("--weights", type=str, default=DEFAULT_WEIGHTS, help="Absolute path to best.pt")
    p.add_argument("--cfg", type=str, default=DEFAULT_CFG, help="Config used only for fallback data resolution.")
    p.add_argument("--data", type=str, default="", help="Optional explicit data.yaml path.")
    p.add_argument("--out_root", type=str, default=DEFAULT_OUT_ROOT, help="Result root.")
    p.add_argument("--tag", type=str, default="a3_b7_d11", help="Requested combo tag for report metadata.")
    p.add_argument("--conf", type=float, default=0.25, help="Image-level decision threshold.")
    p.add_argument("--metric_conf", type=float, default=0.01, help="Prediction conf used before image-level scoring.")
    p.add_argument("--match_iou", type=float, default=0.3, help="GT/pred match IoU.")
    p.add_argument("--nms_iou", type=float, default=0.7, help="NMS IoU.")
    p.add_argument("--max_det", type=int, default=100, help="Max detections after NMS.")
    p.add_argument("--batch", type=int, default=1, help="Eval batch for image-level scans.")
    p.add_argument("--imgsz", type=int, default=640, help="YOLO val imgsz.")
    p.add_argument("--device", type=str, default="", help="Eval device.")
    return p.parse_args()


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML must be a mapping: {path}")
    return data


def make_report_dir(out_root: Path) -> Path:
    ts = dt.datetime.now().strftime("report_%y%m%d%H%M")
    base = out_root / ts
    if not base.exists():
        base.mkdir(parents=True, exist_ok=False)
        return base
    idx = 1
    while True:
        cand = out_root / f"{ts}_{idx:02d}"
        if not cand.exists():
            cand.mkdir(parents=True, exist_ok=False)
            return cand
        idx += 1


def _infer_data_yaml(weights: Path, explicit_data: str, cfg_path: Path) -> Path:
    if explicit_data:
        return Path(explicit_data).resolve()
    exp_dir = weights.parents[2] if len(weights.parents) >= 3 else None
    if exp_dir is not None:
        exp_data = exp_dir / "data.yaml"
        if exp_data.exists():
            return exp_data.resolve()
    cfg = load_yaml(cfg_path)
    data_ref = str(cfg.get("data", "")).strip()
    if not data_ref:
        raise RuntimeError("Unable to resolve data.yaml: no --data and cfg.data is empty.")
    data_path = Path(data_ref)
    if not data_path.is_absolute():
        data_path = (REPO_ROOT / data_path).resolve()
    return data_path


def _resolve_data_entry(data_root: Path, entry: Any) -> List[Path]:
    if entry is None:
        return []
    if isinstance(entry, str):
        refs = [x.strip() for x in entry.split(",") if x.strip()]
    elif isinstance(entry, (list, tuple)):
        refs = [str(x).strip() for x in entry if str(x).strip()]
    else:
        refs = [str(entry).strip()]
    out: List[Path] = []
    for ref in refs:
        p = Path(ref)
        out.append(p if p.is_absolute() else (data_root / p).resolve())
    return out


def _collect_split_sources(data_yaml: Path) -> Dict[str, List[Path]]:
    info = load_yaml(data_yaml)
    data_root = Path(str(info.get("path", data_yaml.parent))).resolve()
    return {
        "train": _resolve_data_entry(data_root, info.get("train")),
        "val": _resolve_data_entry(data_root, info.get("val")),
        "test": _resolve_data_entry(data_root, info.get("test")),
    }


def _safe_metric(metrics: Any, attr: str, default: float = 0.0) -> float:
    try:
        value = getattr(metrics, attr)
        return float(value)
    except Exception:
        return float(default)


def _extract_yolo_metrics(metrics: Any) -> Dict[str, Any]:
    box = getattr(metrics, "box", None)
    speed = getattr(metrics, "speed", {}) or {}
    results_dict = getattr(metrics, "results_dict", {}) or {}
    return {
        "box_mp": _safe_metric(box, "mp"),
        "box_mr": _safe_metric(box, "mr"),
        "box_map50": _safe_metric(box, "map50"),
        "box_map75": _safe_metric(box, "map75"),
        "box_map": _safe_metric(box, "map"),
        "fitness": float(results_dict.get("fitness", 0.0)) if isinstance(results_dict, dict) else 0.0,
        "speed_preprocess_ms": float(speed.get("preprocess", 0.0)) if isinstance(speed, dict) else 0.0,
        "speed_inference_ms": float(speed.get("inference", 0.0)) if isinstance(speed, dict) else 0.0,
        "speed_postprocess_ms": float(speed.get("postprocess", 0.0)) if isinstance(speed, dict) else 0.0,
    }


def _summarize_items(split: str, items: List[Dict[str, object]], yolo_metrics: Dict[str, Any]) -> Dict[str, Any]:
    image_tp = sum(1 for item in items if item.get("outcome") == "TP")
    image_fn = sum(1 for item in items if item.get("outcome") == "FN")
    image_fp = sum(1 for item in items if item.get("outcome") == "FP")
    image_tn = sum(1 for item in items if item.get("outcome") == "TN")
    object_fn = int(sum(int(item.get("obj_fn", 0) or 0) for item in items))
    object_fp = int(sum(int(item.get("obj_fp", 0) or 0) for item in items))
    gt_count = int(sum(int(item.get("gt_count", 0) or 0) for item in items))
    pred_count_thr = int(sum(int(item.get("pred_count_thr", 0) or 0) for item in items))
    matched = int(sum(int(item.get("num_matched", 0) or 0) for item in items))
    return {
        "split": split,
        "images": len(items),
        "image_tp": image_tp,
        "image_fn": image_fn,
        "image_fp": image_fp,
        "image_tn": image_tn,
        "object_fn": object_fn,
        "object_fp": object_fp,
        "gt_count": gt_count,
        "pred_count_thr": pred_count_thr,
        "matched_pred_count": matched,
        **yolo_metrics,
    }


def _merge_summary(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {"split": "all", "images": 0}
    total_images = sum(int(row.get("images", 0) or 0) for row in rows)
    merged = {
        "split": "all",
        "images": total_images,
        "image_tp": sum(int(row.get("image_tp", 0) or 0) for row in rows),
        "image_fn": sum(int(row.get("image_fn", 0) or 0) for row in rows),
        "image_fp": sum(int(row.get("image_fp", 0) or 0) for row in rows),
        "image_tn": sum(int(row.get("image_tn", 0) or 0) for row in rows),
        "object_fn": sum(int(row.get("object_fn", 0) or 0) for row in rows),
        "object_fp": sum(int(row.get("object_fp", 0) or 0) for row in rows),
        "gt_count": sum(int(row.get("gt_count", 0) or 0) for row in rows),
        "pred_count_thr": sum(int(row.get("pred_count_thr", 0) or 0) for row in rows),
        "matched_pred_count": sum(int(row.get("matched_pred_count", 0) or 0) for row in rows),
    }
    weighted_keys = (
        "box_mp",
        "box_mr",
        "box_map50",
        "box_map75",
        "box_map",
        "fitness",
        "speed_preprocess_ms",
        "speed_inference_ms",
        "speed_postprocess_ms",
    )
    for key in weighted_keys:
        if total_images <= 0:
            merged[key] = 0.0
        else:
            merged[key] = sum(float(row.get(key, 0.0) or 0.0) * int(row.get("images", 0) or 0) for row in rows) / total_images
    return merged


def _write_summary_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    headers = [
        "split",
        "images",
        "image_tp",
        "image_fn",
        "image_fp",
        "image_tn",
        "object_fn",
        "object_fp",
        "gt_count",
        "pred_count_thr",
        "matched_pred_count",
        "box_mp",
        "box_mr",
        "box_map50",
        "box_map75",
        "box_map",
        "fitness",
        "speed_preprocess_ms",
        "speed_inference_ms",
        "speed_postprocess_ms",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in headers})


def _loaded_enhance_modules(model: YOLO) -> List[str]:
    names: List[str] = []
    torch_model = getattr(model, "model", None)
    seq = getattr(torch_model, "model", None)
    if seq is None:
        return names
    for module in seq:
        name = module.__class__.__name__
        if "enhance241" in name.lower() or name.startswith(("A", "B", "C", "D")):
            if any(token in name for token in ("SPD", "CARAFE", "DySample", "BRA", "Gate", "Calib", "LSK")):
                names.append(name)
    return names


def _run_yolo_val(
    model: YOLO,
    data_yaml: Path,
    split: str,
    report_dir: Path,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    save_dir = report_dir / "yolo_val" / split
    save_dir.parent.mkdir(parents=True, exist_ok=True)
    metrics = model.val(
        data=str(data_yaml),
        split=split,
        imgsz=int(args.imgsz),
        batch=int(args.batch),
        device=str(args.device),
        conf=float(args.metric_conf),
        iou=float(args.nms_iou),
        max_det=int(args.max_det),
        plots=False,
        verbose=False,
        save_json=False,
        project=str(save_dir.parent),
        name=save_dir.name,
        exist_ok=True,
    )
    return _extract_yolo_metrics(metrics)


def _run_split(
    model: YOLO,
    split: str,
    sources: Sequence[Path],
    data_yaml: Path,
    report_dir: Path,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    metrics_dir = report_dir / "metrics"
    all_items: List[Dict[str, object]] = []
    vis_root = report_dir / f"{split}_vis"
    for src in sources:
        if not src.exists():
            continue
        try:
            items = compute_image_level_results(
                model,
                src,
                conf_threshold=float(args.conf),
                iou_match=float(args.match_iou),
                metric_conf=float(args.metric_conf),
                batch=int(args.batch),
                device=str(args.device),
                nms_iou=float(args.nms_iou),
                max_det=int(args.max_det),
                split=split,
                vis_root=vis_root,
                save_visuals=True,
            )
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            gc.collect()
            items = compute_image_level_results(
                model,
                src,
                conf_threshold=float(args.conf),
                iou_match=float(args.match_iou),
                metric_conf=float(args.metric_conf),
                batch=1,
                device="cpu",
                nms_iou=float(args.nms_iou),
                max_det=int(args.max_det),
                split=split,
                vis_root=vis_root,
                save_visuals=True,
            )
        all_items.extend(items)

    meta = {
        "split": split,
        "conf_threshold": float(args.conf),
        "iou_match": float(args.match_iou),
        "metric_conf": float(args.metric_conf),
        "nms_iou": float(args.nms_iou),
        "max_det": int(args.max_det),
        "weights": str(args.weights),
        "data_yaml": str(data_yaml),
    }
    save_image_level_report(metrics_dir, split, all_items, meta)
    yolo_metrics = _run_yolo_val(model, data_yaml, split, report_dir, args)
    return _summarize_items(split, all_items, yolo_metrics)


def main() -> None:
    args = parse_args()

    weights = Path(args.weights).resolve()
    if not weights.exists():
        raise FileNotFoundError(f"weights not found: {weights}")

    cfg_path = Path(args.cfg).resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f"cfg not found: {cfg_path}")

    data_yaml = _infer_data_yaml(weights, str(args.data), cfg_path)
    if not data_yaml.exists():
        raise FileNotFoundError(f"data yaml not found: {data_yaml}")

    split_sources = _collect_split_sources(data_yaml)
    report_dir = make_report_dir(Path(args.out_root).resolve())
    (report_dir / "metrics").mkdir(parents=True, exist_ok=True)

    model = YOLO(str(weights))

    summary_rows: List[Dict[str, Any]] = []
    for split in ("train", "val", "test"):
        sources = split_sources.get(split, [])
        if not sources:
            continue
        summary_rows.append(_run_split(model, split, sources, data_yaml, report_dir, args))
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        gc.collect()

    summary_rows.append(_merge_summary(summary_rows))
    _write_summary_csv(report_dir / "metrics" / "split_summary.csv", summary_rows)

    meta = {
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        "requested_combo": str(args.tag),
        "weights": str(weights),
        "data_yaml": str(data_yaml),
        "splits": {k: [str(x) for x in v] for k, v in split_sources.items()},
        "conf": float(args.conf),
        "metric_conf": float(args.metric_conf),
        "match_iou": float(args.match_iou),
        "nms_iou": float(args.nms_iou),
        "max_det": int(args.max_det),
        "batch": int(args.batch),
        "imgsz": int(args.imgsz),
        "device": str(args.device),
        "loaded_enhance_modules": _loaded_enhance_modules(model),
        "report_dir": str(report_dir),
    }
    (report_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[done] report_dir={report_dir}")
    print(f"[done] summary_csv={report_dir / 'metrics' / 'split_summary.csv'}")


if __name__ == "__main__":
    """""
    python3 /home/ubuntu/hpproject/yolo/analyze/code/expfindtrain.py \
  --weights /home/ubuntu/hpproject/yolo/experiments/yolo11/defect241__a3__b7__c11__d11/exp_2602280003/train/weights/best.pt \
  --tag a3_b7_d11
    """""
    main()
