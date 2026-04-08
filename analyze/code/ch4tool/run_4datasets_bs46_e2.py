#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
工业缺陷数据集 smoke 脚本：4 数据集，batch=4/6，epoch=2。

特点:
1. 自动查找每个数据集可用的 data.yaml（优先数据集目录，其次 configs 目录）
2. 对每个数据集跑两组 batch（4, 6）短训（epoch=2）检查是否报错
3. 导出目标级指标（P/R/mAP50/mAP50-95）
4. 导出图像级指标（有缺陷 / 无缺陷二分类）
5. 输出到 experiments，不改动原有主流程脚本

用法示例:
1) 全量执行
   conda run -n yolo11 python /home/ubuntu/hpproject/yolo/analyze/code/ch4tool/run_4datasets_bs46_e2.py

2) 指定模型权重
   conda run -n yolo11 python /home/ubuntu/hpproject/yolo/analyze/code/ch4tool/run_4datasets_bs46_e2.py \
     --model /home/ubuntu/hpproject/yolo/experiments/xxx/weights/best.pt

3) 只跑部分数据集
   conda run -n yolo11 python /home/ubuntu/hpproject/yolo/analyze/code/ch4tool/run_4datasets_bs46_e2.py \
     --only DeepPCB_standard neudet_622
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import yaml
from ultralytics import YOLO

PROJECT_ROOT = Path("/home/ubuntu/hpproject/yolo")

DATASET_CANDIDATES = [
    {
        "alias": "gc10det_622_halves",
        "root": "/home/ubuntu/hpproject/yolo/experiments/gc10det_622_halves",
        "data_yaml": None,
    },
    {
        "alias": "DeepPCB_standard",
        "root": "/home/ubuntu/hpproject/yolo/dataset/yolo/DeepPCB_standard",
        "data_yaml": None,
    },
    {
        "alias": "kolektorsdd_622_halves",
        "root": "/home/ubuntu/hpproject/yolo/dataset/yolo/kolektorsdd_622_halves",
        "data_yaml": None,
    },
    {
        "alias": "neudet_622",
        "root": "/home/ubuntu/hpproject/yolo/dataset/yolo/neudet_622",
        "data_yaml": None,
    },
]

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
NEGATIVE_KEYS = {
    "good",
    "ok",
    "normal",
    "background",
    "bg",
    "no_defect",
    "nodefect",
    "non_defect",
    "defect_free",
    "negative",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("run_4datasets_bs46_e2")
    parser.add_argument("--model", default="yolo11n.pt", help="YOLO 权重路径")
    parser.add_argument("--device", default="0", help="device，如 0 / cpu")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batches", nargs="+", type=int, default=[4, 6])
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--only", nargs="*", default=None, help="只跑给定 alias")
    return parser.parse_args()


def read_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        x = yaml.safe_load(f) or {}
    return x if isinstance(x, dict) else {}


def is_data_yaml(cfg: Dict[str, Any]) -> bool:
    return "train" in cfg and ("val" in cfg or "test" in cfg)


def normalize_names(names: Any) -> Dict[int, str]:
    if isinstance(names, list):
        return {i: str(v) for i, v in enumerate(names)}
    if isinstance(names, dict):
        out: Dict[int, str] = {}
        for k, v in names.items():
            try:
                out[int(k)] = str(v)
            except Exception:
                continue
        return out
    return {}


def resolve_path(base: Path, value: Any) -> Path:
    p = Path(str(value))
    if p.is_absolute():
        return p
    return (base / p).resolve()


def collect_images(entry: Any, yaml_dir: Path) -> List[Path]:
    imgs: List[Path] = []
    if entry is None:
        return imgs
    if isinstance(entry, (list, tuple)):
        for e in entry:
            imgs.extend(collect_images(e, yaml_dir))
        return imgs

    p = resolve_path(yaml_dir, entry)

    if p.suffix.lower() == ".txt" and p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            ip = Path(line)
            if not ip.is_absolute():
                ip = (p.parent / ip).resolve()
            imgs.append(ip)
        return imgs

    if p.is_dir():
        for ext in IMG_EXTS:
            imgs.extend(p.rglob(f"*{ext}"))
        return imgs

    if p.is_file() and p.suffix.lower() in IMG_EXTS:
        return [p]

    return imgs


def guess_label_file(img_path: Path) -> Path:
    s = img_path.as_posix()
    if "/images/" in s:
        return Path(s.replace("/images/", "/labels/")).with_suffix(".txt")
    return img_path.with_suffix(".txt")


def is_negative_class(class_name: str) -> bool:
    c = class_name.lower()
    return any(k in c for k in NEGATIVE_KEYS)


def gt_has_defect(label_file: Path, names: Dict[int, str]) -> bool:
    if not label_file.exists():
        return False
    for line in label_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        try:
            cid = int(float(parts[0]))
        except Exception:
            continue
        cname = names.get(cid, str(cid))
        if not is_negative_class(cname):
            return True
    return False


def pred_has_defect(result: Any, names: Dict[int, str]) -> bool:
    boxes = getattr(result, "boxes", None)
    if boxes is None or getattr(boxes, "cls", None) is None:
        return False
    for cid in boxes.cls.tolist():
        cname = names.get(int(cid), str(int(cid)))
        if not is_negative_class(cname):
            return True
    return False


def choose_split(cfg: Dict[str, Any]) -> str:
    if "test" in cfg:
        return "test"
    if "val" in cfg:
        return "val"
    return "train"


def pick_metric(d: Dict[str, Any], keys: Sequence[str], default: float = math.nan) -> float:
    for k in keys:
        if k in d and d[k] is not None:
            try:
                return float(d[k])
            except Exception:
                continue
    return default


def extract_det_metrics(val_ret: Any) -> Dict[str, float]:
    d = getattr(val_ret, "results_dict", {}) or {}
    p = pick_metric(d, ["metrics/precision(B)", "metrics/precision", "precision"])
    r = pick_metric(d, ["metrics/recall(B)", "metrics/recall", "recall"])
    m50 = pick_metric(d, ["metrics/mAP50(B)", "metrics/mAP50", "map50"])
    m5095 = pick_metric(d, ["metrics/mAP50-95(B)", "metrics/mAP50-95", "map"])

    if math.isnan(p) or math.isnan(r) or math.isnan(m50) or math.isnan(m5095):
        box = getattr(val_ret, "box", None)
        if box is not None:
            p = float(getattr(box, "mp", p))
            r = float(getattr(box, "mr", r))
            m50 = float(getattr(box, "map50", m50))
            m5095 = float(getattr(box, "map", m5095))

    return {"precision": p, "recall": r, "map50": m50, "map50_95": m5095}


def calc_img_binary(tp: int, fp: int, fn: int, tn: int) -> Dict[str, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else math.nan
    recall = tp / (tp + fn) if (tp + fn) > 0 else math.nan
    f1 = (
        2 * precision * recall / (precision + recall)
        if (not math.isnan(precision) and not math.isnan(recall) and (precision + recall) > 0)
        else math.nan
    )
    acc = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else math.nan
    return {
        "img_precision": precision,
        "img_recall": recall,
        "img_f1": f1,
        "img_acc": acc,
    }


def eval_image_level(
    model: YOLO,
    data_yaml: Path,
    split: str,
    batch: int,
    conf: float,
    iou: float,
    max_det: int,
    imgsz: int,
    device: str,
) -> Dict[str, Any]:
    cfg = read_yaml(data_yaml)
    names = normalize_names(cfg.get("names", {}))
    imgs = sorted({p.resolve() for p in collect_images(cfg.get(split), data_yaml.parent)})

    if not imgs:
        return {
            "num_images": 0,
            "img_tp": 0,
            "img_fp": 0,
            "img_fn": 0,
            "img_tn": 0,
            "img_precision": math.nan,
            "img_recall": math.nan,
            "img_f1": math.nan,
            "img_acc": math.nan,
        }

    gt_map: Dict[Path, bool] = {}
    for p in imgs:
        gt_map[p] = gt_has_defect(guess_label_file(p), names)

    tp = fp = fn = tn = 0
    for res in model.predict(
        source=[str(x) for x in imgs],
        conf=conf,
        iou=iou,
        max_det=max_det,
        imgsz=imgsz,
        batch=batch,
        device=device,
        stream=True,
        verbose=False,
    ):
        ip = Path(res.path).resolve()
        gt_pos = gt_map.get(ip, False)
        pred_pos = pred_has_defect(res, names)
        if gt_pos and pred_pos:
            tp += 1
        elif (not gt_pos) and pred_pos:
            fp += 1
        elif gt_pos and (not pred_pos):
            fn += 1
        else:
            tn += 1

    return {"num_images": len(imgs), "img_tp": tp, "img_fp": fp, "img_fn": fn, "img_tn": tn, **calc_img_binary(tp, fp, fn, tn)}


def write_rows_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def save_markdown_table(rows: List[Dict[str, Any]], cols: List[str], path: Path) -> None:
    if not rows:
        path.write_text("| empty |\n|---|\n", encoding="utf-8")
        return
    lines = [
        "|" + "|".join(cols) + "|",
        "|" + "|".join(["---"] * len(cols)) + "|",
    ]
    for row in rows:
        vals: List[str] = []
        for c in cols:
            v = row.get(c, "")
            if isinstance(v, float):
                vals.append("NaN" if math.isnan(v) else f"{v:.6f}")
            else:
                vals.append(str(v))
        lines.append("|" + "|".join(vals) + "|")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def find_data_yaml(dataset_root: Path, alias: str, explicit: Optional[str]) -> Optional[Path]:
    if explicit:
        p = Path(explicit).resolve()
        return p if p.exists() else None

    candidates: List[Path] = []

    for base in [dataset_root, PROJECT_ROOT / "configs"]:
        if not base.exists():
            continue
        for p in base.rglob("*.yaml"):
            lp = p.as_posix().lower()
            if (
                alias.lower() in lp
                or dataset_root.name.lower() in lp
                or p.name.lower() in {"data.yaml", "dataset.yaml", "defect.yaml", "defect241.yaml"}
            ):
                candidates.append(p.resolve())

    uniq = sorted(set(candidates), key=lambda x: (0 if str(x).startswith(str(dataset_root)) else 1, len(str(x))))
    for y in uniq:
        try:
            cfg = read_yaml(y)
        except Exception:
            continue
        if is_data_yaml(cfg):
            return y
    return None


def main() -> None:
    args = parse_args()

    out_root = PROJECT_ROOT / "experiments" / f"smoke_4ds_bs46_e2_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    (out_root / "tables").mkdir(parents=True, exist_ok=True)
    (out_root / "logs").mkdir(parents=True, exist_ok=True)

    only = set(args.only or [])
    items = [x for x in DATASET_CANDIDATES if (not only or x["alias"] in only)]

    det_rows: List[Dict[str, Any]] = []
    img_rows: List[Dict[str, Any]] = []
    status_rows: List[Dict[str, Any]] = []

    for ds in items:
        alias = ds["alias"]
        ds_root = Path(ds["root"]).resolve()
        data_yaml = find_data_yaml(ds_root, alias, ds.get("data_yaml"))

        if data_yaml is None:
            for bs in args.batches:
                status_rows.append(
                    {
                        "dataset": alias,
                        "batch": bs,
                        "epochs": args.epochs,
                        "status": "failed",
                        "error": f"data.yaml not found (root={ds_root})",
                    }
                )
            continue

        cfg = read_yaml(data_yaml)
        split = choose_split(cfg)

        for bs in args.batches:
            run_name = f"{alias}_bs{bs}_e{args.epochs}"
            try:
                model = YOLO(args.model)
                train_ret = model.train(
                    data=str(data_yaml),
                    epochs=args.epochs,
                    batch=bs,
                    imgsz=args.imgsz,
                    device=args.device,
                    workers=args.workers,
                    val=True,
                    project=str(out_root / "runs"),
                    name=run_name,
                    exist_ok=True,
                    verbose=True,
                )

                run_dir = Path(getattr(train_ret, "save_dir", out_root / "runs" / run_name))
                best = run_dir / "weights" / "best.pt"
                if not best.exists():
                    best = run_dir / "weights" / "last.pt"

                best_model = YOLO(str(best))
                val_ret = best_model.val(
                    data=str(data_yaml),
                    split=split,
                    batch=bs,
                    imgsz=args.imgsz,
                    device=args.device,
                    conf=args.conf,
                    iou=args.iou,
                    max_det=args.max_det,
                    project=str(out_root / "val"),
                    name=run_name,
                    exist_ok=True,
                    verbose=True,
                )

                det_m = extract_det_metrics(val_ret)
                img_m = eval_image_level(
                    model=best_model,
                    data_yaml=data_yaml,
                    split=split,
                    batch=bs,
                    conf=args.conf,
                    iou=args.iou,
                    max_det=args.max_det,
                    imgsz=args.imgsz,
                    device=args.device,
                )

                det_rows.append(
                    {
                        "dataset": alias,
                        "batch": bs,
                        "epochs": args.epochs,
                        "split": split,
                        "data_yaml": str(data_yaml),
                        "run_dir": str(run_dir),
                        "best_pt": str(best),
                        **det_m,
                    }
                )
                img_rows.append(
                    {
                        "dataset": alias,
                        "batch": bs,
                        "epochs": args.epochs,
                        "split": split,
                        "data_yaml": str(data_yaml),
                        "run_dir": str(run_dir),
                        "best_pt": str(best),
                        **img_m,
                    }
                )
                status_rows.append(
                    {
                        "dataset": alias,
                        "batch": bs,
                        "epochs": args.epochs,
                        "status": "ok",
                        "det_metric_ok": int(all(not math.isnan(det_m[k]) for k in ["precision", "recall", "map50", "map50_95"])),
                        "img_metric_ok": int(img_m["num_images"] > 0),
                        "error": "",
                    }
                )
            except Exception as e:
                status_rows.append(
                    {
                        "dataset": alias,
                        "batch": bs,
                        "epochs": args.epochs,
                        "status": "failed",
                        "error": f"{type(e).__name__}: {e}",
                    }
                )
                with open(out_root / "logs" / f"{run_name}.traceback.log", "w", encoding="utf-8") as f:
                    f.write(traceback.format_exc())

    det_csv = out_root / "tables" / "det_metrics.csv"
    img_csv = out_root / "tables" / "image_metrics.csv"
    status_csv = out_root / "tables" / "run_status.csv"

    det_fields = [
        "dataset",
        "batch",
        "epochs",
        "split",
        "data_yaml",
        "run_dir",
        "best_pt",
        "precision",
        "recall",
        "map50",
        "map50_95",
    ]
    img_fields = [
        "dataset",
        "batch",
        "epochs",
        "split",
        "data_yaml",
        "run_dir",
        "best_pt",
        "num_images",
        "img_tp",
        "img_fp",
        "img_fn",
        "img_tn",
        "img_precision",
        "img_recall",
        "img_f1",
        "img_acc",
    ]
    status_fields = [
        "dataset",
        "batch",
        "epochs",
        "status",
        "det_metric_ok",
        "img_metric_ok",
        "error",
    ]

    write_rows_csv(det_csv, det_rows, det_fields)
    write_rows_csv(img_csv, img_rows, img_fields)
    write_rows_csv(status_csv, status_rows, status_fields)

    join_cols = ("dataset", "batch", "epochs", "split", "data_yaml", "run_dir", "best_pt")
    det_map = {tuple(r.get(k, "") for k in join_cols): r for r in det_rows}
    img_map = {tuple(r.get(k, "") for k in join_cols): r for r in img_rows}
    all_keys = sorted(set(det_map.keys()) | set(img_map.keys()))

    summary_rows: List[Dict[str, Any]] = []
    for k in all_keys:
        row: Dict[str, Any] = {}
        d = det_map.get(k, {})
        i = img_map.get(k, {})
        for c in join_cols:
            row[c] = d.get(c, i.get(c, ""))
        for c in ["precision", "recall", "map50", "map50_95"]:
            row[c] = d.get(c, "")
        for c in ["num_images", "img_tp", "img_fp", "img_fn", "img_tn", "img_precision", "img_recall", "img_f1", "img_acc"]:
            row[c] = i.get(c, "")
        summary_rows.append(row)

    summary_csv = out_root / "tables" / "summary_join.csv"
    summary_md = out_root / "tables" / "summary_join.md"
    summary_fields = [
        "dataset",
        "batch",
        "epochs",
        "split",
        "data_yaml",
        "run_dir",
        "best_pt",
        "precision",
        "recall",
        "map50",
        "map50_95",
        "num_images",
        "img_tp",
        "img_fp",
        "img_fn",
        "img_tn",
        "img_precision",
        "img_recall",
        "img_f1",
        "img_acc",
    ]
    write_rows_csv(summary_csv, summary_rows, summary_fields)
    save_markdown_table(summary_rows, summary_fields, summary_md)

    meta = {
        "project_root": str(PROJECT_ROOT),
        "output_root": str(out_root),
        "model": args.model,
        "device": args.device,
        "imgsz": args.imgsz,
        "epochs": args.epochs,
        "batches": args.batches,
        "conf": args.conf,
        "iou": args.iou,
        "max_det": args.max_det,
        "workers": args.workers,
        "datasets": items,
    }
    with open(out_root / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[done] out={out_root}")
    print(f"[done] det_csv={det_csv}")
    print(f"[done] img_csv={img_csv}")
    print(f"[done] status_csv={status_csv}")
    print(f"[done] summary_md={summary_md}")


if __name__ == "__main__":
    main()
