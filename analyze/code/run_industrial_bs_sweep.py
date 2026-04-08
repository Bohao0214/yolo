#!/usr/bin/env python3
"""
工业缺陷数据集批量训练+评估脚本（4数据集，batch 扫描）。

功能：
1) 对 4 个工业缺陷检测数据集按 batch 列表逐个训练（默认 epoch=2）。
2) 每个组合输出检测指标：Precision/Recall/mAP50/mAP50-95（val/test）。
3) 额外输出图像级指标（有框=有缺陷）：TP/FN/FP/TN、image_precision/image_recall/image_fpr。
4) 结果统一写入 experiments 下新目录，便于服务器直接跑。

默认数据集：
- /home/ubuntu/hpproject/yolo/dataset/yolo/DeepPCB_standard/data.yaml
- /home/ubuntu/hpproject/yolo/dataset/yolo/kolektorsdd_622_halves/data.yaml
- /home/ubuntu/hpproject/yolo/dataset/yolo/neudet_622/data.yaml
- /home/ubuntu/hpproject/yolo/dataset/yolo/gc10det_622_halves/data.yaml

用法：
python /home/ubuntu/hpproject/yolo/analyze/code/run_industrial_bs_sweep.py \
  --epochs 2 \
  --batches 4 6 \
  --device 0

产物：
/home/ubuntu/hpproject/yolo/experiments/industrial_bs_sweep_YYMMDD_HHMM/
  - summary.csv
  - summary.json
  - <dataset>_bs<batch>_e<epoch>/...  (Ultralytics 原生训练输出)
  - 每个训练目录下 image_level/test_image_level.csv|json
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import yaml
from ultralytics import YOLO

try:
    from src.eval import compute_image_level_results, save_image_level_report
except Exception:
    compute_image_level_results = None
    save_image_level_report = None

ROOT = Path("/home/ubuntu/hpproject/yolo")
DEFAULT_MODEL = ROOT / "models" / "pretrained" / "yolo11m.pt"
DEFAULT_OUT_ROOT = ROOT / "experiments"

DEFAULT_DATASET_CANDIDATES: Dict[str, List[Path]] = {
    "DeepPCB_standard": [
        ROOT / "dataset" / "yolo" / "DeepPCB_standard" / "data.yaml",
        ROOT / "dataset" / "yolo" / "DeepPCB_standard" / "dataset.yaml",
    ],
    "kolektorsdd_622_halves": [
        ROOT / "dataset" / "yolo" / "kolektorsdd_622_halves" / "data.yaml",
        ROOT / "dataset" / "yolo" / "kolektorsdd_622_halves" / "dataset.yaml",
    ],
    "neudet_622": [
        ROOT / "dataset" / "yolo" / "neudet_622" / "data.yaml",
        ROOT / "dataset" / "yolo" / "neudet_622" / "dataset.yaml",
    ],
    "gc10det_622_halves": [
        ROOT / "experiments" / "gc10det_622_halves" / "data.yaml",
        ROOT / "experiments" / "gc10det_622_halves" / "dataset.yaml",
        ROOT / "dataset" / "yolo" / "gc10det_622_halves" / "data.yaml",
        ROOT / "dataset" / "yolo" / "gc10det_622_halves" / "dataset.yaml",
    ],
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run 4 industrial datasets with batch sweep and epoch smoke training.")
    p.add_argument("--model", type=str, default=str(DEFAULT_MODEL), help="Path to pretrained model (e.g., yolo11m.pt)")
    p.add_argument("--epochs", type=int, default=2, help="Training epochs for smoke run")
    p.add_argument("--batches", type=int, nargs="+", default=[4, 6], help="Batch sizes to test")
    p.add_argument("--imgsz", type=int, default=640, help="Image size")
    p.add_argument("--device", type=str, default="0", help='CUDA device, e.g. "0" or "cpu"')
    p.add_argument("--workers", type=int, default=4, help="Dataloader workers")
    p.add_argument("--seed", type=int, default=0, help="Random seed")
    p.add_argument("--metric_conf", type=float, default=0.01, help="Predict conf used before image-level scoring")
    p.add_argument("--image_conf_thr", type=float, default=0.25, help="Image-level decision threshold")
    p.add_argument("--image_iou_match", type=float, default=0.3, help="IoU threshold for GT-pred match at image/object analysis")
    p.add_argument("--image_eval_batch", type=int, default=1, help="Batch size for image-level predict (recommended small to avoid OOM)")
    p.add_argument(
        "--image_eval_device",
        type=str,
        default="cpu",
        help='Device for image-level predict, e.g. "cpu" or "0". Empty means follow --device.',
    )
    p.add_argument("--nms_iou", type=float, default=0.7, help="NMS IoU")
    p.add_argument("--max_det", type=int, default=300, help="Max detections after NMS")
    p.add_argument("--run_prefix", type=str, default="industrial_bs_sweep", help="Output directory prefix under experiments")
    p.add_argument("--out_root", type=str, default=str(DEFAULT_OUT_ROOT), help="Output root directory")
    p.add_argument("--train_only", action="store_true", help="只训练并导出产物，不做 val/test 与图像级统计")
    p.add_argument("--dry_run", action="store_true", help="Only print resolved plan, do not run training")
    return p.parse_args()


def safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)


def safe_ratio(n: int, d: int) -> float:
    return float(n) / float(d) if d > 0 else 0.0


def load_yaml(path: Path) -> Dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"YAML top-level is not mapping: {path}")
    return data


def resolve_data_root(data_yaml: Path, info: Dict[str, Any]) -> Path:
    root_raw = str(info.get("path", "")).strip()
    if root_raw:
        root = Path(root_raw)
        if not root.is_absolute():
            root = (data_yaml.parent / root).resolve()
        return root
    return data_yaml.parent.resolve()


def resolve_split_sources(data_yaml: Path, split_key: str) -> List[Path]:
    info = load_yaml(data_yaml)
    data_root = resolve_data_root(data_yaml, info)
    entry = info.get(split_key)
    if entry is None:
        return []

    refs: List[str] = []
    if isinstance(entry, str):
        refs = [x.strip() for x in entry.split(",") if x.strip()]
    elif isinstance(entry, (list, tuple)):
        refs = [str(x).strip() for x in entry if str(x).strip()]
    else:
        refs = [str(entry).strip()]

    out: List[Path] = []
    for ref in refs:
        p = Path(ref)
        if not p.is_absolute():
            p = (data_root / p).resolve()
        out.append(p)
    return out


def choose_eval_split(data_yaml: Path, prefer: str = "test") -> str:
    info = load_yaml(data_yaml)
    if prefer in info and info.get(prefer) not in (None, "", []):
        return prefer
    if "val" in info and info.get("val") not in (None, "", []):
        return "val"
    if "train" in info and info.get("train") not in (None, "", []):
        return "train"
    return prefer


def resolve_dataset_map() -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    missing: Dict[str, List[str]] = {}
    for name, cands in DEFAULT_DATASET_CANDIDATES.items():
        hit: Optional[Path] = None
        for p in cands:
            q = p.resolve()
            if q.exists():
                hit = q
                break
        if hit is None:
            missing[name] = [str(x.resolve()) for x in cands]
        else:
            out[name] = hit

    if missing:
        lines = ["data.yaml not found for datasets:"]
        for name, cands in missing.items():
            lines.append(f"- {name}")
            for c in cands:
                lines.append(f"  * {c}")
        raise FileNotFoundError("\n".join(lines))
    return out


def extract_det_metrics(metrics_obj: Any) -> Dict[str, float | None]:
    out: Dict[str, float | None] = {
        "precision": None,
        "recall": None,
        "map50": None,
        "map50_95": None,
    }
    try:
        rd = getattr(metrics_obj, "results_dict", None)
        if isinstance(rd, dict):
            out["precision"] = rd.get("metrics/precision(B)")
            out["recall"] = rd.get("metrics/recall(B)")
            out["map50"] = rd.get("metrics/mAP50(B)")
            out["map50_95"] = rd.get("metrics/mAP50-95(B)")
    except Exception:
        pass

    try:
        b = getattr(metrics_obj, "box", None)
        if b is not None:
            if out["precision"] is None:
                out["precision"] = getattr(b, "mp", None)
            if out["recall"] is None:
                out["recall"] = getattr(b, "mr", None)
            if out["map50"] is None:
                out["map50"] = getattr(b, "map50", None)
            if out["map50_95"] is None:
                out["map50_95"] = getattr(b, "map", None)
    except Exception:
        pass
    return out


def fmt_metric(v: float | None) -> str:
    if v is None:
        return ""
    try:
        return f"{float(v):.6f}"
    except Exception:
        return ""


def write_rows_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def make_run_dir(out_root: Path, prefix: str) -> Path:
    ts = dt.datetime.now().strftime("%y%m%d_%H%M")
    cand = out_root / f"{prefix}_{ts}"
    if not cand.exists():
        cand.mkdir(parents=True, exist_ok=False)
        return cand
    idx = 1
    while True:
        p = out_root / f"{prefix}_{ts}_{idx:02d}"
        if not p.exists():
            p.mkdir(parents=True, exist_ok=False)
            return p
        idx += 1


def summarize_image_level(items: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    tp = sum(1 for x in items if str(x.get("outcome", "")) == "TP")
    fn = sum(1 for x in items if str(x.get("outcome", "")) == "FN")
    fp = sum(1 for x in items if str(x.get("outcome", "")) == "FP")
    tn = sum(1 for x in items if str(x.get("outcome", "")) == "TN")

    obj_fn = sum(int(x.get("obj_fn", 0) or 0) for x in items)
    obj_fp = sum(int(x.get("obj_fp", 0) or 0) for x in items)

    return {
        "img_tp": int(tp),
        "img_fn": int(fn),
        "img_fp": int(fp),
        "img_tn": int(tn),
        "image_precision": safe_ratio(tp, tp + fp),
        "image_recall": safe_ratio(tp, tp + fn),
        "image_fpr": safe_ratio(fp, fp + tn),
        "obj_fn_total": int(obj_fn),
        "obj_fp_total": int(obj_fp),
    }


def iter_plans(dataset_map: Dict[str, Path], batches: Iterable[int], epochs: int) -> List[Tuple[str, Path, int, int]]:
    plans: List[Tuple[str, Path, int, int]] = []
    for name, y in dataset_map.items():
        for bs in batches:
            plans.append((name, y, int(bs), int(epochs)))
    return plans


def main() -> None:
    args = parse_args()

    model_path = Path(args.model).expanduser().resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"model not found: {model_path}")

    dataset_map = resolve_dataset_map()

    out_root = Path(args.out_root).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    run_dir = make_run_dir(out_root, args.run_prefix)

    plans = iter_plans(dataset_map, args.batches, args.epochs)
    print(f"[plan] run_dir={run_dir}")
    for i, (name, y, bs, ep) in enumerate(plans, 1):
        print(f"[plan] {i}/{len(plans)} dataset={name} batch={bs} epochs={ep} data={y}")

    if args.dry_run:
        return

    rows: List[Dict[str, Any]] = []
    summary_fields = [
        "dataset",
        "batch",
        "epochs",
        "mode",
        "data_yaml",
        "train_run_dir",
        "best_pt",
        "val_precision",
        "val_recall",
        "val_map50",
        "val_map50_95",
        "test_precision",
        "test_recall",
        "test_map50",
        "test_map50_95",
        "img_tp",
        "img_fn",
        "img_fp",
        "img_tn",
        "image_precision",
        "image_recall",
        "image_fpr",
        "obj_fn_total",
        "obj_fp_total",
        "status",
        "error",
    ]

    for idx, (dataset_name, data_yaml, batch, epochs) in enumerate(plans, 1):
        row: Dict[str, Any] = {
            "dataset": dataset_name,
            "batch": int(batch),
            "epochs": int(epochs),
            "mode": "train_only" if args.train_only else "full_eval",
            "data_yaml": str(data_yaml),
            "train_run_dir": "",
            "best_pt": "",
            "val_precision": "",
            "val_recall": "",
            "val_map50": "",
            "val_map50_95": "",
            "test_precision": "",
            "test_recall": "",
            "test_map50": "",
            "test_map50_95": "",
            "img_tp": 0,
            "img_fn": 0,
            "img_fp": 0,
            "img_tn": 0,
            "image_precision": 0.0,
            "image_recall": 0.0,
            "image_fpr": 0.0,
            "obj_fn_total": 0,
            "obj_fp_total": 0,
            "status": "ok",
            "error": "",
        }

        combo_name = f"{dataset_name}_bs{batch}_e{epochs}"
        print(f"\n[run] ({idx}/{len(plans)}) {combo_name}")

        try:
            model = YOLO(str(model_path))
            train_res = model.train(
                data=str(data_yaml),
                epochs=int(epochs),
                batch=int(batch),
                imgsz=int(args.imgsz),
                device=str(args.device),
                workers=int(args.workers),
                project=str(run_dir),
                name=combo_name,
                exist_ok=True,
                pretrained=True,
                optimizer="auto",
                seed=int(args.seed),
                deterministic=True,
                val=(not args.train_only),
                save=True,
                save_json=False,
                plots=True,
                verbose=True,
            )

            save_dir = Path(str(getattr(train_res, "save_dir", run_dir / combo_name))).resolve()
            best_pt = save_dir / "weights" / "best.pt"
            if not best_pt.exists():
                best_pt = save_dir / "weights" / "last.pt"

            row["train_run_dir"] = str(save_dir)
            row["best_pt"] = str(best_pt)

            if args.train_only:
                print(
                    "[done] {name} bs={bs} e={ep} weights={w}".format(
                        name=dataset_name,
                        bs=batch,
                        ep=epochs,
                        w=row["best_pt"],
                    )
                )
                rows.append(row)
                write_rows_csv(run_dir / "summary.csv", rows, summary_fields)
                (run_dir / "summary.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
                continue

            m_val = extract_det_metrics(train_res)
            row["val_precision"] = fmt_metric(m_val["precision"])
            row["val_recall"] = fmt_metric(m_val["recall"])
            row["val_map50"] = fmt_metric(m_val["map50"])
            row["val_map50_95"] = fmt_metric(m_val["map50_95"])

            eval_model = YOLO(str(best_pt))
            eval_split = choose_eval_split(data_yaml, prefer="test")
            test_res = eval_model.val(
                data=str(data_yaml),
                split=eval_split,
                imgsz=int(args.imgsz),
                batch=int(batch),
                device=str(args.device),
                workers=int(args.workers),
                conf=0.001,
                iou=float(args.nms_iou),
                max_det=int(args.max_det),
                plots=False,
                verbose=False,
            )
            m_test = extract_det_metrics(test_res)
            row["test_precision"] = fmt_metric(m_test["precision"])
            row["test_recall"] = fmt_metric(m_test["recall"])
            row["test_map50"] = fmt_metric(m_test["map50"])
            row["test_map50_95"] = fmt_metric(m_test["map50_95"])

            # Image-level metrics are optional side stats. If they fail (often OOM), keep detection metrics.
            try:
                if compute_image_level_results is None or save_image_level_report is None:
                    raise RuntimeError("src.eval not available, image-level skipped")
                image_eval_batch = max(1, int(args.image_eval_batch))
                image_eval_device = str(args.image_eval_device).strip() or str(args.device)
                test_sources = resolve_split_sources(data_yaml, eval_split)
                all_items: List[Dict[str, Any]] = []
                for source in test_sources:
                    if not source.exists():
                        print(f"[warn] test source not found: {source}")
                        continue
                    cur = compute_image_level_results(
                        model=eval_model,
                        source=source,
                        conf_threshold=float(args.image_conf_thr),
                        iou_match=float(args.image_iou_match),
                        metric_conf=float(args.metric_conf),
                        batch=image_eval_batch,
                        device=image_eval_device,
                        nms_iou=float(args.nms_iou),
                        max_det=int(args.max_det),
                        split="test",
                        vis_root=None,
                        save_visuals=False,
                    )
                    all_items.extend(cur)

                if all_items:
                    img_stats = summarize_image_level(all_items)
                    row.update(img_stats)
                    meta = {
                        "dataset": dataset_name,
                        "data_yaml": str(data_yaml),
                        "split": "test",
                        "batch": int(batch),
                        "epochs": int(epochs),
                        "conf_threshold": float(args.image_conf_thr),
                        "iou_match": float(args.image_iou_match),
                        "metric_conf": float(args.metric_conf),
                        "nms_iou": float(args.nms_iou),
                        "max_det": int(args.max_det),
                        "image_eval_batch": int(image_eval_batch),
                        "image_eval_device": image_eval_device,
                    }
                    save_image_level_report(save_dir / "image_level", "test", all_items, meta)
                else:
                    row["status"] = "partial"
                    row["error"] = "image-level items empty (test source missing or predict failed)"
            except Exception as image_exc:
                row["status"] = "partial"
                row["error"] = f"image_level_failed: {type(image_exc).__name__}: {image_exc}"
                print(f"[warn] image-level failed for {combo_name}: {row['error']}")

            print(
                "[done] {name} bs={bs} e={ep} test_map50={m50} image_recall={irec}".format(
                    name=dataset_name,
                    bs=batch,
                    ep=epochs,
                    m50=row["test_map50"],
                    irec=f"{safe_float(row['image_recall']):.6f}",
                )
            )

        except Exception as exc:
            row["status"] = "failed"
            row["error"] = f"{type(exc).__name__}: {exc}"
            print(f"[error] {combo_name}: {row['error']}")
            print(traceback.format_exc())

        rows.append(row)

        # Incremental flush for long jobs.
        write_rows_csv(run_dir / "summary.csv", rows, summary_fields)
        (run_dir / "summary.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n[all_done] {run_dir / 'summary.csv'}")
    print(f"[all_done] {run_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
