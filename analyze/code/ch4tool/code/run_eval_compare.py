#!/usr/bin/env python3
"""
用法说明（统一对比评估，不重训）：

默认会把当前脚本所在目录的上一级作为报告根目录：
/home/ubuntu/hpproject/yolo/analyze/code/ch4tool

执行前请先填写：
1. models_to_eval.txt（每行一个模型权重绝对路径）
2. dataset_override.txt（可选，写 dataset_root=/abs/path 和 data_yaml=/abs/path）

示例：
conda run -n yolo11 python /home/ubuntu/hpproject/yolo/analyze/code/ch4tool/code/run_eval_compare.py \
  --report-root /home/ubuntu/hpproject/yolo/analyze/code/ch4tool

可选参数（覆盖自动参数）：
--imgsz --conf --iou --max-det --batch --device --score-thr --obj-iou --split
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


SCRIPT_PATH = Path(__file__).resolve()
REPORT_ROOT = SCRIPT_PATH.parents[1]
PROJECT_ROOT = SCRIPT_PATH.parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPT_PATH.parent) not in sys.path:
    sys.path.insert(0, str(SCRIPT_PATH.parent))

from tools.eval_detection_benchmark import load_ground_truth
from utils_common import (
    ModelSpec,
    append_log,
    auto_find_data_yaml_from_cfg,
    build_model_specs,
    choose_unified_eval_params,
    compute_fn_fp_breakdown,
    compute_scale_recall_and_image_fp,
    compute_metrics,
    ensure_dir,
    find_nearby_config,
    markdown_table,
    parse_dataset_root,
    parse_eval_params_from_cfg,
    parse_model_paths_txt,
    run_yolo_predictions,
    save_pred_json,
    select_baseline_and_best,
    short_exc,
    write_csv,
    write_json,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run unified YOLO compare evaluation on existing weights (no retraining).")
    p.add_argument("--report-root", type=Path, default=REPORT_ROOT)
    p.add_argument("--models-txt", type=Path, default=None)
    p.add_argument("--dataset-root", type=Path, default=None, help="Optional dataset root override (contains images/labels).")
    p.add_argument("--data-yaml", type=Path, default=None, help="Optional data.yaml override.")
    p.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])

    p.add_argument("--imgsz", type=int, default=None)
    p.add_argument("--conf", type=float, default=None)
    p.add_argument("--iou", type=float, default=None)
    p.add_argument("--max-det", type=int, default=None)
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--device", type=str, default=None)

    p.add_argument("--score-thr", type=float, default=0.25, help="Object-level score threshold for Precision/Recall.")
    p.add_argument("--obj-iou", type=float, default=0.5, help="Object-level IoU threshold for Precision/Recall.")
    return p.parse_args()


def _read_override_file(path: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not path.exists():
        return out
    for ln in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        if "=" in s:
            k, v = s.split("=", 1)
            out[k.strip().lower()] = v.strip()
        else:
            # single line fallback as dataset root
            out.setdefault("dataset_root", s)
    return out


def _ensure_templates(report_root: Path) -> Tuple[Path, Path]:
    models_txt = report_root / "models_to_eval.txt"
    override_txt = report_root / "dataset_override.txt"

    if not models_txt.exists():
        models_txt.write_text(
            """# 每行一个模型权重绝对路径（best.pt/last.pt）
# 你只需要在这个文件里填写路径：
# /abs/path/to/model1/best.pt
# /abs/path/to/model2/best.pt
# /abs/path/to/model3/best.pt
# /abs/path/to/model4/best.pt

/home/ubuntu/hpproject/yolo/experiments/a7b7c7d7/datasetm6c/exp_2604050107/train/weights/best.pt
/home/ubuntu/hpproject/yolo/experiments/a4b7d6/datasetm6c/defect241__a4__b7__d6/exp_2603060619/train/weights/best.pt
/home/ubuntu/hpproject/yolo/experiments/a3b3d3/datasetm6c/exp_2604050042/train/weights/best.pt
/home/ubuntu/hpproject/yolo/experiments/yolo11/defect241__a3__c5/exp_2603032345/train/weights/best.pt
""",
            encoding="utf-8",
        )

    if not override_txt.exists():
        override_txt.write_text(
            """# 可选：手动覆盖数据集路径与 data.yaml
# dataset_root=/abs/path/to/dataset_root
# data_yaml=/abs/path/to/data.yaml
""",
            encoding="utf-8",
        )
    return models_txt, override_txt


def _resolve_dataset_and_params(
    specs: List[ModelSpec],
    args: argparse.Namespace,
    override_kv: Dict[str, str],
) -> Tuple[Optional[Path], Optional[Path], Dict[str, object], List[str]]:
    for spec in specs:
        wp = Path(spec.model_path).expanduser().resolve()
        if not wp.exists():
            spec.status = "missing_model"
            spec.error = f"weights not found: {wp}"
            continue
        cfg = find_nearby_config(wp)
        if cfg is not None:
            spec.config_path = str(cfg)
            data_yaml = auto_find_data_yaml_from_cfg(cfg)
            if data_yaml is not None:
                spec.data_yaml = str(data_yaml)

    # determine unified data.yaml
    data_yaml: Optional[Path] = None
    if args.data_yaml is not None:
        p = args.data_yaml.expanduser().resolve()
        if p.exists():
            data_yaml = p
    if data_yaml is None and "data_yaml" in override_kv:
        p = Path(override_kv["data_yaml"]).expanduser().resolve()
        if p.exists():
            data_yaml = p
    if data_yaml is None:
        for s in specs:
            if s.data_yaml:
                p = Path(s.data_yaml)
                if p.exists():
                    data_yaml = p
                    break

    # eval params from existing configs
    param_dicts = []
    for s in specs:
        if s.config_path:
            param_dicts.append(parse_eval_params_from_cfg(Path(s.config_path)))
    overrides = {
        "imgsz": args.imgsz,
        "conf": args.conf,
        "iou": args.iou,
        "max_det": args.max_det,
        "batch": args.batch,
        "device": args.device,
        "score_thr": args.score_thr,
        "obj_iou": args.obj_iou,
    }
    eval_params, pending_eval = choose_unified_eval_params(param_dicts, overrides)

    # dataset root
    ds_root_override: Optional[Path] = None
    if args.dataset_root is not None:
        ds_root_override = args.dataset_root.expanduser().resolve()
    elif "dataset_root" in override_kv:
        ds_root_override = Path(override_kv["dataset_root"]).expanduser().resolve()

    dataset_root, pending_ds = parse_dataset_root(data_yaml=data_yaml, fallback_root=ds_root_override)
    pending = list(dict.fromkeys(pending_eval + pending_ds))
    return dataset_root, data_yaml, eval_params, pending


def _load_gt_with_possible_fallback(dataset_root: Path, split: str):
    used_split = split
    try:
        image_paths, gt_map, gt_classes = load_ground_truth(dataset_root=dataset_root, split=split)
        return image_paths, gt_map, gt_classes, used_split, ""
    except Exception as e:
        # fallback only when requested test but dataset has no test split
        if split == "test":
            try:
                image_paths, gt_map, gt_classes = load_ground_truth(dataset_root=dataset_root, split="val")
                used_split = "val"
                return image_paths, gt_map, gt_classes, used_split, f"fallback split=test->val due to: {e}"
            except Exception:
                pass
        raise


def main() -> None:
    args = parse_args()
    report_root = args.report_root.expanduser().resolve()
    code_dir = ensure_dir(report_root / "code")
    tables_dir = ensure_dir(report_root / "tables")
    raw_preds_dir = ensure_dir(report_root / "raw_preds")
    logs_dir = ensure_dir(report_root / "logs")
    log_path = logs_dir / "run_eval_compare.log"

    models_txt_default, override_txt = _ensure_templates(report_root)
    models_txt = args.models_txt.expanduser().resolve() if args.models_txt is not None else models_txt_default

    append_log(log_path, f"report_root={report_root}")
    append_log(log_path, f"models_txt={models_txt}")

    model_paths = parse_model_paths_txt(models_txt)
    specs = build_model_specs(model_paths)
    override_kv = _read_override_file(override_txt)

    dataset_root, data_yaml, eval_params, pending = _resolve_dataset_and_params(specs, args, override_kv)

    gt_ready = False
    image_paths = []
    gt_map = {}
    gt_classes = []
    split_used = args.split
    split_note = ""

    if dataset_root is None:
        pending.extend(["dataset_root", "data_yaml", "eval_split(test/val)"])
        pending = list(dict.fromkeys(pending))
        append_log(log_path, "dataset root unresolved; will export status-only tables")
    else:
        try:
            image_paths, gt_map, gt_classes, split_used, split_note = _load_gt_with_possible_fallback(dataset_root, args.split)
            gt_ready = True
            if split_note:
                pending.append("eval_split(test/val)")
            append_log(log_path, f"dataset_root={dataset_root}, split_used={split_used}, n_images={len(image_paths)}")
        except Exception as e:
            pending.extend(["dataset_root", "data_yaml", "eval_split(test/val)"])
            pending = list(dict.fromkeys(pending))
            append_log(log_path, f"load_ground_truth failed: {short_exc(e)}")

    compare_rows: List[dict] = []
    scale_rows_all: List[dict] = []
    fn_rows_all: List[dict] = []
    fp_rows_all: List[dict] = []

    for spec in specs:
        row = {
            "model_name": spec.model_name,
            "exp_name": spec.exp_name,
            "model_path": spec.model_path,
            "ablation_tags": "+".join(spec.ablation_tags),
            "module_groups": "+".join(spec.module_groups),
            "n_tags": len(spec.ablation_tags),
            "status": spec.status if spec.status != "pending" else "pending",
            "error": spec.error,
            "precision": "",
            "recall": "",
            "map50": "",
            "map50_95": "",
            "tp": "",
            "fp": "",
            "fn": "",
            "image_fp_rate": "",
        }

        wp = Path(spec.model_path).expanduser().resolve()
        if not wp.exists():
            row["status"] = "missing_model"
            row["error"] = f"weights not found: {wp}"
            compare_rows.append(row)
            append_log(log_path, f"skip missing model: {wp}")
            continue

        if not gt_ready:
            row["status"] = "missing_dataset"
            row["error"] = "dataset/data.yaml unresolved or unreadable"
            compare_rows.append(row)
            append_log(log_path, f"skip model due to dataset unresolved: {wp}")
            continue

        try:
            append_log(log_path, f"predict start: {spec.model_name}")
            pred_map = run_yolo_predictions(
                model_path=str(wp),
                image_paths=image_paths,
                imgsz=int(eval_params["imgsz"]),
                conf=float(eval_params["conf"]),
                iou=float(eval_params["iou"]),
                max_det=int(eval_params["max_det"]),
                batch=int(eval_params["batch"]),
                device=str(eval_params["device"]),
            )

            pred_json_path = raw_preds_dir / spec.model_name / f"preds_{split_used}.json"
            save_pred_json(
                path=pred_json_path,
                model_name=spec.model_name,
                model_path=str(wp),
                dataset_root=dataset_root,
                split=split_used,
                pred_map=pred_map,
            )

            classes = gt_classes if gt_classes else [0]
            metrics = compute_metrics(
                gt_map=gt_map,
                pred_map=pred_map,
                classes=classes,
                score_thr=float(eval_params["score_thr"]),
                obj_iou=float(eval_params["obj_iou"]),
            )

            scale_rows, img_stats = compute_scale_recall_and_image_fp(
                gt_map=gt_map,
                pred_map=pred_map,
                score_thr=float(eval_params["score_thr"]),
                iou_thr=float(eval_params["obj_iou"]),
            )
            fn_table, fp_table, details = compute_fn_fp_breakdown(
                gt_map=gt_map,
                pred_map=pred_map,
                score_thr=float(eval_params["score_thr"]),
                iou_thr=float(eval_params["obj_iou"]),
            )

            for r in scale_rows:
                scale_rows_all.append({"model_name": spec.model_name, **r})
            for r in fn_table:
                fn_rows_all.append({"model_name": spec.model_name, **r})
            for r in fp_table:
                fp_rows_all.append({"model_name": spec.model_name, **r})

            # optional detail dumps
            write_json(raw_preds_dir / spec.model_name / "fn_fp_cases.json", details)

            row.update(
                {
                    "status": "ok",
                    "error": "",
                    "precision": f"{metrics['object_precision']:.6f}",
                    "recall": f"{metrics['object_recall']:.6f}",
                    "map50": f"{metrics['map50']:.6f}",
                    "map50_95": f"{metrics['map50_95']:.6f}",
                    "tp": int(metrics["tp"]),
                    "fp": int(metrics["fp"]),
                    "fn": int(metrics["fn"]),
                    "image_fp_rate": f"{img_stats['image_fp_rate']:.6f}",
                }
            )
            append_log(log_path, f"predict+eval done: {spec.model_name}")
        except Exception as e:
            row["status"] = "error"
            row["error"] = short_exc(e)
            append_log(log_path, f"error on {spec.model_name}: {short_exc(e)}")

        compare_rows.append(row)

    # persist main tables
    compare_fields = [
        "model_name",
        "exp_name",
        "model_path",
        "ablation_tags",
        "module_groups",
        "n_tags",
        "status",
        "error",
        "precision",
        "recall",
        "map50",
        "map50_95",
        "tp",
        "fp",
        "fn",
        "image_fp_rate",
    ]
    write_csv(tables_dir / "compare_main.csv", compare_rows, compare_fields)

    scale_fields = ["model_name", "scale_bucket", "GT", "TP", "FN", "recall"]
    write_csv(tables_dir / "scale_recall.csv", scale_rows_all, scale_fields)

    fn_fields = ["model_name", "diag_type", "count"]
    write_csv(tables_dir / "fn_mechanism.csv", fn_rows_all, fn_fields)

    fp_fields = ["model_name", "metric", "count"]
    write_csv(tables_dir / "fp_structure.csv", fp_rows_all, fp_fields)

    # markdown lightweight versions
    ok_for_md = []
    for r in compare_rows:
        ok_for_md.append(
            {
                "model": r["model_name"],
                "status": r["status"],
                "P": r["precision"],
                "R": r["recall"],
                "mAP@0.5": r["map50"],
                "mAP@0.5:0.95": r["map50_95"],
            }
        )
    (tables_dir / "compare_main.md").write_text(
        markdown_table(ok_for_md, ["model", "status", "P", "R", "mAP@0.5", "mAP@0.5:0.95"]),
        encoding="utf-8",
    )

    baseline_model, best_model, baseline_note = select_baseline_and_best(compare_rows, specs)

    pending = list(dict.fromkeys(pending))
    # user-requested must-have pending fields when unresolved
    if dataset_root is None:
        for key in ["dataset_root", "data_yaml", "eval_split(test/val)"]:
            if key not in pending:
                pending.append(key)

    metadata = {
        "report_root": str(report_root),
        "project_root": str(PROJECT_ROOT),
        "models_txt": str(models_txt),
        "dataset_override_txt": str(override_txt),
        "data_yaml": str(data_yaml) if data_yaml is not None else "",
        "dataset_root": str(dataset_root) if dataset_root is not None else "",
        "split_requested": args.split,
        "split_used": split_used,
        "split_note": split_note,
        "eval_params": eval_params,
        "pending_user_inputs": pending,
        "baseline_model": baseline_model,
        "best_model": best_model,
        "baseline_note": baseline_note,
        "model_specs": [asdict(s) for s in specs],
        "compare_rows": compare_rows,
        "note": "统一口径：掩膜转最小外接框后对比的是检测能力，不是分割精度。",
    }
    write_json(tables_dir / "metadata.json", metadata)

    print(f"[done] compare_main.csv -> {tables_dir / 'compare_main.csv'}")
    print(f"[done] ablation input metadata -> {tables_dir / 'metadata.json'}")
    print(f"[info] fill model paths in -> {models_txt}")


if __name__ == "__main__":
    main()
