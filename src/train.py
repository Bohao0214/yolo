import argparse
import datetime as dt
import gc
import shutil
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from data.make_split import create_val_split
from yolo_enhance import apply_yolo_enhancements
from eval import (
    compute_ap,
    compute_auc,
    compute_image_scores,
    compute_image_scores_iou,
    compute_image_level_results,
    compute_threshold_metrics,
    save_image_level_report,
    save_metric_plots,
    save_multi_curve,
    save_multi_xy_curves,
    save_threshold_table,
    save_threshold_table_multi,
    annotate_fp_fn_images,
)
from nms_patch import patch_nms


def load_yaml(path: Path) -> Dict[str, Any]:
    import yaml  # type: ignore

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML must be a mapping: {path}")
    return data


def resolve_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_data_root(data_yaml: Path) -> Path:
    info = load_yaml(data_yaml)
    root = info.get("path")
    if root:
        return Path(root).resolve()
    return data_yaml.parent.resolve()


def resolve_data_entry(entry: str, data_root: Path) -> Path:
    path = Path(entry)
    if path.is_absolute():
        return path
    return (data_root / path).resolve()


def resolve_pretrained_weight(project_root: Path, weight_name: str) -> Path:
    return (project_root / "models" / "pretrained" / weight_name).resolve()


def update_args_yaml(exp_dir: Path, cfg: Dict[str, Any], cfg_path: Path) -> None:
    args_path = exp_dir / "train" / "args.yaml"
    if not args_path.exists():
        return
    try:
        import yaml  # type: ignore
    except Exception:
        return
    with open(args_path, "r", encoding="utf-8") as f:
        args_data = yaml.safe_load(f) or {}
    if not isinstance(args_data, dict):
        return
    merged = dict(args_data)
    merged.update(cfg)
    merged["config_path"] = str(cfg_path)
    with open(args_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(merged, f, sort_keys=False, allow_unicode=True)


def build_data_yaml(
    data_yaml: Path,
    save_dir: Path,
    val_split: float,
    seed: int,
    data_root_override: str,
) -> Tuple[Path, Path]:
    info = load_yaml(data_yaml)
    if data_root_override:
        data_root = Path(data_root_override).resolve()
    else:
        data_root = resolve_data_root(data_yaml)
    train_entry = info.get("train", "images/train")
    val_entry = info.get("val")
    test_entry = info.get("test")

    train_path = resolve_data_entry(str(train_entry), data_root)
    val_path = resolve_data_entry(str(val_entry), data_root) if val_entry else None
    test_path = resolve_data_entry(str(test_entry), data_root) if test_entry else None

    splits_dir = save_dir / "splits"
    train_txt = None
    val_txt = None
    if (val_path is None or not val_path.exists()) and val_split > 0:
        train_txt, val_txt = create_val_split(train_path, val_split, seed, splits_dir)

    yaml_lines = [f"path: {data_root}"]
    if train_txt and val_txt:
        yaml_lines.append(f"train: {train_txt}")
        yaml_lines.append(f"val: {val_txt}")
    else:
        yaml_lines.append(f"train: {train_entry}")
        yaml_lines.append(f"val: {val_entry or train_entry}")
    if test_entry:
        yaml_lines.append(f"test: {test_entry}")
    yaml_lines.append(f"nc: {info.get('nc', 1)}")
    yaml_lines.append(f"names: {info.get('names', ['object'])}")

    save_dir.mkdir(parents=True, exist_ok=True)
    out_yaml = save_dir / "data.yaml"
    with open(out_yaml, "w", encoding="utf-8") as f:
        f.write("\n".join(yaml_lines))
    return out_yaml, data_root


def copy_weights(exp_dir: Path, model_dir: Path) -> None:
    weights_dir = exp_dir / "train" / "weights"
    best = weights_dir / "best.pt"
    last = weights_dir / "last.pt"
    exp_name = exp_dir.name
    if best.exists():
        dst = model_dir / exp_name / "best"
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best, dst / "best.pt")
    if last.exists():
        dst = model_dir / exp_name / "last"
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copy2(last, dst / "last.pt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLO training runner.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configs/yolo11/*.yaml",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg_path = Path(args.config).resolve()
    cfg = load_yaml(cfg_path)
    root = resolve_root()
    yolo_version = cfg.get("yolo_version", "yolo11")
    exp_name = cfg.get("exp_name", "defect")
    project_root = Path(cfg.get("project_root", root)).resolve()
    exp_root = project_root / "experiments" / yolo_version / exp_name
    timestamp = dt.datetime.now().strftime("%y%m%d%H%M")
    exp_dir = exp_root / f"exp_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    data_yaml = Path(cfg.get("data", project_root / "configs" / "data" / "defect.yaml")).resolve()
    data_root_override = str(cfg.get("data_root", ""))
    val_split = float(cfg.get("val_split", 0.0))
    seed = int(cfg.get("seed", 42))
    run_data_yaml, data_root = build_data_yaml(
        data_yaml, exp_dir, val_split, seed, data_root_override
    )

    model_path = str(cfg.get("model", "yolo11n.pt"))
    weights_path = str(cfg.get("weights", ""))
    epochs = int(cfg.get("epochs", 50))
    imgsz = int(cfg.get("imgsz", 640))
    batch = int(cfg.get("batch", 16))
    device = str(cfg.get("device", ""))
    workers = int(cfg.get("workers", 8))
    conf = float(cfg.get("conf", 0.25))
    patience = int(cfg.get("patience", 50))
    lr0 = float(cfg.get("lr0", 0.01))
    lrf = float(cfg.get("lrf", 0.1))
    warmup_epochs = float(cfg.get("warmup_epochs", 3.0))
    save_train_pic = bool(cfg.get("save_train_pic", False))
    save_val_pic = bool(cfg.get("save_val_pic", True))
    save_test_pic = bool(cfg.get("save_test_pic", True))
    metric_conf = float(cfg.get("metric_conf", 0.001))
    eval_batch = int(cfg.get("eval_batch", 1))
    eval_device = str(cfg.get("eval_device", device))
    nms_iou = float(cfg.get("nms_iou", cfg.get("iou", 0.7)))
    max_det = int(cfg.get("max_det", 300))
    match_iou = float(cfg.get("match_iou", 0.5))
    image_conf = float(cfg.get("image_conf", conf))
    metric_pre_nms_topk = int(cfg.get("metric_pre_nms_topk", 5000))
    metric_pre_nms_topk_fallback = int(cfg.get("metric_pre_nms_topk_fallback", 2000))
    metric_nms_time_limit = float(cfg.get("metric_nms_time_limit", 0.05))

    from ultralytics import YOLO

    mode = str(cfg.get("mode", "train_test")).lower()
    if mode == "test" and not weights_path:
        raise ValueError("mode=test requires weights in config.")

    if mode == "test":
        init_weights = weights_path
    elif mode == "finetune_test" and weights_path:
        init_weights = weights_path
    else:
        init_weights = model_path

    init_path = Path(init_weights)
    if not init_path.exists() and init_path.suffix == ".pt":
        cache_path = resolve_pretrained_weight(project_root, init_path.name)
        if cache_path.exists():
            init_weights = str(cache_path)
        else:
            print(f"Pretrained weight missing: {init_path}")
            print(f"Downloading to: {cache_path}")
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            init_weights = str(cache_path)
    model = YOLO(init_weights)
    apply_yolo_enhancements(model, cfg)
    if mode in {"train_test", "finetune_test"}:
        model.train(
            data=str(run_data_yaml),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=device,
            workers=workers,
            patience=patience,
            lr0=lr0,
            lrf=lrf,
            warmup_epochs=warmup_epochs,
            project=str(exp_dir),
            name="train",
            exist_ok=True,
        )
        update_args_yaml(exp_dir, cfg, cfg_path)

    best_weights = exp_dir / "train" / "weights" / "best.pt"
    if mode in {"train_test", "finetune_test"}:
        eval_weights = best_weights if best_weights.exists() else (exp_dir / "train" / "weights" / "last.pt")
        # Release training model before evaluation to reduce GPU memory pressure.
        try:
            del model
        except Exception:
            pass
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        gc.collect()
        model = YOLO(str(eval_weights))
        apply_yolo_enhancements(model, cfg)
    elif mode == "test":
        test_weight = Path(weights_path)
        if test_weight.exists():
            model = YOLO(str(test_weight))
            apply_yolo_enhancements(model, cfg)
        else:
            cached = resolve_pretrained_weight(project_root, test_weight.name)
            model = YOLO(str(cached))
            apply_yolo_enhancements(model, cfg)

    data_info = load_yaml(run_data_yaml)
    train_entry = data_info.get("train", "images/train")
    val_entry = data_info.get("val", "images/val")
    test_entry = data_info.get("test", "")
    train_source = resolve_data_entry(str(train_entry), data_root)
    val_source = resolve_data_entry(str(val_entry), data_root)
    test_source = resolve_data_entry(str(test_entry), data_root) if test_entry else None

    nms_cfg = {
        "metric_pre_nms_topk": metric_pre_nms_topk,
        "metric_pre_nms_topk_fallback": metric_pre_nms_topk_fallback,
        "metric_nms_time_limit": metric_nms_time_limit,
    }
    with patch_nms(nms_cfg):
        if save_train_pic and train_source.exists():
            model.predict(
                source=str(train_source),
                save=True,
                conf=conf,
                iou=nms_iou,
                max_det=max_det,
                project=str(exp_dir),
                name="train_vis",
                exist_ok=True,
            )
        if save_val_pic and val_source.exists():
            model.predict(
                source=str(val_source),
                save=True,
                conf=conf,
                iou=nms_iou,
                max_det=max_det,
                project=str(exp_dir),
                name="val_vis",
                exist_ok=True,
            )
        if save_test_pic and test_source and test_source.exists():
            model.predict(
                source=str(test_source),
                save=True,
                conf=conf,
                iou=nms_iou,
                max_det=max_det,
                project=str(exp_dir),
                name="test_vis",
                exist_ok=True,
            )

    metrics_dir = exp_dir / "metrics"
    sweep_cfg = cfg.get("threshold_sweep", {})
    fix_var = str(sweep_cfg.get("fix", "iou")).lower()
    fix_list = sweep_cfg.get("fix_list", [0.1, 0.2, 0.3, 0.4])
    curve_range = sweep_cfg.get("curve", [0.01, 0.01, 1.00])
    table_range = sweep_cfg.get("table", [0.00, 0.05, 1.00])

    if not isinstance(fix_list, (list, tuple)):
        fix_list = [fix_list]

    curve_vals = np.round(
        np.arange(float(curve_range[0]), float(curve_range[2]) + 1e-9, float(curve_range[1])), 2
    )
    table_vals = np.round(
        np.arange(float(table_range[0]), float(table_range[2]) + 1e-9, float(table_range[1])), 2
    )

    def eval_split(tag: str, source: Path) -> None:
        metrics_dir.mkdir(parents=True, exist_ok=True)
        try:
            import json

            sweep_meta = {
                "fix": fix_var,
                "fix_list": fix_list,
                "curve": curve_range,
                "table": table_range,
            }
            (metrics_dir / f"{tag}_threshold_sweep.json").write_text(
                json.dumps(sweep_meta, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception:
            pass
        with patch_nms(nms_cfg):
            try:
                labels, scores = compute_image_scores(
                    model, source, metric_conf, eval_batch, eval_device, nms_iou, max_det
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
                labels, scores = compute_image_scores(model, source, metric_conf, 1, "cpu", nms_iou, max_det)
            if labels.size == 0:
                return
            recall, precision, fpr = compute_threshold_metrics(labels, scores, curve_vals)
            auroc = compute_auc(fpr, recall)
            ap = compute_ap(recall, precision)
            save_metric_plots(metrics_dir, tag, curve_vals, recall, precision, fpr)
            with open(metrics_dir / f"{tag}_summary.txt", "w", encoding="utf-8") as f:
                f.write(f"image_auroc: {auroc:.6f}\n")
                f.write(f"image_ap: {ap:.6f}\n")

            multi_rows: List[dict] = []
            curve_recall = []
            curve_fpr = []
            roc_curves: List[Tuple[str, np.ndarray, np.ndarray]] = []
            pr_curves: List[Tuple[str, np.ndarray, np.ndarray]] = []

            if fix_var == "iou":
                var_vals = curve_vals
                for iou_thr in fix_list:
                    try:
                        labels_i, scores_i = compute_image_scores_iou(
                            model, source, metric_conf, float(iou_thr), eval_batch, eval_device, nms_iou, max_det
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
                        labels_i, scores_i = compute_image_scores_iou(
                            model, source, metric_conf, float(iou_thr), 1, "cpu", nms_iou, max_det
                        )
                    if labels_i.size == 0:
                        continue
                    rec_i, prec_i, fpr_i = compute_threshold_metrics(labels_i, scores_i, var_vals)
                    curve_recall.append((f"iou={float(iou_thr):.2f}", rec_i))
                    curve_fpr.append((f"iou={float(iou_thr):.2f}", fpr_i))
                    roc_curves.append((f"iou={float(iou_thr):.2f}", fpr_i, rec_i))
                    pr_curves.append((f"iou={float(iou_thr):.2f}", rec_i, prec_i))
                    for thr, rec_v, fpr_v in zip(var_vals, rec_i, fpr_i):
                        multi_rows.append(
                            {
                                "fixed_var": "iou",
                                "fixed_value": float(iou_thr),
                                "threshold": float(thr),
                                "recall": float(rec_v),
                                "fpr": float(fpr_v),
                            }
                        )

                save_multi_curve(
                    metrics_dir,
                    f"{tag}_recall_vs_conf_multi.png",
                    var_vals,
                    curve_recall,
                    xlabel="conf threshold",
                    ylabel="recall",
                    title=f"{tag} Recall vs conf (fixed IoU)",
                )
                save_multi_curve(
                    metrics_dir,
                    f"{tag}_fpr_vs_conf_multi.png",
                    var_vals,
                    curve_fpr,
                    xlabel="conf threshold",
                    ylabel="fpr",
                    title=f"{tag} FPR vs conf (fixed IoU)",
                )
                if roc_curves:
                    save_multi_xy_curves(
                        metrics_dir,
                        f"{tag}_roc.png",
                        roc_curves,
                        xlabel="FPR",
                        ylabel="TPR/Recall",
                        title=f"{tag} ROC (fixed IoU)",
                    )
                if pr_curves:
                    save_multi_xy_curves(
                        metrics_dir,
                        f"{tag}_pr.png",
                        pr_curves,
                        xlabel="Recall",
                        ylabel="Precision",
                        title=f"{tag} PR (fixed IoU)",
                    )

            elif fix_var == "conf":
                iou_vals = curve_vals
                scores_by_iou = []
                labels_ref = None
                for iou_thr in iou_vals:
                    try:
                        labels_i, scores_i = compute_image_scores_iou(
                            model, source, metric_conf, float(iou_thr), eval_batch, eval_device, nms_iou, max_det
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
                        labels_i, scores_i = compute_image_scores_iou(
                            model, source, metric_conf, float(iou_thr), 1, "cpu", nms_iou, max_det
                        )
                    if labels_ref is None:
                        labels_ref = labels_i
                    scores_by_iou.append(scores_i)

                if labels_ref is not None and scores_by_iou:
                    for conf_thr in fix_list:
                        thr_arr = np.array([float(conf_thr)], dtype=np.float32)
                        rec_curve = []
                        fpr_curve = []
                        for scores_i in scores_by_iou:
                            rec_i, _, fpr_i = compute_threshold_metrics(labels_ref, scores_i, thr_arr)
                            rec_curve.append(rec_i[0])
                            fpr_curve.append(fpr_i[0])
                        curve_recall.append((f"conf={float(conf_thr):.2f}", np.array(rec_curve)))
                        curve_fpr.append((f"conf={float(conf_thr):.2f}", np.array(fpr_curve)))
                        for thr, rec_v, fpr_v in zip(iou_vals, rec_curve, fpr_curve):
                            multi_rows.append(
                                {
                                    "fixed_var": "conf",
                                    "fixed_value": float(conf_thr),
                                    "threshold": float(thr),
                                    "recall": float(rec_v),
                                    "fpr": float(fpr_v),
                                }
                            )

                    save_multi_curve(
                        metrics_dir,
                        f"{tag}_recall_vs_iou_multi.png",
                        iou_vals,
                        curve_recall,
                        xlabel="iou_match",
                        ylabel="recall",
                        title=f"{tag} Recall vs IoU (fixed conf)",
                    )
                    save_multi_curve(
                        metrics_dir,
                        f"{tag}_fpr_vs_iou_multi.png",
                        iou_vals,
                        curve_fpr,
                        xlabel="iou_match",
                        ylabel="fpr",
                        title=f"{tag} FPR vs IoU (fixed conf)",
                    )

            if multi_rows:
                save_threshold_table_multi(metrics_dir, tag, multi_rows)
            else:
                save_threshold_table(metrics_dir, tag, labels, scores, table_vals)

            # Image-level TP/FP/TN/FN summary and optional visualization marking.
            try:
                items = compute_image_level_results(
                    model,
                    source,
                    conf_threshold=image_conf,
                    iou_match=match_iou,
                    metric_conf=metric_conf,
                    batch=eval_batch,
                    device=eval_device,
                    nms_iou=nms_iou,
                    max_det=max_det,
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
                    source,
                    conf_threshold=image_conf,
                    iou_match=match_iou,
                    metric_conf=metric_conf,
                    batch=1,
                    device="cpu",
                    nms_iou=nms_iou,
                    max_det=max_det,
                )

            meta = {
                "conf_threshold": float(image_conf),
                "iou_match": float(match_iou),
                "nms_iou": float(nms_iou),
                "max_det": int(max_det),
                "metric_conf": float(metric_conf),
                "eval_batch": int(eval_batch),
                "eval_device": str(eval_device),
            }
            save_image_level_report(metrics_dir, tag, items, meta)

            vis_dir = exp_dir / f"{tag}_vis"
            annotate_fp_fn_images(vis_dir, items)

    if val_source.exists():
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        gc.collect()
        eval_split("val", val_source)
    if test_source and test_source.exists():
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        gc.collect()
        eval_split("test", test_source)

    if mode in {"train_test", "finetune_test"}:
        model_dir = project_root / "models" / exp_name
        copy_weights(exp_dir, model_dir)


if __name__ == "__main__":
    main()
