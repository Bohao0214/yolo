#!/usr/bin/env python3
"""
公开数据集评估脚本（支持 USER_EDIT_CONFIG，只改配置即可运行）。

1) 推荐方式：只改 USER_EDIT_CONFIG，然后直接运行
   python /home/ubuntu/hpproject/yolo/analyze/code/ch4tool/run_public_single_eval.py

2) 命令行单模型覆盖（可选）
   python .../run_public_single_eval.py --weight /abs/path/best.pt --dataset-root /abs/dataset
"""

from __future__ import annotations

import argparse
import copy
import csv
import datetime as dt
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml
from ultralytics import YOLO

REPO_ROOT = Path("/home/ubuntu/hpproject/yolo")
DEFAULT_DATASET_ROOT = REPO_ROOT / "dataset" / "yolo"
PUBLIC_DATASETS = ["DeepPCB", "GC10-DET", "KolektorSDD", "NEU-DET"]
PUBLIC_METHODS = ["YOLO11m", "本文方法"]

# 你可以只修改这里，不改其它代码
USER_EDIT_CONFIG: Dict = {
    "models": [
        # {"name": "deeppcb_yolo11m", "path": "/abs/path/to/best.pt", "method": "YOLO11m"},
        # {"name": "deeppcb_our", "path": "/abs/path/to/best.pt", "method": "本文方法"},
    ],
    "data_yaml": "",  # 全局默认 data.yaml；留空则自动推断
    "dataset_root": "",  # 全局默认 dataset_root；留空则自动推断
    "split": "auto",  # auto/test/val/train
    "infer_params": {
        "imgsz": None,  # None=自动读 args.yaml
        "conf": None,
        "iou": None,
        "max_det": None,
        "device": "",  # 空=自动读 args.yaml
        "batch": None,
    },
    "out_root": "/home/ubuntu/hpproject/yolo/analyze/result",
    "report_prefix": "result_",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Public dataset evaluation runner.")
    p.add_argument("--weight", type=Path, default=None, help="Optional single model override")
    p.add_argument("--dataset-root", type=Path, default=None)
    p.add_argument("--data-yaml", type=Path, default=None)
    p.add_argument("--split", type=str, default="")
    p.add_argument("--method-label", type=str, default="")
    p.add_argument("--device", type=str, default="")
    p.add_argument("--config-json", type=str, default="", help="JSON override for USER_EDIT_CONFIG")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def deep_merge(a: Dict, b: Dict) -> Dict:
    out = copy.deepcopy(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def make_result_dir(root: Path, prefix: str) -> Path:
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M")
    out = root / f"{prefix}{ts}"
    if not out.exists():
        out.mkdir(parents=True, exist_ok=False)
        return out
    idx = 1
    while True:
        cand = root / f"{prefix}{ts}_{idx:02d}"
        if not cand.exists():
            cand.mkdir(parents=True, exist_ok=False)
            return cand
        idx += 1


def read_yaml(path: Path) -> Dict:
    obj = yaml.safe_load(path.read_text(encoding="utf-8"))
    return obj if isinstance(obj, dict) else {}


def find_args_yaml(weight: Path) -> Optional[Path]:
    p = weight.resolve()
    cands = []
    if p.parent.name == "weights":
        cands.append(p.parent.parent / "args.yaml")
    for level in range(1, 7):
        cands.append(p.parents[level - 1] / "args.yaml")
    for c in cands:
        if c.exists():
            return c
    return None


def normalize_name(text: str) -> str:
    return text.lower().replace("-", "").replace("_", "")


def infer_dataset_label_from_text(text: str) -> str:
    s = normalize_name(text)
    mapping = [
        ("DeepPCB", ["deeppcb", "deeppcbstandard"]),
        ("GC10-DET", ["gc10det", "gc10"]),
        ("KolektorSDD", ["kolektor", "kolektorsdd"]),
        ("NEU-DET", ["neudet", "neu"]),
    ]
    for label, keys in mapping:
        if any(k in s for k in keys):
            return label
    return "Unknown"


def infer_dataset_root_from_label(label: str) -> Optional[Path]:
    if label == "DeepPCB":
        cands = ["DeepPCB_standard", "deeppcb_standard", "DeepPCB", "deeppcb"]
    elif label == "GC10-DET":
        cands = ["gc10det_622_halves", "GC10DET_622_halves", "GC10-DET", "gc10det"]
    elif label == "KolektorSDD":
        cands = ["kolektorsdd_622_halves", "KolektorSDD", "kolektorsdd"]
    elif label == "NEU-DET":
        cands = ["neudet_622", "NEUDET_622", "NEU-DET", "neudet"]
    else:
        cands = []
    for name in cands:
        p = DEFAULT_DATASET_ROOT / name
        if p.exists():
            return p
    return None


def resolve_data_yaml(dataset_root: Path) -> Optional[Path]:
    for n in ["data.yaml", "dataset.yaml", "data.yml", "dataset.yml"]:
        p = dataset_root / n
        if p.exists():
            return p
    return None


def choose_split(data_yaml: Path, split_arg: str) -> str:
    if split_arg and split_arg != "auto":
        return split_arg
    data = read_yaml(data_yaml)
    if data.get("test") not in (None, "", False):
        return "test"
    if data.get("val") not in (None, "", False):
        return "val"
    return "val"


def pick_method_label(weight: Path, explicit: str) -> str:
    if explicit.strip():
        return explicit.strip()
    s = normalize_name(str(weight))
    if "baseline" in s or "yolo11m" in s:
        return "YOLO11m"
    return "本文方法"


def float_or(v, d: float) -> float:
    try:
        return float(v)
    except Exception:
        return float(d)


def int_or(v, d: int) -> int:
    try:
        return int(v)
    except Exception:
        return int(d)


def choose_eval_params(
    args_yaml_obj: Dict,
    global_infer: Dict,
    model_infer: Dict,
    cli_device: str,
) -> Dict:
    conf_auto = float_or(args_yaml_obj.get("metric_conf", args_yaml_obj.get("conf", 0.001)), 0.001)
    iou_auto = float_or(args_yaml_obj.get("nms_iou", args_yaml_obj.get("iou", 0.7)), 0.7)
    imgsz_auto = int_or(args_yaml_obj.get("imgsz", 640), 640)
    max_det_auto = int_or(args_yaml_obj.get("max_det", 300), 300)
    batch_auto = int_or(args_yaml_obj.get("eval_batch", args_yaml_obj.get("batch", 4)), 4)
    device_auto = str(args_yaml_obj.get("eval_device", args_yaml_obj.get("device", "0")) or "0")

    conf = model_infer.get("conf", global_infer.get("conf", None))
    iou = model_infer.get("iou", global_infer.get("iou", None))
    imgsz = model_infer.get("imgsz", global_infer.get("imgsz", None))
    max_det = model_infer.get("max_det", global_infer.get("max_det", None))
    batch = model_infer.get("batch", global_infer.get("batch", None))

    if conf is None:
        conf = conf_auto
    if iou is None:
        iou = iou_auto
    if imgsz is None:
        imgsz = imgsz_auto
    if max_det is None:
        max_det = max_det_auto
    if batch is None:
        batch = batch_auto

    if cli_device:
        device = cli_device
        device_src = "cli"
    else:
        gd = str(global_infer.get("device", "") or "")
        md = str(model_infer.get("device", "") or "")
        if md:
            device = md
            device_src = "model_infer"
        elif gd:
            device = gd
            device_src = "global_infer"
        else:
            device = device_auto
            device_src = "args_yaml/default"

    return {
        "conf": float_or(conf, conf_auto),
        "iou": float_or(iou, iou_auto),
        "imgsz": int_or(imgsz, imgsz_auto),
        "max_det": int_or(max_det, max_det_auto),
        "batch": int_or(batch, batch_auto),
        "device": device,
        "device_source": device_src,
    }


def extract_metrics(val_res) -> Dict[str, float]:
    rd = getattr(val_res, "results_dict", {}) or {}
    box = getattr(val_res, "box", None)
    p = rd.get("metrics/precision(B)")
    r = rd.get("metrics/recall(B)")
    map50 = rd.get("metrics/mAP50(B)")
    map5095 = rd.get("metrics/mAP50-95(B)")
    if p is None and box is not None:
        p = getattr(box, "mp", 0.0)
    if r is None and box is not None:
        r = getattr(box, "mr", 0.0)
    if map50 is None and box is not None:
        map50 = getattr(box, "map50", 0.0)
    if map5095 is None and box is not None:
        map5095 = getattr(box, "map", 0.0)
    return {
        "map50": float_or(map50, 0.0),
        "map50_95": float_or(map5095, 0.0),
        "obj_precision": float_or(p, 0.0),
        "obj_recall": float_or(r, 0.0),
    }


def write_csv(path: Path, rows: List[Dict], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def pct(v: float) -> str:
    return f"{v * 100.0:.2f}"


def build_public_rows_filled(metrics_rows: List[Dict]) -> List[Dict]:
    table = {
        (d, mtd): {"dataset": d, "method": mtd, "mAP@0.5": "--", "mAP@0.5:0.95": "--", "P_obj/%": "--", "R_obj/%": "--"}
        for d in PUBLIC_DATASETS
        for mtd in PUBLIC_METHODS
    }
    best_key_score = {}
    for r in metrics_rows:
        key = (r["dataset"], r["method"])
        if key not in table:
            continue
        score = float_or(r.get("mAP@0.5:0.95", 0.0), 0.0)
        prev = best_key_score.get(key, -1.0)
        if score >= prev:
            best_key_score[key] = score
            table[key] = {
                "dataset": r["dataset"],
                "method": r["method"],
                "mAP@0.5": pct(float_or(r["mAP@0.5"], 0.0)),
                "mAP@0.5:0.95": pct(float_or(r["mAP@0.5:0.95"], 0.0)),
                "P_obj/%": pct(float_or(r["P_obj"], 0.0)),
                "R_obj/%": pct(float_or(r["R_obj"], 0.0)),
            }
    return [table[(d, mtd)] for d in PUBLIC_DATASETS for mtd in PUBLIC_METHODS]


def write_markdown_table(path: Path, rows: List[Dict]) -> None:
    lines = [
        "# 公开数据集对比实验结果模板（已填当前可得项）",
        "",
        "| 数据集 | 方法 | mAP@0.5 | mAP@0.5:0.95 | P_obj/% | R_obj/% |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['dataset']} | {r['method']} | {r['mAP@0.5']} | {r['mAP@0.5:0.95']} | {r['P_obj/%']} | {r['R_obj/%']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_latex_table(path: Path, rows: List[Dict]) -> None:
    grouped: Dict[str, List[Dict]] = {}
    for r in rows:
        grouped.setdefault(r["dataset"], []).append(r)
    lines = [
        "\\begin{table}[htbp]",
        "    \\centering",
        "    \\caption{公开数据集对比实验结果模板（当前已填部分）}",
        "    \\label{tab:ch4_public_compare}",
        "    \\begin{tabular}{llcccc}",
        "        \\toprule",
        "        数据集 & 方法 & $mAP@0.5$ & $mAP@0.5:0.95$ & $P_{\\mathrm{obj}}$/\\% & $R_{\\mathrm{obj}}$/\\%  \\\\",
        "        \\midrule",
    ]
    for i, d in enumerate(PUBLIC_DATASETS):
        for rr in grouped.get(d, []):
            lines.append(
                f"        {rr['dataset']} & {rr['method']} & {rr['mAP@0.5']} & {rr['mAP@0.5:0.95']} & {rr['P_obj/%']} & {rr['R_obj/%']} \\\\"
            )
        if i != len(PUBLIC_DATASETS) - 1:
            lines.append("        \\midrule")
    lines.extend(["        \\bottomrule", "    \\end{tabular}", "\\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def resolve_dataset_root(
    model_cfg: Dict,
    global_cfg: Dict,
    cli_dataset_root: Optional[Path],
    args_yaml_obj: Dict,
    weight: Path,
) -> Path:
    if cli_dataset_root:
        return cli_dataset_root.resolve()
    if model_cfg.get("dataset_root"):
        return Path(model_cfg["dataset_root"]).expanduser().resolve()
    if global_cfg.get("dataset_root"):
        return Path(global_cfg["dataset_root"]).expanduser().resolve()

    label = infer_dataset_label_from_text(str(weight))
    inferred = infer_dataset_root_from_label(label)
    if inferred and inferred.exists():
        return inferred.resolve()

    data_ref = args_yaml_obj.get("data")
    if isinstance(data_ref, str) and data_ref.strip():
        p = Path(data_ref).expanduser()
        if p.exists():
            if p.is_file():
                data_obj = read_yaml(p)
                base = Path(data_obj.get("path")).expanduser() if data_obj.get("path") else None
                if base and base.exists():
                    return base.resolve()
                return p.parent.resolve()
            return p.resolve()
    raise RuntimeError("无法自动定位 dataset_root，请在 USER_EDIT_CONFIG 或命令行中显式设置")


def resolve_data_yaml_for_model(
    model_cfg: Dict,
    global_cfg: Dict,
    cli_data_yaml: Optional[Path],
    dataset_root: Path,
) -> Path:
    if cli_data_yaml:
        return cli_data_yaml.resolve()
    if model_cfg.get("data_yaml"):
        return Path(model_cfg["data_yaml"]).expanduser().resolve()
    if global_cfg.get("data_yaml"):
        return Path(global_cfg["data_yaml"]).expanduser().resolve()
    p = resolve_data_yaml(dataset_root)
    if p:
        return p.resolve()
    raise RuntimeError(f"未找到 data.yaml/dataset.yaml: {dataset_root}")


def resolve_split_for_model(model_cfg: Dict, global_cfg: Dict, cli_split: str, data_yaml: Path) -> str:
    if cli_split:
        return choose_split(data_yaml, cli_split)
    if model_cfg.get("split"):
        return choose_split(data_yaml, str(model_cfg["split"]))
    return choose_split(data_yaml, str(global_cfg.get("split", "auto")))


def build_runtime_config(args: argparse.Namespace) -> Dict:
    cfg = copy.deepcopy(USER_EDIT_CONFIG)
    if args.config_json:
        cfg = deep_merge(cfg, json.loads(args.config_json))

    # CLI 单模型覆盖：不依赖 USER_EDIT_CONFIG["models"]
    if args.weight:
        single = {
            "name": Path(args.weight).stem,
            "path": str(args.weight),
            "method": args.method_label or "",
        }
        cfg["models"] = [single]

    return cfg


def main() -> None:
    args = parse_args()
    cfg = build_runtime_config(args)
    models = cfg.get("models", [])
    if not isinstance(models, list) or len(models) == 0:
        raise RuntimeError("models 为空。请在 USER_EDIT_CONFIG['models'] 填至少一个模型或使用 --weight。")

    out_root = Path(cfg.get("out_root", "/home/ubuntu/hpproject/yolo/analyze/result")).expanduser().resolve()
    prefix = str(cfg.get("report_prefix", "result_"))
    out_dir = make_result_dir(out_root, prefix)

    logs: List[str] = []
    success_rows: List[Dict] = []
    failures: List[Dict] = []
    used_items: List[Dict] = []

    for mcfg in models:
        if not isinstance(mcfg, dict):
            failures.append({"model": str(mcfg), "error": "model config item is not dict"})
            continue
        model_name = str(mcfg.get("name", "unnamed"))
        weight_path_raw = str(mcfg.get("path", "")).strip()
        if not weight_path_raw:
            failures.append({"model": model_name, "error": "missing model path"})
            continue
        weight = Path(weight_path_raw).expanduser().resolve()
        if not weight.exists():
            failures.append({"model": model_name, "weight": str(weight), "error": "weight not found"})
            continue

        try:
            args_yaml_path = find_args_yaml(weight)
            args_yaml_obj = read_yaml(args_yaml_path) if args_yaml_path and args_yaml_path.exists() else {}
            dataset_root = resolve_dataset_root(mcfg, cfg, args.dataset_root, args_yaml_obj, weight)
            if not dataset_root.exists():
                raise RuntimeError(f"dataset_root not found: {dataset_root}")
            data_yaml = resolve_data_yaml_for_model(mcfg, cfg, args.data_yaml, dataset_root)
            if not data_yaml.exists():
                raise RuntimeError(f"data_yaml not found: {data_yaml}")
            split = resolve_split_for_model(mcfg, cfg, args.split, data_yaml)
            method_label = pick_method_label(weight, str(mcfg.get("method", "") or args.method_label))
            dataset_label = str(mcfg.get("dataset_label", "")).strip() or infer_dataset_label_from_text(str(dataset_root))

            eval_params = choose_eval_params(
                args_yaml_obj=args_yaml_obj,
                global_infer=cfg.get("infer_params", {}) if isinstance(cfg.get("infer_params"), dict) else {},
                model_infer=mcfg.get("infer_params", {}) if isinstance(mcfg.get("infer_params"), dict) else {},
                cli_device=args.device,
            )

            used = {
                "name": model_name,
                "weight": str(weight),
                "args_yaml": str(args_yaml_path) if args_yaml_path else None,
                "dataset_root": str(dataset_root),
                "dataset_label": dataset_label,
                "data_yaml": str(data_yaml),
                "split": split,
                "method": method_label,
                "eval_params": eval_params,
            }
            used_items.append(used)
            logs.append(f"[run] {json.dumps(used, ensure_ascii=False)}")

            if args.dry_run:
                continue

            model = YOLO(str(weight))
            val_res = model.val(
                data=str(data_yaml),
                split=str(split),
                conf=float(eval_params["conf"]),
                iou=float(eval_params["iou"]),
                imgsz=int(eval_params["imgsz"]),
                max_det=int(eval_params["max_det"]),
                batch=int(eval_params["batch"]),
                device=str(eval_params["device"]),
                plots=False,
                save_json=False,
                verbose=False,
            )
            metric = extract_metrics(val_res)
            row = {
                "name": model_name,
                "dataset": dataset_label,
                "method": method_label,
                "weight": str(weight),
                "data_yaml": str(data_yaml),
                "split": split,
                "mAP@0.5": f"{metric['map50']:.6f}",
                "mAP@0.5:0.95": f"{metric['map50_95']:.6f}",
                "P_obj": f"{metric['obj_precision']:.6f}",
                "R_obj": f"{metric['obj_recall']:.6f}",
            }
            success_rows.append(row)
        except Exception as ex:
            failures.append({"model": model_name, "weight": str(weight), "error": str(ex)})

    # 输出文件
    (out_dir / "run.log").write_text("\n".join(logs) + ("\n" if logs else ""), encoding="utf-8")
    (out_dir / "used_params.json").write_text(
        json.dumps({"config": cfg, "used": used_items}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if success_rows:
        write_csv(out_dir / "metrics_all.csv", success_rows, list(success_rows[0].keys()))
        if len(success_rows) == 1:
            write_csv(out_dir / "metrics_single.csv", success_rows, list(success_rows[0].keys()))

    table_rows = build_public_rows_filled(success_rows)
    write_csv(
        out_dir / "public_compare_template.csv",
        table_rows,
        ["dataset", "method", "mAP@0.5", "mAP@0.5:0.95", "P_obj/%", "R_obj/%"],
    )
    write_markdown_table(out_dir / "public_compare_template.md", table_rows)
    write_latex_table(out_dir / "public_compare_table.tex", table_rows)

    if failures:
        (out_dir / "failures.json").write_text(json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[done] result_dir: {out_dir}")
    if success_rows:
        print(f"[done] metrics_all: {out_dir / 'metrics_all.csv'}")
    else:
        print("[warn] no successful model run.")
    print(f"[done] table_csv: {out_dir / 'public_compare_template.csv'}")
    print(f"[done] table_md : {out_dir / 'public_compare_template.md'}")
    print(f"[done] table_tex: {out_dir / 'public_compare_table.tex'}")
    if failures:
        print(f"[warn] failures: {out_dir / 'failures.json'}")


if __name__ == "__main__":
    main()
