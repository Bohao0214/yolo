#!/usr/bin/env python3
"""
公开数据集评估脚本（只改 USER_EDIT_CONFIG 即可）。

目标：
1) 输入权重后优先从同实验目录的 args.yaml 自动解析 data.yaml 与 dataset_root。
2) 不再区分 method，统一用 name 作为模型标识。
3) 输出两类指标：
   - 多类别检测：mAP@0.5, mAP@0.5:0.95, avg_cls_precision, avg_cls_recall
   - 图像级二分类（有缺陷/无缺陷）：bin_img_AP, bin_img_precision, bin_img_recall

运行：
python /home/ubuntu/hpproject/yolo/analyze/code/ch4tool/run_public_single_eval.py
"""

from __future__ import annotations

import argparse
import copy
import csv
import datetime as dt
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import yaml
from ultralytics import YOLO

REPO_ROOT = Path("/home/ubuntu/hpproject/yolo")
DEFAULT_DATASET_ROOT = REPO_ROOT / "dataset" / "yolo"
PUBLIC_DATASETS = ["DeepPCB", "GC10-DET", "KolektorSDD", "NEU-DET"]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# 你可以只修改这里，不改其它代码
USER_EDIT_CONFIG: Dict = {
    "models": [
        {"name": "dpcbbl1", "path": "experiments/smoke241/DeepPCB_standard/bs4_e150/exp_2604090441/train/weights/best.pt"},
        {"name": "dpcbbl2", "path": "experiments/smoke241/DeepPCB_standard/bs10_e150/exp_2604090626/train/weights/best.pt"},
        {"name": "dpcbbl3", "path": "experiments/a4b7d6/DeepPCB_standard/bs2_e150/exp_2604101709/train/weights/best.pt"},
  
        {"name": "dpcbbl4", "path": "experiments/smoke241/DeepPCB_standard/bs6_e150/exp_2604090541/train/weights/best.pt"},

        {"name": "gc10bl1", "path": "experiments/smoke241/gc10det_622_halves/bs4_e150/exp_2604082313/train/weights/best.pt"},
        {"name": "gc10bl2", "path": "experiments/smoke241/gc10det_622_halves/bs6_e150/exp_2604090132/train/weights/best.pt"},
        {"name": "gc10bl3", "path": "/home/ubuntu/hpproject/yolo/experiments/smoke241/gc10det_622_halves/bs10_e150/exp_2604090318/train/weights/best.pt"},  
        {"name": "gc10bl4", "path": "experiments/a4b7d6/gc10det_622_halves/bs2_e150/exp_2604101213/train/weights/best.pt"},  
        {"name": "gc10bl5", "path": "/home/ubuntu/hpproject/yolo/experiments/a4b7d6/gc10det_622_halves/bs4_e150/exp_2604091054/train/weights/best.pt"},  
        {"name": "gc10bl6", "path": "experiments/a4b7d6/gc10det_622_halves/bs6_e150/exp_2604091357/train/weights/best.pt"},  

        {"name": "kolbl1", "path": "experiments/smoke241/kolektorsdd_622_halves/bs4_e150/exp_2604090702/train/weights/best.pt"},  
        {"name": "gc10bl2", "path": "experiments/smoke241/kolektorsdd_622_halves/bs6_e150/exp_2604090727/train/weights/best.pt"},  
        {"name": "gc10bl3", "path": "experiments/smoke241/kolektorsdd_622_halves/bs10_e150/exp_2604090748/train/weights/best.pt"},  
        {"name": "gc10bl4", "path": "experiments/a4b7d6/kolektorsdd_622_halves/bs2_e150/exp_2604101918/train/weights/best.pt"},  


        {"name": "gc10bl1", "path": "experiments/smoke241/neudet_622/bs4_e150/exp_2604090804/train/weights/best.pt"},  
        {"name": "gc10bl2", "path": "experiments/smoke241/neudet_622/bs6_e150/exp_2604090900/train/weights/best.pt"},  
        {"name": "gc10bl3", "path": "experiments/smoke241/neudet_622/bs10_e150/exp_2604090943/train/weights/best.pt"},  
        {"name": "gc10bl4", "path": "experiments/a4b7d6/neudet_622/bs2_e150/exp_2604102007/train/weights/best.pt"},  

        # {"name": "deeppcb_a4b7d6", "path": "/abs/path/to/best.pt"},
        # {"name": "gc10_baseline", "path": "/abs/path/to/best.pt"},
    ],
    "data_yaml": "",  # 全局默认 data.yaml；留空=自动解析（优先 args.yaml 的 data 字段）
    "dataset_root": "",  # 全局默认 dataset_root；留空=自动解析
    "split": "test",  # auto/test/val/train
    "infer_params": {
        # 统一口径对齐 p2604_multi_model_eval.py
        "imgsz": 640,
        "conf": 0.25,
        "iou": 0.7,  # NMS IoU
        "max_det": 100,
        "device": "0",
        "batch": 4,
        "tp_iou": 0.3,  # 预留（本脚本当前不做 GT-Pred 匹配统计）
        "match_metric": "iou",  # 预留: iou | ios
        "score_floor": 0.01,  # 二分类图像级 AP 统计时的最低打分阈值
        "raw_max_det": 3000,  # 二分类图像级 AP 统计时的最大候选框数
    },
    "out_root": "/home/ubuntu/hpproject/yolo/analyze/result",
    "report_prefix": "result_",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Public dataset multi-model evaluation by name.")
    p.add_argument("--weight", type=Path, default=None, help="Optional single model override")
    p.add_argument("--name", type=str, default="", help="Name for --weight")
    p.add_argument("--dataset-root", type=Path, default=None)
    p.add_argument("--data-yaml", type=Path, default=None)
    p.add_argument("--split", type=str, default="")
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


def find_args_yaml(weight: Path) -> Optional[Path]:
    p = weight.resolve()
    cands = []
    if p.parent.name == "weights":
        cands.append(p.parent.parent / "args.yaml")
    for level in range(1, 8):
        cands.append(p.parents[level - 1] / "args.yaml")
    for c in cands:
        if c.exists():
            return c
    return None


def resolve_data_yaml(dataset_root: Path) -> Optional[Path]:
    for n in ["data.yaml", "dataset.yaml", "data.yml", "dataset.yml"]:
        p = dataset_root / n
        if p.exists():
            return p
    return None


def _resolve_ref_path(raw: str, base: Path) -> Path:
    p = Path(raw).expanduser()
    if p.is_absolute():
        return p.resolve()
    return (base / p).resolve()


def infer_from_args_data(args_yaml_obj: Dict, args_yaml_path: Optional[Path]) -> Tuple[Optional[Path], Optional[Path]]:
    """
    从 args.yaml 的 data 字段推断 (data_yaml, dataset_root)。
    支持：
    - data 指向 data.yaml
    - data 指向 dataset_root
    """
    if not isinstance(args_yaml_obj, dict):
        return None, None

    base = args_yaml_path.parent if args_yaml_path else REPO_ROOT
    raw_data = args_yaml_obj.get("data")
    if not isinstance(raw_data, str) or not raw_data.strip():
        return None, None

    data_ref = _resolve_ref_path(raw_data, base)
    data_yaml: Optional[Path] = None
    dataset_root: Optional[Path] = None

    if data_ref.exists():
        if data_ref.is_file():
            data_yaml = data_ref
            data_obj = read_yaml(data_yaml)
            path_key = data_obj.get("path")
            if isinstance(path_key, str) and path_key.strip():
                p = _resolve_ref_path(path_key, data_yaml.parent)
                if p.exists():
                    dataset_root = p
            if dataset_root is None:
                dataset_root = data_yaml.parent
        else:
            dataset_root = data_ref
            candidate = resolve_data_yaml(dataset_root)
            if candidate:
                data_yaml = candidate

    return data_yaml, dataset_root


def choose_split(data_yaml: Path, split_arg: str) -> str:
    if split_arg and split_arg != "auto":
        return split_arg
    d = read_yaml(data_yaml)
    if d.get("test") not in (None, "", False):
        return "test"
    if d.get("val") not in (None, "", False):
        return "val"
    if d.get("train") not in (None, "", False):
        return "train"
    return "val"


def parse_split_spec(split_spec: str, data_yaml: Path) -> List[str]:
    """
    支持:
    - auto
    - test / val / train
    - val+test / test+val
    - val,test
    """
    spec = (split_spec or "auto").strip()
    if not spec or spec == "auto":
        return [choose_split(data_yaml, "auto")]

    raw_parts = spec.replace(",", "+").split("+")
    parts = [p.strip() for p in raw_parts if p.strip()]
    if not parts:
        return [choose_split(data_yaml, "auto")]

    # 去重且保持顺序
    seen = set()
    out: List[str] = []
    for p in parts:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def choose_eval_params(args_yaml_obj: Dict, global_infer: Dict, model_infer: Dict, cli_device: str) -> Dict:
    conf_auto = float_or(args_yaml_obj.get("metric_conf", args_yaml_obj.get("conf", 0.001)), 0.001)
    iou_auto = float_or(args_yaml_obj.get("nms_iou", args_yaml_obj.get("iou", 0.7)), 0.7)
    imgsz_auto = int_or(args_yaml_obj.get("imgsz", 640), 640)
    max_det_auto = int_or(args_yaml_obj.get("max_det", 300), 300)
    batch_auto = int_or(args_yaml_obj.get("eval_batch", args_yaml_obj.get("batch", 4)), 4)
    device_auto = str(args_yaml_obj.get("eval_device", args_yaml_obj.get("device", "0")) or "0")
    tp_iou_auto = float_or(args_yaml_obj.get("tp_iou", 0.3), 0.3)
    match_metric_auto = str(args_yaml_obj.get("match_metric", "iou") or "iou")
    score_floor_auto = float_or(args_yaml_obj.get("score_floor", 0.01), 0.01)
    raw_max_det_auto = int_or(args_yaml_obj.get("raw_max_det", 3000), 3000)

    conf = model_infer.get("conf", global_infer.get("conf", None))
    iou = model_infer.get("iou", global_infer.get("iou", None))
    imgsz = model_infer.get("imgsz", global_infer.get("imgsz", None))
    max_det = model_infer.get("max_det", global_infer.get("max_det", None))
    batch = model_infer.get("batch", global_infer.get("batch", None))
    tp_iou = model_infer.get("tp_iou", global_infer.get("tp_iou", None))
    match_metric = model_infer.get("match_metric", global_infer.get("match_metric", None))
    score_floor = model_infer.get("score_floor", global_infer.get("score_floor", None))
    raw_max_det = model_infer.get("raw_max_det", global_infer.get("raw_max_det", None))
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
    if tp_iou is None:
        tp_iou = tp_iou_auto
    if match_metric is None:
        match_metric = match_metric_auto
    if score_floor is None:
        score_floor = score_floor_auto
    if raw_max_det is None:
        raw_max_det = raw_max_det_auto

    if cli_device:
        device = cli_device
        device_src = "cli"
    else:
        md = str(model_infer.get("device", "") or "")
        gd = str(global_infer.get("device", "") or "")
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
        "tp_iou": float_or(tp_iou, tp_iou_auto),
        "match_metric": str(match_metric or match_metric_auto),
        "score_floor": float_or(score_floor, score_floor_auto),
        "raw_max_det": int_or(raw_max_det, raw_max_det_auto),
        "device": device,
        "device_source": device_src,
    }


def extract_multiclass_metrics(val_res) -> Dict[str, float]:
    rd = getattr(val_res, "results_dict", {}) or {}
    box = getattr(val_res, "box", None)
    map50 = rd.get("metrics/mAP50(B)")
    map5095 = rd.get("metrics/mAP50-95(B)")
    if map50 is None and box is not None:
        map50 = getattr(box, "map50", 0.0)
    if map5095 is None and box is not None:
        map5095 = getattr(box, "map", 0.0)

    cls_ps: List[float] = []
    cls_rs: List[float] = []
    try:
        per_class = val_res.summary(normalize=True, decimals=8)
        if isinstance(per_class, list):
            for c in per_class:
                if isinstance(c, dict):
                    cls_ps.append(float_or(c.get("Box-P", 0.0), 0.0))
                    cls_rs.append(float_or(c.get("Box-R", 0.0), 0.0))
    except Exception:
        pass

    if not cls_ps:
        cls_ps.append(float_or(rd.get("metrics/precision(B)", 0.0), 0.0))
    if not cls_rs:
        cls_rs.append(float_or(rd.get("metrics/recall(B)", 0.0), 0.0))

    avg_cls_p = sum(cls_ps) / max(len(cls_ps), 1)
    avg_cls_r = sum(cls_rs) / max(len(cls_rs), 1)
    return {
        "mAP@0.5": float_or(map50, 0.0),
        "mAP@0.5:0.95": float_or(map5095, 0.0),
        "avg_cls_precision": float(avg_cls_p),
        "avg_cls_recall": float(avg_cls_r),
    }


def extract_instances_weight(val_res) -> float:
    try:
        per_class = val_res.summary(normalize=True, decimals=8)
        if isinstance(per_class, list):
            s = 0.0
            for c in per_class:
                if isinstance(c, dict):
                    s += float_or(c.get("Instances", 0.0), 0.0)
            if s > 0:
                return s
    except Exception:
        pass
    return 0.0


def weighted_avg(values: List[float], weights: List[float]) -> float:
    if not values:
        return 0.0
    if len(values) != len(weights):
        return float(sum(values) / len(values))
    wsum = sum(weights)
    if wsum <= 0:
        return float(sum(values) / len(values))
    return float(sum(v * w for v, w in zip(values, weights)) / wsum)


def resolve_split_paths(data_yaml: Path, dataset_root: Path, split: str) -> List[Path]:
    d = read_yaml(data_yaml)
    entry = d.get(split)
    if entry in (None, "", False):
        if split == "test":
            entry = d.get("val")
        elif split == "val":
            entry = d.get("test")
    if entry in (None, "", False):
        raise RuntimeError(f"split '{split}' 未在 data yaml 中找到: {data_yaml}")

    base_raw = d.get("path")
    if isinstance(base_raw, str) and base_raw.strip():
        base = _resolve_ref_path(base_raw, data_yaml.parent)
    else:
        base = dataset_root

    entries: List[str] = entry if isinstance(entry, list) else [entry]
    paths: List[Path] = []
    for e in entries:
        if not isinstance(e, str):
            continue
        p = Path(e).expanduser()
        if p.is_absolute():
            paths.append(p.resolve())
            continue
        cands = [
            (base / e).resolve(),
            (dataset_root / e).resolve(),
            (data_yaml.parent / e).resolve(),
        ]
        picked = None
        for c in cands:
            if c.exists():
                picked = c
                break
        paths.append(picked if picked is not None else cands[0])
    return paths


def _read_image_list_file(txt_path: Path) -> List[Path]:
    out = []
    for line in txt_path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s:
            continue
        p = Path(s).expanduser()
        if not p.is_absolute():
            p = (txt_path.parent / p).resolve()
        else:
            p = p.resolve()
        if p.exists() and p.suffix.lower() in IMG_EXTS:
            out.append(p)
    return out


def list_split_images(split_paths: Sequence[Path]) -> List[Path]:
    images: List[Path] = []
    for p in split_paths:
        if not p.exists():
            continue
        if p.is_dir():
            images.extend(sorted([x for x in p.rglob("*") if x.is_file() and x.suffix.lower() in IMG_EXTS]))
        elif p.is_file():
            if p.suffix.lower() in IMG_EXTS:
                images.append(p.resolve())
            elif p.suffix.lower() == ".txt":
                images.extend(_read_image_list_file(p.resolve()))
    dedup = sorted({x.resolve() for x in images})
    return dedup


def infer_label_path(image_path: Path, dataset_root: Path) -> Path:
    s = str(image_path.resolve())
    marker = "/images/"
    if marker in s:
        return Path(s.replace(marker, "/labels/", 1)).with_suffix(".txt")
    images_root = dataset_root / "images"
    labels_root = dataset_root / "labels"
    try:
        rel = image_path.resolve().relative_to(images_root.resolve())
        return (labels_root / rel).with_suffix(".txt")
    except Exception:
        return image_path.with_suffix(".txt")


def has_defect_label(label_path: Path) -> bool:
    if not label_path.exists():
        return False
    try:
        for line in label_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                return True
        return False
    except Exception:
        return False


def iter_chunks(items: Sequence[Path], n: int) -> Iterable[Sequence[Path]]:
    n = max(1, int(n))
    for i in range(0, len(items), n):
        yield items[i : i + n]


def average_precision_binary(y_true: List[int], y_score: List[float]) -> float:
    if not y_true:
        return 0.0
    p = sum(1 for v in y_true if v == 1)
    if p <= 0:
        return 0.0
    order = sorted(range(len(y_score)), key=lambda i: float(y_score[i]), reverse=True)
    tp = 0
    fp = 0
    rec = [0.0]
    prec = [1.0]
    for i in order:
        if int(y_true[i]) == 1:
            tp += 1
        else:
            fp += 1
        rec.append(tp / float(p))
        prec.append(tp / float(tp + fp))
    rec.append(1.0)
    prec.append(0.0)
    # precision envelope
    for i in range(len(prec) - 2, -1, -1):
        if prec[i] < prec[i + 1]:
            prec[i] = prec[i + 1]
    ap = 0.0
    for i in range(len(rec) - 1):
        ap += (rec[i + 1] - rec[i]) * prec[i + 1]
    return float(ap)


def compute_binary_image_metrics(
    model: YOLO,
    images: List[Path],
    dataset_root: Path,
    eval_params: Dict,
) -> Dict[str, float]:
    if not images:
        return {
            "bin_img_AP": 0.0,
            "bin_img_precision": 0.0,
            "bin_img_recall": 0.0,
            "pos_images": 0,
            "neg_images": 0,
            "tp_img": 0,
            "fp_img": 0,
            "fn_img": 0,
            "tn_img": 0,
        }

    decision_thr = float_or(eval_params["conf"], 0.001)
    score_floor = float_or(eval_params.get("score_floor", 0.01), 0.01)
    predict_floor = min(score_floor, decision_thr)
    iou = float_or(eval_params["iou"], 0.7)
    imgsz = int_or(eval_params["imgsz"], 640)
    max_det = int_or(eval_params["max_det"], 300)
    raw_max_det = int_or(eval_params.get("raw_max_det", 3000), 3000)
    batch = int_or(eval_params["batch"], 4)
    device = str(eval_params["device"])

    max_score_by_img: Dict[Path, float] = {p.resolve(): 0.0 for p in images}
    for chunk in iter_chunks(images, batch):
        stream = model.predict(
            source=[str(x) for x in chunk],
            conf=predict_floor,
            iou=iou,
            imgsz=imgsz,
            max_det=max(raw_max_det, max_det),
            batch=max(1, batch),
            device=device,
            save=False,
            verbose=False,
            stream=True,
        )
        for res in stream:
            p = Path(str(res.path)).resolve()
            score = 0.0
            if res.boxes is not None and res.boxes.conf is not None and len(res.boxes.conf) > 0:
                score = float(res.boxes.conf.max().item())
            max_score_by_img[p] = max(max_score_by_img.get(p, 0.0), score)

    y_true: List[int] = []
    y_score: List[float] = []
    tp = fp = fn = tn = 0
    for img in images:
        gt_pos = has_defect_label(infer_label_path(img, dataset_root))
        score = float_or(max_score_by_img.get(img.resolve(), 0.0), 0.0)
        pred_pos = score >= decision_thr
        y_true.append(1 if gt_pos else 0)
        y_score.append(score)
        if gt_pos and pred_pos:
            tp += 1
        elif (not gt_pos) and pred_pos:
            fp += 1
        elif gt_pos and (not pred_pos):
            fn += 1
        else:
            tn += 1

    pos = tp + fn
    neg = fp + tn
    precision = tp / float(tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / float(tp + fn) if (tp + fn) > 0 else 0.0
    ap = average_precision_binary(y_true, y_score)

    return {
        "bin_img_AP": float(ap),
        "bin_img_precision": float(precision),
        "bin_img_recall": float(recall),
        "pos_images": int(pos),
        "neg_images": int(neg),
        "tp_img": int(tp),
        "fp_img": int(fp),
        "fn_img": int(fn),
        "tn_img": int(tn),
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


def build_compare_rows_by_name(metrics_rows: List[Dict]) -> List[Dict]:
    rows = []
    for r in metrics_rows:
        rows.append(
            {
                "dataset": r["dataset"],
                "name": r["name"],
                "mAP@0.5": pct(float_or(r["mAP@0.5"], 0.0)),
                "mAP@0.5:0.95": pct(float_or(r["mAP@0.5:0.95"], 0.0)),
                "avg_cls_P/%": pct(float_or(r["avg_cls_precision"], 0.0)),
                "avg_cls_R/%": pct(float_or(r["avg_cls_recall"], 0.0)),
                "bin_img_AP/%": pct(float_or(r["bin_img_AP"], 0.0)),
                "bin_img_P/%": pct(float_or(r["bin_img_precision"], 0.0)),
                "bin_img_R/%": pct(float_or(r["bin_img_recall"], 0.0)),
            }
        )
    order = {k: i for i, k in enumerate(PUBLIC_DATASETS)}
    rows.sort(key=lambda x: (order.get(x["dataset"], 999), x["name"]))
    return rows


def write_markdown_table(path: Path, rows: List[Dict]) -> None:
    lines = [
        "# 公开数据集对比结果（按 name）",
        "",
        "| 数据集 | name | mAP@0.5 | mAP@0.5:0.95 | avg_cls_P/% | avg_cls_R/% | bin_img_AP/% | bin_img_P/% | bin_img_R/% |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['dataset']} | {r['name']} | {r['mAP@0.5']} | {r['mAP@0.5:0.95']} | "
            f"{r['avg_cls_P/%']} | {r['avg_cls_R/%']} | {r['bin_img_AP/%']} | {r['bin_img_P/%']} | {r['bin_img_R/%']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_latex_table(path: Path, rows: List[Dict]) -> None:
    lines = [
        "\\begin{table}[htbp]",
        "    \\centering",
        "    \\caption{公开数据集对比结果（按 name）}",
        "    \\begin{tabular}{llccccccc}",
        "        \\toprule",
        "        数据集 & name & $mAP@0.5$ & $mAP@0.5:0.95$ & $P_{cls}$/\\% & $R_{cls}$/\\% & $AP_{img}$/\\% & $P_{img}$/\\% & $R_{img}$/\\% \\\\",
        "        \\midrule",
    ]
    for r in rows:
        lines.append(
            f"        {r['dataset']} & {r['name']} & {r['mAP@0.5']} & {r['mAP@0.5:0.95']} & "
            f"{r['avg_cls_P/%']} & {r['avg_cls_R/%']} & {r['bin_img_AP/%']} & {r['bin_img_P/%']} & {r['bin_img_R/%']} \\\\"
        )
    lines += ["        \\bottomrule", "    \\end{tabular}", "\\end{table}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def build_runtime_config(args: argparse.Namespace) -> Dict:
    cfg = copy.deepcopy(USER_EDIT_CONFIG)
    if args.config_json:
        cfg = deep_merge(cfg, json.loads(args.config_json))
    if args.weight:
        cfg["models"] = [
            {
                "name": args.name.strip() if args.name.strip() else Path(args.weight).stem,
                "path": str(args.weight),
            }
        ]
    return cfg


def resolve_dataset_root(
    model_cfg: Dict,
    global_cfg: Dict,
    cli_dataset_root: Optional[Path],
    args_yaml_obj: Dict,
    args_yaml_path: Optional[Path],
    weight: Path,
) -> Path:
    # 1) CLI / config 显式给定优先
    if cli_dataset_root:
        return cli_dataset_root.resolve()
    if model_cfg.get("dataset_root"):
        return Path(model_cfg["dataset_root"]).expanduser().resolve()
    if global_cfg.get("dataset_root"):
        return Path(global_cfg["dataset_root"]).expanduser().resolve()

    # 2) 从 args.yaml 的 data 字段优先解析
    _, ds_from_args = infer_from_args_data(args_yaml_obj, args_yaml_path)
    if ds_from_args is not None and ds_from_args.exists():
        return ds_from_args.resolve()

    # 3) 从权重路径推断公开数据集别名
    label = infer_dataset_label_from_text(str(weight))
    inferred = infer_dataset_root_from_label(label)
    if inferred and inferred.exists():
        return inferred.resolve()

    raise RuntimeError("无法自动定位 dataset_root，请在 USER_EDIT_CONFIG 或命令行显式设置")


def resolve_data_yaml_for_model(
    model_cfg: Dict,
    global_cfg: Dict,
    cli_data_yaml: Optional[Path],
    dataset_root: Path,
    args_yaml_obj: Dict,
    args_yaml_path: Optional[Path],
) -> Path:
    if cli_data_yaml:
        return cli_data_yaml.resolve()
    if model_cfg.get("data_yaml"):
        return Path(model_cfg["data_yaml"]).expanduser().resolve()
    if global_cfg.get("data_yaml"):
        return Path(global_cfg["data_yaml"]).expanduser().resolve()

    data_yaml_from_args, _ = infer_from_args_data(args_yaml_obj, args_yaml_path)
    if data_yaml_from_args is not None and data_yaml_from_args.exists():
        return data_yaml_from_args.resolve()

    p = resolve_data_yaml(dataset_root)
    if p is not None:
        return p.resolve()
    raise RuntimeError(f"未找到 data.yaml/dataset.yaml: {dataset_root}")


def resolve_splits_for_model(model_cfg: Dict, global_cfg: Dict, cli_split: str, data_yaml: Path) -> List[str]:
    if cli_split:
        return parse_split_spec(cli_split, data_yaml)
    if model_cfg.get("split"):
        return parse_split_spec(str(model_cfg["split"]), data_yaml)
    return parse_split_spec(str(global_cfg.get("split", "auto")), data_yaml)


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
    per_split_rows: List[Dict] = []
    failures: List[Dict] = []
    used_items: List[Dict] = []

    for mcfg in models:
        if not isinstance(mcfg, dict):
            failures.append({"model": str(mcfg), "error": "model config item is not dict"})
            continue
        name = str(mcfg.get("name", "unnamed")).strip() or "unnamed"
        weight_path_raw = str(mcfg.get("path", "")).strip()
        if not weight_path_raw:
            failures.append({"name": name, "error": "missing model path"})
            continue
        weight = Path(weight_path_raw).expanduser().resolve()
        if not weight.exists():
            failures.append({"name": name, "weight": str(weight), "error": "weight not found"})
            continue

        try:
            args_yaml_path = find_args_yaml(weight)
            args_yaml_obj = read_yaml(args_yaml_path) if args_yaml_path and args_yaml_path.exists() else {}

            dataset_root = resolve_dataset_root(mcfg, cfg, args.dataset_root, args_yaml_obj, args_yaml_path, weight)
            if not dataset_root.exists():
                raise RuntimeError(f"dataset_root not found: {dataset_root}")

            data_yaml = resolve_data_yaml_for_model(mcfg, cfg, args.data_yaml, dataset_root, args_yaml_obj, args_yaml_path)
            if not data_yaml.exists():
                raise RuntimeError(f"data_yaml not found: {data_yaml}")

            splits = resolve_splits_for_model(mcfg, cfg, args.split, data_yaml)
            dataset_label = str(mcfg.get("dataset_label", "")).strip() or infer_dataset_label_from_text(str(dataset_root))
            eval_params = choose_eval_params(
                args_yaml_obj=args_yaml_obj,
                global_infer=cfg.get("infer_params", {}) if isinstance(cfg.get("infer_params"), dict) else {},
                model_infer=mcfg.get("infer_params", {}) if isinstance(mcfg.get("infer_params"), dict) else {},
                cli_device=args.device,
            )

            split_images_map: Dict[str, List[Path]] = {}
            merged_images: List[Path] = []
            for sp in splits:
                sps = resolve_split_paths(data_yaml, dataset_root, sp)
                imgs = list_split_images(sps)
                split_images_map[sp] = imgs
                merged_images.extend(imgs)
            merged_images = sorted({p.resolve() for p in merged_images})

            used = {
                "name": name,
                "weight": str(weight),
                "args_yaml": str(args_yaml_path) if args_yaml_path else None,
                "dataset_root": str(dataset_root),
                "dataset_label": dataset_label,
                "data_yaml": str(data_yaml),
                "split": "+".join(splits),
                "splits": splits,
                "num_images_by_split": {k: len(v) for k, v in split_images_map.items()},
                "num_images_total": len(merged_images),
                "eval_params": eval_params,
            }
            used_items.append(used)
            logs.append(f"[run] {json.dumps(used, ensure_ascii=False)}")

            if args.dry_run:
                continue

            model = YOLO(str(weight))
            split_cls_metrics: List[Dict[str, float]] = []
            split_weights: List[float] = []
            for sp in splits:
                val_res = model.val(
                    data=str(data_yaml),
                    split=str(sp),
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
                cls_m = extract_multiclass_metrics(val_res)
                w = extract_instances_weight(val_res)
                if w <= 0:
                    w = float(len(split_images_map.get(sp, [])))
                split_cls_metrics.append(cls_m)
                split_weights.append(w)
                per_split_rows.append(
                    {
                        "name": name,
                        "dataset": dataset_label,
                        "split": sp,
                        "weight": str(weight),
                        "mAP@0.5": f"{cls_m['mAP@0.5']:.6f}",
                        "mAP@0.5:0.95": f"{cls_m['mAP@0.5:0.95']:.6f}",
                        "avg_cls_precision": f"{cls_m['avg_cls_precision']:.6f}",
                        "avg_cls_recall": f"{cls_m['avg_cls_recall']:.6f}",
                        "weight_for_agg": f"{w:.3f}",
                        "num_images": len(split_images_map.get(sp, [])),
                    }
                )

            cls_metrics = {
                "mAP@0.5": weighted_avg([m["mAP@0.5"] for m in split_cls_metrics], split_weights),
                "mAP@0.5:0.95": weighted_avg([m["mAP@0.5:0.95"] for m in split_cls_metrics], split_weights),
                "avg_cls_precision": weighted_avg([m["avg_cls_precision"] for m in split_cls_metrics], split_weights),
                "avg_cls_recall": weighted_avg([m["avg_cls_recall"] for m in split_cls_metrics], split_weights),
            }
            bin_metrics = compute_binary_image_metrics(model, merged_images, dataset_root, eval_params)

            row = {
                "name": name,
                "dataset": dataset_label,
                "weight": str(weight),
                "data_yaml": str(data_yaml),
                "split": "+".join(splits),
                "mAP@0.5": f"{cls_metrics['mAP@0.5']:.6f}",
                "mAP@0.5:0.95": f"{cls_metrics['mAP@0.5:0.95']:.6f}",
                "avg_cls_precision": f"{cls_metrics['avg_cls_precision']:.6f}",
                "avg_cls_recall": f"{cls_metrics['avg_cls_recall']:.6f}",
                "bin_img_AP": f"{bin_metrics['bin_img_AP']:.6f}",
                "bin_img_precision": f"{bin_metrics['bin_img_precision']:.6f}",
                "bin_img_recall": f"{bin_metrics['bin_img_recall']:.6f}",
                "num_images_eval": len(merged_images),
                "pos_images": int(bin_metrics["pos_images"]),
                "neg_images": int(bin_metrics["neg_images"]),
                "tp_img": int(bin_metrics["tp_img"]),
                "fp_img": int(bin_metrics["fp_img"]),
                "fn_img": int(bin_metrics["fn_img"]),
                "tn_img": int(bin_metrics["tn_img"]),
            }
            success_rows.append(row)
        except Exception as ex:
            failures.append({"name": name, "weight": str(weight), "error": str(ex)})

    # 输出
    (out_dir / "run.log").write_text("\n".join(logs) + ("\n" if logs else ""), encoding="utf-8")
    (out_dir / "used_params.json").write_text(
        json.dumps({"config": cfg, "used": used_items}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if success_rows:
        write_csv(out_dir / "metrics_all.csv", success_rows, list(success_rows[0].keys()))
        if len(success_rows) == 1:
            write_csv(out_dir / "metrics_single.csv", success_rows, list(success_rows[0].keys()))
    if per_split_rows:
        write_csv(out_dir / "metrics_by_split.csv", per_split_rows, list(per_split_rows[0].keys()))
    compare_rows = build_compare_rows_by_name(success_rows)
    write_csv(
        out_dir / "public_compare_by_name.csv",
        compare_rows,
        [
            "dataset",
            "name",
            "mAP@0.5",
            "mAP@0.5:0.95",
            "avg_cls_P/%",
            "avg_cls_R/%",
            "bin_img_AP/%",
            "bin_img_P/%",
            "bin_img_R/%",
        ],
    )
    write_markdown_table(out_dir / "public_compare_by_name.md", compare_rows)
    write_latex_table(out_dir / "public_compare_by_name.tex", compare_rows)

    if failures:
        (out_dir / "failures.json").write_text(json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[done] result_dir: {out_dir}")
    if success_rows:
        print(f"[done] metrics_all: {out_dir / 'metrics_all.csv'}")
    else:
        print("[warn] no successful model run.")
    if per_split_rows:
        print(f"[done] metrics_by_split: {out_dir / 'metrics_by_split.csv'}")
    print(f"[done] table_csv: {out_dir / 'public_compare_by_name.csv'}")
    print(f"[done] table_md : {out_dir / 'public_compare_by_name.md'}")
    print(f"[done] table_tex: {out_dir / 'public_compare_by_name.tex'}")
    if failures:
        print(f"[warn] failures: {out_dir / 'failures.json'}")


if __name__ == "__main__":
    main()
