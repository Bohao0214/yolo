#!/usr/bin/env python3
"""
统一多模型评估脚本（推理 + 统计导出）。

放置路径：
  /home/ubuntu/hpproject/yolo/analyze/code/p2604_multi_model_eval.py

核心目标：
1) 多模型统一推理，输出 mAP/Precision/Recall 与图像级、目标级、分尺度、FN/FP 机制统计。
2) 严格按 conf -> NMS(iou) -> max_det 的最终预测集合统计。
3) 保存完整中间结果（按图像导出 GT/Pred 明细）。
4) 输出 report_YYMMDDHHMM 到 analyze/result 下。

使用方式：
- 直接修改 USER_EDIT_CONFIG 后运行：
    python analyze/code/p2604_multi_model_eval.py
- 或使用参数覆盖：
    python analyze/code/p2604_multi_model_eval.py --config_json '{"models":[...],"data_yaml":"..."}'
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import gc
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# 让自定义 checkpoint（含 third_party.* 模块）在任意 cwd 下都可反序列化
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    import cv2
except Exception as exc:  # pragma: no cover
    cv2 = None  # type: ignore
    CV2_IMPORT_ERROR = exc
else:
    CV2_IMPORT_ERROR = None

try:
    import yaml
except Exception as exc:  # pragma: no cover
    yaml = None  # type: ignore
    YAML_IMPORT_ERROR = exc
else:
    YAML_IMPORT_ERROR = None

try:
    from ultralytics import YOLO
except Exception as exc:  # pragma: no cover
    YOLO = None  # type: ignore
    ULTRALYTICS_IMPORT_ERROR = exc
else:
    ULTRALYTICS_IMPORT_ERROR = None

try:  # 提前触发 namespace 包注册，避免部分环境下 torch.load 找不到 third_party
    import third_party  # type: ignore # noqa: F401
except Exception:
    pass


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
SCALE_BINS: Sequence[Tuple[str, float, float]] = (
    ("<16", 0.0, 16.0),
    ("16-32", 16.0, 32.0),
    ("32-64", 32.0, 64.0),
    (">=64", 64.0, float("inf")),
)
FN_TYPES = ["no_response", "low_score", "regression_or_match_poor", "postproc_filtered"]
FP_KINDS = ["background_fp", "pred_dup"]
FP_TYPES = ["highlight", "edge", "texture_boundary", "other"]


# 你可以只修改这里，不改其它代码
USER_EDIT_CONFIG: Dict[str, Any] = {
    "models": [
        # {"name": "basel   ine", "path": "/abs/path/to/best.pt"},
            {"name": "bsd", "path": "/home/ubuntu/hpproject/yolo/experiments/a3b3d3/datasetm6c/exp_2604050042/train/weights/best.pt"},
            {"name": "baseline", "path": "/home/ubuntu/hpproject/yolo/experiments/baseline/datasetm6c/exp_2603040206/train/weights/best.pt"},
            {"name": "our", "path": "/home/ubuntu/hpproject/yolo/experiments/a4b7d6/datasetm6c/defect241__a4__b7__d6/exp_2603060404/train/weights/best.pt"},
            {"name": "hmc", "path": "/home/ubuntu/hpproject/yolo/experiments/a7b7c7d7/datasetm6c/exp_2604050107/train/weights/best.pt"},
            
            {"name": "al3", "path": "/home/ubuntu/hpproject/yolo/experiments/a4b7d6/datasetm6c/defect241__a4__b7__d11/exp_2603060251/train/weights/best.pt"},
            {"name": "al2", "path": "/home/ubuntu/hpproject/yolo/experiments/yolo11/defect241__a6__b7__d11/exp_2603060315/train/weights/best.pt"},
            {"name": "al1", "path": "/home/ubuntu/hpproject/yolo/experiments/yolo11/defect241__a6__b6__c6__d6/exp_2603060222/train/weights/best.pt"},     

            {"name": "op1", "path": "/home/ubuntu/hpproject/yolo/experiments/yolo11/defect241__a3__c5/exp_2603012143/train/weights/best.pt"},
            {"name": "op2", "path": "/home/ubuntu/hpproject/yolo/experiments/yolo11/defect241__a3__c5/exp_2603012344/train/weights/best.pt"},
            {"name": "op3", "path": "/home/ubuntu/hpproject/yolo/experiments/a4b7d6/datasetm6c/defect241__a4__b7__d11/exp_2603060251/train/weights/best.pt"},              
            

    ],
    "data_yaml": "/home/ubuntu/hpproject/yolo/configs/enhance/datasetm6c/defect241.yaml",  # /abs/path/to/data.yaml（推荐，需你按实际数据集填写）
    "dataset_root": "/home/ubuntu/hpproject/yolo/dataset/yolo/datasetm6c",  # 可选：若 data_yaml 为空，尝试 dataset_root/data.yaml
    "split": "val+test",
    "infer_params": {
        # 下面默认值来自现有评估脚本口径：
        # p2_3_2a_fp_split.py / p23_4_eval_repro_check.py / p23_3_fn_mechanism.py
        "imgsz": 640,
        "conf": 0.25,
        "iou": 0.7,  # 对应 NMS IoU
        "max_det": 100,
        "device": "0",
        "batch": 4,
        "tp_iou": 0.3,
        "match_metric": "iou",  # iou | ios (ios=intersection over smaller box)
        "score_floor": 0.01,
        "raw_max_det": 3000,
    },
    "fp_rule_params": {
        # 下面默认值主要来自 p2_3_2b_fp_type.py / p23_3_fn_mechanism.py
        "white_thresh": 250,
        "highlight_frac": 0.05,
        "edge_margin_ratio": 0.05,
        "edge_white_frac": 0.33,
        "texture_grad_percentile": 90.0,
        "texture_grad_frac": 0.10,
    },
    "fallback_cfg_candidates": [
        "/home/ubuntu/hpproject/yolo/configs/yolo11/enhance241/defect241.yaml",
        "/home/ubuntu/hpproject/yolo/configs/yolo11/defect.yaml",
        "/home/ubuntu/hpproject/yolo/configs/baseline/datasetm6c.yaml",
    ],
    "out_root": "/home/ubuntu/hpproject/yolo/analyze/result",
    "report_prefix": "report_",
}


@dataclass
class InferParams:
    imgsz: int
    conf: float
    iou: float
    max_det: int
    device: str
    batch: int
    tp_iou: float
    match_metric: str
    score_floor: float
    raw_max_det: int


@dataclass
class FPTypeParams:
    white_thresh: int
    highlight_frac: float
    edge_margin_ratio: float
    edge_white_frac: float
    texture_grad_percentile: float
    texture_grad_frac: float


@dataclass
class ImageRecord:
    image_path: Path
    label_path: Path
    split: str
    orig_h: int
    orig_w: int
    letter_h: int
    letter_w: int
    ratio: float
    padw: float
    padh: float
    gt_boxes_orig: np.ndarray
    gt_boxes_letter: np.ndarray
    gt_cls: np.ndarray


def ensure_dependencies() -> None:
    if YOLO is None:
        raise ImportError(
            "无法导入 ultralytics。请先激活含 ultralytics 的环境。"
            f" 原始错误: {ULTRALYTICS_IMPORT_ERROR}"
        )
    if yaml is None:
        raise ImportError(
            "无法导入 pyyaml。请安装 pyyaml。"
            f" 原始错误: {YAML_IMPORT_ERROR}"
        )
    if cv2 is None:
        raise ImportError(
            "无法导入 opencv-python（cv2）。该脚本需要读取图像与 FP 类型规则分析。"
            f" 原始错误: {CV2_IMPORT_ERROR}"
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="多模型统一评估导出脚本")
    p.add_argument("--config_json", type=str, default="", help="JSON 字符串，用于覆盖 USER_EDIT_CONFIG")
    p.add_argument("--dry_run", action="store_true", help="只检查配置并输出将使用的参数，不运行推理")
    return p.parse_args()


def deep_get(d: Dict[str, Any], key: str, default: Any = None) -> Any:
    return d.get(key, default) if isinstance(d, dict) else default


def safe_float(v: Any, default: float) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)


def safe_int(v: Any, default: int) -> int:
    try:
        return int(v)
    except Exception:
        return int(default)


def safe_ratio(num: int, den: int) -> float:
    return float(num) / float(den) if den > 0 else 0.0


def make_report_dir(out_root: Path, prefix: str = "report_") -> Path:
    ts = dt.datetime.now().strftime("%y%m%d%H%M")
    base = out_root / f"{prefix}{ts}"
    if not base.exists():
        base.mkdir(parents=True, exist_ok=False)
        return base
    idx = 1
    while True:
        cand = out_root / f"{prefix}{ts}_{idx:02d}"
        if not cand.exists():
            cand.mkdir(parents=True, exist_ok=False)
            return cand
        idx += 1


def load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"YAML 不存在: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))  # type: ignore[arg-type]
    if not isinstance(data, dict):
        raise ValueError(f"YAML 顶层不是 dict: {path}")
    return data


def _resolve_cfg_key(cfg: Dict[str, Any], keys: Sequence[str]) -> Any:
    for k in keys:
        if "." in k:
            cur: Any = cfg
            ok = True
            for part in k.split("."):
                if isinstance(cur, dict) and part in cur:
                    cur = cur[part]
                else:
                    ok = False
                    break
            if ok and cur is not None:
                return cur
        else:
            if k in cfg and cfg[k] is not None:
                return cfg[k]
    return None


def resolve_param(
    explicit: Any,
    key_candidates: Sequence[str],
    fallback_cfgs: Sequence[Path],
    default: Any,
) -> Tuple[Any, str]:
    if explicit is not None:
        return explicit, "manual_input"
    for cfg_path in fallback_cfgs:
        if not cfg_path.exists():
            continue
        try:
            cfg = load_yaml(cfg_path)
            val = _resolve_cfg_key(cfg, key_candidates)
            if val is not None:
                return val, f"from_cfg:{cfg_path}"
        except Exception:
            continue
    return default, "script_default"


def find_data_yaml(dataset_root: Path) -> Optional[Path]:
    cand = dataset_root / "data.yaml"
    if cand.exists():
        return cand
    cand2 = dataset_root / "dataset.yaml"
    if cand2.exists():
        return cand2
    return None


def resolve_data_yaml(cfg: Dict[str, Any]) -> Path:
    data_yaml = str(cfg.get("data_yaml", "")).strip()
    dataset_root = str(cfg.get("dataset_root", "")).strip()
    if data_yaml:
        p = Path(data_yaml)
        if not p.exists():
            raise FileNotFoundError(f"data_yaml 不存在: {p}")
        return _resolve_data_yaml_from_yaml_or_train_cfg(p)
    if dataset_root:
        p = Path(dataset_root)
        if not p.exists():
            raise FileNotFoundError(f"dataset_root 不存在: {p}")
        found = find_data_yaml(p)
        if found is None:
            raise FileNotFoundError(
                f"dataset_root 下未找到 data.yaml / dataset.yaml: {p}. "
                "mAP 计算依赖 data.yaml，请补充 data_yaml。"
            )
        return found
    raise ValueError("缺少数据集配置：请至少设置 data_yaml，或设置 dataset_root 且其中含 data.yaml。")


def _looks_like_data_yaml(obj: Dict[str, Any]) -> bool:
    return all(k in obj for k in ("train", "val"))


def _resolve_data_yaml_from_yaml_or_train_cfg(path: Path, _depth: int = 0) -> Path:
    if _depth > 3:
        raise RuntimeError(f"data_yaml 解析层级过深，请检查配置引用链: {path}")
    cfg = load_yaml(path)
    if _looks_like_data_yaml(cfg):
        return path
    data_ref = cfg.get("data")
    if isinstance(data_ref, str) and data_ref.strip():
        ref = data_ref.strip()
        rp = Path(ref)
        if rp.is_absolute():
            candidates = [rp]
        else:
            repo_root = Path(__file__).resolve().parents[2]
            candidates = [
                (path.parent / rp).resolve(),
                (Path.cwd() / rp).resolve(),
                (repo_root / rp).resolve(),
            ]
        rp = next((c for c in candidates if c.exists()), candidates[0])
        if not rp.exists():
            raise FileNotFoundError(
                f"从训练配置 {path} 解析 data 引用失败，尝试路径均不存在: "
                f"{[str(c) for c in candidates]}"
            )
        return _resolve_data_yaml_from_yaml_or_train_cfg(rp, _depth + 1)
    raise ValueError(
        f"给定 YAML 既不是 data.yaml（缺 train/val），也不包含可解析的 data 字段: {path}"
    )


def parse_splits(split_raw: str) -> List[str]:
    parts = [s.strip() for s in str(split_raw).replace("+", ",").split(",") if s.strip()]
    if not parts:
        return ["val"]
    allowed = {"train", "val", "test"}
    out: List[str] = []
    for s in parts:
        if s not in allowed:
            raise ValueError(f"split 不支持: {s}（允许: train/val/test，可逗号分隔）")
        if s not in out:
            out.append(s)
    return out


def _remap_legacy_abs_path(p: Path) -> Optional[Path]:
    repo_root = Path(__file__).resolve().parents[2]
    p_str = str(p)
    mapping = [
        ("/home/ubuntu/project/deduibi/yolo/dataset", str(repo_root / "dataset" / "yolo")),
        ("/home/ubuntu/project/deduibi/yolo", str(repo_root)),
        ("/home/ubuntu/project/deduibi", "/home/ubuntu/hpproject"),
    ]
    for old_prefix, new_prefix in mapping:
        if p_str.startswith(old_prefix):
            suffix = p_str[len(old_prefix) :].lstrip("/")
            cand = Path(new_prefix) / suffix
            if cand.exists():
                return cand.resolve()

    # 兜底：旧路径常见 /dataset/<name>/... ，当前项目多为 /dataset/yolo/<name>/...
    if "/dataset/" in p_str and "/dataset/yolo/" not in p_str:
        left, right = p_str.split("/dataset/", 1)
        cand = Path(left + "/dataset/yolo/" + right)
        if cand.exists():
            return cand.resolve()
    return None


def resolve_path(base_dir: Path, value: str) -> Path:
    p = Path(value)
    if p.is_absolute():
        if p.exists():
            return p.resolve()
        remap = _remap_legacy_abs_path(p)
        if remap is not None:
            return remap
        return p
    return (base_dir / p).resolve()


def _remap_data_value(v: Any) -> Any:
    if isinstance(v, str):
        p = Path(v)
        if p.is_absolute():
            remap = _remap_legacy_abs_path(p)
            if remap is not None:
                return str(remap)
        return v
    if isinstance(v, list):
        return [_remap_data_value(x) for x in v]
    return v


def prepare_runtime_data_yaml(data_yaml: Path, out_dir: Path) -> Tuple[Path, Dict[str, Any]]:
    """为 Ultralytics val 准备可用 data.yaml（修复旧机器绝对路径）。"""
    cfg = load_yaml(data_yaml)
    changed = False
    details: Dict[str, Any] = {"src": str(data_yaml), "changed_fields": []}

    # remap path/train/val/test
    for key in ("path", "train", "val", "test"):
        if key not in cfg:
            continue
        old = cfg.get(key)
        new = _remap_data_value(old)
        if new != old:
            cfg[key] = new
            changed = True
            details["changed_fields"].append(key)

    # 若 path 仍是绝对路径且不存在，明确报错，避免后续 val 报错难定位
    root_val = cfg.get("path")
    if isinstance(root_val, str) and root_val.strip():
        rp = Path(root_val.strip())
        if rp.is_absolute() and not rp.exists():
            raise FileNotFoundError(
                f"data.yaml 的 path 不存在且无法自动映射: {rp} (source={data_yaml})"
            )

    runtime_path = out_dir / "runtime_data.yaml"
    runtime_path.write_text(
        yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False),  # type: ignore[arg-type]
        encoding="utf-8",
    )
    details["runtime_data_yaml"] = str(runtime_path)
    details["changed"] = bool(changed)
    return runtime_path, details


def list_images_in_dir(image_dir: Path) -> List[Path]:
    if not image_dir.exists():
        return []
    return sorted([p for p in image_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS])


def collect_split_images(data_cfg: Dict[str, Any], data_yaml_path: Path, split: str) -> List[Path]:
    root = data_cfg.get("path", "")
    root_dir = resolve_path(data_yaml_path.parent, str(root)) if root else data_yaml_path.parent
    split_val = data_cfg.get(split)
    if split_val is None:
        raise ValueError(f"data.yaml 中缺少 split={split}。可用键: {list(data_cfg.keys())}")

    def _collect_one(v: Any) -> List[Path]:
        if isinstance(v, list):
            all_imgs: List[Path] = []
            for it in v:
                all_imgs.extend(_collect_one(it))
            return all_imgs
        if isinstance(v, str):
            p = resolve_path(root_dir, v)
            if p.suffix.lower() == ".txt":
                if not p.exists():
                    raise FileNotFoundError(f"split 列表文件不存在: {p}")
                imgs: List[Path] = []
                for line in p.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    q = resolve_path(root_dir, line)
                    imgs.append(q)
                return imgs
            if p.is_dir():
                return list_images_in_dir(p)
            if p.is_file() and p.suffix.lower() in IMG_EXTS:
                return [p]
            raise FileNotFoundError(f"split={split} 路径无法识别: {p}")
        raise ValueError(f"split={split} 的值类型不支持: {type(v)}")

    images = _collect_one(split_val)
    uniq: List[Path] = []
    seen = set()
    for p in images:
        key = str(p.resolve())
        if key in seen:
            continue
        seen.add(key)
        uniq.append(p.resolve())
    if not uniq:
        raise RuntimeError(f"split={split} 没有找到图片。请检查 data.yaml: {data_yaml_path}")
    return uniq


def infer_label_path(image_path: Path) -> Path:
    parts = list(image_path.parts)
    candidates: List[Path] = []
    for token in ("images", "image"):
        if token in parts:
            idx = parts.index(token)
            for name in ("labels", "label", "lable"):
                candidates.append(Path(*parts[:idx], name, *parts[idx + 1 :]))
    if image_path.parent.name in {"train", "val", "test"}:
        split_name = image_path.parent.name
        pp = image_path.parent.parent
        if pp.name in {"images", "image"}:
            for name in ("labels", "label", "lable"):
                candidates.append(pp.parent / name / split_name / image_path.name)
        for name in ("labels", "label", "lable"):
            candidates.append(image_path.parent.parent / name / image_path.name)
    for name in ("labels", "label", "lable"):
        candidates.append(image_path.parent.parent / name / image_path.name)

    for c in candidates:
        lp = c.with_suffix(".txt")
        if lp.exists():
            return lp
    if candidates:
        return candidates[0].with_suffix(".txt")
    return image_path.with_suffix(".txt")


def read_image_shape(img_path: Path) -> Tuple[int, int]:
    img = cv2.imread(str(img_path))  # type: ignore[arg-type]
    if img is None:
        raise RuntimeError(f"无法读取图像: {img_path}")
    h, w = img.shape[:2]
    return int(h), int(w)


def xywhn_to_xyxy(xc: float, yc: float, w: float, h: float, img_w: int, img_h: int) -> Tuple[float, float, float, float]:
    x1 = (xc - w / 2.0) * img_w
    y1 = (yc - h / 2.0) * img_h
    x2 = (xc + w / 2.0) * img_w
    y2 = (yc + h / 2.0) * img_h
    return (float(x1), float(y1), float(x2), float(y2))


def load_labels_with_cls(label_path: Path, img_w: int, img_h: int) -> Tuple[np.ndarray, np.ndarray]:
    if not label_path.exists():
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int32)
    boxes: List[List[float]] = []
    classes: List[int] = []
    with label_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            try:
                cls_id = int(float(parts[0]))
                xc, yc, w, h = map(float, parts[1:5])
                box = xywhn_to_xyxy(xc, yc, w, h, img_w, img_h)
            except Exception:
                continue
            boxes.append([box[0], box[1], box[2], box[3]])
            classes.append(cls_id)
    if not boxes:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int32)
    return np.asarray(boxes, dtype=np.float32), np.asarray(classes, dtype=np.int32)


def letterbox_meta(orig_h: int, orig_w: int, imgsz: int) -> Tuple[int, int, float, float, float]:
    ratio = min(float(imgsz) / float(orig_h), float(imgsz) / float(orig_w))
    new_w = int(round(orig_w * ratio))
    new_h = int(round(orig_h * ratio))
    pad_w = float(imgsz - new_w) / 2.0
    pad_h = float(imgsz - new_h) / 2.0
    return imgsz, imgsz, ratio, pad_w, pad_h


def boxes_to_letter(boxes: np.ndarray, ratio: float, padw: float, padh: float) -> np.ndarray:
    if boxes.size == 0:
        return np.zeros((0, 4), dtype=np.float32)
    out = boxes.copy().astype(np.float32)
    out[:, [0, 2]] = out[:, [0, 2]] * ratio + padw
    out[:, [1, 3]] = out[:, [1, 3]] * ratio + padh
    return out


def box_short_side(box: np.ndarray) -> float:
    return float(max(0.0, min(box[2] - box[0], box[3] - box[1])))


def scale_bin_name(short_side: float) -> str:
    for name, lo, hi in SCALE_BINS:
        if lo <= short_side < hi:
            return name
    return ">=64"


def _intersection_and_areas(gt_boxes: np.ndarray, pred_boxes: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if gt_boxes.size == 0 or pred_boxes.size == 0:
        return (
            np.zeros((gt_boxes.shape[0], pred_boxes.shape[0]), dtype=np.float32),
            np.zeros((gt_boxes.shape[0],), dtype=np.float32),
            np.zeros((pred_boxes.shape[0],), dtype=np.float32),
        )
    ix1 = np.maximum(gt_boxes[:, None, 0], pred_boxes[None, :, 0])
    iy1 = np.maximum(gt_boxes[:, None, 1], pred_boxes[None, :, 1])
    ix2 = np.minimum(gt_boxes[:, None, 2], pred_boxes[None, :, 2])
    iy2 = np.minimum(gt_boxes[:, None, 3], pred_boxes[None, :, 3])
    iw = np.maximum(0.0, ix2 - ix1)
    ih = np.maximum(0.0, iy2 - iy1)
    inter = (iw * ih).astype(np.float32)
    gt_area = np.maximum(0.0, (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])).astype(np.float32)
    pred_area = np.maximum(0.0, (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])).astype(np.float32)
    return inter, gt_area, pred_area


def compute_iou_matrix(gt_boxes: np.ndarray, pred_boxes: np.ndarray) -> np.ndarray:
    if gt_boxes.size == 0 or pred_boxes.size == 0:
        return np.zeros((gt_boxes.shape[0], pred_boxes.shape[0]), dtype=np.float32)
    inter, gt_area, pred_area = _intersection_and_areas(gt_boxes, pred_boxes)
    union = gt_area[:, None] + pred_area[None, :] - inter
    return np.where(union > 0.0, inter / union, 0.0).astype(np.float32)


def compute_iou_vec(gt_box: np.ndarray, pred_boxes: np.ndarray) -> np.ndarray:
    if pred_boxes.size == 0:
        return np.zeros((0,), dtype=np.float32)
    ix1 = np.maximum(gt_box[0], pred_boxes[:, 0])
    iy1 = np.maximum(gt_box[1], pred_boxes[:, 1])
    ix2 = np.minimum(gt_box[2], pred_boxes[:, 2])
    iy2 = np.minimum(gt_box[3], pred_boxes[:, 3])
    iw = np.maximum(0.0, ix2 - ix1)
    ih = np.maximum(0.0, iy2 - iy1)
    inter = iw * ih
    gt_area = max(0.0, (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1]))
    pred_area = np.maximum(0.0, (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1]))
    union = gt_area + pred_area - inter
    return np.where(union > 0.0, inter / union, 0.0).astype(np.float32)


def compute_ios_matrix(gt_boxes: np.ndarray, pred_boxes: np.ndarray) -> np.ndarray:
    """
    IoS = Intersection over Smaller box area.
    定义：inter / min(area_gt, area_pred)
    """
    if gt_boxes.size == 0 or pred_boxes.size == 0:
        return np.zeros((gt_boxes.shape[0], pred_boxes.shape[0]), dtype=np.float32)
    inter, gt_area, pred_area = _intersection_and_areas(gt_boxes, pred_boxes)
    den = np.minimum(gt_area[:, None], pred_area[None, :])
    return np.where(den > 0.0, inter / den, 0.0).astype(np.float32)


def compute_ios_vec(gt_box: np.ndarray, pred_boxes: np.ndarray) -> np.ndarray:
    if pred_boxes.size == 0:
        return np.zeros((0,), dtype=np.float32)
    ix1 = np.maximum(gt_box[0], pred_boxes[:, 0])
    iy1 = np.maximum(gt_box[1], pred_boxes[:, 1])
    ix2 = np.minimum(gt_box[2], pred_boxes[:, 2])
    iy2 = np.minimum(gt_box[3], pred_boxes[:, 3])
    iw = np.maximum(0.0, ix2 - ix1)
    ih = np.maximum(0.0, iy2 - iy1)
    inter = iw * ih
    gt_area = max(0.0, (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1]))
    pred_area = np.maximum(0.0, (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1]))
    den = np.minimum(gt_area, pred_area)
    return np.where(den > 0.0, inter / den, 0.0).astype(np.float32)


def compute_match_matrix(gt_boxes: np.ndarray, pred_boxes: np.ndarray, metric: str) -> np.ndarray:
    metric = str(metric).lower()
    if metric == "iou":
        return compute_iou_matrix(gt_boxes, pred_boxes)
    if metric == "ios":
        return compute_ios_matrix(gt_boxes, pred_boxes)
    raise ValueError(f"Unsupported match metric: {metric}. Allowed: iou, ios")


def compute_match_vec(gt_box: np.ndarray, pred_boxes: np.ndarray, metric: str) -> np.ndarray:
    metric = str(metric).lower()
    if metric == "iou":
        return compute_iou_vec(gt_box, pred_boxes)
    if metric == "ios":
        return compute_ios_vec(gt_box, pred_boxes)
    raise ValueError(f"Unsupported match metric: {metric}. Allowed: iou, ios")


def hungarian_assign(cost: np.ndarray) -> List[int]:
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


def match_one_to_one(
    gt_boxes: np.ndarray,
    gt_cls: np.ndarray,
    pred_boxes: np.ndarray,
    pred_cls: np.ndarray,
    tp_iou: float,
    match_metric: str,
) -> Tuple[List[Tuple[int, int, float]], List[int], List[int]]:
    matches: List[Tuple[int, int, float]] = []
    if gt_boxes.shape[0] == 0:
        return matches, [], list(range(pred_boxes.shape[0]))
    if pred_boxes.shape[0] == 0:
        return matches, list(range(gt_boxes.shape[0])), []

    matched_gt = set()
    matched_pred = set()
    class_ids = sorted(set(gt_cls.tolist()) | set(pred_cls.tolist()))
    for cid in class_ids:
        gidx = np.where(gt_cls == cid)[0]
        pidx = np.where(pred_cls == cid)[0]
        if gidx.size == 0 or pidx.size == 0:
            continue
        overlap = compute_match_matrix(gt_boxes[gidx], pred_boxes[pidx], match_metric)
        cost = 1.0 - overlap
        assign = hungarian_assign(cost)
        for gi_local, pj_local in enumerate(assign):
            if pj_local < 0 or pj_local >= pidx.size:
                continue
            score_val = float(overlap[gi_local, pj_local])
            if score_val >= float(tp_iou):
                g_abs = int(gidx[gi_local])
                p_abs = int(pidx[pj_local])
                if g_abs in matched_gt or p_abs in matched_pred:
                    continue
                matched_gt.add(g_abs)
                matched_pred.add(p_abs)
                matches.append((g_abs, p_abs, score_val))
    unmatched_gt = [i for i in range(gt_boxes.shape[0]) if i not in matched_gt]
    unmatched_pred = [i for i in range(pred_boxes.shape[0]) if i not in matched_pred]
    return matches, unmatched_gt, unmatched_pred


def _best_score_match_for_gt(
    gt_box: np.ndarray,
    gt_cls: int,
    pred_boxes: np.ndarray,
    pred_scores: np.ndarray,
    pred_cls: np.ndarray,
    match_metric: str,
) -> Tuple[float, float]:
    if pred_boxes.shape[0] == 0:
        return 0.0, 0.0
    mask = pred_cls == gt_cls
    if not np.any(mask):
        return 0.0, 0.0
    boxes = pred_boxes[mask]
    scores = pred_scores[mask]
    scores_match = compute_match_vec(gt_box, boxes, match_metric)
    if scores_match.size == 0:
        return 0.0, 0.0
    overlap_mask = scores_match > 0.0
    if np.any(overlap_mask):
        best_score = float(np.max(scores[overlap_mask]))
    else:
        best_score = 0.0
    best_match = float(np.max(scores_match))
    return best_score, best_match


def _sanitize_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-", "."} else "_" for ch in name).strip("._") or "model"


def predict_for_images(
    model: Any,
    image_paths: Sequence[Path],
    imgsz: int,
    conf: float,
    iou: float,
    max_det: int,
    device: str,
    batch: int,
) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    step = max(1, int(batch))
    for i in range(0, len(image_paths), step):
        chunk = image_paths[i : i + step]
        source = [str(p) for p in chunk]
        results = model.predict(
            source=source,
            imgsz=int(imgsz),
            conf=float(conf),
            iou=float(iou),
            max_det=int(max_det),
            device=device if str(device).strip() else None,
            verbose=False,
            stream=False,
            save=False,
            save_txt=False,
            save_conf=False,
            save_crop=False,
            save_frames=False,
            plots=False,
            show=False,
        )
        if len(results) != len(chunk):
            raise RuntimeError(
                f"predict 返回数量与输入数量不一致: in={len(chunk)} out={len(results)}; "
                f"chunk={[str(p) for p in chunk]}"
            )
        for p, r in zip(chunk, results):
            boxes = r.boxes
            if boxes is None or boxes.xyxy is None:
                pb = np.zeros((0, 4), dtype=np.float32)
                ps = np.zeros((0,), dtype=np.float32)
                pc = np.zeros((0,), dtype=np.int32)
            else:
                pb = boxes.xyxy.detach().cpu().numpy().astype(np.float32)
                ps = boxes.conf.detach().cpu().numpy().astype(np.float32)
                pc = boxes.cls.detach().cpu().numpy().astype(np.int32)
            out[str(p)] = {
                "boxes": pb,
                "scores": ps,
                "cls": pc,
            }
    return out


def _extract_metric_value(obj: Any, key: str, default: Optional[float] = None) -> Optional[float]:
    try:
        if isinstance(obj, dict):
            val = obj.get(key, default)
            return float(val) if val is not None else default
        if hasattr(obj, key):
            val = getattr(obj, key)
            if isinstance(val, (list, tuple, np.ndarray)):
                arr = np.asarray(val, dtype=np.float32)
                return float(np.mean(arr)) if arr.size > 0 else default
            return float(val)
    except Exception:
        pass
    return default


def run_ultralytics_val(
    model: Any,
    data_yaml: Path,
    split: str,
    params: InferParams,
) -> Dict[str, Any]:
    metrics = model.val(
        data=str(data_yaml),
        split=str(split),
        imgsz=int(params.imgsz),
        conf=float(params.conf),
        iou=float(params.iou),
        max_det=int(params.max_det),
        batch=int(params.batch),
        device=params.device if str(params.device).strip() else None,
        verbose=False,
        plots=False,
        save_json=False,
    )
    box = getattr(metrics, "box", None)
    results_dict = getattr(metrics, "results_dict", {}) or {}
    speed = getattr(metrics, "speed", {}) or {}
    map50 = _extract_metric_value(box, "map50", _extract_metric_value(results_dict, "metrics/mAP50(B)", None))
    map5095 = _extract_metric_value(box, "map", _extract_metric_value(results_dict, "metrics/mAP50-95(B)", None))
    precision = _extract_metric_value(box, "mp", _extract_metric_value(results_dict, "metrics/precision(B)", None))
    recall = _extract_metric_value(box, "mr", _extract_metric_value(results_dict, "metrics/recall(B)", None))
    if precision is None:
        precision = _extract_metric_value(box, "p", None)
    if recall is None:
        recall = _extract_metric_value(box, "r", None)
    inf_ms = float(speed.get("inference", 0.0)) if isinstance(speed, dict) else 0.0
    fps = float(1000.0 / inf_ms) if inf_ms > 0 else None
    return {
        "mAP50": map50,
        "mAP50_95": map5095,
        "precision": precision,
        "recall": recall,
        "speed": speed,
        "fps": fps,
        "results_dict": results_dict,
    }


def detect_fp_type(img_bgr: np.ndarray, box: np.ndarray, fp_cfg: FPTypeParams) -> Tuple[str, Dict[str, float]]:
    h, w = img_bgr.shape[:2]
    x1 = max(0, min(w - 1, int(math.floor(float(box[0])))))
    y1 = max(0, min(h - 1, int(math.floor(float(box[1])))))
    x2 = max(0, min(w, int(math.ceil(float(box[2])))))
    y2 = max(0, min(h, int(math.ceil(float(box[3])))))
    if x2 <= x1 or y2 <= y1:
        return "other", {"reason_empty_crop": 1.0}
    crop = img_bgr[y1:y2, x1:x2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    g = gray.astype(np.float32)

    bright_mask = g >= float(fp_cfg.white_thresh)
    bright_ratio = float(bright_mask.mean()) if bright_mask.size > 0 else 0.0

    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx * gx + gy * gy)
    grad_thr = np.percentile(grad, fp_cfg.texture_grad_percentile) if grad.size > 0 else 0.0
    grad_mask = grad >= grad_thr
    grad_ratio = float(grad_mask.mean()) if grad_mask.size > 0 else 0.0

    margin_x = int(round(fp_cfg.edge_margin_ratio * w))
    margin_y = int(round(fp_cfg.edge_margin_ratio * h))
    center_x = 0.5 * (x1 + x2)
    center_y = 0.5 * (y1 + y2)
    near_edge = bool(
        center_x <= margin_x
        or center_x >= (w - margin_x)
        or center_y <= margin_y
        or center_y >= (h - margin_y)
    )

    info = {
        "bright_ratio": bright_ratio,
        "grad_ratio": grad_ratio,
        "near_edge": 1.0 if near_edge else 0.0,
        "white_thresh": float(fp_cfg.white_thresh),
        "grad_percentile": float(fp_cfg.texture_grad_percentile),
    }

    if bright_ratio >= float(fp_cfg.highlight_frac):
        return "highlight", info
    if near_edge or bright_ratio >= float(fp_cfg.edge_white_frac):
        return "edge", info
    if grad_ratio >= float(fp_cfg.texture_grad_frac):
        return "texture_boundary", info
    return "other", info


def write_csv(path: Path, rows: List[Dict[str, Any]], headers: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(headers))
        writer.writeheader()
        for r in rows:
            row = {k: r.get(k, "") for k in headers}
            writer.writerow(row)


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def format_box_xyxy(box: np.ndarray) -> str:
    return f"{float(box[0]):.2f},{float(box[1]):.2f},{float(box[2]):.2f},{float(box[3]):.2f}"


def build_image_records(data_yaml: Path, split: str, imgsz: int) -> Tuple[List[ImageRecord], Dict[str, Any]]:
    data_cfg = load_yaml(data_yaml)
    image_paths = collect_split_images(data_cfg, data_yaml, split)
    records: List[ImageRecord] = []
    missing_labels = 0
    for img_p in image_paths:
        oh, ow = read_image_shape(img_p)
        lb_h, lb_w, ratio, padw, padh = letterbox_meta(oh, ow, imgsz)
        label_path = infer_label_path(img_p)
        gt_boxes_orig, gt_cls = load_labels_with_cls(label_path, ow, oh)
        if not label_path.exists():
            missing_labels += 1
        gt_boxes_letter = boxes_to_letter(gt_boxes_orig, ratio, padw, padh)
        rec = ImageRecord(
            image_path=img_p,
            label_path=label_path,
            split=split,
            orig_h=oh,
            orig_w=ow,
            letter_h=lb_h,
            letter_w=lb_w,
            ratio=ratio,
            padw=padw,
            padh=padh,
            gt_boxes_orig=gt_boxes_orig,
            gt_boxes_letter=gt_boxes_letter,
            gt_cls=gt_cls,
        )
        records.append(rec)
    meta = {
        "data_yaml": str(data_yaml),
        "split": split,
        "image_count": len(records),
        "missing_label_files": int(missing_labels),
        "data_cfg_keys": sorted(list(data_cfg.keys())),
        "names": data_cfg.get("names", None),
        "nc": data_cfg.get("nc", None),
    }
    return records, meta


def run_one_model(
    model_name: str,
    model_path: Path,
    records: List[ImageRecord],
    data_yaml: Path,
    splits: List[str],
    params: InferParams,
    fp_params: FPTypeParams,
    report_dir: Path,
) -> Dict[str, Any]:
    model = YOLO(str(model_path))
    split_metrics: Dict[str, Dict[str, Any]] = {}
    for sp in splits:
        m = run_ultralytics_val(model, data_yaml, sp, params)
        if m.get("mAP50") is None or m.get("mAP50_95") is None:
            raise RuntimeError(
                f"模型 {model_name} 无法计算 mAP（mAP50 或 mAP50_95 为空）。"
                f"请确认 data_yaml={data_yaml}、split={sp} 与标注可用。"
            )
        split_metrics[sp] = m

    split_count: Dict[str, int] = {}
    for r in records:
        split_count[r.split] = split_count.get(r.split, 0) + 1
    total_count = max(1, sum(split_count.get(s, 0) for s in splits))

    def _wavg(key: str) -> Optional[float]:
        num = 0.0
        den = 0.0
        for sp in splits:
            val = split_metrics[sp].get(key)
            if val is None:
                continue
            w = float(split_count.get(sp, 0))
            num += float(val) * w
            den += w
        return (num / den) if den > 0 else None

    val_metrics = {
        "mAP50": _wavg("mAP50"),
        "mAP50_95": _wavg("mAP50_95"),
        "precision": _wavg("precision"),
        "recall": _wavg("recall"),
        "fps": _wavg("fps"),
        "speed": {sp: split_metrics[sp].get("speed", {}) for sp in splits},
        "results_dict": {sp: split_metrics[sp].get("results_dict", {}) for sp in splits},
        "split_metrics": split_metrics,
        "split_count": split_count,
        "split_agg": "weighted_by_image_count",
        "split_total_images": total_count,
    }

    image_paths = [r.image_path for r in records]
    pred_all = predict_for_images(
        model=model,
        image_paths=image_paths,
        imgsz=params.imgsz,
        conf=params.score_floor,
        iou=1.0,
        max_det=params.raw_max_det,
        device=params.device,
        batch=params.batch,
    )
    pred_conf = predict_for_images(
        model=model,
        image_paths=image_paths,
        imgsz=params.imgsz,
        conf=params.conf,
        iou=1.0,
        max_det=params.raw_max_det,
        device=params.device,
        batch=params.batch,
    )
    pred_final = predict_for_images(
        model=model,
        image_paths=image_paths,
        imgsz=params.imgsz,
        conf=params.conf,
        iou=params.iou,
        max_det=params.max_det,
        device=params.device,
        batch=params.batch,
    )

    # 释放 val/predict 内部缓存
    try:
        model.predictor = None
    except Exception:
        pass
    gc.collect()

    gt_total = 0
    pred_total = 0
    tp_total = 0
    fp_total = 0
    fn_total = 0

    hit_img = 0
    miss_img = 0
    fp_img = 0
    tn_img = 0

    scale_stat: Dict[str, Dict[str, int]] = {b[0]: {"gt": 0, "tp": 0, "fn": 0} for b in SCALE_BINS}
    fn_type_count = {k: 0 for k in FN_TYPES}
    fp_kind_count = {k: 0 for k in FP_KINDS}
    fp_type_count = {k: 0 for k in FP_TYPES}

    fn_rows: List[Dict[str, Any]] = []
    fp_rows: List[Dict[str, Any]] = []
    raw_lines_path = report_dir / f"raw_{_sanitize_name(model_name)}.jsonl"

    with raw_lines_path.open("w", encoding="utf-8") as raw_f:
        for rec in records:
            img_key = str(rec.image_path)
            pa = pred_all.get(img_key, {"boxes": np.zeros((0, 4), np.float32), "scores": np.zeros((0,), np.float32), "cls": np.zeros((0,), np.int32)})
            pc = pred_conf.get(img_key, {"boxes": np.zeros((0, 4), np.float32), "scores": np.zeros((0,), np.float32), "cls": np.zeros((0,), np.int32)})
            pf = pred_final.get(img_key, {"boxes": np.zeros((0, 4), np.float32), "scores": np.zeros((0,), np.float32), "cls": np.zeros((0,), np.int32)})

            gt_boxes = rec.gt_boxes_orig
            gt_cls = rec.gt_cls
            pred_boxes = pf["boxes"]
            pred_scores = pf["scores"]
            pred_cls = pf["cls"]

            matches, unmatched_gt, unmatched_pred = match_one_to_one(
                gt_boxes=gt_boxes,
                gt_cls=gt_cls,
                pred_boxes=pred_boxes,
                pred_cls=pred_cls,
                tp_iou=params.tp_iou,
                match_metric=params.match_metric,
            )
            matched_gt_set = {m[0] for m in matches}

            gt_total += int(gt_boxes.shape[0])
            pred_total += int(pred_boxes.shape[0])
            tp_total += int(len(matches))
            fn_total += int(len(unmatched_gt))
            fp_total += int(len(unmatched_pred))

            has_gt = gt_boxes.shape[0] > 0
            has_pred = pred_boxes.shape[0] > 0
            # 图像级判定改为“是否有最终预测框”：
            # - 阳性预测：has_pred=True（不要求 TP）
            # - 图像级FN：has_gt=True 且 has_pred=False
            if has_gt and has_pred:
                hit_img += 1
            elif has_gt and (not has_pred):
                miss_img += 1
            elif (not has_gt) and has_pred:
                fp_img += 1
            elif (not has_gt) and (not has_pred):
                tn_img += 1

            for gi in range(gt_boxes.shape[0]):
                g_bin = scale_bin_name(box_short_side(rec.gt_boxes_letter[gi]) if rec.gt_boxes_letter.shape[0] > gi else 0.0)
                scale_stat[g_bin]["gt"] += 1
                if gi in matched_gt_set:
                    scale_stat[g_bin]["tp"] += 1
                else:
                    scale_stat[g_bin]["fn"] += 1

            # FN 机制
            for gi in unmatched_gt:
                gt_box = gt_boxes[gi]
                gt_c = int(gt_cls[gi]) if gi < gt_cls.shape[0] else 0
                best_score_all, best_iou_all = _best_score_match_for_gt(
                    gt_box, gt_c, pa["boxes"], pa["scores"], pa["cls"], params.match_metric
                )
                _, best_iou_conf = _best_score_match_for_gt(
                    gt_box, gt_c, pc["boxes"], pc["scores"], pc["cls"], params.match_metric
                )
                _, best_iou_final = _best_score_match_for_gt(
                    gt_box, gt_c, pf["boxes"], pf["scores"], pf["cls"], params.match_metric
                )
                if best_score_all < params.score_floor:
                    fn_type = "no_response"
                elif best_score_all < params.conf:
                    fn_type = "low_score"
                elif best_iou_conf < params.tp_iou:
                    fn_type = "regression_or_match_poor"
                elif best_iou_conf >= params.tp_iou and best_iou_final < params.tp_iou:
                    fn_type = "postproc_filtered"
                else:
                    fn_type = "regression_or_match_poor"
                fn_type_count[fn_type] += 1
                fn_rows.append(
                    {
                        "model_name": model_name,
                        "split": rec.split,
                        "image_path": str(rec.image_path),
                        "gt_idx": int(gi),
                        "scale_bin": scale_bin_name(box_short_side(rec.gt_boxes_letter[gi]) if rec.gt_boxes_letter.shape[0] > gi else 0.0),
                        "best_score_all": float(best_score_all),
                        "match_metric": params.match_metric,
                        "best_metric_all": float(best_iou_all),
                        "best_metric_conf": float(best_iou_conf),
                        "best_metric_final": float(best_iou_final),
                        "best_iou_all": float(best_iou_all),
                        "best_iou_conf": float(best_iou_conf),
                        "best_iou_final": float(best_iou_final),
                        "fn_type": fn_type,
                        "gt_cls": int(gt_c),
                        "gt_box_xyxy": format_box_xyxy(gt_box),
                    }
                )

            # FP 结构 + 类型
            img_bgr = cv2.imread(str(rec.image_path))
            if img_bgr is None:
                raise RuntimeError(f"读取图像失败（FP 类型分析需要图像像素）：{rec.image_path}")
            for pi in unmatched_pred:
                pbox = pred_boxes[pi]
                pcls = int(pred_cls[pi]) if pi < pred_cls.shape[0] else -1
                pscore = float(pred_scores[pi]) if pi < pred_scores.shape[0] else 0.0
                if gt_boxes.shape[0] > 0:
                    cls_mask = gt_cls == pcls
                    iou_max = (
                        float(np.max(compute_match_vec(pbox, gt_boxes[cls_mask], params.match_metric)))
                        if np.any(cls_mask)
                        else 0.0
                    )
                else:
                    iou_max = 0.0
                fp_kind = "pred_dup" if iou_max >= params.tp_iou else "background_fp"
                fp_kind_count[fp_kind] += 1
                fp_type, fp_info = detect_fp_type(img_bgr, pbox, fp_params)
                if fp_type not in fp_type_count:
                    fp_type = "other"
                fp_type_count[fp_type] += 1
                fp_rows.append(
                    {
                        "model_name": model_name,
                        "split": rec.split,
                        "image_path": str(rec.image_path),
                        "pred_idx": int(pi),
                        "fp_kind": fp_kind,
                        "fp_type": fp_type,
                        "conf": float(pscore),
                        "cls": int(pcls),
                        "box_xyxy": format_box_xyxy(pbox),
                        "rule_info": json.dumps(fp_info, ensure_ascii=False),
                    }
                )

            raw_obj = {
                "model_name": model_name,
                "split": rec.split,
                "image_path": str(rec.image_path),
                "orig_size": [rec.orig_h, rec.orig_w],
                "letterbox_size": [rec.letter_h, rec.letter_w],
                "letterbox_ratio": rec.ratio,
                "letterbox_pad": [rec.padw, rec.padh],
                "gt": [
                    {"xyxy": [float(b[0]), float(b[1]), float(b[2]), float(b[3])], "cls": int(c)}
                    for b, c in zip(rec.gt_boxes_orig.tolist(), rec.gt_cls.tolist())
                ],
                "pred_final": [
                    {
                        "xyxy": [float(b[0]), float(b[1]), float(b[2]), float(b[3])],
                        "conf": float(s),
                        "cls": int(c),
                    }
                    for b, s, c in zip(
                        pred_boxes.tolist(),
                        pred_scores.tolist(),
                        pred_cls.tolist(),
                    )
                ],
            }
            raw_f.write(json.dumps(raw_obj, ensure_ascii=False) + "\n")

    precision_target = safe_ratio(tp_total, tp_total + fp_total)
    recall_target = safe_ratio(tp_total, tp_total + fn_total)
    image_recall = safe_ratio(hit_img, hit_img + miss_img)
    image_fpr = safe_ratio(fp_img, fp_img + tn_img)

    scale_rows: List[Dict[str, Any]] = []
    for bname, stat in scale_stat.items():
        scale_rows.append(
            {
                "model_name": model_name,
                "scale_bin": bname,
                "gt": int(stat["gt"]),
                "tp": int(stat["tp"]),
                "fn": int(stat["fn"]),
                "recall": float(safe_ratio(stat["tp"], stat["gt"])),
            }
        )

    return {
        "model_name": model_name,
        "model_path": str(model_path),
        "main_row": {
            "model_name": model_name,
            "split_used": ",".join(splits),
            "mAP50": float(val_metrics["mAP50"]),
            "mAP50_95": float(val_metrics["mAP50_95"]),
            "precision": float(val_metrics["precision"]) if val_metrics["precision"] is not None else float(precision_target),
            "recall": float(val_metrics["recall"]) if val_metrics["recall"] is not None else float(recall_target),
            "gt": int(gt_total),
            "pred": int(pred_total),
            "tp": int(tp_total),
            "fp": int(fp_total),
            "fn": int(fn_total),
            "precision_target_level": float(precision_target),
            "recall_target_level": float(recall_target),
            "fps": float(val_metrics["fps"]) if val_metrics["fps"] is not None else "",
        },
        "image_row": {
            "model_name": model_name,
            "hit_img": int(hit_img),
            "miss_img": int(miss_img),
            "fp_img": int(fp_img),
            "tn_img": int(tn_img),
            "image_recall": float(image_recall),
            "image_fpr": float(image_fpr),
        },
        "scale_rows": scale_rows,
        "fn_rows": fn_rows,
        "fp_rows": fp_rows,
        "fn_type_count": fn_type_count,
        "fp_kind_count": fp_kind_count,
        "fp_type_count": fp_type_count,
        "val_metrics": val_metrics,
        "raw_jsonl": str(raw_lines_path),
    }


def build_readme(report_dir: Path, metadata: Dict[str, Any], missing_notes: List[str]) -> None:
    lines: List[str] = []
    lines.append("# 多模型统一评估说明")
    lines.append("")
    lines.append("## 1. 指标与计算规则")
    metric_name = str(deep_get(metadata, "infer_params", {}).get("match_metric", "iou")).lower()
    if metric_name == "ios":
        lines.append("- 目标级匹配：同类别一对一 Hungarian 匹配，IoS>=tp_iou 计为 TP。")
        lines.append("  - IoS 定义：Intersection / min(area_gt, area_pred)。")
    else:
        lines.append("- 目标级匹配：同类别一对一 Hungarian 匹配，IoU>=tp_iou 计为 TP。")
    lines.append("- Precision(目标级)：TP/(TP+FP)")
    lines.append("- Recall(目标级)：TP/(TP+FN)")
    lines.append("- 图像级四格（按“是否有最终预测框”判阳性）：")
    lines.append("  - hit_img：有 GT 且最终预测框数量 > 0")
    lines.append("  - miss_img：有 GT 且最终预测框数量 = 0（图像级 FN）")
    lines.append("  - fp_img：无 GT 且最终预测框数量 > 0")
    lines.append("  - tn_img：无 GT 且最终预测框数量 = 0")
    lines.append("- 图像级召回率：hit_img/(hit_img+miss_img)")
    lines.append("- 图像级误报率：fp_img/(fp_img+tn_img)")
    lines.append("- 分尺度 Recall：按 GT 在输入坐标系（letterbox）短边 s=min(w,h) 分桶统计。")
    lines.append("- FN 机制（互斥，按顺序判定）：")
    lines.append("  1) no_response：best_score_all < score_floor")
    lines.append("  2) low_score：score_floor <= best_score_all < conf")
    lines.append("  3) regression_or_match_poor：best_score_all>=conf 且 best_metric_conf < tp_iou")
    lines.append("  4) postproc_filtered：best_metric_conf>=tp_iou 且 best_metric_final<tp_iou")
    lines.append("- FP 结构：")
    lines.append("  - pred_dup：未匹配预测中与同类任一 GT 的 IoU>=tp_iou")
    lines.append("  - background_fp：其余 FP")
    lines.append("- FP 类型（启发式）：highlight / edge / texture_boundary / other。")
    lines.append("")
    lines.append("## 2. 输出文件")
    lines.append("- compare_main.csv：主指标 + 目标级总量")
    lines.append("- image_level_stats.csv：图像级四格 + 图像级 recall/fpr")
    lines.append("- scale_recall.csv：分尺度 GT/TP/FN/Recall")
    lines.append("- fn_mechanism.csv：每个 FN 的机制判定与关键中间量")
    lines.append("- fp_structure.csv：每个 FP 的结构类型与启发式类型")
    lines.append("- metadata.json：实际参数、来源、数据与模型映射")
    lines.append("- raw_<model>.jsonl：每图原始 GT 与最终预测集合")
    lines.append("")
    lines.append("## 3. 参数来源")
    lines.append("- 优先级：手动输入 > fallback 配置文件 > 脚本默认。")
    lines.append("- 实际采用值及来源写在 metadata.json 的 infer_params 与 infer_param_source。")
    lines.append("")
    if missing_notes:
        lines.append("## 4. 未完成/限制")
        for note in missing_notes:
            lines.append(f"- {note}")
        lines.append("")
    lines.append("## 5. 复现")
    lines.append(f"- 结果目录：`{report_dir}`")
    lines.append(f"- 运行时间：`{metadata.get('run_time', '')}`")
    (report_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def merge_user_config(base_cfg: Dict[str, Any], override_json: str) -> Dict[str, Any]:
    cfg = json.loads(json.dumps(base_cfg))
    if not override_json:
        return cfg
    ov = json.loads(override_json)
    if not isinstance(ov, dict):
        raise ValueError("--config_json 必须是 JSON 对象")
    for k, v in ov.items():
        cfg[k] = v
    return cfg


def main() -> None:
    ensure_dependencies()
    args = parse_args()
    cfg = merge_user_config(USER_EDIT_CONFIG, args.config_json)

    models = cfg.get("models", [])
    if not isinstance(models, list) or len(models) == 0:
        raise ValueError("请在 USER_EDIT_CONFIG['models'] 里至少填写一个模型：[{name, path}, ...]")

    data_yaml = resolve_data_yaml(cfg)
    split_raw = str(cfg.get("split", "val")).strip() or "val"
    splits = parse_splits(split_raw)

    fallback_cfgs = [Path(p) for p in cfg.get("fallback_cfg_candidates", []) if str(p).strip()]
    infer_cfg = cfg.get("infer_params", {}) if isinstance(cfg.get("infer_params"), dict) else {}

    imgsz, src_imgsz = resolve_param(infer_cfg.get("imgsz"), ("imgsz",), fallback_cfgs, 640)
    conf, src_conf = resolve_param(infer_cfg.get("conf"), ("conf", "metric_conf"), fallback_cfgs, 0.3)
    iou, src_iou = resolve_param(infer_cfg.get("iou"), ("nms_iou", "iou"), fallback_cfgs, 0.6)
    max_det, src_max_det = resolve_param(infer_cfg.get("max_det"), ("max_det",), fallback_cfgs, 20)
    device, src_device = resolve_param(infer_cfg.get("device"), ("eval_device", "device"), fallback_cfgs, "")
    batch, src_batch = resolve_param(infer_cfg.get("batch"), ("eval_batch", "batch"), fallback_cfgs, 4)
    tp_iou, src_tp_iou = resolve_param(infer_cfg.get("tp_iou"), ("tp_iou", "match_iou"), fallback_cfgs, 0.2)
    match_metric, src_match_metric = resolve_param(infer_cfg.get("match_metric"), ("match_metric",), fallback_cfgs, "iou")
    score_floor, src_score_floor = resolve_param(infer_cfg.get("score_floor"), ("score_floor", "metric_conf"), fallback_cfgs, 0.01)
    raw_max_det, src_raw_max_det = resolve_param(infer_cfg.get("raw_max_det"), ("raw_max_det",), fallback_cfgs, 3000)

    infer_params = InferParams(
        imgsz=safe_int(imgsz, 640),
        conf=safe_float(conf, 0.3),
        iou=safe_float(iou, 0.6),
        max_det=safe_int(max_det, 20),
        device=str(device) if device is not None else "",
        batch=max(1, safe_int(batch, 4)),
        tp_iou=safe_float(tp_iou, 0.2),
        match_metric=str(match_metric).lower(),
        score_floor=safe_float(score_floor, 0.01),
        raw_max_det=max(50, safe_int(raw_max_det, 3000)),
    )
    if infer_params.match_metric not in {"iou", "ios"}:
        raise ValueError(f"infer_params.match_metric 不支持: {infer_params.match_metric}（允许: iou / ios）")

    fp_cfg_raw = cfg.get("fp_rule_params", {}) if isinstance(cfg.get("fp_rule_params"), dict) else {}
    fp_params = FPTypeParams(
        white_thresh=safe_int(fp_cfg_raw.get("white_thresh", 250), 250),
        highlight_frac=safe_float(fp_cfg_raw.get("highlight_frac", 0.05), 0.05),
        edge_margin_ratio=safe_float(fp_cfg_raw.get("edge_margin_ratio", 0.05), 0.05),
        edge_white_frac=safe_float(fp_cfg_raw.get("edge_white_frac", 0.33), 0.33),
        texture_grad_percentile=safe_float(fp_cfg_raw.get("texture_grad_percentile", 85.0), 85.0),
        texture_grad_frac=safe_float(fp_cfg_raw.get("texture_grad_frac", 0.15), 0.15),
    )

    out_root = Path(str(cfg.get("out_root", "/home/ubuntu/hpproject/yolo/analyze/result")))
    out_root.mkdir(parents=True, exist_ok=True)
    prefix = str(cfg.get("report_prefix", "report_"))
    report_dir = make_report_dir(out_root, prefix=prefix)

    runtime_data_yaml, runtime_data_info = prepare_runtime_data_yaml(data_yaml, report_dir)

    records: List[ImageRecord] = []
    split_metas: Dict[str, Any] = {}
    for sp in splits:
        rec_sp, meta_sp = build_image_records(runtime_data_yaml, sp, infer_params.imgsz)
        records.extend(rec_sp)
        split_metas[sp] = meta_sp
    dataset_meta = {
        "splits": split_metas,
        "image_count_total": int(len(records)),
    }
    if len(records) == 0:
        raise RuntimeError("没有可评估图片。请检查 data.yaml 与 split。")

    metadata: Dict[str, Any] = {
        "run_time": dt.datetime.now().isoformat(timespec="seconds"),
        "report_dir": str(report_dir),
        "data_yaml": str(data_yaml),
        "runtime_data_yaml": str(runtime_data_yaml),
        "runtime_data_info": runtime_data_info,
        "split": ",".join(splits),
        "splits": splits,
        "dataset_meta": dataset_meta,
        "infer_params": {
            "imgsz": infer_params.imgsz,
            "conf": infer_params.conf,
            "iou": infer_params.iou,
            "max_det": infer_params.max_det,
            "device": infer_params.device,
            "batch": infer_params.batch,
            "tp_iou": infer_params.tp_iou,
            "match_metric": infer_params.match_metric,
            "score_floor": infer_params.score_floor,
            "raw_max_det": infer_params.raw_max_det,
        },
        "infer_param_source": {
            "imgsz": src_imgsz,
            "conf": src_conf,
            "iou": src_iou,
            "max_det": src_max_det,
            "device": src_device,
            "batch": src_batch,
            "tp_iou": src_tp_iou,
            "match_metric": src_match_metric,
            "score_floor": src_score_floor,
            "raw_max_det": src_raw_max_det,
        },
        "fp_rule_params": {
            "white_thresh": fp_params.white_thresh,
            "highlight_frac": fp_params.highlight_frac,
            "edge_margin_ratio": fp_params.edge_margin_ratio,
            "edge_white_frac": fp_params.edge_white_frac,
            "texture_grad_percentile": fp_params.texture_grad_percentile,
            "texture_grad_frac": fp_params.texture_grad_frac,
        },
        "image_level_rule": {
            "positive_pred_definition": "final_pred_count>0",
            "hit_img": "has_gt and final_pred_count>0",
            "miss_img": "has_gt and final_pred_count==0",
            "fp_img": "no_gt and final_pred_count>0",
            "tn_img": "no_gt and final_pred_count==0",
        },
        "models": [],
    }

    if args.dry_run:
        write_json(report_dir / "metadata.json", metadata)
        build_readme(report_dir, metadata, missing_notes=[])
        print(f"[dry_run] report_dir={report_dir}")
        print(json.dumps(metadata, ensure_ascii=False, indent=2))
        return

    compare_main_rows: List[Dict[str, Any]] = []
    image_rows: List[Dict[str, Any]] = []
    scale_rows: List[Dict[str, Any]] = []
    fn_rows: List[Dict[str, Any]] = []
    fp_rows: List[Dict[str, Any]] = []
    missing_notes: List[str] = []

    for item in models:
        if not isinstance(item, dict):
            raise ValueError(f"models 项必须是 dict，收到: {item}")
        name = str(item.get("name", "")).strip()
        path = str(item.get("path", "")).strip()
        if not name or not path:
            raise ValueError(f"models 项缺少 name/path: {item}")
        model_path = Path(path)
        if not model_path.exists():
            raise FileNotFoundError(f"模型权重不存在: {model_path}")
        print(f"[run] model={name} weights={model_path}")

        result = run_one_model(
            model_name=name,
            model_path=model_path,
            records=records,
            data_yaml=runtime_data_yaml,
            splits=splits,
            params=infer_params,
            fp_params=fp_params,
            report_dir=report_dir,
        )
        compare_main_rows.append(result["main_row"])
        image_rows.append(result["image_row"])
        scale_rows.extend(result["scale_rows"])
        fn_rows.extend(result["fn_rows"])
        fp_rows.extend(result["fp_rows"])
        metadata["models"].append(
            {
                "name": name,
                "path": str(model_path),
                "split_used": ",".join(splits),
                "raw_jsonl": result["raw_jsonl"],
                "fn_type_count": result["fn_type_count"],
                "fp_kind_count": result["fp_kind_count"],
                "fp_type_count": result["fp_type_count"],
                "val_metrics": result["val_metrics"],
            }
        )

    write_csv(
        report_dir / "compare_main.csv",
        compare_main_rows,
        [
            "model_name",
            "split_used",
            "mAP50",
            "mAP50_95",
            "precision",
            "recall",
            "gt",
            "pred",
            "tp",
            "fp",
            "fn",
            "precision_target_level",
            "recall_target_level",
            "fps",
        ],
    )
    write_csv(
        report_dir / "image_level_stats.csv",
        image_rows,
        [
            "model_name",
            "hit_img",
            "miss_img",
            "fp_img",
            "tn_img",
            "image_recall",
            "image_fpr",
        ],
    )
    write_csv(
        report_dir / "scale_recall.csv",
        scale_rows,
        ["model_name", "scale_bin", "gt", "tp", "fn", "recall"],
    )
    write_csv(
        report_dir / "fn_mechanism.csv",
        fn_rows,
        [
            "model_name",
            "split",
            "image_path",
            "gt_idx",
            "scale_bin",
            "best_score_all",
            "match_metric",
            "best_metric_all",
            "best_metric_conf",
            "best_metric_final",
            "best_iou_all",
            "best_iou_conf",
            "best_iou_final",
            "fn_type",
            "gt_cls",
            "gt_box_xyxy",
        ],
    )
    write_csv(
        report_dir / "fp_structure.csv",
        fp_rows,
        [
            "model_name",
            "split",
            "image_path",
            "pred_idx",
            "fp_kind",
            "fp_type",
            "conf",
            "cls",
            "box_xyxy",
            "rule_info",
        ],
    )
    write_json(report_dir / "metadata.json", metadata)
    build_readme(report_dir, metadata, missing_notes)

    print(f"[done] report_dir={report_dir}")
    print(f"[done] compare_main.csv: {report_dir / 'compare_main.csv'}")
    print(f"[done] image_level_stats.csv: {report_dir / 'image_level_stats.csv'}")
    print(f"[done] scale_recall.csv: {report_dir / 'scale_recall.csv'}")
    print(f"[done] fn_mechanism.csv: {report_dir / 'fn_mechanism.csv'}")
    print(f"[done] fp_structure.csv: {report_dir / 'fp_structure.csv'}")
    print(f"[done] metadata.json: {report_dir / 'metadata.json'}")


if __name__ == "__main__":
    main()


"""
用法说明（快速开始）

1) 直接在本文件里修改 USER_EDIT_CONFIG（推荐）
   - models: [{"name":"模型名","path":"权重绝对路径"}, ...]
   - data_yaml: 数据集 data.yaml 绝对路径
   - split: "val" / "test" / "train"
   - infer_params: 可留空自动从 fallback 配置读取，否则手填覆盖

   运行：
   python analyze/code/p2604_multi_model_eval.py

2) 不改文件，命令行传 --config_json 覆盖
   运行示例：
   python analyze/code/p2604_multi_model_eval.py --config_json '{
     "models":[
       {"name":"baseline","path":"/abs/path/to/best.pt"},
       {"name":"a3_c5","path":"/abs/path/to/best2.pt"}
     ],
     "data_yaml":"/abs/path/to/data.yaml",
     "split":"val",
     "infer_params":{"imgsz":640,"conf":0.3,"iou":0.6,"max_det":20,"batch":4,"tp_iou":0.2,"match_metric":"iou","score_floor":0.01}
   }'

   小框友好匹配（IoS，交集占较小框面积）示例：
   python analyze/code/p2604_multi_model_eval.py --config_json '{
     "models":[{"name":"m","path":"/abs/path/to/best.pt"}],
     "data_yaml":"/abs/path/to/data.yaml",
     "split":"val,test",
     "infer_params":{"match_metric":"ios","tp_iou":0.4}
   }'

3) 仅检查配置是否可解析（不推理）
   python analyze/code/p2604_multi_model_eval.py --dry_run

4) 输出目录
   默认写到：
   /home/ubuntu/hpproject/yolo/analyze/result/report_YYMMDDHHMM
"""
