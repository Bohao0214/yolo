"""P2.3.4 评估口径一致性与可复现性检查（最简可执行版，YOLO）。

目标：
在固定输入与参数条件下，验证评估工具链：
1) 可复现：同一组 (conf, tp_iou, nms_iou, max_det) 在同一数据上连续运行 R 次，关键计数完全一致。
2) 记录性扫描：对少量参数组合做“记录”，保存每组参数的计数（不追求最优）。
3) 多路径合并：支持多个 --image_dir；按 split（例如 val/test）分别统计，并额外给出 all 汇总（计数直接相加）。

P2.3.0 评估口径冻结（禁止漂移）：
后处理流水线：conf 过滤 -> NMS（非极大值抑制，重叠框去重）-> max_det 截断，得到最终预测框集合。

图像级四格（产线是否报警）：
- 图像级命中(hit)：该图存在至少一个 GT（ground_truth，标注框）且预测框非空
- 图像级漏检(miss)：该图存在 GT 但预测框为空
- 图像级误报(false_alarm)：该图无 GT 但预测框非空
- 图像级真阴(true_negative)：该图无 GT 且预测框为空

目标级（诊断用）：
基于预测框与标注框的一对一匹配关系（优先使用匈牙利算法最大化总 IoU），且 IoU >= tp_iou 才算有效匹配。
- 目标级命中：完成有效匹配的预测框（TP）
- 目标级误报：未匹配到任何 GT 的预测框（FP）
- 目标级漏检：未被任何预测框匹配到的 GT（FN）

优化目标声明（只写入报告，不做多余推导）：
优先保证图像级召回率，在此前提下尽量降低图像级误报率，同时维持合理目标级指标用于诊断。

可复制命令示例：
python /home/ubuntu/hpproject/yolo/analyze/code/p23_4_eval_repro_check.py \
  --weights /home/ubuntu/hpproject/yolo/models/defect/exp_260202_base/best/best.pt \
  --image_dir /home/ubuntu/hpproject/yolo/dataset/yolo/datasetm6c/images/val \
  --image_dir /home/ubuntu/hpproject/yolo/dataset/yolo/datasetm6c/images/test \
  --out_root /home/ubuntu/hpproject/yolo/analyze/result \
  --batch 4 --infer_chunk 16 \
  --conf 0.3 --tp_iou 0.2 --nms_iou 0.6 --max_det 20 \
  --repro_runs 3 \
  --conf_list 0.3 0.25 0.2 0.15 0.1 \
  --nms_list 0.5 0.6 0.7 \
  --max_det_list 20 50 100
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import gc
import json
import locale
import os
import sys
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


def _ensure_utf8_locale() -> None:
    """避免在 ASCII locale 下导入 SciPy 时触发 UnicodeDecodeError。"""

    try:
        enc = (locale.getpreferredencoding(False) or "").lower()
        if enc in {"ascii", "ansi_x3.4-1968"}:
            os.environ.setdefault("LANG", "C.UTF-8")
            os.environ.setdefault("LC_ALL", "C.UTF-8")
            locale.setlocale(locale.LC_ALL, "")
    except Exception:
        pass


_ensure_utf8_locale()

try:
    import cv2
except Exception as exc:  # pragma: no cover
    cv2 = None  # type: ignore
    CV2_IMPORT_ERROR = exc
else:
    CV2_IMPORT_ERROR = None

try:
    from ultralytics import YOLO
except Exception as exc:  # pragma: no cover
    YOLO = None  # type: ignore
    ULTRALYTICS_IMPORT_ERROR = exc
else:
    ULTRALYTICS_IMPORT_ERROR = None

try:
    from scipy.optimize import linear_sum_assignment
except Exception as exc:  # pragma: no cover
    linear_sum_assignment = None  # type: ignore
    SCIPY_IMPORT_ERROR = exc
else:
    SCIPY_IMPORT_ERROR = None


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def ensure_ultralytics() -> None:
    if YOLO is None:
        raise ImportError(
            "Failed to import ultralytics. Please install ultralytics in the environment. "
            f"Original error: {ULTRALYTICS_IMPORT_ERROR}"
        )


def ensure_cv2() -> None:
    if cv2 is None:
        raise ImportError(
            "Failed to import cv2. Please install opencv-python in the environment. "
            f"Original error: {CV2_IMPORT_ERROR}"
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="P2.3.4 评估口径一致性与可复现性检查（YOLO）")

    p.add_argument(
        "--weights",
        type=str,
        default="/home/ubuntu/hpproject/yolo/models/defect/exp_2602012004/best/best.pt",
        help="权重路径（默认按 P2.3.4 要求给定）",
    )
    p.add_argument(
        "--image_dir",
        type=str,
        action="append",
        default=[],
        help="数据集 images 目录（如 images/val、images/test），可重复提供或用逗号分隔。",
    )
    p.add_argument(
        "--out_root",
        type=str,
        default="/home/ubuntu/hpproject/yolo/analyze/result",
        help="输出根目录（结果写入 out_root/report_yymmddHHMM/）",
    )

    # 显存与并行约束
    p.add_argument("--batch", type=int, default=4, help="推理 batch（必须 < 8）")
    p.add_argument("--infer_chunk", type=int, default=16, help="每次送入 model.predict 的图片数量（默认 16）")
    p.add_argument("--imgsz", type=int, default=640, help="推理输入尺寸（letterbox，默认 640）")
    p.add_argument("--device", type=str, default="", help="推理设备（例如 0 / cpu；留空交给框架默认）")

    # P2.3.0 默认参数
    p.add_argument("--conf", type=float, default=0.3)
    p.add_argument("--tp_iou", type=float, default=0.2)
    p.add_argument("--nms_iou", type=float, default=0.6)
    p.add_argument("--max_det", type=int, default=20)

    # 可复现性
    p.add_argument("--repro_runs", type=int, default=3, help="固定参数连续复跑次数（默认 3）")

    # 扫描记录（可只给部分列表；未提供则使用默认参数的单值）
    p.add_argument("--conf_list", type=float, nargs="+", default=None)
    p.add_argument("--tp_iou_list", type=float, nargs="+", default=None)
    p.add_argument("--nms_list", type=float, nargs="+", default=None)
    p.add_argument("--max_det_list", type=int, nargs="+", default=None)

    p.add_argument("--seed", type=int, default=0, help="随机种子（用于固定潜在随机性；默认 0）")
    return p.parse_args()


def normalize_path_list(raw: Optional[Sequence[str]]) -> List[Path]:
    if not raw:
        return []
    out: List[Path] = []
    for item in raw:
        if not item:
            continue
        for part in str(item).split(","):
            s = part.strip()
            if not s:
                continue
            out.append(Path(s))
    seen = set()
    uniq: List[Path] = []
    for p in out:
        key = str(p)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(p)
    return uniq


def list_images(image_dir: Path) -> List[Path]:
    if image_dir.is_file() and image_dir.suffix.lower() == ".txt":
        imgs: List[Path] = []
        with image_dir.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                p = Path(s)
                if p.suffix.lower() in IMG_EXTS:
                    imgs.append(p)
        return imgs
    if not image_dir.is_dir():
        return []
    return sorted([p for p in image_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS])


def _label_dir_candidates(image_dir: Path) -> List[Path]:
    parts = list(image_dir.parts)
    candidates: List[Path] = []
    if "images" in parts:
        idx = parts.index("images")
        for name in ("labels", "label", "lable"):
            candidates.append(Path(*parts[:idx], name, *parts[idx + 1 :]))
    if "image" in parts:
        idx = parts.index("image")
        for name in ("labels", "label", "lable"):
            candidates.append(Path(*parts[:idx], name, *parts[idx + 1 :]))
    if image_dir.name in {"train", "val", "test"}:
        parent = image_dir.parent
        if parent.name in {"images", "image"}:
            for name in ("labels", "label", "lable"):
                candidates.append(parent.parent / name / image_dir.name)
        for name in ("labels", "label", "lable"):
            candidates.append(image_dir.parent / name)
    for name in ("labels", "label", "lable"):
        candidates.append(image_dir.parent / name)
    seen = set()
    uniq: List[Path] = []
    for c in candidates:
        if str(c) not in seen:
            uniq.append(c)
            seen.add(str(c))
    return uniq


def infer_label_dir(image_dir: Path) -> Path:
    candidates = _label_dir_candidates(image_dir)
    for cand in candidates:
        if cand.exists():
            return cand
    return candidates[0] if candidates else image_dir.parent / "labels"


def make_report_dir(out_root: Path) -> Path:
    ts = dt.datetime.now().strftime("report_%y%m%d%H%M")
    base = out_root / ts
    if not base.exists():
        base.mkdir(parents=True, exist_ok=False)
        return base
    suffix = 1
    while True:
        cand = out_root / f"{ts}_{suffix:02d}"
        if not cand.exists():
            cand.mkdir(parents=True, exist_ok=False)
            return cand
        suffix += 1


def _dedup_keep_order(seq: Sequence[object]) -> List[object]:
    seen = set()
    out: List[object] = []
    for x in seq:
        k = str(x)
        if k in seen:
            continue
        out.append(x)
        seen.add(k)
    return out


def xywhn_to_xyxy(
    xc: float, yc: float, w: float, h: float, img_w: int, img_h: int
) -> Tuple[float, float, float, float]:
    x1 = (xc - w / 2.0) * img_w
    y1 = (yc - h / 2.0) * img_h
    x2 = (xc + w / 2.0) * img_w
    y2 = (yc + h / 2.0) * img_h
    return (float(x1), float(y1), float(x2), float(y2))


def load_labels(label_path: Path, img_w: int, img_h: int) -> List[Tuple[float, float, float, float]]:
    if not label_path.exists():
        return []
    out: List[Tuple[float, float, float, float]] = []
    with label_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            try:
                xc, yc, w, h = map(float, parts[1:5])
            except ValueError:
                continue
            out.append(xywhn_to_xyxy(xc, yc, w, h, img_w, img_h))
    return out


def compute_iou_matrix(gt_boxes: np.ndarray, pred_boxes: np.ndarray) -> np.ndarray:
    if gt_boxes.size == 0 or pred_boxes.size == 0:
        return np.zeros((gt_boxes.shape[0], pred_boxes.shape[0]), dtype=np.float32)
    ix1 = np.maximum(gt_boxes[:, None, 0], pred_boxes[None, :, 0])
    iy1 = np.maximum(gt_boxes[:, None, 1], pred_boxes[None, :, 1])
    ix2 = np.minimum(gt_boxes[:, None, 2], pred_boxes[None, :, 2])
    iy2 = np.minimum(gt_boxes[:, None, 3], pred_boxes[None, :, 3])
    iw = np.maximum(0.0, ix2 - ix1)
    ih = np.maximum(0.0, iy2 - iy1)
    inter = iw * ih
    gt_area = np.maximum(0.0, (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1]))
    pred_area = np.maximum(0.0, (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1]))
    union = gt_area[:, None] + pred_area[None, :] - inter
    return np.where(union > 0, inter / union, 0.0).astype(np.float32)


def _match_one_to_one_hungarian(iou_mat: np.ndarray, tp_iou: float) -> int:
    """返回 TP 数量（有效匹配数），其余用公式得到 FP/FN。"""

    if linear_sum_assignment is None:
        raise ImportError(
            "Failed to import scipy.optimize.linear_sum_assignment. "
            f"Original error: {SCIPY_IMPORT_ERROR}"
        )
    if iou_mat.size == 0:
        return 0
    # linear_sum_assignment 解最小化；cost=1-IoU 等价于最大化 IoU
    cost = (1.0 - iou_mat).astype(np.float64, copy=False)
    row_ind, col_ind = linear_sum_assignment(cost)
    if row_ind.size == 0:
        return 0
    matched_iou = iou_mat[row_ind, col_ind]
    return int(np.sum(matched_iou >= float(tp_iou)))


def _match_one_to_one_greedy(iou_mat: np.ndarray, tp_iou: float) -> int:
    if iou_mat.size == 0:
        return 0
    pairs: List[Tuple[float, int, int]] = []
    for gi in range(iou_mat.shape[0]):
        for pi in range(iou_mat.shape[1]):
            v = float(iou_mat[gi, pi])
            if v >= float(tp_iou):
                pairs.append((v, gi, pi))
    pairs.sort(reverse=True, key=lambda x: x[0])
    used_g: set[int] = set()
    used_p: set[int] = set()
    tp = 0
    for v, gi, pi in pairs:
        if gi in used_g or pi in used_p:
            continue
        used_g.add(gi)
        used_p.add(pi)
        tp += 1
    return tp


def match_one_to_one(iou_mat: np.ndarray, tp_iou: float) -> int:
    """优先匈牙利算法；不可用时回退到贪心（并在报告中提示）。"""

    if linear_sum_assignment is not None:
        return _match_one_to_one_hungarian(iou_mat, tp_iou)
    return _match_one_to_one_greedy(iou_mat, tp_iou)


@dataclass
class Counts:
    # 图像级四格
    img_hit: int = 0
    img_miss: int = 0
    img_false_alarm: int = 0
    img_true_negative: int = 0
    # 目标级计数
    obj_tp: int = 0
    obj_fp: int = 0
    obj_fn: int = 0
    # 辅助计数（便于排查）
    images_total: int = 0
    gt_total: int = 0
    pred_total: int = 0
    missing_label_files: int = 0
    missing_infer_results: int = 0

    def add(self, other: "Counts") -> None:
        for k in self.__dict__.keys():
            setattr(self, k, int(getattr(self, k)) + int(getattr(other, k)))

    def key_tuple(self) -> Tuple[int, int, int, int, int, int, int]:
        return (
            int(self.img_hit),
            int(self.img_miss),
            int(self.img_false_alarm),
            int(self.img_true_negative),
            int(self.obj_tp),
            int(self.obj_fp),
            int(self.obj_fn),
        )


@dataclass(frozen=True)
class EvalParams:
    conf: float
    tp_iou: float
    nms_iou: float
    max_det: int
    batch: int
    infer_chunk: int
    imgsz: int
    device: str


@dataclass(frozen=True)
class EvalItem:
    img_path: Path
    label_path: Path
    split: str


def _image_to_label_path(img_path: Path, image_dir: Path, label_dir: Path) -> Path:
    try:
        rel = img_path.relative_to(image_dir)
    except Exception:
        return label_dir / f"{img_path.stem}.txt"
    return label_dir / rel.with_suffix(".txt")


def build_items(image_dirs: Sequence[Path]) -> Tuple[List[EvalItem], List[str], List[Path]]:
    """返回 (items, split_order, label_dirs_used). split 规则：split = image_dir 路径末级名。"""

    items: List[EvalItem] = []
    split_order: List[str] = []
    label_dirs: List[Path] = []
    for image_dir in image_dirs:
        split = image_dir.name
        if split not in split_order:
            split_order.append(split)
        label_dir = infer_label_dir(image_dir)
        label_dirs.append(label_dir)
        imgs = list_images(image_dir)
        for img_path in imgs:
            label_path = _image_to_label_path(img_path, image_dir, label_dir)
            items.append(EvalItem(img_path=img_path, label_path=label_path, split=split))
    return items, split_order, label_dirs


def set_global_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def _maybe_empty_cuda_cache() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _abspath(path_like: object) -> str:
    try:
        return os.path.abspath(str(path_like))
    except Exception:
        return str(path_like)


def evaluate_once(model, items: Sequence[EvalItem], params: EvalParams, *, print_warnings: bool = True) -> Dict[str, Counts]:
    """对 items 跑一次评估，返回 split->Counts，并额外包含 'all'。"""

    if int(params.batch) >= 8:
        raise ValueError(f"batch 必须 < 8，当前 batch={params.batch}")
    if int(params.infer_chunk) <= 0:
        raise ValueError(f"infer_chunk 必须 > 0，当前 infer_chunk={params.infer_chunk}")
    if not items:
        raise RuntimeError("未找到图像（items 为空）。")

    counts_by_split: Dict[str, Counts] = {}
    warnings: List[str] = []

    total_images = len(items)
    for start in range(0, total_images, int(params.infer_chunk)):
        chunk = list(items[start : start + int(params.infer_chunk)])
        sources = [_abspath(it.img_path) for it in chunk]
        results = model.predict(
            source=sources,
            imgsz=int(params.imgsz),
            conf=float(params.conf),
            iou=float(params.nms_iou),
            max_det=int(params.max_det),
            save=False,
            verbose=False,
            batch=int(params.batch),
            device=str(params.device) if params.device else None,
            stream=True,
        )
        # chunk 规模不大：用 map 保证按 path 查找更稳
        result_map = {_abspath(res.path): res for res in results}

        for it in chunk:
            split = it.split
            if split not in counts_by_split:
                counts_by_split[split] = Counts()
            st = counts_by_split[split]
            st.images_total += 1

            res = result_map.get(_abspath(it.img_path))
            if res is None:
                st.missing_infer_results += 1
                warnings.append(f"[WARN] 推理结果缺失: {it.img_path}")
                preds_xyxy: List[Tuple[float, float, float, float]] = []
                h = w = 0
            else:
                if getattr(res, "orig_shape", None):
                    h, w = int(res.orig_shape[0]), int(res.orig_shape[1])
                else:
                    ensure_cv2()
                    img = cv2.imread(str(it.img_path))
                    if img is None:
                        raise FileNotFoundError(f"Failed to read image: {it.img_path}")
                    h, w = img.shape[:2]
                    del img

                boxes = res.boxes
                preds_xyxy = []
                if boxes is not None:
                    if getattr(boxes, "xyxyn", None) is not None:
                        xyxyn = boxes.xyxyn.detach().cpu().numpy()
                        xyxy = np.zeros_like(xyxyn)
                        xyxy[:, [0, 2]] = xyxyn[:, [0, 2]] * float(w)
                        xyxy[:, [1, 3]] = xyxyn[:, [1, 3]] * float(h)
                    else:
                        xyxy = boxes.xyxy.detach().cpu().numpy() if boxes.xyxy is not None else np.zeros((0, 4))
                    for box in xyxy:
                        preds_xyxy.append((float(box[0]), float(box[1]), float(box[2]), float(box[3])))

            gt_boxes = load_labels(it.label_path, w, h) if (w > 0 and h > 0) else []
            has_gt = len(gt_boxes) > 0
            has_pred = len(preds_xyxy) > 0

            st.gt_total += len(gt_boxes)
            st.pred_total += len(preds_xyxy)
            if not it.label_path.exists():
                st.missing_label_files += 1

            # 图像级四格：仅看“是否有 gt / 是否有预测框”
            if has_gt and has_pred:
                st.img_hit += 1
            elif has_gt and (not has_pred):
                st.img_miss += 1
            elif (not has_gt) and has_pred:
                st.img_false_alarm += 1
            else:
                st.img_true_negative += 1

            # 目标级：一对一匹配（IoU>=tp_iou 才算有效）
            gt_arr = np.array(gt_boxes, dtype=np.float32) if gt_boxes else np.zeros((0, 4), dtype=np.float32)
            pred_arr = np.array(preds_xyxy, dtype=np.float32) if preds_xyxy else np.zeros((0, 4), dtype=np.float32)
            iou_mat = compute_iou_matrix(gt_arr, pred_arr)
            tp = match_one_to_one(iou_mat, float(params.tp_iou))
            st.obj_tp += int(tp)
            st.obj_fp += int(pred_arr.shape[0]) - int(tp)
            st.obj_fn += int(gt_arr.shape[0]) - int(tp)

            # 及时释放（防止显存/内存累积）
            del res
            del preds_xyxy
            del gt_boxes
            del gt_arr
            del pred_arr
            del iou_mat

        del result_map
        del results
        gc.collect()
        _maybe_empty_cuda_cache()

    # all 汇总
    all_counts = Counts()
    for st in counts_by_split.values():
        all_counts.add(st)
    counts_by_split["all"] = all_counts

    # 控制台仅打印少量提醒
    if print_warnings and warnings:
        print(f"[WARN] 本次评估共 {len(warnings)} 条提醒（仅展示前 5 条）：")
        for w in warnings[:5]:
            print(w)
        if len(warnings) > 5:
            print("[WARN] ... 其余提醒已省略")

    return counts_by_split


def counts_to_row(run_id: int, split: str, params: EvalParams, st: Counts) -> dict:
    return {
        "run(复跑序号)": int(run_id),
        "split(子集)": str(split),
        "conf": float(params.conf),
        "tp_iou": float(params.tp_iou),
        "nms_iou": float(params.nms_iou),
        "max_det": int(params.max_det),
        "图片数(images_total)": int(st.images_total),
        "图像级命中(hit)": int(st.img_hit),
        "图像级漏检(miss)": int(st.img_miss),
        "图像级误报(false_alarm)": int(st.img_false_alarm),
        "图像级真阴(true_negative)": int(st.img_true_negative),
        "目标级TP(命中)": int(st.obj_tp),
        "目标级FP(误报)": int(st.obj_fp),
        "目标级FN(漏检)": int(st.obj_fn),
        "GT总数(gt_total)": int(st.gt_total),
        "预测框总数(pred_total)": int(st.pred_total),
        "缺失标注文件数(missing_label_files)": int(st.missing_label_files),
        "缺失推理结果数(missing_infer_results)": int(st.missing_infer_results),
    }


def counts_to_scan_row(scan_id: int, split: str, params: EvalParams, st: Counts) -> dict:
    row = counts_to_row(0, split, params, st)
    row.pop("run(复跑序号)", None)
    row = {"scan_id(扫描序号)": int(scan_id), **row}
    return row


FIELD_NAME_MAP = {
    "img_hit": "图像级命中(hit)",
    "img_miss": "图像级漏检(miss)",
    "img_false_alarm": "图像级误报(false_alarm)",
    "img_true_negative": "图像级真阴(true_negative)",
    "obj_tp": "目标级TP(命中)",
    "obj_fp": "目标级FP(误报)",
    "obj_fn": "目标级FN(漏检)",
}


def compare_repro(runs: List[Dict[str, Counts]]) -> Tuple[bool, List[dict]]:
    if not runs:
        return False, [{"error": "runs 为空"}]
    base = runs[0]
    diffs: List[dict] = []
    keys = ("img_hit", "img_miss", "img_false_alarm", "img_true_negative", "obj_tp", "obj_fp", "obj_fn")
    for run_id in range(1, len(runs)):
        cur = runs[run_id]
        all_splits = sorted(set(base.keys()) | set(cur.keys()))
        for split in all_splits:
            b = base.get(split)
            c = cur.get(split)
            if b is None or c is None:
                diffs.append(
                    {
                        "run": int(run_id),
                        "split": str(split),
                        "统计项": "split 缺失",
                        "run0": bool(b is not None),
                        "runN": bool(c is not None),
                        "差值": None,
                    }
                )
                continue
            for k in keys:
                bv = int(getattr(b, k))
                cv = int(getattr(c, k))
                if bv != cv:
                    diffs.append(
                        {
                            "run": int(run_id),
                            "split": str(split),
                            "统计项": str(FIELD_NAME_MAP.get(str(k), str(k))),
                            "run0": int(bv),
                            "runN": int(cv),
                            "差值": int(cv - bv),
                        }
                    )
    return (len(diffs) == 0), diffs


def write_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def format_params_list(param_grid: List[Tuple[float, float, float, int]], max_lines: int = 30) -> str:
    if not param_grid:
        return "（无）"
    if len(param_grid) > max_lines:
        head = param_grid[:max_lines]
        tail_n = len(param_grid) - max_lines
        lines = [f"- conf={a:.4g}, tp_iou={b:.4g}, nms_iou={c:.4g}, max_det={d}" for a, b, c, d in head]
        lines.append(f"- ...（其余 {tail_n} 组见 scan_summary.csv）")
        return "\n".join(lines)
    return "\n".join([f"- conf={a:.4g}, tp_iou={b:.4g}, nms_iou={c:.4g}, max_det={d}" for a, b, c, d in param_grid])


def collect_versions() -> Dict[str, str]:
    vers: Dict[str, str] = {"python": sys.version.replace("\n", " ")}
    vers["numpy"] = str(np.__version__)
    if cv2 is not None:
        try:
            vers["opencv(cv2)"] = str(cv2.__version__)
        except Exception:
            pass
    if linear_sum_assignment is not None:
        try:
            import scipy

            vers["scipy"] = str(scipy.__version__)
        except Exception:
            pass
    try:
        import torch

        vers["torch"] = str(torch.__version__)
        vers["cuda_available"] = str(bool(torch.cuda.is_available()))
        if torch.cuda.is_available():
            try:
                vers["cuda_device_name"] = str(torch.cuda.get_device_name(0))
            except Exception:
                pass
    except Exception:
        pass
    if YOLO is not None:
        try:
            import ultralytics

            vers["ultralytics"] = str(ultralytics.__version__)
        except Exception:
            pass
    return vers


def write_report_md(
    report_path: Path,
    cfg: dict,
    repro_pass: bool,
    repro_diffs: List[dict],
    scan_param_grid: List[Tuple[float, float, float, int]],
    used_hungarian: bool,
) -> None:
    lines: List[str] = []
    lines.append("# P2.3.4 评估口径一致性与可复现性检查报告")
    lines.append("")
    lines.append(f"- 生成时间：{cfg.get('created_at')}")
    lines.append(f"- 输出目录：{cfg.get('report_dir')}")
    lines.append("")

    lines.append("## 1) 评估口径（P2.3.0 冻结）")
    lines.append("- 推理后处理：先按 conf 过滤，再做 NMS（非极大值抑制），再做 max_det 截断。")
    lines.append("- 图像级：只看该图是否存在 GT（ground_truth，标注框）、以及最终预测框是否为空（不引入 near/strict 等概念）。")
    lines.append("- 目标级：一对一匹配，IoU（交并比）>= tp_iou 才算有效匹配；TP/FP/FN=命中/误报/漏检。")
    if used_hungarian:
        lines.append("- 一对一匹配实现：匈牙利算法（最大化总 IoU）。")
    else:
        lines.append("- 一对一匹配实现：匈牙利算法不可用，已回退为贪心匹配（按 IoU 从大到小）。")
    lines.append("")

    lines.append("## 2) 优化目标声明（不做多余推导）")
    lines.append("- 优先保证图像级召回率，在此前提下尽量降低图像级误报率，同时维持合理目标级指标用于诊断。")
    lines.append("")

    lines.append("## 3) 本次输入与参数")
    lines.append(f"- 权重：{cfg.get('weights')}")
    lines.append("- 数据路径（image_dir，按提供顺序）：")
    for p in cfg.get("image_dir", []):
        lines.append(f"  - {p}")
    lines.append("- 标注路径（label_dir，自动推断，与 image_dir 一一对应）：")
    for p in cfg.get("label_dir", []):
        lines.append(f"  - {p}")
    lines.append(
        "- 默认参数："
        f"conf={cfg.get('conf')}, tp_iou={cfg.get('tp_iou')}, nms_iou={cfg.get('nms_iou')}, max_det={cfg.get('max_det')}"
    )
    lines.append(f"- 复跑次数 R：{cfg.get('repro_runs')}")
    lines.append(f"- batch={cfg.get('batch')}（要求 <8），infer_chunk={cfg.get('infer_chunk')}（默认 16）")
    lines.append("")

    lines.append("## 4) 可复现性检查结论")
    lines.append(f"- 结论：{'通过' if repro_pass else '不通过'}")
    if not repro_pass:
        lines.append("- 差异项（run0 vs runN）：")
        for d in repro_diffs:
            lines.append(
                f"  - run={d.get('run')}, split={d.get('split')}, 统计项={d.get('统计项')}, "
                f"run0={d.get('run0')}, runN={d.get('runN')}, 差值={d.get('差值')}"
            )
    lines.append("")

    lines.append("## 5) 扫描记录（不是找最优）")
    lines.append(f"- 扫描参数组合数：{len(scan_param_grid)}")
    lines.append(format_params_list(scan_param_grid))
    lines.append("")

    lines.append("## 6) 输出文件说明")
    lines.append("- config.json：本次权重、数据路径列表、R 次复跑次数、扫描参数列表、默认参数与环境版本信息。")
    lines.append("- repro_check.csv：R 次复跑的关键计数表（每次每个 split 一行，含 all）。")
    lines.append("- scan_summary.csv：扫描每组参数的 summary（每组每个 split 一行，含 all）。")
    lines.append("")

    lines.append("## 7) 命令行示例")
    lines.append("```bash")
    lines.append("python /home/ubuntu/hpproject/yolo/analyze/code/p23_4_eval_repro_check.py \\")
    lines.append(f"  --weights {cfg.get('weights')} \\")
    for p in cfg.get("image_dir", []):
        lines.append(f"  --image_dir {p} \\")
    lines.append(f"  --out_root {cfg.get('out_root')} \\")
    lines.append(f"  --batch {cfg.get('batch')} --infer_chunk {cfg.get('infer_chunk')} \\")
    lines.append(
        f"  --conf {cfg.get('conf')} --tp_iou {cfg.get('tp_iou')} --nms_iou {cfg.get('nms_iou')} --max_det {cfg.get('max_det')} \\"
    )
    lines.append(f"  --repro_runs {cfg.get('repro_runs')}")
    lines.append("```")
    lines.append("")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()

    image_dirs = [p.resolve() for p in normalize_path_list(args.image_dir)]
    if not image_dirs:
        raise ValueError("至少需要提供一个 --image_dir")

    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    report_dir = make_report_dir(out_root)

    if int(args.batch) >= 8:
        raise ValueError(f"batch 必须 < 8，当前 batch={args.batch}")
    if int(args.infer_chunk) <= 0:
        raise ValueError(f"infer_chunk 必须 > 0，当前 infer_chunk={args.infer_chunk}")
    if int(args.repro_runs) <= 0:
        raise ValueError(f"repro_runs 必须 > 0，当前 repro_runs={args.repro_runs}")

    ensure_ultralytics()
    set_global_seed(int(args.seed))

    items, split_order, label_dirs = build_items(image_dirs)
    if not items:
        raise RuntimeError("未找到图像。请检查 --image_dir 是否正确。")

    # 扫描列表（未提供则使用默认单值）
    conf_list = [float(x) for x in (args.conf_list if args.conf_list is not None else [args.conf])]
    tp_list = [float(x) for x in (args.tp_iou_list if args.tp_iou_list is not None else [args.tp_iou])]
    nms_list = [float(x) for x in (args.nms_list if args.nms_list is not None else [args.nms_iou])]
    max_det_list = [int(x) for x in (args.max_det_list if args.max_det_list is not None else [args.max_det])]
    conf_list = [float(x) for x in _dedup_keep_order(conf_list)]
    tp_list = [float(x) for x in _dedup_keep_order(tp_list)]
    nms_list = [float(x) for x in _dedup_keep_order(nms_list)]
    max_det_list = [int(x) for x in _dedup_keep_order(max_det_list)]

    scan_param_grid = [(c, t, n, m) for c, t, n, m in product(conf_list, tp_list, nms_list, max_det_list)]

    weights_path = Path(args.weights).resolve()
    model = YOLO(str(weights_path))

    # (A) 固定参数可复现性：连续跑 R 次
    base_params = EvalParams(
        conf=float(args.conf),
        tp_iou=float(args.tp_iou),
        nms_iou=float(args.nms_iou),
        max_det=int(args.max_det),
        batch=int(args.batch),
        infer_chunk=int(args.infer_chunk),
        imgsz=int(args.imgsz),
        device=str(args.device),
    )

    repro_runs: List[Dict[str, Counts]] = []
    repro_rows: List[dict] = []
    for run_id in range(int(args.repro_runs)):
        st_map = evaluate_once(model, items, base_params, print_warnings=(run_id == 0))
        repro_runs.append(st_map)
        for split in (split_order + ["all"]):
            if split not in st_map:
                continue
            repro_rows.append(counts_to_row(run_id, split, base_params, st_map[split]))

    repro_pass, repro_diffs = compare_repro(repro_runs)
    write_csv(report_dir / "repro_check.csv", repro_rows)

    # (B) 扫描记录：每组参数跑 1 次（默认至少会跑 1 组：即默认参数）
    scan_rows: List[dict] = []
    base_key = (float(base_params.conf), float(base_params.tp_iou), float(base_params.nms_iou), int(base_params.max_det))
    base_st_map = repro_runs[0] if repro_runs else {}
    for scan_id, (c, t, n, m) in enumerate(scan_param_grid):
        p = EvalParams(
            conf=float(c),
            tp_iou=float(t),
            nms_iou=float(n),
            max_det=int(m),
            batch=int(args.batch),
            infer_chunk=int(args.infer_chunk),
            imgsz=int(args.imgsz),
            device=str(args.device),
        )
        if (float(c), float(t), float(n), int(m)) == base_key and base_st_map:
            st_map = base_st_map
        else:
            st_map = evaluate_once(model, items, p, print_warnings=False)
        for split in (split_order + ["all"]):
            if split not in st_map:
                continue
            scan_rows.append(counts_to_scan_row(scan_id, split, p, st_map[split]))
    write_csv(report_dir / "scan_summary.csv", scan_rows)

    # config.json（按需求字段 + 版本信息）
    cfg = {
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "weights": str(weights_path),
        "image_dir": [str(p) for p in image_dirs],
        "label_dir": [str(p) for p in label_dirs],
        "out_root": str(out_root),
        "report_dir": str(report_dir),
        "batch": int(args.batch),
        "infer_chunk": int(args.infer_chunk),
        "imgsz": int(args.imgsz),
        "device": str(args.device),
        "conf": float(args.conf),
        "tp_iou": float(args.tp_iou),
        "nms_iou": float(args.nms_iou),
        "max_det": int(args.max_det),
        "repro_runs": int(args.repro_runs),
        "scan": {
            "conf_list": conf_list,
            "tp_iou_list": tp_list,
            "nms_list": nms_list,
            "max_det_list": max_det_list,
            "param_grid_size": int(len(scan_param_grid)),
        },
        "split_rule": "split = image_dir 路径末级名（例如 val/test）",
        "eval_pipeline": "conf -> NMS -> max_det; one-to-one match @ tp_iou",
        "abbr": {
            "IoU": "交并比",
            "NMS": "非极大值抑制（重叠框去重）",
            "GT": "ground_truth（标注框）",
            "TP/FP/FN": "目标级命中/误报/漏检",
        },
        "versions": collect_versions(),
        "seed": int(args.seed),
    }
    with (report_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    used_hungarian = linear_sum_assignment is not None
    write_report_md(
        report_dir / "report.md",
        cfg,
        repro_pass,
        repro_diffs,
        scan_param_grid,
        used_hungarian=used_hungarian,
    )

    print(f"[OK] report_dir: {report_dir}")
    print(f"[OK] 可复现性：{'通过' if repro_pass else '不通过'}（R={int(args.repro_runs)}）")
    if not repro_pass:
        print("[FAIL] 差异项：")
        for d in repro_diffs[:20]:
            print(
                f"  run={d.get('run')}, split={d.get('split')}, 统计项={d.get('统计项')}, "
                f"run0={d.get('run0')}, runN={d.get('runN')}, 差值={d.get('差值')}"
            )
        if len(repro_diffs) > 20:
            print("  ...（其余差异已省略，详见 report.md）")


if __name__ == "__main__":
    main()
