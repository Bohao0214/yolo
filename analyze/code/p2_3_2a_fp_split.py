"""P2.3.2a 目标级误报（FP）结构性拆分（YOLO）。

P2.3.0 评估口径冻结：
- 置信度阈值（conf）过滤 -> NMS（非极大值抑制，重叠框去重）-> max_det
- 匹配阈值 tp_iou（IoU=交并比），目标级一对一匹配

术语（首次出现给出中文解释）：
- GT = ground_truth（标注框）
- FP/TP/FN = 误报/命中/漏检
- IoU = 交并比
- NMS = 非极大值抑制（重叠框去重）

FP 拆分定义（必须写入 run_args.json）：
- unmatched FP（真实误判）：预测框在一对一匹配后未匹配到任何 GT，且与任意 GT 的最大 IoU < tp_iou。
- pred_dup（预测冗余）：同一 GT 被 >=2 个预测框以 IoU >= tp_iou 覆盖，最多 1 个进入 TP，其余预测框计为 FP 并标记 pred_dup。
- both：同时满足 unmatched 与 pred_dup（理论上应为 0，如发生则单列）。
- gt_dup（可选计数）：某个预测框同时覆盖多个 GT，导致 GT 侧潜在漏检风险。

示例：
python /home/ubuntu/project/deduibi/yolo/analyze/code/p2_3_2a_fp_split.py \
  --weights /home/ubuntu/project/deduibi/yolo/models/defect/exp_260202_base/best/best.pt \
  --image_dir /home/ubuntu/project/deduibi/yolo/dataset/yolo/datasetm6c/images/val \
  --image_dir /home/ubuntu/project/deduibi/yolo/dataset/yolo/datasetm6c/images/test \
  --out_root /home/ubuntu/project/deduibi/yolo/analyze/result \
  --conf 0.3 --tp_iou 0.2 --nms_iou 0.6 --max_det 20 \
  --batch 4 --infer_chunk 16

"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import gc
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from ultralytics import YOLO
except Exception as exc:  # pragma: no cover
    YOLO = None  # type: ignore
    ULTRALYTICS_IMPORT_ERROR = exc
else:
    ULTRALYTICS_IMPORT_ERROR = None


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def ensure_ultralytics() -> None:
    if YOLO is None:
        raise ImportError(
            "Failed to import ultralytics. Please install ultralytics in the environment. "
            f"Original error: {ULTRALYTICS_IMPORT_ERROR}"
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="P2.3.2a FP 结构性拆分（unmatched / pred_dup）.")
    p.add_argument("--weights", type=str, required=True)
    p.add_argument(
        "--image_dir",
        type=str,
        required=True,
        action="append",
        help="Dataset image directory (val/test). Can be provided multiple times or comma-separated.",
    )
    p.add_argument("--out_root", type=str, required=True)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--infer_chunk", type=int, default=16)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", type=str, default="")
    p.add_argument("--sample_max", type=int, default=200, help="Max FP samples to save (0 to disable).")

    # P2.3.0 postprocess
    p.add_argument("--conf", type=float, default=0.3)
    p.add_argument("--tp_iou", type=float, default=0.2)
    p.add_argument("--nms_iou", type=float, default=0.6)
    p.add_argument("--max_det", type=int, default=20)
    return p.parse_args()


def normalize_path_list(raw: Optional[Sequence[str]]) -> List[Path]:
    if not raw:
        return []
    out: List[Path] = []
    for item in raw:
        if not item:
            continue
        for part in str(item).split(","):
            p = part.strip()
            if not p:
                continue
            out.append(Path(p))
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


def xywhn_to_xyxy(xc: float, yc: float, w: float, h: float, img_w: int, img_h: int) -> Tuple[float, float, float, float]:
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


def safe_ratio(num: int, den: int) -> float:
    return float(num) / float(den) if den > 0 else 0.0


def write_csv(path: Path, rows: List[dict], headers: List[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def init_stats() -> Dict[str, int]:
    return {
        "fp_total": 0,
        "fp_unmatched": 0,
        "fp_pred_dup": 0,
        "fp_both": 0,
        "fp_other": 0,
        "gt_count": 0,
        "pred_count": 0,
        "gt_dup": 0,
        "images": 0,
        "missing_label_files": 0,
    }


def format_box(box: Tuple[float, float, float, float]) -> str:
    return f"{box[0]:.1f},{box[1]:.1f},{box[2]:.1f},{box[3]:.1f}"


def main() -> None:
    args = parse_args()
    ensure_ultralytics()

    if int(args.batch) >= 8:
        raise ValueError("--batch 必须 < 8")
    if int(args.infer_chunk) > 64:
        raise ValueError("--infer_chunk 不能太高（建议 <= 64）")

    image_dirs = normalize_path_list(args.image_dir)
    if not image_dirs:
        raise ValueError("image_dir 为空")

    for d in image_dirs:
        if not d.exists():
            raise FileNotFoundError(f"image_dir not found: {d}")

    label_dirs = [infer_label_dir(d) for d in image_dirs]
    label_dir_missing = [not d.exists() for d in label_dirs]
    for d, missing in zip(label_dirs, label_dir_missing):
        if missing:
            print(f"[WARN] label_dir 不存在，将按缺失标签处理: {d}")

    report_dir = make_report_dir(Path(args.out_root))

    run_args = {
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "weights": str(args.weights),
        "image_dir": [str(p) for p in image_dirs],
        "label_dir": [str(p) for p in label_dirs],
        "out_root": str(args.out_root),
        "report_dir": str(report_dir),
        "conf": float(args.conf),
        "nms_iou": float(args.nms_iou),
        "max_det": int(args.max_det),
        "tp_iou": float(args.tp_iou),
        "batch": int(args.batch),
        "infer_chunk": int(args.infer_chunk),
        "imgsz": int(args.imgsz),
        "device": str(args.device),
        "sample_max": int(args.sample_max),
        "eval_pipeline": "conf -> NMS -> max_det; one-to-one match @ tp_iou",
        "definitions": {
            "unmatched_fp": "预测框在一对一匹配后未匹配到任何 GT，且与任意 GT 的最大 IoU < tp_iou。",
            "pred_dup": "同一 GT 被 >=2 个预测框以 IoU >= tp_iou 覆盖，最多 1 个进入 TP，其余预测框计为 FP 并标记 pred_dup。",
            "both": "同时满足 unmatched 与 pred_dup（理论上应为 0，如发生则单列）。",
            "both_policy": "both 单列统计，不计入 unmatched/pred_dup。",
            "gt_dup": "某个预测框同时覆盖多个 GT，导致 GT 侧潜在漏检风险（本次仅记录计数）。",
        },
        "source_name_rule": "source_name = image_dir 的路径末级名（如 val/test）",
        "abbr": {
            "IoU": "交并比",
            "NMS": "非极大值抑制（重叠框去重）",
            "GT": "标注框",
            "FP/TP/FN": "误报/命中/漏检",
        },
    }
    with (report_dir / "run_args.json").open("w", encoding="utf-8") as f:
        json.dump(run_args, f, ensure_ascii=False, indent=2)

    stats_all = init_stats()
    stats_by_source: Dict[str, Dict[str, int]] = {}
    source_order: List[str] = []

    items: List[Tuple[Path, Path, str]] = []
    for img_dir, lbl_dir in zip(image_dirs, label_dirs):
        source_name = img_dir.name
        if source_name not in stats_by_source:
            stats_by_source[source_name] = init_stats()
            source_order.append(source_name)
        img_paths = list_images(img_dir)
        if not img_paths:
            print(f"[WARN] image_dir 为空: {img_dir}")
        for img_path in img_paths:
            items.append((img_path, lbl_dir, source_name))

    total_images = len(items)
    if total_images == 0:
        raise RuntimeError("No images found.")

    model = YOLO(str(args.weights))
    sample_rows: List[dict] = []
    sample_max = int(args.sample_max)

    try:
        import torch
    except Exception:
        torch = None  # type: ignore

    for start in range(0, total_images, int(args.infer_chunk)):
        chunk = items[start : start + int(args.infer_chunk)]
        sources = [str(p[0]) for p in chunk]
        results = model.predict(
            source=sources,
            imgsz=int(args.imgsz),
            conf=float(args.conf),
            iou=float(args.nms_iou),
            max_det=int(args.max_det),
            save=False,
            verbose=False,
            batch=int(args.batch),
            device=str(args.device) if args.device else None,
            stream=True,
        )
        result_map = {str(res.path): res for res in results}

        for img_path, label_dir, source_name in chunk:
            res = result_map.get(str(img_path))
            if res is None:
                print(f"[WARN] 推理结果缺失: {img_path}")
                continue

            if getattr(res, "orig_shape", None):
                h, w = int(res.orig_shape[0]), int(res.orig_shape[1])
            else:
                # fallback: 仅在极少数情况下使用
                import cv2

                img = cv2.imread(str(img_path))
                if img is None:
                    raise FileNotFoundError(f"Failed to read image: {img_path}")
                h, w = img.shape[:2]
                del img

            label_path = label_dir / f"{img_path.stem}.txt"
            gt_boxes = load_labels(label_path, w, h)

            stats_all["images"] += 1
            stats_by_source[source_name]["images"] += 1
            stats_all["gt_count"] += len(gt_boxes)
            stats_by_source[source_name]["gt_count"] += len(gt_boxes)
            if not label_path.exists():
                stats_all["missing_label_files"] += 1
                stats_by_source[source_name]["missing_label_files"] += 1

            boxes = res.boxes
            preds_xyxy: List[Tuple[float, float, float, float]] = []
            scores: List[float] = []
            if boxes is not None:
                if getattr(boxes, "xyxyn", None) is not None:
                    xyxyn = boxes.xyxyn.detach().cpu().numpy()
                    xyxy = np.zeros_like(xyxyn)
                    xyxy[:, [0, 2]] = xyxyn[:, [0, 2]] * float(w)
                    xyxy[:, [1, 3]] = xyxyn[:, [1, 3]] * float(h)
                else:
                    xyxy = boxes.xyxy.detach().cpu().numpy() if boxes.xyxy is not None else np.zeros((0, 4))
                confs = boxes.conf.detach().cpu().numpy() if getattr(boxes, "conf", None) is not None else np.zeros(len(xyxy))
                for box, score in zip(xyxy, confs):
                    preds_xyxy.append((float(box[0]), float(box[1]), float(box[2]), float(box[3])))
                    scores.append(float(score))

            stats_all["pred_count"] += len(preds_xyxy)
            stats_by_source[source_name]["pred_count"] += len(preds_xyxy)

            gt_arr = np.array(gt_boxes, dtype=np.float32) if gt_boxes else np.zeros((0, 4), dtype=np.float32)
            pred_arr = np.array(preds_xyxy, dtype=np.float32) if preds_xyxy else np.zeros((0, 4), dtype=np.float32)
            iou_mat = compute_iou_matrix(gt_arr, pred_arr)

            assigned_pred_for_gt: Dict[int, int] = {}
            assigned_gt_for_pred: Dict[int, int] = {}
            if iou_mat.size > 0:
                pairs: List[Tuple[float, int, int]] = []
                for gi in range(iou_mat.shape[0]):
                    for pi in range(iou_mat.shape[1]):
                        v = float(iou_mat[gi, pi])
                        if v >= float(args.tp_iou):
                            pairs.append((v, gi, pi))
                pairs.sort(reverse=True, key=lambda x: x[0])
                for v, gi, pi in pairs:
                    if gi in assigned_pred_for_gt or pi in assigned_gt_for_pred:
                        continue
                    assigned_pred_for_gt[gi] = pi
                    assigned_gt_for_pred[pi] = gi

            pred_dup: set[int] = set()
            if iou_mat.size > 0:
                for gi in range(iou_mat.shape[0]):
                    cand = np.where(iou_mat[gi] >= float(args.tp_iou))[0].tolist()
                    if len(cand) < 2:
                        continue
                    if gi in assigned_pred_for_gt:
                        keep_pi = assigned_pred_for_gt[gi]
                    else:
                        keep_pi = max(cand, key=lambda pi: (float(iou_mat[gi, pi]), float(scores[pi] if pi < len(scores) else 0.0)))
                    for pi in cand:
                        if pi != keep_pi:
                            pred_dup.add(pi)

            gt_dup = set()
            if iou_mat.size > 0:
                for gi in range(iou_mat.shape[0]):
                    if gi in assigned_pred_for_gt:
                        continue
                    cand = np.where(iou_mat[gi] >= float(args.tp_iou))[0].tolist()
                    for pi in cand:
                        if pi in assigned_gt_for_pred:
                            gt_dup.add(gi)
                            break

            for pi, pred in enumerate(preds_xyxy):
                if pi in assigned_gt_for_pred:
                    continue
                max_iou = float(iou_mat[:, pi].max()) if iou_mat.size > 0 else 0.0
                is_unmatched = max_iou < float(args.tp_iou)
                is_pred_dup = pi in pred_dup
                if is_unmatched and is_pred_dup:
                    tag = "both"
                elif is_unmatched:
                    tag = "unmatched"
                elif is_pred_dup:
                    tag = "pred_dup"
                else:
                    tag = "other"

                stats_all["fp_total"] += 1
                stats_by_source[source_name]["fp_total"] += 1
                if tag == "unmatched":
                    stats_all["fp_unmatched"] += 1
                    stats_by_source[source_name]["fp_unmatched"] += 1
                elif tag == "pred_dup":
                    stats_all["fp_pred_dup"] += 1
                    stats_by_source[source_name]["fp_pred_dup"] += 1
                elif tag == "both":
                    stats_all["fp_both"] += 1
                    stats_by_source[source_name]["fp_both"] += 1
                else:
                    stats_all["fp_other"] += 1
                    stats_by_source[source_name]["fp_other"] += 1

                if sample_max > 0 and len(sample_rows) < sample_max:
                    sample_rows.append(
                        {
                            "image_id": img_path.stem,
                            "pred_box": format_box(pred),
                            "pred_score": f"{scores[pi]:.4f}" if pi < len(scores) else "",
                            "max_iou": f"{max_iou:.4f}",
                            "fp_tag": tag,
                            "source_name": source_name,
                        }
                    )

            stats_all["gt_dup"] += len(gt_dup)
            stats_by_source[source_name]["gt_dup"] += len(gt_dup)

            del gt_arr, pred_arr, iou_mat

        del result_map, results, sources, chunk
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary_rows: List[dict] = []
    sources = ["all"] + source_order
    stats_map = {"all": stats_all, **stats_by_source}
    for name in sources:
        s = stats_map[name]
        fp_total = s["fp_total"]
        summary_rows.append(
            {
                "source_name": name,
                "FP_total": fp_total,
                "FP_unmatched": s["fp_unmatched"],
                "FP_pred_dup": s["fp_pred_dup"],
                "FP_both": s["fp_both"],
                "ratio_unmatched": f"{safe_ratio(s['fp_unmatched'], fp_total):.6f}",
                "ratio_pred_dup": f"{safe_ratio(s['fp_pred_dup'], fp_total):.6f}",
                "GT_count": s["gt_count"],
            }
        )

    summary_path = report_dir / "p2_3_2a_fp_split_summary.csv"
    write_csv(
        summary_path,
        summary_rows,
        [
            "source_name",
            "FP_total",
            "FP_unmatched",
            "FP_pred_dup",
            "FP_both",
            "ratio_unmatched",
            "ratio_pred_dup",
            "GT_count",
        ],
    )

    if sample_rows:
        samples_path = report_dir / "p2_3_2a_fp_samples.csv"
        write_csv(
            samples_path,
            sample_rows,
            ["image_id", "pred_box", "pred_score", "max_iou", "fp_tag", "source_name"],
        )

    warnings: List[str] = []
    for name in sources:
        s = stats_map[name]
        if s["gt_count"] == 0:
            warnings.append(f"- [警告] {name} 的 GT_count=0，可能存在标签缺失或数据源无标注。")
        if s["missing_label_files"] > 0:
            warnings.append(f"- [提醒] {name} 缺失标签文件数: {s['missing_label_files']}")
        if s["fp_other"] > 0:
            warnings.append(f"- [提醒] {name} 出现 FP_other={s['fp_other']}（不属于 unmatched/pred_dup/both）。")

    notes_path = report_dir / "p2_3_2a_fp_split_notes.md"
    lines: List[str] = []
    lines.append("# P2.3.2a 目标级误报（FP）结构性拆分")
    lines.append("")
    lines.append(f"- created_at: {dt.datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"- report_dir: {report_dir}")
    lines.append("")
    lines.append("## 结果解读")
    lines.append(
        "本报告在 P2.3.0 口径下统计 FP，并将其拆分为 unmatched（真实误判）与 pred_dup（预测冗余）。"
        "ratio_* 为对应 FP 占 FP_total 的比例；若存在 both，则双方占比不一定相加为 1。"
    )
    lines.append(
        "pred_dup 反映同一 GT 被多个预测框重复覆盖的冗余；unmatched 反映与任意 GT 的最大 IoU < tp_iou 的背景误判。"
    )
    if warnings:
        lines.append("")
        lines.append("## 提醒")
        lines.extend(warnings)

    with notes_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    if warnings:
        print("[WARN] 统计提醒：")
        for w in warnings:
            print(w)

    print(f"[DONE] report_dir: {report_dir}")
    print(f"[DONE] summary: {summary_path}")
    if sample_rows:
        print(f"[DONE] samples: {report_dir / 'p2_3_2a_fp_samples.csv'}")


if __name__ == "__main__":
    main()
