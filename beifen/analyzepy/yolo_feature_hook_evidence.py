import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_images(image_dir: Path) -> List[Path]:
    return sorted([p for p in image_dir.rglob("*") if p.suffix.lower() in IMG_EXTS and p.is_file()])


def infer_label_dir(image_dir: Path) -> Path:
    parts = list(image_dir.parts)
    if "images" in parts:
        idx = parts.index("images")
        return Path(*parts[:idx], "labels", *parts[idx + 1 :])
    if image_dir.name in {"train", "val", "test"}:
        return image_dir.parent.parent / "labels" / image_dir.name
    return image_dir.parent / "labels"


def load_yolo_labels(label_path: Path, img_w: int, img_h: int) -> List[List[float]]:
    if not label_path.exists():
        return []
    boxes = []
    with open(label_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            _, xc, yc, bw, bh = parts[:5]
            try:
                xc_f, yc_f, bw_f, bh_f = float(xc), float(yc), float(bw), float(bh)
            except ValueError:
                continue
            x1 = (xc_f - bw_f / 2) * img_w
            y1 = (yc_f - bh_f / 2) * img_h
            x2 = (xc_f + bw_f / 2) * img_w
            y2 = (yc_f + bh_f / 2) * img_h
            boxes.append([x1, y1, x2, y2])
    return boxes


def compute_iou_matrix(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    if gt.size == 0 or pred.size == 0:
        return np.zeros((gt.shape[0], pred.shape[0]), dtype=np.float32)
    ix1 = np.maximum(gt[:, None, 0], pred[None, :, 0])
    iy1 = np.maximum(gt[:, None, 1], pred[None, :, 1])
    ix2 = np.minimum(gt[:, None, 2], pred[None, :, 2])
    iy2 = np.minimum(gt[:, None, 3], pred[None, :, 3])
    iw = np.maximum(0, ix2 - ix1)
    ih = np.maximum(0, iy2 - iy1)
    inter = iw * ih
    gt_area = (gt[:, 2] - gt[:, 0]) * (gt[:, 3] - gt[:, 1])
    pred_area = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
    union = gt_area[:, None] + pred_area[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


def hungarian_assign(cost: np.ndarray) -> np.ndarray:
    n0, m0 = cost.shape
    transposed = False
    if n0 > m0:
        cost = cost.T
        transposed = True
    n, m = cost.shape
    u = np.zeros(n + 1, dtype=np.float32)
    v = np.zeros(m + 1, dtype=np.float32)
    p = np.zeros(m + 1, dtype=int)
    way = np.zeros(m + 1, dtype=int)
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
                if not used[j]:
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
    assignment = -np.ones(n, dtype=int)
    for j in range(1, m + 1):
        if p[j] > 0:
            assignment[p[j] - 1] = j - 1
    if not transposed:
        return assignment
    assignment_orig = -np.ones(n0, dtype=int)
    for row_t, col_t in enumerate(assignment):
        if col_t != -1:
            orig_row = col_t
            orig_col = row_t
            assignment_orig[orig_row] = orig_col
    return assignment_orig


def hungarian_match(gt: np.ndarray, pred: np.ndarray, iou_candidate: float, iou_match: float) -> List[Tuple[int, int, float]]:
    if gt.size == 0 or pred.size == 0:
        return []
    iou_mat = compute_iou_matrix(gt, pred)
    cost = 1.0 - iou_mat
    cost[iou_mat < iou_candidate] = 1e6
    assignment = hungarian_assign(cost)
    matched = []
    for gi, pi in enumerate(assignment):
        if pi != -1 and iou_mat[gi, pi] >= iou_match:
            matched.append((gi, pi, float(iou_mat[gi, pi])))
    return matched


def letterbox_image(img: np.ndarray, size: int) -> Tuple[np.ndarray, float, int, int, int, int]:
    h, w = img.shape[:2]
    r = min(size / w, size / h)
    new_w, new_h = int(round(w * r)), int(round(h * r))
    resized = cv2.resize(img, (new_w, new_h))
    canvas = np.full((size, size, 3), 114, dtype=img.dtype)
    pad_x = (size - new_w) // 2
    pad_y = (size - new_h) // 2
    canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized
    return canvas, r, pad_x, pad_y, new_w, new_h


def img_to_tensor(img: np.ndarray, device: torch.device) -> torch.Tensor:
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(img_rgb).float() / 255.0
    t = t.permute(2, 0, 1).unsqueeze(0).to(device)
    return t


def run_inference(
    model: YOLO,
    images: List[Path],
    imgsz: int,
    conf: float,
    nms_iou: float,
    batch: int,
    device: str,
    half: bool,
) -> Dict[str, List[Dict]]:
    def _run(run_device: str, run_half: bool, run_batch: int) -> Dict[str, List[Dict]]:
        results = model.predict(
            source=[str(p) for p in images],
            imgsz=int(imgsz),
            conf=float(conf),
            iou=float(nms_iou),
            save=False,
            verbose=False,
            batch=int(run_batch),
            device=run_device if run_device else None,
            half=run_half,
            stream=True,
        )
        preds: Dict[str, List[Dict]] = {}
        for res in results:
            img_path = Path(res.path)
            items = []
            if res.boxes is not None:
                xyxy = res.boxes.xyxy.cpu().numpy()
                confs = res.boxes.conf.cpu().numpy()
                for b, c in zip(xyxy, confs):
                    items.append({"xyxy": b.tolist(), "conf": float(c)})
            preds[str(img_path)] = items
        if hasattr(model, "predictor"):
            model.predictor = None
        return preds

    try:
        return _run(device, half, batch)
    except Exception as exc:
        if device == "cpu":
            raise
        if "out of memory" not in str(exc).lower():
            raise
        print("CUDA OOM in inference, retrying on CPU with batch=1.")
        return _run("cpu", False, 1)


def find_latest_report(out_root: Path) -> Path:
    candidates = [p for p in out_root.iterdir() if p.is_dir() and p.name.startswith("report_")]
    if not candidates:
        raise RuntimeError(f"No report_*/ found under {out_root}")
    return sorted(candidates, key=lambda p: p.name)[-1]


def parse_topk_csv(path: Path) -> List[Dict]:
    items = []
    if not path.exists():
        return items
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            boxes = json.loads(row.get("boxes", "[]"))
            data = dict(row)
            data["image_path"] = row.get("image_path", "")
            data["boxes"] = boxes
            items.append(data)
    return items


def select_fn_samples(items: List[Dict], max_n: int) -> List[Dict]:
    samples = []
    for row in items[:max_n]:
        img_path = row.get("image_path", "")
        gt_boxes = [b["xyxy"] for b in row.get("boxes", []) if b.get("type") == "gt"]
        pred_boxes = [b["xyxy"] for b in row.get("boxes", []) if b.get("type") == "pred"]
        if img_path and gt_boxes:
            samples.append(
                {
                    "image": img_path,
                    "gt_boxes": gt_boxes,
                    "pred_boxes": pred_boxes,
                    "label": "FN",
                }
            )
    return samples


def fp_label(fp_type: str) -> str:
    if fp_type == "FP_image":
        return "image"
    if fp_type == "FP_strict":
        return "strict"
    if fp_type == "FP_near":
        return "near"
    return "fp"


def select_fp_samples(items: List[Dict], max_n: int) -> List[Dict]:
    samples = []
    for row in items[:max_n]:
        img_path = row.get("image_path", "")
        pred_boxes = [b["xyxy"] for b in row.get("boxes", []) if b.get("type") == "pred"]
        gt_boxes = [b["xyxy"] for b in row.get("boxes", []) if b.get("type") == "gt"]
        fp_type = row.get("fp_type", "")
        if img_path and pred_boxes:
            samples.append(
                {
                    "image": img_path,
                    "gt_boxes": gt_boxes,
                    "pred_boxes": pred_boxes,
                    "label": fp_label(fp_type),
                }
            )
    return samples


def collect_tp_samples(
    model: YOLO,
    images: List[Path],
    image_dir: Path,
    label_dir: Path,
    imgsz: int,
    conf: float,
    nms_iou: float,
    iou_match: float,
    iou_candidate: float,
    batch: int,
    device: str,
    half: bool,
    max_samples: int,
) -> List[Dict]:
    collected: List[Dict] = []
    batch = max(1, int(batch))
    for i in range(0, len(images), batch):
        chunk = images[i : i + batch]
        preds = run_inference(model, chunk, imgsz, conf, nms_iou, batch, device, half)
        for img_path in chunk:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]
            try:
                rel = img_path.relative_to(image_dir)
                label_path = (label_dir / rel).with_suffix(".txt")
            except Exception:
                label_path = (label_dir / img_path.name).with_suffix(".txt")
            gt_boxes = load_yolo_labels(label_path, w, h)
            if not gt_boxes:
                continue
            pred_items = preds.get(str(img_path), [])
            pred_arr = np.array([p["xyxy"] for p in pred_items], dtype=np.float32) if pred_items else np.zeros((0, 4))
            gt_arr = np.array(gt_boxes, dtype=np.float32)
            matched = hungarian_match(gt_arr, pred_arr, iou_candidate, iou_match)
            for gi, pi, iou in matched:
                collected.append(
                    {
                        "image": str(img_path),
                        "gt_boxes": [gt_boxes[gi]],
                        "pred_boxes": [pred_items[pi]["xyxy"]],
                        "iou": float(iou),
                        "label": "TP",
                    }
                )
        if len(collected) >= max_samples * 5:
            break
    if not collected:
        return []
    collected = sorted(collected, key=lambda x: -x["iou"])
    return collected[:max_samples]


def select_layers(
    model: torch.nn.Module,
    sample_tensor: torch.Tensor,
    keywords: List[str],
    max_candidates: int = 30,
    max_layers: int = 3,
) -> Tuple[List[str], Dict[str, str]]:
    candidates = [(n, m) for n, m in model.named_modules() if any(k in n.lower() for k in keywords)]
    if len(candidates) > max_candidates:
        candidates = candidates[-max_candidates:]

    feats: Dict[str, torch.Tensor] = {}
    hooks = []

    def make_hook(name: str):
        def fn(_, __, y):
            feats[name] = y

        return fn

    for name, module in candidates:
        hooks.append(module.register_forward_hook(make_hook(name)))

    with torch.no_grad():
        _ = model(sample_tensor)

    for h in hooks:
        h.remove()

    outputs = []
    key_to_module: Dict[str, str] = {}
    for name, out in feats.items():
        if torch.is_tensor(out) and out.ndim == 4:
            outputs.append((name, out))
            key_to_module[name] = name
        elif isinstance(out, (list, tuple)):
            for i, o in enumerate(out):
                if torch.is_tensor(o) and o.ndim == 4:
                    key = f"{name}.{i}"
                    outputs.append((key, o))
                    key_to_module[key] = name

    groups: Dict[Tuple[int, int], List[Tuple[str, torch.Tensor]]] = {}
    for key, t in outputs:
        h, w = int(t.shape[-2]), int(t.shape[-1])
        groups.setdefault((h, w), []).append((key, t))

    selected_keys = []
    for (h, w), items in sorted(groups.items(), key=lambda kv: -(kv[0][0] * kv[0][1])):
        key, _ = max(items, key=lambda it: it[1].shape[1])
        selected_keys.append(key)
        if len(selected_keys) >= max_layers:
            break

    return selected_keys, key_to_module


def compute_heatmap(
    feat: torch.Tensor,
    imgsz: int,
    pad_x: int,
    pad_y: int,
    new_w: int,
    new_h: int,
    orig_w: int,
    orig_h: int,
) -> np.ndarray:
    if feat.ndim == 4:
        feat = feat[0]
    heat = torch.mean(torch.abs(feat), dim=0).cpu().numpy()
    heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-6)
    heat_lb = cv2.resize(heat, (imgsz, imgsz))
    heat_crop = heat_lb[pad_y : pad_y + new_h, pad_x : pad_x + new_w]
    heat_orig = cv2.resize(heat_crop, (orig_w, orig_h))
    return heat_orig


def overlay_heatmap(img: np.ndarray, heat: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    heat_u8 = np.clip(heat * 255, 0, 255).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    return cv2.addWeighted(img, 1 - alpha, heat_color, alpha, 0)


def clip_box(xyxy: List[float], w: int, h: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = xyxy
    x1 = max(0, min(int(round(x1)), w - 1))
    y1 = max(0, min(int(round(y1)), h - 1))
    x2 = max(0, min(int(round(x2)), w))
    y2 = max(0, min(int(round(y2)), h))
    if x2 <= x1:
        x2 = min(w, x1 + 1)
    if y2 <= y1:
        y2 = min(h, y1 + 1)
    return x1, y1, x2, y2


def compute_ratio(heat: np.ndarray, box: List[float]) -> float:
    h, w = heat.shape[:2]
    x1, y1, x2, y2 = clip_box(box, w, h)
    inside = heat[y1:y2, x1:x2]
    if inside.size == 0:
        return float("nan")
    outside_mask = np.ones_like(heat, dtype=bool)
    outside_mask[y1:y2, x1:x2] = False
    outside = heat[outside_mask]
    eps = 1e-6
    return float((inside.mean() + eps) / (outside.mean() + eps))


def build_grid(
    samples: List[Dict],
    layer_keys: List[str],
    overlays: Dict[Tuple[int, str], np.ndarray],
    out_path: Path,
    title: str,
    tile_size: int = 320,
) -> None:
    if not samples or not layer_keys:
        return
    rows = len(layer_keys)
    cols = len(samples)
    canvas = np.zeros((rows * tile_size + 30, cols * tile_size, 3), dtype=np.uint8)
    cv2.putText(canvas, title, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    for r, layer in enumerate(layer_keys):
        for c, sample in enumerate(samples):
            key = (c, layer)
            if key not in overlays:
                continue
            img = cv2.resize(overlays[key], (tile_size, tile_size))
            y0 = r * tile_size + 30
            x0 = c * tile_size
            canvas[y0 : y0 + tile_size, x0 : x0 + tile_size] = img
            short = ".".join(layer.split(".")[-2:])
            cv2.putText(canvas, short, (x0 + 4, y0 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            label = Path(sample["image"]).stem[:12]
            cv2.putText(
                canvas,
                label,
                (x0 + 4, y0 + tile_size - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
            )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), canvas)


def write_hook_summary(
    out_path: Path,
    layer_keys: List[str],
    tp_ratios: Dict[str, List[float]],
    fn_ratios: Dict[str, List[float]],
    fp_pred_ratios: Dict[str, List[float]],
) -> None:
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "layer",
                "tp_ratio_gt_p10",
                "tp_ratio_gt_p50",
                "tp_ratio_gt_p90",
                "fn_ratio_gt_p10",
                "fn_ratio_gt_p50",
                "fn_ratio_gt_p90",
                "fp_ratio_pred_p10",
                "fp_ratio_pred_p50",
                "fp_ratio_pred_p90",
                "tp_count",
                "fn_count",
                "fp_count",
            ]
        )
        for layer in layer_keys:
            tp_vals = tp_ratios.get(layer, [])
            fn_vals = fn_ratios.get(layer, [])
            fp_vals = fp_pred_ratios.get(layer, [])
            tp_p10 = np.percentile(tp_vals, 10) if tp_vals else float("nan")
            tp_p50 = np.percentile(tp_vals, 50) if tp_vals else float("nan")
            tp_p90 = np.percentile(tp_vals, 90) if tp_vals else float("nan")
            fn_p10 = np.percentile(fn_vals, 10) if fn_vals else float("nan")
            fn_p50 = np.percentile(fn_vals, 50) if fn_vals else float("nan")
            fn_p90 = np.percentile(fn_vals, 90) if fn_vals else float("nan")
            fp_p10 = np.percentile(fp_vals, 10) if fp_vals else float("nan")
            fp_p50 = np.percentile(fp_vals, 50) if fp_vals else float("nan")
            fp_p90 = np.percentile(fp_vals, 90) if fp_vals else float("nan")
            w.writerow(
                [
                    layer,
                    f"{tp_p10:.6f}" if not math.isnan(tp_p10) else "",
                    f"{tp_p50:.6f}" if not math.isnan(tp_p50) else "",
                    f"{tp_p90:.6f}" if not math.isnan(tp_p90) else "",
                    f"{fn_p10:.6f}" if not math.isnan(fn_p10) else "",
                    f"{fn_p50:.6f}" if not math.isnan(fn_p50) else "",
                    f"{fn_p90:.6f}" if not math.isnan(fn_p90) else "",
                    f"{fp_p10:.6f}" if not math.isnan(fp_p10) else "",
                    f"{fp_p50:.6f}" if not math.isnan(fp_p50) else "",
                    f"{fp_p90:.6f}" if not math.isnan(fp_p90) else "",
                    len(tp_vals),
                    len(fn_vals),
                    len(fp_vals),
                ]
            )


def load_base_metrics(report_dir: Path) -> Dict[str, float]:
    base_path = report_dir / "metrics" / "base_metrics.csv"
    if not base_path.exists():
        return {}
    with open(base_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        return {}
    row = rows[0]
    out = {}
    for key, value in row.items():
        try:
            out[key] = float(value)
        except Exception:
            continue
    return out


def update_report(
    report_path: Path,
    layer_keys: List[str],
    tp_ratios: Dict[str, List[float]],
    fn_ratios: Dict[str, List[float]],
    fp_pred_ratios: Dict[str, List[float]],
    base_metrics: Dict[str, float],
) -> None:
    collapse_layers = []
    for layer in layer_keys:
        tp_vals = tp_ratios.get(layer, [])
        fn_vals = fn_ratios.get(layer, [])
        if not tp_vals or not fn_vals:
            continue
        tp_p50 = np.percentile(tp_vals, 50)
        fn_p50 = np.percentile(fn_vals, 50)
        if tp_p50 > 0 and fn_p50 / tp_p50 < 0.7:
            collapse_layers.append(layer)

    lines = []
    lines.append("- 说明：heatmap 是 feature saliency，不等价于最终出框。\n")
    lines.append("- 已生成 hook/hook_summary.csv 与热图拼图：hook/tp_heatmaps.png、hook/fn_heatmaps.png。\n")
    if collapse_layers:
        layers_str = ", ".join(collapse_layers)
        lines.append(f"- E1：FN 在层 {layers_str} 的 ratio_gt 中位数显著低于 TP，提示表征衰减/弱信号不可见。\n")
        lines.append("- 路线建议：优先尝试提高有效分辨率（切片/更高输入）或调整检测头以增强高层表征。\n")
    else:
        lines.append("- 未观察到明显层级坍塌（FN ratio_gt 与 TP 接近）。\n")

    fp_strict = base_metrics.get("fp_strict_per_img", 0.0)
    fp_near = base_metrics.get("fp_near_per_img", 0.0)
    if not collapse_layers and fp_strict > fp_near:
        lines.append("- E2：TP/FN ratio_gt 接近但 FP_strict 占比更高，支持伪异常驱动（需 hard negatives）。\n")

    if report_path.exists():
        content = report_path.read_text(encoding="utf-8")
        placeholder = "- 待生成 hook/hook_summary.csv 与热图拼图后补充结论。"
        if placeholder in content:
            content = content.replace(placeholder, "".join(lines).strip())
            report_path.write_text(content, encoding="utf-8")
        else:
            report_path.write_text(content + "\n\n## Hook 证据（Work2）\n" + "".join(lines), encoding="utf-8")
    else:
        report_path.write_text("# YOLO Eval Report\n\n## Hook 证据（Work2）\n" + "".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="/home/ubuntu/project/deduibi/yolo/models/defect/exp_2601261106/best/best.pt")
    parser.add_argument("--image_dir", type=str, default="/home/ubuntu/project/deduibi/yolo/dataset/yolo/datasetm6c/images/val")
    parser.add_argument("--out_root", type=str, default="/home/ubuntu/project/deduibi/yolo/analysis")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--conf", type=float, default=0.2)
    parser.add_argument("--nms_iou", type=float, default=0.5)
    parser.add_argument("--iou_match", type=float, default=0.5)
    parser.add_argument("--iou_candidate", type=float, default=0.1)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--report_dir", type=str, default="")
    parser.add_argument("--hook_n", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_dir = Path(args.image_dir)
    label_dir = infer_label_dir(image_dir)
    if not image_dir.exists():
        raise FileNotFoundError(f"image_dir not found: {image_dir}")
    if not label_dir.exists():
        raise FileNotFoundError(f"label_dir not found: {label_dir}")

    out_root = Path(args.out_root)
    report_dir = Path(args.report_dir) if args.report_dir else find_latest_report(out_root)
    topk_dir = report_dir / "topk"
    hook_dir = report_dir / "hook"
    hook_dir.mkdir(parents=True, exist_ok=True)

    fn_items = parse_topk_csv(topk_dir / "image_fn_topk.csv")
    fp_items = parse_topk_csv(topk_dir / "fp_topk.csv")

    max_n = max(3, min(6, args.hook_n))
    fn_samples = select_fn_samples(fn_items, max_n)
    fp_samples = select_fp_samples(fp_items, max_n)

    images = list_images(image_dir)
    if not images:
        raise RuntimeError(f"No images found in {image_dir}")

    model = YOLO(args.weights)
    if args.device:
        model.model.to(torch.device(args.device))
    model.model.eval()
    device = next(model.model.parameters()).device

    tp_samples = collect_tp_samples(
        model,
        images,
        image_dir,
        label_dir,
        args.imgsz,
        args.conf,
        args.nms_iou,
        args.iou_match,
        args.iou_candidate,
        args.batch,
        args.device,
        args.half,
        max_n,
    )

    sample_img_path = None
    for s in fn_samples + tp_samples + fp_samples:
        sample_img_path = Path(s["image"])
        if sample_img_path.exists():
            break
    if sample_img_path is None:
        raise RuntimeError("No valid samples for hook.")
    sample_img = cv2.imread(str(sample_img_path))
    if sample_img is None:
        raise RuntimeError("Failed to read sample image for hook.")
    sample_lb, _, _, _, _, _ = letterbox_image(sample_img, args.imgsz)
    sample_tensor = img_to_tensor(sample_lb, device)
    layer_keys, key_to_module = select_layers(
        model.model,
        sample_tensor,
        keywords=["neck", "fpn", "pan", "p3", "p4", "p5", "cv2", "cv3", "detect"],
        max_layers=3,
    )
    if not layer_keys:
        raise RuntimeError("Failed to select hook layers.")

    selected_modules = {key_to_module[k] for k in layer_keys if k in key_to_module}
    feats: Dict[str, torch.Tensor] = {}
    hooks = []

    def make_hook(name: str):
        def fn(_, __, y):
            if torch.is_tensor(y):
                feats[name] = y.detach()
            elif isinstance(y, (list, tuple)):
                for i, yi in enumerate(y):
                    if torch.is_tensor(yi):
                        feats[f"{name}.{i}"] = yi.detach()

        return fn

    for name, module in model.model.named_modules():
        if name in selected_modules:
            hooks.append(module.register_forward_hook(make_hook(name)))

    tp_ratio_gt: Dict[str, List[float]] = {k: [] for k in layer_keys}
    fn_ratio_gt: Dict[str, List[float]] = {k: [] for k in layer_keys}
    fp_ratio_pred: Dict[str, List[float]] = {k: [] for k in layer_keys}
    tp_overlays: Dict[Tuple[int, str], np.ndarray] = {}
    fn_overlays: Dict[Tuple[int, str], np.ndarray] = {}
    fp_overlays: Dict[Tuple[int, str], np.ndarray] = {}

    def process_samples(
        samples: List[Dict],
        overlays: Dict[Tuple[int, str], np.ndarray],
        ratio_gt_store: Dict[str, List[float]] = None,
        ratio_pred_store: Dict[str, List[float]] = None,
    ) -> None:
        for idx, sample in enumerate(samples):
            img_path = Path(sample["image"])
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            orig_h, orig_w = img.shape[:2]
            lb, _, pad_x, pad_y, new_w, new_h = letterbox_image(img, args.imgsz)
            x = img_to_tensor(lb, device)
            feats.clear()
            with torch.no_grad():
                _ = model.model(x)
            gt_boxes = sample.get("gt_boxes", [])
            pred_boxes = sample.get("pred_boxes", [])
            label = sample.get("label", "")
            for layer in layer_keys:
                if layer not in feats:
                    continue
                heat = compute_heatmap(feats[layer], args.imgsz, pad_x, pad_y, new_w, new_h, orig_w, orig_h)
                overlay = overlay_heatmap(img, heat, alpha=0.4)
                for b in gt_boxes:
                    x1, y1, x2, y2 = clip_box(b, orig_w, orig_h)
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
                for b in pred_boxes:
                    x1, y1, x2, y2 = clip_box(b, orig_w, orig_h)
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), 2)
                text = f"GT={len(gt_boxes)} Pred={len(pred_boxes)} {label}"
                cv2.putText(overlay, text, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                overlays[(idx, layer)] = overlay
                if ratio_gt_store is not None:
                    for b in gt_boxes:
                        ratio = compute_ratio(heat, b)
                        if not math.isnan(ratio):
                            ratio_gt_store[layer].append(ratio)
                if ratio_pred_store is not None:
                    for b in pred_boxes:
                        ratio = compute_ratio(heat, b)
                        if not math.isnan(ratio):
                            ratio_pred_store[layer].append(ratio)

    process_samples(tp_samples, tp_overlays, ratio_gt_store=tp_ratio_gt)
    process_samples(fn_samples, fn_overlays, ratio_gt_store=fn_ratio_gt)
    if fp_samples:
        process_samples(fp_samples, fp_overlays, ratio_pred_store=fp_ratio_pred)

    for h in hooks:
        h.remove()

    if tp_samples:
        build_grid(tp_samples, layer_keys, tp_overlays, hook_dir / "tp_heatmaps.png", "TP heatmaps")
    if fn_samples:
        build_grid(fn_samples, layer_keys, fn_overlays, hook_dir / "fn_heatmaps.png", "FN heatmaps")
    if fp_samples:
        build_grid(fp_samples, layer_keys, fp_overlays, hook_dir / "fp_heatmaps.png", "FP heatmaps")

    write_hook_summary(hook_dir / "hook_summary.csv", layer_keys, tp_ratio_gt, fn_ratio_gt, fp_ratio_pred)
    base_metrics = load_base_metrics(report_dir)
    update_report(report_dir / "report.md", layer_keys, tp_ratio_gt, fn_ratio_gt, fp_ratio_pred, base_metrics)

    print(f"Report updated at: {report_dir}")


if __name__ == "__main__":
    main()
