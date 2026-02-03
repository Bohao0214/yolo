import argparse
import csv
import datetime as dt
import gc
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_images(dir_path: Path) -> List[Path]:
    return sorted([p for p in dir_path.iterdir() if p.suffix.lower() in IMG_EXTS])


def label_path_for(img_path: Path, dataset_root: Path) -> Path:
    rel = img_path.relative_to(dataset_root / "images")
    return (dataset_root / "labels" / rel).with_suffix(".txt")


def load_yolo_labels(label_path: Path, w: int, h: int) -> List[Tuple[int, float, float, float, float]]:
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
            cls, xc, yc, bw, bh = parts[:5]
            try:
                boxes.append((int(cls), float(xc), float(yc), float(bw), float(bh)))
            except ValueError:
                continue
    return boxes


def xywhn_to_xyxy(xc, yc, w, h, img_w, img_h):
    x1 = (xc - w / 2) * img_w
    y1 = (yc - h / 2) * img_h
    x2 = (xc + w / 2) * img_w
    y2 = (yc + h / 2) * img_h
    return [x1, y1, x2, y2]


def letterbox_params(w: int, h: int, size: int):
    r = min(size / w, size / h)
    new_w, new_h = w * r, h * r
    pad_x = (size - new_w) / 2
    pad_y = (size - new_h) / 2
    return r, pad_x, pad_y


def to_letterbox_xyxy(xyxy, r, pad_x, pad_y):
    x1, y1, x2, y2 = xyxy
    return [
        x1 * r + pad_x,
        y1 * r + pad_y,
        x2 * r + pad_x,
        y2 * r + pad_y,
    ]


def letterbox_image(img, size: int):
    h, w = img.shape[:2]
    r = min(size / w, size / h)
    new_w, new_h = int(round(w * r)), int(round(h * r))
    resized = cv2.resize(img, (new_w, new_h))
    canvas = np.full((size, size, 3), 114, dtype=img.dtype)
    pad_x = (size - new_w) // 2
    pad_y = (size - new_h) // 2
    canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized
    return canvas, r, pad_x, pad_y


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


def greedy_match(gt: np.ndarray, pred: np.ndarray, iou_thr: float):
    iou = compute_iou_matrix(gt, pred)
    gt_match = -np.ones(len(gt), dtype=int)
    pred_match = -np.ones(len(pred), dtype=int)
    while True:
        if iou.size == 0:
            break
        idx = np.unravel_index(np.argmax(iou), iou.shape)
        max_iou = iou[idx]
        if max_iou < iou_thr:
            break
        gi, pi = idx
        gt_match[gi] = pi
        pred_match[pi] = gi
        iou[gi, :] = -1
        iou[:, pi] = -1
    return gt_match, pred_match


def compute_proxy_stats(img, xyxy):
    x1, y1, x2, y2 = [int(max(0, v)) for v in xyxy]
    x2 = min(x2, img.shape[1] - 1)
    y2 = min(y2, img.shape[0] - 1)
    if x2 <= x1 or y2 <= y1:
        return None
    roi = img[y1:y2, x1:x2]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]
    v_mean = float(v.mean())
    v_std = float(v.std())
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    lap_var = float(lap.var())
    sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(sobelx**2 + sobely**2)
    grad_energy = float(np.mean(grad))
    return v_mean, v_std, lap_var, grad_energy


def draw_boxes(img, boxes, color, labels=None, thickness=2):
    for i, b in enumerate(boxes):
        x1, y1, x2, y2 = [int(v) for v in b]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        if labels:
            cv2.putText(
                img,
                labels[i],
                (x1, max(0, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )


def build_collage(items, out_path: Path, title: str, tile_size: int):
    if not items:
        return
    cols = 4
    rows = int(np.ceil(len(items) / cols))
    tile = tile_size
    canvas = np.zeros((rows * tile + 30, cols * tile, 3), dtype=np.uint8)
    cv2.putText(canvas, title, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    for idx, img in enumerate(items):
        r = idx // cols
        c = idx % cols
        y0 = r * tile + 30
        x0 = c * tile
        canvas[y0 : y0 + tile, x0 : x0 + tile] = img
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), canvas)


def predict_images(
    model,
    images: List[Path],
    conf: float,
    iou_nms: float,
    imgsz: int,
    batch: int,
    device: str,
    half: bool,
    stream: bool,
    collect_boxes: bool = True,
):
    try:
        import torch
    except Exception:
        torch = None  # type: ignore

    imgsz = int(imgsz)
    conf = float(conf)
    iou_nms = float(iou_nms)
    batch = int(batch)

    def _run(run_device: str, run_half: bool, run_batch: int):
        if torch is not None and run_device != "cpu" and torch.cuda.is_available():
            torch.cuda.empty_cache()
        results = model.predict(
            source=[str(p) for p in images],
            imgsz=imgsz,
            conf=float(conf),
            iou=float(iou_nms),
            save=False,
            verbose=False,
            batch=int(run_batch),
            device=run_device,
            half=run_half,
            stream=stream,
        )
        preds = {}
        counts = {}
        if stream:
            for res in results:
                img_path = Path(res.path)
                counts[str(img_path)] = len(res.boxes) if res.boxes is not None else 0
                if collect_boxes:
                    items = []
                    if res.boxes is not None:
                        for xyxy, score in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy()):
                            items.append({"xyxy": xyxy.tolist(), "conf": float(score), "cls": 0})
                    preds[str(img_path)] = items
        else:
            for img_path, res in zip(images, results):
                counts[str(img_path)] = len(res.boxes) if res.boxes is not None else 0
                if collect_boxes:
                    items = []
                    if res.boxes is not None:
                        for xyxy, score in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy()):
                            items.append({"xyxy": xyxy.tolist(), "conf": float(score), "cls": 0})
                    preds[str(img_path)] = items
        # drop predictor to release dataloader buffers
        if hasattr(model, "predictor"):
            model.predictor = None
        if torch is not None and run_device != "cpu" and torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
        return preds, counts

    try:
        return _run(device, half, batch)
    except Exception as exc:
        if torch is None or device == "cpu":
            raise
        if "out of memory" not in str(exc).lower():
            raise
        print("CUDA OOM in inference, retrying on CPU with batch=1.")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return _run("cpu", False, 1)


def clear_memory(device: str) -> None:
    try:
        import torch

        if device != "cpu" and torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
    except Exception:
        pass
    gc.collect()


def write_pred_files(out_dir: Path, split: str, preds: Dict[str, List[Dict]], counts: Dict[str, Tuple[int, int]]):
    split_dir = out_dir / "preds" / split
    split_dir.mkdir(parents=True, exist_ok=True)
    for img_path, items in preds.items():
        pre_nms, post_nms = counts.get(img_path, (0, 0))
        txt = split_dir / (Path(img_path).stem + ".txt")
        lines = [f"# pre_nms={pre_nms} post_nms={post_nms}"]
        for it in items:
            x1, y1, x2, y2 = it["xyxy"]
            lines.append(f"{it['cls']} {it['conf']:.6f} {x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f}")
        txt.write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="")
    parser.add_argument("--data_root", type=str, default="")
    parser.add_argument("--weights", type=str, default="")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.2)
    parser.add_argument("--nms_iou", type=float, default=0.5)
    parser.add_argument("--match_iou", type=float, default=0.5)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--split", nargs="+", default=["val", "test"])
    parser.add_argument("--out_root", type=str, required=True)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--batch", type=int, default=1, help="inference batch size")
    parser.add_argument("--half", action="store_true", help="use fp16 for inference")
    parser.add_argument("--stream", action="store_true", default=True)
    return parser.parse_args()


def load_config(path: str) -> Dict:
    try:
        import yaml  # type: ignore
    except Exception as exc:
        raise RuntimeError("PyYAML is required to load --config.") from exc
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("Config YAML must be a mapping.")
    return data


def resolve_path(base_dir: Path, value: str) -> str:
    p = Path(value)
    if p.is_absolute():
        return str(p)
    return str((base_dir / p).resolve())


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    weights = args.weights
    img_size = args.imgsz
    conf = args.conf
    nms_iou = args.nms_iou
    match_iou = args.match_iou
    topk = args.topk
    splits = args.split
    out_root = Path(args.out_root)
    device = args.device
    batch = args.batch
    half = args.half
    stream = args.stream

    if args.config:
        cfg_path = Path(args.config).resolve()
        cfg = load_config(str(cfg_path))
        base_dir = cfg_path.parent
        if cfg.get("data_root"):
            data_root = Path(resolve_path(base_dir, str(cfg["data_root"])))
        weights = resolve_path(base_dir, str(cfg.get("weights") or cfg.get("model") or weights))
        img_size = int(cfg.get("imgsz", img_size))
        conf = float(cfg.get("conf", conf))
        nms_iou = float(cfg.get("nms_iou", cfg.get("iou", nms_iou)))
        match_iou = float(cfg.get("match_iou", match_iou))
        device = str(cfg.get("device", device))
        if cfg.get("eval_batch"):
            batch = int(cfg.get("eval_batch"))
        topk = int(cfg.get("topk", topk))

    # CLI overrides (highest priority)
    if args.data_root:
        data_root = Path(args.data_root)
    if args.weights:
        weights = args.weights
    if args.imgsz:
        img_size = args.imgsz
    if args.conf >= 0:
        conf = args.conf
    if args.nms_iou >= 0:
        nms_iou = args.nms_iou
    if args.match_iou >= 0:
        match_iou = args.match_iou
    if args.topk:
        topk = args.topk
    if args.split:
        splits = args.split
    if args.out_root:
        out_root = Path(args.out_root)
    if args.device:
        device = args.device
    if args.batch:
        batch = args.batch
    if args.half:
        half = True
    if args.stream is not None:
        stream = args.stream

    if not data_root or not weights:
        raise RuntimeError("Please provide --data_root and --weights (or use --config with these fields).")

    timestamp = dt.datetime.now().strftime("%y%m%d%H%M")
    out_dir = out_root / f"report_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    from ultralytics import YOLO

    model = YOLO(weights)

    all_stats_rows = []
    error_rows = []
    proxy_rows = []
    sweep_rows = []
    pred_count_rows = []

    all_fn_items = []
    all_fp_items = []
    all_dup_items = []

    for split in splits:
        img_dir = data_root / "images" / split
        images = list_images(img_dir)
        if not images:
            continue

        # Pre-NMS counts using iou=1.0 (approximate pre-NMS after conf)
        _, pre_counts = predict_images(
            model, images, conf, 1.0, img_size, batch, device, half, stream, collect_boxes=False
        )
        clear_memory(device)
        preds, post_counts = predict_images(
            model, images, conf, nms_iou, img_size, batch, device, half, stream, collect_boxes=True
        )
        clear_memory(device)

        counts = {}
        for img in [str(p) for p in images]:
            counts[img] = (pre_counts.get(img, 0), post_counts.get(img, 0))
            pred_count_rows.append([img, split, counts[img][0], counts[img][1]])

        write_pred_files(out_dir, split, preds, counts)

        for img_path in images:
            im = cv2.imread(str(img_path))
            if im is None:
                continue
            h, w = im.shape[:2]
            r, pad_x, pad_y = letterbox_params(w, h, img_size)
            label_path = label_path_for(img_path, data_root)
            gt_yolo = load_yolo_labels(label_path, w, h)
            gt_xyxy = []
            gt_area_ratio = []
            for cls, xc, yc, bw, bh in gt_yolo:
                xyxy = xywhn_to_xyxy(xc, yc, bw, bh, w, h)
                lb = to_letterbox_xyxy(xyxy, r, pad_x, pad_y)
                w_lb = max(0.0, lb[2] - lb[0])
                h_lb = max(0.0, lb[3] - lb[1])
                area_ratio = (w_lb * h_lb) / (img_size * img_size)
                gt_xyxy.append(xyxy)
                gt_area_ratio.append(area_ratio)
                all_stats_rows.append(
                    [str(img_path), split, area_ratio, w_lb, h_lb, lb[0], lb[1], lb[2], lb[3]]
                )

            pred_items = preds.get(str(img_path), [])
            pred_xyxy = np.array([p["xyxy"] for p in pred_items], dtype=np.float32) if pred_items else np.zeros((0, 4))
            pred_conf = np.array([p["conf"] for p in pred_items], dtype=np.float32) if pred_items else np.zeros((0,))

            gt_xyxy_arr = np.array(gt_xyxy, dtype=np.float32) if gt_xyxy else np.zeros((0, 4))
            gt_match, pred_match = greedy_match(gt_xyxy_arr, pred_xyxy, match_iou)

            fn_count = int((gt_match == -1).sum())
            fp_count = int((pred_match == -1).sum())

            dup_count = 0
            dup_map = {}
            if gt_xyxy_arr.size and pred_xyxy.size:
                iou_mat = compute_iou_matrix(gt_xyxy_arr, pred_xyxy)
                overlaps = (iou_mat >= match_iou)
                dup_counts = overlaps.sum(axis=1)
                dup_count = int(np.sum(dup_counts > 1))
                for gi, cnt in enumerate(dup_counts):
                    if cnt > 1:
                        dup_map[gi] = list(np.where(overlaps[gi])[0])

            error_rows.append(
                [str(img_path), split, len(gt_xyxy), len(pred_xyxy), fn_count, fp_count, dup_count]
            )

            # Collect FN/FP for crops and TopK
            for idx, area_ratio in enumerate(gt_area_ratio):
                if gt_match[idx] == -1:
                    all_fn_items.append((area_ratio, str(img_path), gt_xyxy[idx]))
                    proxy = compute_proxy_stats(im, gt_xyxy[idx])
                    if proxy:
                        v_mean, v_std, lap_var, grad = proxy
                        proxy_rows.append(
                            ["FN", str(img_path), split, *gt_xyxy[idx], area_ratio, v_mean, v_std, lap_var, grad]
                        )
                        crop = im[int(gt_xyxy[idx][1]) : int(gt_xyxy[idx][3]), int(gt_xyxy[idx][0]) : int(gt_xyxy[idx][2])]
                        if crop.size:
                            crop_dir = out_dir / "crops" / "fn"
                            crop_dir.mkdir(parents=True, exist_ok=True)
                            cv2.imwrite(str(crop_dir / f"{Path(img_path).stem}_fn_{idx}.png"), crop)

            for pidx, confv in enumerate(pred_conf):
                if pred_match[pidx] == -1:
                    all_fp_items.append((float(confv), str(img_path), pred_items[pidx]["xyxy"]))
                    proxy = compute_proxy_stats(im, pred_items[pidx]["xyxy"])
                    if proxy:
                        v_mean, v_std, lap_var, grad = proxy
                        proxy_rows.append(
                            ["FP", str(img_path), split, *pred_items[pidx]["xyxy"], 0.0, v_mean, v_std, lap_var, grad]
                        )
                        crop = im[
                            int(pred_items[pidx]["xyxy"][1]) : int(pred_items[pidx]["xyxy"][3]),
                            int(pred_items[pidx]["xyxy"][0]) : int(pred_items[pidx]["xyxy"][2]),
                        ]
                        if crop.size:
                            crop_dir = out_dir / "crops" / "fp"
                            crop_dir.mkdir(parents=True, exist_ok=True)
                            cv2.imwrite(str(crop_dir / f"{Path(img_path).stem}_fp_{pidx}.png"), crop)

            if dup_count > 0:
                all_dup_items.append((dup_count, str(img_path), gt_xyxy, pred_items, dup_map))

        # end split

    # Save prediction counts
    with open(out_dir / "pred_counts.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image", "split", "pre_nms", "post_nms"])
        w.writerows(pred_count_rows)

    # Save bbox stats CSV
    stats_csv = out_dir / "bbox_stats.csv"
    with open(stats_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "split", "area_ratio", "w_lb", "h_lb", "x1_lb", "y1_lb", "x2_lb", "y2_lb"])
        writer.writerows(all_stats_rows)

    # Save error slicing
    error_csv = out_dir / "error_slicing.csv"
    with open(error_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "split", "gt", "pred", "fn", "fp", "duplicate"])
        writer.writerows(sorted(error_rows, key=lambda x: (x[4] + x[5] + x[6]), reverse=True))

    # Proxy stats CSV
    proxy_csv = out_dir / "proxy_stats.csv"
    with open(proxy_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["type", "image", "split", "x1", "y1", "x2", "y2", "area_ratio", "v_mean", "v_std", "lap_var", "sobel"]
        )
        writer.writerows(proxy_rows)

    # TopK collages
    fn_top = sorted(all_fn_items, key=lambda x: x[0])[:topk]
    fp_top = sorted(all_fp_items, key=lambda x: -x[0])[:topk]
    dup_top = sorted(all_dup_items, key=lambda x: -x[0])[:topk]

    fn_imgs = []
    for _, img_path, gt_box in fn_top:
        im = cv2.imread(img_path)
        if im is None:
            continue
        vis, r, pad_x, pad_y = letterbox_image(im, img_size)
        lb = to_letterbox_xyxy(gt_box, r, pad_x, pad_y)
        draw_boxes(vis, [lb], (0, 255, 0), labels=["FN"])
        fn_imgs.append(vis)

    fp_imgs = []
    for confv, img_path, pred_box in fp_top:
        im = cv2.imread(img_path)
        if im is None:
            continue
        vis, r, pad_x, pad_y = letterbox_image(im, img_size)
        lb = to_letterbox_xyxy(pred_box, r, pad_x, pad_y)
        draw_boxes(vis, [lb], (0, 0, 255), labels=[f"FP {confv:.2f}"])
        fp_imgs.append(vis)

    dup_imgs = []
    for _, img_path, gt_boxes, pred_items, dup_map in dup_top:
        im = cv2.imread(img_path)
        if im is None:
            continue
        vis, r, pad_x, pad_y = letterbox_image(im, img_size)
        for gi, pred_ids in dup_map.items():
            gt_lb = to_letterbox_xyxy(gt_boxes[gi], r, pad_x, pad_y)
            draw_boxes(vis, [gt_lb], (0, 255, 0), labels=[f"GT{gi}"])
            for pi in pred_ids:
                pb = pred_items[pi]["xyxy"]
                confv = pred_items[pi]["conf"]
                pb_lb = to_letterbox_xyxy(pb, r, pad_x, pad_y)
                draw_boxes(vis, [pb_lb], (0, 0, 255), labels=[f"P{pi}:{confv:.2f}"])
        dup_imgs.append(vis)

    build_collage(fn_imgs, out_dir / "FN_TopK_small.png", "FN TopK (small)", img_size)
    build_collage(fp_imgs, out_dir / "FP_TopK.png", "FP TopK", img_size)
    build_collage(dup_imgs, out_dir / "Duplicate_TopK.png", "Duplicate TopK", img_size)

    # Sweep
    sweep_confs = [0.15, 0.2, 0.25]
    sweep_ious = [0.4, 0.5, 0.6]
    for s_conf in sweep_confs:
        for s_iou in sweep_ious:
            tp_img = 0
            pos_img = 0
            total_fp = 0
            total_dup = 0
            ious_all = []
            for split in splits:
                img_dir = data_root / "images" / split
                images = list_images(img_dir)
                if not images:
                    continue
                s_preds, _ = predict_images(
                    model, images, s_conf, s_iou, img_size, batch, device, half, stream, collect_boxes=True
                )
                clear_memory(device)
                for img_path in images:
                    im = cv2.imread(str(img_path))
                    if im is None:
                        continue
                    h, w = im.shape[:2]
                    label_path = label_path_for(img_path, data_root)
                    gt_yolo = load_yolo_labels(label_path, w, h)
                    gt_xyxy = [xywhn_to_xyxy(xc, yc, bw, bh, w, h) for _, xc, yc, bw, bh in gt_yolo]
                    gt_arr = np.array(gt_xyxy, dtype=np.float32) if gt_xyxy else np.zeros((0, 4))
                    pred_items = s_preds.get(str(img_path), [])
                    pred_xyxy = np.array([p["xyxy"] for p in pred_items], dtype=np.float32) if pred_items else np.zeros((0, 4))
                    gt_match, pred_match = greedy_match(gt_arr, pred_xyxy, match_iou)
                    if len(gt_arr) > 0:
                        pos_img += 1
                        if (gt_match != -1).any():
                            tp_img += 1
                    total_fp += int((pred_match == -1).sum())
                    if gt_arr.size and pred_xyxy.size:
                        iou_mat = compute_iou_matrix(gt_arr, pred_xyxy)
                        overlaps = (iou_mat >= match_iou)
                        dup_counts = overlaps.sum(axis=1)
                        total_dup += int(np.sum(dup_counts > 1))
                        matched = iou_mat[gt_match != -1, gt_match[gt_match != -1]]
                        if matched.size:
                            ious_all.extend(matched.tolist())

            recall_img = tp_img / pos_img if pos_img > 0 else 0.0
            fp_per_img = total_fp / max(1, pos_img)
            dup_per_img = total_dup / max(1, pos_img)
            if ious_all:
                p10, p50, p90 = np.percentile(ious_all, [10, 50, 90])
            else:
                p10 = p50 = p90 = 0.0
            sweep_rows.append([s_conf, s_iou, recall_img, fp_per_img, dup_per_img, p10, p50, p90])

    with open(out_dir / "sweep_report.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["conf", "nms_iou", "recall_img", "fp_per_img", "dup_per_img", "iou_p10", "iou_p50", "iou_p90"])
        w.writerows(sweep_rows)

    # Simple report
    report_md = out_dir / "report.md"
    with open(report_md, "w", encoding="utf-8") as f:
        f.write("# Error Audit Report\n\n")
        f.write(f"- weights: {weights}\n")
        f.write(f"- imgsz={img_size}, conf={conf}, nms_iou={nms_iou}, match_iou={match_iou}\n")
        f.write(f"- splits: {', '.join(splits)}\n\n")
        f.write("## Key Files\n")
        f.write("- bbox_stats.csv\n")
        f.write("- error_slicing.csv\n")
        f.write("- proxy_stats.csv\n")
        f.write("- pred_counts.csv\n")
        f.write("- FN_TopK_small.png\n")
        f.write("- FP_TopK.png\n")
        f.write("- Duplicate_TopK.png\n")
        f.write("- sweep_report.csv\n\n")
        f.write("## Findings (auto)\n")
        f.write("- Review proxy_stats.csv: compare FN vs FP for V mean/std and Laplacian variance.\n")
        f.write("- Review sweep_report.csv to pick a conf/nms_iou trade-off.\n")

    print(f"Report saved to: {out_dir}")


if __name__ == "__main__":
    main()
"""""
/home/ubuntu/anaconda3/envs/yolo11/bin/python \
  /home/ubuntu/project/deduibi/yolo/tools/yolo_error_audit.py \
  --config /home/ubuntu/project/deduibi/yolo/configs/yolo11/defect.yaml \
  --weights /home/ubuntu/project/deduibi/yolo/models/defect/exp_2601261106/best/best.pt \
  --split val test \
  --out_root /home/ubuntu/project/deduibi/yolo/analysis \
  --batch 1 \
  --half
"""""
