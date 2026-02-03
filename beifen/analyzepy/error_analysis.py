import argparse
import csv
import datetime as dt
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


def to_letterbox_xyxy(xyxy, r, pad_x, pad_y):
    x1, y1, x2, y2 = xyxy
    return [
        x1 * r + pad_x,
        y1 * r + pad_y,
        x2 * r + pad_x,
        y2 * r + pad_y,
    ]


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
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    mean = float(gray.mean())
    std = float(gray.std())
    p5, p95 = np.percentile(gray, [5, 95])
    contrast = float(p95 - p5)
    sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(sobelx**2 + sobely**2)
    grad_energy = float(np.mean(grad))
    return mean, std, contrast, grad_energy


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


def run_inference(
    images: List[Path],
    out_dir: Path,
    weights: str,
    device: str,
    imgsz: int,
    conf: float,
    iou_nms: float,
    batch: int,
    half: bool,
    stream: bool,
) -> Dict[str, List[Dict]]:
    from ultralytics import YOLO
    import torch

    model = YOLO(weights)
    imgsz = int(imgsz)
    conf = float(conf)
    iou_nms = float(iou_nms)
    batch = int(batch)

    def _run(run_device: str, run_half: bool, run_batch: int) -> Dict[str, List[Dict]]:
        if run_device != "cpu" and torch.cuda.is_available():
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
        if stream:
            for res in results:
                img_path = Path(res.path)
                items = []
                if res.boxes is not None:
                    for xyxy, score in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy()):
                        items.append({"xyxy": xyxy.tolist(), "conf": float(score), "cls": 0})
                preds[str(img_path)] = items
        else:
            for img_path, res in zip(images, results):
                items = []
                if res.boxes is not None:
                    for xyxy, score in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy()):
                        items.append({"xyxy": xyxy.tolist(), "conf": float(score), "cls": 0})
                preds[str(img_path)] = items
        if hasattr(model, "predictor"):
            model.predictor = None
        if run_device != "cpu" and torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
        return preds

    try:
        preds = _run(device, half, batch)
    except torch.OutOfMemoryError:
        if device == "cpu":
            raise
        print("CUDA OOM in inference, retrying on CPU with batch=1.")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        preds = _run("cpu", False, 1)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "predictions.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "cls", "conf", "x1", "y1", "x2", "y2"])
        for img_path, items in preds.items():
            for it in items:
                x1, y1, x2, y2 = it["xyxy"]
                writer.writerow([img_path, it["cls"], it["conf"], x1, y1, x2, y2])
    return preds


def load_predictions_from_txt(images: List[Path], pred_dir: Path, pred_format: str) -> Dict[str, List[Dict]]:
    preds = {}
    for img_path in images:
        im = cv2.imread(str(img_path))
        if im is None:
            preds[str(img_path)] = []
            continue
        h, w = im.shape[:2]
        pred_file = pred_dir / (img_path.stem + ".txt")
        items = []
        if pred_file.exists():
            with open(pred_file, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 6:
                        continue
                    cls, xc, yc, bw, bh, conf = parts[:6]
                    try:
                        cls_i = int(cls)
                        conf_f = float(conf)
                        if pred_format == "yolo":
                            xc_f, yc_f = float(xc), float(yc)
                            bw_f, bh_f = float(bw), float(bh)
                            xyxy = xywhn_to_xyxy(xc_f, yc_f, bw_f, bh_f, w, h)
                        else:
                            x1, y1, x2, y2 = float(xc), float(yc), float(bw), float(bh)
                            xyxy = [x1, y1, x2, y2]
                        items.append({"xyxy": xyxy, "conf": conf_f, "cls": cls_i})
                    except Exception:
                        continue
        preds[str(img_path)] = items
    return preds


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="")
    parser.add_argument("--data_root", type=str, default="")
    parser.add_argument("--weights", type=str, default="")
    parser.add_argument("--imgsz", type=int, default=0)
    parser.add_argument("--conf", type=float, default=-1.0)
    parser.add_argument("--nms_iou", type=float, default=-1.0)
    parser.add_argument("--match_iou", type=float, default=-1.0)
    parser.add_argument("--topk", type=int, default=0)
    parser.add_argument("--split", nargs="+", default=[])
    parser.add_argument("--out_root", type=str, default="")
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--batch", type=int, default=0)
    parser.add_argument("--pred_dir", type=str, default="")
    parser.add_argument("--pred_format", type=str, default="yolo")
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--stream", action="store_true", default=True)
    args = parser.parse_args()

    # Defaults
    dataset_root = Path("/home/ubuntu/project/deduibi/yolo/dataset/yolo/datasetm6c")
    splits = ["test"]
    weights = "/home/ubuntu/project/deduibi/yolo/models/defect/exp_2601261106/best/best.pt"
    device = ""
    img_size = 640
    conf = 0.25
    iou_nms = 0.7
    match_iou = 0.5
    topk = 20
    pred_batch = 1
    pred_half = True
    pred_stream = True
    pred_dir = ""
    pred_format = "yolo"
    output_root = Path("/home/ubuntu/project/deduibi/yolo/analysis")

    if args.config:
        cfg_path = Path(args.config).resolve()
        cfg = load_config(str(cfg_path))
        base_dir = cfg_path.parent
        if cfg.get("data_root"):
            dataset_root = Path(resolve_path(base_dir, str(cfg["data_root"])))
        weights = str(cfg.get("weights") or cfg.get("model") or weights)
        if weights:
            weights = resolve_path(base_dir, weights)
        img_size = int(cfg.get("imgsz", img_size))
        conf = float(cfg.get("conf", conf))
        iou_nms = float(cfg.get("nms_iou", cfg.get("iou", iou_nms)))
        match_iou = float(cfg.get("match_iou", match_iou))
        device = str(cfg.get("device", device))
        if cfg.get("eval_batch"):
            pred_batch = int(cfg.get("eval_batch"))
        topk = int(cfg.get("topk", topk))

    # CLI overrides (highest priority)
    if args.data_root:
        dataset_root = Path(args.data_root)
    if args.weights:
        weights = args.weights
    if args.imgsz > 0:
        img_size = args.imgsz
    if args.conf >= 0:
        conf = args.conf
    if args.nms_iou >= 0:
        iou_nms = args.nms_iou
    if args.match_iou >= 0:
        match_iou = args.match_iou
    if args.topk > 0:
        topk = args.topk
    if args.split:
        splits = args.split
    if args.out_root:
        output_root = Path(args.out_root)
    if args.device:
        device = args.device
    if args.batch > 0:
        pred_batch = args.batch
    if args.pred_dir:
        pred_dir = args.pred_dir
    if args.pred_format:
        pred_format = args.pred_format
    if args.half:
        pred_half = True
    pred_stream = args.stream

    timestamp = dt.datetime.now().strftime("%y%m%d%H%M")
    out_dir = output_root / f"report_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_stats_rows = []
    error_rows = []
    fn_rows = []
    fp_rows = []
    dup_rows = []
    proxy_rows = []

    all_fn_items = []
    all_fp_items = []
    all_dup_items = []

    for split in splits:
        img_dir = dataset_root / "images" / split
        images = list_images(img_dir)
        if not images:
            continue

        if weights and not pred_dir:
            try:
                preds = run_inference(
                    images,
                    out_dir / "preds",
                    weights,
                    device,
                    img_size,
                    conf,
                    iou_nms,
                    pred_batch,
                    pred_half,
                    pred_stream,
                )
            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower():
                    raise
                preds = run_inference(
                    images,
                    out_dir / "preds",
                    weights,
                    "cpu",
                    img_size,
                    conf,
                    iou_nms,
                    1,
                    False,
                    True,
                )
        elif pred_dir:
            preds = load_predictions_from_txt(images, Path(pred_dir), pred_format)
        else:
            raise RuntimeError("Please set WEIGHTS for inference or provide PRED_DIR with supported format.")

        for img_path in images:
            im = cv2.imread(str(img_path))
            if im is None:
                continue
            h, w = im.shape[:2]
            r, pad_x, pad_y = letterbox_params(w, h, img_size)
            label_path = label_path_for(img_path, dataset_root)
            gt_yolo = load_yolo_labels(label_path, w, h)
            gt_xyxy = []
            gt_area_ratio = []
            gt_stats = []
            for cls, xc, yc, bw, bh in gt_yolo:
                xyxy = xywhn_to_xyxy(xc, yc, bw, bh, w, h)
                lb = to_letterbox_xyxy(xyxy, r, pad_x, pad_y)
                w_lb = max(0.0, lb[2] - lb[0])
                h_lb = max(0.0, lb[3] - lb[1])
                area_ratio = (w_lb * h_lb) / (img_size * img_size)
                proxy = compute_proxy_stats(im, xyxy)
                gt_xyxy.append(xyxy)
                gt_area_ratio.append(area_ratio)
                gt_stats.append(proxy)
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

            # Collect FN/TP proxy stats
            for idx, proxy in enumerate(gt_stats):
                if proxy is None:
                    continue
                mean, std, contrast, grad = proxy
                is_fn = gt_match[idx] == -1
                proxy_rows.append(
                    [str(img_path), split, int(is_fn), gt_area_ratio[idx], mean, std, contrast, grad]
                )

            # FN TopK items
            for idx, area_ratio in enumerate(gt_area_ratio):
                if gt_match[idx] == -1:
                    contrast = gt_stats[idx][2] if gt_stats[idx] else 0.0
                    all_fn_items.append((area_ratio, contrast, str(img_path), gt_xyxy[idx]))

            # FP TopK items
            for pidx, conf in enumerate(pred_conf):
                if pred_match[pidx] == -1:
                    all_fp_items.append((float(conf), str(img_path), pred_items[pidx]["xyxy"]))

            # Duplicate TopK items
            if dup_count > 0:
                all_dup_items.append((dup_count, str(img_path), gt_xyxy, pred_items, dup_map))

        # end split

    # Save bbox stats CSV
    stats_csv = out_dir / "bbox_stats.csv"
    with open(stats_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["image", "split", "area_ratio", "w_lb", "h_lb", "x1_lb", "y1_lb", "x2_lb", "y2_lb"]
        )
        writer.writerows(all_stats_rows)

    # Percentiles summary
    area_ratios = np.array([r[2] for r in all_stats_rows], dtype=np.float32) if all_stats_rows else np.array([])
    summary_md = out_dir / "summary.md"
    with open(summary_md, "w", encoding="utf-8") as f:
        f.write("# BBox Area Ratio Summary\n")
        if area_ratios.size:
            f.write(
                f"min={area_ratios.min():.6f}, p1={np.percentile(area_ratios,1):.6f}, "
                f"p5={np.percentile(area_ratios,5):.6f}, p10={np.percentile(area_ratios,10):.6f}, "
                f"p50={np.percentile(area_ratios,50):.6f}, p90={np.percentile(area_ratios,90):.6f}\n"
            )
            for t in [0.002, 0.005, 0.01]:
                f.write(f"area_ratio < {t*100:.1f}% : {(area_ratios < t).mean()*100:.2f}%\n")
        else:
            f.write("No boxes found.\n")

        f.write("\n# Inference Params\n")
        f.write(f"imgsz={img_size}, conf={conf}, iou={iou_nms}\n")
        if weights:
            f.write(f"weights={weights}\n")

    # Proxy stats CSV
    proxy_csv = out_dir / "proxy_stats.csv"
    with open(proxy_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "split", "is_fn", "area_ratio", "mean", "std", "contrast", "grad_energy"])
        writer.writerows(proxy_rows)

    # Error slicing CSV
    error_csv = out_dir / "error_slicing.csv"
    with open(error_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "split", "gt", "pred", "fn", "fp", "duplicate"])
        writer.writerows(sorted(error_rows, key=lambda x: (x[4] + x[5] + x[6]), reverse=True))

    # TopK lists and collages (basic)
    fn_top = sorted(all_fn_items, key=lambda x: x[0])[:topk]
    fn_top_low = sorted(all_fn_items, key=lambda x: x[1])[:topk]
    fp_top = sorted(all_fp_items, key=lambda x: -x[0])[:topk]
    dup_top = sorted(all_dup_items, key=lambda x: -x[0])[:topk]

    fn_imgs = []
    for _, _, img_path, gt_box in fn_top:
        im = cv2.imread(img_path)
        if im is None:
            continue
        vis, r, pad_x, pad_y = letterbox_image(im, img_size)
        lb = to_letterbox_xyxy(gt_box, r, pad_x, pad_y)
        draw_boxes(vis, [lb], (0, 255, 0), labels=["FN"])
        fn_imgs.append(vis)

    fp_imgs = []
    for conf, img_path, pred_box in fp_top:
        im = cv2.imread(img_path)
        if im is None:
            continue
        vis, r, pad_x, pad_y = letterbox_image(im, img_size)
        lb = to_letterbox_xyxy(pred_box, r, pad_x, pad_y)
        draw_boxes(vis, [lb], (0, 0, 255), labels=[f"FP {conf:.2f}"])
        fp_imgs.append(vis)

    dup_imgs = []
    for _, img_path, gt_boxes, pred_items, dup_map in dup_top:
        im = cv2.imread(img_path)
        if im is None:
            continue
        vis, r, pad_x, pad_y = letterbox_image(im, img_size)
        # draw all duplicate GTs and their preds
        for gi, pred_ids in dup_map.items():
            gt_lb = to_letterbox_xyxy(gt_boxes[gi], r, pad_x, pad_y)
            draw_boxes(vis, [gt_lb], (0, 255, 0), labels=[f"GT{gi}"])
            for pi in pred_ids:
                pb = pred_items[pi]["xyxy"]
                conf = pred_items[pi]["conf"]
                pb_lb = to_letterbox_xyxy(pb, r, pad_x, pad_y)
                draw_boxes(vis, [pb_lb], (0, 0, 255), labels=[f"P{pi}:{conf:.2f}"])
        dup_imgs.append(vis)

    build_collage(fn_imgs, out_dir / "fn_topk.png", "FN TopK (small area)", img_size)
    build_collage(fp_imgs, out_dir / "fp_topk.png", "FP TopK", img_size)
    build_collage(dup_imgs, out_dir / "dup_topk.png", "Duplicate TopK", img_size)

    # Save TopK CSVs
    with open(out_dir / "fn_topk_small.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["area_ratio", "contrast", "image", "x1", "y1", "x2", "y2"])
        for ar, ct, img, box in fn_top:
            w.writerow([ar, ct, img, *box])
    with open(out_dir / "fn_topk_lowcontrast.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["area_ratio", "contrast", "image", "x1", "y1", "x2", "y2"])
        for ar, ct, img, box in fn_top_low:
            w.writerow([ar, ct, img, *box])
    with open(out_dir / "fp_topk.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["conf", "image", "x1", "y1", "x2", "y2"])
        for conf, img, box in fp_top:
            w.writerow([conf, img, *box])

    print(f"Report saved to: {out_dir}")


if __name__ == "__main__":
    main()
    
"""""
/home/ubuntu/anaconda3/envs/yolo11/bin/python \
  /home/ubuntu/project/deduibi/yolo/tools/error_analysis.py \
  --config /home/ubuntu/project/deduibi/yolo/configs/yolo11/defect.yaml \
  --weights /home/ubuntu/project/deduibi/yolo/models/defect/exp_2601261106/best/best.pt  \
  --split val test \
  --out_root /home/ubuntu/project/deduibi/yolo/analysis
"""""
