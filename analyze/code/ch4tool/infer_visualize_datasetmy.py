#!/usr/bin/env python3
"""
Infer + visualize for datasetmy (train/test/val/bal), with 1/4 image size output.

Task A: run Faster R-CNN + RT-DETR together
python analyze/code/ch4tool/infer_visualize_datasetmy.py \
  --dataset-root /home/ubuntu/hpproject/yolo/dataset/yolo/datasetmy \
  --model "name=faster_rcnn,type=fasterrcnn,weights=/ABS/PATH/TO/faster_rcnn_best.pt" \
  --model "name=rt_detr,type=rtdetr,weights=/ABS/PATH/TO/rt_detr_best.pt" \
  --out-root /home/ubuntu/hpproject/yolo/experiments/vis_datasetmy_two_models \
  --splits train,test,val,bal \
  --device 0 --conf 0.05 --imgsz 640 --batch 4 --resize-ratio 0.25

Task B: run any YOLO weight
python analyze/code/ch4tool/infer_visualize_datasetmy.py \
  --dataset-root /home/ubuntu/hpproject/yolo/dataset/yolo/datasetmy \
  --model "name=my_yolo,type=yolo,weights=/ABS/PATH/TO/yolo_best.pt" \
  --out-root /home/ubuntu/hpproject/yolo/experiments/vis_datasetmy_yolo \
  --splits train,test,val,bal \
  --device 0 --conf 0.05 --imgsz 640 --batch 4 --resize-ratio 0.25

Output layout:
  <out-root>/
    raw_preds/<model_name>/<split>.json
    overlays/<model_name>/<split>/**/*.jpg
    run_summary.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw

import torch

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
PALETTE = [
    (255, 56, 56),
    (255, 157, 151),
    (255, 112, 31),
    (255, 178, 29),
    (207, 210, 49),
    (72, 249, 10),
    (146, 204, 23),
    (61, 219, 134),
    (26, 147, 52),
    (0, 212, 187),
    (44, 153, 168),
    (0, 194, 255),
    (52, 69, 147),
    (100, 115, 255),
    (0, 24, 236),
    (132, 56, 255),
    (82, 0, 133),
    (203, 56, 255),
    (255, 149, 200),
    (255, 55, 199),
]


@dataclass
class ModelSpec:
    name: str
    model_type: str
    weights: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run model inference on datasetmy and save compressed overlay images."
    )
    parser.add_argument("--dataset-root", type=Path, required=True, help="Dataset root (YOLO format).")
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        help='Repeatable. Format: "name=<name>,type=<fasterrcnn|rtdetr|yolo>,weights=<abs_path>".',
    )
    parser.add_argument("--out-root", type=Path, required=True, help="Output root directory.")
    parser.add_argument("--splits", type=str, default="train,test,val,bal")
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--resize-ratio", type=float, default=0.25)
    parser.add_argument("--jpeg-quality", type=int, default=85)
    parser.add_argument("--line-width", type=int, default=0, help="0 means auto.")
    parser.add_argument("--max-samples", type=int, default=0, help="0 means all images.")
    return parser.parse_args()


def parse_model_spec(raw: str) -> ModelSpec:
    parts = [x.strip() for x in raw.split(",") if x.strip()]
    kv: Dict[str, str] = {}
    for item in parts:
        if "=" not in item:
            raise ValueError(f"Invalid --model token: {item!r}")
        k, v = item.split("=", 1)
        kv[k.strip().lower()] = v.strip()
    name = kv.get("name", "")
    model_type = kv.get("type", "").lower().replace("-", "").replace("_", "")
    weights = kv.get("weights", "")
    if not name or not model_type or not weights:
        raise ValueError(f"Invalid --model spec: {raw}")
    if model_type not in {"fasterrcnn", "rtdetr", "yolo"}:
        raise ValueError(f"Unsupported model type: {model_type}")
    return ModelSpec(name=name, model_type=model_type, weights=Path(weights).expanduser().resolve())


def resolve_device(device_arg: str) -> str:
    if device_arg:
        return device_arg
    return "0" if torch.cuda.is_available() else "cpu"


def list_images(image_dir: Path) -> List[Path]:
    if not image_dir.exists():
        return []
    return sorted([p for p in image_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS])


def iter_batches(items: Sequence[Path], batch_size: int) -> Iterable[Sequence[Path]]:
    if batch_size <= 0:
        batch_size = 1
    for idx in range(0, len(items), batch_size):
        yield items[idx : idx + batch_size]


def choose_line_width(width: int, height: int, user_line_width: int) -> int:
    if user_line_width > 0:
        return user_line_width
    base = min(width, height)
    return max(2, int(round(base / 280)))


def clamp_box(x1: float, y1: float, x2: float, y2: float, w: int, h: int) -> Tuple[float, float, float, float]:
    x1 = max(0.0, min(float(w - 1), x1))
    y1 = max(0.0, min(float(h - 1), y1))
    x2 = max(0.0, min(float(w - 1), x2))
    y2 = max(0.0, min(float(h - 1), y2))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def text_size(draw: ImageDraw.ImageDraw, text: str) -> Tuple[int, int]:
    if hasattr(draw, "textbbox"):
        left, top, right, bottom = draw.textbbox((0, 0), text=text)
        return right - left, bottom - top
    return draw.textsize(text=text)  # type: ignore[attr-defined]


def draw_overlay(
    image_path: Path,
    boxes: List[List[float]],
    scores: List[float],
    labels: List[int],
    save_path: Path,
    resize_ratio: float,
    jpeg_quality: int,
    line_width: int,
) -> None:
    src = Image.open(image_path).convert("RGB")
    src_w, src_h = src.size
    dst_w = max(1, int(round(src_w * resize_ratio)))
    dst_h = max(1, int(round(src_h * resize_ratio)))
    canvas = src.resize((dst_w, dst_h), Image.Resampling.BILINEAR)
    draw = ImageDraw.Draw(canvas)

    sx = dst_w / float(src_w)
    sy = dst_h / float(src_h)
    lw = choose_line_width(dst_w, dst_h, line_width)
    pad = max(1, lw // 2)

    order = sorted(range(len(scores)), key=lambda i: float(scores[i]), reverse=True)
    for i in order:
        box = boxes[i]
        score = float(scores[i])
        cls_id = int(labels[i]) if i < len(labels) else 0
        color = PALETTE[cls_id % len(PALETTE)]

        x1, y1, x2, y2 = box[:4]
        x1, y1, x2, y2 = clamp_box(x1 * sx, y1 * sy, x2 * sx, y2 * sy, dst_w, dst_h)
        for t in range(lw):
            draw.rectangle((x1 - t, y1 - t, x2 + t, y2 + t), outline=color, width=1)

        label_txt = f"c{cls_id}:{score:.2f}"
        tw, th = text_size(draw, label_txt)
        tx = max(0, int(x1))
        ty = max(0, int(y1) - th - 2 * pad)
        draw.rectangle((tx, ty, tx + tw + 2 * pad, ty + th + 2 * pad), fill=color)
        draw.text((tx + pad, ty + pad), label_txt, fill=(0, 0, 0))

    save_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(save_path, format="JPEG", quality=jpeg_quality, optimize=True)


class FasterRCNNRunner:
    def __init__(self, weights: Path, device: str):
        try:
            from torchvision.models.detection import fasterrcnn_resnet50_fpn
            from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
            from torchvision.transforms import functional as F
        except Exception as ex:  # pragma: no cover
            raise RuntimeError("torchvision is required for fasterrcnn inference.") from ex

        self._f_to_tensor = F.to_tensor
        self._device = torch.device(f"cuda:{device}" if device.isdigit() else device)

        try:
            ckpt = torch.load(weights, map_location="cpu", weights_only=False)
        except TypeError:
            ckpt = torch.load(weights, map_location="cpu")
        if not isinstance(ckpt, dict) or "model_state" not in ckpt:
            raise RuntimeError(f"Invalid Faster R-CNN checkpoint: {weights}")
        num_classes = int(ckpt.get("num_classes", 1))

        model = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes + 1)
        model.load_state_dict(ckpt["model_state"], strict=True)
        model.to(self._device).eval()
        self.model = model

    def predict_batch(self, image_paths: Sequence[Path], conf: float) -> List[Dict[str, List]]:
        inputs = [self._f_to_tensor(Image.open(p).convert("RGB")).to(self._device) for p in image_paths]
        with torch.no_grad():
            outputs = self.model(inputs)
        merged: List[Dict[str, List]] = []
        for out in outputs:
            boxes = out["boxes"].detach().cpu().numpy().tolist()
            scores = out["scores"].detach().cpu().numpy().tolist()
            labels = out["labels"].detach().cpu().numpy().tolist()
            kept_boxes: List[List[float]] = []
            kept_scores: List[float] = []
            kept_labels: List[int] = []
            for b, s, c in zip(boxes, scores, labels):
                s = float(s)
                if s < conf:
                    continue
                kept_boxes.append([float(v) for v in b[:4]])
                kept_scores.append(s)
                kept_labels.append(max(0, int(c) - 1))
            merged.append({"boxes": kept_boxes, "scores": kept_scores, "labels": kept_labels})
        return merged


class UltraRunner:
    def __init__(self, weights: Path, model_type: str):
        if model_type == "rtdetr":
            try:
                from ultralytics import RTDETR
            except Exception as ex:  # pragma: no cover
                raise RuntimeError("ultralytics is required for rtdetr inference.") from ex
            self.model = RTDETR(str(weights))
        else:
            try:
                from ultralytics import YOLO
            except Exception as ex:  # pragma: no cover
                raise RuntimeError("ultralytics is required for yolo inference.") from ex
            self.model = YOLO(str(weights))

    def predict_batch(
        self,
        image_paths: Sequence[Path],
        conf: float,
        iou: float,
        imgsz: int,
        max_det: int,
        batch: int,
        device: str,
    ) -> List[Dict[str, List]]:
        kwargs = dict(
            source=[str(p) for p in image_paths],
            conf=float(conf),
            iou=float(iou),
            imgsz=int(imgsz),
            max_det=int(max_det),
            batch=int(max(1, batch)),
            save=False,
            verbose=False,
            stream=True,
        )
        if device:
            kwargs["device"] = device

        by_path: Dict[Path, Dict[str, List]] = {}
        stream = self.model.predict(**kwargs)
        for res in stream:
            img_path = Path(str(res.path)).resolve()
            boxes_obj = res.boxes
            if boxes_obj is None or boxes_obj.xyxy is None:
                boxes: List[List[float]] = []
                scores: List[float] = []
                labels: List[int] = []
            else:
                boxes = boxes_obj.xyxy.detach().cpu().numpy().tolist()
                confs = boxes_obj.conf.detach().cpu().numpy().tolist() if boxes_obj.conf is not None else []
                clss = boxes_obj.cls.detach().cpu().numpy().tolist() if boxes_obj.cls is not None else []
                scores = [float(x) for x in confs] if confs else [1.0] * len(boxes)
                labels = [int(x) for x in clss] if clss else [0] * len(boxes)
            by_path[img_path] = {
                "boxes": [[float(v) for v in b[:4]] for b in boxes],
                "scores": scores,
                "labels": labels,
            }

        merged: List[Dict[str, List]] = []
        for p in image_paths:
            merged.append(by_path.get(p.resolve(), {"boxes": [], "scores": [], "labels": []}))
        return merged


def build_runner(spec: ModelSpec, device: str):
    if not spec.weights.exists():
        raise FileNotFoundError(f"Weight not found: {spec.weights}")
    if spec.model_type == "fasterrcnn":
        return FasterRCNNRunner(spec.weights, device=device)
    if spec.model_type in {"rtdetr", "yolo"}:
        return UltraRunner(spec.weights, model_type=spec.model_type)
    raise ValueError(f"Unsupported model type: {spec.model_type}")


def resolve_split_dir(dataset_root: Path, split_name: str) -> Tuple[str, Optional[Path]]:
    images_root = dataset_root / "images"
    direct = images_root / split_name
    if direct.exists():
        return split_name, direct

    alias_map = {"bal": "val", "val": "bal"}
    alias = alias_map.get(split_name)
    if alias:
        alt = images_root / alias
        if alt.exists():
            print(f"[warn] split '{split_name}' not found, fallback to '{alias}'.")
            return alias, alt
    return split_name, None


def run_one_model(
    runner,
    spec: ModelSpec,
    dataset_root: Path,
    requested_splits: List[str],
    out_root: Path,
    conf: float,
    iou: float,
    imgsz: int,
    max_det: int,
    batch: int,
    device: str,
    resize_ratio: float,
    jpeg_quality: int,
    line_width: int,
    max_samples: int,
) -> Dict[str, int]:
    stats: Dict[str, int] = {}
    for split_req in requested_splits:
        split_name, split_dir = resolve_split_dir(dataset_root, split_req)
        if split_dir is None:
            print(f"[warn] skip split '{split_req}': {dataset_root / 'images' / split_req} not found.")
            continue
        image_paths = list_images(split_dir)
        if max_samples > 0:
            image_paths = image_paths[:max_samples]
        if not image_paths:
            print(f"[warn] split '{split_name}' has no images, skip.")
            continue

        print(
            f"[run] model={spec.name} type={spec.model_type} split={split_name} "
            f"images={len(image_paths)} conf={conf}"
        )

        preds: List[Dict] = []
        total = len(image_paths)
        done = 0
        for chunk in iter_batches(image_paths, batch):
            if isinstance(runner, FasterRCNNRunner):
                out_chunk = runner.predict_batch(chunk, conf=conf)
            else:
                out_chunk = runner.predict_batch(
                    chunk, conf=conf, iou=iou, imgsz=imgsz, max_det=max_det, batch=batch, device=device
                )
            for img_path, pred in zip(chunk, out_chunk):
                rel = img_path.relative_to(split_dir)
                save_jpg = (out_root / "overlays" / spec.name / split_name / rel).with_suffix(".jpg")
                draw_overlay(
                    image_path=img_path,
                    boxes=pred["boxes"],
                    scores=pred["scores"],
                    labels=pred["labels"],
                    save_path=save_jpg,
                    resize_ratio=resize_ratio,
                    jpeg_quality=jpeg_quality,
                    line_width=line_width,
                )
                preds.append(
                    {
                        "image_id": img_path.stem,
                        "image_path": str(img_path.resolve()),
                        "boxes": pred["boxes"],
                        "scores": [float(s) for s in pred["scores"]],
                        "labels": [int(c) for c in pred["labels"]],
                    }
                )
                done += 1
                if done % 100 == 0 or done == total:
                    print(f"[progress] model={spec.name} split={split_name} {done}/{total}")

        pred_json = out_root / "raw_preds" / spec.name / f"{split_name}.json"
        pred_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": "det_preds_v1",
            "model": spec.name,
            "model_type": spec.model_type,
            "weights": str(spec.weights),
            "dataset_root": str(dataset_root.resolve()),
            "split": split_name,
            "predictions": preds,
        }
        pred_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[done] pred json -> {pred_json}")
        stats[split_name] = len(preds)
    return stats


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    out_root = args.out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    if not (dataset_root / "images").exists():
        raise RuntimeError(f"Invalid dataset root: {dataset_root}, missing images/ directory.")

    if not (0 < float(args.resize_ratio) <= 1.0):
        raise ValueError("--resize-ratio must be in (0, 1].")

    model_specs = [parse_model_spec(x) for x in args.model]
    requested_splits = [x.strip() for x in str(args.splits).split(",") if x.strip()]
    device = resolve_device(str(args.device).strip())

    summary = {
        "dataset_root": str(dataset_root),
        "out_root": str(out_root),
        "splits": requested_splits,
        "device": device,
        "models": [],
        "args": vars(args),
    }
    for spec in model_specs:
        print(f"[init] model={spec.name} type={spec.model_type} weights={spec.weights}")
        runner = build_runner(spec, device=device)
        model_stats = run_one_model(
            runner=runner,
            spec=spec,
            dataset_root=dataset_root,
            requested_splits=requested_splits,
            out_root=out_root,
            conf=float(args.conf),
            iou=float(args.iou),
            imgsz=int(args.imgsz),
            max_det=int(args.max_det),
            batch=int(args.batch),
            device=device,
            resize_ratio=float(args.resize_ratio),
            jpeg_quality=int(args.jpeg_quality),
            line_width=int(args.line_width),
            max_samples=int(args.max_samples),
        )
        summary["models"].append(
            {
                "name": spec.name,
                "type": spec.model_type,
                "weights": str(spec.weights),
                "num_images_by_split": model_stats,
            }
        )

    summary_path = out_root / "run_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] summary -> {summary_path}")
    print(f"[done] overlays root -> {out_root / 'overlays'}")


if __name__ == "__main__":
    main()
