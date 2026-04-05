#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import torch
from PIL import Image
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(THIS_DIR))

from dataset import list_images  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Faster R-CNN inference and export predictions to JSON.")
    p.add_argument("--weights", type=Path, required=True, help="Checkpoint path from train.py (best.pt/last.pt).")
    p.add_argument("--dataset-root", type=Path, required=True, help="Dataset root, e.g. dataset/yolo/datasetm6c")
    p.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    p.add_argument("--output-json", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-classes", type=int, default=1, help="Foreground class count if missing in checkpoint.")
    p.add_argument("--conf", type=float, default=0.001, help="Score threshold for exported predictions.")
    p.add_argument("--device", type=str, default="")
    p.add_argument("--max-samples", type=int, default=0, help="Only run first N images.")
    return p.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg:
        if device_arg.isdigit():
            return torch.device(f"cuda:{device_arg}")
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(num_classes: int) -> torch.nn.Module:
    model = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes + 1)
    return model


def batched(items: List[Path], batch_size: int):
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def main() -> None:
    args = parse_args()
    try:
        ckpt = torch.load(args.weights, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(args.weights, map_location="cpu")
    if not isinstance(ckpt, dict) or "model_state" not in ckpt:
        raise RuntimeError(f"Invalid checkpoint format: {args.weights}")

    num_classes = int(ckpt.get("num_classes", args.num_classes))
    model = build_model(num_classes=num_classes)
    model.load_state_dict(ckpt["model_state"], strict=True)

    device = resolve_device(args.device)
    model.to(device).eval()

    image_dir = args.dataset_root.resolve() / "images" / args.split
    image_paths = list_images(image_dir)
    if args.max_samples > 0:
        image_paths = image_paths[: args.max_samples]

    results = []
    with torch.no_grad():
        for chunk in batched(image_paths, int(args.batch_size)):
            images = [F.to_tensor(Image.open(p).convert("RGB")).to(device) for p in chunk]
            outputs = model(images)
            for img_path, out in zip(chunk, outputs):
                boxes = out["boxes"].detach().cpu().numpy().tolist()
                scores = out["scores"].detach().cpu().numpy().tolist()
                labels = out["labels"].detach().cpu().numpy().tolist()

                kept_boxes = []
                kept_scores = []
                kept_labels = []
                for box, score, label in zip(boxes, scores, labels):
                    score_f = float(score)
                    if score_f < float(args.conf):
                        continue
                    kept_boxes.append([float(v) for v in box[:4]])
                    kept_scores.append(score_f)
                    kept_labels.append(max(0, int(label) - 1))

                results.append(
                    {
                        "image_id": img_path.stem,
                        "image_path": str(img_path.resolve()),
                        "boxes": kept_boxes,
                        "scores": kept_scores,
                        "labels": kept_labels,
                    }
                )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "det_preds_v1",
        "model": "faster_rcnn",
        "weights": str(args.weights.resolve()),
        "dataset_root": str(args.dataset_root.resolve()),
        "split": args.split,
        "predictions": results,
    }
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] wrote {len(results)} image predictions -> {args.output_json}")


if __name__ == "__main__":
    main()
