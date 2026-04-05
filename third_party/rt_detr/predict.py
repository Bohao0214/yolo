#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

from ultralytics import RTDETR

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run RT-DETR inference and export predictions to JSON.")
    p.add_argument("--weights", type=Path, required=True, help="RT-DETR .pt weight path.")
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    p.add_argument("--output-json", type=Path, required=True)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.001)
    p.add_argument("--iou", type=float, default=0.7)
    p.add_argument("--max-det", type=int, default=300)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--device", type=str, default="")
    p.add_argument("--max-samples", type=int, default=0)
    return p.parse_args()


def list_images(image_dir: Path) -> List[Path]:
    return sorted([p for p in image_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS])


def batched(items: List[Path], batch_size: int):
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def main() -> None:
    args = parse_args()
    model = RTDETR(str(args.weights))

    image_dir = args.dataset_root.resolve() / "images" / args.split
    image_paths = list_images(image_dir)
    if args.max_samples > 0:
        image_paths = image_paths[: args.max_samples]

    results = []
    for chunk in batched(image_paths, int(args.batch)):
        predict_kwargs = dict(
            source=[str(p) for p in chunk],
            imgsz=int(args.imgsz),
            conf=float(args.conf),
            iou=float(args.iou),
            max_det=int(args.max_det),
            batch=int(args.batch),
            save=False,
            verbose=False,
            stream=True,
        )
        if args.device:
            predict_kwargs["device"] = args.device
        stream = model.predict(**predict_kwargs)
        for res in stream:
            img_path = Path(res.path).resolve()
            boxes_obj = res.boxes
            if boxes_obj is None or boxes_obj.xyxy is None:
                boxes = []
                scores = []
                labels = []
            else:
                boxes = boxes_obj.xyxy.detach().cpu().numpy().tolist()
                scores = boxes_obj.conf.detach().cpu().numpy().tolist() if boxes_obj.conf is not None else [1.0] * len(boxes)
                labels = boxes_obj.cls.detach().cpu().numpy().tolist() if boxes_obj.cls is not None else [0] * len(boxes)
            results.append(
                {
                    "image_id": img_path.stem,
                    "image_path": str(img_path),
                    "boxes": [[float(v) for v in b[:4]] for b in boxes],
                    "scores": [float(s) for s in scores],
                    "labels": [int(c) for c in labels],
                }
            )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "det_preds_v1",
        "model": "rt_detr",
        "weights": str(args.weights.resolve()),
        "dataset_root": str(args.dataset_root.resolve()),
        "split": args.split,
        "predictions": results,
    }
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] wrote {len(results)} image predictions -> {args.output_json}")


if __name__ == "__main__":
    main()
