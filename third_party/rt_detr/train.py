#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import List

from ultralytics import RTDETR


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train RT-DETR on datasetm6c-style YOLO dataset.")
    p.add_argument("--dataset-root", type=Path, required=True, help="Dataset root, e.g. dataset/yolo/datasetm6c")
    p.add_argument("--output-dir", type=Path, required=True, help="Output dir for run and copied best/last weights.")
    p.add_argument("--model", type=str, default="rtdetr-l.yaml", help="Ultralytics model yaml/pt.")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--num-classes", type=int, default=1)
    p.add_argument("--names", type=str, default="defect", help='Comma separated class names, e.g. "defect".')
    p.add_argument("--device", type=str, default="", help='GPU id like "0" or explicit "cuda:0"/"cpu".')
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--patience", type=int, default=0)
    p.add_argument("--cache", action="store_true")
    p.add_argument("--pretrained", action="store_true", help="Use Ultralytics pretrained init when supported.")
    return p.parse_args()


def parse_names(csv_text: str, num_classes: int) -> List[str]:
    names = [x.strip() for x in csv_text.split(",") if x.strip()]
    if not names:
        names = [f"class_{i}" for i in range(num_classes)]
    if len(names) != num_classes:
        if len(names) == 1 and num_classes == 1:
            return names
        raise ValueError(f"names count ({len(names)}) must match num_classes ({num_classes})")
    return names


def write_data_yaml(dataset_root: Path, out_dir: Path, num_classes: int, names: List[str]) -> Path:
    yaml_text = "\n".join(
        [
            f"path: {dataset_root.resolve()}",
            "train: images/train",
            "val: images/val",
            "test: images/test",
            f"nc: {int(num_classes)}",
            "names:",
            *[f"  - {name}" for name in names],
            "",
        ]
    )
    data_yaml = out_dir / "data.yaml"
    data_yaml.write_text(yaml_text, encoding="utf-8")
    return data_yaml


def main() -> None:
    args = parse_args()
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if not args.dataset_root.exists():
        raise FileNotFoundError(f"dataset root not found: {args.dataset_root}")

    names = parse_names(args.names, int(args.num_classes))
    data_yaml = write_data_yaml(args.dataset_root, out_dir, int(args.num_classes), names)

    model = RTDETR(args.model)
    train_kwargs = dict(
        data=str(data_yaml),
        epochs=int(args.epochs),
        imgsz=int(args.imgsz),
        batch=int(args.batch),
        workers=int(args.workers),
        project=str(out_dir),
        name="train",
        seed=int(args.seed),
        patience=max(0, int(args.patience)),
        verbose=True,
    )
    model_name = str(args.model).lower()
    if bool(args.pretrained):
        train_kwargs["pretrained"] = True
    elif model_name.endswith(".yaml") or model_name.endswith(".yml"):
        train_kwargs["pretrained"] = False
    if args.device:
        train_kwargs["device"] = args.device
    if args.cache:
        train_kwargs["cache"] = True

    model.train(**train_kwargs)

    run_dir = Path(getattr(model.trainer, "save_dir", out_dir / "train")).resolve()
    best_src = run_dir / "weights" / "best.pt"
    last_src = run_dir / "weights" / "last.pt"
    best_dst = out_dir / "best.pt"
    last_dst = out_dir / "last.pt"
    if best_src.exists():
        shutil.copy2(best_src, best_dst)
    if last_src.exists():
        shutil.copy2(last_src, last_dst)

    summary = {
        "dataset_root": str(args.dataset_root.resolve()),
        "data_yaml": str(data_yaml),
        "run_dir": str(run_dir),
        "best_weight": str(best_dst if best_dst.exists() else best_src),
        "last_weight": str(last_dst if last_dst.exists() else last_src),
        "model": args.model,
        "epochs": int(args.epochs),
        "imgsz": int(args.imgsz),
        "batch": int(args.batch),
        "device": args.device,
    }
    (out_dir / "train_meta.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] run_dir={run_dir}")
    print(f"[done] best={summary['best_weight']}")


if __name__ == "__main__":
    main()
