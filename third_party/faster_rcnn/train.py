#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path
from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights, fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(THIS_DIR))

from dataset import YoloDetectionDataset, collate_fn, infer_num_classes  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Faster R-CNN on YOLO-format dataset.")
    p.add_argument("--dataset-root", type=Path, required=True, help="Dataset root, e.g. dataset/yolo/datasetm6c")
    p.add_argument("--output-dir", type=Path, required=True, help="Output directory.")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=0.005)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--num-classes", type=int, default=1, help="Foreground classes count.")
    p.add_argument("--pretrained-coco", action="store_true", help="Use torchvision COCO pretrained weights.")
    p.add_argument("--device", type=str, default="", help='cuda device like "0", "cuda:0", or "cpu".')
    p.add_argument("--max-train-samples", type=int, default=0, help="Use only first N train images for quick checks.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--print-freq", type=int, default=20)
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg:
        if device_arg.isdigit():
            return torch.device(f"cuda:{device_arg}")
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(num_classes: int, pretrained_coco: bool) -> torch.nn.Module:
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained_coco else None
    model = fasterrcnn_resnet50_fpn(weights=weights, weights_backbone=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=num_classes + 1)  # +1 background
    return model


def move_targets_to_device(targets, device: torch.device):
    moved = []
    for t in targets:
        moved.append({k: (v.to(device) if torch.is_tensor(v) else v) for k, v in t.items()})
    return moved


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    print_freq: int,
) -> Tuple[float, Dict[str, float]]:
    model.train()
    loss_total = 0.0
    n_steps = 0
    parts_total: Dict[str, float] = {}

    for step, (images, targets) in enumerate(loader, start=1):
        images = [img.to(device) for img in images]
        targets = move_targets_to_device(targets, device)

        loss_dict = model(images, targets)
        losses = sum(loss_dict.values())
        if not torch.isfinite(losses):
            raise RuntimeError(f"Non-finite loss detected: {float(losses.detach().cpu())}")

        optimizer.zero_grad(set_to_none=True)
        losses.backward()
        optimizer.step()

        loss_v = float(losses.detach().cpu().item())
        loss_total += loss_v
        n_steps += 1
        for k, v in loss_dict.items():
            parts_total[k] = parts_total.get(k, 0.0) + float(v.detach().cpu().item())

        if print_freq > 0 and step % print_freq == 0:
            print(f"[train] step={step}/{len(loader)} loss={loss_v:.6f}")

    avg_loss = loss_total / max(n_steps, 1)
    avg_parts = {k: v / max(n_steps, 1) for k, v in parts_total.items()}
    return avg_loss, avg_parts


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    train_set = YoloDetectionDataset(args.dataset_root, split="train", max_samples=args.max_train_samples)
    if len(train_set) == 0:
        raise RuntimeError("Empty train split.")

    inferred_num_classes = infer_num_classes(train_set.label_dir)
    num_classes = int(args.num_classes)
    if num_classes < inferred_num_classes:
        print(
            f"[warn] --num-classes={num_classes} is smaller than inferred={inferred_num_classes}, use inferred value."
        )
        num_classes = inferred_num_classes

    loader = DataLoader(
        train_set,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
    )

    device = resolve_device(args.device)
    model = build_model(num_classes=num_classes, pretrained_coco=bool(args.pretrained_coco)).to(device)
    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(args.lr),
        momentum=float(args.momentum),
        weight_decay=float(args.weight_decay),
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, args.epochs // 3), gamma=0.1)

    history_path = out_dir / "train_log.csv"
    with history_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "lr", "loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg", "is_best"])

    best_loss = float("inf")
    for epoch in range(1, int(args.epochs) + 1):
        t0 = time.time()
        train_loss, parts = train_one_epoch(model, loader, optimizer, device, print_freq=int(args.print_freq))
        scheduler.step()
        is_best = train_loss < best_loss
        if is_best:
            best_loss = train_loss

        ckpt = {
            "epoch": epoch,
            "num_classes": int(num_classes),
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "args": vars(args),
            "best_loss": float(best_loss),
        }
        torch.save(ckpt, out_dir / "last.pt")
        if is_best:
            torch.save(ckpt, out_dir / "best.pt")

        lr = float(optimizer.param_groups[0]["lr"])
        with history_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    epoch,
                    f"{train_loss:.8f}",
                    f"{lr:.8f}",
                    f"{parts.get('loss_classifier', 0.0):.8f}",
                    f"{parts.get('loss_box_reg', 0.0):.8f}",
                    f"{parts.get('loss_objectness', 0.0):.8f}",
                    f"{parts.get('loss_rpn_box_reg', 0.0):.8f}",
                    int(is_best),
                ]
            )

        dt = time.time() - t0
        print(
            f"[epoch {epoch:03d}] loss={train_loss:.6f} best={best_loss:.6f} lr={lr:.6g} time={dt:.1f}s"
        )

    (out_dir / "train_meta.json").write_text(
        json.dumps(
            {
                "dataset_root": str(args.dataset_root.resolve()),
                "num_classes": int(num_classes),
                "inferred_num_classes": int(inferred_num_classes),
                "epochs": int(args.epochs),
                "best_loss": float(best_loss),
                "device": str(device),
                "output_dir": str(out_dir),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"[done] best checkpoint: {out_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
