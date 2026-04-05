#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as F

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def list_images(image_dir: Path) -> List[Path]:
    return sorted([p for p in image_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS])


def xywhn_to_xyxy(xc: float, yc: float, w: float, h: float, img_w: int, img_h: int) -> Tuple[float, float, float, float]:
    x1 = (xc - w / 2.0) * img_w
    y1 = (yc - h / 2.0) * img_h
    x2 = (xc + w / 2.0) * img_w
    y2 = (yc + h / 2.0) * img_h
    return float(x1), float(y1), float(x2), float(y2)


def seg_norm_to_xyxy(coords: Sequence[float], img_w: int, img_h: int) -> Tuple[float, float, float, float]:
    xs = [float(v) * img_w for v in coords[0::2]]
    ys = [float(v) * img_h for v in coords[1::2]]
    return float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))


def parse_yolo_label_file(
    label_path: Path,
    img_w: int,
    img_h: int,
    class_offset: int = 1,
) -> Tuple[List[Tuple[float, float, float, float]], List[int]]:
    boxes: List[Tuple[float, float, float, float]] = []
    labels: List[int] = []

    if not label_path.exists():
        return boxes, labels

    for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        try:
            cls_id = int(float(parts[0]))
            coords = [float(x) for x in parts[1:]]
        except ValueError:
            continue

        if len(coords) > 4 and len(coords) % 2 == 0:
            x1, y1, x2, y2 = seg_norm_to_xyxy(coords, img_w, img_h)
        else:
            if len(coords) < 4:
                continue
            x1, y1, x2, y2 = xywhn_to_xyxy(coords[0], coords[1], coords[2], coords[3], img_w, img_h)

        x1 = max(0.0, min(float(img_w), x1))
        x2 = max(0.0, min(float(img_w), x2))
        y1 = max(0.0, min(float(img_h), y1))
        y2 = max(0.0, min(float(img_h), y2))
        if x2 <= x1 or y2 <= y1:
            continue
        boxes.append((x1, y1, x2, y2))
        labels.append(cls_id + class_offset)

    return boxes, labels


def infer_num_classes(label_dir: Path) -> int:
    max_cls = -1
    for txt in sorted(label_dir.rglob("*.txt")):
        for line in txt.read_text(encoding="utf-8", errors="ignore").splitlines():
            parts = line.strip().split()
            if not parts:
                continue
            try:
                cls_id = int(float(parts[0]))
            except ValueError:
                continue
            if cls_id > max_cls:
                max_cls = cls_id
    return max(1, max_cls + 1)


class YoloDetectionDataset(Dataset):
    def __init__(self, dataset_root: Path, split: str, max_samples: int = 0) -> None:
        self.dataset_root = Path(dataset_root).resolve()
        self.split = split
        self.image_dir = self.dataset_root / "images" / split
        self.label_dir = self.dataset_root / "labels" / split
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image dir not found: {self.image_dir}")
        if not self.label_dir.exists():
            raise FileNotFoundError(f"Label dir not found: {self.label_dir}")

        image_paths = list_images(self.image_dir)
        if max_samples > 0:
            image_paths = image_paths[:max_samples]
        self.image_paths = image_paths

    def __len__(self) -> int:
        return len(self.image_paths)

    def get_image_path(self, idx: int) -> Path:
        return self.image_paths[idx]

    def __getitem__(self, idx: int):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        img_w, img_h = image.size
        label_path = self.label_dir / f"{img_path.stem}.txt"
        boxes, labels = parse_yolo_label_file(label_path=label_path, img_w=img_w, img_h=img_h, class_offset=1)

        if boxes:
            boxes_t = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
            labels_t = torch.as_tensor(labels, dtype=torch.int64)
            area = (boxes_t[:, 2] - boxes_t[:, 0]) * (boxes_t[:, 3] - boxes_t[:, 1])
        else:
            boxes_t = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros((0,), dtype=torch.int64)
            area = torch.zeros((0,), dtype=torch.float32)

        target = {
            "boxes": boxes_t,
            "labels": labels_t,
            "image_id": torch.tensor([idx], dtype=torch.int64),
            "iscrowd": torch.zeros((labels_t.shape[0],), dtype=torch.int64),
            "area": area,
        }
        return F.to_tensor(image), target


def collate_fn(batch):
    return tuple(zip(*batch))
