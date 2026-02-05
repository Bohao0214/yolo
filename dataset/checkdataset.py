#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.patheffects as patheffects
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.widgets import RectangleSelector
import numpy as np
from PIL import Image

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def list_images(root: Path) -> List[Path]:
    images: List[Path] = []
    for split in ["train", "val", "test"]:
        img_dir = root / "images" / split
        if not img_dir.exists():
            continue
        for p in sorted(img_dir.rglob("*")):
            if p.is_file() and p.suffix.lower() in IMG_EXTS:
                images.append(p)

    if not images:
        img_dir = root / "images"
        if img_dir.exists():
            for p in sorted(img_dir.rglob("*")):
                if p.is_file() and p.suffix.lower() in IMG_EXTS:
                    images.append(p)

    return images


def rel_image_path(image_path: Path, root: Path) -> Path:
    try:
        return image_path.relative_to(root)
    except ValueError:
        return image_path


def label_path_for(image_path: Path, root: Path, labels_root: Path) -> Path:
    try:
        rel = image_path.relative_to(root / "images")
    except ValueError:
        rel = image_path.relative_to(root)
    return labels_root / rel.with_suffix(".txt")


def load_image(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        return np.array(img.convert("RGB"))


def parse_yolo_labels(path: Path) -> List[Tuple[int, float, float, float, float]]:
    boxes: List[Tuple[int, float, float, float, float]] = []
    if not path.exists():
        return boxes
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                cls = int(float(parts[0]))
                cx, cy, w, h = map(float, parts[1:5])
            except ValueError:
                continue
            boxes.append((cls, cx, cy, w, h))
    return boxes


def save_yolo_labels(
    path: Path, boxes: List[Tuple[int, float, float, float, float]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not boxes:
        path.write_text("", encoding="utf-8")
        return
    lines = [f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}" for cls, cx, cy, w, h in boxes]
    path.write_text("\n".join(lines), encoding="utf-8")


def xyxy_to_yolo(
    x0: float, y0: float, x1: float, y1: float, width: int, height: int
) -> Tuple[int, float, float, float, float]:
    x_min = max(0.0, min(x0, x1))
    y_min = max(0.0, min(y0, y1))
    x_max = min(float(width), max(x0, x1))
    y_max = min(float(height), max(y0, y1))
    w = max(1.0, x_max - x_min)
    h = max(1.0, y_max - y_min)
    cx = (x_min + x_max) / 2.0 / width
    cy = (y_min + y_max) / 2.0 / height
    nw = w / width
    nh = h / height
    cx = min(max(cx, 0.0), 1.0)
    cy = min(max(cy, 0.0), 1.0)
    nw = min(max(nw, 0.0), 1.0)
    nh = min(max(nh, 0.0), 1.0)
    return 0, cx, cy, nw, nh


def yolo_to_xyxy(
    box: Tuple[int, float, float, float, float], width: int, height: int
) -> Tuple[float, float, float, float]:
    _, cx, cy, w, h = box
    x0 = (cx - w / 2.0) * width
    y0 = (cy - h / 2.0) * height
    x1 = (cx + w / 2.0) * width
    y1 = (cy + h / 2.0) * height
    return x0, y0, x1, y1


class YoloReviewer:
    def __init__(self, root: Path, out_root: Path) -> None:
        self.root = root
        self.out_root = out_root
        self.images = list_images(root)
        if not self.images:
            raise RuntimeError(f"No images found under {root / 'images'}")
        self.index = 0

        self.boxes_cache: Dict[Path, List[Tuple[int, float, float, float, float]]] = {}
        self.dirty: set[Path] = set()

        self.current_image: Optional[np.ndarray] = None
        self.current_path: Optional[Path] = None
        self.current_hw: Tuple[int, int] = (1, 1)
        self.pending_box: Optional[Tuple[float, float, float, float]] = None
        self.pending_patch: Optional[Rectangle] = None
        self.box_patches: List[Rectangle] = []

        self.fig, self.ax = plt.subplots()
        self.fig.patch.set_facecolor("white")
        self.ax.set_facecolor("white")
        self.ax.axis("off")
        self.image_artist = None

        self.text_effects = [patheffects.withStroke(linewidth=3, foreground="black")]
        self.info_text = self.ax.text(
            0.01,
            0.99,
            "",
            transform=self.ax.transAxes,
            va="top",
            ha="left",
            color="white",
            fontsize=10,
            path_effects=self.text_effects,
        )
        self.path_text = self.ax.text(
            0.01,
            0.01,
            "",
            transform=self.ax.transAxes,
            va="bottom",
            ha="left",
            color="white",
            fontsize=10,
            path_effects=self.text_effects,
        )

        self._init_selector()
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)

        self.load_current()

    def _init_selector(self) -> None:
        props = dict(edgecolor="yellow", linewidth=1.5, fill=False, linestyle="--")
        try:
            self.selector = RectangleSelector(
                self.ax,
                self.on_select,
                useblit=True,
                button=[1],
                minspanx=5,
                minspany=5,
                spancoords="pixels",
                props=props,
            )
        except TypeError:
            self.selector = RectangleSelector(
                self.ax,
                self.on_select,
                useblit=True,
                button=[1],
                minspanx=5,
                minspany=5,
                rectprops=props,
            )

    def load_current(self) -> None:
        path = self.images[self.index]
        self.current_path = path
        img = load_image(path)
        self.current_image = img
        height, width = img.shape[:2]
        self.current_hw = (height, width)

        if self.image_artist is None:
            self.image_artist = self.ax.imshow(img)
        else:
            self.image_artist.set_data(img)
        self.image_artist.set_extent((0, width, height, 0))
        self.ax.set_xlim(0, width)
        self.ax.set_ylim(height, 0)

        self.pending_box = None
        if self.pending_patch is not None:
            self.pending_patch.remove()
            self.pending_patch = None

        boxes = self.get_boxes(path)
        self.draw_boxes(boxes)
        self.update_overlay()
        self.fig.canvas.draw_idle()

    def get_boxes(self, path: Path) -> List[Tuple[int, float, float, float, float]]:
        if path in self.boxes_cache:
            return self.boxes_cache[path]

        out_label = label_path_for(path, self.root, self.out_root / "labels")
        src_label = label_path_for(path, self.root, self.root / "labels")
        label_path = out_label if out_label.exists() else src_label
        boxes = parse_yolo_labels(label_path)
        self.boxes_cache[path] = boxes
        return boxes

    def draw_boxes(self, boxes: List[Tuple[int, float, float, float, float]]) -> None:
        for patch in self.box_patches:
            patch.remove()
        self.box_patches = []

        height, width = self.current_hw
        for box in boxes:
            x0, y0, x1, y1 = yolo_to_xyxy(box, width, height)
            rect = Rectangle(
                (x0, y0),
                x1 - x0,
                y1 - y0,
                linewidth=2,
                edgecolor="lime",
                facecolor="none",
            )
            self.ax.add_patch(rect)
            self.box_patches.append(rect)

    def update_overlay(self) -> None:
        if self.current_path is None:
            return
        rel = rel_image_path(self.current_path, self.root)
        dirty_mark = "*" if self.current_path in self.dirty else ""
        self.fig.suptitle(
            f"{dirty_mark}{rel.as_posix()} ({self.index + 1}/{len(self.images)})",
            fontsize=12,
        )
        self.info_text.set_text(
            "Keys: q prev | e next | r clear | b confirm | u save | j jump | esc quit\n"
            "Mouse: drag to draw box | boxes = defect | no box = normal"
        )
        self.path_text.set_text(f"File: {rel.as_posix()}")

    def on_select(self, eclick, erelease) -> None:
        if eclick.xdata is None or erelease.xdata is None:
            return
        x0, y0 = float(eclick.xdata), float(eclick.ydata)
        x1, y1 = float(erelease.xdata), float(erelease.ydata)
        if abs(x1 - x0) < 2 or abs(y1 - y0) < 2:
            return
        self.pending_box = (x0, y0, x1, y1)
        self.update_pending_patch()

    def update_pending_patch(self) -> None:
        if self.pending_box is None:
            if self.pending_patch is not None:
                self.pending_patch.remove()
                self.pending_patch = None
            self.fig.canvas.draw_idle()
            return
        x0, y0, x1, y1 = self.pending_box
        x_min = min(x0, x1)
        y_min = min(y0, y1)
        width = abs(x1 - x0)
        height = abs(y1 - y0)
        if self.pending_patch is None:
            self.pending_patch = Rectangle(
                (x_min, y_min),
                width,
                height,
                linewidth=1.5,
                edgecolor="yellow",
                facecolor="none",
                linestyle="--",
            )
            self.ax.add_patch(self.pending_patch)
        else:
            self.pending_patch.set_xy((x_min, y_min))
            self.pending_patch.set_width(width)
            self.pending_patch.set_height(height)
        self.fig.canvas.draw_idle()

    def confirm_pending(self) -> None:
        if self.pending_box is None or self.current_path is None:
            return
        height, width = self.current_hw
        box = xyxy_to_yolo(*self.pending_box, width=width, height=height)
        boxes = self.get_boxes(self.current_path)
        boxes.append(box)
        self.dirty.add(self.current_path)
        self.pending_box = None
        self.update_pending_patch()
        self.draw_boxes(boxes)
        self.update_overlay()
        self.fig.canvas.draw_idle()

    def clear_boxes(self) -> None:
        if self.current_path is None:
            return
        self.boxes_cache[self.current_path] = []
        self.dirty.add(self.current_path)
        self.pending_box = None
        self.update_pending_patch()
        self.draw_boxes([])
        self.update_overlay()
        self.fig.canvas.draw_idle()

    def save_current(self) -> None:
        if self.current_path is None:
            return
        try:
            rel = self.current_path.relative_to(self.root / "images")
        except ValueError:
            rel = self.current_path.relative_to(self.root)
        out_image = self.out_root / "images" / rel
        out_label = self.out_root / "labels" / rel.with_suffix(".txt")
        out_image.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self.current_path, out_image)
        save_yolo_labels(out_label, self.get_boxes(self.current_path))
        self.dirty.discard(self.current_path)
        self.update_overlay()
        self.fig.canvas.draw_idle()

    def jump_to_path(self) -> None:
        if self.current_path is None:
            return
        prompt = (
            "Jump to image path (absolute or relative to dataset root, "
            "e.g. images/train/xxx.jpg): "
        )
        target = input(prompt).strip()
        if not target:
            return
        candidate = Path(target)
        if not candidate.is_absolute():
            candidate = (self.root / target).resolve()
        if candidate in self.images:
            self.index = self.images.index(candidate)
            self.load_current()
            return
        alt = (self.root / "images" / target).resolve()
        if alt in self.images:
            self.index = self.images.index(alt)
            self.load_current()
            return
        name_matches = [p for p in self.images if p.name == Path(target).name]
        if len(name_matches) == 1:
            self.index = self.images.index(name_matches[0])
            self.load_current()
            return
        print(f"Path not found in dataset: {target}")

    def on_key(self, event) -> None:
        if event.key in {"q"}:
            self.index = (self.index - 1) % len(self.images)
            self.load_current()
        elif event.key in {"e"}:
            self.index = (self.index + 1) % len(self.images)
            self.load_current()
        elif event.key in {"r"}:
            self.clear_boxes()
        elif event.key in {"b"}:
            self.confirm_pending()
        elif event.key in {"u"}:
            self.save_current()
        elif event.key in {"j"}:
            self.jump_to_path()
        elif event.key in {"escape", "esc"}:
            plt.close(self.fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="YOLO dataset reviewer and label editor (matplotlib)."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("dataset/raw/dataset1"),
        help="Input dataset root (default: dataset/raw/dataset1)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("dataset/raw/dataset1c"),
        help="Output dataset root (default: dataset/raw/dataset1c)",
    )
    args = parser.parse_args()

    reviewer = YoloReviewer(args.root.resolve(), args.out.resolve())
    plt.show()


if __name__ == "__main__":
    main()
