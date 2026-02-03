"""Merge multiple images into a single grid image.

Example:
python analyzepy/merge_images_grid.py \
  --images /path/a.jpg /path/b.jpg /path/c.jpg \
  --rows 2 --cols 2 \
  --out /tmp/grid.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge images into a grid and save one output image.")
    parser.add_argument("--images", nargs="+", required=True, help="List of image paths")
    parser.add_argument("--rows", type=int, required=True, help="Grid rows")
    parser.add_argument("--cols", type=int, required=True, help="Grid cols")
    parser.add_argument("--out", type=str, required=True, help="Output image path")
    parser.add_argument("--bg", type=int, nargs=3, default=[0, 0, 0], help="Background color B G R")
    parser.add_argument("--keep_aspect", action="store_true", help="Keep aspect ratio with padding")
    return parser.parse_args()


def read_images(paths: List[str]) -> List[np.ndarray]:
    images: List[np.ndarray] = []
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            raise FileNotFoundError(f"Failed to read image: {p}")
        images.append(img)
    return images


def compute_cell_size(images: List[np.ndarray]) -> Tuple[int, int]:
    heights = [img.shape[0] for img in images]
    widths = [img.shape[1] for img in images]
    return max(widths), max(heights)


def resize_into_cell(
    img: np.ndarray,
    cell_w: int,
    cell_h: int,
    keep_aspect: bool,
    bg_color: Tuple[int, int, int],
) -> np.ndarray:
    if not keep_aspect:
        return cv2.resize(img, (cell_w, cell_h), interpolation=cv2.INTER_AREA)

    h, w = img.shape[:2]
    scale = min(cell_w / w, cell_h / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
    canvas[:] = bg_color
    x0 = (cell_w - new_w) // 2
    y0 = (cell_h - new_h) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return canvas


def main() -> None:
    args = parse_args()

    if args.rows <= 0 or args.cols <= 0:
        raise ValueError("rows and cols must be positive integers")

    images = read_images(args.images)
    max_slots = args.rows * args.cols
    if len(images) > max_slots:
        print(f"[warn] got {len(images)} images but grid has {max_slots} slots; extra images will be ignored")
        images = images[:max_slots]

    cell_w, cell_h = compute_cell_size(images)
    bg_color = tuple(int(v) for v in args.bg)

    grid_h = args.rows * cell_h
    grid_w = args.cols * cell_w
    canvas = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
    canvas[:] = bg_color

    for idx, img in enumerate(images):
        row = idx // args.cols
        col = idx % args.cols
        cell_img = resize_into_cell(img, cell_w, cell_h, args.keep_aspect, bg_color)
        y1 = row * cell_h
        x1 = col * cell_w
        canvas[y1 : y1 + cell_h, x1 : x1 + cell_w] = cell_img

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out_path), canvas):
        raise RuntimeError(f"Failed to write output: {out_path}")

    print(f"Saved grid image to: {out_path}")


if __name__ == "__main__":
    main()

"""""
python analyzepy/merge_images_grid.py \
  --images \
    /home/ubuntu/project/deduibi/yolo/analysis/report_260127173548/crops/FP_strict/0030_pred0_s0.674.png \
    /home/ubuntu/project/deduibi/yolo/analysis/report_260127173548/crops/FP_strict/0068_pred4_s0.371.png \
    /home/ubuntu/project/deduibi/yolo/analysis/report_260127173548/crops/FP_strict/0172_pred2_s0.296.png \
    /home/ubuntu/project/deduibi/yolo/analysis/report_260127173548/crops/FP_strict/0226_pred1_s0.883.png \
    /home/ubuntu/project/deduibi/yolo/analysis/report_260127173548/crops/FP_strict/0371_pred2_s0.496.png \
    /home/ubuntu/project/deduibi/yolo/analysis/report_260127173548/crops/FP_strict/0414_pred0_s0.852.png \
  --rows 2 --cols 3 \
  --out /home/ubuntu/project/deduibi/yolo/analysis/report_260127173548/crops/FP_strict/grid_fp_strict.png \
  --keep_aspect
"""""
