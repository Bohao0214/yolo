import os
import shutil
from typing import List, Tuple

import cv2


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def copy_image(src: str, dst: str) -> None:
    if os.path.abspath(src) == os.path.abspath(dst):
        return
    if not os.path.exists(dst):
        shutil.copy2(src, dst)


def yolo_line(x: int, y: int, w: int, h: int, img_w: int, img_h: int) -> str:
    cx = (x + w / 2.0) / img_w
    cy = (y + h / 2.0) / img_h
    bw = w / img_w
    bh = h / img_h
    return f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def mask_to_boxes(mask_path: str, min_area: int) -> List[Tuple[int, int, int, int]]:
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return []
    _, bin_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    bin_mask = cv2.morphologyEx(bin_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes: List[Tuple[int, int, int, int]] = []
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 0 and h > 0:
            boxes.append((x, y, w, h))
    return boxes


def write_labels(label_path: str, lines: List[str]) -> None:
    with open(label_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def resolve_mask_path(src_root: str, name: str) -> str:
    base = os.path.splitext(name)[0]
    direct = os.path.join(src_root, "ground_truth", "abnormal", f"{base}.png")
    if os.path.exists(direct):
        return direct
    mask_suffix = os.path.join(src_root, "ground_truth", "abnormal", f"{base}_mask.png")
    if os.path.exists(mask_suffix):
        return mask_suffix
    return direct


def process_split(src_root: str, dst_root: str, split: str, min_area: int) -> None:
    for cls in ["good", "abnormal"]:
        src_dir = os.path.join(src_root, split, cls)
        if not os.path.isdir(src_dir):
            continue
        for name in sorted(os.listdir(src_dir)):
            if not name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                continue
            src_img_path = os.path.join(src_dir, name)
            img = cv2.imread(src_img_path)
            if img is None:
                print(f"Skip unreadable image: {src_img_path}")
                continue
            h, w = img.shape[:2]

            dst_img_dir = os.path.join(dst_root, "images", split)
            dst_label_dir = os.path.join(dst_root, "labels", split)
            ensure_dir(dst_img_dir)
            ensure_dir(dst_label_dir)

            dst_img_path = os.path.join(dst_img_dir, name)
            dst_label_path = os.path.join(dst_label_dir, os.path.splitext(name)[0] + ".txt")

            label_lines: List[str] = []
            if cls == "abnormal":
                mask_path = resolve_mask_path(src_root, name)
                boxes = mask_to_boxes(mask_path, min_area)
                for x, y, bw, bh in boxes:
                    label_lines.append(yolo_line(x, y, bw, bh, w, h))

            write_labels(dst_label_path, label_lines)
            copy_image(src_img_path, dst_img_path)


def main() -> None:
    src_root = "/home/ubuntu/project/iad/Sealant_Task/GLASS/datasets/mvtec/mvtec-mirror6"
    dst_root = "/home/ubuntu/project/deduibi/yolo/dataset/datasetm6"
    min_area = 15

    if not os.path.isdir(src_root):
        raise SystemExit(f"Source not found: {src_root}")

    ensure_dir(dst_root)
    process_split(src_root, dst_root, "train", min_area)
    process_split(src_root, dst_root, "test", min_area)

    print(f"Done. YOLO dataset saved to {dst_root}")


if __name__ == "__main__":
    main()
