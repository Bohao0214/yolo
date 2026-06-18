from __future__ import annotations

from pathlib import Path
from typing import Iterable

try:
    from func import get_detector  # run from parent: python -m func.test_batch_infer
except Exception:
    from sd_yolo11_detector import get_detector  # run inside func: python test_batch_infer.py


def _iter_images(folder: Path) -> Iterable[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    for p in sorted(folder.iterdir()):
        if p.is_file() and p.suffix.lower() in exts:
            yield p


def main() -> None:
    image_dir = Path("/home/ubuntu/hpproject/yolo/dataset/yolo/datasetm6c/images/val")
    weights = Path("/home/ubuntu/hpproject/yolo/best.pt")

    if not image_dir.exists():
        raise FileNotFoundError(f"image dir not found: {image_dir}")
    if not weights.exists():
        raise FileNotFoundError(f"weights not found: {weights}")

    detector = get_detector(
        weights=weights,
        device="0",
        conf=0.25,
        iou=0.7,
        imgsz=640,
        max_det=100,
    )

    images = list(_iter_images(image_dir))[:10]
    if not images:
        raise RuntimeError(f"no images found in: {image_dir}")

    print(f"[run] total_images={len(images)} dir={image_dir}")
    for idx, img_path in enumerate(images, start=1):
        results = detector.predict(img_path)
        print(f"\n[{idx:02d}/{len(images):02d}] {img_path.name} det_count={len(results)}")
        for j, det in enumerate(results, start=1):
            box = det["xyxy"]
            print(
                f"  - #{j:02d} cls={det['class_name']}({det['class_id']}) "
                f"conf={det['confidence']:.4f} "
                f"xyxy=[{box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f}, {box[3]:.1f}]"
            )


if __name__ == "__main__":
    main()
