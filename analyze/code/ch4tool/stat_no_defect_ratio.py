#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统计工业缺陷数据集的“无缺陷图像比例”（按 split 和 overall）。

定义：
- 图像对应的 label txt 不存在，或存在但为空（仅空白），记为“无缺陷图像”
- 否则记为“有缺陷图像”

默认统计 4 个数据集：
- /home/ubuntu/hpproject/yolo/dataset/yolo/gc10det_622_halves
- /home/ubuntu/hpproject/yolo/dataset/yolo/DeepPCB_standard
- /home/ubuntu/hpproject/yolo/dataset/yolo/kolektorsdd_622_halves
- /home/ubuntu/hpproject/yolo/dataset/yolo/neudet_622

用法：
python /home/ubuntu/hpproject/yolo/analyze/code/ch4tool/stat_no_defect_ratio.py

可选：
python .../stat_no_defect_ratio.py \
  --out /home/ubuntu/hpproject/yolo/experiments/_runlogs/no_defect_ratio.csv
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple


DEFAULT_DATASETS: Dict[str, Path] = {
    "gc10det_622_halves": Path("/home/ubuntu/hpproject/yolo/dataset/yolo/gc10det_622_halves"),
    "DeepPCB_standard": Path("/home/ubuntu/hpproject/yolo/dataset/yolo/DeepPCB_standard"),
    "kolektorsdd_622_halves": Path("/home/ubuntu/hpproject/yolo/dataset/yolo/kolektorsdd_622_halves"),
    "neudet_622": Path("/home/ubuntu/hpproject/yolo/dataset/yolo/neudet_622"),
}

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("stat_no_defect_ratio")
    p.add_argument(
        "--out",
        default="",
        help="输出 csv 路径；为空则写到 /home/ubuntu/hpproject/yolo/experiments/_runlogs/no_defect_ratio_时间戳.csv",
    )
    return p.parse_args()


def list_images(split_dir: Path) -> List[Path]:
    if not split_dir.exists():
        return []
    imgs: List[Path] = []
    for ext in IMG_EXTS:
        imgs.extend(split_dir.rglob(f"*{ext}"))
    return sorted(imgs)


def label_for_image(img_path: Path, dataset_root: Path, split: str) -> Path:
    labels_dir = dataset_root / "labels" / split
    return labels_dir / f"{img_path.stem}.txt"


def is_no_defect(label_path: Path) -> bool:
    if not label_path.exists():
        return True
    txt = label_path.read_text(encoding="utf-8", errors="ignore")
    return len(txt.strip()) == 0


def count_split(dataset_root: Path, split: str) -> Tuple[int, int, int]:
    imgs = list_images(dataset_root / "images" / split)
    total = len(imgs)
    no_def = 0
    for img in imgs:
        if is_no_defect(label_for_image(img, dataset_root, split)):
            no_def += 1
    defect = total - no_def
    return total, no_def, defect


def safe_ratio(n: int, d: int) -> float:
    return (float(n) / float(d)) if d > 0 else 0.0


def main() -> None:
    args = parse_args()
    if args.out.strip():
        out_csv = Path(args.out).resolve()
    else:
        out_csv = Path("/home/ubuntu/hpproject/yolo/experiments/_runlogs") / f"no_defect_ratio_{datetime.now().strftime('%y%m%d_%H%M%S')}.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, object]] = []
    for name, root in DEFAULT_DATASETS.items():
        if not root.exists():
            rows.append(
                {
                    "dataset": name,
                    "split": "all",
                    "num_images": 0,
                    "num_no_defect": 0,
                    "num_defect": 0,
                    "no_defect_ratio": 0.0,
                    "exists": 0,
                    "root": str(root),
                }
            )
            continue

        agg_total = agg_no_def = agg_def = 0
        for split in SPLITS:
            total, no_def, defect = count_split(root, split)
            agg_total += total
            agg_no_def += no_def
            agg_def += defect
            rows.append(
                {
                    "dataset": name,
                    "split": split,
                    "num_images": total,
                    "num_no_defect": no_def,
                    "num_defect": defect,
                    "no_defect_ratio": safe_ratio(no_def, total),
                    "exists": 1,
                    "root": str(root),
                }
            )
        rows.append(
            {
                "dataset": name,
                "split": "all",
                "num_images": agg_total,
                "num_no_defect": agg_no_def,
                "num_defect": agg_def,
                "no_defect_ratio": safe_ratio(agg_no_def, agg_total),
                "exists": 1,
                "root": str(root),
            }
        )

    fields = ["dataset", "split", "num_images", "num_no_defect", "num_defect", "no_defect_ratio", "exists", "root"]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"[done] csv={out_csv}")


if __name__ == "__main__":
    main()

