#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import yaml


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML must be a mapping: {path}")
    return data


def _resolve_ref(ref: str, base: Path) -> Path:
    p = Path(ref).expanduser()
    if p.is_absolute():
        return p
    return (base / p).resolve()


def _label_dir_from_images(images_dir: Path) -> Path:
    parts = list(images_dir.parts)
    if "images" in parts:
        idx = parts.index("images")
        parts[idx] = "labels"
        return Path(*parts)
    return images_dir.parent / "labels" / images_dir.name


def _iter_label_files(data_yaml: Path, split: str) -> Iterable[Path]:
    data = _load_yaml(data_yaml)
    root_ref = str(data.get("path", "")).strip()
    root = _resolve_ref(root_ref, data_yaml.parent) if root_ref else data_yaml.parent.resolve()

    split_ref = data.get(split)
    if split_ref is None:
        return []
    if isinstance(split_ref, (list, tuple)):
        refs: List[str] = [str(x) for x in split_ref if str(x).strip()]
    else:
        refs = [str(split_ref)]

    out: List[Path] = []
    for ref in refs:
        p = _resolve_ref(ref, root)
        if p.is_dir():
            lbl = _label_dir_from_images(p)
            if lbl.exists():
                out.extend(sorted(lbl.rglob("*.txt")))
        elif p.is_file() and p.suffix.lower() == ".txt":
            with p.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    img = _resolve_ref(line, p.parent)
                    lbl = _label_dir_from_images(img.parent) / f"{img.stem}.txt"
                    if lbl.exists():
                        out.append(lbl)
    return out


def _count_objects(label_path: Path) -> int:
    try:
        with label_path.open("r", encoding="utf-8") as f:
            return sum(1 for ln in f if ln.strip())
    except Exception:
        return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute recommended c7_n for MCBAM from YOLO labels.")
    parser.add_argument("--data", required=True, help="Path to data yaml")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"], help="Dataset split")
    parser.add_argument("--clip-min", type=int, default=1, help="Minimum clipped n")
    parser.add_argument("--clip-max", type=int, default=6, help="Maximum clipped n")
    args = parser.parse_args()

    data_yaml = Path(args.data).expanduser().resolve()
    labels = list(_iter_label_files(data_yaml, args.split))
    if not labels:
        raise SystemExit(f"No label files resolved from split={args.split} data={data_yaml}")

    counts = [_count_objects(p) for p in labels]
    total_images = len(counts)
    total_objects = sum(counts)
    avg_objects_per_image = float(total_objects) / float(max(1, total_images))

    # MCBAM paper-inspired n statistic proxy:
    # n = (sum_i sum_j x_ij) / (sum_i m_i) -> average object complexity per image.
    n_raw = int(round(avg_objects_per_image))
    n_clip = max(int(args.clip_min), min(int(args.clip_max), n_raw))

    print(f"data_yaml={data_yaml}")
    print(f"split={args.split}")
    print(f"images={total_images}")
    print(f"objects={total_objects}")
    print(f"avg_objects_per_image={avg_objects_per_image:.4f}")
    print(f"recommended_c7_n_raw={n_raw}")
    print(f"recommended_c7_n_clipped={n_clip}")


if __name__ == "__main__":
    main()
