import random
from pathlib import Path
from typing import List, Tuple


def list_images(root: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    return sorted([p for p in root.iterdir() if p.suffix.lower() in exts])


def create_val_split(
    train_dir: Path, split_ratio: float, seed: int, save_dir: Path
) -> Tuple[Path, Path]:
    images = list_images(train_dir)
    if not images:
        raise RuntimeError(f"No train images found in {train_dir}")
    rng = random.Random(seed)
    rng.shuffle(images)
    val_count = max(1, int(len(images) * split_ratio))
    val_images = images[:val_count]
    train_images = images[val_count:]
    save_dir.mkdir(parents=True, exist_ok=True)
    train_txt = save_dir / "train.txt"
    val_txt = save_dir / "val.txt"
    with open(train_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(str(p) for p in train_images))
    with open(val_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(str(p) for p in val_images))
    return train_txt, val_txt
