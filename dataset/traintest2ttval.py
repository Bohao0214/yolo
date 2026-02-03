import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def list_images(root: Path) -> List[Path]:
    return sorted([p for p in root.iterdir() if p.suffix.lower() in IMG_EXTS])


def label_path_for(image_path: Path, raw_root: Path) -> Path:
    rel = image_path.relative_to(raw_root / "images")
    return raw_root / "labels" / rel.with_suffix(".txt")


def is_defect(label_path: Path) -> bool:
    if not label_path.exists():
        return False
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                return True
    return False


def ensure_empty_label(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text("", encoding="utf-8")


def copy_pair(image_path: Path, label_path: Path, dst_root: Path, split: str, name: str) -> None:
    img_dst = dst_root / "images" / split / name
    lbl_dst = dst_root / "labels" / split / Path(name).with_suffix(".txt").name
    img_dst.parent.mkdir(parents=True, exist_ok=True)
    lbl_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(image_path, img_dst)
    if label_path.exists():
        shutil.copy2(label_path, lbl_dst)
    else:
        ensure_empty_label(lbl_dst)


def write_summary(
    out_root: Path,
    train_names: List[str],
    val_names: List[str],
    test_names: List[str],
) -> None:
    lines = [
        "Label rule:",
        "- defect: label file exists and has at least one line",
        "- normal: label file missing or empty",
        "",
        f"train_count: {len(train_names)}",
        f"val_count: {len(val_names)}",
        f"test_count: {len(test_names)}",
        "",
        "[train]",
        *train_names,
        "",
        "[val]",
        *val_names,
        "",
        "[test]",
        *test_names,
    ]
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "README.txt").write_text("\n".join(lines), encoding="utf-8")


def classify_train_test(raw_root: Path, out_root: Path) -> None:
    for split in ["train", "test"]:
        img_dir = raw_root / "images" / split
        for img_path in list_images(img_dir):
            lbl_path = label_path_for(img_path, raw_root)
            cls = "defect" if is_defect(lbl_path) else "normal"
            copy_pair(
                img_path,
                lbl_path,
                out_root / cls,
                split,
                img_path.name,
            )


def split_train_val(
    train_images: List[Path], val_ratio: float, seed: int
) -> Tuple[List[Path], List[Path]]:
    rng = random.Random(seed)
    images = train_images[:]
    rng.shuffle(images)
    val_count = max(1, int(len(images) * val_ratio)) if val_ratio > 0 else 0
    return images[val_count:], images[:val_count]


def split_train_val_test(
    images: List[Path], train_ratio: float, test_ratio: float, seed: int
) -> Tuple[List[Path], List[Path], List[Path]]:
    if train_ratio + test_ratio >= 1.0:
        raise ValueError("train_ratio + test_ratio must be < 1.0")
    val_ratio = 1.0 - train_ratio - test_ratio
    rng = random.Random(seed)
    pool = images[:]
    rng.shuffle(pool)
    train_count = int(len(pool) * train_ratio)
    test_count = int(len(pool) * test_ratio)
    train_split = pool[:train_count]
    test_split = pool[train_count : train_count + test_count]
    val_split = pool[train_count + test_count :]
    return train_split, val_split, test_split


def balance_train_images(
    images: List[Path],
    raw_root: Path,
    seed: int,
    strategy: str,
) -> List[Tuple[Path, str]]:
    rng = random.Random(seed)
    defect = []
    normal = []
    for img_path in images:
        lbl_path = label_path_for(img_path, raw_root)
        (defect if is_defect(lbl_path) else normal).append(img_path)

    if strategy == "undersample":
        target = min(len(defect), len(normal))
        defect = rng.sample(defect, target) if len(defect) > target else defect
        normal = rng.sample(normal, target) if len(normal) > target else normal
        return [(p, p.name) for p in defect + normal]

    if strategy == "oversample":
        if len(defect) == 0 or len(normal) == 0:
            return [(p, p.name) for p in images]
        if len(defect) < len(normal):
            short, long = defect, normal
        else:
            short, long = normal, defect
        needed = len(long) - len(short)
        dup = [rng.choice(short) for _ in range(needed)]
        result = [(p, p.name) for p in defect + normal]
        for idx, p in enumerate(dup):
            stem = p.stem
            result.append((p, f"{stem}_dup{idx}{p.suffix}"))
        return result

    return [(p, p.name) for p in images]


def build_yolo_dataset(
    raw_root: Path,
    out_root: Path,
    train_ratio: float,
    test_ratio: float,
    seed: int,
    balance_mode: str,
) -> None:
    train_images = list_images(raw_root / "images" / "train")
    test_images = list_images(raw_root / "images" / "test")
    all_images = train_images + test_images
    train_split, val_split, test_split = split_train_val_test(
        all_images, train_ratio, test_ratio, seed
    )

    strategy = {"a": "undersample", "b": "oversample", "n": "none"}.get(
        balance_mode, balance_mode
    )
    balanced_train = balance_train_images(train_split, raw_root, seed, strategy)
    train_names = []
    for img_path, name in balanced_train:
        lbl_path = label_path_for(img_path, raw_root)
        copy_pair(img_path, lbl_path, out_root, "train", name)
        train_names.append(name)

    val_names = []
    for img_path in val_split:
        lbl_path = label_path_for(img_path, raw_root)
        copy_pair(img_path, lbl_path, out_root, "val", img_path.name)
        val_names.append(img_path.name)

    test_names = []
    for img_path in test_split:
        lbl_path = label_path_for(img_path, raw_root)
        copy_pair(img_path, lbl_path, out_root, "test", img_path.name)
        test_names.append(img_path.name)

    write_summary(out_root, train_names, val_names, test_names)


def build_small_dataset(
    raw_root: Path,
    out_root: Path,
    train_ratio: float,
    test_ratio: float,
    seed: int,
    balance_mode: str,
    small_counts: Dict[str, int],
) -> None:
    train_images = list_images(raw_root / "images" / "train")
    test_images = list_images(raw_root / "images" / "test")
    all_images = train_images + test_images
    train_split, val_split, test_split = split_train_val_test(
        all_images, train_ratio, test_ratio, seed
    )
    rng = random.Random(seed)

    strategy = {"a": "undersample", "b": "oversample", "n": "none"}.get(
        balance_mode, balance_mode
    )
    balanced_train = balance_train_images(train_split, raw_root, seed, strategy)
    if small_counts.get("train", 0) > 0:
        balanced_train = rng.sample(balanced_train, min(len(balanced_train), small_counts["train"]))
    train_names = []
    for img_path, name in balanced_train:
        lbl_path = label_path_for(img_path, raw_root)
        copy_pair(img_path, lbl_path, out_root, "train", name)
        train_names.append(name)

    val_names = []
    if small_counts.get("val", 0) > 0 and val_split:
        val_pick = rng.sample(val_split, min(len(val_split), small_counts["val"]))
        for img_path in val_pick:
            lbl_path = label_path_for(img_path, raw_root)
            copy_pair(img_path, lbl_path, out_root, "val", img_path.name)
            val_names.append(img_path.name)

    test_names = []
    if small_counts.get("test", 0) > 0 and test_split:
        test_pick = rng.sample(test_split, min(len(test_split), small_counts["test"]))
        for img_path in test_pick:
            lbl_path = label_path_for(img_path, raw_root)
            copy_pair(img_path, lbl_path, out_root, "test", img_path.name)
            test_names.append(img_path.name)

    write_summary(out_root, train_names, val_names, test_names)


def main() -> None:
    raw_root = Path("/home/ubuntu/project/deduibi/yolo/dataset/raw/datasetm6c")
    classified_root = raw_root / "classified"
    yolo_root = Path("/home/ubuntu/project/deduibi/yolo/dataset/yolo/datasetm6c")
    yolo_small_root = Path("/home/ubuntu/project/deduibi/yolo/dataset/yolo/datasetm6c_small")

    seed = 42
    split_ratios = {"train": 0.7, "test": 0.15}  # val = 1 - train - test
    balance_mode = "n"  # a: undersample, b: oversample, n: none
    small_counts = {"train": 120, "val": 20, "test": 40}

    run_classify = True
    run_yolo = True
    run_small = True

    if run_classify:
        classify_train_test(raw_root, classified_root)

    if run_yolo:
        build_yolo_dataset(
            raw_root,
            yolo_root,
            split_ratios["train"],
            split_ratios["test"],
            seed,
            balance_mode,
        )

    if run_small:
        build_small_dataset(
            raw_root,
            yolo_small_root,
            split_ratios["train"],
            split_ratios["test"],
            seed,
            balance_mode,
            small_counts,
        )


if __name__ == "__main__":
    main()
