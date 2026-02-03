import argparse
import datetime as dt
import re
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO


def sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def letterbox(img: np.ndarray, size: int) -> Tuple[np.ndarray, float, int, int]:
    h, w = img.shape[:2]
    r = min(size / w, size / h)
    new_w, new_h = int(round(w * r)), int(round(h * r))
    resized = cv2.resize(img, (new_w, new_h))
    canvas = np.full((size, size, 3), 114, dtype=img.dtype)
    pad_x = (size - new_w) // 2
    pad_y = (size - new_h) // 2
    canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized
    return canvas, r, pad_x, pad_y


def img_to_tensor(img: np.ndarray, device: torch.device) -> torch.Tensor:
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(img_rgb).float() / 255.0
    t = t.permute(2, 0, 1).unsqueeze(0).to(device)
    return t


def save_heatmap(feat: torch.Tensor, out_path: Path) -> None:
    if feat.ndim == 4:
        feat = feat[0]
    # mean over channels
    heat = feat.mean(dim=0).cpu().numpy()
    heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-6)
    heat = (heat * 255).astype(np.uint8)
    heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), heat)


def build_hooks(model, keywords: List[str]) -> Dict[str, torch.Tensor]:
    feats: Dict[str, torch.Tensor] = {}

    def make_hook(name: str):
        def fn(_, __, y):
            if isinstance(y, (list, tuple)):
                for i, yi in enumerate(y):
                    if torch.is_tensor(yi):
                        feats[f"{name}.{i}"] = yi.detach().cpu()
            elif torch.is_tensor(y):
                feats[name] = y.detach().cpu()

        return fn

    for n, m in model.named_modules():
        if any(k in n for k in keywords):
            m.register_forward_hook(make_hook(n))
    return feats


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--data_root", type=str, default="", help="YOLO dataset root containing images/<split> dirs")
    parser.add_argument("--split", nargs="+", default=["val"], help="splits under images/, e.g. val test")
    parser.add_argument("--image", type=str, default="")
    parser.add_argument("--image_dir", type=str, default="")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--out_root", type=str, default="/home/ubuntu/project/deduibi/yolo/analysis")
    parser.add_argument(
        "--keywords",
        nargs="+",
        default=["neck", "fpn", "pan", "cv2", "cv3", "c2f", "p3", "p4", "p5"],
        help="Module name keywords to hook.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    out_root = Path(args.out_root)
    timestamp = dt.datetime.now().strftime("%y%m%d%H%M")
    report_dir = out_root / f"report_{timestamp}" / "features"
    report_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.weights).model
    model.eval()
    device = next(model.parameters()).device

    feats = build_hooks(model, args.keywords)

    if args.data_root:
        data_root = Path(args.data_root)
        img_paths = []
        for split in args.split:
            split_dir = data_root / "images" / split
            if not split_dir.exists():
                raise RuntimeError(f"Split dir not found: {split_dir}")
            img_paths.extend(
                sorted(
                    p
                    for p in split_dir.iterdir()
                    if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
                )
            )
    elif args.image_dir:
        img_paths = sorted(
            p for p in Path(args.image_dir).iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        )
    elif args.image:
        img_paths = [Path(args.image)]
    else:
        raise RuntimeError("Provide --data_root (and --split) or --image_dir or --image")

    total = 0
    for img_path in img_paths:
        feats.clear()
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img_lb, _, _, _ = letterbox(img, args.imgsz)
        x = img_to_tensor(img_lb, device)
        with torch.no_grad():
            _ = model(x)
        # save heatmaps per image
        subdir = report_dir / img_path.stem
        for idx, (name, feat) in enumerate(feats.items()):
            out_name = f"{sanitize(name)}_{idx}.png"
            save_heatmap(feat, subdir / out_name)
        total += 1

    print(f"Saved feature maps for {total} images to: {report_dir}")


if __name__ == "__main__":
    main()
"""""
/home/ubuntu/anaconda3/envs/yolo11/bin/python \
  /home/ubuntu/project/deduibi/yolo/tools/feature_probe.py \
  --weights /home/ubuntu/project/deduibi/yolo/models/defect/exp_2601261106/best/best.pt \
  --data_root /home/ubuntu/project/deduibi/yolo/dataset/yolo/datasetm6c \
  --split val test \
  --out_root /home/ubuntu/project/deduibi/yolo/analysis
"""""
