import argparse
import datetime as dt
import re
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO

# 你想保存的关键层名字，可按实际需求自定义
SAVE_LAYERS = ["p3", "p4", "p5"]  # 只保留这几层

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

def save_heatmap_and_overlay(
    feat: torch.Tensor, 
    out_path: Path, 
    orig_img: np.ndarray, 
    alpha: float = 0.45
) -> None:
    if feat.ndim == 4:
        feat = feat[0]
    heat = feat.mean(dim=0).cpu().numpy()
    heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-6)
    heat_uint8 = (heat * 255).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # 单独保存 heatmap
    cv2.imwrite(str(out_path), heat_color)
    # 保存叠加图
    heat_resized = cv2.resize(heat_color, (orig_img.shape[1], orig_img.shape[0]))
    overlay = cv2.addWeighted(orig_img, 1 - alpha, heat_resized, alpha, 0)
    overlay_path = out_path.parent / (out_path.stem + "_overlay.png")
    cv2.imwrite(str(overlay_path), overlay)

def draw_yolo_boxes(img: np.ndarray, label_path: Path, color=(0,255,0), thickness=2) -> np.ndarray:
    # 读 YOLO txt 标注并画框
    h, w = img.shape[:2]
    if not label_path.exists():
        return img.copy()
    img = img.copy()
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls, xc, yc, bw, bh = map(float, parts[:5])
            cx, cy, boxw, boxh = xc * w, yc * h, bw * w, bh * h
            x1 = int(cx - boxw / 2)
            y1 = int(cy - boxh / 2)
            x2 = int(cx + boxw / 2)
            y2 = int(cy + boxh / 2)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    return img

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

    for idx, (n, m) in enumerate(model.named_modules()):
        if any(k in n for k in keywords):
            m.register_forward_hook(make_hook(n))
    return feats

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--data_root", type=str, default="", help="YOLO dataset root containing images/<split> dirs")
    parser.add_argument("--split", nargs="+", default=["val"], help="splits under images/, e.g. val test")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--out_root", type=str, default="/home/ubuntu/project/deduibi/yolo/analysis")
    parser.add_argument(
        "--keywords",
        nargs="+",
        default=SAVE_LAYERS,  # 用 SAVE_LAYERS 控制
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
    # 搜集所有图片路径
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
        # 输出目录
        subdir = report_dir / img_path.stem
        subdir.mkdir(parents=True, exist_ok=True)
        # ——①保存原图和带框原图
        label_path = img_path.parent.parent.parent / "labels" / img_path.parent.name / (img_path.stem + ".txt")
        # 保存“原图” (原始无标注) 
        cv2.imwrite(str(subdir / "orig.png"), img)
        # 保存“带标注的原图”
        img_with_boxes = draw_yolo_boxes(img, label_path)
        cv2.imwrite(str(subdir / "orig_with_boxes.png"), img_with_boxes)
        # ——②只保存关键层的 heatmap/overlay，并编号
        for idx, (name, feat) in enumerate(feats.items()):
            for key_idx, layer in enumerate(SAVE_LAYERS):
                if layer in name:
                    out_name = f"{sanitize(layer)}_{key_idx+1}.png"
                    save_heatmap_and_overlay(feat, subdir / out_name, img)
        total += 1
    print(f"Saved features, overlays, origs for {total} images to: {report_dir}")

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
