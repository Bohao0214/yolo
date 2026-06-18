from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from ultralytics import YOLO

IMG_SUFFIX = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def mkdir(p: str | Path):
    Path(p).mkdir(parents=True, exist_ok=True)


# -----------------------------
# data utils
# -----------------------------
def list_images(source: str) -> List[Path]:
    p = Path(source)
    if p.is_file() and p.suffix.lower() == ".txt":
        out: List[Path] = []
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                q = Path(s)
                if q.exists() and q.suffix.lower() in IMG_SUFFIX:
                    out.append(q)
        return out
    if p.is_file() and p.suffix.lower() in IMG_SUFFIX:
        return [p]
    if p.is_dir():
        return sorted([x for x in p.rglob("*") if x.suffix.lower() in IMG_SUFFIX])
    raise FileNotFoundError(f"Invalid source: {source}")


def infer_label_path(image_path: Path) -> Path:
    parts = list(image_path.parts)
    for key in ["images", "image"]:
        if key in parts:
            idx = parts.index(key)
            parts[idx] = "labels"
            return Path(*parts).with_suffix(".txt")
    return image_path.with_suffix(".txt")


def read_yolo_boxes(image_path: Path, img_hw: Tuple[int, int]) -> List[Tuple[int, int, int, int]]:
    h, w = img_hw
    label_path = infer_label_path(image_path)
    if not label_path.exists():
        return []
    boxes = []
    with label_path.open("r", encoding="utf-8") as f:
        for line in f:
            ss = line.strip().split()
            if len(ss) < 5:
                continue
            _, xc, yc, bw, bh = map(float, ss[:5])
            x1 = int((xc - bw / 2) * w)
            y1 = int((yc - bh / 2) * h)
            x2 = int((xc + bw / 2) * w)
            y2 = int((yc + bh / 2) * h)
            x1 = max(0, min(w - 1, x1))
            y1 = max(0, min(h - 1, y1))
            x2 = max(0, min(w - 1, x2))
            y2 = max(0, min(h - 1, y2))
            boxes.append((x1, y1, x2, y2))
    return boxes


def draw_gt_boxes(img_rgb: np.ndarray, boxes: List[Tuple[int, int, int, int]]) -> np.ndarray:
    out = img_rgb.copy()
    for (x1, y1, x2, y2) in boxes:
        cv2.rectangle(out, (x1, y1), (x2, y2), (255, 0, 0), 2)
    return out


def load_image_rgb(image_path: str, imgsz: int = 640) -> Tuple[np.ndarray, torch.Tensor]:
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
    x = torch.from_numpy(img_rgb).float() / 255.0
    x = x.permute(2, 0, 1).unsqueeze(0).contiguous()
    return img_rgb, x


# -----------------------------
# feature aggregation / vis
# -----------------------------
def aggregate_feature(feat: torch.Tensor, mode: str = "mean") -> np.ndarray:
    if feat.dim() == 4:
        feat = feat[0]
    feat = feat.detach().float().cpu()
    if mode == "max":
        fmap = feat.max(dim=0).values
    elif mode == "absmean":
        fmap = feat.abs().mean(dim=0)
    else:
        fmap = feat.mean(dim=0)
    return fmap.numpy()


def normalize_map(fmap: np.ndarray, vmin: Optional[float] = None, vmax: Optional[float] = None) -> np.ndarray:
    if vmin is None:
        vmin = float(fmap.min())
    if vmax is None:
        vmax = float(fmap.max())
    if vmax > vmin:
        out = (fmap - vmin) / (vmax - vmin)
    else:
        out = np.zeros_like(fmap, dtype=np.float32)
    return np.clip(out, 0.0, 1.0)


def resize_map(fmap: np.ndarray, out_hw: Tuple[int, int]) -> np.ndarray:
    h, w = out_hw
    return cv2.resize(fmap, (w, h), interpolation=cv2.INTER_CUBIC)


def apply_colormap(fmap: np.ndarray, cmap: str = "viridis") -> np.ndarray:
    cm = plt.get_cmap(cmap)
    color = cm(fmap)[..., :3]
    return (color * 255).astype(np.uint8)


def overlay_heatmap(image_rgb: np.ndarray, fmap: np.ndarray, alpha: float = 0.45, cmap: str = "viridis") -> np.ndarray:
    color = apply_colormap(fmap, cmap=cmap)
    out = image_rgb.astype(np.float32) * (1 - alpha) + color.astype(np.float32) * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def fmap_to_vis(image_rgb: np.ndarray, fmap: np.ndarray, overlay: bool, cmap: str = "viridis") -> np.ndarray:
    fmap = normalize_map(fmap)
    fmap = resize_map(fmap, image_rgb.shape[:2])
    return overlay_heatmap(image_rgb, fmap, cmap=cmap) if overlay else apply_colormap(fmap, cmap=cmap)


# -----------------------------
# hooks
# -----------------------------
class FeatureHook:
    def __init__(self):
        self.outputs: Dict[str, torch.Tensor] = {}
        self.handles = []

    def add(self, module: torch.nn.Module, name: str):
        def fn(_, __, output):
            if isinstance(output, torch.Tensor):
                self.outputs[name] = output.detach()
            elif isinstance(output, (tuple, list)):
                for item in output:
                    if isinstance(item, torch.Tensor):
                        self.outputs[name] = item.detach()
                        break
        self.handles.append(module.register_forward_hook(fn))

    def clear(self):
        self.outputs.clear()

    def close(self):
        for h in self.handles:
            h.remove()
        self.handles = []


# -----------------------------
# discovery
# -----------------------------
def find_named_modules_by_class(model: torch.nn.Module, class_name: str) -> List[Tuple[str, torch.nn.Module]]:
    out = []
    for name, m in model.named_modules():
        if m.__class__.__name__ == class_name:
            out.append((name, m))
    return out


def get_spd_modules(det_model: torch.nn.Module) -> List[Tuple[str, torch.nn.Module]]:
    return find_named_modules_by_class(det_model, "SPDConvDownsample")


def get_carafe_modules(det_model: torch.nn.Module) -> List[Tuple[str, torch.nn.Module, str]]:
    found = find_named_modules_by_class(det_model, "CARAFEUpsampleSafe")
    found = sorted(found, key=lambda x: x[0])
    out = []
    for i, (name, m) in enumerate(found):
        if i == 0:
            stage = "p5_to_p4"
        elif i == 1:
            stage = "p4_to_p3"
        else:
            stage = f"carafe_{i}"
        out.append((name, m, stage))
    return out


# -----------------------------
# figure saving
# -----------------------------
def save_panels(panels: List[Tuple[str, np.ndarray]], save_path: str, suptitle: Optional[str] = None):
    n = len(panels)
    plt.figure(figsize=(4.1 * n, 4.8))
    if suptitle:
        plt.suptitle(suptitle, fontsize=12)
    for i, (title, img) in enumerate(panels, start=1):
        ax = plt.subplot(1, n, i)
        ax.imshow(img)
        ax.set_title(title, fontsize=11)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close()


# -----------------------------
# visualization
# -----------------------------
def visualize_spd_one(
    det_model: torch.nn.Module,
    image_path: Path,
    save_dir: Path,
    imgsz: int,
    agg: str,
    overlay: bool,
    draw_gt: bool,
    spd_view: str,
):
    found = get_spd_modules(det_model)
    if not found:
        print("[WARN] No SPDConvDownsample found.")
        return

    name, mod = found[0]
    hook = FeatureHook()
    hook.add(mod.enhance241_a4_base, "base")
    hook.add(mod.enhance241_a4_post, "spd_raw")
    hook.add(mod, "fused")

    img_rgb, x = load_image_rgb(str(image_path), imgsz=imgsz)
    device = next(det_model.parameters()).device
    x = x.to(device)

    hook.clear()
    with torch.no_grad():
        _ = det_model(x)

    if not {"base", "spd_raw", "fused"}.issubset(set(hook.outputs.keys())):
        print(f"[WARN] Missing SPD outputs for {image_path.name}: {hook.outputs.keys()}")
        hook.close()
        return

    # rebuild weighted branch explicitly for faithful visualization
    y_base = hook.outputs["base"]
    y_spd_raw = hook.outputs["spd_raw"]
    alpha_raw = mod.enhance241_a4_alpha.detach().float().cpu().item()
    alpha = float(torch.tanh(torch.tensor(alpha_raw)) * float(mod.alpha_cap))
    y_spd_weighted = y_spd_raw * alpha
    y_fused = hook.outputs["fused"]

    gt_img = img_rgb
    if draw_gt:
        boxes = read_yolo_boxes(image_path, img_rgb.shape[:2])
        gt_img = draw_gt_boxes(img_rgb, boxes)

    # base and fused use shared normalization for fair comparison
    base_map = aggregate_feature(y_base, mode=agg)
    fused_map = aggregate_feature(y_fused, mode=agg)
    joint_min = min(float(base_map.min()), float(fused_map.min()))
    joint_max = max(float(base_map.max()), float(fused_map.max()))
    base_vis = resize_map(normalize_map(base_map, joint_min, joint_max), img_rgb.shape[:2])
    fused_vis = resize_map(normalize_map(fused_map, joint_min, joint_max), img_rgb.shape[:2])
    base_vis = overlay_heatmap(img_rgb, base_vis) if overlay else apply_colormap(base_vis)
    fused_vis = overlay_heatmap(img_rgb, fused_vis) if overlay else apply_colormap(fused_vis)

    if spd_view == "raw":
        third_map = aggregate_feature(y_spd_raw, mode=agg)
        third_title = "SPD原始分支特征"
    else:
        # weighted residual is more meaningful for explaining fusion
        third_map = aggregate_feature(y_spd_weighted, mode="absmean" if agg == "mean" else agg)
        third_title = f"SPD加权残差特征 (alpha={alpha:.4f})"
    third_vis = fmap_to_vis(img_rgb, third_map, overlay=overlay)

    panels = [
        ("原图+GT" if draw_gt else "原图", gt_img),
        ("原始下采样特征", base_vis),
        (third_title, third_vis),
        ("融合输出特征", fused_vis),
    ]

    out_name = f"{image_path.stem}_spd.png"
    save_panels(panels, str(save_dir / out_name), suptitle=f"SPD ({name}) | alpha={alpha:.4f} | {image_path.name}")
    hook.close()


def visualize_carafe_one(
    det_model: torch.nn.Module,
    image_path: Path,
    save_dir: Path,
    imgsz: int,
    agg: str,
    overlay: bool,
    draw_gt: bool,
    stage_filter: str,
    carafe_view: str,
):
    carafes = get_carafe_modules(det_model)
    if not carafes:
        print("[WARN] No CARAFEUpsampleSafe found.")
        return

    img_rgb, x = load_image_rgb(str(image_path), imgsz=imgsz)
    device = next(det_model.parameters()).device
    x = x.to(device)

    gt_img = img_rgb
    if draw_gt:
        boxes = read_yolo_boxes(image_path, img_rgb.shape[:2])
        gt_img = draw_gt_boxes(img_rgb, boxes)

    for name, mod, stage in carafes:
        if stage_filter != "all" and stage != stage_filter:
            continue

        hook = FeatureHook()
        hook.add(mod.enhance241_b7_base, "base")
        hook.add(mod.enhance241_b7_carafe, "carafe_raw")
        hook.add(mod, "fused")

        hook.clear()
        with torch.no_grad():
            _ = det_model(x)

        if not {"base", "carafe_raw", "fused"}.issubset(set(hook.outputs.keys())):
            print(f"[WARN] Missing CARAFE outputs for {image_path.name}, {stage}: {hook.outputs.keys()}")
            hook.close()
            continue

        y_base = hook.outputs["base"]
        y_carafe_raw = hook.outputs["carafe_raw"]
        alpha_raw = mod.enhance241_b7_alpha.detach().float().cpu().item()
        alpha = float(torch.tanh(torch.tensor(alpha_raw)) * float(mod.alpha_cap))
        y_fused = hook.outputs["fused"]
        y_delta = (y_carafe_raw - y_base) * alpha

        base_map = aggregate_feature(y_base, mode=agg)
        fused_map = aggregate_feature(y_fused, mode=agg)
        joint_min = min(float(base_map.min()), float(fused_map.min()))
        joint_max = max(float(base_map.max()), float(fused_map.max()))
        base_vis = resize_map(normalize_map(base_map, joint_min, joint_max), img_rgb.shape[:2])
        fused_vis = resize_map(normalize_map(fused_map, joint_min, joint_max), img_rgb.shape[:2])
        base_vis = overlay_heatmap(img_rgb, base_vis) if overlay else apply_colormap(base_vis)
        fused_vis = overlay_heatmap(img_rgb, fused_vis) if overlay else apply_colormap(fused_vis)

        if carafe_view == "raw":
            third_map = aggregate_feature(y_carafe_raw, mode=agg)
            third_title = f"{stage} CARAFE原始分支"
        else:
            third_map = aggregate_feature(y_delta, mode="absmean" if agg == "mean" else agg)
            third_title = f"{stage} CARAFE加权残差 (alpha={alpha:.4f})"
        third_vis = fmap_to_vis(img_rgb, third_map, overlay=overlay)

        panels = [
            ("原图+GT" if draw_gt else "原图", gt_img),
            (f"{stage} 原始上采样特征", base_vis),
            (third_title, third_vis),
            (f"{stage} 融合输出特征", fused_vis),
        ]

        out_name = f"{image_path.stem}_{stage}.png"
        save_panels(panels, str(save_dir / out_name), suptitle=f"CARAFE ({stage}) | alpha={alpha:.4f} | {image_path.name}")
        hook.close()


# -----------------------------
# dataset-level driver
# -----------------------------
def pick_gt_examples(images: List[Path], max_n: int) -> List[Path]:
    out = []
    for p in images:
        if infer_label_path(p).exists():
            out.append(p)
        if len(out) >= max_n:
            break
    return out


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--source", type=str, required=True, help="image / dir / txt")
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--target", type=str, choices=["spd", "carafe", "both"], default="both")
    parser.add_argument("--agg", type=str, choices=["mean", "max", "absmean"], default="mean")
    parser.add_argument("--overlay", action="store_true")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max_images", type=int, default=-1)
    parser.add_argument("--gt_examples", type=int, default=6)
    parser.add_argument("--carafe_stage", type=str, choices=["all", "p5_to_p4", "p4_to_p3"], default="all")
    parser.add_argument("--spd_view", type=str, choices=["weighted", "raw"], default="weighted",
                        help="third panel for SPD: weighted residual or raw SPD branch")
    parser.add_argument("--carafe_view", type=str, choices=["delta", "raw"], default="delta",
                        help="third panel for CARAFE: weighted delta or raw CARAFE branch")
    return parser.parse_args()


def main():
    args = parse_args()

    save_dir = Path(args.save_dir)
    mkdir(save_dir)
    mkdir(save_dir / "spd_all")
    mkdir(save_dir / "carafe_all")
    mkdir(save_dir / "spd_gt_examples")
    mkdir(save_dir / "carafe_gt_examples")

    images = list_images(args.source)
    if args.max_images > 0:
        images = images[: args.max_images]

    print(f"[INFO] total images = {len(images)}")

    yolo = YOLO(args.weights)
    det_model = yolo.model.to(args.device).eval()

    for idx, image_path in enumerate(images, start=1):
        print(f"[{idx}/{len(images)}] {image_path}")
        try:
            if args.target in ["spd", "both"]:
                visualize_spd_one(
                    det_model=det_model,
                    image_path=image_path,
                    save_dir=save_dir / "spd_all",
                    imgsz=args.imgsz,
                    agg=args.agg,
                    overlay=args.overlay,
                    draw_gt=False,
                    spd_view=args.spd_view,
                )
            if args.target in ["carafe", "both"]:
                visualize_carafe_one(
                    det_model=det_model,
                    image_path=image_path,
                    save_dir=save_dir / "carafe_all",
                    imgsz=args.imgsz,
                    agg=args.agg,
                    overlay=args.overlay,
                    draw_gt=False,
                    stage_filter=args.carafe_stage,
                    carafe_view=args.carafe_view,
                )
        except Exception as e:
            print(f"[WARN] failed on {image_path}: {e}")

    gt_imgs = pick_gt_examples(images, args.gt_examples)
    print(f"[INFO] gt examples = {len(gt_imgs)}")

    for image_path in gt_imgs:
        try:
            if args.target in ["spd", "both"]:
                visualize_spd_one(
                    det_model=det_model,
                    image_path=image_path,
                    save_dir=save_dir / "spd_gt_examples",
                    imgsz=args.imgsz,
                    agg=args.agg,
                    overlay=args.overlay,
                    draw_gt=True,
                    spd_view=args.spd_view,
                )
            if args.target in ["carafe", "both"]:
                visualize_carafe_one(
                    det_model=det_model,
                    image_path=image_path,
                    save_dir=save_dir / "carafe_gt_examples",
                    imgsz=args.imgsz,
                    agg=args.agg,
                    overlay=args.overlay,
                    draw_gt=True,
                    stage_filter=args.carafe_stage,
                    carafe_view=args.carafe_view,
                )
        except Exception as e:
            print(f"[WARN] gt example failed on {image_path}: {e}")

    print("[DONE]")


if __name__ == "__main__":
    main()
"""""
1）画某个模型里的 SPD 特征图
python /home/ubuntu/hpproject/yolo/analyze/code/visualize_a4_b7_features.py \
  --weights best.pt \
  --source dataset/yolo/datasetm6c/images/val \
  --save_dir analyze/result/report_2605192024 \
  --target spd \
  --imgsz 640 \
  --agg mean \
  --device cuda:0

  --spd_view raw
  
python /home/ubuntu/hpproject/yolo/analyze/code/visualize_a4_b7_features.py \
  --weights best.pt \
  --image dataset/yolo/datasetm6c/images/train/0003.png \
  --save_dir analyze/result/report_2605192024 \
  --target spd \
  --module_index 0 \
  --imgsz 640 \
  --agg mean \
  --device cuda:0


2）画某个模型里的 CARAFE 特征图
python /home/ubuntu/hpproject/yolo/analyze/code/visualize_a4_b7_features.py \
  --weights best.pt \
  --source dataset/yolo/datasetm6c/images/train \
  --save_dir analyze/result/report_2605192024 \
  --target carafe \
  --carafe_stage p5_to_p4 \
  --imgsz 640 \
  --agg mean \
  --device cuda:0


3）两个一起画
python /home/ubuntu/hpproject/yolo/analyze/code/visualize_a4_b7_features.py \
  --weights /home/ubuntu/hpproject/yolo/experiments/a4b7d6/datasetm6c/defect241__a4__b7__d6/exp_2603060619/train/weights/best.pt \
  --image /你的测试图片路径/xxx.jpg \
  --save_dir /home/ubuntu/hpproject/yolo/analyze/result/feature_vis \
  --target both \
  --module_index 0 \
  --imgsz 640 \
  --agg mean \
  --device cuda:0
4）如果想叠加到原图上

加一个参数：

--overlay

"""""