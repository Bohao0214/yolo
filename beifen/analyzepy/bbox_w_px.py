import os, glob
import cv2
import numpy as np

IMG_ROOT = "/home/ubuntu/project/deduibi/yolo/dataset/yolo/datasetm6c/images/val"  # 按你实际改
LBL_ROOT = "/home/ubuntu/project/deduibi/yolo/dataset/yolo/datasetm6c/labels/val"  # 按你实际改
IMGSZ = 640

# 支持常见图片后缀
IMG_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]

def find_image_for_label(label_path):
    """根据 label 文件名在 IMG_ROOT 里找同名图片（不同后缀都试）"""
    stem = os.path.splitext(os.path.basename(label_path))[0]
    # 若你的目录层级是 train/val/test，建议保持 images/xxx 与 labels/xxx 同结构
    # 这里做两种策略：1) 直接在 IMG_ROOT 递归找 2) 同目录映射
    candidates = []
    for ext in IMG_EXTS:
        candidates += glob.glob(os.path.join(IMG_ROOT, "**", stem + ext), recursive=True)
    return candidates[0] if candidates else None

ws, hs = [], []
miss_img = 0
bad_lines = 0

label_files = glob.glob(os.path.join(LBL_ROOT, "**", "*.txt"), recursive=True)

for lp in label_files:
    img_path = find_image_for_label(lp)
    if img_path is None:
        miss_img += 1
        continue

    im = cv2.imread(img_path)
    if im is None:
        miss_img += 1
        continue

    H0, W0 = im.shape[:2]
    r = min(IMGSZ / W0, IMGSZ / H0)  # letterbox scale

    with open(lp, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                bad_lines += 1
                continue

            # YOLO det: cls xc yc w h（归一化于原图）
            # YOLO seg: cls xc yc w h x1 y1 ...（>=6列），取前 5 列
            _, xc, yc, w, h = parts[:5]

            try:
                w0 = float(w) * W0
                h0 = float(h) * H0
            except ValueError:
                bad_lines += 1
                continue

            # 映射到 640 输入空间（宽高只受缩放 r 影响）
            ws.append(w0 * r)
            hs.append(h0 * r)

ws = np.array(ws, dtype=np.float32)
hs = np.array(hs, dtype=np.float32)

print("labels:", len(label_files))
print("missing/failed images:", miss_img)
print("bad lines:", bad_lines)
print("valid boxes:", len(ws))

if len(ws) == 0:
    raise SystemExit("No valid boxes. 请检查 images/labels 路径与同名匹配逻辑。")

s = np.minimum(ws, hs)

def pct(x): return float(np.mean(s < x))*100.0

print("\nmin(w,h) in 640-letterbox space (px):")
print("min=%.2f  p1=%.2f  p5=%.2f  p10=%.2f  p50=%.2f  p90=%.2f" %
      (s.min(), np.percentile(s,1), np.percentile(s,5), np.percentile(s,10),
       np.percentile(s,50), np.percentile(s,90)))
print("<8px:  %.2f%%" % pct(8))
print("<12px: %.2f%%" % pct(12))
print("<16px: %.2f%%" % pct(16))
