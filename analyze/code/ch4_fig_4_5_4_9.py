#!/usr/bin/env python3
"""生成论文图4-9与图4-5（基于已有 FP/FN 导出）。"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

IMG_EXTS = [".png", ".jpg", ".jpeg", ".bmp", ".webp"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="生成图4-9（误报与重复预测）和图4-5（特征映射示意）")
    p.add_argument("--dataset_root", type=str, required=True, help="如 /home/ubuntu/hpproject/yolo/dataset/yolo/datasetm6c")
    p.add_argument("--fp_report_dir", type=str, required=True, help="包含 p2_3_2a_fp_samples.csv")
    p.add_argument("--fn_report_dir", type=str, required=True, help="包含 fn_cases.csv")
    p.add_argument("--out_root", type=str, default="/home/ubuntu/hpproject/yolo/analyze/result")
    p.add_argument("--report_name", type=str, default="")
    p.add_argument("--fig49_each", type=int, default=3, help="图4-9 每类示例数（unmatched/pred_dup）")
    return p.parse_args()


def make_report_dir(out_root: Path, report_name: str) -> Path:
    if report_name:
        out = out_root / report_name
        out.mkdir(parents=True, exist_ok=False)
        return out
    ts = dt.datetime.now().strftime("report_%y%m%d%H%M")
    out = out_root / f"{ts}_ch4_figs"
    if not out.exists():
        out.mkdir(parents=True, exist_ok=False)
        return out
    idx = 1
    while True:
        cand = out_root / f"{ts}_ch4_figs_{idx:02d}"
        if not cand.exists():
            cand.mkdir(parents=True, exist_ok=False)
            return cand
        idx += 1


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        lines = [ln for ln in f if ln.strip() and not ln.lstrip().startswith("#")]
    if not lines:
        return []
    return list(csv.DictReader(lines))


def parse_box(s: str) -> Tuple[float, float, float, float]:
    vals = [float(x.strip()) for x in s.split(",")]
    if len(vals) != 4:
        raise ValueError(f"invalid box: {s}")
    return vals[0], vals[1], vals[2], vals[3]


def find_image(dataset_root: Path, source_name: str, image_id: str) -> Optional[Path]:
    base = dataset_root / "images" / source_name
    for ext in IMG_EXTS:
        p = base / f"{image_id}{ext}"
        if p.exists():
            return p
    # fallback: glob by stem
    if base.exists():
        cand = list(base.glob(f"{image_id}.*"))
        if cand:
            return cand[0]
    return None


def fit_size(w: int, h: int, max_w: int, max_h: int) -> Tuple[int, int]:
    if w <= 0 or h <= 0:
        return 1, 1
    s = min(max_w / float(w), max_h / float(h))
    if s <= 0:
        return 1, 1
    return max(1, int(round(w * s))), max(1, int(round(h * s)))


def draw_box_thumb(img_path: Path, box: Tuple[float, float, float, float], color: Tuple[int, int, int], label: str, max_w: int, max_h: int) -> Image.Image:
    im = Image.open(img_path).convert("RGB")
    nw, nh = fit_size(im.width, im.height, max_w, max_h)
    im = im.resize((nw, nh), Image.BILINEAR)
    sx = nw / float(max(1, Image.open(img_path).width))
    sy = nh / float(max(1, Image.open(img_path).height))
    x1, y1, x2, y2 = box
    b = (x1 * sx, y1 * sy, x2 * sx, y2 * sy)
    dr = ImageDraw.Draw(im)
    dr.rectangle(b, outline=color, width=3)
    dr.rectangle((4, 4, min(nw - 4, 240), 28), fill=(0, 0, 0))
    dr.text((8, 8), label, fill=(255, 255, 255), font=ImageFont.load_default())
    return im


def build_fig49(dataset_root: Path, fp_rows: List[Dict[str, str]], out_path: Path, each_n: int) -> None:
    grouped = {"unmatched": [], "pred_dup": []}
    seen = {"unmatched": set(), "pred_dup": set()}
    for r in fp_rows:
        tag = r.get("fp_tag", "")
        if tag not in grouped:
            continue
        key = (r.get("source_name", ""), r.get("image_id", ""))
        if key in seen[tag]:
            continue
        img = find_image(dataset_root, r.get("source_name", ""), r.get("image_id", ""))
        if img is None:
            continue
        seen[tag].add(key)
        grouped[tag].append((r, img))

    unmatched = grouped["unmatched"][:each_n]
    pred_dup = grouped["pred_dup"][:each_n]

    pad = 16
    cell_w, cell_h = 360, 260
    cols = max(each_n, 1)
    W = pad + cols * (cell_w + pad)
    H = pad + 2 * (cell_h + 56 + pad)
    canvas = Image.new("RGB", (W, H), (245, 246, 250))
    dr = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    dr.text((pad, 8), "图4-9 基线模型典型误报与重复预测示例（上：unmatched，下：pred_dup）", fill=(20, 20, 20), font=font)

    def paste_row(items, row_idx: int, color: Tuple[int, int, int], title: str):
        y0 = pad + 24 + row_idx * (cell_h + 56 + pad)
        dr.text((pad, y0), title, fill=color, font=font)
        for i, (r, img_path) in enumerate(items):
            x0 = pad + i * (cell_w + pad)
            thumb = draw_box_thumb(
                img_path,
                parse_box(r["pred_box"]),
                color,
                f"{r['source_name']}/{r['image_id']} score={float(r.get('pred_score','0')):.3f}",
                cell_w,
                cell_h,
            )
            canvas.paste(thumb, (x0, y0 + 18))

    paste_row(unmatched, 0, (220, 60, 60), "unmatched（真实误判）")
    paste_row(pred_dup, 1, (50, 90, 220), "pred_dup（重复预测）")
    canvas.save(out_path)


def choose_small_fn(fn_rows: List[Dict[str, str]]) -> Dict[str, str]:
    cand = [r for r in fn_rows if r.get("diag_type") == "no_response"]
    if not cand:
        cand = fn_rows
    cand.sort(key=lambda r: float(r.get("gt_short_side_px", "1e9")))
    return cand[0]


def to_gray_arr(im: Image.Image) -> np.ndarray:
    return np.array(im.convert("L"), dtype=np.float32)


def upsample_patch(gray_crop: np.ndarray, stride: int, out_size: int = 180) -> Image.Image:
    h, w = gray_crop.shape[:2]
    sw = max(1, int(round(w / float(stride))))
    sh = max(1, int(round(h / float(stride))))
    down = Image.fromarray(gray_crop.astype(np.uint8)).resize((sw, sh), Image.BILINEAR)
    up = down.resize((out_size, out_size), Image.NEAREST)
    rgb = Image.merge("RGB", (up, up, up))
    return rgb


def build_fig45(fn_rows: List[Dict[str, str]], out_path: Path) -> None:
    row = choose_small_fn(fn_rows)
    img_path = Path(row["image_path"])
    gt = parse_box(row["gt_xyxy"])
    im = Image.open(img_path).convert("RGB")

    # left: original + GT
    left_w = 520
    left_h = 360
    nw, nh = fit_size(im.width, im.height, left_w, left_h)
    left = im.resize((nw, nh), Image.BILINEAR)
    sx, sy = nw / float(im.width), nh / float(im.height)
    x1, y1, x2, y2 = gt
    draw_left = ImageDraw.Draw(left)
    draw_left.rectangle((x1 * sx, y1 * sy, x2 * sx, y2 * sy), outline=(255, 80, 80), width=3)

    crop = im.crop((int(max(0, x1)), int(max(0, y1)), int(min(im.width, x2)), int(min(im.height, y2))))
    gray_crop = to_gray_arr(crop)

    patches = []
    notes = []
    short_side = float(row.get("gt_short_side_px", "0"))
    for stride in (8, 16, 32):
        patches.append(upsample_patch(gray_crop, stride))
        proj = short_side / float(stride)
        if proj < 0.5:
            tag = "高层几乎无响应"
        elif proj < 1.0:
            tag = "响应很弱"
        else:
            tag = "可形成响应"
        notes.append((stride, proj, tag))

    W, H = 1220, 720
    canvas = Image.new("RGB", (W, H), (245, 246, 250))
    dr = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    dr.text((20, 16), "图4-5 特征映射示意图（小缺陷在高层特征图响应减弱）", fill=(20, 20, 20), font=font)

    canvas.paste(left, (20, 60))
    dr.text((20, 430), f"样例: {img_path.name}  GT短边={short_side:.1f}px  机制={row.get('diag_type','')}", fill=(30, 30, 30), font=font)

    x0 = 580
    y0 = 80
    for i, (patch, (stride, proj, tag)) in enumerate(zip(patches, notes)):
        yy = y0 + i * 210
        canvas.paste(patch, (x0, yy))
        dr.rectangle((x0, yy, x0 + patch.width, yy + patch.height), outline=(60, 60, 60), width=2)
        dr.text((x0 + 200, yy + 8), f"stride={stride}", fill=(30, 30, 30), font=font)
        dr.text((x0 + 200, yy + 30), f"投影短边≈{proj:.2f} cells", fill=(30, 30, 30), font=font)
        dr.text((x0 + 200, yy + 52), tag, fill=(180, 40, 40) if "无响应" in tag or "弱" in tag else (30, 120, 30), font=font)

    dr.text((20, 470), "说明：stride 越大，目标在特征图上的投影单元越少；当投影<0.5 cell 时，高层分支难以稳定响应。", fill=(60, 60, 60), font=font)
    canvas.save(out_path)


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    fp_rows = read_csv(Path(args.fp_report_dir) / "p2_3_2a_fp_samples.csv")
    fn_rows = read_csv(Path(args.fn_report_dir) / "fn_cases.csv")

    if not fp_rows:
        raise RuntimeError("未找到 FP 样本：p2_3_2a_fp_samples.csv 为空或缺失")
    if not fn_rows:
        raise RuntimeError("未找到 FN 明细：fn_cases.csv 为空或缺失")

    out_dir = make_report_dir(Path(args.out_root), args.report_name)
    fig49 = out_dir / "fig_4_9_fp_typical_and_dup.png"
    fig45 = out_dir / "fig_4_5_feature_mapping_schematic.png"

    build_fig49(dataset_root, fp_rows, fig49, max(1, args.fig49_each))
    build_fig45(fn_rows, fig45)

    readme = out_dir / "README_figs.md"
    readme.write_text(
        "\n".join(
            [
                "# 图4-5 / 图4-9 生成说明",
                f"- fig_4_9: {fig49.name}",
                f"- fig_4_5: {fig45.name}",
                "- 图4-9来源：p2_3_2a_fp_samples.csv（unmatched 与 pred_dup 示例）",
                "- 图4-5来源：fn_cases.csv（选择小尺度且无响应 FN 样例，做 stride=8/16/32 投影示意）",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[DONE] fig_report_dir: {out_dir}")
    print(f"[DONE] {fig49}")
    print(f"[DONE] {fig45}")


if __name__ == "__main__":
    main()
