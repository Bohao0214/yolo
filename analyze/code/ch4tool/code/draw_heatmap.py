#!/usr/bin/env python3
"""
用法说明（热力图/响应图）：

优先级：EigenCAM -> 中间特征响应 -> 框分数密度图（自动回退）。
输出到 figures/heatmaps，并记录方法到 tables/heatmap_method.json。
默认报告根目录为：
/home/ubuntu/hpproject/yolo/analyze/code/ch4tool

示例：
conda run -n yolo11 python /home/ubuntu/hpproject/yolo/analyze/code/ch4tool/code/draw_heatmap.py \
  --report-root /home/ubuntu/hpproject/yolo/analyze/code/ch4tool \
  --num-cases 5

可选参数：
--imgsz
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

SCRIPT_PATH = Path(__file__).resolve()
REPORT_ROOT = SCRIPT_PATH.parents[1]
PROJECT_ROOT = SCRIPT_PATH.parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPT_PATH.parent) not in sys.path:
    sys.path.insert(0, str(SCRIPT_PATH.parent))

from utils_common import append_log, ensure_dir, stack_h, write_csv, write_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Draw heatmaps for baseline vs improved models.")
    p.add_argument("--report-root", type=Path, default=REPORT_ROOT)
    p.add_argument("--num-cases", type=int, default=5)
    p.add_argument("--imgsz", type=int, default=None)
    p.add_argument("--try-cam", action="store_true", default=True)
    return p.parse_args()


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> List[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _title(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 24), (0, 0, 0), -1)
    cv2.putText(out, text, (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def _norm_heat(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    x = x - float(np.min(x))
    mx = float(np.max(x))
    if mx > 1e-8:
        x = x / mx
    return np.clip(x, 0.0, 1.0)


def _overlay_heat(img_bgr: np.ndarray, heat01: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    heat = cv2.resize(heat01, (w, h), interpolation=cv2.INTER_LINEAR)
    heat_u8 = (np.clip(heat, 0.0, 1.0) * 255.0).astype(np.uint8)
    cmap = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    return cv2.addWeighted(img_bgr, 1.0 - alpha, cmap, alpha, 0)


def _find_target_layer(net) -> Optional[object]:
    modules = list(net.modules())
    for m in reversed(modules):
        name = m.__class__.__name__.lower()
        if "conv" in name or "c2f" in name or "bottleneck" in name or "spp" in name:
            return m
    return modules[-1] if modules else None


def _prep_tensor(image_bgr: np.ndarray, imgsz: int):
    import torch

    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
    ten = torch.from_numpy(resized).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    return ten


def _try_cam_overlay(model_path: str, image_bgr: np.ndarray, imgsz: int):
    import torch
    from ultralytics import YOLO

    try:
        from pytorch_grad_cam import EigenCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
    except Exception as e:
        raise RuntimeError(f"pytorch_grad_cam unavailable: {e}")

    y = YOLO(model_path)
    net = y.model.model
    net.eval()

    target_layer = _find_target_layer(net)
    if target_layer is None:
        raise RuntimeError("no target layer found for CAM")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)

    ten = _prep_tensor(image_bgr, imgsz).to(device)
    with torch.no_grad():
        _ = net(ten)

    cam = EigenCAM(model=net, target_layers=[target_layer], use_cuda=(device.type == "cuda"))
    cam_map = cam(input_tensor=ten)
    if cam_map is None or len(cam_map) == 0:
        raise RuntimeError("empty CAM map")

    cam01 = _norm_heat(cam_map[0])
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    vis_rgb = show_cam_on_image(rgb, cv2.resize(cam01, (rgb.shape[1], rgb.shape[0])), use_rgb=True)
    vis_bgr = cv2.cvtColor(vis_rgb, cv2.COLOR_RGB2BGR)
    return vis_bgr, f"EigenCAM@{target_layer.__class__.__name__}"


def _feature_response_overlay(model_path: str, image_bgr: np.ndarray, imgsz: int):
    import torch
    from ultralytics import YOLO

    y = YOLO(model_path)
    net = y.model.model
    net.eval()

    target_layer = _find_target_layer(net)
    if target_layer is None:
        raise RuntimeError("no target layer for feature response")

    feature_holder = {}

    def _hook(_m, _inp, out):
        feature_holder["feat"] = out

    handle = target_layer.register_forward_hook(_hook)
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net.to(device)
        ten = _prep_tensor(image_bgr, imgsz).to(device)
        with torch.no_grad():
            _ = net(ten)
        feat = feature_holder.get("feat", None)
        if feat is None:
            raise RuntimeError("hook did not capture feature")
        if isinstance(feat, (list, tuple)):
            feat = feat[0]
        if feat.ndim != 4:
            raise RuntimeError(f"unexpected feature ndim={feat.ndim}")
        fmap = feat[0].detach().float().cpu().numpy()  # C,H,W
        heat = np.mean(np.maximum(fmap, 0.0), axis=0)
        if float(np.max(heat)) < 1e-8:
            heat = np.mean(np.abs(fmap), axis=0)
        heat01 = _norm_heat(heat)
        overlay = _overlay_heat(image_bgr, heat01)
        return overlay, f"feature_mean@{target_layer.__class__.__name__}"
    finally:
        handle.remove()


def _score_density_overlay(model_path: str, image_path: str, image_bgr: np.ndarray, imgsz: int, conf: float, iou: float):
    from ultralytics import YOLO

    y = YOLO(model_path)
    res = y.predict(source=image_path, imgsz=imgsz, conf=conf, iou=iou, save=False, verbose=False)[0]
    h, w = image_bgr.shape[:2]
    heat = np.zeros((h, w), dtype=np.float32)
    if res.boxes is not None and res.boxes.xyxy is not None:
        boxes = res.boxes.xyxy.detach().cpu().numpy()
        confs = res.boxes.conf.detach().cpu().numpy() if res.boxes.conf is not None else np.ones((len(boxes),), dtype=np.float32)
        for b, s in zip(boxes, confs):
            x1, y1, x2, y2 = [int(round(v)) for v in b.tolist()]
            x1 = max(0, min(w - 1, x1))
            y1 = max(0, min(h - 1, y1))
            x2 = max(0, min(w, x2))
            y2 = max(0, min(h, y2))
            if x2 <= x1 or y2 <= y1:
                continue
            heat[y1:y2, x1:x2] += float(s)
    heat = cv2.GaussianBlur(heat, (0, 0), sigmaX=9.0, sigmaY=9.0)
    return _overlay_heat(image_bgr, _norm_heat(heat)), "score_box_density"


def _model_path_map(compare_rows: List[dict]) -> Dict[str, str]:
    out = {}
    for r in compare_rows:
        name = str(r.get("model_name", "")).strip()
        mp = str(r.get("model_path", "")).strip()
        if name and mp:
            out[name] = mp
    return out


def _select_cases(report_root: Path, num_cases: int, fallback_split: str, dataset_root: Optional[Path]) -> List[str]:
    q = _read_csv(report_root / "tables" / "qualitative_cases.csv")
    from_q = []
    for r in q:
        p = str(r.get("image_path", "")).strip()
        if p and p not in from_q:
            from_q.append(p)
    if from_q:
        return from_q[:num_cases]

    if dataset_root is None:
        return []
    try:
        from tools.eval_detection_benchmark import load_ground_truth

        image_paths, _, _ = load_ground_truth(dataset_root=dataset_root, split=fallback_split)
        return [str(p.resolve()) for p in image_paths[:num_cases]]
    except Exception:
        return []


def main() -> None:
    args = parse_args()
    report_root = args.report_root.expanduser().resolve()
    figs_dir = ensure_dir(report_root / "figures" / "heatmaps")
    tables_dir = ensure_dir(report_root / "tables")
    log_path = report_root / "logs" / "draw_heatmap.log"

    metadata = _read_json(tables_dir / "metadata.json")
    compare_rows = _read_csv(tables_dir / "compare_main.csv")
    ok_rows = [r for r in compare_rows if str(r.get("status", "")) == "ok"]

    baseline = metadata.get("baseline_model")
    best = metadata.get("best_model")
    if not baseline and ok_rows:
        baseline = ok_rows[0].get("model_name")
    if not best and ok_rows:
        best = sorted(ok_rows, key=lambda r: float(r.get("map50", 0.0) or 0.0), reverse=True)[0].get("model_name")

    if not baseline or not best:
        append_log(log_path, "no available baseline/best model for heatmap")
        write_csv(tables_dir / "heatmap_manifest.csv", [], ["case_id", "image_path", "output_png", "baseline_method", "improved_method"])
        write_json(
            tables_dir / "heatmap_method.json",
            {
                "status": "skipped",
                "reason": "no successful model rows",
                "method_priority": ["EigenCAM", "feature_response", "score_box_density"],
            },
        )
        print("[warn] no successful model rows, heatmap skipped.")
        return

    mp_map = _model_path_map(compare_rows)
    baseline_path = Path(mp_map.get(baseline, "")).expanduser().resolve() if mp_map.get(baseline) else None
    best_path = Path(mp_map.get(best, "")).expanduser().resolve() if mp_map.get(best) else None

    if baseline_path is None or best_path is None or (not baseline_path.exists()) or (not best_path.exists()):
        append_log(log_path, f"model paths missing: baseline={baseline_path}, best={best_path}")
        write_csv(tables_dir / "heatmap_manifest.csv", [], ["case_id", "image_path", "output_png", "baseline_method", "improved_method"])
        write_json(
            tables_dir / "heatmap_method.json",
            {
                "status": "skipped",
                "reason": "weights missing on this machine",
                "baseline": str(baseline_path) if baseline_path else "",
                "best": str(best_path) if best_path else "",
                "method_priority": ["EigenCAM", "feature_response", "score_box_density"],
            },
        )
        print("[warn] model weights missing, heatmap skipped.")
        return

    eval_params = metadata.get("eval_params", {})
    imgsz = int(args.imgsz or eval_params.get("imgsz", 640))
    conf = float(eval_params.get("conf", 0.001))
    iou = float(eval_params.get("iou", 0.7))
    split = metadata.get("split_used", metadata.get("split_requested", "test"))
    dataset_root = Path(metadata.get("dataset_root", "")).expanduser().resolve() if metadata.get("dataset_root") else None

    cases = _select_cases(report_root, int(args.num_cases), split, dataset_root)
    if not cases:
        append_log(log_path, "no cases selected for heatmap")
        write_csv(tables_dir / "heatmap_manifest.csv", [], ["case_id", "image_path", "output_png", "baseline_method", "improved_method"])
        write_json(
            tables_dir / "heatmap_method.json",
            {
                "status": "skipped",
                "reason": "no images available for heatmap",
                "method_priority": ["EigenCAM", "feature_response", "score_box_density"],
            },
        )
        print("[warn] no candidate images, heatmap skipped.")
        return

    manifest = []
    failures = []

    def render_for_model(model_path: Path, image_path: str) -> Tuple[np.ndarray, str]:
        img = cv2.imread(image_path)
        if img is None:
            raise RuntimeError(f"failed to read image: {image_path}")
        if args.try_cam:
            try:
                out, m = _try_cam_overlay(str(model_path), img, imgsz)
                return out, m
            except Exception as e:
                failures.append({"stage": "cam", "model_path": str(model_path), "image_path": image_path, "error": str(e)[:300]})
        try:
            out, m = _feature_response_overlay(str(model_path), img, imgsz)
            return out, m
        except Exception as e:
            failures.append(
                {"stage": "feature_response", "model_path": str(model_path), "image_path": image_path, "error": str(e)[:300]}
            )
        out, m = _score_density_overlay(str(model_path), image_path, img, imgsz=imgsz, conf=conf, iou=iou)
        return out, m

    for idx, img_path in enumerate(cases, start=1):
        orig = cv2.imread(img_path)
        if orig is None:
            continue
        try:
            ov_b, method_b = render_for_model(baseline_path, img_path)
            ov_i, method_i = render_for_model(best_path, img_path)

            panel = stack_h(
                [
                    _title(orig, "ORIG"),
                    _title(ov_b, f"{baseline} | {method_b}"),
                    _title(ov_i, f"{best} | {method_i}"),
                ]
            )
            out_png = figs_dir / f"{baseline}_vs_{best}_case{idx:03d}.png"
            cv2.imwrite(str(out_png), panel)
            manifest.append(
                {
                    "case_id": idx,
                    "image_path": img_path,
                    "output_png": str(out_png),
                    "baseline_method": method_b,
                    "improved_method": method_i,
                }
            )
        except Exception as e:
            failures.append({"stage": "render_case", "image_path": img_path, "error": str(e)[:300]})

    write_csv(tables_dir / "heatmap_manifest.csv", manifest, ["case_id", "image_path", "output_png", "baseline_method", "improved_method"])
    write_json(
        tables_dir / "heatmap_method.json",
        {
            "status": "ok" if manifest else "partial_or_failed",
            "baseline": baseline,
            "best": best,
            "method_priority": ["EigenCAM", "feature_response", "score_box_density"],
            "num_cases_requested": int(args.num_cases),
            "num_cases_done": len(manifest),
            "failures": failures,
        },
    )
    append_log(log_path, f"heatmap done: {len(manifest)} cases")
    print(f"[done] heatmaps -> {figs_dir}")


if __name__ == "__main__":
    main()
