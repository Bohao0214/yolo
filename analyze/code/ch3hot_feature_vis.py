#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

# 允许反序列化自定义模块（third_party.*）
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
try:
    import third_party  # noqa: F401
except Exception:
    pass

try:
    from ultralytics import YOLO
except Exception as exc:  # pragma: no cover
    raise ImportError(f"无法导入 ultralytics: {exc}")


USER_EDIT_CONFIG: Dict[str, Any] = {
    # 至少包含 baseline + a4/b7 其中之一
    "models": [
        # {"name": "baseline", "weight": "/abs/path/to/best.pt"},
        # {"name": "a4", "weight": "/abs/path/to/a4_best.pt"},
        # {"name": "b7", "weight": "/abs/path/to/b7_best.pt"},
    ],
    "images": [
        # "/abs/path/to/image1.jpg",
    ],
    "reference_model": "baseline",  # 用于对比图的参考模型
    "imgsz": 640,
    "device": "0",
    "conf": 0.001,
    "iou": 0.7,
    "max_det": 100,
    "output_root": "/home/ubuntu/hpproject/yolo/analyze/result",
    "report_prefix": "report_",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Feature response visualization for baseline/A4/B7")
    p.add_argument("--config_json", type=str, default="", help="JSON to override USER_EDIT_CONFIG")
    p.add_argument("--dry_run", action="store_true", help="Only check config; no inference")
    return p.parse_args()


def merge_cfg(base: Dict[str, Any], override_json: str) -> Dict[str, Any]:
    cfg = json.loads(json.dumps(base))
    if override_json.strip():
        ov = json.loads(override_json)
        if not isinstance(ov, dict):
            raise ValueError("--config_json must be a JSON object")
        cfg.update(ov)
    return cfg


def make_report_dir(root: Path, prefix: str) -> Path:
    ts = dt.datetime.now().strftime("%y%m%d%H%M")
    out = root / f"{prefix}{ts}"
    if not out.exists():
        out.mkdir(parents=True, exist_ok=False)
        return out
    i = 1
    while True:
        cand = root / f"{prefix}{ts}_{i:02d}"
        if not cand.exists():
            cand.mkdir(parents=True, exist_ok=False)
            return cand
        i += 1


def _extract_tensor(x: Any) -> Optional[np.ndarray]:
    if x is None:
        return None
    if isinstance(x, (list, tuple)):
        for it in x:
            arr = _extract_tensor(it)
            if arr is not None:
                return arr
        return None
    # torch.Tensor
    if hasattr(x, "detach") and hasattr(x, "cpu"):
        t = x.detach().float().cpu().numpy()
        return t
    return None


def _to_response_map(arr: np.ndarray) -> Optional[np.ndarray]:
    if arr is None:
        return None
    if arr.ndim == 4:  # B,C,H,W
        arr = arr[0]
    if arr.ndim == 3:  # C,H,W
        m = np.mean(np.abs(arr), axis=0)
        return m.astype(np.float32)
    if arr.ndim == 2:
        return arr.astype(np.float32)
    return None


def _find_detect_and_p3_idx(seq: Any) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    detect_idx = None
    for i, m in enumerate(seq):
        if hasattr(m, "f") and hasattr(m, "cv2"):
            detect_idx = i
            break
    if detect_idx is None:
        return None, None, None
    detect = seq[detect_idx]
    f = getattr(detect, "f", [])
    if isinstance(f, (list, tuple)) and len(f) > 0:
        p3 = int(f[0])
        p4 = int(f[1]) if len(f) > 1 else None
        return detect_idx, p3, p4
    if isinstance(f, int):
        return detect_idx, int(f), None
    return detect_idx, None, None


def _find_module_idx(seq: Any, keywords: Sequence[str]) -> Optional[int]:
    keys = [k.lower() for k in keywords]
    for i, m in enumerate(seq):
        name = m.__class__.__name__.lower()
        t = str(getattr(m, "type", "")).lower()
        s = f"{name} {t}"
        if any(k in s for k in keys):
            return i
    return None


def _register_capture_hooks(yolo: Any) -> Tuple[List[Any], Dict[str, Any], Dict[str, np.ndarray]]:
    seq = yolo.model.model
    hooks: List[Any] = []
    info: Dict[str, Any] = {}
    captured: Dict[str, np.ndarray] = {}

    detect_idx, p3_idx, p4_idx = _find_detect_and_p3_idx(seq)
    info["detect_idx"] = detect_idx
    info["p3_star_idx"] = p3_idx
    info["p4_star_idx"] = p4_idx

    p3_backbone_idx: Optional[int] = None
    if p3_idx is not None and 0 <= p3_idx < len(seq):
        f_p3 = getattr(seq[p3_idx], "f", None)
        if isinstance(f_p3, (list, tuple)) and len(f_p3) >= 2:
            try:
                p3_backbone_idx = int(f_p3[1])
            except Exception:
                p3_backbone_idx = None
    info["p3_backbone_idx"] = p3_backbone_idx

    a4_idx = _find_module_idx(seq, ["a4", "241a4", "spddownsample", "dualdelta", "spd"])
    b7_idx = _find_module_idx(seq, ["b7", "241b7", "carafe"])
    info["a4_idx"] = a4_idx
    info["b7_idx"] = b7_idx

    def _mk_hook(tag: str):
        def _hook(_m, _inp, out):
            if tag in captured:
                return
            arr = _extract_tensor(out)
            if arr is None:
                return
            m = _to_response_map(arr)
            if m is None:
                return
            captured[tag] = m
        return _hook

    if p3_idx is not None and 0 <= p3_idx < len(seq):
        hooks.append(seq[p3_idx].register_forward_hook(_mk_hook("p3_star")))
    if p4_idx is not None and 0 <= p4_idx < len(seq):
        hooks.append(seq[p4_idx].register_forward_hook(_mk_hook("p4_star")))
    if p3_backbone_idx is not None and 0 <= p3_backbone_idx < len(seq):
        hooks.append(seq[p3_backbone_idx].register_forward_hook(_mk_hook("p3_backbone")))
    if a4_idx is not None and 0 <= a4_idx < len(seq):
        hooks.append(seq[a4_idx].register_forward_hook(_mk_hook("a4")))
    if b7_idx is not None and 0 <= b7_idx < len(seq):
        hooks.append(seq[b7_idx].register_forward_hook(_mk_hook("b7")))

    return hooks, info, captured


def _normalize_maps(maps: List[np.ndarray]) -> List[np.ndarray]:
    vals = []
    for m in maps:
        if m.size > 0:
            vals.append(np.percentile(m, [1, 99]))
    if not vals:
        return maps
    lo = float(min(v[0] for v in vals))
    hi = float(max(v[1] for v in vals))
    den = max(hi - lo, 1e-6)
    out = []
    for m in maps:
        n = np.clip((m - lo) / den, 0, 1)
        out.append(n.astype(np.float32))
    return out


def _map_to_color(m: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    mr = cv2.resize(m, (out_w, out_h), interpolation=cv2.INTER_CUBIC)
    u8 = np.clip(mr * 255.0, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(u8, cv2.COLORMAP_VIRIDIS)


def _put_title(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 30), (255, 255, 255), -1)
    cv2.putText(out, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20, 20, 20), 2, cv2.LINE_AA)
    return out


def _save_panel(image_bgr: np.ndarray, maps: Dict[str, np.ndarray], out_path: Path) -> None:
    h, w = image_bgr.shape[:2]
    keys = list(maps.keys())
    norm_list = _normalize_maps([maps[k] for k in keys])
    tiles = [_put_title(image_bgr, "image")]
    for k, m in zip(keys, norm_list):
        tiles.append(_put_title(_map_to_color(m, w, h), k))
    panel = cv2.hconcat(tiles)
    cv2.imwrite(str(out_path), panel)


def _save_compare_panel(
    image_bgr: np.ndarray,
    base_map: np.ndarray,
    target_map: np.ndarray,
    base_name: str,
    target_name: str,
    out_path: Path,
) -> None:
    h, w = image_bgr.shape[:2]
    maps = _normalize_maps([base_map, target_map])
    bm = cv2.resize(maps[0], (w, h), interpolation=cv2.INTER_CUBIC)
    tm = cv2.resize(maps[1], (w, h), interpolation=cv2.INTER_CUBIC)
    diff = np.abs(tm - bm).astype(np.float32)
    diff = _normalize_maps([diff])[0]
    tiles = [
        _put_title(image_bgr, "image"),
        _put_title(_map_to_color(bm, w, h), f"{base_name}"),
        _put_title(_map_to_color(tm, w, h), f"{target_name}"),
        _put_title(_map_to_color(diff, w, h), "abs_diff"),
    ]
    panel = cv2.hconcat(tiles)
    cv2.imwrite(str(out_path), panel)


def main() -> None:
    args = parse_args()
    cfg = merge_cfg(USER_EDIT_CONFIG, args.config_json)

    models_cfg = cfg.get("models", [])
    images_cfg = cfg.get("images", [])
    if not isinstance(models_cfg, list) or len(models_cfg) == 0:
        raise ValueError("USER_EDIT_CONFIG.models 不能为空")
    if not isinstance(images_cfg, list) or len(images_cfg) == 0:
        raise ValueError("USER_EDIT_CONFIG.images 不能为空")

    imgsz = int(cfg.get("imgsz", 640))
    device = str(cfg.get("device", "0"))
    conf = float(cfg.get("conf", 0.001))
    iou = float(cfg.get("iou", 0.7))
    max_det = int(cfg.get("max_det", 100))
    ref_name = str(cfg.get("reference_model", "baseline"))

    out_root = Path(str(cfg.get("output_root", "/home/ubuntu/hpproject/yolo/analyze/result")))
    out_root.mkdir(parents=True, exist_ok=True)
    report_dir = make_report_dir(out_root, str(cfg.get("report_prefix", "report_")))
    cap_dir = report_dir / "captures"
    fig_dir = report_dir / "figures"
    cap_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 加载模型
    models: Dict[str, Any] = {}
    model_info: Dict[str, Any] = {}
    for m in models_cfg:
        name = str(m.get("name", "")).strip()
        weight = str(m.get("weight", "")).strip()
        if not name or not weight:
            raise ValueError(f"模型项缺少 name/weight: {m}")
        wp = Path(weight)
        if not wp.exists():
            raise FileNotFoundError(f"权重不存在: {wp}")
        y = YOLO(str(wp))
        models[name] = y
        model_info[name] = {"weight": str(wp)}

    if args.dry_run:
        dry = {
            "report_dir": str(report_dir),
            "imgsz": imgsz,
            "device": device,
            "conf": conf,
            "iou": iou,
            "max_det": max_det,
            "reference_model": ref_name,
            "models": model_info,
            "images": images_cfg,
        }
        (report_dir / "config_dump.json").write_text(json.dumps(dry, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(dry, ensure_ascii=False, indent=2))
        return

    all_maps: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}
    for img_str in images_cfg:
        ip = Path(str(img_str))
        if not ip.exists():
            raise FileNotFoundError(f"图像不存在: {ip}")
        img_bgr = cv2.imread(str(ip))
        if img_bgr is None:
            raise RuntimeError(f"无法读取图像: {ip}")
        stem = ip.stem
        all_maps[str(ip)] = {}

        for model_name, y in models.items():
            hooks, info, captured = _register_capture_hooks(y)
            try:
                _ = y.predict(
                    source=str(ip),
                    imgsz=imgsz,
                    conf=conf,
                    iou=iou,
                    max_det=max_det,
                    device=device if device else None,
                    verbose=False,
                    save=False,
                    plots=False,
                )
            finally:
                for h in hooks:
                    h.remove()

            all_maps[str(ip)][model_name] = captured
            model_info[model_name]["hook_info"] = info

            # 保存原始响应图
            raw_model_dir = cap_dir / model_name
            raw_model_dir.mkdir(parents=True, exist_ok=True)
            for k, m in captured.items():
                np.save(str(raw_model_dir / f"{stem}_{k}.npy"), m.astype(np.float32))

            # 单模型面板
            if captured:
                panel_path = fig_dir / f"{stem}__{model_name}__panel.png"
                _save_panel(img_bgr, captured, panel_path)

        # 基于 reference model 做同层对比（优先：p3_backbone/p3_star/p4_star）
        if ref_name in all_maps[str(ip)]:
            ref_maps = all_maps[str(ip)][ref_name]
            for model_name, maps in all_maps[str(ip)].items():
                if model_name == ref_name:
                    continue
                for key in ("p3_backbone", "p3_star", "p4_star"):
                    if key in ref_maps and key in maps:
                        outp = fig_dir / f"{stem}__cmp_{ref_name}_vs_{model_name}_{key}.png"
                        _save_compare_panel(
                            img_bgr,
                            ref_maps[key],
                            maps[key],
                            f"{ref_name}:{key}",
                            f"{model_name}:{key}",
                            outp,
                        )

                # 额外导出模块输出对比（若存在）
                if "p3_backbone" in ref_maps and "a4" in maps:
                    outp = fig_dir / f"{stem}__cmp_{ref_name}_p3_backbone_vs_{model_name}_a4.png"
                    _save_compare_panel(
                        img_bgr,
                        ref_maps["p3_backbone"],
                        maps["a4"],
                        f"{ref_name}:p3_backbone",
                        f"{model_name}:a4",
                        outp,
                    )
                if "p4_star" in ref_maps and "b7" in maps:
                    outp = fig_dir / f"{stem}__cmp_{ref_name}_p4_star_vs_{model_name}_b7.png"
                    _save_compare_panel(
                        img_bgr,
                        ref_maps["p4_star"],
                        maps["b7"],
                        f"{ref_name}:p4_star",
                        f"{model_name}:b7",
                        outp,
                    )

    run_meta = {
        "time": dt.datetime.now().isoformat(timespec="seconds"),
        "report_dir": str(report_dir),
        "imgsz": imgsz,
        "device": device,
        "conf": conf,
        "iou": iou,
        "max_det": max_det,
        "reference_model": ref_name,
        "models": model_info,
        "images": [str(x) for x in images_cfg],
    }
    (report_dir / "config_dump.json").write_text(json.dumps(run_meta, ensure_ascii=False, indent=2), encoding="utf-8")

    md = []
    md.append("# ch3hot 特征响应可视化报告")
    md.append("")
    md.append("## 这是什么")
    md.append("- 该脚本用于比较同一输入图像在不同权重模型上的中间特征响应。")
    md.append("- 默认抓取：`p3_backbone`（骨干到颈部融合前的 P3）、`p3_star`、`p4_star`（检测头输入尺度层）以及模块输出 `a4`/`b7`。")
    md.append("")
    md.append("## 结果有什么用")
    md.append("- 对比 `p3_backbone/p3_star`：验证 SPD(A4) 替换下采样后是否提升 P3 细节响应。")
    md.append("- 对比 `p4_star/p3_star`：验证 B7(CARAFE) 上采样后是否改善融合层响应。")
    md.append("- 通过 `abs_diff` 对比图定位“增强模块真正改变了哪里”。")
    md.append("")
    md.append("## 输出文件")
    md.append(f"- `figures/`：可视化图（单模型面板 + baseline 对比面板）")
    md.append(f"- `captures/`：每张图每个捕获点的原始响应 `npy`")
    md.append(f"- `config_dump.json`：本次运行参数与模型信息")
    md.append("")
    md.append("## 关键参数")
    md.append(f"- imgsz={imgsz}, device={device}, conf={conf}, iou={iou}, max_det={max_det}")
    md.append(f"- reference_model={ref_name}")
    (report_dir / "README.md").write_text("\n".join(md), encoding="utf-8")

    print(f"[done] report_dir={report_dir}")
    print(f"[done] figures={fig_dir}")
    print(f"[done] captures={cap_dir}")


if __name__ == "__main__":
    main()
