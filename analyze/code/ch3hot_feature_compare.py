#!/usr/bin/env python3
from __future__ import annotations

"""
Feature heatmap comparison for baseline YOLO11m vs improved SD-YOLO11.

Outputs (per comparison):
- a4_p3_compare.png
- a4_internal_spd_module.png
- b7_p5_to_p4_internal.png
- b7_p4_to_p3_internal.png
- b7_p4star_compare.png
- b7_p3star_compare.png
- feature_meta.json
- README_feature_compare.md
"""

import argparse
import copy
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


# -----------------------------
# User-edit defaults (requested)
# -----------------------------
USER_EDIT_CONFIG: Dict[str, Any] = {
    "models": [
        {
            "name": "baseline",
            "weight": "experiments/yolo11-baseline/datasetm6c/exp_2603040206/train/weights/best.pt",
        },
        {
            "name": "oura4",
            "weight": "experiments/yolo11a4b7d6/datasetm6c/defect241__a4__b7__d6/exp_2603060404/train/weights/best.pt",
        },
        {
            "name": "ourb7",
            "weight": "experiments/yolo11a4b7d6/datasetm6c/defect241__a4__b7__d6/exp_2603060404/train/weights/best.pt",
        },
    ],
    "images": [
        "dataset/yolo/datasetm6c/images/train/0013.png",
    ],
    "reference_model": "baseline",
    "imgsz": 640,
    "device": "0",
    "conf": 0.001,
    "iou": 0.7,
    "max_det": 100,
    "output_root": "/home/ubuntu/hpproject/yolo/analyze/result",
    "report_prefix": "report_",
}


# Ensure repo root in sys.path so custom third_party modules are importable from checkpoints.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ultralytics import YOLO  # noqa: E402

# Reuse enhance241 locate logic and module classes (do not hardcode structure).
from third_party.yolo11.enhance241.yolo11_241a4 import (  # noqa: E402
    SPDConvDownsample,
    _locate_p4_to_p3_fuse as locate_p4_to_p3_fuse_a4,
)
from third_party.yolo11.enhance241.yolo11_241b7 import (  # noqa: E402
    CARAFEUpsampleSafe,
    _locate_p4_to_p3_fuse as locate_p4_to_p3_fuse_b7,
    _locate_p5_to_p4_fuse as locate_p5_to_p4_fuse_b7,
)


@dataclass
class LayerChoice:
    index: int
    name: str


@dataclass
class NormInfo:
    lo: float
    hi: float
    mode: str


def _now_tag() -> str:
    return datetime.now().strftime("%y%m%d%H%M")


def _resolve_path(path_like: str | Path) -> Path:
    p = Path(path_like)
    return p if p.is_absolute() else (REPO_ROOT / p).resolve()


def _safe_tensor(x: Any) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x
    if isinstance(x, (list, tuple)):
        for item in x:
            if isinstance(item, torch.Tensor):
                return item
    if isinstance(x, dict):
        for _, v in x.items():
            if isinstance(v, torch.Tensor):
                return v
    raise TypeError(f"Cannot convert object to tensor for feature visualization: type={type(x)}")


def _aggregate_feature_map(feat: torch.Tensor, method: str = "abs_mean") -> np.ndarray:
    t = feat.detach().float()
    if t.ndim == 4:
        t = t[0]
    if t.ndim != 3:
        raise ValueError(f"Expected feature tensor with ndim=3/4, got {tuple(t.shape)}")
    if method == "abs_mean":
        a = t.abs().mean(dim=0)
    elif method == "l2":
        a = torch.sqrt((t * t).mean(dim=0).clamp_min(1e-12))
    else:
        raise ValueError(f"Unsupported aggregation mode: {method}")
    return a.cpu().numpy()


def _resize_map(arr: np.ndarray, out_hw: Tuple[int, int]) -> np.ndarray:
    h, w = out_hw
    return cv2.resize(arr.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)


def _robust_joint_range(arrays: Sequence[np.ndarray], p_lo: float = 1.0, p_hi: float = 99.0) -> NormInfo:
    flat = np.concatenate([a.reshape(-1) for a in arrays], axis=0)
    lo = float(np.percentile(flat, p_lo))
    hi = float(np.percentile(flat, p_hi))
    if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
        lo = float(np.min(flat))
        hi = float(np.max(flat))
        if hi <= lo:
            hi = lo + 1e-6
    return NormInfo(lo=lo, hi=hi, mode=f"joint_p{p_lo:g}_p{p_hi:g}")


def _normalize(arr: np.ndarray, norm: NormInfo) -> np.ndarray:
    x = (arr - norm.lo) / (norm.hi - norm.lo + 1e-12)
    return np.clip(x, 0.0, 1.0)


def _normalize_diff(diff: np.ndarray, p: float = 99.0) -> Tuple[np.ndarray, NormInfo]:
    hi = float(np.percentile(np.abs(diff), p))
    hi = max(hi, 1e-6)
    norm = NormInfo(lo=-hi, hi=hi, mode=f"symmetric_p{p:g}")
    x = (diff - norm.lo) / (norm.hi - norm.lo + 1e-12)
    return np.clip(x, 0.0, 1.0), norm


def _plot_panels(
    out_path: Path,
    titles: Sequence[str],
    images: Sequence[np.ndarray],
    *,
    cmap: str,
    overlay: bool,
    base_rgb: np.ndarray,
    diff_indices: Optional[Sequence[int]] = None,
) -> None:
    diff_set = set(diff_indices or [])
    n = len(images)
    fig_w = 3.5 * n
    fig_h = 3.6
    fig, axs = plt.subplots(1, n, figsize=(fig_w, fig_h), dpi=180)
    if n == 1:
        axs = [axs]

    for i, (ax, t, img) in enumerate(zip(axs, titles, images)):
        if i == 0 and img.ndim == 3:
            ax.imshow(img)
        else:
            if img.ndim == 3:
                ax.imshow(img)
            else:
                if overlay:
                    ax.imshow(base_rgb)
                    ax.imshow(img, cmap="coolwarm" if i in diff_set else cmap, alpha=0.55)
                else:
                    ax.imshow(img, cmap="coolwarm" if i in diff_set else cmap)
        ax.set_title(t, fontsize=9)
        ax.axis("off")
    fig.tight_layout(pad=0.5)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), bbox_inches="tight")
    plt.close(fig)


def _pick_device(device_str: str) -> torch.device:
    d = str(device_str).strip()
    if d in {"", "cpu"}:
        return torch.device("cpu")
    if d.isdigit() and torch.cuda.is_available():
        return torch.device(f"cuda:{d}")
    if d.startswith("cuda") and torch.cuda.is_available():
        return torch.device(d)
    return torch.device("cpu")


def _load_letterbox(img: np.ndarray, imgsz: int) -> np.ndarray:
    try:
        from ultralytics.data.augment import LetterBox

        lb = LetterBox(new_shape=(imgsz, imgsz), auto=False, scaleFill=False, scaleup=True, stride=32)
        return lb(image=img)
    except Exception:
        h, w = img.shape[:2]
        scale = min(imgsz / max(h, 1), imgsz / max(w, 1))
        nh, nw = int(round(h * scale)), int(round(w * scale))
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((imgsz, imgsz, 3), 114, dtype=np.uint8)
        top = (imgsz - nh) // 2
        left = (imgsz - nw) // 2
        canvas[top : top + nh, left : left + nw] = resized
        return canvas


def _prep_input(image_path: Path, imgsz: int, device: torch.device, dtype: torch.dtype) -> Tuple[np.ndarray, torch.Tensor, Dict[str, Any]]:
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    orig_h, orig_w = bgr.shape[:2]
    lb_bgr = _load_letterbox(bgr, imgsz)
    rgb = cv2.cvtColor(lb_bgr, cv2.COLOR_BGR2RGB)
    x = torch.from_numpy(rgb).to(device=device)
    x = x.permute(2, 0, 1).contiguous().unsqueeze(0)
    x = (x.float() / 255.0).to(dtype=dtype)
    orig_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    meta = {
        "image_path": str(image_path),
        "orig_shape": [int(orig_h), int(orig_w)],
        "letterbox_shape": [int(lb_bgr.shape[0]), int(lb_bgr.shape[1])],
    }
    return orig_rgb, x, meta


def _extract_seq(yolo_model: YOLO) -> Any:
    m = yolo_model.model
    if hasattr(m, "model"):
        return m.model
    raise RuntimeError("YOLO model has no .model.model sequence")


def _layer_name(idx: int, mod: torch.nn.Module) -> str:
    return f"model.model[{idx}]::{mod.__class__.__name__}"


def _capture_layer_outputs(yolo_model: YOLO, x: torch.Tensor, indices: Sequence[int]) -> Dict[int, torch.Tensor]:
    seq = _extract_seq(yolo_model)
    outputs: Dict[int, torch.Tensor] = {}
    hooks = []

    for i in sorted(set(indices)):
        m = seq[i]

        def _mk_hook(idx: int):
            def _hook(_m: torch.nn.Module, _inp: Tuple[Any, ...], out: Any) -> None:
                outputs[idx] = _safe_tensor(out).detach()

            return _hook

        hooks.append(m.register_forward_hook(_mk_hook(i)))

    with torch.inference_mode():
        _ = yolo_model.model(x)

    for h in hooks:
        h.remove()

    miss = [i for i in indices if i not in outputs]
    if miss:
        raise RuntimeError(f"Failed to capture outputs for layer indices: {miss}")
    return outputs


def _capture_module_input(yolo_model: YOLO, module_idx: int, x: torch.Tensor) -> torch.Tensor:
    seq = _extract_seq(yolo_model)
    cache: Dict[str, torch.Tensor] = {}

    def _pre(_m: torch.nn.Module, inp: Tuple[Any, ...]) -> None:
        cache["x"] = _safe_tensor(inp[0]).detach()

    h = seq[module_idx].register_forward_pre_hook(_pre)
    with torch.inference_mode():
        _ = yolo_model.model(x)
    h.remove()
    if "x" not in cache:
        raise RuntimeError(f"Failed to capture input for layer {module_idx}")
    return cache["x"]


def _is_spd_module(mod: torch.nn.Module) -> bool:
    return isinstance(mod, SPDConvDownsample) or mod.__class__.__name__ == "SPDConvDownsample"


def _is_b7_module(mod: torch.nn.Module) -> bool:
    return isinstance(mod, CARAFEUpsampleSafe) or mod.__class__.__name__ == "CARAFEUpsampleSafe"


def _list_stride2_candidates(seq: Any) -> List[str]:
    rows: List[str] = []
    for i, m in enumerate(seq):
        hit = False
        for sm in m.modules():
            if isinstance(sm, torch.nn.Conv2d) and tuple(sm.stride) == (2, 2):
                hit = True
                break
        if hit:
            rows.append(f"idx={i}, name={m.__class__.__name__}, f={getattr(m, 'f', None)}")
    return rows


def _find_a4_spd(
    seq: Any,
    manual_idx: Optional[int],
    manual_name: Optional[str],
) -> Tuple[LayerChoice, List[LayerChoice]]:
    found = [LayerChoice(i, _layer_name(i, m)) for i, m in enumerate(seq) if _is_spd_module(m)]
    if manual_idx is not None:
        if manual_idx < 0 or manual_idx >= len(seq):
            raise RuntimeError(f"manual a4 index out of range: {manual_idx}")
        m = seq[manual_idx]
        if not _is_spd_module(m):
            raise RuntimeError(f"manual a4 index={manual_idx} is not SPDConvDownsample ({m.__class__.__name__})")
        return LayerChoice(manual_idx, _layer_name(manual_idx, m)), found
    if manual_name:
        for i, m in enumerate(seq):
            if manual_name in _layer_name(i, m):
                if not _is_spd_module(m):
                    raise RuntimeError(f"manual a4 name matched non-SPD module: {_layer_name(i, m)}")
                return LayerChoice(i, _layer_name(i, m)), found
        raise RuntimeError(f"manual a4 name not found: {manual_name}")

    if not found:
        cand = _list_stride2_candidates(seq)
        raise RuntimeError(
            "No SPDConvDownsample found in improved model. stride=2 candidates: " + "; ".join(cand)
        )
    return found[0], found


def _nearest_b7_before(seq: Any, end_idx: int) -> Optional[int]:
    for i in range(end_idx - 1, -1, -1):
        if _is_b7_module(seq[i]):
            return i
    return None


def _find_b7_modules(
    seq: Any,
    manual_p5p4: Optional[int],
    manual_p4p3: Optional[int],
) -> Tuple[LayerChoice, LayerChoice, Dict[str, Any]]:
    all_b7 = [LayerChoice(i, _layer_name(i, m)) for i, m in enumerate(seq) if _is_b7_module(m)]
    if len(all_b7) < 2 and (manual_p5p4 is None or manual_p4p3 is None):
        raise RuntimeError(f"Expected >=2 CARAFEUpsampleSafe modules, got {len(all_b7)}")

    p3_concat_idx, _ = locate_p4_to_p3_fuse_b7(seq)
    p4_concat_idx, _ = locate_p5_to_p4_fuse_b7(seq, p3_concat_idx)

    if manual_p5p4 is not None:
        p5p4_idx = manual_p5p4
    else:
        p5p4_idx = _nearest_b7_before(seq, p4_concat_idx)
    if manual_p4p3 is not None:
        p4p3_idx = manual_p4p3
    else:
        p4p3_idx = _nearest_b7_before(seq, p3_concat_idx)

    if p5p4_idx is None or p4p3_idx is None:
        raise RuntimeError(
            f"Cannot align b7 modules with concat locations. p4_concat_idx={p4_concat_idx}, p3_concat_idx={p3_concat_idx}, b7={all_b7}"
        )
    if p5p4_idx == p4p3_idx:
        raise RuntimeError(f"b7 module mapping conflict: both routes map to idx={p5p4_idx}")

    m1, m2 = seq[p5p4_idx], seq[p4p3_idx]
    if not _is_b7_module(m1) or not _is_b7_module(m2):
        raise RuntimeError("Mapped b7 indices are not CARAFEUpsampleSafe modules")

    info = {
        "p4_to_p3_concat_idx": int(p3_concat_idx),
        "p5_to_p4_concat_idx": int(p4_concat_idx),
        "all_b7_indices": [c.index for c in all_b7],
    }
    return LayerChoice(p5p4_idx, _layer_name(p5p4_idx, m1)), LayerChoice(p4p3_idx, _layer_name(p4p3_idx, m2)), info


def _find_p3_idx(seq: Any, manual_p3_idx: Optional[int] = None) -> int:
    if manual_p3_idx is not None:
        return int(manual_p3_idx)
    _, p3_idx = locate_p4_to_p3_fuse_a4(seq)
    return int(p3_idx)


def _find_pstar_indices(seq: Any, manual_p4star_idx: Optional[int], manual_p3star_idx: Optional[int]) -> Tuple[int, int, Dict[str, int]]:
    p3_concat_idx, _ = locate_p4_to_p3_fuse_b7(seq)
    p4_concat_idx, _ = locate_p5_to_p4_fuse_b7(seq, p3_concat_idx)

    p4_star = int(manual_p4star_idx) if manual_p4star_idx is not None else int(p4_concat_idx + 1)
    p3_star = int(manual_p3star_idx) if manual_p3star_idx is not None else int(p3_concat_idx + 1)
    if p4_star >= len(seq) or p3_star >= len(seq):
        raise RuntimeError(
            f"P* index out of range: p4*={p4_star}, p3*={p3_star}, len(seq)={len(seq)}"
        )
    return p4_star, p3_star, {"p4_concat_idx": int(p4_concat_idx), "p3_concat_idx": int(p3_concat_idx)}


def _alpha_from_param(alpha_param: torch.Tensor, alpha_cap: float, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    raw = alpha_param.to(dtype=dtype, device=device)
    return torch.tanh(raw) * float(alpha_cap)


def _reconstruct_a4_internal(mod: torch.nn.Module, x_in: torch.Tensor) -> Dict[str, torch.Tensor]:
    y_base = mod.enhance241_a4_base(x_in)
    y_pre = mod.enhance241_a4_pre(x_in)
    y_s2d = mod.enhance241_a4_s2d(y_pre)
    y_spd = mod.enhance241_a4_post(y_s2d)
    cap = float(getattr(mod, "alpha_cap", 0.5))
    alpha = _alpha_from_param(mod.enhance241_a4_alpha, cap, y_base.dtype, y_base.device)
    out = y_base + alpha * y_spd
    return {
        "x_in": x_in,
        "y_base": y_base,
        "y_spd": y_spd,
        "out": out,
        "alpha": alpha,
    }


def _reconstruct_b7_internal(mod: torch.nn.Module, x_in: torch.Tensor) -> Dict[str, torch.Tensor]:
    y_base = mod.enhance241_b7_base(x_in)
    y_carafe = mod.enhance241_b7_carafe(x_in)
    cap = float(getattr(mod, "alpha_cap", 0.5))
    alpha = _alpha_from_param(mod.enhance241_b7_alpha, cap, y_base.dtype, y_base.device)
    out = y_base + alpha * (y_carafe - y_base)
    return {
        "x_in": x_in,
        "y_base": y_base,
        "y_carafe": y_carafe,
        "out": out,
        "alpha": alpha,
    }


def _collect_maps(
    tensors: Dict[str, torch.Tensor],
    *,
    agg: str,
    out_hw: Tuple[int, int],
) -> Dict[str, np.ndarray]:
    maps: Dict[str, np.ndarray] = {}
    for k, v in tensors.items():
        if not isinstance(v, torch.Tensor):
            continue
        m = _aggregate_feature_map(v, method=agg)
        maps[k] = _resize_map(m, out_hw)
    return maps


def _run_compare_once(
    baseline_name: str,
    baseline_weight: Path,
    improved_name: str,
    improved_weight: Path,
    image_path: Path,
    out_dir: Path,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)

    device = _pick_device(args.device)

    baseline_yolo = YOLO(str(baseline_weight))
    improved_yolo = YOLO(str(improved_weight))

    baseline_yolo.model.to(device).eval()
    improved_yolo.model.to(device).eval()

    dtype = next(improved_yolo.model.parameters()).dtype
    orig_rgb, x, image_meta = _prep_input(image_path, int(args.imgsz), device, dtype)
    out_hw = (orig_rgb.shape[0], orig_rgb.shape[1])

    bseq = _extract_seq(baseline_yolo)
    iseq = _extract_seq(improved_yolo)

    # a4 locate
    a4_choice, a4_all = _find_a4_spd(iseq, args.manual_a4_index, args.manual_a4_name)
    if not a4_all:
        raise RuntimeError("improved model has no a4 module (SPDConvDownsample)")

    # b7 locate
    b7_p5p4, b7_p4p3, b7_info = _find_b7_modules(iseq, args.manual_b7_p5p4_index, args.manual_b7_p4p3_index)

    # P3, P4*, P3* indices
    ip3_idx = _find_p3_idx(iseq, args.manual_p3_index)
    ip4s_idx, ip3s_idx, pstar_info_i = _find_pstar_indices(iseq, args.manual_p4star_index, args.manual_p3star_index)

    # baseline alignment (same locate logic on baseline; fallback to improved index)
    align_notes: List[str] = []
    try:
        bp3_idx = _find_p3_idx(bseq, None)
    except Exception:
        bp3_idx = ip3_idx
        align_notes.append(f"baseline P3 fallback to improved index={ip3_idx}")

    try:
        bp4s_idx, bp3s_idx, pstar_info_b = _find_pstar_indices(bseq, None, None)
    except Exception:
        bp4s_idx, bp3s_idx = ip4s_idx, ip3s_idx
        pstar_info_b = {"p4_concat_idx": -1, "p3_concat_idx": -1}
        align_notes.append(f"baseline P4*/P3* fallback to improved indices: {ip4s_idx}, {ip3s_idx}")

    if max(bp3_idx, bp4s_idx, bp3s_idx) >= len(bseq):
        raise RuntimeError(
            f"baseline aligned indices out of range: p3={bp3_idx}, p4*={bp4s_idx}, p3*={bp3s_idx}, len={len(bseq)}"
        )

    # Capture endpoint features
    b_outs = _capture_layer_outputs(baseline_yolo, x, [bp3_idx, bp4s_idx, bp3s_idx])
    i_outs = _capture_layer_outputs(improved_yolo, x, [ip3_idx, ip4s_idx, ip3s_idx])

    # Reconstruct a4 internals from captured module input
    a4_in = _capture_module_input(improved_yolo, a4_choice.index, x)
    a4_mod = iseq[a4_choice.index]
    a4_tensors = _reconstruct_a4_internal(a4_mod, a4_in)

    # Reconstruct b7 internals (two routes)
    b7_m1_in = _capture_module_input(improved_yolo, b7_p5p4.index, x)
    b7_m2_in = _capture_module_input(improved_yolo, b7_p4p3.index, x)
    b7_m1 = iseq[b7_p5p4.index]
    b7_m2 = iseq[b7_p4p3.index]
    b7_r1 = _reconstruct_b7_internal(b7_m1, b7_m1_in)
    b7_r2 = _reconstruct_b7_internal(b7_m2, b7_m2_in)

    # Aggregate maps
    base_p3 = _resize_map(_aggregate_feature_map(b_outs[bp3_idx], args.agg), out_hw)
    imp_p3 = _resize_map(_aggregate_feature_map(i_outs[ip3_idx], args.agg), out_hw)

    base_p4s = _resize_map(_aggregate_feature_map(b_outs[bp4s_idx], args.agg), out_hw)
    imp_p4s = _resize_map(_aggregate_feature_map(i_outs[ip4s_idx], args.agg), out_hw)

    base_p3s = _resize_map(_aggregate_feature_map(b_outs[bp3s_idx], args.agg), out_hw)
    imp_p3s = _resize_map(_aggregate_feature_map(i_outs[ip3s_idx], args.agg), out_hw)

    a4_maps = _collect_maps({
        "y_base": a4_tensors["y_base"],
        "y_spd": a4_tensors["y_spd"],
        "out": a4_tensors["out"],
    }, agg=args.agg, out_hw=out_hw)

    b7_r1_maps = _collect_maps({
        "y_base": b7_r1["y_base"],
        "y_carafe": b7_r1["y_carafe"],
        "out": b7_r1["out"],
    }, agg=args.agg, out_hw=out_hw)

    b7_r2_maps = _collect_maps({
        "y_base": b7_r2["y_base"],
        "y_carafe": b7_r2["y_carafe"],
        "out": b7_r2["out"],
    }, agg=args.agg, out_hw=out_hw)

    # ---------- Figure A1: P3 compare ----------
    norm_a1 = _robust_joint_range([base_p3, imp_p3], p_lo=1, p_hi=99)
    a1_b = _normalize(base_p3, norm_a1)
    a1_i = _normalize(imp_p3, norm_a1)
    a1_diff_raw = imp_p3 - base_p3
    a1_d, norm_a1d = _normalize_diff(a1_diff_raw, p=99)

    _plot_panels(
        out_dir / "a4_p3_compare.png",
        [
            "Input Image",
            f"Baseline P3 ({args.agg})",
            f"Improved P3 ({args.agg})",
            "Diff (Improved - Baseline)",
        ],
        [orig_rgb, a1_b, a1_i, a1_d],
        cmap=args.cmap,
        overlay=args.overlay,
        base_rgb=orig_rgb,
        diff_indices=[3],
    )

    # ---------- Figure A2: a4 internals ----------
    norm_a2 = _robust_joint_range([a4_maps["y_base"], a4_maps["y_spd"], a4_maps["out"]], p_lo=1, p_hi=99)
    a2_base = _normalize(a4_maps["y_base"], norm_a2)
    a2_spd = _normalize(a4_maps["y_spd"], norm_a2)
    a2_out = _normalize(a4_maps["out"], norm_a2)
    a2_diff_raw = a4_maps["out"] - a4_maps["y_base"]
    a2_diff, norm_a2d = _normalize_diff(a2_diff_raw, p=99)

    _plot_panels(
        out_dir / "a4_internal_spd_module.png",
        [
            "Input Image",
            f"a4 y_base ({args.agg})",
            f"a4 y_spd ({args.agg})",
            f"a4 out ({args.agg})",
            "Diff (out - base)",
        ],
        [orig_rgb, a2_base, a2_spd, a2_out, a2_diff],
        cmap=args.cmap,
        overlay=args.overlay,
        base_rgb=orig_rgb,
        diff_indices=[4],
    )

    # ---------- Figure B1 route 1 ----------
    norm_b1 = _robust_joint_range([b7_r1_maps["y_base"], b7_r1_maps["y_carafe"], b7_r1_maps["out"]], p_lo=1, p_hi=99)
    b1_base = _normalize(b7_r1_maps["y_base"], norm_b1)
    b1_carafe = _normalize(b7_r1_maps["y_carafe"], norm_b1)
    b1_out = _normalize(b7_r1_maps["out"], norm_b1)
    b1_diff_raw = b7_r1_maps["out"] - b7_r1_maps["y_base"]
    b1_diff, norm_b1d = _normalize_diff(b1_diff_raw, p=99)

    _plot_panels(
        out_dir / "b7_p5_to_p4_internal.png",
        [
            "Input Image",
            f"b7(P5->P4) y_base ({args.agg})",
            f"b7(P5->P4) y_carafe ({args.agg})",
            f"b7(P5->P4) out ({args.agg})",
            "Diff (out - base)",
        ],
        [orig_rgb, b1_base, b1_carafe, b1_out, b1_diff],
        cmap=args.cmap,
        overlay=args.overlay,
        base_rgb=orig_rgb,
        diff_indices=[4],
    )

    # ---------- Figure B1 route 2 ----------
    norm_b2 = _robust_joint_range([b7_r2_maps["y_base"], b7_r2_maps["y_carafe"], b7_r2_maps["out"]], p_lo=1, p_hi=99)
    b2_base = _normalize(b7_r2_maps["y_base"], norm_b2)
    b2_carafe = _normalize(b7_r2_maps["y_carafe"], norm_b2)
    b2_out = _normalize(b7_r2_maps["out"], norm_b2)
    b2_diff_raw = b7_r2_maps["out"] - b7_r2_maps["y_base"]
    b2_diff, norm_b2d = _normalize_diff(b2_diff_raw, p=99)

    _plot_panels(
        out_dir / "b7_p4_to_p3_internal.png",
        [
            "Input Image",
            f"b7(P4*->P3) y_base ({args.agg})",
            f"b7(P4*->P3) y_carafe ({args.agg})",
            f"b7(P4*->P3) out ({args.agg})",
            "Diff (out - base)",
        ],
        [orig_rgb, b2_base, b2_carafe, b2_out, b2_diff],
        cmap=args.cmap,
        overlay=args.overlay,
        base_rgb=orig_rgb,
        diff_indices=[4],
    )

    # ---------- Figure B2: P4* compare ----------
    norm_p4s = _robust_joint_range([base_p4s, imp_p4s], p_lo=1, p_hi=99)
    p4s_b = _normalize(base_p4s, norm_p4s)
    p4s_i = _normalize(imp_p4s, norm_p4s)
    p4s_diff_raw = imp_p4s - base_p4s
    p4s_d, norm_p4sd = _normalize_diff(p4s_diff_raw, p=99)

    _plot_panels(
        out_dir / "b7_p4star_compare.png",
        [
            "Input Image",
            f"Baseline P4* ({args.agg})",
            f"Improved P4* ({args.agg})",
            "Diff (Improved - Baseline)",
        ],
        [orig_rgb, p4s_b, p4s_i, p4s_d],
        cmap=args.cmap,
        overlay=args.overlay,
        base_rgb=orig_rgb,
        diff_indices=[3],
    )

    # ---------- Figure B2: P3* compare ----------
    norm_p3s = _robust_joint_range([base_p3s, imp_p3s], p_lo=1, p_hi=99)
    p3s_b = _normalize(base_p3s, norm_p3s)
    p3s_i = _normalize(imp_p3s, norm_p3s)
    p3s_diff_raw = imp_p3s - base_p3s
    p3s_d, norm_p3sd = _normalize_diff(p3s_diff_raw, p=99)

    _plot_panels(
        out_dir / "b7_p3star_compare.png",
        [
            "Input Image",
            f"Baseline P3* ({args.agg})",
            f"Improved P3* ({args.agg})",
            "Diff (Improved - Baseline)",
        ],
        [orig_rgb, p3s_b, p3s_i, p3s_d],
        cmap=args.cmap,
        overlay=args.overlay,
        base_rgb=orig_rgb,
        diff_indices=[3],
    )

    meta: Dict[str, Any] = {
        "baseline": {"name": baseline_name, "weight": str(baseline_weight)},
        "improved": {"name": improved_name, "weight": str(improved_weight)},
        "image": image_meta,
        "config": {
            "imgsz": int(args.imgsz),
            "device": str(device),
            "conf": float(args.conf),
            "iou": float(args.iou),
            "max_det": int(args.max_det),
            "agg": args.agg,
            "overlay": bool(args.overlay),
            "cmap": args.cmap,
            "norm_policy": "joint 1%-99% per comparison group; diff uses symmetric abs 99%",
        },
        "identified_layers": {
            "a4_spd_selected": a4_choice.__dict__,
            "a4_spd_all": [x.__dict__ for x in a4_all],
            "b7_p5_to_p4": b7_p5p4.__dict__,
            "b7_p4_to_p3": b7_p4p3.__dict__,
            "b7_locate_info": b7_info,
            "improved_p3_idx": int(ip3_idx),
            "improved_p4star_idx": int(ip4s_idx),
            "improved_p3star_idx": int(ip3s_idx),
            "improved_pstar_locate": pstar_info_i,
            "baseline_p3_idx": int(bp3_idx),
            "baseline_p4star_idx": int(bp4s_idx),
            "baseline_p3star_idx": int(bp3s_idx),
            "baseline_pstar_locate": pstar_info_b,
            "alignment_notes": align_notes,
        },
        "tensor_shapes": {
            "baseline_p3": list(b_outs[bp3_idx].shape),
            "improved_p3": list(i_outs[ip3_idx].shape),
            "baseline_p4star": list(b_outs[bp4s_idx].shape),
            "improved_p4star": list(i_outs[ip4s_idx].shape),
            "baseline_p3star": list(b_outs[bp3s_idx].shape),
            "improved_p3star": list(i_outs[ip3s_idx].shape),
            "a4_y_base": list(a4_tensors["y_base"].shape),
            "a4_y_spd": list(a4_tensors["y_spd"].shape),
            "a4_out": list(a4_tensors["out"].shape),
            "b7_p5p4_y_base": list(b7_r1["y_base"].shape),
            "b7_p5p4_y_carafe": list(b7_r1["y_carafe"].shape),
            "b7_p5p4_out": list(b7_r1["out"].shape),
            "b7_p4p3_y_base": list(b7_r2["y_base"].shape),
            "b7_p4p3_y_carafe": list(b7_r2["y_carafe"].shape),
            "b7_p4p3_out": list(b7_r2["out"].shape),
        },
        "alpha_values": {
            "a4_alpha": float(a4_tensors["alpha"].detach().cpu().item()),
            "b7_p5p4_alpha": float(b7_r1["alpha"].detach().cpu().item()),
            "b7_p4p3_alpha": float(b7_r2["alpha"].detach().cpu().item()),
        },
        "normalization_ranges": {
            "a4_p3_compare": {"main": norm_a1.__dict__, "diff": norm_a1d.__dict__},
            "a4_internal_spd": {"main": norm_a2.__dict__, "diff": norm_a2d.__dict__},
            "b7_p5_to_p4_internal": {"main": norm_b1.__dict__, "diff": norm_b1d.__dict__},
            "b7_p4_to_p3_internal": {"main": norm_b2.__dict__, "diff": norm_b2d.__dict__},
            "b7_p4star_compare": {"main": norm_p4s.__dict__, "diff": norm_p4sd.__dict__},
            "b7_p3star_compare": {"main": norm_p3s.__dict__, "diff": norm_p3sd.__dict__},
        },
    }

    (out_dir / "feature_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    readme_lines = [
        "# Feature Compare Report",
        "",
        "## 输入信息",
        f"- baseline: `{baseline_name}` -> `{baseline_weight}`",
        f"- improved: `{improved_name}` -> `{improved_weight}`",
        f"- image: `{image_path}`",
        f"- 聚合方式: `{args.agg}`",
        f"- 归一化策略: `同组联合 1%-99% 分位；diff 使用对称 abs 99%`",
        "",
        "## 自动识别关键层",
        f"- a4 SPD 模块: `{a4_choice.name}`",
        f"- b7(P5->P4): `{b7_p5p4.name}`",
        f"- b7(P4*->P3): `{b7_p4p3.name}`",
        f"- improved P3 idx: `{ip3_idx}`",
        f"- improved P4* idx: `{ip4s_idx}`",
        f"- improved P3* idx: `{ip3s_idx}`",
        f"- baseline P3 idx: `{bp3_idx}`",
        f"- baseline P4* idx: `{bp4s_idx}`",
        f"- baseline P3* idx: `{bp3s_idx}`",
    ]
    if align_notes:
        readme_lines.extend(["", "## baseline 对齐说明"])
        for n in align_notes:
            readme_lines.append(f"- {n}")

    readme_lines.extend(
        [
            "",
            "## 输出图语义",
            "- `a4_p3_compare.png`: baseline/improved 最终 P3 响应 + diff（improved-baseline）",
            "- `a4_internal_spd_module.png`: improved 的 a4 内部 `y_base / y_spd / out` + diff(out-base)",
            "- `b7_p5_to_p4_internal.png`: improved 的 b7(P5->P4) 内部 `y_base / y_carafe / out` + diff(out-base)",
            "- `b7_p4_to_p3_internal.png`: improved 的 b7(P4*->P3) 内部 `y_base / y_carafe / out` + diff(out-base)",
            "- `b7_p4star_compare.png`: baseline vs improved 的 P4* 响应 + diff",
            "- `b7_p3star_compare.png`: baseline vs improved 的 P3* 响应 + diff",
            "",
            "## 解释建议",
            "- a4: 观察 P3 和 a4 内部 diff 是否在小目标局部区域增强（响应更集中/更亮）。",
            "- b7: 观察 P4*、P3* 及 b7 内部 diff 是否提升 top-down 融合后的响应集中度与边缘可分性。",
            "- diff 图为 `improved - baseline` 或 `out - base`，红/暖色表示增强，蓝/冷色表示抑制。",
        ]
    )

    (out_dir / "README_feature_compare.md").write_text("\n".join(readme_lines), encoding="utf-8")

    print("[done]", out_dir)
    print("[layers] a4:", a4_choice)
    print("[layers] b7 p5->p4:", b7_p5p4)
    print("[layers] b7 p4->p3:", b7_p4p3)
    print("[layers] p3/p4*/p3* improved:", ip3_idx, ip4s_idx, ip3s_idx)
    print("[layers] p3/p4*/p3* baseline:", bp3_idx, bp4s_idx, bp3s_idx)

    return meta


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Compare intermediate feature heatmaps between baseline YOLO11m and improved SD-YOLO11 (a4/b7)."
    )
    p.add_argument("--baseline-weight", type=str, default="", help="Baseline weight path")
    p.add_argument("--improved-weight", type=str, default="", help="Improved weight path")
    p.add_argument("--baseline-name", type=str, default="baseline")
    p.add_argument("--improved-name", type=str, default="improved")
    p.add_argument("--image", type=str, default="", help="Input image path")

    p.add_argument("--imgsz", type=int, default=int(USER_EDIT_CONFIG["imgsz"]))
    p.add_argument("--device", type=str, default=str(USER_EDIT_CONFIG["device"]))
    p.add_argument("--conf", type=float, default=float(USER_EDIT_CONFIG["conf"]))
    p.add_argument("--iou", type=float, default=float(USER_EDIT_CONFIG["iou"]))
    p.add_argument("--max-det", type=int, default=int(USER_EDIT_CONFIG["max_det"]))

    p.add_argument("--agg", type=str, default="abs_mean", choices=["abs_mean", "l2"])
    p.add_argument("--cmap", type=str, default="viridis")
    p.add_argument("--overlay", action="store_true", default=True, help="Overlay heatmap on original image")
    p.add_argument("--no-overlay", dest="overlay", action="store_false")

    p.add_argument("--output-root", type=str, default=str(USER_EDIT_CONFIG["output_root"]))
    p.add_argument("--report-prefix", type=str, default=str(USER_EDIT_CONFIG["report_prefix"]))

    p.add_argument("--auto-layers", action="store_true", default=True)
    p.add_argument("--manual-a4-index", type=int, default=None)
    p.add_argument("--manual-a4-name", type=str, default="")
    p.add_argument("--manual-b7-p5p4-index", type=int, default=None)
    p.add_argument("--manual-b7-p4p3-index", type=int, default=None)
    p.add_argument("--manual-p3-index", type=int, default=None)
    p.add_argument("--manual-p4star-index", type=int, default=None)
    p.add_argument("--manual-p3star-index", type=int, default=None)

    p.add_argument(
        "--use-user-config",
        action="store_true",
        default=False,
        help="Run using USER_EDIT_CONFIG model list (reference_model vs others)",
    )
    return p


def main() -> None:
    args = _build_parser().parse_args()

    output_root = _resolve_path(args.output_root)
    report_dir = output_root / f"{args.report_prefix}{_now_tag()}"
    report_dir.mkdir(parents=True, exist_ok=True)

    runs: List[Dict[str, Any]] = []

    if args.use_user_config or (not args.baseline_weight and not args.improved_weight):
        cfg = copy.deepcopy(USER_EDIT_CONFIG)
        ref_name = str(cfg.get("reference_model", "baseline"))
        models = cfg.get("models", [])
        if not models:
            raise RuntimeError("USER_EDIT_CONFIG.models is empty")

        model_map = {m["name"]: _resolve_path(m["weight"]) for m in models}
        if ref_name not in model_map:
            raise RuntimeError(f"reference_model='{ref_name}' not found in USER_EDIT_CONFIG.models")

        image = args.image or (cfg.get("images") or [""])[0]
        if not image:
            raise RuntimeError("No input image provided (use --image or USER_EDIT_CONFIG.images)")
        image_path = _resolve_path(image)

        baseline_weight = model_map[ref_name]
        for m in models:
            name = m["name"]
            if name == ref_name:
                continue
            improved_weight = model_map[name]
            subdir = report_dir / f"{ref_name}_vs_{name}"
            meta = _run_compare_once(
                ref_name,
                baseline_weight,
                name,
                improved_weight,
                image_path,
                subdir,
                args,
            )
            runs.append({"tag": f"{ref_name}_vs_{name}", "dir": str(subdir), "meta": meta})
    else:
        if not args.baseline_weight or not args.improved_weight or not args.image:
            raise RuntimeError("Single-run mode requires --baseline-weight --improved-weight --image")

        baseline_weight = _resolve_path(args.baseline_weight)
        improved_weight = _resolve_path(args.improved_weight)
        image_path = _resolve_path(args.image)

        meta = _run_compare_once(
            args.baseline_name,
            baseline_weight,
            args.improved_name,
            improved_weight,
            image_path,
            report_dir,
            args,
        )
        runs.append({"tag": f"{args.baseline_name}_vs_{args.improved_name}", "dir": str(report_dir), "meta": meta})

    summary = {
        "report_dir": str(report_dir),
        "runs": [{"tag": r["tag"], "dir": r["dir"]} for r in runs],
    }
    (report_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[report]", report_dir)


if __name__ == "__main__":
    main()
