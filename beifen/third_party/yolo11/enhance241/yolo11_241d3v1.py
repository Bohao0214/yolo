from __future__ import annotations

# Path: third_party/yolo11/enhance241/yolo11_241d1.py
# Purpose: enhance241 d3 (temperature on P3 cls logits) with d1 backward compatibility.

from typing import Any, Dict, Optional, Tuple

import torch

from .yolo11_241a3 import _bind_module_debug, _get_module_recorder, get_check_recorder

ENHANCE241_AUDIT_KEYS = ["enhance241_d3", "enhance241_d1"]  # enhance241-audit


def _deep_get(mapping: Any, *keys: str, default: Any = None) -> Any:
    cur = mapping
    for k in keys:
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(k)
            continue
        if hasattr(cur, k):
            cur = getattr(cur, k)
            continue
        return default
    return cur if cur is not None else default


def _safe_float(x: Any, default: float) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


class P3LogitTemperature(torch.nn.Module):
    """Apply learnable temperature scaling to P3 cls logits only."""

    def __init__(
        self,
        base_cls_head: torch.nn.Module,
        temp_init: float = 1.0,
        t_min: float = 0.5,
        t_max: float = 4.0,
    ) -> None:
        super().__init__()
        self.enhance241_d3_base_cls = base_cls_head
        self.enhance241_d3_temp = torch.nn.Parameter(torch.tensor(float(temp_init), dtype=torch.float32))
        self.t_min = float(t_min)
        self.t_max = float(t_max)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        recorder = _get_module_recorder(self)
        prefix = str(getattr(self, "_enhance241_prefix", "d3"))
        step = int(getattr(self, "_enhance241_fwd_step", 0))

        logits = self.enhance241_d3_base_cls(x)
        t = torch.clamp(self.enhance241_d3_temp, min=self.t_min, max=self.t_max)
        t = t.to(dtype=logits.dtype, device=logits.device)
        out = logits / t

        if recorder is not None:
            recorder.record_scalar_curve(f"{prefix}.temperature", float(t.item()), step, max_steps=30)
            conf = torch.sigmoid(out.detach().float()).reshape(-1)
            recorder.record_distribution(f"{prefix}.conf", conf, step, max_steps=30)
            if step == 0:
                recorder.record_a1_payload(
                    f"{prefix}.temperature_cfg",
                    {
                        "t_init": float(t.item()),
                        "t_min": float(self.t_min),
                        "t_max": float(self.t_max),
                    },
                )
                if out.requires_grad:
                    try:
                        out.register_hook(lambda g, n=f"{prefix}.logits": recorder.record_output_grad(n, g))
                    except Exception:
                        pass
            if step == 1:
                recorder.capture_param_delta(self, prefix)

        setattr(self, "_enhance241_fwd_step", step + 1)
        return out


def _extract_model_seq(model: Any) -> Tuple[Any, Any, Optional[Any]]:
    yolo_obj = model
    if hasattr(model, "model") and hasattr(getattr(model, "model"), "model"):
        det_model = getattr(model, "model")
        seq = getattr(det_model, "model", None)
    else:
        det_model = model
        seq = getattr(det_model, "model", None)
    return yolo_obj, det_model, seq


def _locate_detect(seq: Any) -> Tuple[int, Any]:
    if not seq:
        raise RuntimeError("Empty model sequence.")
    idx = len(seq) - 1
    det = seq[idx]
    if det.__class__.__name__.lower() == "detect":
        return idx, det
    for i, layer in enumerate(seq):
        if layer.__class__.__name__.lower() == "detect":
            return i, layer
    raise RuntimeError("Detect layer not found.")


def apply(model: Any, cfg: Any) -> Any:
    """Apply d3 temperature scaling (d1 backward compatible flag)."""

    enh = _deep_get(cfg, "enhance241", default={}) or {}
    if not isinstance(enh, dict):
        enh = {}

    enable_d3 = bool(enh.get("d3", False)) or bool(enh.get("d1", False))
    if not enable_d3:
        return model

    yolo_obj, _, seq = _extract_model_seq(model)
    if seq is None:
        raise RuntimeError("enhance241.d3 requires an ultralytics YOLO/DetectionModel-like object with a .model sequence.")

    detect_idx, detect = _locate_detect(seq)
    cls_heads = getattr(detect, "cv3", None)
    if cls_heads is None or not isinstance(cls_heads, torch.nn.ModuleList) or len(cls_heads) < 1:
        raise RuntimeError("enhance241.d3 expects Detect.cv3 ModuleList with at least one head.")

    temp_init = _safe_float(enh.get("d3_temp_init", enh.get("d1_temp_init", 1.0)), 1.0)
    t_min = _safe_float(enh.get("d3_temp_min", 0.5), 0.5)
    t_max = _safe_float(enh.get("d3_temp_max", 4.0), 4.0)

    if isinstance(cls_heads[0], P3LogitTemperature):
        wrapped = cls_heads[0]
        info = {
            "enabled": True,
            "mode": "d3_temperature",
            "existing_count": 1,
            "patched_count": 0,
            "detect_idx": int(detect_idx),
            "p3_head_index": 0,
            "detect_heads_after": int(getattr(detect, "nl", len(cls_heads))),
            "temp": float(torch.clamp(wrapped.enhance241_d3_temp.detach(), min=t_min, max=t_max).item()),
            "temp_range": [float(t_min), float(t_max)],
        }
        setattr(yolo_obj, "_enhance241_d1_info", info)
        setattr(yolo_obj, "_enhance241_d3_info", info)

        recorder = get_check_recorder("d3", cfg, patch_info=info)
        if recorder is not None:
            _bind_module_debug(wrapped, recorder, prefix=f"d3.detect{detect_idx}.p3head.existing")
            recorder.register_module_params(wrapped, f"d3.detect{detect_idx}.p3head.existing")
            recorder.maybe_run_val_separability(yolo_obj, cfg)
        return yolo_obj

    old_cls = cls_heads[0]
    wrapped = P3LogitTemperature(old_cls, temp_init=temp_init, t_min=t_min, t_max=t_max)

    try:
        p = next(old_cls.parameters())
        wrapped = wrapped.to(device=p.device, dtype=p.dtype)
    except Exception:
        pass

    cls_heads[0] = wrapped

    info = {
        "enabled": True,
        "mode": "d3_temperature",
        "existing_count": 0,
        "patched_count": 1,
        "detect_idx": int(detect_idx),
        "p3_head_index": 0,
        "detect_heads_after": int(getattr(detect, "nl", len(cls_heads))),
        "temp_init": float(temp_init),
        "temp_range": [float(t_min), float(t_max)],
    }
    setattr(yolo_obj, "_enhance241_d1_info", info)
    setattr(yolo_obj, "_enhance241_d3_info", info)

    recorder = get_check_recorder("d3", cfg, patch_info=info)
    if recorder is not None:
        _bind_module_debug(wrapped, recorder, prefix=f"d3.detect{detect_idx}.p3head")
        recorder.register_module_params(wrapped, f"d3.detect{detect_idx}.p3head")
        recorder.maybe_run_val_separability(yolo_obj, cfg)

    return yolo_obj
