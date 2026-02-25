from __future__ import annotations

# Path: third_party/yolo11/enhance241/yolo11_241d9.py
# Purpose: enhance241 d9 patch (P3 head score-calib residual block, baseline-safe).

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from .yolo11_241a3 import (
    _bind_module_debug,
    _get_module_recorder,
    _infer_device_dtype,
    _locate_detect,
    _should_capture_delta,
    _tensor_stats,
    _to_float,
    get_check_recorder,
)

ENHANCE241_AUDIT_KEYS = ["enhance241_d9"]  # enhance241-audit


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


def _safe_int(x: Any, default: int) -> int:
    try:
        return int(x)
    except Exception:
        return int(default)


class _ScoreCalibDelta(torch.nn.Module):
    def __init__(self, channels: int, mode: str = "dw") -> None:
        super().__init__()
        c = int(channels)
        m = str(mode).lower()
        if m == "conv":
            self.conv3 = torch.nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1, bias=True)
        else:
            self.conv3 = torch.nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1, groups=c, bias=True)
        self.conv1 = torch.nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0, bias=True)
        torch.nn.init.zeros_(self.conv1.weight)
        if self.conv1.bias is not None:
            torch.nn.init.zeros_(self.conv1.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv1(F.silu(self.conv3(x)))


class D9HeadScoreCalib(torch.nn.Module):
    """x_head = x + alpha * conv1x1(conv3x3(x)) on P3 input."""

    def __init__(
        self,
        base_module: torch.nn.Module,
        channels: int,
        mode: str = "dw",
        alpha_init: float = 0.0,
        alpha_cap: float = 0.5,
    ) -> None:
        super().__init__()
        self.enhance241_d9_base = base_module
        self.enhance241_d9_delta = _ScoreCalibDelta(channels=int(channels), mode=str(mode))
        self.alpha_cap = float(max(1e-6, abs(alpha_cap)))
        alpha_init = float(max(-self.alpha_cap * 0.95, min(self.alpha_cap * 0.95, alpha_init)))
        alpha_raw = torch.atanh(torch.tensor(alpha_init / self.alpha_cap, dtype=torch.float32))
        self.enhance241_d9_alpha = torch.nn.Parameter(alpha_raw)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        recorder = _get_module_recorder(self)
        prefix = str(getattr(self, "_enhance241_prefix", "d9"))
        step = int(getattr(self, "_enhance241_fwd_step", 0))
        if recorder is not None and step == 0:
            recorder.capture_param_before(self, prefix)

        y_base = self.enhance241_d9_base(x)
        delta_raw = self.enhance241_d9_delta(y_base)
        alpha_raw = self.enhance241_d9_alpha.to(dtype=y_base.dtype, device=y_base.device)
        alpha = torch.tanh(alpha_raw) * self.alpha_cap
        delta = alpha * delta_raw
        out = y_base + delta

        if recorder is not None:
            if step == 0:
                recorder.record_module_compare(f"{prefix}.p3", patched=out, baseline=y_base, input_tensor=x)
                recorder.record_a1_payload(
                    f"{prefix}.gate1",
                    {
                        "alpha_raw": _to_float(alpha_raw.item()),
                        "alpha": _to_float(alpha.item()),
                        "base_stats": _tensor_stats(y_base),
                        "delta_stats": _tensor_stats(delta),
                        "out_stats": _tensor_stats(out),
                    },
                )
                recorder.record_scalar_curve(f"{prefix}.alpha_raw", _to_float(alpha_raw.item()), step, max_steps=100)
                recorder.record_scalar_curve(f"{prefix}.alpha", _to_float(alpha.item()), step, max_steps=100)
                if out.requires_grad:
                    try:
                        out.register_hook(lambda g, n=f"{prefix}.p3": recorder.record_output_grad(n, g))
                    except Exception:
                        pass
            elif step <= 100:
                recorder.record_scalar_curve(f"{prefix}.alpha_raw", _to_float(alpha_raw.item()), step, max_steps=100)
                recorder.record_scalar_curve(f"{prefix}.alpha", _to_float(alpha.item()), step, max_steps=100)
            if _should_capture_delta(step):
                recorder.capture_param_delta(self, prefix)
        setattr(self, "_enhance241_fwd_step", step + 1)
        return out


def _extract_model_seq(model: Any) -> Tuple[Any, Optional[Any]]:
    yolo_obj = model
    if hasattr(model, "model") and hasattr(getattr(model, "model"), "model"):
        det_model = getattr(model, "model")
        seq = getattr(det_model, "model", None)
    else:
        det_model = model
        seq = getattr(det_model, "model", None)
    return yolo_obj, seq


def _infer_p3_output_index(seq: Any) -> int:
    _, detect = _locate_detect(seq)
    f = getattr(detect, "f", [])
    if isinstance(f, int):
        return int(f)
    if isinstance(f, (list, tuple)) and f:
        return int(f[0])
    raise RuntimeError(f"Unable to locate P3 output index from Detect.f={f}")


def _infer_p3_channels(detect: Any) -> int:
    try:
        return int(detect.cv2[0][0].conv.in_channels)  # type: ignore[index]
    except Exception:
        pass
    try:
        first = detect.cv3[0]  # type: ignore[index]
        for m in first.modules():
            if isinstance(m, torch.nn.Conv2d):
                return int(m.in_channels)
    except Exception:
        pass
    raise RuntimeError("Unable to infer P3 channels for d9.")


def apply(model: Any, cfg: Any) -> Any:
    enable_d9 = bool(_deep_get(cfg, "enhance241", "d9", default=False))
    if not enable_d9:
        return model
    if any(bool(_deep_get(cfg, "enhance241", k, default=False)) for k in ("d3", "d5", "d7", "d1")):
        raise RuntimeError("enhance241.d9 conflicts with d1/d3/d5/d7; enable only one D-class module.")

    yolo_obj, seq = _extract_model_seq(model)
    if seq is None:
        raise RuntimeError("enhance241.d9 requires YOLO/DetectionModel-like object with .model sequence.")
    p3_idx = _infer_p3_output_index(seq)
    detect_idx, detect = _locate_detect(seq)
    old = seq[p3_idx]

    if isinstance(old, D9HeadScoreCalib):
        info = {
            "enabled": True,
            "existing_count": 1,
            "patched_count": 0,
            "p3_index": int(p3_idx),
            "detect_idx": int(detect_idx),
            "note": "already_patched",
        }
        setattr(yolo_obj, "_enhance241_d9_info", info)
        recorder = get_check_recorder("d9", cfg, patch_info=info)
        if recorder is not None:
            try:
                _bind_module_debug(old, recorder, prefix=f"d9.idx{p3_idx}.existing")
                recorder.register_module_params(old, f"d9.idx{p3_idx}.existing")
                recorder.attach_detect_hooks(detect, detect_idx)
                recorder.maybe_run_val_separability(yolo_obj, cfg)
            except Exception as exc:
                recorder.add_note(f"d9_prepatched_hook_failed:{exc}")
        return yolo_obj

    channels = _infer_p3_channels(detect)
    mode = str(_deep_get(cfg, "enhance241", "d9_mode", default="dw")).lower()
    alpha_init = _safe_float(_deep_get(cfg, "enhance241", "d9_alpha_init", default=0.0), 0.0)
    alpha_cap = _safe_float(_deep_get(cfg, "enhance241", "d9_alpha_cap", default=0.3), 0.3)
    alpha_auto_fallback = False

    wrapped = D9HeadScoreCalib(
        old,
        channels=channels,
        mode=mode,
        alpha_init=alpha_init,
        alpha_cap=alpha_cap,
    )
    for attr in ("i", "f", "type"):
        if hasattr(old, attr):
            setattr(wrapped, attr, getattr(old, attr))
    device, dtype = _infer_device_dtype(seq, p3_idx)
    if device is not None:
        if dtype is not None:
            wrapped = wrapped.to(device=device, dtype=dtype)
        else:
            wrapped = wrapped.to(device=device)
    seq[p3_idx] = wrapped

    old_params = sum(int(p.numel()) for p in old.parameters())
    new_params = sum(int(p.numel()) for p in wrapped.parameters())
    info = {
        "enabled": True,
        "existing_count": 0,
        "patched_count": 1,
        "p3_index": int(p3_idx),
        "detect_idx": int(detect_idx),
        "base_type": old.__class__.__name__,
        "new_type": "D9HeadScoreCalib",
        "channels": int(channels),
        "mode": str(mode),
        "alpha_init": _to_float(alpha_init),
        "alpha_auto_fallback": bool(alpha_auto_fallback),
        "alpha_cap": _to_float(alpha_cap),
        "params_old": int(old_params),
        "params_new": int(new_params),
        "delta_params": int(new_params - old_params),
    }
    setattr(yolo_obj, "_enhance241_d9_info", info)

    recorder = get_check_recorder("d9", cfg, patch_info=info)
    if recorder is not None:
        _bind_module_debug(wrapped, recorder, prefix=f"d9.idx{p3_idx}")
        recorder.register_module_params(wrapped, f"d9.idx{p3_idx}")
        try:
            recorder.attach_detect_hooks(detect, detect_idx)
            recorder.maybe_run_val_separability(yolo_obj, cfg)
        except Exception as exc:
            recorder.add_note(f"d9_hook_failed:{exc}")
    return yolo_obj
