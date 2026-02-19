from __future__ import annotations

# Path: third_party/yolo11/enhance241/yolo11_241a5.py
# Purpose: enhance241 a5 patch (P3-side residual-safe lightweight enhancement).

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from .yolo11_241a3 import (
    _bind_module_debug,
    _get_module_recorder,
    _infer_device_dtype,
    _locate_detect,
    _tensor_stats,
    _to_float,
    get_check_recorder,
)

ENHANCE241_AUDIT_KEYS = ["enhance241_a5"]  # enhance241-audit


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


def _safe_int(x: Any, default: int) -> int:
    try:
        return int(x)
    except Exception:
        return int(default)


def _safe_float(x: Any, default: float) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


class _DWSeparableConv(torch.nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.dw = torch.nn.Conv2d(
            channels, channels, kernel_size=3, stride=1, padding=1, groups=channels, bias=True
        )
        self.act1 = torch.nn.SiLU(inplace=True)
        self.pw = torch.nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.act2 = torch.nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act2(self.pw(self.act1(self.dw(x))))


class _ConvStack(torch.nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = torch.nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.act1 = torch.nn.SiLU(inplace=True)
        self.conv2 = torch.nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.act2 = torch.nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act2(self.conv2(self.act1(self.conv1(x))))


class A5P3Residual(torch.nn.Module):
    """Residual-safe enhancer on P3 path: out = base(x) + alpha * delta(base(x))."""

    def __init__(self, base_module: torch.nn.Module, channels: int, refine: str = "dw") -> None:
        super().__init__()
        self.enhance241_a5_base = base_module
        self.channels = int(channels)
        refine = str(refine).lower()
        if refine == "conv":
            self.enhance241_a5_delta = _ConvStack(self.channels)  # enhance241-audit
        else:
            self.enhance241_a5_delta = _DWSeparableConv(self.channels)  # enhance241-audit

        # Safe start: force delta branch output close to zero at step0.
        last_conv = None
        for m in self.enhance241_a5_delta.modules():
            if isinstance(m, torch.nn.Conv2d):
                last_conv = m
        if last_conv is not None:
            torch.nn.init.zeros_(last_conv.weight)
            if last_conv.bias is not None:
                torch.nn.init.zeros_(last_conv.bias)

        self.enhance241_a5_alpha = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        recorder = _get_module_recorder(self)
        prefix = str(getattr(self, "_enhance241_prefix", "a5"))
        step = int(getattr(self, "_enhance241_fwd_step", 0))

        if recorder is not None and step == 0:
            recorder.capture_param_before(self, prefix)

        y_base = self.enhance241_a5_base(x)
        delta_raw = self.enhance241_a5_delta(y_base)
        alpha = self.enhance241_a5_alpha.to(dtype=y_base.dtype, device=y_base.device)
        delta = alpha * delta_raw
        out = y_base + delta

        if recorder is not None:
            if step == 0:
                recorder.record_module_compare(f"{prefix}.p3", patched=out, baseline=y_base, input_tensor=x)
                v_base = _safe_float(y_base.detach().float().var(unbiased=False).item(), 0.0)
                v_out = _safe_float(out.detach().float().var(unbiased=False).item(), 0.0)
                cos = _safe_float(
                    F.cosine_similarity(
                        out.detach().float().reshape(out.shape[0], -1),
                        y_base.detach().float().reshape(y_base.shape[0], -1),
                        dim=1,
                    ).mean().item(),
                    0.0,
                )
                gate1_ok = bool(cos >= 0.98 and (v_out / (v_base + 1e-12)) >= 0.5)
                recorder.record_a1_payload(
                    f"{prefix}.gate1",
                    {
                        "cosine_out_vs_base": cos,
                        "var_ratio_out_over_base": v_out / (v_base + 1e-12),
                        "thresholds": {"cosine_min": 0.98, "var_ratio_min": 0.5},
                        "pass": gate1_ok,
                        "alpha": _to_float(alpha.item()),
                        "base_stats": _tensor_stats(y_base),
                        "delta_stats": _tensor_stats(delta),
                        "out_stats": _tensor_stats(out),
                    },
                )
                if not gate1_ok:
                    recorder.add_note(f"{prefix}: Gate-1 failed (cos={cos:.4f}, var_ratio={v_out/(v_base+1e-12):.4f})")
                recorder.record_scalar_curve(f"{prefix}.alpha", _to_float(alpha.item()), step, max_steps=100)
                if out.requires_grad:
                    try:
                        out.register_hook(lambda g, n=f"{prefix}.p3": recorder.record_output_grad(n, g))
                    except Exception:
                        pass
            elif step <= 100:
                recorder.record_scalar_curve(f"{prefix}.alpha", _to_float(alpha.item()), step, max_steps=100)
            if step == 1:
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
    raise RuntimeError("Unable to infer P3 channels for a5.")


def apply(model: Any, cfg: Any) -> Any:
    enable_a5 = bool(_deep_get(cfg, "enhance241", "a5", default=False))
    if not enable_a5:
        return model

    yolo_obj, seq = _extract_model_seq(model)
    if seq is None:
        raise RuntimeError("enhance241.a5 requires YOLO/DetectionModel-like object with .model sequence.")

    p3_idx = _infer_p3_output_index(seq)
    detect_idx, detect = _locate_detect(seq)
    old = seq[p3_idx]

    if isinstance(old, A5P3Residual):
        info = {
            "enabled": True,
            "existing_count": 1,
            "patched_count": 0,
            "p3_index": int(p3_idx),
            "detect_idx": int(detect_idx),
            "note": "already_patched",
        }
        setattr(yolo_obj, "_enhance241_a5_info", info)
        recorder = get_check_recorder("a5", cfg, patch_info=info)
        if recorder is not None:
            try:
                _bind_module_debug(old, recorder, prefix=f"a5.idx{p3_idx}.existing")
                recorder.register_module_params(old, f"a5.idx{p3_idx}.existing")
                recorder.attach_detect_hooks(detect, detect_idx)
                recorder.maybe_run_val_separability(yolo_obj, cfg)
            except Exception as exc:
                recorder.add_note(f"a5_prepatched_hook_failed:{exc}")
        return yolo_obj

    refine = str(_deep_get(cfg, "enhance241", "a5_refine", default="dw"))
    _ = _safe_int(_deep_get(cfg, "enhance241", "a5_depth", default=1), 1)  # reserved
    channels = _infer_p3_channels(detect)

    wrapped = A5P3Residual(old, channels=channels, refine=refine)
    for attr in ("i", "f", "type"):
        if hasattr(old, attr):
            setattr(wrapped, attr, getattr(old, attr))
    device, dtype = _infer_device_dtype(seq, p3_idx)
    if device is not None:
        if dtype is not None:
            wrapped = wrapped.to(device=device, dtype=dtype)
        else:
            wrapped = wrapped.to(device=device)

    old_params = sum(int(p.numel()) for p in old.parameters())
    new_params = sum(int(p.numel()) for p in wrapped.parameters())
    seq[p3_idx] = wrapped

    info = {
        "enabled": True,
        "existing_count": 0,
        "patched_count": 1,
        "p3_index": int(p3_idx),
        "detect_idx": int(detect_idx),
        "base_type": old.__class__.__name__,
        "new_type": "A5P3Residual",
        "channels": int(channels),
        "params_old": int(old_params),
        "params_new": int(new_params),
        "delta_params": int(new_params - old_params),
        "alpha_init": 0.0,
        "refine": refine,
        "gate1_thresholds": {"cosine_min": 0.98, "var_ratio_min": 0.5},
    }
    setattr(yolo_obj, "_enhance241_a5_info", info)

    recorder = get_check_recorder("a5", cfg, patch_info=info)
    if recorder is not None:
        _bind_module_debug(wrapped, recorder, prefix=f"a5.idx{p3_idx}")
        recorder.register_module_params(wrapped, f"a5.idx{p3_idx}")
        try:
            recorder.attach_detect_hooks(detect, detect_idx)
        except Exception as exc:
            recorder.add_note(f"a5_detect_hook_failed:{exc}")
        try:
            recorder.maybe_run_val_separability(yolo_obj, cfg)
        except Exception as exc:
            recorder.add_note(f"a5_val_check_failed:{exc}")

    return yolo_obj
