from __future__ import annotations

# Path: third_party/yolo11/enhance241/yolo11_241c9.py
# Purpose: enhance241 c9 patch (SE-SAM guardrail before head, residual-safe).

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

ENHANCE241_AUDIT_KEYS = ["enhance241_c9"]  # enhance241-audit


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


class _ChannelSE(torch.nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        c = int(channels)
        hidden = max(4, c // max(1, int(reduction)))
        self.fc1 = torch.nn.Conv2d(c, hidden, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc2 = torch.nn.Conv2d(hidden, c, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = F.adaptive_avg_pool2d(x, output_size=1)
        s = torch.sigmoid(self.fc2(F.silu(self.fc1(z))))
        return x * s


class _SpatialSAM(torch.nn.Module):
    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        k = max(3, int(kernel_size))
        if k % 2 == 0:
            k += 1
        self.conv = torch.nn.Conv2d(2, 1, kernel_size=k, stride=1, padding=k // 2, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_map = x.mean(dim=1, keepdim=True)
        max_map = x.amax(dim=1, keepdim=True)
        m = torch.sigmoid(self.conv(torch.cat([avg_map, max_map], dim=1)))
        return x * m


class _SESAMCore(torch.nn.Module):
    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7, mode: str = "full") -> None:
        super().__init__()
        self.mode = str(mode).lower()
        self.se = _ChannelSE(channels=int(channels), reduction=int(reduction))
        self.sam = _SpatialSAM(kernel_size=int(kernel_size))
        self.out = torch.nn.Conv2d(int(channels), int(channels), kernel_size=1, stride=1, padding=0, bias=True)
        torch.nn.init.zeros_(self.out.weight)
        if self.out.bias is not None:
            torch.nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x
        if self.mode in ("full", "channel"):
            y = self.se(y)
        if self.mode in ("full", "spatial"):
            y = self.sam(y)
        return self.out(y)


class C9SESAMGuard(torch.nn.Module):
    """Guardrail gate before head: out = base + alpha * delta(base)."""

    def __init__(
        self,
        base_module: torch.nn.Module,
        channels: int,
        reduction: int = 16,
        kernel_size: int = 7,
        mode: str = "full",
        alpha_init: float = 0.0,
        alpha_cap: float = 0.5,
    ) -> None:
        super().__init__()
        self.enhance241_c9_base = base_module
        self.enhance241_c9_delta = _SESAMCore(
            channels=int(channels),
            reduction=int(reduction),
            kernel_size=int(kernel_size),
            mode=str(mode),
        )
        self.alpha_cap = float(max(1e-6, abs(alpha_cap)))
        alpha_init = float(max(-self.alpha_cap * 0.95, min(self.alpha_cap * 0.95, alpha_init)))
        alpha_raw = torch.atanh(torch.tensor(alpha_init / self.alpha_cap, dtype=torch.float32))
        self.enhance241_c9_alpha = torch.nn.Parameter(alpha_raw)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        recorder = _get_module_recorder(self)
        prefix = str(getattr(self, "_enhance241_prefix", "c9"))
        step = int(getattr(self, "_enhance241_fwd_step", 0))
        if recorder is not None and step == 0:
            recorder.capture_param_before(self, prefix)

        y_base = self.enhance241_c9_base(x)
        delta_raw = self.enhance241_c9_delta(y_base)
        alpha_raw = self.enhance241_c9_alpha.to(dtype=y_base.dtype, device=y_base.device)
        alpha = torch.tanh(alpha_raw) * self.alpha_cap
        delta = alpha * delta_raw
        out = y_base + delta

        if recorder is not None:
            if step == 0:
                recorder.record_module_compare(
                    f"{prefix}.p3",
                    patched=out,
                    baseline=y_base,
                    input_tensor=_to_input_tensor(x),
                )
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


def _to_input_tensor(x: Any) -> Optional[torch.Tensor]:
    if isinstance(x, torch.Tensor):
        return x
    if isinstance(x, (list, tuple)):
        for item in x:
            if isinstance(item, torch.Tensor):
                return item
    return None


def _infer_p3_head_index(detect: Any) -> int:
    f = getattr(detect, "f", [])
    if isinstance(f, int):
        return 0
    if isinstance(f, (list, tuple)):
        if len(f) >= 4:
            return 1
        if len(f) >= 1:
            return 0
    raise RuntimeError(f"Unable to locate P3 head index from Detect.f={f}")


def _infer_p3_output_index(seq: Any) -> int:
    _, detect = _locate_detect(seq)
    f = getattr(detect, "f", [])
    if isinstance(f, int):
        return int(f)
    if isinstance(f, (list, tuple)) and f:
        return int(f[_infer_p3_head_index(detect)])
    raise RuntimeError(f"Unable to locate P3 output index from Detect.f={f}")


def _infer_p3_channels(detect: Any) -> int:
    p3_head_idx = _infer_p3_head_index(detect)
    try:
        return int(detect.cv2[p3_head_idx][0].conv.in_channels)  # type: ignore[index]
    except Exception:
        pass
    try:
        first = detect.cv3[p3_head_idx]  # type: ignore[index]
        for m in first.modules():
            if isinstance(m, torch.nn.Conv2d):
                return int(m.in_channels)
    except Exception:
        pass
    raise RuntimeError("Unable to infer P3 channels for c9.")


def apply(model: Any, cfg: Any) -> Any:
    enable_c9 = bool(_deep_get(cfg, "enhance241", "c9", default=False))
    if not enable_c9:
        return model
    if any(bool(_deep_get(cfg, "enhance241", k, default=False)) for k in ("c5", "c7")):
        raise RuntimeError("enhance241.c9 conflicts with c5/c7; enable only one C-class module.")

    yolo_obj, seq = _extract_model_seq(model)
    if seq is None:
        raise RuntimeError("enhance241.c9 requires YOLO/DetectionModel-like object with .model sequence.")
    p3_idx = _infer_p3_output_index(seq)
    detect_idx, detect = _locate_detect(seq)
    old = seq[p3_idx]

    if isinstance(old, C9SESAMGuard):
        info = {
            "enabled": True,
            "existing_count": 1,
            "patched_count": 0,
            "p3_index": int(p3_idx),
            "detect_idx": int(detect_idx),
            "note": "already_patched",
        }
        setattr(yolo_obj, "_enhance241_c9_info", info)
        recorder = get_check_recorder("c9", cfg, patch_info=info)
        if recorder is not None:
            try:
                _bind_module_debug(old, recorder, prefix=f"c9.idx{p3_idx}.existing")
                recorder.register_module_params(old, f"c9.idx{p3_idx}.existing")
                recorder.attach_detect_hooks(detect, detect_idx)
                recorder.maybe_run_val_separability(yolo_obj, cfg)
            except Exception as exc:
                recorder.add_note(f"c9_prepatched_hook_failed:{exc}")
        return yolo_obj

    channels = _infer_p3_channels(detect)
    reduction = _safe_int(_deep_get(cfg, "enhance241", "c9_reduction", default=16), 16)
    kernel_size = _safe_int(_deep_get(cfg, "enhance241", "c9_kernel_size", default=7), 7)
    mode = str(_deep_get(cfg, "enhance241", "c9_mode", default="full")).lower()
    alpha_init = _safe_float(_deep_get(cfg, "enhance241", "c9_alpha_init", default=0.0), 0.0)
    alpha_cap = _safe_float(_deep_get(cfg, "enhance241", "c9_alpha_cap", default=0.3), 0.3)
    alpha_auto_fallback = False

    wrapped = C9SESAMGuard(
        old,
        channels=channels,
        reduction=reduction,
        kernel_size=kernel_size,
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
        "new_type": "C9SESAMGuard",
        "channels": int(channels),
        "reduction": int(reduction),
        "kernel_size": int(kernel_size),
        "mode": str(mode),
        "alpha_init": _to_float(alpha_init),
        "alpha_auto_fallback": bool(alpha_auto_fallback),
        "alpha_cap": _to_float(alpha_cap),
        "params_old": int(old_params),
        "params_new": int(new_params),
        "delta_params": int(new_params - old_params),
    }
    setattr(yolo_obj, "_enhance241_c9_info", info)

    recorder = get_check_recorder("c9", cfg, patch_info=info)
    if recorder is not None:
        _bind_module_debug(wrapped, recorder, prefix=f"c9.idx{p3_idx}")
        recorder.register_module_params(wrapped, f"c9.idx{p3_idx}")
        try:
            recorder.attach_detect_hooks(detect, detect_idx)
            recorder.maybe_run_val_separability(yolo_obj, cfg)
        except Exception as exc:
            recorder.add_note(f"c9_hook_failed:{exc}")
    return yolo_obj
