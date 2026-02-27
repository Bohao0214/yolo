from __future__ import annotations

# Path: third_party/yolo11/enhance241/yolo11_241a7.py
# Purpose: enhance241 a7 patch (HorNet/C3HB-like residual-safe P3 enhancer).

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

ENHANCE241_AUDIT_KEYS = ["enhance241_a7"]  # enhance241-audit


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


class _DWConvBlock(torch.nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.dw = torch.nn.Conv2d(
            channels, channels, kernel_size=3, stride=1, padding=1, groups=channels, bias=True
        )
        self.pw = torch.nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.act = torch.nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.pw(self.act(self.dw(x))))


class _ConvBlock(torch.nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.act = torch.nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(x))


class HorNetC3HBDelta(torch.nn.Module):
    """HorNet-inspired recursive gated interaction delta branch.

    Paper-style notation (simplified engineering form):
      P0 = f(x)
      C0 = C / 2^(alpha-1)
      phi(Q0) = [Q1, Q2, ..., Q_alpha]
      P_i = f(P_{i-1} ⊙ Q_{i-1}), i >= 1
    Here we keep channel shape invariant and realize recursive gating by
    repeated gated blocks with learnable gates from x.
    """

    def __init__(self, channels: int, order: int = 3, refine: str = "dw") -> None:
        super().__init__()
        self.channels = int(channels)
        self.order = max(2, int(order))
        refine = str(refine).lower()

        self.pre = torch.nn.Conv2d(self.channels, self.channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.gate = torch.nn.Conv2d(
            self.channels, self.channels * self.order, kernel_size=1, stride=1, padding=0, bias=True
        )
        block_cls = _ConvBlock if refine == "conv" else _DWConvBlock
        self.blocks = torch.nn.ModuleList([block_cls(self.channels) for _ in range(self.order)])
        self.out = torch.nn.Conv2d(self.channels, self.channels, kernel_size=1, stride=1, padding=0, bias=True)

        # Safe start: delta branch final projection starts from zero.
        torch.nn.init.zeros_(self.out.weight)
        if self.out.bias is not None:
            torch.nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p = self.pre(x)
        gates = torch.chunk(self.gate(x), self.order, dim=1)
        for i, blk in enumerate(self.blocks):
            p = blk(p * torch.sigmoid(gates[i]))
        return self.out(p)


class C3HBHorNetSafe(torch.nn.Module):
    """Residual-safe a7 wrapper: out = base(x) + alpha * HorNetDelta(base(x))."""

    def __init__(
        self,
        base_module: torch.nn.Module,
        channels: int,
        order: int = 3,
        refine: str = "dw",
        alpha_init: float = 0.0,
        alpha_cap: float = 0.5,
    ) -> None:
        super().__init__()
        self.enhance241_a7_base = base_module
        self.alpha_cap = float(max(1e-6, abs(alpha_cap)))
        self.enhance241_a7_delta = HorNetC3HBDelta(channels=int(channels), order=int(order), refine=str(refine))

        alpha_init = float(max(-self.alpha_cap * 0.95, min(self.alpha_cap * 0.95, alpha_init)))
        alpha_ratio = alpha_init / self.alpha_cap
        alpha_raw = torch.atanh(torch.tensor(alpha_ratio, dtype=torch.float32))
        self.enhance241_a7_alpha = torch.nn.Parameter(alpha_raw)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        recorder = _get_module_recorder(self)
        prefix = str(getattr(self, "_enhance241_prefix", "a7"))
        step = int(getattr(self, "_enhance241_fwd_step", 0))

        if recorder is not None and step == 0:
            recorder.capture_param_before(self, prefix)

        y_base = self.enhance241_a7_base(x)
        delta_raw = self.enhance241_a7_delta(y_base)
        alpha_raw = self.enhance241_a7_alpha.to(dtype=y_base.dtype, device=y_base.device)
        alpha = torch.tanh(alpha_raw) * self.alpha_cap
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
                        "alpha_raw": _to_float(alpha_raw.item()),
                        "alpha": _to_float(alpha.item()),
                        "alpha_cap": _to_float(self.alpha_cap),
                        "base_stats": _tensor_stats(y_base),
                        "delta_stats": _tensor_stats(delta),
                        "out_stats": _tensor_stats(out),
                    },
                )
                if not gate1_ok:
                    recorder.add_note(f"{prefix}: Gate-1 failed (cos={cos:.4f}, var_ratio={v_out/(v_base+1e-12):.4f})")
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
    raise RuntimeError("Unable to infer P3 channels for a7.")


def apply(model: Any, cfg: Any) -> Any:
    enable_a7 = bool(_deep_get(cfg, "enhance241", "a7", default=False))
    if not enable_a7:
        return model

    if any(bool(_deep_get(cfg, "enhance241", k, default=False)) for k in ("a3", "a5", "a9")):
        raise RuntimeError("enhance241.a7 conflicts with a3/a5/a9; enable only one A-class module.")

    yolo_obj, seq = _extract_model_seq(model)
    if seq is None:
        raise RuntimeError("enhance241.a7 requires YOLO/DetectionModel-like object with .model sequence.")

    p3_idx = _infer_p3_output_index(seq)
    detect_idx, detect = _locate_detect(seq)
    old = seq[p3_idx]

    if isinstance(old, C3HBHorNetSafe):
        info = {
            "enabled": True,
            "existing_count": 1,
            "patched_count": 0,
            "p3_index": int(p3_idx),
            "detect_idx": int(detect_idx),
            "note": "already_patched",
        }
        setattr(yolo_obj, "_enhance241_a7_info", info)
        recorder = get_check_recorder("a7", cfg, patch_info=info)
        if recorder is not None:
            try:
                _bind_module_debug(old, recorder, prefix=f"a7.idx{p3_idx}.existing")
                recorder.register_module_params(old, f"a7.idx{p3_idx}.existing")
                recorder.attach_detect_hooks(detect, detect_idx)
                recorder.maybe_run_val_separability(yolo_obj, cfg)
            except Exception as exc:
                recorder.add_note(f"a7_prepatched_hook_failed:{exc}")
        return yolo_obj

    channels = _infer_p3_channels(detect)
    order = _safe_int(_deep_get(cfg, "enhance241", "a7_order", default=3), 3)
    refine = str(_deep_get(cfg, "enhance241", "a7_refine", default="dw"))
    alpha_init = _safe_float(_deep_get(cfg, "enhance241", "a7_alpha_init", default=0.05), 0.05)
    alpha_auto_fallback = False
    if abs(alpha_init) < 1e-8:
        # Zero alpha + zero-init tail makes the branch effectively frozen at startup.
        alpha_init = 0.05
        alpha_auto_fallback = True
    alpha_cap = _safe_float(_deep_get(cfg, "enhance241", "a7_alpha_cap", default=0.5), 0.5)

    wrapped = C3HBHorNetSafe(
        old,
        channels=channels,
        order=order,
        refine=refine,
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
        "new_type": "C3HBHorNetSafe",
        "channels": int(channels),
        "order": int(order),
        "refine": str(refine),
        "alpha_init": _to_float(alpha_init),
        "alpha_auto_fallback": bool(alpha_auto_fallback),
        "alpha_cap": _to_float(alpha_cap),
        "params_old": int(old_params),
        "params_new": int(new_params),
        "delta_params": int(new_params - old_params),
        "gate1_thresholds": {"cosine_min": 0.98, "var_ratio_min": 0.5},
        "gate2_thresholds": {"delta_l2_min": 0.0, "nan_forbidden": True},
    }
    setattr(yolo_obj, "_enhance241_a7_info", info)

    recorder = get_check_recorder("a7", cfg, patch_info=info)
    if recorder is not None:
        _bind_module_debug(wrapped, recorder, prefix=f"a7.idx{p3_idx}")
        recorder.register_module_params(wrapped, f"a7.idx{p3_idx}")
        try:
            recorder.attach_detect_hooks(detect, detect_idx)
        except Exception as exc:
            recorder.add_note(f"a7_detect_hook_failed:{exc}")
        try:
            recorder.maybe_run_val_separability(yolo_obj, cfg)
        except Exception as exc:
            recorder.add_note(f"a7_val_check_failed:{exc}")

    return yolo_obj
