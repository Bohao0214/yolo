from __future__ import annotations

# Path: third_party/yolo11/enhance241/yolo11_241a4.py
# Purpose: enhance241 a4 patch (fuse a3 + a7 dual-delta on P3 downsample).

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from .yolo11_241a3 import (
    _Conv3x3,
    _DWSeparableConv,
    _SpaceToDepth,
    _bind_module_debug,
    _find_first_conv,
    _find_stride_conv,
    _get_module_recorder,
    _infer_device_dtype,
    _locate_detect,
    _locate_p4_to_p3_fuse,
    _module_stride,
    _should_capture_delta,
    _tensor_stats,
    _to_float,
    get_check_recorder,
)
from .yolo11_241a7 import HorNetC3HBDelta

ENHANCE241_AUDIT_KEYS = ["enhance241_a4"]  # enhance241-audit


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


class A4DualDeltaSafe(torch.nn.Module):
    """Dual-delta residual injection:
    y = y_base + alpha1 * delta_a3(x) + alpha2 * delta_a7(y_base)
    """

    def __init__(
        self,
        base_downsample: torch.nn.Module,
        in_ch: int,
        out_ch: int,
        a3_pre_div: int = 4,
        a4_refine: str = "dw",
        a4_order: int = 3,
        a4_alpha1_init: float = 0.05,
        a4_alpha1_cap: float = 0.5,
        a4_alpha2_init: float = 0.0,
        a4_alpha2_cap: float = 0.5,
    ) -> None:
        super().__init__()
        self.enhance241_a4_base = base_downsample
        self.in_ch = int(in_ch)
        self.out_ch = int(out_ch)
        self.a4_refine = str(a4_refine).lower()
        self.a4_order = max(2, int(a4_order))
        self.a4_pre_div = max(1, int(a3_pre_div))

        if self.out_ch > 0 and self.out_ch % self.a4_pre_div == 0:
            pre_ch = self.out_ch // self.a4_pre_div
        else:
            pre_ch = self.out_ch

        if self.a4_refine == "conv":
            self.enhance241_a4_a3_pre = _Conv3x3(self.in_ch, pre_ch)
        else:
            self.enhance241_a4_a3_pre = _DWSeparableConv(self.in_ch, pre_ch)
        self.enhance241_a4_a3_s2d = _SpaceToDepth(2)
        self.enhance241_a4_a3_post = torch.nn.Conv2d(pre_ch * 4, self.out_ch, kernel_size=1, stride=1, padding=0, bias=True)
        torch.nn.init.zeros_(self.enhance241_a4_a3_post.weight)
        if self.enhance241_a4_a3_post.bias is not None:
            torch.nn.init.zeros_(self.enhance241_a4_a3_post.bias)

        self.enhance241_a4_a7_delta = HorNetC3HBDelta(
            channels=int(self.out_ch),
            order=int(self.a4_order),
            refine=str(self.a4_refine),
        )

        self.alpha1_cap = float(max(1e-6, abs(a4_alpha1_cap)))
        self.alpha2_cap = float(max(1e-6, abs(a4_alpha2_cap)))

        a1 = float(max(-self.alpha1_cap * 0.95, min(self.alpha1_cap * 0.95, float(a4_alpha1_init))))
        a2 = float(max(-self.alpha2_cap * 0.95, min(self.alpha2_cap * 0.95, float(a4_alpha2_init))))
        self.enhance241_a4_alpha1 = torch.nn.Parameter(torch.atanh(torch.tensor(a1 / self.alpha1_cap, dtype=torch.float32)))
        self.enhance241_a4_alpha2 = torch.nn.Parameter(torch.atanh(torch.tensor(a2 / self.alpha2_cap, dtype=torch.float32)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        recorder = _get_module_recorder(self)
        prefix = str(getattr(self, "_enhance241_prefix", "a4"))
        step = int(getattr(self, "_enhance241_fwd_step", 0))

        if recorder is not None and step == 0:
            recorder.capture_param_before(self, prefix)

        y_base = self.enhance241_a4_base(x)

        delta_a3 = self.enhance241_a4_a3_pre(x)
        delta_a3 = self.enhance241_a4_a3_s2d(delta_a3)
        delta_a3 = self.enhance241_a4_a3_post(delta_a3)
        if delta_a3.shape[-2:] != y_base.shape[-2:]:
            delta_a3 = F.interpolate(delta_a3, size=y_base.shape[-2:], mode="nearest")

        delta_a7 = self.enhance241_a4_a7_delta(y_base)

        alpha1_raw = self.enhance241_a4_alpha1.to(dtype=y_base.dtype, device=y_base.device)
        alpha2_raw = self.enhance241_a4_alpha2.to(dtype=y_base.dtype, device=y_base.device)
        alpha1 = torch.tanh(alpha1_raw) * self.alpha1_cap
        alpha2 = torch.tanh(alpha2_raw) * self.alpha2_cap
        delta = alpha1 * delta_a3 + alpha2 * delta_a7
        out = y_base + delta

        if recorder is not None:
            if step == 0:
                recorder.record_module_compare(f"{prefix}.p3_down", patched=out, baseline=y_base, input_tensor=x)
                v_base = _safe_float(y_base.detach().float().var(unbiased=False).item(), 0.0)
                v_out = _safe_float(out.detach().float().var(unbiased=False).item(), 0.0)
                v_delta = _safe_float(delta.detach().float().var(unbiased=False).item(), 0.0)
                cos = _safe_float(
                    F.cosine_similarity(
                        out.detach().float().reshape(out.shape[0], -1),
                        y_base.detach().float().reshape(y_base.shape[0], -1),
                        dim=1,
                    ).mean().item(),
                    0.0,
                )
                gate1_ok = bool(cos >= 0.98 and (v_out / (v_base + 1e-12)) >= 0.5 and v_delta >= 0.0)
                recorder.record_a1_payload(
                    f"{prefix}.gate1",
                    {
                        "pass": gate1_ok,
                        "cosine_out_vs_base": cos,
                        "var_ratio_out_over_base": v_out / (v_base + 1e-12),
                        "delta_var": v_delta,
                        "alpha1_raw": _to_float(alpha1_raw.item()),
                        "alpha1": _to_float(alpha1.item()),
                        "alpha1_cap": _to_float(self.alpha1_cap),
                        "alpha2_raw": _to_float(alpha2_raw.item()),
                        "alpha2": _to_float(alpha2.item()),
                        "alpha2_cap": _to_float(self.alpha2_cap),
                        "base_stats": _tensor_stats(y_base),
                        "delta_a3_stats": _tensor_stats(delta_a3),
                        "delta_a7_stats": _tensor_stats(delta_a7),
                        "delta_stats": _tensor_stats(delta),
                        "out_stats": _tensor_stats(out),
                    },
                )
                recorder.record_scalar_curve(f"{prefix}.alpha1", _to_float(alpha1.item()), step, max_steps=100)
                recorder.record_scalar_curve(f"{prefix}.alpha2", _to_float(alpha2.item()), step, max_steps=100)
                if out.requires_grad:
                    try:
                        out.register_hook(lambda g, n=f"{prefix}.p3_down": recorder.record_output_grad(n, g))
                    except Exception:
                        pass
            elif step <= 100:
                recorder.record_scalar_curve(f"{prefix}.alpha1", _to_float(alpha1.item()), step, max_steps=100)
                recorder.record_scalar_curve(f"{prefix}.alpha2", _to_float(alpha2.item()), step, max_steps=100)
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


def apply(model: Any, cfg: Any) -> Any:
    enable_a4 = bool(_deep_get(cfg, "enhance241", "a4", default=False))
    if not enable_a4:
        return model

    if any(bool(_deep_get(cfg, "enhance241", k, default=False)) for k in ("a3", "a5", "a7", "a9", "a11", "a21")):
        raise RuntimeError("enhance241.a4 conflicts with a3/a5/a7/a9/a11/a21; enable only one A-class module.")

    yolo_obj, seq = _extract_model_seq(model)
    if seq is None:
        raise RuntimeError("enhance241.a4 requires YOLO/DetectionModel-like object with .model sequence.")

    prepatched = [i for i, layer in enumerate(seq) if isinstance(layer, A4DualDeltaSafe)]
    if prepatched:
        info = {
            "enabled": True,
            "existing_count": len(prepatched),
            "patched_count": 0,
            "patched_indices": [int(x) for x in prepatched],
            "note": "already_patched",
        }
        setattr(yolo_obj, "_enhance241_a4_info", info)
        recorder = get_check_recorder("a4", cfg, patch_info=info)
        if recorder is not None:
            try:
                idx0 = int(prepatched[0])
                mod0 = seq[idx0]
                _bind_module_debug(mod0, recorder, prefix=f"a4.idx{idx0}.existing")
                recorder.register_module_params(mod0, f"a4.idx{idx0}.existing")
                detect_idx, detect_layer = _locate_detect(seq)
                recorder.attach_detect_hooks(detect_layer, detect_idx)
                recorder.maybe_run_val_separability(yolo_obj, cfg)
            except Exception as exc:
                recorder.add_note(f"a4_prepatched_hook_failed:{exc}")
        return yolo_obj

    _, p3_idx = _locate_p4_to_p3_fuse(seq)
    ds_idx: Optional[int] = None
    for i in range(p3_idx, -1, -1):
        if _module_stride(seq[i]) == 2:
            ds_idx = i
            break
    if ds_idx is None:
        raise RuntimeError("enhance241.a4: unable to locate stride=2 downsample before P3 output.")

    old = seq[ds_idx]
    conv = _find_stride_conv(old, stride=2) or _find_first_conv(old)
    if conv is None:
        raise RuntimeError(f"enhance241.a4: unable to infer in/out channels at idx={ds_idx}.")

    in_ch = int(conv.in_channels)
    out_ch = int(conv.out_channels)
    a4_pre_div = _safe_int(_deep_get(cfg, "enhance241", "a4_pre_div", default=4), 4)
    a4_refine = str(_deep_get(cfg, "enhance241", "a4_refine", default="dw"))
    a4_order = _safe_int(_deep_get(cfg, "enhance241", "a4_order", default=3), 3)
    a4_alpha1_init = _safe_float(_deep_get(cfg, "enhance241", "a4_alpha1_init", default=0.05), 0.05)
    a4_alpha1_cap = _safe_float(_deep_get(cfg, "enhance241", "a4_alpha1_cap", default=0.5), 0.5)
    a4_alpha2_init = _safe_float(_deep_get(cfg, "enhance241", "a4_alpha2_init", default=0.0), 0.0)
    a4_alpha2_cap = _safe_float(_deep_get(cfg, "enhance241", "a4_alpha2_cap", default=0.5), 0.5)

    fuse = A4DualDeltaSafe(
        base_downsample=old,
        in_ch=in_ch,
        out_ch=out_ch,
        a3_pre_div=a4_pre_div,
        a4_refine=a4_refine,
        a4_order=a4_order,
        a4_alpha1_init=a4_alpha1_init,
        a4_alpha1_cap=a4_alpha1_cap,
        a4_alpha2_init=a4_alpha2_init,
        a4_alpha2_cap=a4_alpha2_cap,
    )
    for attr in ("i", "f", "type"):
        if hasattr(old, attr):
            setattr(fuse, attr, getattr(old, attr))
    device, dtype = _infer_device_dtype(seq, ds_idx)
    if device is not None:
        if dtype is not None:
            fuse = fuse.to(device=device, dtype=dtype)
        else:
            fuse = fuse.to(device=device)
    seq[ds_idx] = fuse

    info = {
        "enabled": True,
        "existing_count": 0,
        "patched_count": 1,
        "patched_indices": [int(ds_idx)],
        "p3_index": int(p3_idx),
        "orig_type": old.__class__.__name__,
        "new_type": "A4DualDeltaSafe",
        "a4_refine": str(a4_refine),
        "a4_order": int(a4_order),
        "alpha1_init": _to_float(a4_alpha1_init),
        "alpha1_cap": _to_float(a4_alpha1_cap),
        "alpha2_init": _to_float(a4_alpha2_init),
        "alpha2_cap": _to_float(a4_alpha2_cap),
    }
    setattr(yolo_obj, "_enhance241_a4_info", info)
    print(
        f"[enhance241] a4 enabled: patched model.model[{int(ds_idx)}] "
        f"-> A4DualDeltaSafe(refine={a4_refine}, order={a4_order}, alpha1_init={a4_alpha1_init:.3f}, alpha2_init={a4_alpha2_init:.3f})"
    )

    recorder = get_check_recorder("a4", cfg, patch_info=info)
    if recorder is not None:
        _bind_module_debug(fuse, recorder, prefix=f"a4.idx{int(ds_idx)}")
        recorder.register_module_params(fuse, f"a4.idx{int(ds_idx)}")
        try:
            detect_idx, detect_layer = _locate_detect(seq)
            recorder.attach_detect_hooks(detect_layer, detect_idx)
        except Exception as exc:
            recorder.add_note(f"a4_detect_hook_failed:{exc}")
        try:
            recorder.maybe_run_val_separability(yolo_obj, cfg)
        except Exception as exc:
            recorder.add_note(f"a4_val_check_failed:{exc}")

    return yolo_obj
