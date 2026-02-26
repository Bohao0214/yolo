from __future__ import annotations

# Path: third_party/yolo11/enhance241/yolo11_241c4.py
# Purpose: enhance241 c4 patch (fuse c5 + c11 as sequential residual-safe head-input guardrail).

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
from .yolo11_241c11 import _Conv1x1Gate, _SEGate
from .yolo11_241c5 import BRACore

ENHANCE241_AUDIT_KEYS = ["enhance241_c4"]  # enhance241-audit


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
    raise RuntimeError("Unable to infer P3 channels for c4.")


class C4C5C11Inject(torch.nn.Module):
    """Sequential residual-safe fusion:
    z = y_base + alpha5 * delta_c5(y_base)
    y = z + alpha11 * delta_c11(z)
    """

    def __init__(
        self,
        base_module: torch.nn.Module,
        channels: int,
        c5_topk: int = 4,
        c5_window_size: int = 8,
        c5_num_heads: int = 4,
        c5_kv_downsample_mode: str = "avg",
        c5_soft_routing: bool = True,
        c4_gate_mode: str = "se",
        c4_gate_reduction: int = 16,
        alpha5_init: float = 0.05,
        alpha5_cap: float = 0.5,
        alpha11_init: float = 0.01,
        alpha11_cap: float = 0.5,
    ) -> None:
        super().__init__()
        self.enhance241_c4_base = base_module
        self.enhance241_c4_c5 = BRACore(
            channels=int(channels),
            num_heads=int(c5_num_heads),
            window_size=int(c5_window_size),
            topk=int(c5_topk),
            kv_downsample_mode=str(c5_kv_downsample_mode),
            soft_routing=bool(c5_soft_routing),
        )
        gate_mode = str(c4_gate_mode).lower()
        if gate_mode in {"1x1", "1x1_gate", "conv1x1"}:
            self.enhance241_c4_c11_gate = _Conv1x1Gate(int(channels))
            self.gate_mode = "1x1_gate"
        else:
            # Keep same structure as c11 SE gate, reduction is accepted for forward compatibility.
            _ = int(c4_gate_reduction)
            self.enhance241_c4_c11_gate = _SEGate(int(channels), reduction=max(1, int(c4_gate_reduction)))
            self.gate_mode = "se"

        self.alpha5_cap = float(max(1e-6, abs(alpha5_cap)))
        self.alpha11_cap = float(max(1e-6, abs(alpha11_cap)))
        a5 = float(max(-self.alpha5_cap * 0.95, min(self.alpha5_cap * 0.95, float(alpha5_init))))
        a11 = float(max(-self.alpha11_cap * 0.95, min(self.alpha11_cap * 0.95, float(alpha11_init))))
        self.enhance241_c4_alpha5 = torch.nn.Parameter(torch.atanh(torch.tensor(a5 / self.alpha5_cap, dtype=torch.float32)))
        self.enhance241_c4_alpha11 = torch.nn.Parameter(
            torch.atanh(torch.tensor(a11 / self.alpha11_cap, dtype=torch.float32))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        recorder = _get_module_recorder(self)
        prefix = str(getattr(self, "_enhance241_prefix", "c4"))
        step = int(getattr(self, "_enhance241_fwd_step", 0))

        if recorder is not None and step == 0:
            recorder.capture_param_before(self, prefix)

        y_base = self.enhance241_c4_base(x)
        delta5 = self.enhance241_c4_c5(y_base)
        alpha5_raw = self.enhance241_c4_alpha5.to(dtype=y_base.dtype, device=y_base.device)
        alpha5 = torch.tanh(alpha5_raw) * self.alpha5_cap
        z = y_base + alpha5 * delta5

        gate = self.enhance241_c4_c11_gate(z)
        delta11 = gate * z
        alpha11_raw = self.enhance241_c4_alpha11.to(dtype=z.dtype, device=z.device)
        alpha11 = torch.tanh(alpha11_raw) * self.alpha11_cap
        out = z + alpha11 * delta11

        if recorder is not None:
            if step == 0:
                recorder.record_module_compare(f"{prefix}.p3", patched=out, baseline=y_base, input_tensor=_to_input_tensor(x))
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
                recorder.record_a1_payload(
                    f"{prefix}.gate1",
                    {
                        "pass": bool(cos >= 0.98 and (v_out / (v_base + 1e-12)) >= 0.5),
                        "cosine_out_vs_base": cos,
                        "var_ratio_out_over_base": v_out / (v_base + 1e-12),
                        "alpha5": _to_float(alpha5.item()),
                        "alpha11": _to_float(alpha11.item()),
                        "gate_mode": self.gate_mode,
                        "base_stats": _tensor_stats(y_base),
                        "delta5_stats": _tensor_stats(delta5),
                        "delta11_stats": _tensor_stats(delta11),
                        "out_stats": _tensor_stats(out),
                    },
                )
                recorder.record_scalar_curve(f"{prefix}.alpha5", _to_float(alpha5.item()), step, max_steps=100)
                recorder.record_scalar_curve(f"{prefix}.alpha11", _to_float(alpha11.item()), step, max_steps=100)
                if out.requires_grad:
                    try:
                        out.register_hook(lambda g, n=f"{prefix}.p3": recorder.record_output_grad(n, g))
                    except Exception:
                        pass
            elif step <= 100:
                recorder.record_scalar_curve(f"{prefix}.alpha5", _to_float(alpha5.item()), step, max_steps=100)
                recorder.record_scalar_curve(f"{prefix}.alpha11", _to_float(alpha11.item()), step, max_steps=100)
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
    enable_c4 = bool(_deep_get(cfg, "enhance241", "c4", default=False))
    if not enable_c4:
        return model

    if any(bool(_deep_get(cfg, "enhance241", k, default=False)) for k in ("c5", "c7", "c9", "c11", "c21")):
        raise RuntimeError("enhance241.c4 conflicts with c5/c7/c9/c11/c21; enable only one C-class module.")

    yolo_obj, seq = _extract_model_seq(model)
    if seq is None:
        raise RuntimeError("enhance241.c4 requires YOLO/DetectionModel-like object with .model sequence.")

    p3_idx = _infer_p3_output_index(seq)
    detect_idx, detect = _locate_detect(seq)
    old = seq[p3_idx]

    if isinstance(old, C4C5C11Inject):
        info = {
            "enabled": True,
            "existing_count": 1,
            "patched_count": 0,
            "p3_index": int(p3_idx),
            "detect_idx": int(detect_idx),
            "note": "already_patched",
        }
        setattr(yolo_obj, "_enhance241_c4_info", info)
        recorder = get_check_recorder("c4", cfg, patch_info=info)
        if recorder is not None:
            try:
                _bind_module_debug(old, recorder, prefix=f"c4.idx{p3_idx}.existing")
                recorder.register_module_params(old, f"c4.idx{p3_idx}.existing")
                recorder.attach_detect_hooks(detect, detect_idx)
                recorder.maybe_run_val_separability(yolo_obj, cfg)
            except Exception as exc:
                recorder.add_note(f"c4_prepatched_hook_failed:{exc}")
        return yolo_obj

    channels = _infer_p3_channels(detect)
    c4_topk = _safe_int(_deep_get(cfg, "enhance241", "c4_topk", default=4), 4)
    c4_window = _safe_int(_deep_get(cfg, "enhance241", "c4_window_size", default=8), 8)
    c4_heads = _safe_int(_deep_get(cfg, "enhance241", "c4_num_heads", default=4), 4)
    c4_kv = str(_deep_get(cfg, "enhance241", "c4_kv_downsample_mode", default="avg"))
    c4_soft = bool(_deep_get(cfg, "enhance241", "c4_soft_routing", default=True))
    c4_gate_mode = str(_deep_get(cfg, "enhance241", "c4_gate_mode", default="se"))
    c4_gate_reduction = _safe_int(_deep_get(cfg, "enhance241", "c4_gate_reduction", default=16), 16)
    c4_alpha5_init = _safe_float(_deep_get(cfg, "enhance241", "c4_alpha5_init", default=0.05), 0.05)
    c4_alpha5_cap = _safe_float(_deep_get(cfg, "enhance241", "c4_alpha5_cap", default=0.5), 0.5)
    c4_alpha11_init = _safe_float(_deep_get(cfg, "enhance241", "c4_alpha11_init", default=0.01), 0.01)
    c4_alpha11_cap = _safe_float(_deep_get(cfg, "enhance241", "c4_alpha11_cap", default=0.5), 0.5)

    wrapped = C4C5C11Inject(
        base_module=old,
        channels=channels,
        c5_topk=c4_topk,
        c5_window_size=c4_window,
        c5_num_heads=c4_heads,
        c5_kv_downsample_mode=c4_kv,
        c5_soft_routing=c4_soft,
        c4_gate_mode=c4_gate_mode,
        c4_gate_reduction=c4_gate_reduction,
        alpha5_init=c4_alpha5_init,
        alpha5_cap=c4_alpha5_cap,
        alpha11_init=c4_alpha11_init,
        alpha11_cap=c4_alpha11_cap,
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

    info = {
        "enabled": True,
        "existing_count": 0,
        "patched_count": 1,
        "p3_index": int(p3_idx),
        "detect_idx": int(detect_idx),
        "new_type": "C4C5C11Inject",
        "c4_topk": int(c4_topk),
        "c4_window_size": int(c4_window),
        "c4_num_heads": int(c4_heads),
        "c4_kv_downsample_mode": str(c4_kv),
        "c4_soft_routing": bool(c4_soft),
        "c4_gate_mode": str(c4_gate_mode),
        "c4_gate_reduction": int(c4_gate_reduction),
        "alpha5_init": _to_float(c4_alpha5_init),
        "alpha11_init": _to_float(c4_alpha11_init),
    }
    setattr(yolo_obj, "_enhance241_c4_info", info)
    print(
        f"[enhance241] c4 enabled: patched model.model[{int(p3_idx)}] "
        f"-> C4C5C11Inject(topk={c4_topk}, window={c4_window}, gate={c4_gate_mode})"
    )

    recorder = get_check_recorder("c4", cfg, patch_info=info)
    if recorder is not None:
        _bind_module_debug(wrapped, recorder, prefix=f"c4.idx{p3_idx}")
        recorder.register_module_params(wrapped, f"c4.idx{p3_idx}")
        try:
            recorder.attach_detect_hooks(detect, detect_idx)
            recorder.maybe_run_val_separability(yolo_obj, cfg)
        except Exception as exc:
            recorder.add_note(f"c4_hook_failed:{exc}")

    return yolo_obj
