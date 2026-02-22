from __future__ import annotations

# Path: third_party/yolo11/enhance241/yolo11_241b9.py
# Purpose: enhance241 b9 patch (Light-PDD improved_CSP at P4->P3 fusion, residual-safe).

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from .yolo11_241a3 import (
    _bind_module_debug,
    _f_as_list,
    _get_module_recorder,
    _infer_device_dtype,
    _locate_detect,
    _should_capture_delta,
    _tensor_stats,
    _to_float,
    get_check_recorder,
)

ENHANCE241_AUDIT_KEYS = ["enhance241_b9"]  # enhance241-audit


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


def _concat_candidates(seq: Any) -> List[str]:
    out: List[str] = []
    for i, layer in enumerate(seq):
        if layer.__class__.__name__ == "Concat":
            out.append(f"idx={i}, f={getattr(layer, 'f', None)}")
    return out


def _locate_p4_to_p3_fuse(seq: Any) -> int:
    _, detect = _locate_detect(seq)
    detect_f = _f_as_list(getattr(detect, "f", []))
    if detect_f:
        p3_out = int(detect_f[0])
        p3_fuse = p3_out - 1
        if 0 <= p3_fuse < len(seq):
            layer = seq[p3_fuse]
            if layer.__class__.__name__ == "Concat":
                return p3_fuse
    candidates = []
    for i, layer in enumerate(seq):
        f = _f_as_list(getattr(layer, "f", []))
        if layer.__class__.__name__ == "Concat" and len(f) == 2 and f[0] == -1:
            candidates.append(i)
    if candidates:
        return candidates[-1]
    raise RuntimeError(f"Unable to locate P4->P3 fusion Concat. Concat candidates: {_concat_candidates(seq)}")


def _infer_concat_channels(seq: Any, fuse_idx: int) -> int:
    next_layer = seq[fuse_idx + 1] if (fuse_idx + 1) < len(seq) else None
    c_in = None
    try:
        c_in = int(next_layer.cv1.conv.in_channels)  # type: ignore[attr-defined]
    except Exception:
        c_in = None
    if not c_in and next_layer is not None:
        try:
            for m in next_layer.modules():
                if isinstance(m, torch.nn.Conv2d):
                    c_in = int(m.in_channels)
                    break
        except Exception:
            c_in = None
    if not c_in or c_in % 2 != 0:
        raise RuntimeError(f"Unable to infer concat channels from next layer idx={fuse_idx + 1}; c_in={c_in}")
    return c_in // 2


class _BottleneckLite(torch.nn.Module):
    def __init__(self, channels: int, shortcut: bool = True) -> None:
        super().__init__()
        c = int(channels)
        self.cv1 = torch.nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0, bias=True)
        self.cv2 = torch.nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1, groups=c, bias=True)
        self.cv3 = torch.nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0, bias=True)
        self.shortcut = bool(shortcut)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.silu(self.cv1(x))
        y = F.silu(self.cv2(y))
        y = self.cv3(y)
        if self.shortcut:
            y = y + x
        return F.silu(y)


class ImprovedCSP(torch.nn.Module):
    """improved_CSP style refinement with split -> stack -> concat -> projection."""

    def __init__(self, in_ch: int, out_ch: int, n: int = 2, shortcut: bool = True) -> None:
        super().__init__()
        c_in = int(in_ch)
        c_out = int(out_ch)
        hidden = max(16, c_in // 2)
        self.cv1 = torch.nn.Conv2d(c_in, hidden, kernel_size=1, stride=1, padding=0, bias=True)
        self.cv2 = torch.nn.Conv2d(c_in, hidden, kernel_size=1, stride=1, padding=0, bias=True)
        self.blocks = torch.nn.Sequential(*[_BottleneckLite(hidden, shortcut=shortcut) for _ in range(max(1, int(n)))])
        self.cv3 = torch.nn.Conv2d(hidden * 2, c_out, kernel_size=1, stride=1, padding=0, bias=True)
        torch.nn.init.zeros_(self.cv3.weight)
        if self.cv3.bias is not None:
            torch.nn.init.zeros_(self.cv3.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.blocks(self.cv1(x))
        x2 = self.cv2(x)
        return self.cv3(torch.cat([x1, x2], dim=1))


class B9ImprovedCSPFuseSafe(torch.nn.Module):
    """Residual-safe fusion refinement at concat node."""

    def __init__(
        self,
        channels_per_branch: int,
        n: int = 2,
        shortcut: bool = True,
        upsample_mode: str = "nearest",
        alpha_init: float = 0.0,
        alpha_cap: float = 0.5,
    ) -> None:
        super().__init__()
        self.c = int(channels_per_branch)
        self.upsample_mode = str(upsample_mode).lower()
        self.enhance241_b9_csp = ImprovedCSP(2 * self.c, 2 * self.c, n=int(n), shortcut=bool(shortcut))
        self.alpha_cap = float(max(1e-6, abs(alpha_cap)))
        alpha_init = float(max(-self.alpha_cap * 0.95, min(self.alpha_cap * 0.95, alpha_init)))
        alpha_raw = torch.atanh(torch.tensor(alpha_init / self.alpha_cap, dtype=torch.float32))
        self.enhance241_b9_alpha = torch.nn.Parameter(alpha_raw)

    def _split_input(self, x: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(x, (list, tuple)) and len(x) == 2:
            return x[0], x[1]
        if isinstance(x, torch.Tensor):
            if x.shape[1] != self.c * 2:
                raise ValueError(f"Expected concat tensor with 2C channels, got {x.shape}")
            return torch.split(x, self.c, dim=1)
        raise TypeError(f"B9ImprovedCSPFuseSafe expects [hi,lo] or concat tensor, got {type(x)}")

    def _maybe_align(self, hi: torch.Tensor, lo: torch.Tensor) -> torch.Tensor:
        if hi.shape[-2:] == lo.shape[-2:]:
            return hi
        if self.upsample_mode == "bilinear":
            return F.interpolate(hi, size=lo.shape[-2:], mode="bilinear", align_corners=False)
        return F.interpolate(hi, size=lo.shape[-2:], mode="nearest")

    def forward(self, x: Any) -> torch.Tensor:
        recorder = _get_module_recorder(self)
        prefix = str(getattr(self, "_enhance241_prefix", "b9"))
        step = int(getattr(self, "_enhance241_fwd_step", 0))

        hi, lo = self._split_input(x)
        hi = self._maybe_align(hi, lo)
        base = torch.cat([hi, lo], dim=1)

        if recorder is not None and step == 0:
            recorder.capture_param_before(self, prefix)

        delta_raw = self.enhance241_b9_csp(base)
        alpha_raw = self.enhance241_b9_alpha.to(dtype=base.dtype, device=base.device)
        alpha = torch.tanh(alpha_raw) * self.alpha_cap
        delta = alpha * delta_raw
        out = base + delta

        if recorder is not None:
            if step == 0:
                recorder.record_module_compare(f"{prefix}.fuse", patched=out, baseline=base, input_tensor=lo)
                v_base = _safe_float(base.detach().float().var(unbiased=False).item(), 0.0)
                v_out = _safe_float(out.detach().float().var(unbiased=False).item(), 0.0)
                cos = _safe_float(
                    F.cosine_similarity(
                        out.detach().float().reshape(out.shape[0], -1),
                        base.detach().float().reshape(base.shape[0], -1),
                        dim=1,
                    ).mean().item(),
                    0.0,
                )
                recorder.record_a1_payload(
                    f"{prefix}.gate1",
                    {
                        "cosine_out_vs_base": cos,
                        "var_ratio_out_over_base": v_out / (v_base + 1e-12),
                        "thresholds": {"cosine_min": 0.98, "var_ratio_min": 0.5},
                        "alpha_raw": _to_float(alpha_raw.item()),
                        "alpha": _to_float(alpha.item()),
                        "base_stats": _tensor_stats(base),
                        "delta_stats": _tensor_stats(delta),
                        "out_stats": _tensor_stats(out),
                    },
                )
                recorder.record_scalar_curve(f"{prefix}.alpha_raw", _to_float(alpha_raw.item()), step, max_steps=100)
                recorder.record_scalar_curve(f"{prefix}.alpha", _to_float(alpha.item()), step, max_steps=100)
                if out.requires_grad:
                    try:
                        out.register_hook(lambda g, n=f"{prefix}.fuse": recorder.record_output_grad(n, g))
                    except Exception:
                        pass
            elif step <= 100:
                recorder.record_scalar_curve(f"{prefix}.alpha_raw", _to_float(alpha_raw.item()), step, max_steps=100)
                recorder.record_scalar_curve(f"{prefix}.alpha", _to_float(alpha.item()), step, max_steps=100)
            if _should_capture_delta(step):
                recorder.capture_param_delta(self, prefix)

        setattr(self, "_enhance241_fwd_step", step + 1)
        return out


def apply(model: Any, cfg: Any) -> Any:
    enable_b9 = bool(_deep_get(cfg, "enhance241", "b9", default=False))
    if not enable_b9:
        return model

    if any(bool(_deep_get(cfg, "enhance241", key, default=False)) for key in ("b1", "b2", "b3", "b5", "b7")):
        raise RuntimeError("enhance241.b9 conflicts with b1/b2/b3/b5/b7; enable only one B-class module.")

    yolo_obj = model
    if hasattr(model, "model") and hasattr(getattr(model, "model"), "model"):
        det_model = getattr(model, "model")
        seq = getattr(det_model, "model", None)
    else:
        det_model = model
        seq = getattr(det_model, "model", None)
    if seq is None:
        raise RuntimeError("enhance241.b9 requires YOLO/DetectionModel-like object with .model sequence.")

    prepatched = [i for i, layer in enumerate(seq) if isinstance(layer, B9ImprovedCSPFuseSafe)]
    if prepatched:
        detect_idx, detect = _locate_detect(seq)
        info = {
            "enabled": True,
            "patched_count": 0,
            "existing_count": len(prepatched),
            "patched_indices": [int(x) for x in prepatched],
            "note": "already_patched",
        }
        setattr(yolo_obj, "_enhance241_b9_info", info)
        recorder = get_check_recorder("b9", cfg, patch_info=info)
        if recorder is not None:
            try:
                for idx in prepatched:
                    layer = seq[idx]
                    _bind_module_debug(layer, recorder, prefix=f"b9.idx{idx}.existing")
                    recorder.register_module_params(layer, f"b9.idx{idx}.existing")
                recorder.attach_detect_hooks(detect, detect_idx)
                recorder.maybe_run_val_separability(yolo_obj, cfg)
            except Exception as exc:
                recorder.add_note(f"b9_prepatched_hook_failed:{exc}")
        return yolo_obj

    p3_concat_idx = _locate_p4_to_p3_fuse(seq)
    old = seq[p3_concat_idx]
    detect_idx, detect = _locate_detect(seq)

    if isinstance(old, B9ImprovedCSPFuseSafe):
        info = {
            "enabled": True,
            "patched_count": 0,
            "existing_count": 1,
            "patched_indices": [int(p3_concat_idx)],
            "note": "already_patched",
        }
        setattr(yolo_obj, "_enhance241_b9_info", info)
        recorder = get_check_recorder("b9", cfg, patch_info=info)
        if recorder is not None:
            try:
                _bind_module_debug(old, recorder, prefix=f"b9.idx{p3_concat_idx}.existing")
                recorder.register_module_params(old, f"b9.idx{p3_concat_idx}.existing")
                recorder.attach_detect_hooks(detect, detect_idx)
                recorder.maybe_run_val_separability(yolo_obj, cfg)
            except Exception as exc:
                recorder.add_note(f"b9_prepatched_hook_failed:{exc}")
        return yolo_obj

    n = _safe_int(_deep_get(cfg, "enhance241", "b9_depth", default=2), 2)
    shortcut = bool(_deep_get(cfg, "enhance241", "b9_shortcut", default=True))
    upsample = str(_deep_get(cfg, "enhance241", "b9_upsample", default="nearest")).lower()
    alpha_init = _safe_float(_deep_get(cfg, "enhance241", "b9_alpha_init", default=0.0), 0.0)
    alpha_cap = _safe_float(_deep_get(cfg, "enhance241", "b9_alpha_cap", default=0.5), 0.5)
    alpha_auto_fallback = False
    if abs(alpha_init) < 1e-8:
        alpha_init = 0.05
        alpha_auto_fallback = True

    c = _infer_concat_channels(seq, p3_concat_idx)
    mod = B9ImprovedCSPFuseSafe(
        channels_per_branch=int(c),
        n=int(n),
        shortcut=bool(shortcut),
        upsample_mode=str(upsample),
        alpha_init=float(alpha_init),
        alpha_cap=float(alpha_cap),
    )
    for attr in ("i", "f", "type"):
        if hasattr(old, attr):
            setattr(mod, attr, getattr(old, attr))
    device, dtype = _infer_device_dtype(seq, p3_concat_idx)
    if device is not None:
        if dtype is not None:
            mod = mod.to(device=device, dtype=dtype)
        else:
            mod = mod.to(device=device)
    seq[p3_concat_idx] = mod

    old_params = sum(int(p.numel()) for p in old.parameters())
    new_params = sum(int(p.numel()) for p in mod.parameters())
    info = {
        "enabled": True,
        "patched_count": 1,
        "existing_count": 0,
        "patched_indices": [int(p3_concat_idx)],
        "concat_candidates": _concat_candidates(seq),
        "base_type": old.__class__.__name__,
        "new_type": "B9ImprovedCSPFuseSafe",
        "depth": int(n),
        "shortcut": bool(shortcut),
        "upsample": str(upsample),
        "alpha_init": _to_float(alpha_init),
        "alpha_auto_fallback": bool(alpha_auto_fallback),
        "alpha_cap": _to_float(alpha_cap),
        "params_old": int(old_params),
        "params_new": int(new_params),
        "delta_params": int(new_params - old_params),
    }
    setattr(yolo_obj, "_enhance241_b9_info", info)

    recorder = get_check_recorder("b9", cfg, patch_info=info)
    if recorder is not None:
        try:
            _bind_module_debug(mod, recorder, prefix=f"b9.idx{p3_concat_idx}")
            recorder.register_module_params(mod, f"b9.idx{p3_concat_idx}")
            recorder.attach_detect_hooks(detect, detect_idx)
            recorder.maybe_run_val_separability(yolo_obj, cfg)
        except Exception as exc:
            recorder.add_note(f"b9_hook_failed:{exc}")

    return yolo_obj
