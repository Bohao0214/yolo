from __future__ import annotations

# Path: third_party/yolo11/enhance241/yolo11_241c7.py
# Purpose: enhance241 c7 patch (MCBAM residual-safe gate before Detect P3 input).

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

ENHANCE241_AUDIT_KEYS = ["enhance241_c7"]  # enhance241-audit


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


class _ChannelAttention(torch.nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        c = int(channels)
        hidden = max(4, c // max(1, int(reduction)))
        self.fc1 = torch.nn.Conv2d(c, hidden, kernel_size=1, stride=1, padding=0, bias=True)
        self.act = torch.nn.SiLU(inplace=True)
        self.fc2 = torch.nn.Conv2d(hidden, c, kernel_size=1, stride=1, padding=0, bias=True)

    def _shared_mlp(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = F.adaptive_avg_pool2d(x, output_size=1)
        mx = F.adaptive_max_pool2d(x, output_size=1)
        return torch.sigmoid(self._shared_mlp(avg) + self._shared_mlp(mx))


class _SpatialAttentionBySubspace(torch.nn.Module):
    """MCBAM-style spatial attention with n*n subspace max and two avg-pool aggregations.

    Paper note:
    n = (sum_{i=1..lambda} sum_{j=1..m_i} x_{ij}) / (sum_{i=1..lambda} m_i)
    This module consumes precomputed/configured n (offline statistic).
    """

    def __init__(self, n: int = 3) -> None:
        super().__init__()
        self.n = max(1, int(n))
        self.proj = torch.nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=True)

    def _subspace_max_grid(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        n = max(1, min(self.n, h, w))
        ys = torch.linspace(0, h, steps=n + 1, device=x.device, dtype=torch.float32).round().long()
        xs = torch.linspace(0, w, steps=n + 1, device=x.device, dtype=torch.float32).round().long()

        rows = []
        for iy in range(n):
            y0 = int(ys[iy].item())
            y1 = int(ys[iy + 1].item())
            if y1 <= y0:
                y1 = min(h, y0 + 1)
            cols = []
            for ix in range(n):
                x0 = int(xs[ix].item())
                x1 = int(xs[ix + 1].item())
                if x1 <= x0:
                    x1 = min(w, x0 + 1)
                cell = x[:, :, y0:y1, x0:x1]
                # Subspace max response.
                val = cell.amax(dim=(1, 2, 3), keepdim=True)
                cols.append(val)
            rows.append(torch.cat(cols, dim=-1))
        return torch.cat(rows, dim=-2)  # [B, 1, n, n]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        grid = self._subspace_max_grid(x)
        agg = F.avg_pool2d(grid, kernel_size=3, stride=1, padding=1)
        agg = F.avg_pool2d(agg, kernel_size=3, stride=1, padding=1)
        up = F.interpolate(agg, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return torch.sigmoid(self.proj(up))


class MCBAMCore(torch.nn.Module):
    def __init__(self, channels: int, n: int = 3, reduction: int = 16, mode: str = "full") -> None:
        super().__init__()
        self.mode = str(mode).lower()
        self.channel = _ChannelAttention(channels=int(channels), reduction=int(reduction))
        self.spatial = _SpatialAttentionBySubspace(n=int(n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mode = self.mode
        attn_c: Optional[torch.Tensor] = None
        attn_s: Optional[torch.Tensor] = None
        if mode in ("full", "channel"):
            attn_c = self.channel(x)
        if mode in ("full", "spatial"):
            attn_s = self.spatial(x)

        if attn_c is None and attn_s is None:
            return x
        if attn_c is None:
            attn = attn_s
        elif attn_s is None:
            attn = attn_c
        else:
            attn = attn_c * attn_s
        return x * attn


class C7MCBAMInject(torch.nn.Module):
    """Residual-safe c7 wrapper: out = base(x) + alpha * (MCBAM(base(x)) - base(x))."""

    def __init__(
        self,
        base_module: torch.nn.Module,
        channels: int,
        n: int = 3,
        reduction: int = 16,
        mode: str = "full",
        alpha_init: float = 0.0,
        alpha_cap: float = 0.5,
    ) -> None:
        super().__init__()
        self.enhance241_c7_base = base_module
        self.enhance241_c7_mcbam = MCBAMCore(channels=int(channels), n=int(n), reduction=int(reduction), mode=str(mode))
        self.alpha_cap = float(max(1e-6, abs(alpha_cap)))

        alpha_init = float(max(-self.alpha_cap * 0.95, min(self.alpha_cap * 0.95, alpha_init)))
        alpha_ratio = alpha_init / self.alpha_cap
        alpha_raw = torch.atanh(torch.tensor(alpha_ratio, dtype=torch.float32))
        self.enhance241_c7_alpha = torch.nn.Parameter(alpha_raw)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        recorder = _get_module_recorder(self)
        prefix = str(getattr(self, "_enhance241_prefix", "c7"))
        step = int(getattr(self, "_enhance241_fwd_step", 0))

        if recorder is not None and step == 0:
            recorder.capture_param_before(self, prefix)

        y_base = self.enhance241_c7_base(x)
        y_gate = self.enhance241_c7_mcbam(y_base)
        delta = y_gate - y_base
        alpha_raw = self.enhance241_c7_alpha.to(dtype=y_base.dtype, device=y_base.device)
        alpha = torch.tanh(alpha_raw) * self.alpha_cap
        out = y_base + alpha * delta

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
    raise RuntimeError("Unable to infer P3 channels for c7.")


def apply(model: Any, cfg: Any) -> Any:
    enable_c7 = bool(_deep_get(cfg, "enhance241", "c7", default=False))
    if not enable_c7:
        return model

    if any(bool(_deep_get(cfg, "enhance241", k, default=False)) for k in ("c5", "c9")):
        raise RuntimeError("enhance241.c7 conflicts with c5/c9; enable only one C-class module.")

    yolo_obj, seq = _extract_model_seq(model)
    if seq is None:
        raise RuntimeError("enhance241.c7 requires YOLO/DetectionModel-like object with .model sequence.")

    p3_idx = _infer_p3_output_index(seq)
    detect_idx, detect = _locate_detect(seq)
    old = seq[p3_idx]

    if isinstance(old, C7MCBAMInject):
        info = {
            "enabled": True,
            "existing_count": 1,
            "patched_count": 0,
            "p3_index": int(p3_idx),
            "detect_idx": int(detect_idx),
            "note": "already_patched",
        }
        setattr(yolo_obj, "_enhance241_c7_info", info)
        recorder = get_check_recorder("c7", cfg, patch_info=info)
        if recorder is not None:
            try:
                _bind_module_debug(old, recorder, prefix=f"c7.idx{p3_idx}.existing")
                recorder.register_module_params(old, f"c7.idx{p3_idx}.existing")
                recorder.attach_detect_hooks(detect, detect_idx)
                recorder.maybe_run_val_separability(yolo_obj, cfg)
            except Exception as exc:
                recorder.add_note(f"c7_prepatched_hook_failed:{exc}")
        return yolo_obj

    channels = _infer_p3_channels(detect)
    n = _safe_int(_deep_get(cfg, "enhance241", "c7_n", default=3), 3)
    n = max(1, min(6, int(n)))
    mode = str(_deep_get(cfg, "enhance241", "c7_mode", default="full")).lower()
    reduction = _safe_int(_deep_get(cfg, "enhance241", "c7_reduction", default=16), 16)
    alpha_init = _safe_float(_deep_get(cfg, "enhance241", "c7_alpha_init", default=0.0), 0.0)
    alpha_cap = _safe_float(_deep_get(cfg, "enhance241", "c7_alpha_cap", default=0.5), 0.5)

    wrapped = C7MCBAMInject(
        old,
        channels=channels,
        n=n,
        reduction=reduction,
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
        "new_type": "C7MCBAMInject",
        "channels": int(channels),
        "n": int(n),
        "mode": str(mode),
        "reduction": int(reduction),
        "alpha_init": _to_float(alpha_init),
        "alpha_cap": _to_float(alpha_cap),
        "params_old": int(old_params),
        "params_new": int(new_params),
        "delta_params": int(new_params - old_params),
        "gate1_thresholds": {"cosine_min": 0.98, "var_ratio_min": 0.5},
        "gate2_thresholds": {"delta_l2_min": 0.0, "nan_forbidden": True},
    }
    setattr(yolo_obj, "_enhance241_c7_info", info)

    recorder = get_check_recorder("c7", cfg, patch_info=info)
    if recorder is not None:
        _bind_module_debug(wrapped, recorder, prefix=f"c7.idx{p3_idx}")
        recorder.register_module_params(wrapped, f"c7.idx{p3_idx}")
        try:
            recorder.attach_detect_hooks(detect, detect_idx)
        except Exception as exc:
            recorder.add_note(f"c7_detect_hook_failed:{exc}")
        try:
            recorder.maybe_run_val_separability(yolo_obj, cfg)
        except Exception as exc:
            recorder.add_note(f"c7_val_check_failed:{exc}")

    return yolo_obj
