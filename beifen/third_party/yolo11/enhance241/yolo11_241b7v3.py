from __future__ import annotations

# Path: third_party/yolo11/enhance241/yolo11_241b7.py
# Purpose: enhance241 b7 patch (CARAFE residual-safe upsample for neck top-down links).

from typing import Any, Dict, List, Optional

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

ENHANCE241_AUDIT_KEYS = ["enhance241_b7"]  # enhance241-audit


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


def _locate_p5_to_p4_fuse(seq: Any, p3_concat_idx: int) -> int:
    candidates = []
    for i, layer in enumerate(seq):
        f = _f_as_list(getattr(layer, "f", []))
        if layer.__class__.__name__ == "Concat" and i < p3_concat_idx and len(f) == 2 and f[0] == -1:
            candidates.append(i)
    if candidates:
        return candidates[-1]
    raise RuntimeError(
        f"Unable to locate P5->P4 Concat before idx={p3_concat_idx}. Concat candidates: {_concat_candidates(seq)}"
    )


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


def _infer_scale_factor(module: Any) -> int:
    s = getattr(module, "scale_factor", 2)
    if isinstance(s, (list, tuple)):
        s = s[0] if s else 2
    try:
        return max(2, int(round(float(s))))
    except Exception:
        return 2


class CARAFECore(torch.nn.Module):
    """CARAFE two-stage upsampler: kernel prediction + feature reassembly."""

    def __init__(
        self,
        channels: int,
        scale: int = 2,
        kernel_size: int = 5,
        compress: int = 64,
        chunk_channels: int = 64,
    ) -> None:
        super().__init__()
        c = int(channels)
        s = max(2, int(scale))
        k = max(3, int(kernel_size))
        if k % 2 == 0:
            k += 1
        cm = max(8, min(int(compress), c))

        self.channels = c
        self.scale = s
        self.kernel_size = k
        self.compress = cm
        self.chunk_channels = max(8, int(chunk_channels))

        self.comp = torch.nn.Conv2d(c, cm, kernel_size=1, stride=1, padding=0, bias=True)
        self.encoder = torch.nn.Conv2d(cm, (k * k) * (s * s), kernel_size=3, stride=1, padding=1, bias=True)
        self.out_proj = torch.nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0, bias=True)

        # Safe start: CARAFE branch initially produces near-zero residual.
        torch.nn.init.zeros_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            torch.nn.init.zeros_(self.out_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        s = self.scale
        k = self.kernel_size
        hs, ws = h * s, w * s

        kernel_logits = self.encoder(self.comp(x))
        kernel = F.pixel_shuffle(kernel_logits, upscale_factor=s)  # [B, k*k, H*s, W*s]
        kernel = kernel.float()
        kernel = kernel - kernel.amax(dim=1, keepdim=True)
        kernel = torch.softmax(kernel, dim=1).to(dtype=x.dtype)

        x_up = F.interpolate(x, size=(hs, ws), mode="nearest")
        kflat = kernel.view(b, k * k, hs * ws).unsqueeze(1)

        # Memory-safe feature reassembly: process channels in chunks to cap unfold peak memory.
        out = x_up.new_zeros((b, c, hs, ws))
        step = int(self.chunk_channels)
        for c0 in range(0, c, step):
            c1 = min(c, c0 + step)
            part = x_up[:, c0:c1, :, :]
            patches = F.unfold(part, kernel_size=k, dilation=1, padding=k // 2, stride=1)
            patches = patches.view(b, c1 - c0, k * k, hs * ws)
            y_part = (patches * kflat).sum(dim=2).view(b, c1 - c0, hs, ws)
            out[:, c0:c1, :, :] = y_part

        return self.out_proj(out)


class CARAFEUpsampleSafe(torch.nn.Module):
    """Residual-safe b7 wrapper: out = base_up(x) + alpha * (carafe(x) - base_up(x))."""

    def __init__(
        self,
        base_module: torch.nn.Module,
        channels: int,
        scale: int = 2,
        kernel_size: int = 5,
        compress: int = 64,
        chunk_channels: int = 64,
        alpha_init: float = 0.0,
        alpha_cap: float = 0.5,
        tag: str = "",
    ) -> None:
        super().__init__()
        self.enhance241_b7_base = base_module
        self.enhance241_b7_carafe = CARAFECore(
            channels=int(channels),
            scale=int(scale),
            kernel_size=int(kernel_size),
            compress=int(compress),
            chunk_channels=int(chunk_channels),
        )
        self.tag = str(tag)
        self.alpha_cap = float(max(1e-6, abs(alpha_cap)))

        alpha_init = float(max(-self.alpha_cap * 0.95, min(self.alpha_cap * 0.95, alpha_init)))
        alpha_ratio = alpha_init / self.alpha_cap
        alpha_raw = torch.atanh(torch.tensor(alpha_ratio, dtype=torch.float32))
        self.enhance241_b7_alpha = torch.nn.Parameter(alpha_raw)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        recorder = _get_module_recorder(self)
        prefix = str(getattr(self, "_enhance241_prefix", "b7"))
        step = int(getattr(self, "_enhance241_fwd_step", 0))

        if recorder is not None and step == 0:
            recorder.capture_param_before(self, prefix)

        y_base = self.enhance241_b7_base(x)
        y_carafe = self.enhance241_b7_carafe(x)
        if y_carafe.shape[-2:] != y_base.shape[-2:]:
            y_carafe = F.interpolate(y_carafe, size=y_base.shape[-2:], mode="nearest")

        alpha_raw = self.enhance241_b7_alpha.to(dtype=y_base.dtype, device=y_base.device)
        alpha = torch.tanh(alpha_raw) * self.alpha_cap
        delta = y_carafe - y_base
        out = y_base + alpha * delta

        if recorder is not None:
            if step == 0:
                recorder.record_module_compare(f"{prefix}.upsample", patched=out, baseline=y_base, input_tensor=x)
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
                        "tag": self.tag,
                        "cosine_out_vs_base": cos,
                        "var_ratio_out_over_base": v_out / (v_base + 1e-12),
                        "thresholds": {"cosine_min": 0.98, "var_ratio_min": 0.5},
                        "pass": gate1_ok,
                        "alpha_raw": _to_float(alpha_raw.item()),
                        "alpha": _to_float(alpha.item()),
                        "alpha_cap": _to_float(self.alpha_cap),
                        "base_stats": _tensor_stats(y_base),
                        "carafe_stats": _tensor_stats(y_carafe),
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
                        out.register_hook(lambda g, n=f"{prefix}.upsample": recorder.record_output_grad(n, g))
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
    enable_b7 = bool(_deep_get(cfg, "enhance241", "b7", default=False))
    if not enable_b7:
        return model

    if any(bool(_deep_get(cfg, "enhance241", key, default=False)) for key in ("b1", "b2", "b3", "b5", "b9")):
        raise RuntimeError("enhance241.b7 conflicts with b1/b2/b3/b5/b9; enable only one B-class module.")

    yolo_obj = model
    if hasattr(model, "model") and hasattr(getattr(model, "model"), "model"):
        det_model = getattr(model, "model")
        seq = getattr(det_model, "model", None)
    else:
        det_model = model
        seq = getattr(det_model, "model", None)
    if seq is None:
        raise RuntimeError("enhance241.b7 requires YOLO/DetectionModel-like object with .model sequence.")

    prepatched = [i for i, layer in enumerate(seq) if isinstance(layer, CARAFEUpsampleSafe)]
    if prepatched:
        info = {
            "enabled": True,
            "patched_indices": prepatched,
            "patched_count": 0,
            "existing_count": len(prepatched),
            "note": "already_patched",
        }
        setattr(yolo_obj, "_enhance241_b7_info", info)
        recorder = get_check_recorder("b7", cfg, patch_info=info)
        if recorder is not None:
            try:
                for idx in prepatched:
                    m = seq[int(idx)]
                    _bind_module_debug(m, recorder, prefix=f"b7.idx{int(idx)}.existing")
                    recorder.register_module_params(m, f"b7.idx{int(idx)}.existing")
                detect_idx, detect_layer = _locate_detect(seq)
                recorder.attach_detect_hooks(detect_layer, detect_idx)
                recorder.maybe_run_val_separability(yolo_obj, cfg)
            except Exception as exc:
                recorder.add_note(f"b7_prepatched_hook_failed:{exc}")
        if len(prepatched) >= 2:
            return yolo_obj
        raise RuntimeError(f"enhance241.b7 expects two patched upsample modules, got {len(prepatched)}")

    p3_concat_idx = _locate_p4_to_p3_fuse(seq)
    p5_concat_idx = _locate_p5_to_p4_fuse(seq, p3_concat_idx)

    kernel_size = _safe_int(_deep_get(cfg, "enhance241", "b7_kernel_size", default=5), 5)
    compress = _safe_int(_deep_get(cfg, "enhance241", "b7_compress", default=64), 64)
    chunk_channels = _safe_int(_deep_get(cfg, "enhance241", "b7_chunk_channels", default=64), 64)
    alpha_init = _safe_float(_deep_get(cfg, "enhance241", "b7_alpha_init", default=0.05), 0.05)
    alpha_auto_fallback = False
    if abs(alpha_init) < 1e-8:
        # Avoid dead-start where CARAFE branch receives no gradient in the first steps.
        alpha_init = 0.05
        alpha_auto_fallback = True
    alpha_cap = _safe_float(_deep_get(cfg, "enhance241", "b7_alpha_cap", default=0.5), 0.5)

    patched_indices: List[int] = []
    recorder = get_check_recorder("b7", cfg)

    def _patch_one(concat_idx: int, tag: str) -> None:
        up_idx = int(concat_idx) - 1
        if up_idx < 0:
            raise RuntimeError(f"Invalid upsample index for concat idx={concat_idx}")
        old = seq[up_idx]
        old_name = old.__class__.__name__.lower()
        if "upsample" not in old_name:
            raise RuntimeError(
                f"Expected upsample module at idx={up_idx} before concat idx={concat_idx}, got {old.__class__.__name__}"
            )

        channels = _infer_concat_channels(seq, concat_idx)
        scale = _infer_scale_factor(old)
        mod = CARAFEUpsampleSafe(
            old,
            channels=channels,
            scale=scale,
            kernel_size=kernel_size,
            compress=compress,
            chunk_channels=chunk_channels,
            alpha_init=alpha_init,
            alpha_cap=alpha_cap,
            tag=tag,
        )
        for attr in ("i", "f", "type"):
            if hasattr(old, attr):
                setattr(mod, attr, getattr(old, attr))
        device, dtype = _infer_device_dtype(seq, up_idx)
        if device is not None:
            if dtype is not None:
                mod = mod.to(device=device, dtype=dtype)
            else:
                mod = mod.to(device=device)
        seq[up_idx] = mod
        patched_indices.append(up_idx)

        if recorder is not None:
            try:
                _bind_module_debug(mod, recorder, prefix=f"b7.idx{up_idx}.{tag}")
                recorder.register_module_params(mod, f"b7.idx{up_idx}.{tag}")
            except Exception as exc:
                recorder.add_note(f"b7_attach_debug_failed:{up_idx}:{exc}")

    _patch_one(p5_concat_idx, "p5p4")
    _patch_one(p3_concat_idx, "p4p3")

    info = {
        "enabled": True,
        "patched_indices": [int(x) for x in patched_indices],
        "patched_count": len(patched_indices),
        "existing_count": 0,
        "concat_indices": [int(p5_concat_idx), int(p3_concat_idx)],
        "concat_candidates": _concat_candidates(seq),
        "kernel_size": int(kernel_size),
        "compress": int(compress),
        "chunk_channels": int(chunk_channels),
        "alpha_init": _to_float(alpha_init),
        "alpha_auto_fallback": bool(alpha_auto_fallback),
        "alpha_cap": _to_float(alpha_cap),
        "mode": "carafe_residual_safe",
    }
    setattr(yolo_obj, "_enhance241_b7_info", info)

    if recorder is not None:
        recorder.set_patch_info(info)
        try:
            detect_idx, detect_layer = _locate_detect(seq)
            recorder.attach_detect_hooks(detect_layer, detect_idx)
        except Exception as exc:
            recorder.add_note(f"b7_detect_hook_failed:{exc}")
        try:
            recorder.maybe_run_val_separability(yolo_obj, cfg)
        except Exception as exc:
            recorder.add_note(f"b7_val_check_failed:{exc}")

    return yolo_obj
