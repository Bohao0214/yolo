from __future__ import annotations

# Path: third_party/yolo11/enhance241/yolo11_241b5.py
# Purpose: enhance241 b5 patch (GFPN-like CSPStage fusion refinement).

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from .yolo11_241a3 import (
    _bind_module_debug,
    _f_as_list,
    _get_module_recorder,
    _infer_device_dtype,
    _locate_detect,
    _tensor_stats,
    _to_float,
    get_check_recorder,
)

ENHANCE241_AUDIT_KEYS = ["enhance241_b5"]  # enhance241-audit


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
    if not c_in or c_in % 2 != 0:
        raise RuntimeError(f"Unable to infer concat channels from next layer idx={fuse_idx + 1}; c_in={c_in}")
    return c_in // 2


class _DWSeparable(torch.nn.Module):
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


class _Conv3x3(torch.nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.act = torch.nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(x))


class CSPStageLite(torch.nn.Module):
    """GFPN-style CSP refinement core."""

    def __init__(self, in_ch: int, out_ch: int, depth: int = 2, expansion: float = 0.5, refine: str = "dw") -> None:
        super().__init__()
        hidden = max(16, int(out_ch * float(expansion)))
        depth = max(1, int(depth))
        refine = str(refine).lower()

        self.cv1 = torch.nn.Conv2d(in_ch, hidden * 2, kernel_size=1, stride=1, padding=0, bias=True)
        if refine == "conv":
            self.blocks = torch.nn.ModuleList([_Conv3x3(hidden) for _ in range(depth)])
        else:
            self.blocks = torch.nn.ModuleList([_DWSeparable(hidden) for _ in range(depth)])
        self.cv2 = torch.nn.Conv2d(hidden * (2 + depth), out_ch, kernel_size=1, stride=1, padding=0, bias=True)
        # Safe-start for residual injection.
        torch.nn.init.zeros_(self.cv2.weight)
        if self.cv2.bias is not None:
            torch.nn.init.zeros_(self.cv2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = torch.chunk(self.cv1(x), 2, dim=1)
        parts = [x1, x2]
        y = x2
        for blk in self.blocks:
            y = blk(y)
            parts.append(y)
        return self.cv2(torch.cat(parts, dim=1))


class B5GFPNFuse(torch.nn.Module):
    """GFPN-lite fusion replacement for Concat output (2C channels kept)."""

    def __init__(
        self,
        channels_per_branch: int,
        depth: int = 2,
        expansion: float = 0.5,
        refine: str = "dw",
        upsample_mode: str = "nearest",
        tag: str = "",
    ) -> None:
        super().__init__()
        self.c = int(channels_per_branch)
        self.tag = str(tag)
        self.upsample_mode = str(upsample_mode).lower()
        self.enhance241_b5_csp = CSPStageLite(2 * self.c, 2 * self.c, depth=depth, expansion=expansion, refine=refine)
        self.enhance241_b5_alpha = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

    def _split_input(self, x: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(x, (list, tuple)) and len(x) == 2:
            return x[0], x[1]
        if isinstance(x, torch.Tensor):
            if x.shape[1] != self.c * 2:
                raise ValueError(f"Expected 2C concat tensor, got {x.shape}")
            return torch.split(x, self.c, dim=1)
        raise TypeError(f"B5GFPNFuse expects [hi, lo] or concat tensor, got {type(x)}")

    def _maybe_align(self, hi: torch.Tensor, lo: torch.Tensor) -> torch.Tensor:
        if hi.shape[-2:] == lo.shape[-2:]:
            return hi
        if self.upsample_mode == "bilinear":
            return F.interpolate(hi, size=lo.shape[-2:], mode="bilinear", align_corners=False)
        return F.interpolate(hi, size=lo.shape[-2:], mode="nearest")

    def forward(self, x: Any) -> torch.Tensor:
        recorder = _get_module_recorder(self)
        prefix = str(getattr(self, "_enhance241_prefix", "b5"))
        step = int(getattr(self, "_enhance241_fwd_step", 0))

        hi, lo = self._split_input(x)
        hi = self._maybe_align(hi, lo)
        base = torch.cat((hi, lo), dim=1)
        if recorder is not None and step == 0:
            recorder.capture_param_before(self, prefix)

        delta = self.enhance241_b5_csp(base)
        alpha = self.enhance241_b5_alpha.to(dtype=base.dtype, device=base.device)
        out = base + alpha * delta

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
                    f"{prefix}.gfpn",
                    {
                        "tag": self.tag,
                        "alpha": _to_float(alpha.item()),
                        "cosine_out_vs_base": cos,
                        "var_ratio_out_over_base": v_out / (v_base + 1e-12),
                        "hi_stats": _tensor_stats(hi),
                        "lo_stats": _tensor_stats(lo),
                        "out_stats": _tensor_stats(out),
                    },
                )
                recorder.record_scalar_curve(f"{prefix}.alpha", _to_float(alpha.item()), step, max_steps=100)
                if out.requires_grad:
                    try:
                        out.register_hook(lambda g, n=f"{prefix}.fuse": recorder.record_output_grad(n, g))
                    except Exception:
                        pass
            elif step <= 100:
                recorder.record_scalar_curve(f"{prefix}.alpha", _to_float(alpha.item()), step, max_steps=100)
            if step == 1:
                recorder.capture_param_delta(self, prefix)

        setattr(self, "_enhance241_fwd_step", step + 1)
        return out


def apply(model: Any, cfg: Any) -> Any:
    enable_b5 = bool(_deep_get(cfg, "enhance241", "b5", default=False))
    if not enable_b5:
        return model

    if any(
        bool(_deep_get(cfg, "enhance241", key, default=False))
        for key in ("b1", "b2", "b3")
    ):
        raise RuntimeError("enhance241.b5 conflicts with b1/b2/b3; enable only one B-class module.")

    yolo_obj = model
    if hasattr(model, "model") and hasattr(getattr(model, "model"), "model"):
        det_model = getattr(model, "model")
        seq = getattr(det_model, "model", None)
    else:
        det_model = model
        seq = getattr(det_model, "model", None)
    if seq is None:
        raise RuntimeError("enhance241.b5 requires YOLO/DetectionModel-like object with .model sequence.")

    prepatched = [i for i, layer in enumerate(seq) if isinstance(layer, B5GFPNFuse)]
    if prepatched:
        info = {
            "enabled": True,
            "patched_indices": prepatched,
            "patched_count": len(prepatched),
            "existing_count": len(prepatched),
            "note": "already_patched",
        }
        setattr(yolo_obj, "_enhance241_b5_info", info)
        recorder = get_check_recorder("b5", cfg, patch_info=info)
        if recorder is not None:
            try:
                for idx in prepatched:
                    m = seq[int(idx)]
                    _bind_module_debug(m, recorder, prefix=f"b5.idx{int(idx)}.existing")
                    recorder.register_module_params(m, f"b5.idx{int(idx)}.existing")
                detect_idx, detect_layer = _locate_detect(seq)
                recorder.attach_detect_hooks(detect_layer, detect_idx)
                recorder.maybe_run_val_separability(yolo_obj, cfg)
            except Exception as exc:
                recorder.add_note(f"b5_prepatched_hook_failed:{exc}")
        if len(prepatched) >= 2:
            return yolo_obj
        raise RuntimeError(f"enhance241.b5 expects two B5GFPNFuse modules, got {len(prepatched)}")

    p3_concat_idx = _locate_p4_to_p3_fuse(seq)
    p5_concat_idx = _locate_p5_to_p4_fuse(seq, p3_concat_idx)

    depth = _safe_int(_deep_get(cfg, "enhance241", "b5_depth", default=2), 2)
    expansion = _safe_float(_deep_get(cfg, "enhance241", "b5_expansion", default=0.5), 0.5)
    refine = str(_deep_get(cfg, "enhance241", "b5_refine", default="dw"))
    upsample = str(_deep_get(cfg, "enhance241", "b5_upsample", default="nearest")).lower()

    patched_indices: List[int] = []
    recorder = get_check_recorder("b5", cfg)

    def _patch_one(idx: int, tag: str) -> None:
        old = seq[idx]
        if old.__class__.__name__ != "Concat":
            raise RuntimeError(
                f"Expected Concat at idx={idx} for B5GFPNFuse({tag}), got {old.__class__.__name__}."
            )
        c = _infer_concat_channels(seq, idx)
        mod = B5GFPNFuse(c, depth=depth, expansion=expansion, refine=refine, upsample_mode=upsample, tag=tag)
        for attr in ("i", "f", "type"):
            if hasattr(old, attr):
                setattr(mod, attr, getattr(old, attr))
        device, dtype = _infer_device_dtype(seq, idx)
        if device is not None:
            if dtype is not None:
                mod = mod.to(device=device, dtype=dtype)
            else:
                mod = mod.to(device=device)
        seq[idx] = mod
        patched_indices.append(idx)

        if recorder is not None:
            try:
                _bind_module_debug(mod, recorder, prefix=f"b5.idx{idx}.{tag}")
                recorder.register_module_params(mod, f"b5.idx{idx}.{tag}")
            except Exception as exc:
                recorder.add_note(f"b5_attach_debug_failed:{idx}:{exc}")

    _patch_one(p5_concat_idx, "p5p4")
    _patch_one(p3_concat_idx, "p4p3")

    info = {
        "enabled": True,
        "patched_indices": patched_indices,
        "patched_count": len(patched_indices),
        "existing_count": 0,
        "concat_candidates": _concat_candidates(seq),
        "depth": int(depth),
        "expansion": float(expansion),
        "refine": str(refine),
        "upsample": str(upsample),
        "alpha_init": 0.0,
        "mode": "gfpn_csp_residual_safe",
    }
    setattr(yolo_obj, "_enhance241_b5_info", info)

    if recorder is not None:
        recorder.set_patch_info(info)
        try:
            detect_idx, detect_layer = _locate_detect(seq)
            recorder.attach_detect_hooks(detect_layer, detect_idx)
        except Exception as exc:
            recorder.add_note(f"b5_detect_hook_failed:{exc}")
        try:
            recorder.maybe_run_val_separability(yolo_obj, cfg)
        except Exception as exc:
            recorder.add_note(f"b5_val_check_failed:{exc}")

    return yolo_obj
