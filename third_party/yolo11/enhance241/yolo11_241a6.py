from __future__ import annotations

# Path: third_party/yolo11/enhance241/yolo11_241a6.py
# Purpose: enhance241 a6 (a4 geometry-preserving SPD + LSK semantic block on P3 route).

from typing import Any, Optional, Tuple

import torch
import torch.nn.functional as F

from .yolo11_241a3 import (
    _bind_module_debug,
    _get_module_recorder,
    _locate_detect,
    _should_capture_delta,
    _tensor_stats,
    get_check_recorder,
)
from .yolo11_241a4 import (
    SPDConvDownsample,
    _deep_get,
    _find_first_conv,
    _find_stride_conv,
    _infer_device_dtype,
    _locate_p4_to_p3_fuse,
    _module_stride,
    _safe_float,
    _safe_int,
    _to_float,
)

ENHANCE241_AUDIT_KEYS = ["enhance241_a6"]  # enhance241-audit


class LSKBlock(torch.nn.Module):
    """PKI-style pyramid knowledge injection block.

    Keeps the historical class name for checkpoint compatibility.
    Old checkpoints without PKI attrs fall back to the legacy LSK path.
    """

    def __init__(self, channels: int, kernel_size: int = 7, dilation: int = 3, reduction: int = 16) -> None:
        super().__init__()
        c = int(channels)
        d3 = max(3, int(dilation))
        hidden = max(4, c // max(1, int(reduction)))

        self.enhance241_a6_pki_d1 = torch.nn.Conv2d(
            c,
            c,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1,
            groups=c,
            bias=True,
        )
        self.enhance241_a6_pki_d2 = torch.nn.Conv2d(
            c,
            c,
            kernel_size=3,
            stride=1,
            padding=2,
            dilation=2,
            groups=c,
            bias=True,
        )
        self.enhance241_a6_pki_d3 = torch.nn.Conv2d(
            c,
            c,
            kernel_size=3,
            stride=1,
            padding=d3,
            dilation=d3,
            groups=c,
            bias=True,
        )
        self.enhance241_a6_pki_proj = torch.nn.Conv2d(c * 3, c, kernel_size=1, stride=1, padding=0, bias=True)
        self.enhance241_a6_pki_ca1 = torch.nn.Conv2d(c, hidden, kernel_size=1, stride=1, padding=0, bias=True)
        self.enhance241_a6_pki_ca2 = torch.nn.Conv2d(hidden, c, kernel_size=1, stride=1, padding=0, bias=True)
        torch.nn.init.zeros_(self.enhance241_a6_pki_ca2.weight)
        if self.enhance241_a6_pki_ca2.bias is not None:
            torch.nn.init.zeros_(self.enhance241_a6_pki_ca2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self, "enhance241_a6_pki_proj"):
            p1 = self.enhance241_a6_pki_d1(x)
            p2 = self.enhance241_a6_pki_d2(x)
            p3 = self.enhance241_a6_pki_d3(x)
            fused = self.enhance241_a6_pki_proj(torch.cat((p1, p2, p3), dim=1))
            pooled = F.adaptive_avg_pool2d(fused, output_size=1)
            gate = self.enhance241_a6_pki_ca2(F.silu(self.enhance241_a6_pki_ca1(pooled)))
            gate = torch.sigmoid(gate) * 2.0
            return fused * gate

        local_feat = self.dw_local(x)
        median_feat = self.dw_median(local_feat)
        large_feat = self.dw_large(local_feat)
        gate_logits = self.enhance241_a6_mix(torch.cat((median_feat, large_feat), dim=1))
        gate = torch.sigmoid(gate_logits) * 2.0
        return x * gate


class A6LSKSPDDownsampleSafe(torch.nn.Module):
    """a6 = a4 geometry branch + residual-safe PKI semantic shaping."""

    def __init__(
        self,
        base_downsample: torch.nn.Module,
        in_ch: int,
        out_ch: int,
        pre_div: int = 4,
        refine: str = "dw",
        lsk_kernel: int = 7,
        lsk_dilation: int = 3,
        attn_reduction: int = 16,
        alpha_init: float = 0.02,
        alpha_cap: float = 0.5,
    ) -> None:
        super().__init__()
        self.enhance241_a6_geom = SPDConvDownsample(
            base_downsample=base_downsample,
            in_ch=int(in_ch),
            out_ch=int(out_ch),
            pre_div=int(pre_div),
            refine=str(refine),
            alpha_init=float(alpha_init),
            alpha_cap=float(alpha_cap),
        )
        self.enhance241_a6_pki = LSKBlock(
            int(out_ch),
            kernel_size=int(lsk_kernel),
            dilation=int(lsk_dilation),
            reduction=int(attn_reduction),
        )
        self.alpha_cap = float(max(1e-6, abs(alpha_cap)))
        alpha_init = float(max(-self.alpha_cap * 0.95, min(self.alpha_cap * 0.95, alpha_init)))
        alpha_ratio = alpha_init / self.alpha_cap
        alpha_raw = torch.atanh(torch.tensor(alpha_ratio, dtype=torch.float32))
        self.enhance241_a6_alpha = torch.nn.Parameter(alpha_raw)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        recorder = _get_module_recorder(self)
        prefix = str(getattr(self, "_enhance241_prefix", "a6"))
        step = int(getattr(self, "_enhance241_fwd_step", 0))

        if recorder is not None and step == 0:
            recorder.capture_param_before(self, prefix)

        y_geom = self.enhance241_a6_geom(x)
        pki = getattr(self, "enhance241_a6_pki", None)
        if pki is None:
            pki = getattr(self, "enhance241_a6_lsk")
        y_pki = pki(y_geom)
        delta = y_pki - y_geom
        alpha_raw = self.enhance241_a6_alpha.to(dtype=y_geom.dtype, device=y_geom.device)
        alpha = torch.tanh(alpha_raw) * self.alpha_cap
        out = y_geom + alpha * delta

        if recorder is not None:
            if step == 0:
                recorder.record_module_compare(f"{prefix}.p3", patched=out, baseline=y_geom, input_tensor=x)
                recorder.record_a1_payload(
                    f"{prefix}.gate1",
                    {
                        "alpha": _to_float(alpha.item()),
                        "alpha_cap": _to_float(self.alpha_cap),
                        "geom_stats": _tensor_stats(y_geom),
                        "pki_stats": _tensor_stats(y_pki),
                        "delta_stats": _tensor_stats(delta),
                        "out_stats": _tensor_stats(out),
                    },
                )
                if out.requires_grad:
                    try:
                        out.register_hook(lambda g, n=f"{prefix}.p3": recorder.record_output_grad(n, g))
                    except Exception:
                        pass
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
    enable_a6 = bool(_deep_get(cfg, "enhance241", "a6", default=False))
    if not enable_a6:
        return model

    yolo_obj, seq = _extract_model_seq(model)
    if seq is None:
        raise RuntimeError("enhance241.a6 requires an ultralytics YOLO/DetectionModel-like object with a .model.")

    prepatched = [i for i, layer in enumerate(seq) if isinstance(layer, A6LSKSPDDownsampleSafe)]
    if prepatched:
        info = {"enabled": True, "existing_count": len(prepatched), "patched_count": 0, "patched_indices": prepatched}
        setattr(yolo_obj, "_enhance241_a6_info", info)
        recorder = get_check_recorder("a6", cfg, patch_info=info)
        if recorder is not None:
            try:
                idx0 = int(prepatched[0])
                mod0 = seq[idx0]
                _bind_module_debug(mod0, recorder, prefix=f"a6.idx{idx0}.existing")
                recorder.register_module_params(mod0, f"a6.idx{idx0}.existing")
                detect_idx, detect_layer = _locate_detect(seq)
                recorder.attach_detect_hooks(detect_layer, detect_idx)
            except Exception as exc:
                recorder.add_note(f"a6_prepatched_hook_failed:{exc}")
        return yolo_obj

    _, p3_idx = _locate_p4_to_p3_fuse(seq)
    ds_idx: Optional[int] = None
    for i in range(p3_idx, -1, -1):
        if _module_stride(seq[i]) == 2:
            ds_idx = i
            break
    if ds_idx is None:
        raise RuntimeError("enhance241.a6: unable to locate stride=2 downsample before P3 output.")

    old = seq[ds_idx]
    conv = _find_stride_conv(old, stride=2) or _find_first_conv(old)
    if conv is None:
        raise RuntimeError(f"enhance241.a6: unable to infer in/out channels at idx={ds_idx}.")

    in_ch = int(conv.in_channels)
    out_ch = int(conv.out_channels)
    a6_pre_div = _safe_int(_deep_get(cfg, "enhance241", "a6_pre_div", default=4), 4)
    a6_refine = str(_deep_get(cfg, "enhance241", "a6_refine", default="dw"))
    a6_lsk_kernel = _safe_int(_deep_get(cfg, "enhance241", "a6_lsk_kernel", default=7), 7)
    a6_lsk_dilation = _safe_int(_deep_get(cfg, "enhance241", "a6_lsk_dilation", default=3), 3)
    a6_attn_reduction = _safe_int(_deep_get(cfg, "enhance241", "a6_attn_reduction", default=16), 16)
    a6_alpha_init = _safe_float(_deep_get(cfg, "enhance241", "a6_alpha_init", default=0.02), 0.02)
    a6_alpha_cap = _safe_float(_deep_get(cfg, "enhance241", "a6_alpha_cap", default=0.5), 0.5)

    fuse = A6LSKSPDDownsampleSafe(
        base_downsample=old,
        in_ch=in_ch,
        out_ch=out_ch,
        pre_div=a6_pre_div,
        refine=a6_refine,
        lsk_kernel=a6_lsk_kernel,
        lsk_dilation=a6_lsk_dilation,
        attn_reduction=a6_attn_reduction,
        alpha_init=a6_alpha_init,
        alpha_cap=a6_alpha_cap,
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
        "new_type": "A6LSKSPDDownsampleSafe",
        "a6_refine": str(a6_refine),
        "a6_lsk_kernel": int(a6_lsk_kernel),
        "a6_lsk_dilation": int(a6_lsk_dilation),
        "a6_attn_reduction": int(a6_attn_reduction),
        "alpha_init": _to_float(a6_alpha_init),
        "alpha_cap": _to_float(a6_alpha_cap),
    }
    setattr(yolo_obj, "_enhance241_a6_info", info)
    print(
        f"[enhance241] a6 enabled: patched model.model[{int(ds_idx)}] "
        f"-> A6LSKSPDDownsampleSafe(PKI d=1/2/{a6_lsk_dilation}, reduction={a6_attn_reduction})"
    )

    recorder = get_check_recorder("a6", cfg, patch_info=info)
    if recorder is not None:
        _bind_module_debug(fuse, recorder, prefix=f"a6.idx{int(ds_idx)}")
        recorder.register_module_params(fuse, f"a6.idx{int(ds_idx)}")
        try:
            detect_idx, detect_layer = _locate_detect(seq)
            recorder.attach_detect_hooks(detect_layer, detect_idx)
        except Exception as exc:
            recorder.add_note(f"a6_detect_hook_failed:{exc}")

    return yolo_obj
