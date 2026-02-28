from __future__ import annotations

# Path: third_party/yolo11/enhance241/yolo11_241c6.py
# Purpose: enhance241 c6 (Gated-BRA sparse guardrail before head).

import math
from typing import Any, Dict, List, Optional, Tuple

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
from .yolo11_241c5 import (
    _deep_get,
    _extract_model_seq,
    _infer_p3_channels,
    _infer_p3_output_index,
    _safe_float,
    _safe_int,
    _window_partition,
    _window_reverse,
)

ENHANCE241_AUDIT_KEYS = ["enhance241_c6"]  # enhance241-audit


def _to_input_tensor(x: Any) -> Optional[torch.Tensor]:
    if isinstance(x, torch.Tensor):
        return x
    if isinstance(x, (list, tuple)):
        for item in x:
            if isinstance(item, torch.Tensor):
                return item
    return None


class GatedBRACore(torch.nn.Module):
    """BRA + route-mask export for sparse gated enhancement."""

    def __init__(
        self,
        channels: int,
        num_heads: int = 4,
        window_size: int = 8,
        topk: int = 4,
        kv_downsample_mode: str = "avg",
        soft_routing: bool = True,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.num_heads = max(1, int(num_heads))
        if self.channels % self.num_heads != 0:
            self.num_heads = 1
        self.head_dim = self.channels // self.num_heads
        self.window_size = max(2, int(window_size))
        self.topk = max(1, int(topk))
        self.kv_downsample_mode = str(kv_downsample_mode).lower()
        self.soft_routing = bool(soft_routing)
        self.eps = float(eps)
        self.enhance241_c6_qkv = torch.nn.Conv2d(self.channels, self.channels * 3, kernel_size=1, stride=1, padding=0)
        self.enhance241_c6_proj = torch.nn.Conv2d(self.channels, self.channels, kernel_size=1, stride=1, padding=0)
        torch.nn.init.zeros_(self.enhance241_c6_proj.weight)
        if self.enhance241_c6_proj.bias is not None:
            torch.nn.init.zeros_(self.enhance241_c6_proj.bias)

    def _prepare(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int, int, int]:
        b, c, h, w = x.shape
        ws = self.window_size
        pad_h = (ws - h % ws) % ws
        pad_w = (ws - w % ws) % ws
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
        hp, wp = x.shape[-2:]
        return x, h, w, hp, wp

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_pad, h, w, hp, wp = self._prepare(x)
        qkv = self.enhance241_c6_qkv(x_pad)
        q, k, v = torch.chunk(qkv, 3, dim=1)

        ws = self.window_size
        q_w = _window_partition(q, ws)
        k_w = _window_partition(k, ws)
        v_w = _window_partition(v, ws)

        if self.kv_downsample_mode == "max":
            k_tok = k_w.amax(dim=2, keepdim=True)
            v_tok = v_w.amax(dim=2, keepdim=True)
        elif self.kv_downsample_mode == "identity":
            k_tok = k_w
            v_tok = v_w
        else:
            k_tok = k_w.mean(dim=2, keepdim=True)
            v_tok = v_w.mean(dim=2, keepdim=True)

        q_mean = q_w.mean(dim=2)
        k_mean = k_w.mean(dim=2)
        scale_win = 1.0 / math.sqrt(max(1, self.channels))
        routing_logits = torch.matmul(q_mean.float(), k_mean.float().transpose(-1, -2)) * scale_win
        topk = min(self.topk, int(routing_logits.shape[-1]))
        topv, topi = torch.topk(routing_logits, k=topk, dim=-1)

        routing_w: Optional[torch.Tensor] = None
        if self.soft_routing:
            topv = topv - topv.amax(dim=-1, keepdim=True)
            routing_w = torch.softmax(topv, dim=-1).to(dtype=q_w.dtype)
            route_strength = routing_w.amax(dim=-1)
        else:
            route_strength = torch.sigmoid(topv.float()).mean(dim=-1).to(dtype=q_w.dtype)

        bsz, nwin, tq, c = q_w.shape
        tk = int(k_tok.shape[2])
        out_windows: List[torch.Tensor] = []
        scale_head = 1.0 / math.sqrt(max(1, self.head_dim))

        for b in range(bsz):
            q_b = q_w[b]
            k_b = k_tok[b]
            v_b = v_tok[b]
            idx_b = topi[b]

            k_sel = k_b[idx_b]
            v_sel = v_b[idx_b]

            qh = q_b.view(nwin, tq, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            kh = k_sel.view(nwin, topk * tk, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            vh = v_sel.view(nwin, topk * tk, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

            logits = torch.matmul(qh.float(), kh.float().transpose(-1, -2)) * scale_head
            logits = logits - logits.amax(dim=-1, keepdim=True)
            attn = torch.softmax(logits, dim=-1).to(dtype=qh.dtype)

            if routing_w is not None:
                rw = routing_w[b].repeat_interleave(tk, dim=-1)
                rw = rw.unsqueeze(1).unsqueeze(2)
                attn = attn * rw
                attn = attn / attn.sum(dim=-1, keepdim=True).clamp_min(self.eps)

            out_h = torch.matmul(attn.float(), vh.float()).to(dtype=qh.dtype)
            out_b = out_h.permute(0, 2, 1, 3).contiguous().view(nwin, tq, c)
            out_windows.append(out_b)

        out_w = torch.stack(out_windows, dim=0)
        out = _window_reverse(out_w, ws, hp, wp)
        out = out[..., :h, :w]
        out = self.enhance241_c6_proj(out)

        route_mask_windows = route_strength.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, ws * ws, 1)
        route_mask = _window_reverse(route_mask_windows, ws, hp, wp)
        route_mask = route_mask[..., :h, :w]
        return out, route_mask.to(dtype=out.dtype, device=out.device)


class C6GatedBRAInject(torch.nn.Module):
    """Gated-BRA: out = base + sigmoid(route_mask) * BRA(base)."""

    def __init__(
        self,
        base_module: torch.nn.Module,
        channels: int,
        num_heads: int = 4,
        window_size: int = 8,
        topk: int = 4,
        kv_downsample_mode: str = "avg",
        soft_routing: bool = True,
        gate_scale: float = 6.0,
        gate_bias: float = -0.5,
    ) -> None:
        super().__init__()
        self.enhance241_c6_base = base_module
        self.enhance241_c6_bra = GatedBRACore(
            channels=channels,
            num_heads=num_heads,
            window_size=window_size,
            topk=topk,
            kv_downsample_mode=kv_downsample_mode,
            soft_routing=soft_routing,
        )
        self.gate_scale = float(gate_scale)
        self.gate_bias = float(gate_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        recorder = _get_module_recorder(self)
        prefix = str(getattr(self, "_enhance241_prefix", "c6"))
        step = int(getattr(self, "_enhance241_fwd_step", 0))

        if recorder is not None and step == 0:
            recorder.capture_param_before(self, prefix)

        y_base = self.enhance241_c6_base(x)
        delta, route_mask = self.enhance241_c6_bra(y_base)
        gate = torch.sigmoid((route_mask + self.gate_bias) * self.gate_scale)
        out = y_base + gate * delta

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
                        "gate_scale": _to_float(self.gate_scale),
                        "gate_bias": _to_float(self.gate_bias),
                        "route_mask_stats": _tensor_stats(route_mask),
                        "gate_stats": _tensor_stats(gate),
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


def apply(model: Any, cfg: Any) -> Any:
    enable_c6 = bool(_deep_get(cfg, "enhance241", "c6", default=False))
    if not enable_c6:
        return model

    yolo_obj, seq = _extract_model_seq(model)
    if seq is None:
        raise RuntimeError("enhance241.c6 requires YOLO/DetectionModel-like object with .model sequence.")

    p3_idx = _infer_p3_output_index(seq)
    detect_idx, detect = _locate_detect(seq)
    old = seq[p3_idx]

    if isinstance(old, C6GatedBRAInject):
        info = {
            "enabled": True,
            "existing_count": 1,
            "patched_count": 0,
            "p3_index": int(p3_idx),
            "detect_idx": int(detect_idx),
            "note": "already_patched",
        }
        setattr(yolo_obj, "_enhance241_c6_info", info)
        return yolo_obj

    channels = _infer_p3_channels(detect)
    num_heads = _safe_int(_deep_get(cfg, "enhance241", "c6_num_heads", default=4), 4)
    window_size = _safe_int(_deep_get(cfg, "enhance241", "c6_window_size", default=8), 8)
    topk = _safe_int(_deep_get(cfg, "enhance241", "c6_topk", default=4), 4)
    kv_mode = str(_deep_get(cfg, "enhance241", "c6_kv_downsample_mode", default="avg")).lower()
    soft_routing = bool(_deep_get(cfg, "enhance241", "c6_soft_routing", default=True))
    gate_scale = _safe_float(_deep_get(cfg, "enhance241", "c6_gate_scale", default=6.0), 6.0)
    gate_bias = _safe_float(_deep_get(cfg, "enhance241", "c6_gate_bias", default=-0.5), -0.5)

    wrapped = C6GatedBRAInject(
        old,
        channels=channels,
        num_heads=num_heads,
        window_size=window_size,
        topk=topk,
        kv_downsample_mode=kv_mode,
        soft_routing=soft_routing,
        gate_scale=gate_scale,
        gate_bias=gate_bias,
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
        "new_type": "C6GatedBRAInject",
        "channels": int(channels),
        "num_heads": int(num_heads),
        "window_size": int(window_size),
        "topk": int(topk),
        "kv_downsample_mode": str(kv_mode),
        "soft_routing": bool(soft_routing),
        "gate_scale": _to_float(gate_scale),
        "gate_bias": _to_float(gate_bias),
    }
    setattr(yolo_obj, "_enhance241_c6_info", info)
    print(
        f"[enhance241] c6 enabled: patched model.model[{int(p3_idx)}] "
        f"-> C6GatedBRAInject(topk={topk}, window={window_size}, gate_scale={gate_scale:.2f})"
    )

    recorder = get_check_recorder("c6", cfg, patch_info=info)
    if recorder is not None:
        _bind_module_debug(wrapped, recorder, prefix=f"c6.idx{p3_idx}")
        recorder.register_module_params(wrapped, f"c6.idx{p3_idx}")
        try:
            recorder.attach_detect_hooks(detect, detect_idx)
        except Exception as exc:
            recorder.add_note(f"c6_detect_hook_failed:{exc}")

    return yolo_obj
