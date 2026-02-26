from __future__ import annotations

# Path: third_party/yolo11/enhance241/yolo11_241c5.py
# Purpose: enhance241 c5 patch (BRA residual-safe injection on P3 path).

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

ENHANCE241_AUDIT_KEYS = ["enhance241_c5"]  # enhance241-audit


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


def _window_partition(x: torch.Tensor, ws: int) -> torch.Tensor:
    b, c, h, w = x.shape
    x = x.view(b, c, h // ws, ws, w // ws, ws)
    x = x.permute(0, 2, 4, 3, 5, 1).contiguous()
    return x.view(b, (h // ws) * (w // ws), ws * ws, c)


def _window_reverse(windows: torch.Tensor, ws: int, h: int, w: int) -> torch.Tensor:
    b, nw, tokens, c = windows.shape
    nh = h // ws
    nw_w = w // ws
    if nw != nh * nw_w or tokens != ws * ws:
        raise ValueError(f"window_reverse mismatch nw={nw}, tokens={tokens}, h={h}, w={w}, ws={ws}")
    x = windows.view(b, nh, nw_w, ws, ws, c)
    x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
    return x.view(b, c, h, w)


class BRACore(torch.nn.Module):
    """Bi-level Routing Attention (lightweight implementation)."""

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

        self.enhance241_c5_qkv = torch.nn.Conv2d(self.channels, self.channels * 3, kernel_size=1, stride=1, padding=0)
        self.enhance241_c5_proj = torch.nn.Conv2d(self.channels, self.channels, kernel_size=1, stride=1, padding=0)
        # Safe-start: BRA residual branch begins near-zero.
        torch.nn.init.zeros_(self.enhance241_c5_proj.weight)
        if self.enhance241_c5_proj.bias is not None:
            torch.nn.init.zeros_(self.enhance241_c5_proj.bias)

    def _prepare(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int, int, int]:
        b, c, h, w = x.shape
        ws = self.window_size
        pad_h = (ws - h % ws) % ws
        pad_w = (ws - w % ws) % ws
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
        hp, wp = x.shape[-2:]
        return x, h, w, hp, wp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_pad, h, w, hp, wp = self._prepare(x)
        qkv = self.enhance241_c5_qkv(x_pad)
        q, k, v = torch.chunk(qkv, 3, dim=1)

        ws = self.window_size
        q_w = _window_partition(q, ws)  # [B, Nw, Tq, C]
        k_w = _window_partition(k, ws)  # [B, Nw, Tk0, C]
        v_w = _window_partition(v, ws)

        if self.kv_downsample_mode == "max":
            k_tok = k_w.amax(dim=2, keepdim=True)
            v_tok = v_w.amax(dim=2, keepdim=True)
        elif self.kv_downsample_mode == "identity":
            k_tok = k_w
            v_tok = v_w
        else:
            # default avg
            k_tok = k_w.mean(dim=2, keepdim=True)
            v_tok = v_w.mean(dim=2, keepdim=True)

        q_mean = q_w.mean(dim=2)  # [B, Nw, C]
        k_mean = k_w.mean(dim=2)  # [B, Nw, C]

        scale_win = 1.0 / math.sqrt(max(1, self.channels))
        routing_logits = torch.matmul(q_mean.float(), k_mean.float().transpose(-1, -2)) * scale_win
        topk = min(self.topk, int(routing_logits.shape[-1]))
        topv, topi = torch.topk(routing_logits, k=topk, dim=-1)  # [B, Nw, K]

        routing_w: Optional[torch.Tensor] = None
        if self.soft_routing:
            topv = topv - topv.amax(dim=-1, keepdim=True)
            routing_w = torch.softmax(topv, dim=-1).to(dtype=q_w.dtype)

        bsz, nwin, tq, c = q_w.shape
        tk = int(k_tok.shape[2])
        out_windows: List[torch.Tensor] = []
        scale_head = 1.0 / math.sqrt(max(1, self.head_dim))

        for b in range(bsz):
            q_b = q_w[b]  # [Nw, Tq, C]
            k_b = k_tok[b]  # [Nw, Tk, C]
            v_b = v_tok[b]  # [Nw, Tk, C]
            idx_b = topi[b]  # [Nw, K]

            # Gather selected windows for each query window.
            k_sel = k_b[idx_b]  # [Nw, K, Tk, C]
            v_sel = v_b[idx_b]  # [Nw, K, Tk, C]

            qh = q_b.view(nwin, tq, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [Nw, H, Tq, D]
            kh = k_sel.view(nwin, topk * tk, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [Nw, H, Tk', D]
            vh = v_sel.view(nwin, topk * tk, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

            logits = torch.matmul(qh.float(), kh.float().transpose(-1, -2)) * scale_head
            logits = logits - logits.amax(dim=-1, keepdim=True)
            attn = torch.softmax(logits, dim=-1).to(dtype=qh.dtype)

            if routing_w is not None:
                rw = routing_w[b].repeat_interleave(tk, dim=-1)  # [Nw, Tk']
                rw = rw.unsqueeze(1).unsqueeze(2)  # [Nw, 1, 1, Tk']
                attn = attn * rw
                attn = attn / attn.sum(dim=-1, keepdim=True).clamp_min(self.eps)

            out_h = torch.matmul(attn.float(), vh.float()).to(dtype=qh.dtype)  # [Nw, H, Tq, D]
            out_b = out_h.permute(0, 2, 1, 3).contiguous().view(nwin, tq, c)  # [Nw, Tq, C]
            out_windows.append(out_b)

        out_w = torch.stack(out_windows, dim=0)  # [B, Nw, Tq, C]
        out = _window_reverse(out_w, ws, hp, wp)
        out = out[..., :h, :w]
        out = self.enhance241_c5_proj(out)
        return out


class C5BRAInject(torch.nn.Module):
    """Residual-safe wrapper: out = base(x) + alpha * BRA(base(x))."""

    def __init__(
        self,
        base_module: torch.nn.Module,
        channels: int,
        num_heads: int = 4,
        window_size: int = 8,
        topk: int = 4,
        kv_downsample_mode: str = "avg",
        soft_routing: bool = True,
        alpha_init: float = 0.05,
        alpha_cap: float = 0.5,
    ) -> None:
        super().__init__()
        self.enhance241_c5_base = base_module
        self.alpha_cap = float(max(1e-6, abs(alpha_cap)))
        self.enhance241_c5_bra = BRACore(
            channels=channels,
            num_heads=num_heads,
            window_size=window_size,
            topk=topk,
            kv_downsample_mode=kv_downsample_mode,
            soft_routing=soft_routing,
        )
        alpha_init = float(max(-self.alpha_cap * 0.95, min(self.alpha_cap * 0.95, alpha_init)))
        alpha_ratio = alpha_init / self.alpha_cap
        alpha_raw = torch.atanh(torch.tensor(alpha_ratio, dtype=torch.float32))
        self.enhance241_c5_alpha = torch.nn.Parameter(alpha_raw)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        recorder = _get_module_recorder(self)
        prefix = str(getattr(self, "_enhance241_prefix", "c5"))
        step = int(getattr(self, "_enhance241_fwd_step", 0))

        if recorder is not None and step == 0:
            recorder.capture_param_before(self, prefix)

        y_base = self.enhance241_c5_base(x)
        delta = self.enhance241_c5_bra(y_base)
        alpha_raw = self.enhance241_c5_alpha.to(dtype=y_base.dtype, device=y_base.device)
        alpha = torch.tanh(alpha_raw) * self.alpha_cap
        out = y_base + alpha * delta

        if recorder is not None:
            if step == 0:
                recorder.record_module_compare(
                    f"{prefix}.p3",
                    patched=out,
                    baseline=y_base,
                    input_tensor=_to_input_tensor(x),
                )
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
    raise RuntimeError("Unable to infer P3 channels for c5.")


def apply(model: Any, cfg: Any) -> Any:
    enable_c5 = bool(_deep_get(cfg, "enhance241", "c5", default=False))
    if not enable_c5:
        return model

    yolo_obj, seq = _extract_model_seq(model)
    if seq is None:
        raise RuntimeError("enhance241.c5 requires YOLO/DetectionModel-like object with .model sequence.")

    p3_idx = _infer_p3_output_index(seq)
    detect_idx, detect = _locate_detect(seq)
    old = seq[p3_idx]

    if isinstance(old, C5BRAInject):
        info = {
            "enabled": True,
            "existing_count": 1,
            "patched_count": 0,
            "p3_index": int(p3_idx),
            "detect_idx": int(detect_idx),
            "note": "already_patched",
        }
        setattr(yolo_obj, "_enhance241_c5_info", info)
        recorder = get_check_recorder("c5", cfg, patch_info=info)
        if recorder is not None:
            try:
                _bind_module_debug(old, recorder, prefix=f"c5.idx{p3_idx}.existing")
                recorder.register_module_params(old, f"c5.idx{p3_idx}.existing")
                recorder.attach_detect_hooks(detect, detect_idx)
                recorder.maybe_run_val_separability(yolo_obj, cfg)
            except Exception as exc:
                recorder.add_note(f"c5_prepatched_hook_failed:{exc}")
        return yolo_obj

    channels = _infer_p3_channels(detect)
    num_heads = _safe_int(_deep_get(cfg, "enhance241", "c5_num_heads", default=4), 4)
    window_size = _safe_int(_deep_get(cfg, "enhance241", "c5_window_size", default=8), 8)
    topk = _safe_int(_deep_get(cfg, "enhance241", "c5_topk", default=4), 4)
    kv_mode = str(_deep_get(cfg, "enhance241", "c5_kv_downsample_mode", default="avg")).lower()
    soft_routing = bool(_deep_get(cfg, "enhance241", "c5_soft_routing", default=True))
    alpha_init = _safe_float(_deep_get(cfg, "enhance241", "c5_alpha_init", default=0.05), 0.05)
    alpha_cap = _safe_float(_deep_get(cfg, "enhance241", "c5_alpha_cap", default=0.5), 0.5)

    wrapped = C5BRAInject(
        old,
        channels=channels,
        num_heads=num_heads,
        window_size=window_size,
        topk=topk,
        kv_downsample_mode=kv_mode,
        soft_routing=soft_routing,
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
        "new_type": "C5BRAInject",
        "channels": int(channels),
        "num_heads": int(num_heads),
        "window_size": int(window_size),
        "topk": int(topk),
        "kv_downsample_mode": str(kv_mode),
        "soft_routing": bool(soft_routing),
        "alpha_init": _to_float(alpha_init),
        "alpha_cap": _to_float(alpha_cap),
        "params_old": int(old_params),
        "params_new": int(new_params),
        "delta_params": int(new_params - old_params),
        "gate1_thresholds": {"cosine_min": 0.98, "var_ratio_min": 0.5},
        "gate2_thresholds": {"delta_l2_min": 0.0, "nan_forbidden": True},
    }
    setattr(yolo_obj, "_enhance241_c5_info", info)

    recorder = get_check_recorder("c5", cfg, patch_info=info)
    if recorder is not None:
        _bind_module_debug(wrapped, recorder, prefix=f"c5.idx{p3_idx}")
        recorder.register_module_params(wrapped, f"c5.idx{p3_idx}")
        try:
            recorder.attach_detect_hooks(detect, detect_idx)
        except Exception as exc:
            recorder.add_note(f"c5_detect_hook_failed:{exc}")
        try:
            recorder.maybe_run_val_separability(yolo_obj, cfg)
        except Exception as exc:
            recorder.add_note(f"c5_val_check_failed:{exc}")

    return yolo_obj
