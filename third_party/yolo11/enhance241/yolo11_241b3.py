from __future__ import annotations

# Path: third_party/yolo11/enhance241/yolo11_241b3.py
# Purpose: enhance241 b3 (NASFPNLite) module + apply hook.

from typing import Any, Dict, List, Optional, Tuple

import torch

from .yolo11_241a3 import _bind_module_debug, _get_module_recorder, _should_capture_delta, get_check_recorder

ENHANCE241_AUDIT_KEYS = ["enhance241_b3"]  # enhance241-audit


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


def _safe_float(x: Any, default: float) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


class _DWSeparableConv(torch.nn.Module):
    """Depthwise-separable 3x3 + pointwise 1x1 (no BN)."""

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


def _zero_init_last_conv(module: torch.nn.Module) -> None:
    convs = [m for m in module.modules() if isinstance(m, torch.nn.Conv2d)]
    if not convs:
        return
    last = convs[-1]
    torch.nn.init.zeros_(last.weight)
    if last.bias is not None:
        torch.nn.init.zeros_(last.bias)


class NASFPNLiteFuse(torch.nn.Module):
    """NASFPNLite fuse: learnable weighted fusion + lightweight refine."""

    def __init__(
        self,
        channels_per_branch: int,
        weight_init: float = 1.0,
        refine: str = "dw",
        upsample_mode: str = "nearest",
        eps: float = 1e-4,
        tag: str = "",
    ) -> None:
        super().__init__()
        self.c = int(channels_per_branch)
        if self.c <= 0:
            raise ValueError(f"channels_per_branch must be positive, got: {channels_per_branch}")
        self.upsample_mode = str(upsample_mode).lower()
        self.eps = float(eps)
        self.tag = str(tag)

        refine = str(refine).lower()
        if refine == "conv":
            self.enhance241_b3_refine = _Conv3x3(self.c)  # enhance241-audit
        else:
            self.enhance241_b3_refine = _DWSeparableConv(self.c)  # enhance241-audit
        _zero_init_last_conv(self.enhance241_b3_refine)

        self.enhance241_b3_weight = torch.nn.Parameter(  # enhance241-audit
            torch.tensor([float(weight_init), float(weight_init)], dtype=torch.float32)
        )
        self.enhance241_b3_alpha = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

    def _maybe_align(self, hi: torch.Tensor, lo: torch.Tensor) -> torch.Tensor:
        if hi.shape[-2:] == lo.shape[-2:]:
            return hi
        if self.upsample_mode == "bilinear":
            return torch.nn.functional.interpolate(hi, size=lo.shape[-2:], mode="bilinear", align_corners=False)
        return torch.nn.functional.interpolate(hi, size=lo.shape[-2:], mode="nearest")

    def _split_input(self, x: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(x, (list, tuple)) and len(x) == 2:
            return x[0], x[1]
        if isinstance(x, torch.Tensor):
            if x.shape[1] != self.c * 2:
                raise ValueError(f"Expected concat tensor with 2C channels, got {x.shape}")
            return torch.split(x, self.c, dim=1)
        raise TypeError(f"NASFPNLiteFuse expects [hi,lo] or concat tensor, got: {type(x)}")

    def forward(self, x: Any) -> torch.Tensor:
        recorder = _get_module_recorder(self)
        prefix = str(getattr(self, "_enhance241_prefix", "b3"))
        step = int(getattr(self, "_enhance241_fwd_step", 0))

        hi, lo = self._split_input(x)
        hi = self._maybe_align(hi, lo)
        if hi.shape[1] != self.c or lo.shape[1] != self.c:
            raise ValueError(f"Channel mismatch: hi={hi.shape} lo={lo.shape} expected C={self.c}")

        if recorder is not None and step == 0:
            recorder.capture_param_before(self, prefix)

        w = torch.softmax(self.enhance241_b3_weight, dim=0)
        if recorder is not None:
            recorder.record_b3_weight(prefix, self.tag, w.detach().float(), step)

        mix = w[0] * hi + w[1] * lo
        delta = self.enhance241_b3_refine(mix - lo)
        alpha = self.enhance241_b3_alpha.to(dtype=lo.dtype, device=lo.device)
        lo_out = lo + alpha * delta
        out = torch.cat((hi, lo_out), dim=1)

        if recorder is not None:
            if step == 0:
                baseline = torch.cat((hi.detach(), lo.detach()), dim=1)
                recorder.record_module_compare(f"{prefix}.fused", patched=out, baseline=baseline, input_tensor=lo)
                recorder.record_b3_feature_relation(prefix, lo, delta)
                recorder.record_scalar_curve(f"{prefix}.alpha", float(alpha.item()), step, max_steps=30)
                recorder.record_a1_payload(
                    f"{prefix}.residual_safe",
                    {
                        "alpha": float(alpha.item()),
                        "var_delta_over_lo": float(delta.detach().float().var(unbiased=False).item())
                        / (float(lo.detach().float().var(unbiased=False).item()) + 1e-12),
                        "cosine_delta_lo": float(
                            torch.nn.functional.cosine_similarity(
                                delta.detach().float().reshape(delta.shape[0], -1),
                                lo.detach().float().reshape(lo.shape[0], -1),
                                dim=1,
                            ).mean().item()
                        ),
                    },
                )
                if out.requires_grad:
                    try:
                        out.register_hook(lambda g, n=f"{prefix}.fused": recorder.record_output_grad(n, g))
                    except Exception:
                        pass
            elif step <= 30:
                recorder.record_scalar_curve(f"{prefix}.alpha", float(alpha.item()), step, max_steps=30)
            if _should_capture_delta(step):
                recorder.capture_param_delta(self, prefix)

        setattr(self, "_enhance241_fwd_step", step + 1)
        return out


def _concat_candidates(seq: Any) -> List[str]:
    cand = []
    for i, layer in enumerate(seq):
        if layer.__class__.__name__ == "Concat":
            cand.append(f"idx={i}, f={getattr(layer, 'f', None)}")
    return cand


def _f_as_list(v: Any) -> List[int]:
    if isinstance(v, int):
        return [int(v)]
    if isinstance(v, (list, tuple)):
        out: List[int] = []
        for x in v:
            try:
                out.append(int(x))
            except Exception:
                continue
        return out
    return []


def _locate_detect(seq: Any) -> Tuple[int, Any]:
    if not seq:
        raise RuntimeError("Empty model sequence.")
    idx = len(seq) - 1
    det = seq[idx]
    if det.__class__.__name__.lower() == "detect":
        return idx, det
    for i, layer in enumerate(seq):
        if layer.__class__.__name__.lower() == "detect":
            return i, layer
    raise RuntimeError("Detect layer not found.")


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
    if not candidates:
        raise RuntimeError(
            f"Unable to locate P5->P4 Concat before idx={p3_concat_idx}. "
            f"Concat candidates: {_concat_candidates(seq)}"
        )
    return candidates[-1]


def _infer_device_dtype(seq: Any, start_idx: int) -> Tuple[Optional[torch.device], Optional[torch.dtype]]:
    for step in range(0, 6):
        for idx in (start_idx + step, start_idx - step):
            if idx < 0 or idx >= len(seq):
                continue
            layer = seq[idx]
            try:
                p = next(layer.parameters())
                return p.device, p.dtype
            except StopIteration:
                continue
            except Exception:
                continue
    return None, None


def _infer_concat_channels(seq: Any, fuse_idx: int) -> int:
    next_layer = seq[fuse_idx + 1] if (fuse_idx + 1) < len(seq) else None
    c_in = None
    try:
        c_in = int(next_layer.cv1.conv.in_channels)  # type: ignore[attr-defined]
    except Exception:
        c_in = None
    if not c_in or c_in % 2 != 0:
        raise RuntimeError(f"Unable to infer concat channels from next layer at idx={fuse_idx+1}. c_in={c_in}")
    return c_in // 2


def apply(model: Any, cfg: Any) -> Any:
    """Apply enhance241 b3 NASFPNLite to P5->P4 and P4->P3 top-down fuses."""

    enable_b3 = bool(_deep_get(cfg, "enhance241", "b3", default=False))
    if not enable_b3:
        return model

    if bool(_deep_get(cfg, "enhance241", "b1", default=False)) or bool(_deep_get(cfg, "enhance241", "b2", default=False)):
        raise RuntimeError("enhance241.b3 conflicts with b1/b2; enable only one B-class module.")

    yolo_obj = model
    if hasattr(model, "model") and hasattr(getattr(model, "model"), "model"):
        det_model = getattr(model, "model")
        seq = getattr(det_model, "model", None)
    else:
        det_model = model
        seq = getattr(det_model, "model", None)
    if seq is None:
        raise RuntimeError("enhance241.b3 requires an ultralytics YOLO/DetectionModel-like object with a .model sequence.")

    prepatched = [i for i, layer in enumerate(seq) if isinstance(layer, NASFPNLiteFuse)]
    if prepatched:
        info = {
            "enabled": True,
            "patched_indices": prepatched,
            "patched_count": len(prepatched),
            "note": "already_patched",
        }
        setattr(yolo_obj, "_enhance241_b3_info", info)

        recorder = get_check_recorder("b3", cfg, patch_info=info)
        if recorder is not None:
            try:
                for idx in prepatched:
                    mod = seq[int(idx)]
                    _bind_module_debug(mod, recorder, prefix=f"b3.idx{int(idx)}.existing")
                    recorder.register_module_params(mod, f"b3.idx{int(idx)}.existing")
                detect_idx, detect_layer = _locate_detect(seq)
                recorder.attach_detect_hooks(detect_layer, detect_idx)
                recorder.maybe_run_val_separability(yolo_obj, cfg)
            except Exception as exc:
                recorder.add_note(f"b3_prepatched_hook_failed:{exc}")

        if len(prepatched) >= 2:
            return yolo_obj
        raise RuntimeError(f"enhance241.b3 expects two NASFPNLiteFuse modules, got {len(prepatched)}")

    p3_concat_idx = _locate_p4_to_p3_fuse(seq)
    p5_concat_idx = _locate_p5_to_p4_fuse(seq, p3_concat_idx)

    enhance_cfg = _deep_get(cfg, "enhance241", default={}) or {}
    if not isinstance(enhance_cfg, dict):
        enhance_cfg = {}
    weight_init = _safe_float(enhance_cfg.get("b3_weight_init", 1.0), 1.0)
    refine = str(enhance_cfg.get("b3_refine", "dw"))
    upsample_mode = str(enhance_cfg.get("b3_upsample", "nearest")).lower()
    eps = _safe_float(enhance_cfg.get("b3_eps", 1e-4), 1e-4)

    patched_indices: List[int] = []
    weights_info: List[Dict[str, Any]] = []
    recorder = get_check_recorder("b3", cfg)

    def _patch_at(idx: int, tag: str) -> None:
        old = seq[idx]
        if isinstance(old, NASFPNLiteFuse):
            patched_indices.append(idx)
            return
        if old.__class__.__name__ != "Concat":
            raise RuntimeError(
                f"Expected Concat at idx={idx} for NASFPNLite({tag}), got {old.__class__.__name__}. "
                f"Concat candidates: {_concat_candidates(seq)}"
            )
        c = _infer_concat_channels(seq, idx)
        fuse = NASFPNLiteFuse(
            channels_per_branch=c,
            weight_init=weight_init,
            refine=refine,
            upsample_mode=upsample_mode,
            eps=eps,
            tag=tag,
        )
        for attr in ("i", "f", "type"):
            if hasattr(old, attr):
                setattr(fuse, attr, getattr(old, attr))
        device, dtype = _infer_device_dtype(seq, idx)
        if device is not None:
            if dtype is not None:
                fuse = fuse.to(device=device, dtype=dtype)
            else:
                fuse = fuse.to(device=device)
        seq[idx] = fuse
        patched_indices.append(idx)
        weight_param = fuse.enhance241_b3_weight
        weights_info.append(
            {
                "idx": idx,
                "tag": tag,
                "weight_init": [float(weight_param[0].item()), float(weight_param[1].item())],
                "requires_grad": bool(weight_param.requires_grad),
                "channels": int(c),
            }
        )

        if recorder is not None:
            try:
                _bind_module_debug(fuse, recorder, prefix=f"b3.idx{idx}.{tag}")
                recorder.register_module_params(fuse, f"b3.idx{idx}.{tag}")
            except Exception as exc:
                recorder.add_note(f"b3_attach_debug_failed:{idx}:{exc}")

    _patch_at(p5_concat_idx, "p5p4")
    _patch_at(p3_concat_idx, "p4p3")

    info = {
        "enabled": True,
        "patched_indices": patched_indices,
        "patched_count": len(patched_indices),
        "concat_candidates": _concat_candidates(seq),
        "weights": weights_info,
        "weight_init": float(weight_init),
        "refine": str(refine),
        "upsample": str(upsample_mode),
        "mode": "residual_safe",
        "alpha_init": 0.0,
    }
    setattr(yolo_obj, "_enhance241_b3_info", info)

    if recorder is not None:
        recorder.set_patch_info(info)
        try:
            detect_idx, detect_layer = _locate_detect(seq)
            recorder.attach_detect_hooks(detect_layer, detect_idx)
        except Exception as exc:
            recorder.add_note(f"b3_detect_hook_failed:{exc}")
        try:
            recorder.maybe_run_val_separability(yolo_obj, cfg)
        except Exception as exc:
            recorder.add_note(f"b3_val_check_failed:{exc}")

    return yolo_obj
