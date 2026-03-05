from __future__ import annotations

# Path: third_party/yolo11/enhance241/yolo11_241d6.py
# Purpose: enhance241 d6 (scale-sensitive score calibration; d11-derived isolated variant).

from typing import Any, List

import torch

from .yolo11_241a3 import _bind_module_debug, _locate_detect, _to_float, get_check_recorder
from .yolo11_241d11 import (
    D11ClsScoreCalib,
    _cast_like_module,
    _deep_get,
    _extract_model_seq,
    _infer_head_stride,
    _parse_apply_to,
    _safe_bool,
    _safe_float,
    _target_head_indices,
)

ENHANCE241_AUDIT_KEYS = ["enhance241_d6"]  # enhance241-audit


class D6ScaleAwareCalib(D11ClsScoreCalib):
    """Compatibility wrapper with d6-specific config namespace and linear scale bias."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        recorder = getattr(self, "_enhance241_recorder", None)
        prefix = str(getattr(self, "_enhance241_prefix", "d6"))
        step = int(getattr(self, "_enhance241_fwd_step", 0))

        if recorder is not None and step == 0:
            recorder.capture_param_before(self, prefix)

        logits = self.enhance241_d11_base_cls(x)
        t = torch.clamp(self.enhance241_d11_temp, min=self.t_min, max=self.t_max).to(dtype=logits.dtype, device=logits.device)
        b = self.enhance241_d11_bias.to(dtype=logits.dtype, device=logits.device)
        alpha_raw = self.enhance241_d11_alpha_raw.to(dtype=logits.dtype, device=logits.device)
        alpha = torch.tanh(alpha_raw) * self.alpha_cap
        gamma = logits.new_tensor(self.head_stride * self.stride_gamma_mul)
        linear_limit = max(float(self.scale_lambda), 1e-6)
        ratio = torch.clamp(logits.new_tensor(1.0) - (gamma / linear_limit), min=0.0)
        scale_bias = logits.new_tensor(self.scale_beta) * ratio
        logits_calib = logits / t + (b + scale_bias)
        if self.score_domain:
            score_base = torch.sigmoid(logits)
            score_calib = torch.sigmoid(logits_calib)
            score_out = score_base + alpha * (score_calib - score_base)
            score_out = torch.clamp(score_out, min=1e-5, max=1.0 - 1e-5)
            out = torch.log(score_out / (1.0 - score_out))
        else:
            out = logits + alpha * (logits_calib - logits)

        if recorder is not None:
            recorder.record_scalar_curve(f"{prefix}.temperature", float(t.item()), step, max_steps=100)
            recorder.record_scalar_curve(f"{prefix}.bias", float(b.item()), step, max_steps=100)
            recorder.record_scalar_curve(f"{prefix}.alpha", float(alpha.item()), step, max_steps=100)
            recorder.record_scalar_curve(f"{prefix}.scale_bias", float(scale_bias.item()), step, max_steps=100)
            if step == 0:
                recorder.record_a1_payload(
                    f"{prefix}.cfg",
                    {
                        "temp_init": _to_float(self.enhance241_d11_temp.detach().item()),
                        "temp_min": _to_float(self.t_min),
                        "temp_max": _to_float(self.t_max),
                        "bias_init": _to_float(self.enhance241_d11_bias.detach().item()),
                        "head_stride": _to_float(self.head_stride),
                        "stride_gamma_mul": _to_float(self.stride_gamma_mul),
                        "gamma_proxy": _to_float(gamma.item()),
                        "scale_beta": _to_float(self.scale_beta),
                        "scale_lambda": _to_float(self.scale_lambda),
                        "scale_threshold": _to_float(self.scale_threshold),
                        "linear_ratio": _to_float(ratio.item()),
                        "scale_bias": _to_float(scale_bias.item()),
                        "score_domain": bool(self.score_domain),
                        "alpha": _to_float(alpha.item()),
                        "alpha_cap": _to_float(self.alpha_cap),
                    },
                )
                if out.requires_grad:
                    try:
                        out.register_hook(lambda g, n=f"{prefix}.logits": recorder.record_output_grad(n, g))
                    except Exception:
                        pass
            if hasattr(recorder, "capture_param_delta"):
                from .yolo11_241a3 import _should_capture_delta

                if _should_capture_delta(step):
                    recorder.capture_param_delta(self, prefix)

        setattr(self, "_enhance241_fwd_step", step + 1)
        return out


def apply(model: Any, cfg: Any) -> Any:
    enable_d6 = bool(_deep_get(cfg, "enhance241", "d6", default=False))
    if not enable_d6:
        return model

    yolo_obj, seq = _extract_model_seq(model)
    if seq is None:
        raise RuntimeError("enhance241.d6 requires YOLO/DetectionModel-like object with .model sequence.")

    detect_idx, detect = _locate_detect(seq)
    cls_heads = getattr(detect, "cv3", None)
    if cls_heads is None or not isinstance(cls_heads, torch.nn.ModuleList) or len(cls_heads) < 1:
        raise RuntimeError("enhance241.d6 expects Detect.cv3 ModuleList with at least one head.")

    apply_to = _parse_apply_to(_deep_get(cfg, "enhance241", "d6_apply_to", default=["P3", "P2_new"]))
    target_heads = _target_head_indices(detect, apply_to)
    if not target_heads:
        raise RuntimeError(f"enhance241.d6 apply_to={apply_to} matched no valid detect heads.")

    temp_init = _safe_float(_deep_get(cfg, "enhance241", "d6_temp_init", default=1.0), 1.0)
    t_min = _safe_float(_deep_get(cfg, "enhance241", "d6_temp_min", default=0.5), 0.5)
    t_max = _safe_float(_deep_get(cfg, "enhance241", "d6_temp_max", default=4.0), 4.0)
    scale_beta = _safe_float(_deep_get(cfg, "enhance241", "d6_scale_beta", default=0.20), 0.20)
    scale_lambda = _safe_float(_deep_get(cfg, "enhance241", "d6_scale_lambda", default=32.0), 32.0)
    scale_threshold = _safe_float(_deep_get(cfg, "enhance241", "d6_scale_threshold", default=64.0), 64.0)
    stride_gamma_mul = _safe_float(_deep_get(cfg, "enhance241", "d6_stride_gamma_mul", default=4.0), 4.0)
    score_domain = _safe_bool(_deep_get(cfg, "enhance241", "d6_score_domain", default=True), True)
    alpha_init = _safe_float(_deep_get(cfg, "enhance241", "d6_alpha_init", default=0.05), 0.05)
    alpha_cap = _safe_float(_deep_get(cfg, "enhance241", "d6_alpha_cap", default=0.5), 0.5)

    patched_heads: List[int] = []
    existing = 0
    recorder = get_check_recorder("d6", cfg)
    for head_idx in target_heads:
        old_cls = cls_heads[head_idx]
        if isinstance(old_cls, D6ScaleAwareCalib):
            existing += 1
            continue

        wrapped = D6ScaleAwareCalib(
            old_cls,
            temp_init=temp_init,
            t_min=t_min,
            t_max=t_max,
            bias_init=0.0,
            head_stride=_infer_head_stride(detect, head_idx=head_idx, default=8.0),
            scale_beta=scale_beta,
            scale_lambda=scale_lambda,
            scale_threshold=scale_threshold,
            stride_gamma_mul=stride_gamma_mul,
            score_domain=score_domain,
            alpha_init=alpha_init,
            alpha_cap=alpha_cap,
        )
        wrapped = _cast_like_module(old_cls, wrapped)
        cls_heads[head_idx] = wrapped
        patched_heads.append(int(head_idx))

        if recorder is not None:
            try:
                _bind_module_debug(wrapped, recorder, prefix=f"d6.detect{detect_idx}.head{head_idx}")
                recorder.register_module_params(wrapped, f"d6.detect{detect_idx}.head{head_idx}")
            except Exception as exc:
                recorder.add_note(f"d6_attach_debug_failed:{head_idx}:{exc}")

        one2one_cv3 = getattr(detect, "one2one_cv3", None)
        if isinstance(one2one_cv3, torch.nn.ModuleList) and head_idx < len(one2one_cv3):
            old_o2o = one2one_cv3[head_idx]
            if not isinstance(old_o2o, D6ScaleAwareCalib):
                wrapped_o2o = D6ScaleAwareCalib(
                    old_o2o,
                    temp_init=temp_init,
                    t_min=t_min,
                    t_max=t_max,
                    bias_init=0.0,
                    head_stride=_infer_head_stride(detect, head_idx=head_idx, default=8.0),
                    scale_beta=scale_beta,
                    scale_lambda=scale_lambda,
                    scale_threshold=scale_threshold,
                    stride_gamma_mul=stride_gamma_mul,
                    score_domain=score_domain,
                    alpha_init=alpha_init,
                    alpha_cap=alpha_cap,
                )
                one2one_cv3[head_idx] = _cast_like_module(old_o2o, wrapped_o2o)

    info = {
        "enabled": True,
        "existing_count": int(existing),
        "patched_count": int(len(patched_heads)),
        "detect_idx": int(detect_idx),
        "patched_heads": [int(x) for x in patched_heads],
        "target_heads": [int(x) for x in target_heads],
        "temp_init": _to_float(temp_init),
        "temp_range": [_to_float(t_min), _to_float(t_max)],
        "scale_beta": _to_float(scale_beta),
        "scale_lambda": _to_float(scale_lambda),
        "scale_threshold": _to_float(scale_threshold),
        "stride_gamma_mul": _to_float(stride_gamma_mul),
        "score_domain": bool(score_domain),
        "alpha_init": _to_float(alpha_init),
        "alpha_cap": _to_float(alpha_cap),
    }
    setattr(yolo_obj, "_enhance241_d6_info", info)
    print(
        f"[enhance241] d6 enabled: patched cls heads={patched_heads or 'none(new)'} "
        f"target_heads={target_heads} beta={scale_beta:.3f} lambda={scale_lambda:.1f}"
    )

    if recorder is not None:
        recorder.set_patch_info(info)
        try:
            recorder.attach_detect_hooks(detect, detect_idx)
        except Exception as exc:
            recorder.add_note(f"d6_detect_hook_failed:{exc}")

    return yolo_obj
