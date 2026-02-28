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
    """Compatibility wrapper with d6-specific config namespace."""


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
