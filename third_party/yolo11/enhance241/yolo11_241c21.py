from __future__ import annotations

# Path: third_party/yolo11/enhance241/yolo11_241c21.py
# Purpose: enhance241 c21 patch (c5-primary with light c11 guardrail).

from typing import Any, Optional, Tuple

from .yolo11_241a3 import _bind_module_debug, _infer_device_dtype, _locate_detect, _to_float, get_check_recorder
from .yolo11_241c4 import (
    C4C5C11Inject,
    _deep_get,
    _infer_p3_channels,
    _infer_p3_output_index,
    _safe_float,
    _safe_int,
)

ENHANCE241_AUDIT_KEYS = ["enhance241_c21"]  # enhance241-audit


class C21C5C11Inject(C4C5C11Inject):
    pass


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
    enable_c21 = bool(_deep_get(cfg, "enhance241", "c21", default=False))
    if not enable_c21:
        return model

    if any(bool(_deep_get(cfg, "enhance241", k, default=False)) for k in ("c4", "c5", "c7", "c9", "c11")):
        raise RuntimeError("enhance241.c21 conflicts with c4/c5/c7/c9/c11; enable only one C-class module.")

    yolo_obj, seq = _extract_model_seq(model)
    if seq is None:
        raise RuntimeError("enhance241.c21 requires YOLO/DetectionModel-like object with .model sequence.")

    p3_idx = _infer_p3_output_index(seq)
    detect_idx, detect = _locate_detect(seq)
    old = seq[p3_idx]

    if isinstance(old, C21C5C11Inject):
        info = {
            "enabled": True,
            "existing_count": 1,
            "patched_count": 0,
            "p3_index": int(p3_idx),
            "detect_idx": int(detect_idx),
            "note": "already_patched",
        }
        setattr(yolo_obj, "_enhance241_c21_info", info)
        recorder = get_check_recorder("c21", cfg, patch_info=info)
        if recorder is not None:
            try:
                _bind_module_debug(old, recorder, prefix=f"c21.idx{p3_idx}.existing")
                recorder.register_module_params(old, f"c21.idx{p3_idx}.existing")
                recorder.attach_detect_hooks(detect, detect_idx)
                recorder.maybe_run_val_separability(yolo_obj, cfg)
            except Exception as exc:
                recorder.add_note(f"c21_prepatched_hook_failed:{exc}")
        return yolo_obj

    channels = _infer_p3_channels(detect)
    c21_topk = _safe_int(_deep_get(cfg, "enhance241", "c21_topk", default=4), 4)
    c21_window = _safe_int(_deep_get(cfg, "enhance241", "c21_window_size", default=8), 8)
    c21_heads = _safe_int(_deep_get(cfg, "enhance241", "c21_num_heads", default=4), 4)
    c21_kv = str(_deep_get(cfg, "enhance241", "c21_kv_downsample_mode", default="avg"))
    c21_soft = bool(_deep_get(cfg, "enhance241", "c21_soft_routing", default=True))
    c21_gate_mode = str(_deep_get(cfg, "enhance241", "c21_gate_mode", default="se"))
    c21_gate_reduction = _safe_int(_deep_get(cfg, "enhance241", "c21_gate_reduction", default=16), 16)
    c21_alpha5_init = _safe_float(_deep_get(cfg, "enhance241", "c21_alpha5_init", default=0.05), 0.05)
    c21_alpha5_cap = _safe_float(_deep_get(cfg, "enhance241", "c21_alpha5_cap", default=0.5), 0.5)
    c21_alpha11_init = _safe_float(_deep_get(cfg, "enhance241", "c21_alpha11_init", default=0.01), 0.01)
    c21_alpha11_cap = _safe_float(_deep_get(cfg, "enhance241", "c21_alpha11_cap", default=0.5), 0.5)

    wrapped = C21C5C11Inject(
        base_module=old,
        channels=channels,
        c5_topk=c21_topk,
        c5_window_size=c21_window,
        c5_num_heads=c21_heads,
        c5_kv_downsample_mode=c21_kv,
        c5_soft_routing=c21_soft,
        c4_gate_mode=c21_gate_mode,
        c4_gate_reduction=c21_gate_reduction,
        alpha5_init=c21_alpha5_init,
        alpha5_cap=c21_alpha5_cap,
        alpha11_init=c21_alpha11_init,
        alpha11_cap=c21_alpha11_cap,
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
        "new_type": "C21C5C11Inject",
        "c21_topk": int(c21_topk),
        "c21_window_size": int(c21_window),
        "c21_num_heads": int(c21_heads),
        "c21_kv_downsample_mode": str(c21_kv),
        "c21_soft_routing": bool(c21_soft),
        "c21_gate_mode": str(c21_gate_mode),
        "c21_gate_reduction": int(c21_gate_reduction),
        "alpha5_init": _to_float(c21_alpha5_init),
        "alpha11_init": _to_float(c21_alpha11_init),
    }
    setattr(yolo_obj, "_enhance241_c21_info", info)
    print(
        f"[enhance241] c21 enabled: patched model.model[{int(p3_idx)}] "
        f"-> C21C5C11Inject(topk={c21_topk}, window={c21_window}, gate={c21_gate_mode})"
    )

    recorder = get_check_recorder("c21", cfg, patch_info=info)
    if recorder is not None:
        _bind_module_debug(wrapped, recorder, prefix=f"c21.idx{p3_idx}")
        recorder.register_module_params(wrapped, f"c21.idx{p3_idx}")
        try:
            recorder.attach_detect_hooks(detect, detect_idx)
            recorder.maybe_run_val_separability(yolo_obj, cfg)
        except Exception as exc:
            recorder.add_note(f"c21_hook_failed:{exc}")

    return yolo_obj
