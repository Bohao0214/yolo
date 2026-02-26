from __future__ import annotations

# Path: third_party/yolo11/enhance241/yolo11_241b21.py
# Purpose: enhance241 b21 patch (P4->P3 only CARAFE residual-safe fusion enhancer).

from typing import Any, Optional, Tuple

from .yolo11_241a3 import _bind_module_debug, _infer_device_dtype, _locate_detect, _to_float, get_check_recorder
from .yolo11_241b7 import (
    CARAFEUpsampleSafe,
    _concat_candidates,
    _deep_get,
    _infer_concat_channels,
    _infer_scale_factor,
    _locate_p4_to_p3_fuse,
    _safe_float,
    _safe_int,
)

ENHANCE241_AUDIT_KEYS = ["enhance241_b21"]  # enhance241-audit


class B21CARAFEUpsampleSafe(CARAFEUpsampleSafe):
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
    enable_b21 = bool(_deep_get(cfg, "enhance241", "b21", default=False))
    if not enable_b21:
        return model

    if any(bool(_deep_get(cfg, "enhance241", k, default=False)) for k in ("b1", "b2", "b3", "b5", "b7", "b9", "b11")):
        raise RuntimeError("enhance241.b21 conflicts with b1/b2/b3/b5/b7/b9/b11; enable only one B-class module.")

    yolo_obj, seq = _extract_model_seq(model)
    if seq is None:
        raise RuntimeError("enhance241.b21 requires YOLO/DetectionModel-like object with .model sequence.")

    prepatched = [i for i, layer in enumerate(seq) if isinstance(layer, B21CARAFEUpsampleSafe)]
    if prepatched:
        info = {
            "enabled": True,
            "existing_count": len(prepatched),
            "patched_count": 0,
            "patched_indices": [int(x) for x in prepatched],
            "note": "already_patched",
        }
        setattr(yolo_obj, "_enhance241_b21_info", info)
        recorder = get_check_recorder("b21", cfg, patch_info=info)
        if recorder is not None:
            try:
                idx0 = int(prepatched[0])
                mod0 = seq[idx0]
                _bind_module_debug(mod0, recorder, prefix=f"b21.idx{idx0}.existing")
                recorder.register_module_params(mod0, f"b21.idx{idx0}.existing")
                detect_idx, detect_layer = _locate_detect(seq)
                recorder.attach_detect_hooks(detect_layer, detect_idx)
                recorder.maybe_run_val_separability(yolo_obj, cfg)
            except Exception as exc:
                recorder.add_note(f"b21_prepatched_hook_failed:{exc}")
        return yolo_obj

    p3_concat_idx = _locate_p4_to_p3_fuse(seq)
    up_idx = int(p3_concat_idx) - 1
    if up_idx < 0:
        raise RuntimeError(f"enhance241.b21 invalid upsample index for concat idx={p3_concat_idx}")
    old = seq[up_idx]
    old_name = old.__class__.__name__.lower()
    if "upsample" not in old_name:
        raise RuntimeError(
            f"enhance241.b21 expected upsample at idx={up_idx} before concat idx={p3_concat_idx}, got {old.__class__.__name__}"
        )

    channels = _infer_concat_channels(seq, p3_concat_idx)
    scale = _infer_scale_factor(old)
    kernel_size = _safe_int(_deep_get(cfg, "enhance241", "b21_kernel_size", default=5), 5)
    compress = _safe_int(_deep_get(cfg, "enhance241", "b21_compress", default=64), 64)
    chunk_channels = _safe_int(_deep_get(cfg, "enhance241", "b21_chunk_channels", default=64), 64)
    alpha_init = _safe_float(_deep_get(cfg, "enhance241", "b21_alpha_init", default=0.02), 0.02)
    alpha_cap = _safe_float(_deep_get(cfg, "enhance241", "b21_alpha_cap", default=0.3), 0.3)

    mod = B21CARAFEUpsampleSafe(
        old,
        channels=channels,
        scale=scale,
        kernel_size=kernel_size,
        compress=compress,
        chunk_channels=chunk_channels,
        alpha_init=alpha_init,
        alpha_cap=alpha_cap,
        tag="b21_p4p3",
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

    info = {
        "enabled": True,
        "existing_count": 0,
        "patched_count": 1,
        "patched_indices": [int(up_idx)],
        "concat_idx": int(p3_concat_idx),
        "concat_candidates": _concat_candidates(seq),
        "kernel_size": int(kernel_size),
        "compress": int(compress),
        "chunk_channels": int(chunk_channels),
        "alpha_init": _to_float(alpha_init),
        "alpha_cap": _to_float(alpha_cap),
        "mode": "carafe_residual_safe_p4p3_only",
    }
    setattr(yolo_obj, "_enhance241_b21_info", info)
    print(
        f"[enhance241] b21 enabled: patched model.model[{int(up_idx)}] "
        f"-> B21CARAFEUpsampleSafe(kernel={kernel_size}, compress={compress}, alpha_init={alpha_init:.3f})"
    )

    recorder = get_check_recorder("b21", cfg, patch_info=info)
    if recorder is not None:
        _bind_module_debug(mod, recorder, prefix=f"b21.idx{int(up_idx)}")
        recorder.register_module_params(mod, f"b21.idx{int(up_idx)}")
        try:
            detect_idx, detect_layer = _locate_detect(seq)
            recorder.attach_detect_hooks(detect_layer, detect_idx)
        except Exception as exc:
            recorder.add_note(f"b21_detect_hook_failed:{exc}")
        try:
            recorder.maybe_run_val_separability(yolo_obj, cfg)
        except Exception as exc:
            recorder.add_note(f"b21_val_check_failed:{exc}")

    return yolo_obj
