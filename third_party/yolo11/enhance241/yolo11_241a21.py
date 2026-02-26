from __future__ import annotations

# Path: third_party/yolo11/enhance241/yolo11_241a21.py
# Purpose: enhance241 a21 patch (a3-primary + optional weak a7 delta).

from typing import Any, Optional, Tuple

from .yolo11_241a3 import (
    _bind_module_debug,
    _find_first_conv,
    _find_stride_conv,
    _infer_device_dtype,
    _locate_detect,
    _locate_p4_to_p3_fuse,
    _module_stride,
    _to_float,
    get_check_recorder,
)
from .yolo11_241a4 import A4DualDeltaSafe, _deep_get, _safe_float, _safe_int

ENHANCE241_AUDIT_KEYS = ["enhance241_a21"]  # enhance241-audit


class A21DualDeltaSafe(A4DualDeltaSafe):
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
    enable_a21 = bool(_deep_get(cfg, "enhance241", "a21", default=False))
    if not enable_a21:
        return model

    if any(bool(_deep_get(cfg, "enhance241", k, default=False)) for k in ("a3", "a4", "a5", "a7", "a9", "a11")):
        raise RuntimeError("enhance241.a21 conflicts with a3/a4/a5/a7/a9/a11; enable only one A-class module.")

    yolo_obj, seq = _extract_model_seq(model)
    if seq is None:
        raise RuntimeError("enhance241.a21 requires YOLO/DetectionModel-like object with .model sequence.")

    prepatched = [i for i, layer in enumerate(seq) if isinstance(layer, A21DualDeltaSafe)]
    if prepatched:
        info = {
            "enabled": True,
            "existing_count": len(prepatched),
            "patched_count": 0,
            "patched_indices": [int(x) for x in prepatched],
            "note": "already_patched",
        }
        setattr(yolo_obj, "_enhance241_a21_info", info)
        recorder = get_check_recorder("a21", cfg, patch_info=info)
        if recorder is not None:
            try:
                idx0 = int(prepatched[0])
                mod0 = seq[idx0]
                _bind_module_debug(mod0, recorder, prefix=f"a21.idx{idx0}.existing")
                recorder.register_module_params(mod0, f"a21.idx{idx0}.existing")
                detect_idx, detect_layer = _locate_detect(seq)
                recorder.attach_detect_hooks(detect_layer, detect_idx)
                recorder.maybe_run_val_separability(yolo_obj, cfg)
            except Exception as exc:
                recorder.add_note(f"a21_prepatched_hook_failed:{exc}")
        return yolo_obj

    _, p3_idx = _locate_p4_to_p3_fuse(seq)
    ds_idx: Optional[int] = None
    for i in range(p3_idx, -1, -1):
        if _module_stride(seq[i]) == 2:
            ds_idx = i
            break
    if ds_idx is None:
        raise RuntimeError("enhance241.a21: unable to locate stride=2 downsample before P3 output.")

    old = seq[ds_idx]
    conv = _find_stride_conv(old, stride=2) or _find_first_conv(old)
    if conv is None:
        raise RuntimeError(f"enhance241.a21: unable to infer in/out channels at idx={ds_idx}.")

    in_ch = int(conv.in_channels)
    out_ch = int(conv.out_channels)
    a21_pre_div = _safe_int(_deep_get(cfg, "enhance241", "a21_pre_div", default=4), 4)
    a21_refine = str(_deep_get(cfg, "enhance241", "a21_refine", default="dw"))
    a21_order = _safe_int(_deep_get(cfg, "enhance241", "a21_order", default=3), 3)
    a21_alpha1_init = _safe_float(_deep_get(cfg, "enhance241", "a21_alpha1_init", default=0.05), 0.05)
    a21_alpha1_cap = _safe_float(_deep_get(cfg, "enhance241", "a21_alpha1_cap", default=0.5), 0.5)
    a21_alpha2_init = _safe_float(_deep_get(cfg, "enhance241", "a21_alpha2_init", default=0.0), 0.0)
    a21_alpha2_cap = _safe_float(_deep_get(cfg, "enhance241", "a21_alpha2_cap", default=0.5), 0.5)

    fuse = A21DualDeltaSafe(
        base_downsample=old,
        in_ch=in_ch,
        out_ch=out_ch,
        a3_pre_div=a21_pre_div,
        a4_refine=a21_refine,
        a4_order=a21_order,
        a4_alpha1_init=a21_alpha1_init,
        a4_alpha1_cap=a21_alpha1_cap,
        a4_alpha2_init=a21_alpha2_init,
        a4_alpha2_cap=a21_alpha2_cap,
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
        "new_type": "A21DualDeltaSafe",
        "a21_refine": str(a21_refine),
        "a21_order": int(a21_order),
        "alpha1_init": _to_float(a21_alpha1_init),
        "alpha1_cap": _to_float(a21_alpha1_cap),
        "alpha2_init": _to_float(a21_alpha2_init),
        "alpha2_cap": _to_float(a21_alpha2_cap),
    }
    setattr(yolo_obj, "_enhance241_a21_info", info)
    print(
        f"[enhance241] a21 enabled: patched model.model[{int(ds_idx)}] "
        f"-> A21DualDeltaSafe(alpha1_init={a21_alpha1_init:.3f}, alpha2_init={a21_alpha2_init:.3f})"
    )

    recorder = get_check_recorder("a21", cfg, patch_info=info)
    if recorder is not None:
        _bind_module_debug(fuse, recorder, prefix=f"a21.idx{int(ds_idx)}")
        recorder.register_module_params(fuse, f"a21.idx{int(ds_idx)}")
        try:
            detect_idx, detect_layer = _locate_detect(seq)
            recorder.attach_detect_hooks(detect_layer, detect_idx)
        except Exception as exc:
            recorder.add_note(f"a21_detect_hook_failed:{exc}")
        try:
            recorder.maybe_run_val_separability(yolo_obj, cfg)
        except Exception as exc:
            recorder.add_note(f"a21_val_check_failed:{exc}")

    return yolo_obj
