from __future__ import annotations

# Path: third_party/yolo11/enhance241/yolo11_241d7.py
# Purpose: enhance241 d7 patch (small-target-only detection head, keep P3 stride-8 branch).

from typing import Any, Dict, List, Optional, Tuple

import torch

from .yolo11_241a3 import (
    _bind_module_debug,
    _f_as_list,
    _locate_detect,
    get_check_recorder,
)

ENHANCE241_AUDIT_KEYS = ["enhance241_d7"]  # enhance241-audit


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


def _extract_model_seq(model: Any) -> Tuple[Any, Any, Optional[Any]]:
    yolo_obj = model
    if hasattr(model, "model") and hasattr(getattr(model, "model"), "model"):
        det_model = getattr(model, "model")
        seq = getattr(det_model, "model", None)
    else:
        det_model = model
        seq = getattr(det_model, "model", None)
    return yolo_obj, det_model, seq


def _keep_first_modulelist(module_list: Any) -> Any:
    if not isinstance(module_list, torch.nn.ModuleList):
        return module_list
    if len(module_list) <= 1:
        return module_list
    return torch.nn.ModuleList([module_list[0]])


def _head_stride_list(stride: Any) -> List[float]:
    if isinstance(stride, torch.Tensor):
        return [float(x) for x in stride.detach().float().cpu().tolist()]
    if isinstance(stride, (list, tuple)):
        return [float(x) for x in stride]
    try:
        return [float(stride)]
    except Exception:
        return []


def _d7_stride_tensor_for_loss_compat(stride: Any) -> torch.Tensor:
    """Keep at least 2 stride values for TAL compatibility (tal.py uses stride[1])."""

    vals = _head_stride_list(stride)
    if len(vals) >= 2:
        return torch.tensor([vals[0], vals[1]], dtype=torch.float32)
    if len(vals) == 1:
        return torch.tensor([vals[0], vals[0] * 2.0], dtype=torch.float32)
    return torch.tensor([8.0, 16.0], dtype=torch.float32)


def _shift_keep_head_cls_bias(detect: Any, delta: float) -> bool:
    ok = False

    def _shift_from_head(head: Any) -> bool:
        try:
            if isinstance(head, torch.nn.Sequential) and len(head) > 0:
                last = head[-1]
            else:
                last = head
            if isinstance(last, torch.nn.Conv2d) and last.bias is not None:
                with torch.no_grad():
                    last.bias.add_(float(delta))
                return True
        except Exception:
            return False
        return False

    try:
        if hasattr(detect, "cv3") and len(detect.cv3) > 0:
            ok = _shift_from_head(detect.cv3[0]) or ok
    except Exception:
        pass

    try:
        one2one_cv3 = getattr(detect, "one2one_cv3", None)
        if isinstance(one2one_cv3, torch.nn.ModuleList) and len(one2one_cv3) > 0:
            ok = _shift_from_head(one2one_cv3[0]) or ok
    except Exception:
        pass

    return ok


def apply(model: Any, cfg: Any) -> Any:
    enable_d7 = bool(_deep_get(cfg, "enhance241", "d7", default=False))
    if not enable_d7:
        return model

    if any(bool(_deep_get(cfg, "enhance241", k, default=False)) for k in ("d1", "d3", "d5", "d9")):
        raise RuntimeError("enhance241.d7 conflicts with d1/d3/d5/d9; enable only one D-class module.")

    yolo_obj, det_model, seq = _extract_model_seq(model)
    if seq is None:
        raise RuntimeError("enhance241.d7 requires YOLO/DetectionModel-like object with .model sequence.")

    detect_idx, detect = _locate_detect(seq)
    f_before = _f_as_list(getattr(detect, "f", []))
    heads_before = int(getattr(detect, "nl", len(f_before) if f_before else 0))
    stride_before = _head_stride_list(getattr(detect, "stride", []))

    already_small_only = heads_before == 1 and len(f_before) == 1
    if already_small_only:
        stride_after = _d7_stride_tensor_for_loss_compat(getattr(detect, "stride", []))
        detect.stride = stride_after
        info = {
            "enabled": True,
            "patched_count": 0,
            "existing_count": 1,
            "detect_idx": int(detect_idx),
            "detect_heads_before": int(heads_before),
            "detect_heads_after": int(heads_before),
            "detect_f_before": [int(x) for x in f_before],
            "detect_f_after": [int(x) for x in f_before],
            "stride_before": stride_before,
            "stride_after": _head_stride_list(stride_after),
            "head_keep": "p3_only",
            "note": "already_patched",
            "tal_compat_stride_len": int(len(_head_stride_list(stride_after))),
        }
        setattr(yolo_obj, "_enhance241_d7_info", info)
        recorder = get_check_recorder("d7", cfg, patch_info=info)
        if recorder is not None:
            try:
                _bind_module_debug(detect, recorder, prefix=f"d7.detect.idx{detect_idx}.existing")
                recorder.register_module_params(detect, f"d7.detect.idx{detect_idx}.existing")
                recorder.attach_detect_hooks(detect, detect_idx)
                recorder.maybe_run_val_separability(yolo_obj, cfg)
            except Exception as exc:
                recorder.add_note(f"d7_prepatched_hook_failed:{exc}")
        return yolo_obj

    if not f_before:
        raise RuntimeError(f"enhance241.d7 invalid Detect.f={getattr(detect, 'f', None)}")

    keep_idx = 0  # keep smallest stride branch (P3 in this codebase)
    keep_f = int(f_before[keep_idx])

    detect.f = [keep_f]
    detect.cv2 = _keep_first_modulelist(getattr(detect, "cv2", None))
    detect.cv3 = _keep_first_modulelist(getattr(detect, "cv3", None))

    one2one_cv2 = getattr(detect, "one2one_cv2", None)
    if isinstance(one2one_cv2, torch.nn.ModuleList) and len(one2one_cv2) > 1:
        detect.one2one_cv2 = torch.nn.ModuleList([one2one_cv2[0]])

    one2one_cv3 = getattr(detect, "one2one_cv3", None)
    if isinstance(one2one_cv3, torch.nn.ModuleList) and len(one2one_cv3) > 1:
        detect.one2one_cv3 = torch.nn.ModuleList([one2one_cv3[0]])

    detect.nl = 1
    detect.stride = _d7_stride_tensor_for_loss_compat(getattr(detect, "stride", []))

    try:
        detect.bias_init()
    except Exception:
        pass

    d7_cls_bias_shift = float(_deep_get(cfg, "enhance241", "d7_cls_bias_shift", default=-0.25))
    cls_bias_shift_ok = _shift_keep_head_cls_bias(detect, d7_cls_bias_shift) if abs(d7_cls_bias_shift) > 1e-12 else False

    # Keep full neck graph unchanged for safety; only Detect branches are pruned.
    info = {
        "enabled": True,
        "patched_count": 1,
        "existing_count": 0,
        "detect_idx": int(detect_idx),
        "detect_heads_before": int(heads_before),
        "detect_heads_after": int(getattr(detect, "nl", 1)),
        "detect_f_before": [int(x) for x in f_before],
        "detect_f_after": [int(x) for x in _f_as_list(getattr(detect, "f", []))],
        "stride_before": stride_before,
        "stride_after": _head_stride_list(getattr(detect, "stride", [])),
        "head_keep": "p3_only",
        "dropped_heads": max(0, int(heads_before) - 1),
        "loss_scope": "only_p3_branch",
        "tal_compat_stride_len": int(len(_head_stride_list(getattr(detect, "stride", [])))),
        "d7_cls_bias_shift": float(d7_cls_bias_shift),
        "d7_cls_bias_shift_applied": bool(cls_bias_shift_ok),
    }
    setattr(yolo_obj, "_enhance241_d7_info", info)

    recorder = get_check_recorder("d7", cfg, patch_info=info)
    if recorder is not None:
        try:
            _bind_module_debug(detect, recorder, prefix=f"d7.detect.idx{detect_idx}")
            recorder.register_module_params(detect, f"d7.detect.idx{detect_idx}")
            recorder.attach_detect_hooks(detect, detect_idx)
            recorder.maybe_run_val_separability(yolo_obj, cfg)
        except Exception as exc:
            recorder.add_note(f"d7_detect_hook_failed:{exc}")

    return yolo_obj
