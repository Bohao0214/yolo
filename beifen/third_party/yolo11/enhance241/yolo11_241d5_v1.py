from __future__ import annotations

# Path: third_party/yolo11/enhance241/yolo11_241d5.py
# Purpose: enhance241 d5 patch (add P6 stride=64 head to Detect).

import copy
from typing import Any, Dict, Optional, Tuple

import torch

from .yolo11_241a3 import (
    _bind_module_debug,
    _infer_device_dtype,
    _locate_detect,
    get_check_recorder,
)

ENHANCE241_AUDIT_KEYS = ["enhance241_d5"]  # enhance241-audit


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


class P6Downsample(torch.nn.Module):
    """Lightweight P5->P6 downsample block (stride=2)."""

    def __init__(self, channels: int, mode: str = "dw") -> None:
        super().__init__()
        c = int(channels)
        mode = str(mode).lower()
        if mode == "conv":
            self.conv = torch.nn.Conv2d(c, c, kernel_size=3, stride=2, padding=1, bias=True)
            self.act = torch.nn.SiLU(inplace=True)
            self.dw = None
            self.pw = None
        else:
            self.dw = torch.nn.Conv2d(c, c, kernel_size=3, stride=2, padding=1, groups=c, bias=True)
            self.act = torch.nn.SiLU(inplace=True)
            self.pw = torch.nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0, bias=True)
            self.conv = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.conv is not None:
            return self.act(self.conv(x))
        return self.act(self.pw(self.act(self.dw(x))))


def _extract_model_seq(model: Any) -> Tuple[Any, Any, Optional[Any]]:
    yolo_obj = model
    if hasattr(model, "model") and hasattr(getattr(model, "model"), "model"):
        det_model = getattr(model, "model")
        seq = getattr(det_model, "model", None)
    else:
        det_model = model
        seq = getattr(det_model, "model", None)
    return yolo_obj, det_model, seq


def _infer_p5_channels(detect: Any) -> int:
    try:
        return int(detect.cv2[-1][0].conv.in_channels)  # type: ignore[index]
    except Exception:
        pass
    try:
        last = detect.cv3[-1]  # type: ignore[index]
        for m in last.modules():
            if isinstance(m, torch.nn.Conv2d):
                return int(m.in_channels)
    except Exception:
        pass
    raise RuntimeError("Unable to infer P5 channels for d5.")


def _stride_with_p6(old_stride: Any) -> torch.Tensor:
    if isinstance(old_stride, torch.Tensor) and old_stride.numel() > 0:
        s = old_stride.detach().float().cpu()
        return torch.cat([s, s[-1:].clone() * 2.0], dim=0)
    if isinstance(old_stride, (list, tuple)) and len(old_stride) > 0:
        vals = [float(x) for x in old_stride]
        vals.append(vals[-1] * 2.0)
        return torch.tensor(vals, dtype=torch.float32)
    return torch.tensor([8.0, 16.0, 32.0, 64.0], dtype=torch.float32)


def apply(model: Any, cfg: Any) -> Any:
    enable_d5 = bool(_deep_get(cfg, "enhance241", "d5", default=False))
    if not enable_d5:
        return model

    yolo_obj, det_model, seq = _extract_model_seq(model)
    if seq is None:
        raise RuntimeError("enhance241.d5 requires YOLO/DetectionModel-like object with .model sequence.")

    detect_idx, detect = _locate_detect(seq)
    old_f_raw = getattr(detect, "f", [])
    old_f = [int(x) for x in (old_f_raw if isinstance(old_f_raw, (list, tuple)) else [old_f_raw])]

    prepatched = bool(len(old_f) >= 4 and getattr(detect, "nl", len(old_f)) >= 4)
    if prepatched:
        info = {
            "enabled": True,
            "patched_count": 0,
            "existing_count": 1,
            "detect_idx": int(detect_idx),
            "detect_heads_before": int(getattr(detect, "nl", len(old_f))),
            "detect_heads_after": int(getattr(detect, "nl", len(old_f))),
            "detect_f_before": old_f,
            "detect_f_after": old_f,
            "stride_after": [float(x) for x in getattr(detect, "stride", [])] if hasattr(detect, "stride") else [],
            "note": "already_patched",
        }
        setattr(yolo_obj, "_enhance241_d5_info", info)
        recorder = get_check_recorder("d5", cfg, patch_info=info)
        if recorder is not None:
            try:
                recorder.attach_detect_hooks(detect, detect_idx)
                recorder.maybe_run_val_separability(yolo_obj, cfg)
            except Exception as exc:
                recorder.add_note(f"d5_prepatched_hook_failed:{exc}")
        return yolo_obj

    if len(old_f) < 3:
        raise RuntimeError(f"enhance241.d5 expects Detect with >=3 heads, got f={old_f}")

    p5_idx = int(old_f[-1])
    if not (0 <= p5_idx < detect_idx):
        raise RuntimeError(f"enhance241.d5 invalid P5 source index {p5_idx} for detect_idx={detect_idx}")

    d5_mode = str(_deep_get(cfg, "enhance241", "d5_downsample", default="dw"))
    channels = _infer_p5_channels(detect)
    p6_down = P6Downsample(channels=channels, mode=d5_mode)
    p6_down.f = p5_idx
    p6_down.i = detect_idx
    p6_down.type = p6_down.__class__.__name__

    device, dtype = _infer_device_dtype(seq, detect_idx - 1)
    if device is not None:
        if dtype is not None:
            p6_down = p6_down.to(device=device, dtype=dtype)
        else:
            p6_down = p6_down.to(device=device)

    # Insert P6 producer before Detect.
    seq.insert(detect_idx, p6_down)

    # Refresh detect handle after insert.
    detect = seq[detect_idx + 1]
    if hasattr(detect, "i"):
        detect.i = int(detect_idx + 1)
    new_p6_idx = detect_idx
    detect_f_after = [int(x) for x in (list(getattr(detect, "f", old_f)) + [new_p6_idx])]
    detect.f = detect_f_after

    # Ensure new P6 feature is retained in y[] for Detect.f lookup.
    if hasattr(det_model, "save"):
        try:
            save = [int(x) for x in list(getattr(det_model, "save", []))]
            if new_p6_idx not in save:
                save.append(int(new_p6_idx))
            det_model.save = sorted(set(save))
        except Exception:
            pass

    # Extend Detect branches by cloning P5 branch template.
    detect.cv2.append(copy.deepcopy(detect.cv2[-1]))
    detect.cv3.append(copy.deepcopy(detect.cv3[-1]))
    if hasattr(detect, "one2one_cv2") and isinstance(detect.one2one_cv2, torch.nn.ModuleList):
        detect.one2one_cv2.append(copy.deepcopy(detect.one2one_cv2[-1]))
    if hasattr(detect, "one2one_cv3") and isinstance(detect.one2one_cv3, torch.nn.ModuleList):
        detect.one2one_cv3.append(copy.deepcopy(detect.one2one_cv3[-1]))

    heads_before = int(getattr(detect, "nl", 3))
    detect.nl = int(len(detect_f_after))
    stride_before = getattr(detect, "stride", torch.tensor([8.0, 16.0, 32.0], dtype=torch.float32))
    detect.stride = _stride_with_p6(stride_before).to(dtype=torch.float32)

    try:
        # Re-init only after structural head count change.
        detect.bias_init()
    except Exception:
        pass

    info = {
        "enabled": True,
        "patched_count": 1,
        "existing_count": 0,
        "detect_idx_before": int(detect_idx),
        "detect_idx_after": int(detect_idx + 1),
        "p5_source_idx": int(p5_idx),
        "p6_module_idx": int(new_p6_idx),
        "detect_heads_before": int(heads_before),
        "detect_heads_after": int(getattr(detect, "nl", len(detect_f_after))),
        "detect_f_before": old_f,
        "detect_f_after": detect_f_after,
        "stride_before": [float(x) for x in (stride_before.tolist() if isinstance(stride_before, torch.Tensor) else stride_before)],
        "stride_after": [float(x) for x in detect.stride.tolist()],
        "p6_channels": int(channels),
        "downsample_mode": d5_mode,
    }
    setattr(yolo_obj, "_enhance241_d5_info", info)

    recorder = get_check_recorder("d5", cfg, patch_info=info)
    if recorder is not None:
        try:
            _bind_module_debug(p6_down, recorder, prefix=f"d5.p6.idx{new_p6_idx}")
            recorder.register_module_params(p6_down, f"d5.p6.idx{new_p6_idx}")
        except Exception as exc:
            recorder.add_note(f"d5_module_debug_failed:{exc}")
        try:
            recorder.attach_detect_hooks(detect, detect_idx + 1)
            recorder.maybe_run_val_separability(yolo_obj, cfg)
        except Exception as exc:
            recorder.add_note(f"d5_detect_hook_failed:{exc}")

    return yolo_obj
