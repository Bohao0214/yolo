from __future__ import annotations

# Path: third_party/yolo11/enhance241/yolo11_241d5.py
# Purpose: enhance241 d5 patch (paper-aligned P2 160x160 fourth head).

import copy
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from .yolo11_241a3 import (
    _bind_module_debug,
    _f_as_list,
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


class P2LiteFuse(torch.nn.Module):
    """Fuse shallow P2 and upsampled P3 into a new P2 detect input."""

    def __init__(
        self,
        p2_channels: int,
        p3_channels: int,
        out_channels: int,
        mode: str = "dw",
        upsample: str = "nearest",
    ) -> None:
        super().__init__()
        c2 = int(p2_channels)
        c3 = int(p3_channels)
        co = int(out_channels)
        self.upsample = str(upsample).lower()

        self.p2_proj = torch.nn.Identity() if c2 == co else torch.nn.Conv2d(c2, co, kernel_size=1, stride=1, padding=0, bias=True)
        self.p3_proj = torch.nn.Identity() if c3 == co else torch.nn.Conv2d(c3, co, kernel_size=1, stride=1, padding=0, bias=True)

        self.mix = torch.nn.Conv2d(2 * co, co, kernel_size=1, stride=1, padding=0, bias=True)
        if str(mode).lower() == "conv":
            self.refine = torch.nn.Conv2d(co, co, kernel_size=3, stride=1, padding=1, bias=True)
        else:
            self.refine = torch.nn.Sequential(
                torch.nn.Conv2d(co, co, kernel_size=3, stride=1, padding=1, groups=co, bias=True),
                torch.nn.SiLU(inplace=True),
                torch.nn.Conv2d(co, co, kernel_size=1, stride=1, padding=0, bias=True),
            )
        self.act = torch.nn.SiLU(inplace=True)

    def forward(self, x: Any) -> torch.Tensor:
        if not isinstance(x, (list, tuple)) or len(x) != 2:
            raise TypeError(f"P2LiteFuse expects [p3, p2], got {type(x)}")
        p3, p2 = x
        if not isinstance(p3, torch.Tensor) or not isinstance(p2, torch.Tensor):
            raise TypeError("P2LiteFuse inputs must be tensors")

        if p3.shape[-2:] != p2.shape[-2:]:
            if self.upsample == "bilinear":
                p3 = F.interpolate(p3, size=p2.shape[-2:], mode="bilinear", align_corners=False)
            else:
                p3 = F.interpolate(p3, size=p2.shape[-2:], mode="nearest")

        p2p = self.p2_proj(p2)
        p3p = self.p3_proj(p3)
        y = torch.cat((p3p, p2p), dim=1)
        y = self.act(self.mix(y))
        y = self.act(self.refine(y))
        return y


def _extract_model_seq(model: Any) -> Tuple[Any, Any, Optional[Any]]:
    yolo_obj = model
    if hasattr(model, "model") and hasattr(getattr(model, "model"), "model"):
        det_model = getattr(model, "model")
        seq = getattr(det_model, "model", None)
    else:
        det_model = model
        seq = getattr(det_model, "model", None)
    return yolo_obj, det_model, seq


def _infer_head_in_channels(detect: Any, head_idx: int = 0) -> int:
    try:
        return int(detect.cv2[head_idx][0].conv.in_channels)  # type: ignore[index]
    except Exception:
        pass
    try:
        branch = detect.cv2[head_idx]  # type: ignore[index]
        for m in branch.modules():
            if isinstance(m, torch.nn.Conv2d):
                return int(m.in_channels)
    except Exception:
        pass
    raise RuntimeError("Unable to infer head input channels for d5.")


def _infer_layer_out_channels(layer: Any) -> int:
    # Ultralytics blocks (e.g. C3k2/C2f) usually expose the final projection as cv2.conv.
    try:
        cv2 = getattr(layer, "cv2", None)
        if cv2 is not None and hasattr(cv2, "conv") and hasattr(cv2.conv, "out_channels"):
            return int(cv2.conv.out_channels)
    except Exception:
        pass
    try:
        conv = getattr(layer, "conv", None)
        if conv is not None and hasattr(conv, "out_channels"):
            return int(conv.out_channels)
    except Exception:
        pass
    if hasattr(layer, "c2"):
        try:
            return int(getattr(layer, "c2"))
        except Exception:
            pass
    if hasattr(layer, "out_channels"):
        try:
            return int(getattr(layer, "out_channels"))
        except Exception:
            pass
    out_c = None
    try:
        for m in layer.modules():
            if isinstance(m, torch.nn.Conv2d):
                out_c = int(m.out_channels)
    except Exception:
        out_c = None
    if out_c is None:
        raise RuntimeError(f"Unable to infer output channels from layer type={layer.__class__.__name__}")
    return int(out_c)


def _locate_p2_source(seq: Any, p3_idx: int, detect_idx: int) -> int:
    # Prefer: infer from the concat that feeds current P3 head path.
    p3_concat_idx = int(p3_idx) - 1
    if 0 <= p3_concat_idx < len(seq):
        layer = seq[p3_concat_idx]
        if layer.__class__.__name__ == "Concat":
            f = _f_as_list(getattr(layer, "f", []))
            non_neg = [int(v) for v in f if int(v) >= 0]
            if non_neg:
                p3_backbone_idx = int(non_neg[-1])
                candidate = p3_backbone_idx - 2  # YOLO11 topology: e.g. 4 -> 2
                if 0 <= candidate < detect_idx:
                    return int(candidate)

    # Stable fallback for YOLO11 family.
    if 0 <= 2 < detect_idx:
        return 2

    # Last fallback: earliest valid backbone index.
    for idx in range(detect_idx):
        if idx != p3_idx:
            return int(idx)
    raise RuntimeError("Unable to locate P2 source index for d5.")


def _stride_with_p2(old_stride: Any) -> torch.Tensor:
    if isinstance(old_stride, torch.Tensor) and old_stride.numel() > 0:
        s = old_stride.detach().float().cpu()
        return torch.cat([s[:1] / 2.0, s], dim=0)
    if isinstance(old_stride, (list, tuple)) and len(old_stride) > 0:
        vals = [float(x) for x in old_stride]
        return torch.tensor([vals[0] / 2.0] + vals, dtype=torch.float32)
    return torch.tensor([4.0, 8.0, 16.0, 32.0], dtype=torch.float32)


def _prepend_branch(module_list: Any) -> Any:
    first = copy.deepcopy(module_list[0])
    try:
        module_list.insert(0, first)
        return module_list
    except Exception:
        return torch.nn.ModuleList([first] + list(module_list))


def apply(model: Any, cfg: Any) -> Any:
    enable_d5 = bool(_deep_get(cfg, "enhance241", "d5", default=False))
    if not enable_d5:
        return model

    yolo_obj, det_model, seq = _extract_model_seq(model)
    if seq is None:
        raise RuntimeError("enhance241.d5 requires YOLO/DetectionModel-like object with .model sequence.")

    detect_idx, detect = _locate_detect(seq)
    old_f = _f_as_list(getattr(detect, "f", []))
    heads_before = int(getattr(detect, "nl", len(old_f)))

    has_p2_fuse = any(getattr(layer, "__class__", type(layer)).__name__ == "P2LiteFuse" for layer in seq[:detect_idx])
    has_legacy_p6 = any(getattr(layer, "__class__", type(layer)).__name__ == "P6Downsample" for layer in seq[:detect_idx])

    if len(old_f) >= 4 or heads_before >= 4:
        if has_legacy_p6 and not has_p2_fuse:
            raise RuntimeError(
                "enhance241.d5 detected legacy P6-style 4-head weights. "
                "Please retrain with corrected d5 (P2 head) to keep experiment semantics consistent."
            )

        info = {
            "enabled": True,
            "patched_count": 0,
            "existing_count": 1,
            "detect_idx": int(detect_idx),
            "detect_heads_before": int(heads_before),
            "detect_heads_after": int(getattr(detect, "nl", len(old_f))),
            "detect_f_before": [int(x) for x in old_f],
            "detect_f_after": [int(x) for x in old_f],
            "stride_after": [float(x) for x in getattr(detect, "stride", [])] if hasattr(detect, "stride") else [],
            "note": "already_patched",
            "head_type": "p2",
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

    p3_idx = int(old_f[0])
    if not (0 <= p3_idx < detect_idx):
        raise RuntimeError(f"enhance241.d5 invalid P3 source index {p3_idx} for detect_idx={detect_idx}")

    p2_idx = _locate_p2_source(seq, p3_idx=p3_idx, detect_idx=detect_idx)
    if not (0 <= p2_idx < detect_idx):
        raise RuntimeError(f"enhance241.d5 invalid inferred P2 index {p2_idx} for detect_idx={detect_idx}")

    head_channels = _infer_head_in_channels(detect, head_idx=0)
    p2_channels = _infer_layer_out_channels(seq[p2_idx])

    d5_mode = str(_deep_get(cfg, "enhance241", "d5_downsample", default="dw")).lower()
    d5_upsample = str(_deep_get(cfg, "enhance241", "d5_upsample", default="nearest")).lower()

    p2_fuse = P2LiteFuse(
        p2_channels=p2_channels,
        p3_channels=head_channels,
        out_channels=head_channels,
        mode=d5_mode,
        upsample=d5_upsample,
    )
    p2_fuse.f = [p3_idx, p2_idx]
    p2_fuse.i = int(detect_idx)
    p2_fuse.type = p2_fuse.__class__.__name__

    device, dtype = _infer_device_dtype(seq, detect_idx - 1)
    if device is not None:
        if dtype is not None:
            p2_fuse = p2_fuse.to(device=device, dtype=dtype)
        else:
            p2_fuse = p2_fuse.to(device=device)

    seq.insert(detect_idx, p2_fuse)

    detect = seq[detect_idx + 1]
    detect.i = int(detect_idx + 1)
    new_p2_idx = int(detect_idx)

    detect_f_after = [new_p2_idx] + [int(x) for x in old_f]
    detect.f = detect_f_after

    if hasattr(det_model, "save"):
        try:
            save = [int(x) for x in list(getattr(det_model, "save", []))]
            for idx in detect_f_after + [p2_idx, p3_idx]:
                if idx not in save:
                    save.append(int(idx))
            det_model.save = sorted(set(save))
        except Exception:
            pass

    detect.cv2 = _prepend_branch(detect.cv2)
    detect.cv3 = _prepend_branch(detect.cv3)
    if hasattr(detect, "one2one_cv2") and isinstance(detect.one2one_cv2, torch.nn.ModuleList) and len(detect.one2one_cv2):
        detect.one2one_cv2 = _prepend_branch(detect.one2one_cv2)
    if hasattr(detect, "one2one_cv3") and isinstance(detect.one2one_cv3, torch.nn.ModuleList) and len(detect.one2one_cv3):
        detect.one2one_cv3 = _prepend_branch(detect.one2one_cv3)

    detect.nl = int(len(detect_f_after))
    stride_before = getattr(detect, "stride", torch.tensor([8.0, 16.0, 32.0], dtype=torch.float32))
    detect.stride = _stride_with_p2(stride_before).to(dtype=torch.float32)

    try:
        detect.bias_init()
    except Exception:
        pass

    info = {
        "enabled": True,
        "patched_count": 1,
        "existing_count": 0,
        "head_type": "p2",
        "detect_idx_before": int(detect_idx),
        "detect_idx_after": int(detect_idx + 1),
        "p2_source_idx": int(p2_idx),
        "p3_source_idx": int(p3_idx),
        "p2_fuse_idx": int(new_p2_idx),
        "detect_heads_before": int(heads_before),
        "detect_heads_after": int(getattr(detect, "nl", len(detect_f_after))),
        "detect_f_before": [int(x) for x in old_f],
        "detect_f_after": [int(x) for x in detect_f_after],
        "stride_before": [float(x) for x in (stride_before.tolist() if isinstance(stride_before, torch.Tensor) else stride_before)],
        "stride_after": [float(x) for x in detect.stride.tolist()],
        "p2_channels": int(p2_channels),
        "head_channels": int(head_channels),
        "fuse_mode": d5_mode,
        "upsample_mode": d5_upsample,
    }
    setattr(yolo_obj, "_enhance241_d5_info", info)

    recorder = get_check_recorder("d5", cfg, patch_info=info)
    if recorder is not None:
        try:
            _bind_module_debug(p2_fuse, recorder, prefix=f"d5.p2.idx{new_p2_idx}")
            recorder.register_module_params(p2_fuse, f"d5.p2.idx{new_p2_idx}")
        except Exception as exc:
            recorder.add_note(f"d5_module_debug_failed:{exc}")
        try:
            recorder.attach_detect_hooks(detect, detect_idx + 1)
            recorder.maybe_run_val_separability(yolo_obj, cfg)
        except Exception as exc:
            recorder.add_note(f"d5_detect_hook_failed:{exc}")

    return yolo_obj
