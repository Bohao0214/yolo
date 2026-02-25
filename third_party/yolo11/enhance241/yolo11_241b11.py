from __future__ import annotations

# Path: third_party/yolo11/enhance241/yolo11_241b11.py
# Purpose: enhance241 b11 (Tiny Detection Layer, stride=4 branch) for P4->P3 family.

import copy
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from .yolo11_241a3 import (
    _bind_module_debug,
    _f_as_list,
    _get_module_recorder,
    _infer_device_dtype,
    _locate_detect,
    _should_capture_delta,
    _tensor_stats,
    _to_float,
    get_check_recorder,
)

ENHANCE241_AUDIT_KEYS = ["enhance241_b11"]  # enhance241-audit


def _deep_get(mapping: Any, *keys: str, default: Any = None) -> Any:
    cur = mapping
    for key in keys:
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(key)
            continue
        if hasattr(cur, key):
            cur = getattr(cur, key)
            continue
        return default
    return cur if cur is not None else default


def _safe_int(x: Any, default: int) -> int:
    try:
        return int(x)
    except Exception:
        return int(default)


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _zero_init_last_conv(module: torch.nn.Module) -> None:
    convs = [m for m in module.modules() if isinstance(m, torch.nn.Conv2d)]
    if not convs:
        return
    last = convs[-1]
    torch.nn.init.zeros_(last.weight)
    if last.bias is not None:
        torch.nn.init.zeros_(last.bias)


class _DWSeparable(torch.nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        c = int(channels)
        self.dw = torch.nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1, groups=c, bias=True)
        self.act1 = torch.nn.SiLU(inplace=True)
        self.pw = torch.nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0, bias=True)
        self.act2 = torch.nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act2(self.pw(self.act1(self.dw(x))))


class _Conv3x3(torch.nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        c = int(channels)
        self.conv = torch.nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1, bias=True)
        self.act = torch.nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(x))


class B11TinyP2Fuse(torch.nn.Module):
    """Tiny detection fusion:
    use P3 upsample + backbone_l3(P2-like) to build a new stride=4 head feature.
    """

    def __init__(
        self,
        p2_channels: int,
        p3_channels: int,
        out_channels: int,
        hidden_channels: int = 128,
        fuse_mode: str = "concat+conv",
        refine: str = "dw",
        upsample: str = "nearest",
        alpha_init: float = 0.0,
        alpha_cap: float = 0.5,
    ) -> None:
        super().__init__()
        c2 = int(p2_channels)
        c3 = int(p3_channels)
        co = int(out_channels)
        ch = int(max(16, hidden_channels))
        self.upsample = str(upsample).lower()
        self.fuse_mode = str(fuse_mode).lower()
        self.alpha_cap = float(max(1e-6, abs(alpha_cap)))

        self.p2_proj = torch.nn.Conv2d(c2, ch, kernel_size=1, stride=1, padding=0, bias=True)
        self.p3_proj = torch.nn.Conv2d(c3, ch, kernel_size=1, stride=1, padding=0, bias=True)
        self.base_proj = torch.nn.Conv2d(c2, co, kernel_size=1, stride=1, padding=0, bias=True)

        if self.fuse_mode == "add+conv":
            self.mix = torch.nn.Identity()
            mix_in = ch
        else:
            self.mix = torch.nn.Conv2d(ch * 2, ch, kernel_size=1, stride=1, padding=0, bias=True)
            mix_in = ch

        if str(refine).lower() == "conv":
            self.refine = _Conv3x3(mix_in)
        else:
            self.refine = _DWSeparable(mix_in)
        self.out_proj = torch.nn.Conv2d(ch, co, kernel_size=1, stride=1, padding=0, bias=True)
        _zero_init_last_conv(self.out_proj)

        alpha_init = float(max(-self.alpha_cap * 0.95, min(self.alpha_cap * 0.95, alpha_init)))
        alpha_ratio = alpha_init / self.alpha_cap
        alpha_raw = torch.atanh(torch.tensor(alpha_ratio, dtype=torch.float32))
        self.enhance241_b11_alpha_raw = torch.nn.Parameter(alpha_raw)  # enhance241-audit

    def forward(self, x: Any) -> torch.Tensor:
        recorder = _get_module_recorder(self)
        prefix = str(getattr(self, "_enhance241_prefix", "b11"))
        step = int(getattr(self, "_enhance241_fwd_step", 0))

        if not isinstance(x, (list, tuple)) or len(x) != 2:
            raise TypeError(f"B11TinyP2Fuse expects [p3, p2], got {type(x)}")
        p3, p2 = x
        if not isinstance(p3, torch.Tensor) or not isinstance(p2, torch.Tensor):
            raise TypeError("B11TinyP2Fuse inputs must be tensors")

        if p3.shape[-2:] != p2.shape[-2:]:
            if self.upsample == "bilinear":
                p3 = F.interpolate(p3, size=p2.shape[-2:], mode="bilinear", align_corners=False)
            else:
                p3 = F.interpolate(p3, size=p2.shape[-2:], mode="nearest")

        if recorder is not None and step == 0:
            recorder.capture_param_before(self, prefix)

        p2h = self.p2_proj(p2)
        p3h = self.p3_proj(p3)
        if self.fuse_mode == "add+conv":
            mixed = p2h + p3h
        else:
            mixed = self.mix(torch.cat((p3h, p2h), dim=1))
        delta_h = self.refine(mixed)
        delta = self.out_proj(delta_h)
        base = self.base_proj(p2)
        alpha_raw = self.enhance241_b11_alpha_raw.to(dtype=base.dtype, device=base.device)
        alpha = torch.tanh(alpha_raw) * self.alpha_cap
        out = base + alpha * delta

        if recorder is not None:
            if step == 0:
                recorder.record_module_compare(f"{prefix}.tiny_p2", patched=out, baseline=base, input_tensor=p2)
                recorder.record_a1_payload(
                    f"{prefix}.gate0",
                    {
                        "alpha": _to_float(alpha.item()),
                        "alpha_cap": _to_float(self.alpha_cap),
                        "p3_stats": _tensor_stats(p3),
                        "p2_stats": _tensor_stats(p2),
                        "delta_stats": _tensor_stats(delta),
                        "out_stats": _tensor_stats(out),
                    },
                )
                recorder.record_scalar_curve(f"{prefix}.alpha", _to_float(alpha.item()), step, max_steps=100)
                if out.requires_grad:
                    try:
                        out.register_hook(lambda g, n=f"{prefix}.tiny_p2": recorder.record_output_grad(n, g))
                    except Exception:
                        pass
            elif step <= 100:
                recorder.record_scalar_curve(f"{prefix}.alpha", _to_float(alpha.item()), step, max_steps=100)
            if _should_capture_delta(step):
                recorder.capture_param_delta(self, prefix)

        setattr(self, "_enhance241_fwd_step", step + 1)
        return out


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
    raise RuntimeError("Unable to infer detect head input channels for b11.")


def _infer_layer_out_channels(layer: Any) -> int:
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
    p3_concat_idx = int(p3_idx) - 1
    if 0 <= p3_concat_idx < len(seq):
        layer = seq[p3_concat_idx]
        if layer.__class__.__name__ == "Concat":
            f = _f_as_list(getattr(layer, "f", []))
            non_neg = [int(v) for v in f if int(v) >= 0]
            if non_neg:
                p3_backbone_idx = int(non_neg[-1])
                cand = p3_backbone_idx - 2
                if 0 <= cand < detect_idx:
                    return int(cand)
    if 0 <= 2 < detect_idx:
        return 2
    for idx in range(detect_idx):
        if idx != p3_idx:
            return int(idx)
    raise RuntimeError("Unable to locate P2 source index for b11.")


def _stride_with_p2(old_stride: Any, new_stride: int = 4) -> torch.Tensor:
    if isinstance(old_stride, torch.Tensor) and old_stride.numel() > 0:
        s = old_stride.detach().float().cpu()
        return torch.cat([torch.tensor([float(new_stride)], dtype=torch.float32), s], dim=0)
    if isinstance(old_stride, (list, tuple)) and len(old_stride) > 0:
        vals = [float(x) for x in old_stride]
        return torch.tensor([float(new_stride)] + vals, dtype=torch.float32)
    return torch.tensor([float(new_stride), 8.0, 16.0, 32.0], dtype=torch.float32)


def _prepend_branch(module_list: Any) -> Any:
    first = copy.deepcopy(module_list[0])
    try:
        module_list.insert(0, first)
        return module_list
    except Exception:
        return torch.nn.ModuleList([first] + list(module_list))


def _register_optimizer_check(yolo_obj: Any, b11_module: torch.nn.Module) -> None:
    if not hasattr(yolo_obj, "add_callback") or getattr(yolo_obj, "_enhance241_b11_callbacks", False):
        return
    setattr(yolo_obj, "_enhance241_b11_callbacks", True)

    def on_train_start(trainer: Any) -> None:
        opt = getattr(trainer, "optimizer", None)
        if opt is None:
            return
        opt_params = {id(p) for g in opt.param_groups for p in g.get("params", [])}
        missing = [p for p in b11_module.parameters() if id(p) not in opt_params]
        if missing:
            opt.add_param_group({"params": list(missing)})
            print(f"[enhance241] b11 optimizer: added {len(missing)} params to optimizer param_groups")
        else:
            print("[enhance241] b11 optimizer: all b11 params already registered")

    yolo_obj.add_callback("on_train_start", on_train_start)


def apply(model: Any, cfg: Any) -> Any:
    enable_b11 = bool(_deep_get(cfg, "enhance241", "b11", default=False))
    if not enable_b11:
        return model

    yolo_obj, det_model, seq = _extract_model_seq(model)
    if seq is None:
        raise RuntimeError("enhance241.b11 requires YOLO/DetectionModel-like object with .model sequence.")

    detect_idx, detect = _locate_detect(seq)
    old_f = _f_as_list(getattr(detect, "f", []))
    heads_before = int(getattr(detect, "nl", len(old_f)))

    prepatched = [i for i, layer in enumerate(seq[:detect_idx]) if isinstance(layer, B11TinyP2Fuse)]
    if prepatched:
        info = {
            "enabled": True,
            "existing_count": len(prepatched),
            "patched_count": 0,
            "detect_idx_before": int(detect_idx),
            "detect_heads_before": int(heads_before),
            "detect_heads_after": int(getattr(detect, "nl", len(_f_as_list(getattr(detect, "f", []))))),
            "note": "already_patched",
        }
        setattr(yolo_obj, "_enhance241_b11_info", info)
        recorder = get_check_recorder("b11", cfg, patch_info=info)
        if recorder is not None:
            try:
                idx0 = int(prepatched[0])
                mod0 = seq[idx0]
                _bind_module_debug(mod0, recorder, prefix=f"b11.idx{idx0}.existing")
                recorder.register_module_params(mod0, f"b11.idx{idx0}.existing")
                recorder.attach_detect_hooks(detect, detect_idx)
                recorder.maybe_run_val_separability(yolo_obj, cfg)
            except Exception as exc:
                recorder.add_note(f"b11_prepatched_hook_failed:{exc}")
        return yolo_obj

    if len(old_f) < 1:
        raise RuntimeError(f"enhance241.b11 invalid Detect.f={getattr(detect, 'f', None)}")

    p3_idx = int(old_f[0])
    b11_fuse_from = str(_deep_get(cfg, "enhance241", "b11_fuse_from", default="backbone_l3")).lower()
    if b11_fuse_from != "backbone_l3":
        raise RuntimeError(f"enhance241.b11 currently supports b11_fuse_from=backbone_l3 only, got {b11_fuse_from}")
    p2_idx = _locate_p2_source(seq, p3_idx=p3_idx, detect_idx=detect_idx)
    p2_channels = _infer_layer_out_channels(seq[p2_idx])
    p3_channels = _infer_head_in_channels(detect, head_idx=0)
    out_channels = p3_channels

    b11_stride = _safe_int(_deep_get(cfg, "enhance241", "b11_stride", default=4), 4)
    b11_hidden = _safe_int(_deep_get(cfg, "enhance241", "b11_channels", default=128), 128)
    b11_fuse_mode = str(_deep_get(cfg, "enhance241", "b11_fuse_mode", default="concat+conv"))
    b11_refine = str(_deep_get(cfg, "enhance241", "b11_refine", default="dw"))
    b11_upsample = str(_deep_get(cfg, "enhance241", "b11_upsample", default="nearest"))
    b11_alpha_init = _safe_float(_deep_get(cfg, "enhance241", "b11_alpha_init", default=0.0), 0.0)
    b11_alpha_cap = _safe_float(_deep_get(cfg, "enhance241", "b11_alpha_cap", default=0.5), 0.5)

    b11_fuse = B11TinyP2Fuse(
        p2_channels=p2_channels,
        p3_channels=p3_channels,
        out_channels=out_channels,
        hidden_channels=b11_hidden,
        fuse_mode=b11_fuse_mode,
        refine=b11_refine,
        upsample=b11_upsample,
        alpha_init=b11_alpha_init,
        alpha_cap=b11_alpha_cap,
    )
    b11_fuse.f = [int(p3_idx), int(p2_idx)]
    b11_fuse.i = int(detect_idx)
    b11_fuse.type = b11_fuse.__class__.__name__

    device, dtype = _infer_device_dtype(seq, detect_idx - 1)
    if device is not None:
        if dtype is not None:
            b11_fuse = b11_fuse.to(device=device, dtype=dtype)
        else:
            b11_fuse = b11_fuse.to(device=device)

    seq.insert(detect_idx, b11_fuse)
    detect = seq[detect_idx + 1]
    detect.i = int(detect_idx + 1)
    detect_f_after = [int(detect_idx)] + [int(x) for x in old_f]
    detect.f = detect_f_after

    detect.cv2 = _prepend_branch(detect.cv2)
    detect.cv3 = _prepend_branch(detect.cv3)
    if hasattr(detect, "one2one_cv2") and isinstance(detect.one2one_cv2, torch.nn.ModuleList) and len(detect.one2one_cv2):
        detect.one2one_cv2 = _prepend_branch(detect.one2one_cv2)
    if hasattr(detect, "one2one_cv3") and isinstance(detect.one2one_cv3, torch.nn.ModuleList) and len(detect.one2one_cv3):
        detect.one2one_cv3 = _prepend_branch(detect.one2one_cv3)

    detect.nl = int(len(detect_f_after))
    stride_before = getattr(detect, "stride", torch.tensor([8.0, 16.0, 32.0], dtype=torch.float32))
    detect.stride = _stride_with_p2(stride_before, new_stride=b11_stride).to(dtype=torch.float32)
    try:
        detect.bias_init()
    except Exception:
        pass

    # Gate-0: print detect input feature shapes/strides once to verify P2 branch enters head.
    if not getattr(detect, "_enhance241_b11_shape_hooked", False):
        detect._enhance241_b11_shape_hooked = True  # type: ignore[attr-defined]
        _once = {"done": False}

        def _pre_hook(mod: Any, inputs: Tuple[Any, ...]) -> None:
            if _once["done"]:
                return
            feats = inputs[0] if inputs else None
            if isinstance(feats, (list, tuple)):
                shapes = [tuple(t.shape) for t in feats if isinstance(t, torch.Tensor)]
                stride_list = _f_as_list(getattr(mod, "stride", []))
                print(f"[enhance241] b11 Gate-0: detect input shapes={shapes} strides={stride_list}")
                _once["done"] = True

        try:
            detect.register_forward_pre_hook(_pre_hook)
        except Exception:
            pass

    info = {
        "enabled": True,
        "existing_count": 0,
        "patched_count": 1,
        "detect_idx_before": int(detect_idx),
        "detect_idx_after": int(detect_idx + 1),
        "detect_heads_before": int(heads_before),
        "detect_heads_after": int(getattr(detect, "nl", len(detect_f_after))),
        "detect_f_before": [int(x) for x in old_f],
        "detect_f_after": [int(x) for x in detect_f_after],
        "p2_source_idx": int(p2_idx),
        "p3_source_idx": int(p3_idx),
        "p2_fuse_idx": int(detect_idx),
        "stride_after": [float(x) for x in detect.stride.tolist()],
        "fuse_from": str(b11_fuse_from),
        "fuse_mode": str(b11_fuse_mode),
        "refine": str(b11_refine),
        "hidden_channels": int(b11_hidden),
        "out_channels": int(out_channels),
        "alpha_init": _to_float(b11_alpha_init),
        "alpha_cap": _to_float(b11_alpha_cap),
    }
    setattr(yolo_obj, "_enhance241_b11_info", info)

    print(
        f"[enhance241] b11 enabled: inserted tiny head branch at model.model[{detect_idx}] "
        f"-> B11TinyP2Fuse(p2={p2_idx}, p3={p3_idx}, stride={b11_stride}, hidden={b11_hidden})"
    )
    _register_optimizer_check(yolo_obj, b11_fuse)

    recorder = get_check_recorder("b11", cfg, patch_info=info)
    if recorder is not None:
        _bind_module_debug(b11_fuse, recorder, prefix=f"b11.idx{detect_idx}")
        recorder.register_module_params(b11_fuse, f"b11.idx{detect_idx}")
        try:
            recorder.attach_detect_hooks(detect, detect_idx + 1)
        except Exception as exc:
            recorder.add_note(f"b11_detect_hook_failed:{exc}")
        try:
            recorder.maybe_run_val_separability(yolo_obj, cfg)
        except Exception as exc:
            recorder.add_note(f"b11_val_check_failed:{exc}")

    return yolo_obj
