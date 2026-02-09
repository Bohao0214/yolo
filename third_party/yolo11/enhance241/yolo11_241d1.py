from __future__ import annotations

# Path: third_party/yolo11/enhance241/yolo11_241d1.py
# Purpose: enhance241 d1 (AddP2DetectHead) module + apply hook.

from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import torch

ENHANCE241_AUDIT_KEYS = ["enhance241_d1"]  # enhance241-audit


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


def _safe_int(x: Any, default: int) -> int:
    try:
        return int(x)
    except Exception:
        return int(default)


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


class _Conv1x1(torch.nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=True)
        self.act = torch.nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(x))


class P2LiteFuse(torch.nn.Module):
    """Fuse P2 (stride=4) with upsampled P3 to produce P2 head input."""

    def __init__(
        self,
        p2_ch: int,
        p3_ch: int,
        out_ch: int,
        weight_init: float = 1.0,
        refine: str = "dw",
        upsample_mode: str = "nearest",
        eps: float = 1e-4,
    ) -> None:
        super().__init__()
        self.p2_ch = int(p2_ch)
        self.p3_ch = int(p3_ch)
        self.out_ch = int(out_ch)
        self.upsample_mode = str(upsample_mode).lower()
        self.eps = float(eps)

        self.enhance241_d1_p2_proj = _Conv1x1(self.p2_ch, self.out_ch)  # enhance241-audit
        self.enhance241_d1_p3_proj = _Conv1x1(self.p3_ch, self.out_ch)  # enhance241-audit

        refine = str(refine).lower()
        if refine == "conv":
            self.enhance241_d1_refine = _Conv3x3(self.out_ch)  # enhance241-audit
        else:
            self.enhance241_d1_refine = _DWSeparableConv(self.out_ch)  # enhance241-audit

        self.enhance241_d1_weight = torch.nn.Parameter(  # enhance241-audit
            torch.tensor([float(weight_init), float(weight_init)], dtype=torch.float32)
        )

    def _maybe_align(self, p3: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
        if p3.shape[-2:] == p2.shape[-2:]:
            return p3
        if self.upsample_mode == "bilinear":
            return torch.nn.functional.interpolate(p3, size=p2.shape[-2:], mode="bilinear", align_corners=False)
        return torch.nn.functional.interpolate(p3, size=p2.shape[-2:], mode="nearest")

    def forward(self, x: Any) -> torch.Tensor:
        if not isinstance(x, (list, tuple)) or len(x) != 2:
            raise TypeError("P2LiteFuse expects [p2, p3] inputs.")
        p2, p3 = x[0], x[1]
        p3 = self._maybe_align(p3, p2)

        p2p = self.enhance241_d1_p2_proj(p2)
        p3p = self.enhance241_d1_p3_proj(p3)
        w = torch.relu(self.enhance241_d1_weight)
        w = w / (w.sum() + self.eps)
        fused = w[0] * p2p + w[1] * p3p
        return self.enhance241_d1_refine(fused)


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


def _locate_p4_to_p3_fuse(seq: Any) -> Tuple[int, int]:
    detect_idx, detect = _locate_detect(seq)
    detect_f = _f_as_list(getattr(detect, "f", []))
    if detect_f:
        p3_idx = int(detect_f[0])
        fuse_idx = p3_idx - 1
        if 0 <= fuse_idx < detect_idx:
            f_list = _f_as_list(getattr(seq[fuse_idx], "f", []))
            if len(f_list) >= 2:
                return fuse_idx, int(f_list[1])

    candidates: List[Tuple[int, List[int]]] = []
    for i, layer in enumerate(seq):
        f_list = _f_as_list(getattr(layer, "f", []))
        if layer.__class__.__name__ == "Concat" and len(f_list) == 2 and f_list[0] == -1:
            candidates.append((i, f_list))
    if candidates:
        fuse_idx, f_list = candidates[-1]
        return fuse_idx, int(f_list[1])

    raise RuntimeError(f"Unable to locate P4->P3 fuse. Concat candidates: {_concat_candidates(seq)}")


def _stride_to_int(s: Any) -> Optional[int]:
    try:
        if isinstance(s, (list, tuple)):
            return int(s[0]) if s else None
        if isinstance(s, torch.Tensor):
            if s.numel() == 1:
                return int(s.item())
            return int(s.flatten()[0].item())
        return int(s)
    except Exception:
        return None


def _module_stride(mod: Any) -> Optional[int]:
    if hasattr(mod, "stride"):
        s = _stride_to_int(getattr(mod, "stride"))
        if s is not None:
            return s
    for attr in ("conv", "cv1", "dw"):
        sub = getattr(mod, attr, None)
        if sub is None:
            continue
        conv = getattr(sub, "conv", None) if hasattr(sub, "conv") else sub
        if hasattr(conv, "stride"):
            s = _stride_to_int(getattr(conv, "stride"))
            if s is not None:
                return s
    for layer in mod.modules():
        if isinstance(layer, torch.nn.Conv2d):
            s = _stride_to_int(layer.stride)
            if s is not None:
                return s
    return None


def _resolve_from(idx: int, f: Any) -> int:
    if isinstance(f, int):
        return idx + f if f < 0 else f
    if isinstance(f, (list, tuple)) and f:
        f0 = f[0]
        if isinstance(f0, int):
            return idx + f0 if f0 < 0 else f0
    return idx - 1


def _infer_out_channels(mod: torch.nn.Module) -> Optional[int]:
    convs = [m for m in mod.modules() if isinstance(m, torch.nn.Conv2d)]
    if not convs:
        return None
    return int(convs[-1].out_channels)


def _infer_feature_channels_by_probe(
    seq: Any,
    layer_indices: List[int],
    device: Optional[torch.device],
    dtype: Optional[torch.dtype],
    imgsz: int = 640,
) -> Dict[int, int]:
    if not layer_indices:
        return {}
    max_idx = max(int(i) for i in layer_indices)
    x = torch.zeros(1, 3, int(imgsz), int(imgsz), device=device, dtype=dtype or torch.float32)
    y: List[Any] = []
    out: Dict[int, int] = {}
    with torch.no_grad():
        for i, m in enumerate(seq):
            f = getattr(m, "f", -1)
            if isinstance(f, int):
                xin = x if f == -1 else y[f]
            else:
                fs = _f_as_list(f)
                xin = [x if j == -1 else y[j] for j in fs]
            x = m(xin)
            y.append(x)
            if i in layer_indices and isinstance(x, torch.Tensor):
                out[int(i)] = int(x.shape[1])
            if i >= max_idx:
                break
    return out


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


def _locate_detect(seq: Any) -> Tuple[int, Any]:
    if not seq:
        raise RuntimeError("Empty model sequence.")
    detect_idx = len(seq) - 1
    detect = seq[detect_idx]
    if detect.__class__.__name__.lower() != "detect":
        for i, layer in enumerate(seq):
            if layer.__class__.__name__.lower() == "detect":
                detect_idx, detect = i, layer
                break
    return detect_idx, detect


def _expand_detect_heads(detect: Any, old_nl: int) -> None:
    for name, mod in list(getattr(detect, "_modules", {}).items()):
        if isinstance(mod, torch.nn.ModuleList) and len(mod) == old_nl:
            mod.insert(0, deepcopy(mod[0]))


def _ensure_save_indices(det_model: Any, indices: List[int]) -> List[int]:
    save = list(getattr(det_model, "save", []) or [])
    merged = set()
    for x in save:
        try:
            merged.add(int(x))
        except Exception:
            continue
    for x in indices:
        try:
            merged.add(int(x))
        except Exception:
            continue
    out = sorted(merged)
    setattr(det_model, "save", out)
    return out


def apply(model: Any, cfg: Any) -> Any:
    """Apply enhance241 d1: add P2 detect head (4-head)."""

    enable_d1 = bool(_deep_get(cfg, "enhance241", "d1", default=False))
    if not enable_d1:
        return model

    yolo_obj = model
    if hasattr(model, "model") and hasattr(getattr(model, "model"), "model"):
        det_model = getattr(model, "model")
        seq = getattr(det_model, "model", None)
    else:
        det_model = model
        seq = getattr(det_model, "model", None)
    if seq is None:
        raise RuntimeError("enhance241.d1 requires an ultralytics YOLO/DetectionModel-like object with a .model sequence.")

    prepatched = [i for i, layer in enumerate(seq) if isinstance(layer, P2LiteFuse)]
    detect_idx, detect = _locate_detect(seq)
    detect_f = _f_as_list(getattr(detect, "f", [])) if hasattr(detect, "f") else []
    if prepatched or len(detect_f) >= 4:
        if prepatched:
            pf = _f_as_list(getattr(seq[prepatched[0]], "f", []))
            needed = []
            needed.extend(pf)
            needed.extend(detect_f)
            needed.extend(prepatched)
            _ensure_save_indices(det_model, needed)
        info = {
            "enabled": True,
            "note": "already_patched",
            "detect_heads_before": len(detect_f),
            "detect_heads_after": len(detect_f),
            "p2_index": prepatched[0] if prepatched else None,
        }
        setattr(yolo_obj, "_enhance241_d1_info", info)
        return yolo_obj

    if not detect_f:
        raise RuntimeError("enhance241.d1 requires Detect.f to be present.")
    p3_idx = int(detect_f[0])

    _, p3_skip_idx = _locate_p4_to_p3_fuse(seq)
    ds_idx: Optional[int] = None
    if p3_skip_idx - 1 >= 0:
        ds_idx = p3_skip_idx - 1
    if ds_idx is None:
        for i in range(p3_skip_idx, -1, -1):
            if _module_stride(seq[i]) == 2:
                ds_idx = i
                break
    if ds_idx is None:
        raise RuntimeError("Unable to locate downsample before P3 for P2 head.")
    ds_layer = seq[ds_idx]
    p2_idx = _resolve_from(ds_idx, getattr(ds_layer, "f", -1))

    probe_device, probe_dtype = _infer_device_dtype(seq, max(p2_idx, p3_idx))
    probe_channels = _infer_feature_channels_by_probe(
        seq=seq,
        layer_indices=[p2_idx, p3_idx],
        device=probe_device,
        dtype=probe_dtype,
        imgsz=_safe_int(_deep_get(cfg, "imgsz", default=640), 640),
    )
    p2_ch = probe_channels.get(p2_idx)
    p3_ch = probe_channels.get(p3_idx)

    if p2_ch is None or p3_ch is None:
        p2_ch = p2_ch or _infer_out_channels(seq[p2_idx])
        p3_ch = p3_ch or _infer_out_channels(seq[p3_idx])
    if p2_ch is None or p3_ch is None:
        raise RuntimeError(f"Unable to infer channel sizes for P2/P3 heads. p2_idx={p2_idx}, p3_idx={p3_idx}")

    enhance_cfg = _deep_get(cfg, "enhance241", default={}) or {}
    if not isinstance(enhance_cfg, dict):
        enhance_cfg = {}
    weight_init = _safe_float(enhance_cfg.get("d1_weight_init", 1.0), 1.0)
    refine = str(enhance_cfg.get("d1_refine", "dw"))
    upsample_mode = str(enhance_cfg.get("d1_upsample", "nearest")).lower()
    eps = _safe_float(enhance_cfg.get("d1_eps", 1e-4), 1e-4)
    out_ch_req = _safe_int(enhance_cfg.get("d1_out_channels", p3_ch), int(p3_ch))
    out_ch = out_ch_req
    if out_ch <= 0 or out_ch != int(p3_ch):
        out_ch = int(p3_ch)

    fuse = P2LiteFuse(
        p2_ch=p2_ch,
        p3_ch=p3_ch,
        out_ch=out_ch,
        weight_init=weight_init,
        refine=refine,
        upsample_mode=upsample_mode,
        eps=eps,
    )
    setattr(fuse, "i", int(detect_idx))
    setattr(fuse, "f", [p2_idx, p3_idx])
    setattr(fuse, "type", fuse.__class__.__name__)

    device, dtype = _infer_device_dtype(seq, detect_idx - 1)
    if device is not None:
        if dtype is not None:
            fuse = fuse.to(device=device, dtype=dtype)
        else:
            fuse = fuse.to(device=device)

    # Insert fuse before Detect
    try:
        seq.insert(detect_idx, fuse)
    except Exception:
        modules = list(seq)
        modules.insert(detect_idx, fuse)
        seq = torch.nn.ModuleList(modules)
        if hasattr(det_model, "model"):
            det_model.model = seq

    # Reindex modules (for metadata only).
    for i in range(detect_idx, len(seq)):
        try:
            setattr(seq[i], "i", i)
        except Exception:
            pass

    detect_idx, detect = _locate_detect(seq)
    old_f = list(detect_f)
    detect.f = [detect_idx - 1] + old_f

    if hasattr(detect, "ch"):
        old_ch = list(getattr(detect, "ch"))
        detect.ch = [int(out_ch)] + old_ch
    old_nl = len(old_f)
    _expand_detect_heads(detect, old_nl)
    if hasattr(detect, "nl"):
        detect.nl = old_nl + 1
    save_after = _ensure_save_indices(det_model, [p2_idx, p3_idx, detect_idx - 1] + list(getattr(detect, "f", [])))

    # Update stride if available.
    strides = None
    if hasattr(detect, "stride"):
        try:
            s_list = list(getattr(detect, "stride"))
            if s_list:
                first = s_list[0] / 2.0
                strides = [first] + s_list
                detect.stride = torch.tensor(strides)
        except Exception:
            strides = None

    # Record head resolution info (using imgsz if possible).
    imgsz = _safe_int(_deep_get(cfg, "imgsz", default=640), 640)
    head_res = []
    if strides:
        for s in strides:
            try:
                h = int(round(float(imgsz) / float(s)))
                head_res.append({"stride": float(s), "hw": [h, h]})
            except Exception:
                continue

    info = {
        "enabled": True,
        "detect_heads_before": old_nl,
        "detect_heads_after": old_nl + 1,
        "detect_f_before": old_f,
        "detect_f_after": _f_as_list(getattr(detect, "f", [])),
        "p2_index": detect_idx - 1,
        "p3_index": p3_idx,
        "out_channels": int(out_ch),
        "out_channels_requested": int(out_ch_req),
        "stride": strides,
        "head_resolutions": head_res,
        "weight_init": float(weight_init),
        "refine": str(refine),
        "save_indices_len": len(save_after),
        "save_indices_tail": save_after[-12:],
    }
    setattr(yolo_obj, "_enhance241_d1_info", info)
    return yolo_obj
