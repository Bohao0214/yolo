from __future__ import annotations

# Path: third_party/yolo11/enhance241/yolo11_241a3.py
# Purpose: enhance241 a3 (SPDConvDownsample) module + apply hook.

from typing import Any, Dict, List, Optional, Tuple

import torch

ENHANCE241_AUDIT_KEYS = ["enhance241_a3"]  # enhance241-audit


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


def _safe_int(x: Any, default: int) -> int:
    try:
        return int(x)
    except Exception:
        return int(default)


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


def _find_stride_conv(mod: torch.nn.Module, stride: int = 2) -> Optional[torch.nn.Conv2d]:
    for layer in mod.modules():
        if isinstance(layer, torch.nn.Conv2d):
            s = _stride_to_int(layer.stride)
            if s == stride:
                return layer
    return None


def _find_first_conv(mod: torch.nn.Module) -> Optional[torch.nn.Conv2d]:
    for layer in mod.modules():
        if isinstance(layer, torch.nn.Conv2d):
            return layer
    return None


class _DWSeparableConv(torch.nn.Module):
    """Depthwise-separable 3x3 + pointwise 1x1 (no BN)."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.dw = torch.nn.Conv2d(
            in_ch, in_ch, kernel_size=3, stride=1, padding=1, groups=in_ch, bias=True
        )
        self.act1 = torch.nn.SiLU(inplace=True)
        self.pw = torch.nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=True)
        self.act2 = torch.nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act2(self.pw(self.act1(self.dw(x))))


class _Conv3x3(torch.nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True)
        self.act = torch.nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(x))


class _SpaceToDepth(torch.nn.Module):
    def __init__(self, block: int = 2) -> None:
        super().__init__()
        self.block = int(block)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        if h % self.block != 0 or w % self.block != 0:
            raise ValueError(f"SpaceToDepth expects H/W divisible by {self.block}, got: {x.shape}")
        bh = h // self.block
        bw = w // self.block
        x = x.view(b, c, bh, self.block, bw, self.block)
        x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
        return x.view(b, c * (self.block ** 2), bh, bw)


class SPDConvDownsample(torch.nn.Module):
    """SPDConvDownsample: non-stride conv + space-to-depth for 2x downsample."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        pre_div: int = 4,
        refine: str = "dw",
    ) -> None:
        super().__init__()
        self.in_ch = int(in_ch)
        self.out_ch = int(out_ch)
        self.pre_div = int(max(1, pre_div))
        refine = str(refine).lower()

        if self.out_ch > 0 and self.out_ch % self.pre_div == 0:
            pre_ch = self.out_ch // self.pre_div
        else:
            pre_ch = self.out_ch

        if refine == "conv":
            self.enhance241_a3_pre = _Conv3x3(self.in_ch, pre_ch)  # enhance241-audit
        else:
            self.enhance241_a3_pre = _DWSeparableConv(self.in_ch, pre_ch)  # enhance241-audit

        self.enhance241_a3_s2d = _SpaceToDepth(2)  # enhance241-audit

        post_in = pre_ch * 4
        if post_in != self.out_ch:
            self.enhance241_a3_post = torch.nn.Conv2d(
                post_in, self.out_ch, kernel_size=1, stride=1, padding=0, bias=True
            )  # enhance241-audit
            self.enhance241_a3_post_act = torch.nn.SiLU(inplace=True)
        else:
            self.enhance241_a3_post = None
            self.enhance241_a3_post_act = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.enhance241_a3_pre(x)
        x = self.enhance241_a3_s2d(x)
        if self.enhance241_a3_post is not None:
            x = self.enhance241_a3_post_act(self.enhance241_a3_post(x))
        return x


def _concat_candidates(seq: Any) -> List[str]:
    out: List[str] = []
    for i, layer in enumerate(seq):
        if layer.__class__.__name__ == "Concat":
            out.append(f"idx={i}, f={getattr(layer, 'f', None)}")
    return out


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


def _locate_detect(seq: Any) -> Tuple[int, Any]:
    if not seq:
        raise RuntimeError("Empty model sequence.")
    idx = len(seq) - 1
    det = seq[idx]
    if det.__class__.__name__.lower() == "detect":
        return idx, det
    for i, layer in enumerate(seq):
        if layer.__class__.__name__.lower() == "detect":
            return i, layer
    raise RuntimeError("Detect layer not found.")


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


def apply(model: Any, cfg: Any) -> Any:
    """Apply enhance241 a3 SPDConvDownsample at the P3 downsample point."""

    enable_a3 = bool(_deep_get(cfg, "enhance241", "a3", default=False))
    if not enable_a3:
        return model

    yolo_obj = model
    if hasattr(model, "model") and hasattr(getattr(model, "model"), "model"):
        det_model = getattr(model, "model")
        seq = getattr(det_model, "model", None)
    else:
        det_model = model
        seq = getattr(det_model, "model", None)
    if seq is None:
        raise RuntimeError("enhance241.a3 requires an ultralytics YOLO/DetectionModel-like object with a .model.")

    prepatched = [i for i, layer in enumerate(seq) if isinstance(layer, SPDConvDownsample)]
    if prepatched:
        info = {
            "enabled": True,
            "existing_count": len(prepatched),
            "existing_indices": prepatched,
            "replaced_count": 0,
        }
        setattr(yolo_obj, "_enhance241_a3_info", info)
        if len(prepatched) == 1:
            return yolo_obj
        raise RuntimeError(f"enhance241.a3 expects exactly one SPDConvDownsample, got {len(prepatched)}")

    _, p3_idx = _locate_p4_to_p3_fuse(seq)
    ds_idx: Optional[int] = None
    for i in range(p3_idx, -1, -1):
        if _module_stride(seq[i]) == 2:
            ds_idx = i
            break
    if ds_idx is None:
        raise RuntimeError("Unable to locate stride=2 downsample before P3 output.")

    old = seq[ds_idx]
    if isinstance(old, SPDConvDownsample):
        return yolo_obj

    stride = _module_stride(old)
    if stride != 2:
        raise RuntimeError(f"Expected stride=2 at idx={ds_idx}, got stride={stride}")

    conv = _find_stride_conv(old, stride=2) or _find_first_conv(old)
    if conv is None:
        raise RuntimeError(f"Unable to infer in/out channels for layer idx={ds_idx} ({old.__class__.__name__}).")

    in_ch = int(conv.in_channels)
    out_ch = int(conv.out_channels)
    pre_div = _safe_int(_deep_get(cfg, "enhance241", "a3_pre_div", default=4), 4)
    refine = str(_deep_get(cfg, "enhance241", "a3_refine", default="dw"))

    fuse = SPDConvDownsample(in_ch=in_ch, out_ch=out_ch, pre_div=pre_div, refine=refine)
    for attr in ("i", "f", "type"):
        if hasattr(old, attr):
            setattr(fuse, attr, getattr(old, attr))

    device, dtype = _infer_device_dtype(seq, ds_idx)
    if device is not None:
        if dtype is not None:
            fuse = fuse.to(device=device, dtype=dtype)
        else:
            fuse = fuse.to(device=device)

    old_params = sum(int(p.numel()) for p in old.parameters())
    new_params = sum(int(p.numel()) for p in fuse.parameters())
    seq[ds_idx] = fuse

    info = {
        "enabled": True,
        "replaced_count": 1,
        "existing_count": 0,
        "replaced_idx": ds_idx,
        "p3_index": p3_idx,
        "orig_type": old.__class__.__name__,
        "orig_stride": stride,
        "new_type": "SPDConvDownsample",
        "params_old": int(old_params),
        "params_new": int(new_params),
        "delta_params": int(new_params - old_params),
        "pre_div": int(pre_div),
        "refine": str(refine),
    }
    setattr(yolo_obj, "_enhance241_a3_info", info)
    return yolo_obj
