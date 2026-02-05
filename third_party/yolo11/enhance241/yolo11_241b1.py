from __future__ import annotations

from typing import Any, Tuple

import torch


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


class _DSConv(torch.nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.dw = torch.nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=channels,
            bias=False,
        )
        self.bn1 = torch.nn.BatchNorm2d(channels, eps=1e-3, momentum=0.03)
        self.act1 = torch.nn.SiLU(inplace=True)
        self.pw = torch.nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(channels, eps=1e-3, momentum=0.03)
        self.act2 = torch.nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act1(self.bn1(self.dw(x)))
        x = self.act2(self.bn2(self.pw(x)))
        return x


class P3AlignedFuse(torch.nn.Module):
    """P3AlignedFuse: lightweight P4->P3 fusion enhancer (2.4.1 b1).

    Inputs:
      - [p4, p3] where p4 is higher-level feature (from P4 branch),
        p3 is lateral feature (from P3 branch).

    Outputs:
      - Tensor with channels = 2 * channels_per_branch (keeps original concat-compatible shape),
        intended to replace the original Concat at the P4->P3 fusion point.
    """

    def __init__(self, channels_per_branch: int, upsample_mode: str = "bilinear") -> None:
        super().__init__()
        self.c = int(channels_per_branch)
        if self.c <= 0:
            raise ValueError(f"channels_per_branch must be positive, got: {channels_per_branch}")
        self.upsample_mode = str(upsample_mode)

        # (1) Alignment after upsample: local smoothing/align conv (depthwise, cheap)
        self.p4_align = torch.nn.Sequential(
            torch.nn.Conv2d(self.c, self.c, kernel_size=3, stride=1, padding=1, groups=self.c, bias=False),
            torch.nn.BatchNorm2d(self.c, eps=1e-3, momentum=0.03),
            torch.nn.SiLU(inplace=True),
        )

        # (2) Learnable fusion weights: per-channel softmax weights for (p3, p4)
        self.fuse_w = torch.nn.Parameter(torch.zeros(2, self.c, 1, 1))

        # (3) Lightweight refinement on fused feature: depthwise separable conv + residual
        self.refine = _DSConv(self.c)

    def forward(self, x: Any) -> torch.Tensor:
        if not isinstance(x, (list, tuple)) or len(x) != 2:
            raise TypeError(f"P3AlignedFuse expects [p4, p3], got: {type(x)} len={getattr(x, '__len__', None)}")
        p4, p3 = x  # expected order from original Concat(f=[-1, P3])
        if p4.shape[1] != self.c or p3.shape[1] != self.c:
            raise ValueError(f"Channel mismatch: p4={p4.shape} p3={p3.shape} expected C={self.c}")

        if p4.shape[-2:] != p3.shape[-2:]:
            p4 = torch.nn.functional.interpolate(
                p4, size=p3.shape[-2:], mode=self.upsample_mode, align_corners=False if self.upsample_mode == "bilinear" else None
            )

        p4_aligned = self.p4_align(p4)

        w = torch.softmax(self.fuse_w, dim=0)
        fused = w[0] * p3 + w[1] * p4_aligned
        fused = fused + self.refine(fused)

        # Keep original concat-compatible shape (2C) so downstream neck stays unchanged.
        return torch.cat((p4_aligned, fused), dim=1)


def _locate_p4_to_p3_fuse(seq: Any) -> Tuple[int, Optional[Any]]:
    """Return (fuse_idx, detect_layer) for the P4->P3 fusion point."""
    detect = None
    if isinstance(seq, (list, tuple)) and seq:
        detect = seq[-1]
    else:
        try:
            detect = seq[-1]
        except Exception:
            detect = None

    # Preferred: derive from Detect.f = [P3, P4, P5] output indices
    detect_f = getattr(detect, "f", None)
    if isinstance(detect_f, (list, tuple)) and detect_f:
        try:
            p3_out = int(detect_f[0])
            fuse_idx = p3_out - 1
            return fuse_idx, detect
        except Exception:
            pass

    # Fallback: find Concat with f=[-1, 4] (YOLO11 default P4->P3 fuse)
    for i, layer in enumerate(seq):
        if layer.__class__.__name__ == "Concat" and list(getattr(layer, "f", [])) == [-1, 4]:
            return i, detect
    raise RuntimeError("Unable to locate P4->P3 fusion point in model.")


def apply(model: Any, cfg: Any) -> Any:
    """Apply 2.4.1 b1 enhancement (small-scale box recall) to YOLO11 model.

    - If cfg.enhance241.b1 is false: return model unchanged.
    - If true: replace the neck P4->P3 fusion Concat with P3AlignedFuse.
    """

    enable_b1 = bool(_deep_get(cfg, "enhance241", "b1", default=False))
    if not enable_b1:
        return model

    # Support both ultralytics.YOLO and raw nn.Module-like inputs.
    yolo_obj = model
    det_model = getattr(model, "model", None)
    if det_model is None:
        det_model = model

    seq = getattr(det_model, "model", None)
    if seq is None:
        raise RuntimeError("enhance241.b1 requires an ultralytics DetectionModel-like object with .model (Sequential).")

    fuse_idx, _ = _locate_p4_to_p3_fuse(seq)
    if fuse_idx < 0 or fuse_idx >= len(seq):
        raise RuntimeError(f"Invalid fuse_idx={fuse_idx} for model length={len(seq)}")

    old = seq[fuse_idx]
    if isinstance(old, P3AlignedFuse):
        print(f"[enhance241] b1 enabled: already patched at model.model[{fuse_idx}] -> P3AlignedFuse(C={old.c})")
        return yolo_obj

    if old.__class__.__name__ != "Concat":
        raise RuntimeError(
            f"Expected Concat at P4->P3 fuse idx={fuse_idx}, got {old.__class__.__name__}. Refusing to patch."
        )

    next_layer = seq[fuse_idx + 1] if (fuse_idx + 1) < len(seq) else None
    c_in = None
    try:
        c_in = int(next_layer.cv1.conv.in_channels)  # type: ignore[attr-defined]
    except Exception:
        c_in = None
    if not c_in or c_in % 2 != 0:
        raise RuntimeError(f"Unable to infer concat channels from next layer at idx={fuse_idx+1}. c_in={c_in}")

    c = c_in // 2
    fuse = P3AlignedFuse(channels_per_branch=c, upsample_mode="bilinear")

    # Preserve Ultralytics layer meta attributes used by its forward graph.
    for attr in ("i", "f", "type"):
        if hasattr(old, attr):
            setattr(fuse, attr, getattr(old, attr))

    seq[fuse_idx] = fuse

    print(
        f"[enhance241] b1 enabled: patched neck P4->P3 fuse at model.model[{fuse_idx}] "
        f"(Concat f={getattr(old, 'f', None)}) -> P3AlignedFuse(C={c})"
    )
    return yolo_obj
