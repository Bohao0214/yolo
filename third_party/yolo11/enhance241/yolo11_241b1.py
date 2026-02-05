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


class _DWSeparableConv(torch.nn.Module):
    """Depthwise-separable 3x3 + pointwise 1x1 (no BN; batch<8 friendly)."""

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


class P3BiFPNLiteFuse(torch.nn.Module):
    """2.4.1 b1 (BiFPN-lite): normalized weighted sum + zero-init residual gate.

    Only patches the neck P4->P3 fusion point.

    Core:
      P3_fuse = (w3*P3 + w4*P4_up) / (w3+w4+eps)
      P3_out  = P3 + g * Conv(P3_fuse)   (g is zero-init => baseline-equivalent at start)

    Output keeps concat-compatible shape:
      cat([P4_up, P3_out]) -> 2C channels (downstream neck unchanged).
    """

    def __init__(self, channels_per_branch: int, eps: float = 1e-4, log_every: int = 500) -> None:
        super().__init__()
        self.c = int(channels_per_branch)
        if self.c <= 0:
            raise ValueError(f"channels_per_branch must be positive, got: {channels_per_branch}")
        self.eps = float(eps)
        self.log_every = int(log_every)

        # BiFPN normalized weights (per-channel, non-negative -> normalized)
        self.w = torch.nn.Parameter(torch.ones(2, self.c, 1, 1))

        # Lightweight refinement on fused feature (depthwise-separable; no BN)
        self.refine = _DWSeparableConv(self.c)

        # Zero-init residual gate: start identical to baseline.
        self.g = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

        self._step: int = 0

    def _norm_w(self) -> torch.Tensor:
        w_pos = torch.nn.functional.softplus(self.w)  # ensure non-negative, smooth gradients
        denom = w_pos.sum(dim=0, keepdim=True) + self.eps
        return w_pos / denom

    def forward(self, x: Any) -> torch.Tensor:
        if not isinstance(x, (list, tuple)) or len(x) != 2:
            raise TypeError(
                f"P3BiFPNLiteFuse expects [p4_up, p3], got: {type(x)} len={getattr(x, '__len__', None)}"
            )
        p4_up, p3 = x  # expected order from original Concat(f=[-1, P3])
        if p4_up.shape[-2:] != p3.shape[-2:]:
            p4_up = torch.nn.functional.interpolate(p4_up, size=p3.shape[-2:], mode="nearest")
        if p4_up.shape[1] != self.c or p3.shape[1] != self.c:
            raise ValueError(f"Channel mismatch: p4_up={p4_up.shape} p3={p3.shape} expected C={self.c}")

        w_norm = self._norm_w()
        p3_fuse = w_norm[0] * p3 + w_norm[1] * p4_up
        delta = self.refine(p3_fuse)
        gate = torch.tanh(self.g)  # zero-init => baseline-equivalent; bounded for stability
        p3_out = p3 + gate * delta

        # Optional lightweight monitoring (avoid log spam).
        if self.training and torch.is_grad_enabled() and self.log_every > 0 and (self._step % self.log_every == 0):
            with torch.no_grad():
                w3_m = float(w_norm[0].mean().detach().cpu())
                w4_m = float(w_norm[1].mean().detach().cpu())
                g_m = float(gate.detach().cpu())
                diff = (p3_out - p3).detach()
                diff_abs_mean = float(diff.abs().mean().cpu())
                diff_var = float(diff.var(unbiased=False).cpu())
            print(
                f"[enhance241] b1 mon step={self._step} w3_mean={w3_m:.4f} w4_mean={w4_m:.4f} "
                f"g={g_m:.4f} |diff|_mean={diff_abs_mean:.6f} diff_var={diff_var:.6f}"
            )
        self._step += 1

        return torch.cat((p4_up, p3_out), dim=1)


def _locate_p4_to_p3_fuse(seq: Any) -> Tuple[int, Any]:
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
    - If true: replace the neck P4->P3 fusion Concat with P3BiFPNLiteFuse.
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
    if isinstance(old, P3BiFPNLiteFuse):
        print(f"[enhance241] b1 enabled: already patched at model.model[{fuse_idx}] -> P3BiFPNLiteFuse(C={old.c})")
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
    fuse = P3BiFPNLiteFuse(channels_per_branch=c, eps=1e-4, log_every=500)

    # Preserve Ultralytics layer meta attributes used by its forward graph.
    for attr in ("i", "f", "type"):
        if hasattr(old, attr):
            setattr(fuse, attr, getattr(old, attr))

    seq[fuse_idx] = fuse

    print(
        f"[enhance241] b1 enabled: patched neck P4->P3 fuse at model.model[{fuse_idx}] "
        f"(Concat f={getattr(old, 'f', None)}) -> P3BiFPNLiteFuse(C={c})"
    )
    return yolo_obj
