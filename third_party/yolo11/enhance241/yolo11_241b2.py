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


class P3ResidualFuseSafe(torch.nn.Module):
    """2.4.1 b2-safe: residual P4->P3 fusion (baseline-safe init, no BN).

    Strategy:
      F3_new = F3_old + alpha * Delta(P4_up, F3_old)

    - alpha starts small (default 0.10) and is learnable (bounded to (0, alpha_max]).
    - Delta head last conv is zero-initialized => initial Delta == 0 => output equals baseline.
    - No BatchNorm (batch < 8 is a hard constraint).
    - Prints fused feature mean/var once (for hidden "near-zero output" issues).

    Input:
      [p4_up, p3]  (same order as original Concat(f=[-1, P3]))

    Output:
      cat([p4_up, p3_new]) with channels = 2C (concat-compatible), only changing the P3 branch.
    """

    def __init__(self, channels_per_branch: int, alpha_init: float = 0.10, alpha_max: float = 0.20) -> None:
        super().__init__()
        self.c = int(channels_per_branch)
        if self.c <= 0:
            raise ValueError(f"channels_per_branch must be positive, got: {channels_per_branch}")

        self.alpha_max = float(alpha_max)
        if self.alpha_max <= 0:
            raise ValueError(f"alpha_max must be positive, got: {alpha_max}")

        alpha_init = float(alpha_init)
        alpha_init = max(0.0, min(alpha_init, self.alpha_max))
        # Use sigmoid(logit) * alpha_max to keep alpha positive and bounded.
        init_ratio = (alpha_init / self.alpha_max) if self.alpha_max > 0 else 0.0
        init_ratio = max(1e-6, min(init_ratio, 1 - 1e-6))
        init_logit = float(torch.log(torch.tensor(init_ratio / (1 - init_ratio))))
        self.alpha_logit = torch.nn.Parameter(torch.full((1, self.c, 1, 1), init_logit))

        hidden = max(8, self.c // 4)
        self.delta = torch.nn.Sequential(
            torch.nn.Conv2d(2 * self.c, hidden, kernel_size=1, stride=1, padding=0, bias=True),
            torch.nn.SiLU(inplace=True),
            torch.nn.Conv2d(hidden, hidden, kernel_size=3, stride=1, padding=1, groups=hidden, bias=True),
            torch.nn.SiLU(inplace=True),
            torch.nn.Conv2d(hidden, self.c, kernel_size=1, stride=1, padding=0, bias=True),
        )
        # Baseline-safe: initial Delta == 0 regardless of other weights.
        torch.nn.init.zeros_(self.delta[-1].weight)
        torch.nn.init.zeros_(self.delta[-1].bias)

        self._stats_printed: bool = False

    def reset_stats(self) -> None:
        self._stats_printed = False

    def _alpha(self) -> torch.Tensor:
        return torch.sigmoid(self.alpha_logit) * self.alpha_max

    def forward(self, x: Any) -> torch.Tensor:
        if not isinstance(x, (list, tuple)) or len(x) != 2:
            raise TypeError(
                f"P3ResidualFuseSafe expects [p4_up, p3], got: {type(x)} len={getattr(x, '__len__', None)}"
            )
        p4_up, p3 = x
        if p4_up.shape[-2:] != p3.shape[-2:]:
            p4_up = torch.nn.functional.interpolate(p4_up, size=p3.shape[-2:], mode="nearest")

        if p4_up.shape[1] != self.c or p3.shape[1] != self.c:
            raise ValueError(f"Channel mismatch: p4_up={p4_up.shape} p3={p3.shape} expected C={self.c}")

        delta = self.delta(torch.cat((p4_up, p3), dim=1))
        p3_new = p3 + self._alpha() * delta

        if not self._stats_printed:
            self._stats_printed = True
            with torch.no_grad():
                mu = float(p3_new.mean().detach().cpu())
                var = float(p3_new.var(unbiased=False).detach().cpu())
                a = float(self._alpha().mean().detach().cpu())
            # Print only once per model instance.
            print(f"[enhance241] b2 stats: fused_p3 mean={mu:.6f} var={var:.6f} alpha_mean={a:.4f}")

        return torch.cat((p4_up, p3_new), dim=1)


def _locate_p4_to_p3_fuse(seq: Any) -> Tuple[int, Any]:
    detect = None
    try:
        detect = seq[-1]
    except Exception:
        detect = None

    detect_f = getattr(detect, "f", None)
    if isinstance(detect_f, (list, tuple)) and detect_f:
        try:
            p3_out = int(detect_f[0])
            fuse_idx = p3_out - 1
            return fuse_idx, detect
        except Exception:
            pass

    for i, layer in enumerate(seq):
        if layer.__class__.__name__ == "Concat" and list(getattr(layer, "f", [])) == [-1, 4]:
            return i, detect
    raise RuntimeError("Unable to locate P4->P3 fusion point in model.")


def apply(model: Any, cfg: Any) -> Any:
    """Apply 2.4.1 b2-safe enhancement to YOLO11 model (neck P4->P3 fusion point)."""

    enable_b2 = bool(_deep_get(cfg, "enhance241", "b2", default=False))
    if not enable_b2:
        return model

    yolo_obj = model
    det_model = getattr(model, "model", None) or model
    seq = getattr(det_model, "model", None)
    if seq is None:
        raise RuntimeError("enhance241.b2 requires an ultralytics DetectionModel-like object with .model (Sequential).")

    fuse_idx, _ = _locate_p4_to_p3_fuse(seq)
    if fuse_idx < 0 or fuse_idx >= len(seq):
        raise RuntimeError(f"Invalid fuse_idx={fuse_idx} for model length={len(seq)}")

    old = seq[fuse_idx]
    if isinstance(old, P3ResidualFuseSafe):
        old.reset_stats()
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
    fuse = P3ResidualFuseSafe(channels_per_branch=c, alpha_init=0.10, alpha_max=0.20)

    for attr in ("i", "f", "type"):
        if hasattr(old, attr):
            setattr(fuse, attr, getattr(old, attr))

    seq[fuse_idx] = fuse

    print(
        f"[enhance241] b2 enabled: patched neck P4->P3 fuse at model.model[{fuse_idx}] "
        f"(Concat f={getattr(old, 'f', None)}) -> P3ResidualFuseSafe(C={c}, alpha_init=0.10)"
    )
    return yolo_obj

