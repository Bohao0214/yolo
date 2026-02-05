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

    - alpha starts small (default 0.10) and is learnable.
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

        # Prefer a plain scalar nn.Parameter for alpha to make optimizer inclusion obvious.
        alpha_init = max(0.0, min(float(alpha_init), self.alpha_max))
        self.alpha = torch.nn.Parameter(torch.tensor(alpha_init, dtype=torch.float32))

        # B2-1 (low risk): residual + gate (only enhance high-response regions on P3)
        #   g = sigmoid(gate(P3))
        #   delta = g * semantic(P4_up)
        #   P3_new = P3 + alpha * delta
        hidden = max(8, self.c // 4)
        self.semantic = torch.nn.Sequential(
            torch.nn.Conv2d(self.c, hidden, kernel_size=1, stride=1, padding=0, bias=True),
            torch.nn.SiLU(inplace=True),
            torch.nn.Conv2d(hidden, hidden, kernel_size=3, stride=1, padding=1, groups=hidden, bias=True),
            torch.nn.SiLU(inplace=True),
            torch.nn.Conv2d(hidden, self.c, kernel_size=1, stride=1, padding=0, bias=True),
        )
        # Baseline-safe: initial semantic == 0 regardless of other weights => delta == 0.
        torch.nn.init.zeros_(self.semantic[-1].weight)
        torch.nn.init.zeros_(self.semantic[-1].bias)

        # Gate from P3 only: starts slightly "off" to bias toward keeping baseline P3 details.
        self.gate = torch.nn.Conv2d(self.c, 1, kernel_size=1, stride=1, padding=0, bias=True)
        torch.nn.init.zeros_(self.gate.weight)
        torch.nn.init.constant_(self.gate.bias, -2.0)  # sigmoid(-2)≈0.119

        self._debug_printed: bool = False
        self._stats_printed: bool = False

    def reset_stats(self) -> None:
        self._stats_printed = False

    def _alpha(self) -> torch.Tensor:
        # Backward compatibility: older checkpoints may have alpha_logit instead of alpha.
        if hasattr(self, "alpha") and isinstance(getattr(self, "alpha"), torch.nn.Parameter):
            a = torch.clamp(self.alpha, 0.0, self.alpha_max)
            return a
        if hasattr(self, "alpha_logit"):
            a_logit = getattr(self, "alpha_logit")
            if isinstance(a_logit, torch.Tensor):
                return torch.sigmoid(a_logit) * self.alpha_max
        # Fallback constant (should not happen for newly created modules)
        return torch.tensor(0.0, device=next(self.parameters()).device, dtype=torch.float32)

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

        if self.training and not self._debug_printed:
            self._debug_printed = True
            alpha_param = getattr(self, "alpha", None)
            alpha_param_name = "alpha"
            if alpha_param is None:
                alpha_param = getattr(self, "alpha_logit", None)
                alpha_param_name = "alpha_logit"

            is_param = isinstance(alpha_param, torch.nn.Parameter)
            req_grad = bool(getattr(alpha_param, "requires_grad", False))
            params = list(self.parameters())
            alpha_in_params = id(alpha_param) in {id(p) for p in params}
            param_elems = sum(int(p.numel()) for p in params)
            print(
                f"[enhance241] b2 debug: {alpha_param_name} is nn.Parameter={is_param} requires_grad={req_grad}"
            )
            print(
                f"[enhance241] b2 debug: {alpha_param_name}_in_module_params={alpha_in_params} "
                f"{alpha_param_name}_id={id(alpha_param)} "
                f"module_param_tensors={len(params)} module_param_elems={param_elems}"
            )

        # Prefer gated semantic injection (B2-1). Fallback to legacy 'delta' if loading older checkpoints.
        if hasattr(self, "semantic") and hasattr(self, "gate"):
            g = torch.sigmoid(self.gate(p3))  # [B,1,H,W]
            delta = self.semantic(p4_up) * g  # [B,C,H,W]  broadcast
        elif hasattr(self, "delta"):
            delta = self.delta(torch.cat((p4_up, p3), dim=1))
        else:
            raise RuntimeError("P3ResidualFuseSafe has no semantic/gate or delta branch to compute residual.")

        p3_new = p3 + self._alpha() * delta

        if not self._stats_printed:
            self._stats_printed = True
            with torch.no_grad():
                mu = float(p3_new.mean().detach().cpu())
                var = float(p3_new.var(unbiased=False).detach().cpu())
                a = float(self._alpha().mean().detach().cpu())
            # Print only once per model instance.
            print(f"[enhance241] b2 stats: fused_p3 mean={mu:.6f} var={var:.6f} alpha_mean={a:.6f}")

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
