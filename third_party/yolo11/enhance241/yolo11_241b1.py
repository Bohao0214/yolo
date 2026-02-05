from __future__ import annotations

# 路径: third_party/yolo11/enhance241/yolo11_241b1.py
# 作用: enhance241 b1_v3（ASFF-lite 空间自适应融合）模块与注入入口

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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


def _is_rank0() -> bool:
    try:
        from ultralytics.utils import RANK  # type: ignore

        return int(RANK) in {-1, 0}
    except Exception:
        return True


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


class _DWSeparableConv(torch.nn.Module):
    """Depthwise-separable 3x3 + pointwise 1x1 (no BatchNorm/BN; batch<8 friendly)."""

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


class P3ASFFLiteFuse(torch.nn.Module):
    """2.4.1 b1_v3: ASFF-lite 空间自适应融合，仅作用于 P4->P3 融合点。

    - 权重预测器:
        w = Conv1x1(ReLU(Conv1x1([p3,p4_up]))) -> (B,2,H,W)
        W = softmax(w, dim=1)
        fused = W0*p3 + W1*p4_up
    - refine: depthwise separable (默认) 或普通 3x3 conv
    - 输出: p3_out = p3 + alpha * refine(fused)
    - 兼容原 Concat 输出: cat([p4_up, p3_out]) -> 2C
    """

    def __init__(
        self,
        channels_per_branch: int,
        alpha_init: float = 0.05,
        weight_hidden: int = 64,
        refine: str = "dw",
        upsample_mode: str = "nearest",
        mon_every: int = 100,
    ) -> None:
        super().__init__()
        self.c = int(channels_per_branch)
        if self.c <= 0:
            raise ValueError(f"channels_per_branch must be positive, got: {channels_per_branch}")

        self.upsample_mode = str(upsample_mode).lower()
        self.mon_every = int(mon_every)

        hidden = int(max(8, weight_hidden))
        self.weight_net = torch.nn.Sequential(
            torch.nn.Conv2d(self.c * 2, hidden, kernel_size=1, stride=1, padding=0, bias=True),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(hidden, 2, kernel_size=1, stride=1, padding=0, bias=True),
        )

        refine = str(refine).lower()
        if refine == "conv":
            self.refine = _Conv3x3(self.c)
        else:
            self.refine = _DWSeparableConv(self.c)

        # Learnable alpha (scalar or per-channel). Here: scalar for stability.
        self.alpha = torch.nn.Parameter(torch.tensor(float(alpha_init), dtype=torch.float32))

        self._step: int = 0
        self._last_mon: Optional[Dict[str, float]] = None
        self._last_mon_step: int = -1

    def pop_monitor(self) -> Optional[Dict[str, float]]:
        row = self._last_mon
        self._last_mon = None
        return row

    def _maybe_align(self, p4_up: torch.Tensor, p3: torch.Tensor) -> torch.Tensor:
        if p4_up.shape[-2:] == p3.shape[-2:]:
            return p4_up
        if self.upsample_mode == "bilinear":
            return torch.nn.functional.interpolate(p4_up, size=p3.shape[-2:], mode="bilinear", align_corners=False)
        return torch.nn.functional.interpolate(p4_up, size=p3.shape[-2:], mode="nearest")

    def _split_input(self, x: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(x, (list, tuple)) and len(x) == 2:
            return x[0], x[1]
        if isinstance(x, torch.Tensor):
            if x.shape[1] != self.c * 2:
                raise ValueError(f"Expected concat tensor with 2C channels, got {x.shape}")
            return torch.split(x, self.c, dim=1)
        raise TypeError(f"P3ASFFLiteFuse expects [p4_up,p3] or concat tensor, got: {type(x)}")

    def forward(self, x: Any) -> torch.Tensor:
        p4_up, p3 = self._split_input(x)
        p4_up = self._maybe_align(p4_up, p3)
        if p4_up.shape[1] != self.c or p3.shape[1] != self.c:
            raise ValueError(f"Channel mismatch: p4_up={p4_up.shape} p3={p3.shape} expected C={self.c}")

        w_logits = self.weight_net(torch.cat((p3, p4_up), dim=1))
        w = torch.softmax(w_logits, dim=1)
        fused = w[:, 0:1] * p3 + w[:, 1:2] * p4_up
        refined = self.refine(fused)
        p3_out = p3 + self.alpha * refined

        # Lightweight monitoring: cache stats, flush by callbacks (no file I/O in forward).
        if self.training and torch.is_grad_enabled():
            if self.mon_every > 0 and self._step % self.mon_every == 0 and self._last_mon_step != self._step:
                with torch.no_grad():
                    row = {
                        "step": float(self._step),
                        "alpha_mean": float(self.alpha.mean().detach().cpu()),
                        "w_mean": float(w.mean().detach().cpu()),
                        "w_var": float(w.var(unbiased=False).detach().cpu()),
                        "fused_mean": float(p3_out.mean().detach().cpu()),
                        "fused_var": float(p3_out.var(unbiased=False).detach().cpu()),
                    }
                self._last_mon = row
                self._last_mon_step = self._step
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

    detect_f = getattr(detect, "f", None)
    if isinstance(detect_f, (list, tuple)) and detect_f:
        try:
            p3_out = int(detect_f[0])
            return p3_out - 1, detect
        except Exception:
            pass

    for i, layer in enumerate(seq):
        if layer.__class__.__name__ == "Concat" and list(getattr(layer, "f", [])) == [-1, 4]:
            return i, detect
    raise RuntimeError("Unable to locate P4->P3 fusion point in model.")


def apply(model: Any, cfg: Any) -> Any:
    """Apply enhance241 b1_v3 (ASFF-lite) to YOLO11 neck P4->P3 fusion point."""

    enable_b1 = bool(_deep_get(cfg, "enhance241", "b1", default=False))
    if not enable_b1:
        return model

    enhance_cfg = _deep_get(cfg, "enhance241", default={}) or {}
    if not isinstance(enhance_cfg, dict):
        enhance_cfg = {}
    version = str(enhance_cfg.get("b1_version", "v3")).lower()
    if version not in {"v3", "asff", "asff-lite", "asff_lite"}:
        raise ValueError(f"Unsupported enhance241.b1_version={version}. Only v3(asff-lite) is supported.")

    # Support both ultralytics.YOLO and raw DetectionModel-like inputs.
    yolo_obj = model
    if hasattr(model, "model") and hasattr(getattr(model, "model"), "model"):
        det_model = getattr(model, "model")
        seq = getattr(det_model, "model", None)
    else:
        det_model = model
        seq = getattr(det_model, "model", None)
    if seq is None:
        raise RuntimeError("enhance241.b1 requires an ultralytics YOLO/DetectionModel-like object with a .model sequence.")

    fuse_idx, _ = _locate_p4_to_p3_fuse(seq)
    if fuse_idx < 0 or fuse_idx >= len(seq):
        raise RuntimeError(f"Invalid fuse_idx={fuse_idx} for model length={len(seq)}")

    old = seq[fuse_idx]
    if isinstance(old, P3ASFFLiteFuse):
        print(f"[enhance241] b1 enabled: already patched at model.model[{fuse_idx}] -> P3ASFFLiteFuse(C={old.c})")
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
    alpha_init = _safe_float(enhance_cfg.get("b1_alpha_init", 0.05), 0.05)
    weight_hidden = int(_safe_float(enhance_cfg.get("b1_weight_hidden", 64), 64))
    refine = str(enhance_cfg.get("b1_refine", "dw"))
    upsample_mode = str(enhance_cfg.get("b1_upsample", "nearest")).lower()
    mon_every = int(_safe_float(enhance_cfg.get("b1_mon_every", 100), 100))

    fuse = P3ASFFLiteFuse(
        channels_per_branch=c,
        alpha_init=alpha_init,
        weight_hidden=weight_hidden,
        refine=refine,
        upsample_mode=upsample_mode,
        mon_every=mon_every,
    )

    # Preserve Ultralytics layer meta attributes used by its forward graph.
    for attr in ("i", "f", "type"):
        if hasattr(old, attr):
            setattr(fuse, attr, getattr(old, attr))

    seq[fuse_idx] = fuse

    print(
        f"[enhance241] b1 enabled: patched neck P4->P3 fuse at model.model[{fuse_idx}] "
        f"-> P3ASFFLiteFuse(C={c}, alpha_init={alpha_init})"
    )

    # Lightweight monitor via Ultralytics callbacks (train only).
    if hasattr(yolo_obj, "add_callback") and not getattr(yolo_obj, "_enhance241_b1_callbacks", False):
        setattr(yolo_obj, "_enhance241_b1_callbacks", True)

        train_state: Dict[str, Any] = {"epoch": None}

        def _write_csv_row(path: Path, row: Dict[str, Any], fieldnames: List[str]) -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            exists = path.exists()
            with open(path, "a", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                if not exists:
                    w.writeheader()
                w.writerow(row)

        def on_train_epoch_end(trainer: Any) -> None:
            train_state["epoch"] = getattr(trainer, "epoch", None)

        def on_train_batch_end(trainer: Any) -> None:
            if not _is_rank0():
                return
            save_dir = Path(getattr(trainer, "save_dir", ""))
            if not save_dir:
                return
            row = fuse.pop_monitor()
            if not row:
                return
            row_out = dict(row)
            row_out["epoch"] = train_state.get("epoch")
            fieldnames = ["epoch", "step", "alpha_mean", "w_mean", "w_var", "fused_mean", "fused_var"]
            _write_csv_row(save_dir / "enhance241_b1_v3_monitor.csv", row_out, fieldnames)
            print(
                f"[enhance241] b1_v3 mon step={int(row_out['step'])} alpha={row_out['alpha_mean']:.4f} "
                f"w_mean={row_out['w_mean']:.4f} w_var={row_out['w_var']:.6f} "
                f"fused_mean={row_out['fused_mean']:.4f} fused_var={row_out['fused_var']:.6f}"
            )

        yolo_obj.add_callback("on_train_epoch_end", on_train_epoch_end)
        yolo_obj.add_callback("on_train_batch_end", on_train_batch_end)

    return yolo_obj
