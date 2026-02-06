from __future__ import annotations

# 路径: third_party/yolo11/enhance241/yolo11_241b2.py
# 作用: enhance241 b2 v2（P4->P3 对齐式门控融合，feature alignment + gated inject）

from typing import Any, List, Optional, Tuple

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


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _logit(p: float) -> float:
    p = float(p)
    p = max(1e-6, min(p, 1 - 1e-6))
    return float(torch.log(torch.tensor(p / (1 - p))).item())


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


class P4P3GateAlignFuse(torch.nn.Module):
    """2.4.1 b2 v2: 对齐式门控融合（P4->P3）。

    - 输入: Concat 张量（2C）或 [p4_up, p3]
    - 对齐: align=off / flow / dcn（若 dcn 不可用则报错）
    - 门控融合: p3_out = p3 + gate * F([p3, p4_aligned])
      gate 为 sigmoid 参数化，可通过 b2_safe / b2_strong 设置初值
    - 输出: cat([p4_aligned, p3_out]) 维持下游结构
    """

    def __init__(
        self,
        channels_per_branch: int,
        align: str = "flow",
        gate_init: float = 0.05,
        flow_max: float = 2.0,
        refine: str = "dw",
        upsample_mode: str = "nearest",
        debug_stats: bool = False,
    ) -> None:
        super().__init__()
        self.c = int(channels_per_branch)
        if self.c <= 0:
            raise ValueError(f"channels_per_branch must be positive, got: {channels_per_branch}")

        self.upsample_mode = str(upsample_mode).lower()
        self.align = str(align).lower()
        self.flow_max = float(flow_max)
        self.debug_stats = bool(debug_stats)
        self._debug_printed = False
        self._align_logged = False

        refine = str(refine).lower()
        if refine == "conv":
            self.refine = _Conv3x3(self.c)
        else:
            self.refine = _DWSeparableConv(self.c)

        self.fuse = torch.nn.Conv2d(self.c * 2, self.c, kernel_size=1, stride=1, padding=0, bias=True)
        self.gate = torch.nn.Parameter(torch.full((1, self.c, 1, 1), _logit(float(gate_init)), dtype=torch.float32))

        self._dcn = None
        self._offset = None
        self._flow = None

        if self.align == "dcn":
            try:
                from torchvision.ops import DeformConv2d  # type: ignore

                self._offset = torch.nn.Conv2d(self.c * 2, 18, kernel_size=3, stride=1, padding=1, bias=True)
                self._dcn = DeformConv2d(self.c, self.c, kernel_size=3, stride=1, padding=1, bias=True)
            except Exception as exc:
                raise RuntimeError(f"enhance241.b2 align=dcn requested but DCN unavailable: {exc}") from exc
        elif self.align == "flow":
            self._flow = torch.nn.Conv2d(self.c * 2, 2, kernel_size=3, stride=1, padding=1, bias=True)
        elif self.align == "off":
            pass
        else:
            raise ValueError(f"Unknown b2_align={self.align}, must be one of: dcn|flow|off")

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
        raise TypeError(f"P4P3GateAlignFuse expects [p4_up,p3] or concat tensor, got: {type(x)}")

    def _align(self, p4_up: torch.Tensor, p3: torch.Tensor) -> torch.Tensor:
        if self.align == "dcn" and self._dcn is not None and self._offset is not None:
            offset = self._offset(torch.cat((p3, p4_up), dim=1))
            return self._dcn(p4_up, offset)
        if self.align == "flow" and self._flow is not None:
            flow = torch.tanh(self._flow(torch.cat((p3, p4_up), dim=1))) * self.flow_max
            b, _, h, w = flow.shape
            grid_y, grid_x = torch.meshgrid(
                torch.linspace(-1.0, 1.0, h, device=flow.device, dtype=flow.dtype),
                torch.linspace(-1.0, 1.0, w, device=flow.device, dtype=flow.dtype),
                indexing="ij",
            )
            base = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0).expand(b, -1, -1, -1)
            if w > 1 and h > 1:
                flow_norm = torch.stack(
                    (
                        flow[:, 0] / ((w - 1) / 2.0),
                        flow[:, 1] / ((h - 1) / 2.0),
                    ),
                    dim=-1,
                )
            else:
                flow_norm = torch.stack((flow[:, 0] * 0.0, flow[:, 1] * 0.0), dim=-1)
            grid = base + flow_norm
            return torch.nn.functional.grid_sample(
                p4_up, grid, mode="bilinear", padding_mode="border", align_corners=True
            )
        return p4_up

    def forward(self, x: Any) -> torch.Tensor:
        p4_up, p3 = self._split_input(x)
        p4_up = self._maybe_align(p4_up, p3)
        if p4_up.shape[1] != self.c or p3.shape[1] != self.c:
            raise ValueError(f"Channel mismatch: p4_up={p4_up.shape} p3={p3.shape} expected C={self.c}")

        if not self._align_logged:
            self._align_logged = True
            if self.align == "off":
                print("[enhance241] b2 v2 align: disabled -> fallback conv")
            else:
                print(f"[enhance241] b2 v2 align: {self.align}")

        p4_aligned = self._align(p4_up, p3)
        fused = self.refine(self.fuse(torch.cat((p3, p4_aligned), dim=1)))
        gate = torch.sigmoid(self.gate)
        p3_out = p3 + gate * fused

        if self.debug_stats and not self._debug_printed:
            self._debug_printed = True
            with torch.no_grad():
                stats = {
                    "p3_in": (float(p3.mean().cpu()), float(p3.var(unbiased=False).cpu())),
                    "p4_aligned": (float(p4_aligned.mean().cpu()), float(p4_aligned.var(unbiased=False).cpu())),
                    "fused": (float(fused.mean().cpu()), float(fused.var(unbiased=False).cpu())),
                    "p3_out": (float(p3_out.mean().cpu()), float(p3_out.var(unbiased=False).cpu())),
                    "gate": (
                        float(gate.mean().cpu()),
                        float(gate.min().cpu()),
                        float(gate.max().cpu()),
                    ),
                }
            print(
                "[enhance241] b2 v2 debug_stats: "
                f"p3_in(mean={stats['p3_in'][0]:.6f},var={stats['p3_in'][1]:.6f}) "
                f"p4_aligned(mean={stats['p4_aligned'][0]:.6f},var={stats['p4_aligned'][1]:.6f}) "
                f"fused(mean={stats['fused'][0]:.6f},var={stats['fused'][1]:.6f}) "
                f"p3_out(mean={stats['p3_out'][0]:.6f},var={stats['p3_out'][1]:.6f}) "
                f"gate(mean={stats['gate'][0]:.4f},min={stats['gate'][1]:.4f},max={stats['gate'][2]:.4f})"
            )

        return torch.cat((p4_aligned, p3_out), dim=1)


class P3FuseChain(torch.nn.Module):
    """Chain two P4->P3 fuse modules without changing downstream shape."""

    def __init__(self, first: torch.nn.Module, second: torch.nn.Module, channels_per_branch: int) -> None:
        super().__init__()
        self.first = first
        self.second = second
        self.c = int(channels_per_branch)

    def forward(self, x: Any) -> torch.Tensor:
        y = self.first(x)
        if not isinstance(y, torch.Tensor) or y.shape[1] != self.c * 2:
            raise ValueError(f"P3FuseChain expects concat tensor with 2C channels, got {getattr(y, 'shape', None)}")
        p4_up, p3 = torch.split(y, self.c, dim=1)
        return self.second([p4_up, p3])


def _concat_candidates(seq: Any) -> List[str]:
    cand = []
    for i, layer in enumerate(seq):
        if layer.__class__.__name__ == "Concat":
            cand.append(f"idx={i}, f={getattr(layer, 'f', None)}")
    return cand


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
            return p3_out - 1, detect
        except Exception:
            pass

    for i, layer in enumerate(seq):
        if layer.__class__.__name__ == "Concat" and list(getattr(layer, "f", [])) == [-1, 4]:
            return i, detect
    candidates = _concat_candidates(seq)
    raise RuntimeError(f"Unable to locate P4->P3 fusion point. Concat candidates: {candidates}")


def apply(model: Any, cfg: Any) -> Any:
    """Apply enhance241 b2 v2 to YOLO11 neck P4->P3 fusion point."""

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
    if isinstance(old, P4P3GateAlignFuse):
        return yolo_obj

    # b1/b2 chain: if b1 already patched, wrap b2 after it
    try:
        from third_party.yolo11.enhance241.yolo11_241b1 import P3ASFFLiteFuse  # type: ignore
    except Exception:
        P3ASFFLiteFuse = None  # type: ignore

    enhance_cfg = _deep_get(cfg, "enhance241", default={}) or {}
    if not isinstance(enhance_cfg, dict):
        enhance_cfg = {}

    align = str(enhance_cfg.get("b2_align", "flow")).lower()
    mode = str(enhance_cfg.get("b2_mode", "safe")).lower()
    gate_init = _safe_float(enhance_cfg.get("b2_gate_init", 0.0), 0.0)
    if gate_init <= 0:
        gate_init = 0.20 if mode == "strong" else 0.05
    flow_max = _safe_float(enhance_cfg.get("b2_flow_max", 2.0), 2.0)
    refine = str(enhance_cfg.get("b2_refine", "dw"))
    upsample_mode = str(enhance_cfg.get("b2_upsample", "nearest")).lower()
    debug_stats = bool(enhance_cfg.get("debug_stats", False))

    def _build_fuse(c: int) -> P4P3GateAlignFuse:
        return P4P3GateAlignFuse(
            channels_per_branch=c,
            align=align,
            gate_init=gate_init,
            flow_max=flow_max,
            refine=refine,
            upsample_mode=upsample_mode,
            debug_stats=debug_stats,
        )

    if P3ASFFLiteFuse is not None and isinstance(old, P3ASFFLiteFuse):
        c = int(getattr(old, "c", 0)) or None
        if not c:
            raise RuntimeError("Unable to infer channels from existing b1 fuse module.")
        fuse = _build_fuse(c)
        chain = P3FuseChain(old, fuse, c)
        for attr in ("i", "f", "type"):
            if hasattr(old, attr):
                setattr(chain, attr, getattr(old, attr))
        seq[fuse_idx] = chain
        print(
            f"[enhance241] b2 v2 enabled: chained after b1 at model.model[{fuse_idx}] "
            f"-> P4P3GateAlignFuse(align={align}, mode={mode}, gate_init={gate_init:.3f})"
        )
        return yolo_obj

    if old.__class__.__name__ != "Concat":
        raise RuntimeError(
            f"Expected Concat at P4->P3 fuse idx={fuse_idx}, got {old.__class__.__name__}. "
            f"Concat candidates: {_concat_candidates(seq)}"
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
    fuse = _build_fuse(c)
    for attr in ("i", "f", "type"):
        if hasattr(old, attr):
            setattr(fuse, attr, getattr(old, attr))
    seq[fuse_idx] = fuse

    num_params = sum(int(p.numel()) for p in fuse.parameters())
    gate_is_param = isinstance(fuse.gate, torch.nn.Parameter)
    gate_req_grad = bool(getattr(fuse.gate, "requires_grad", False))
    print(
        f"[enhance241] b2 v2 enabled: patched neck P4->P3 fuse at model.model[{fuse_idx}] "
        f"(orig={old.__class__.__name__}, f={getattr(old, 'f', None)}) -> "
        f"P4P3GateAlignFuse(align={align}, mode={mode}, gate_init={gate_init:.3f})"
    )
    print(
        f"[enhance241] b2 v2 params={num_params} gate_is_param={gate_is_param} gate_requires_grad={gate_req_grad}"
    )

    return yolo_obj
