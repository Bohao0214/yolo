from __future__ import annotations

# 路径: third_party/yolo11/enhance241/yolo11_241b2.py
# 作用: enhance241 b2（P4->P3 对齐式融合，feature alignment fusion）

from typing import Any, Optional, Tuple

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


class P4P3AlignFuse(torch.nn.Module):
    """2.4.1 b2: P4->P3 对齐式融合（feature alignment fusion）。

    - 对齐优先级：
        A) DCN (torchvision.ops.DeformConv2d)
        B) flow (grid_sample)
        C) off (fallback conv, 对齐禁用)
    - 融合：fused = Conv1x1([p3, p4_aligned]) -> Conv3x3(refine)
    - 残差写回：p3_out = p3 + beta * fused
    - 输出：cat([p4_up, p3_out]) 保持下游形状/顺序不变
    """

    def __init__(
        self,
        channels_per_branch: int,
        align: str = "dcn",
        beta_init: float = 0.05,
        flow_max: float = 2.0,
        refine: str = "dw",
        upsample_mode: str = "nearest",
    ) -> None:
        super().__init__()
        self.c = int(channels_per_branch)
        if self.c <= 0:
            raise ValueError(f"channels_per_branch must be positive, got: {channels_per_branch}")

        self.upsample_mode = str(upsample_mode).lower()
        self.align_req = str(align).lower()
        self.align_use = "off"
        self.align_reason: Optional[str] = None

        # refine
        refine = str(refine).lower()
        if refine == "conv":
            self.refine = _Conv3x3(self.c)
        else:
            self.refine = _DWSeparableConv(self.c)

        # fusion conv
        self.fuse = torch.nn.Conv2d(self.c * 2, self.c, kernel_size=1, stride=1, padding=0, bias=True)

        # learnable beta (ensure optimizer sees it)
        self.beta = torch.nn.Parameter(torch.tensor(float(beta_init), dtype=torch.float32))

        # alignment modules (created depending on availability)
        self.flow_max = float(flow_max)
        self._dcn = None
        self._offset = None
        self._flow = None

        if self.align_req == "dcn":
            try:
                from torchvision.ops import DeformConv2d  # type: ignore

                self._offset = torch.nn.Conv2d(self.c * 2, 18, kernel_size=3, stride=1, padding=1, bias=True)
                self._dcn = DeformConv2d(self.c, self.c, kernel_size=3, stride=1, padding=1, bias=True)
                self.align_use = "dcn"
            except Exception as exc:
                self.align_reason = f"DCN unavailable: {exc}"
                self.align_use = "flow"

        if self.align_req == "flow" or self.align_use == "flow":
            self._flow = torch.nn.Conv2d(self.c * 2, 2, kernel_size=3, stride=1, padding=1, bias=True)
            self.align_use = "flow"

        if self.align_req == "off":
            self.align_use = "off"

        self._align_logged = False

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
        raise TypeError(f"P4P3AlignFuse expects [p4_up,p3] or concat tensor, got: {type(x)}")

    def _align(self, p4_up: torch.Tensor, p3: torch.Tensor) -> torch.Tensor:
        if self.align_use == "dcn" and self._dcn is not None and self._offset is not None:
            offset = self._offset(torch.cat((p3, p4_up), dim=1))
            return self._dcn(p4_up, offset)
        if self.align_use == "flow" and self._flow is not None:
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
            if self.align_use == "off":
                print("[enhance241] b2 align: disabled -> fallback conv")
            elif self.align_use == "flow" and self.align_reason:
                print(f"[enhance241] b2 align: requested=dcn -> using=flow ({self.align_reason})")

        p4_aligned = self._align(p4_up, p3)
        fused = self.fuse(torch.cat((p3, p4_aligned), dim=1))
        fused = self.refine(fused)
        p3_out = p3 + self.beta * fused
        return torch.cat((p4_up, p3_out), dim=1)


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
    raise RuntimeError("Unable to locate P4->P3 fusion point in model.")


def apply(model: Any, cfg: Any) -> Any:
    """Apply 2.4.1 b2 enhancement (P4->P3 alignment fusion) to YOLO11."""

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
    if isinstance(old, P4P3AlignFuse):
        return yolo_obj

    # If b1 already patched the fuse point, chain b2 after it.
    try:
        from third_party.yolo11.enhance241.yolo11_241b1 import P3ASFFLiteFuse  # type: ignore
    except Exception:
        P3ASFFLiteFuse = None  # type: ignore

    enhance_cfg = _deep_get(cfg, "enhance241", default={}) or {}
    if not isinstance(enhance_cfg, dict):
        enhance_cfg = {}

    align = str(enhance_cfg.get("b2_align", "dcn")).lower()
    beta_init = _safe_float(enhance_cfg.get("b2_beta_init", 0.05), 0.05)
    flow_max = _safe_float(enhance_cfg.get("b2_flow_max", 2.0), 2.0)
    refine = str(enhance_cfg.get("b2_refine", "dw"))
    upsample_mode = str(enhance_cfg.get("b2_upsample", "nearest")).lower()

    if P3ASFFLiteFuse is not None and isinstance(old, P3ASFFLiteFuse):
        c = int(getattr(old, "c", 0)) or None
        if not c:
            raise RuntimeError("Unable to infer channels from existing b1 fuse module.")
        fuse = P4P3AlignFuse(
            channels_per_branch=c,
            align=align,
            beta_init=beta_init,
            flow_max=flow_max,
            refine=refine,
            upsample_mode=upsample_mode,
        )
        chain = P3FuseChain(old, fuse, c)
        for attr in ("i", "f", "type"):
            if hasattr(old, attr):
                setattr(chain, attr, getattr(old, attr))
        seq[fuse_idx] = chain
        print(
            f"[enhance241] b2 enabled: chained after b1 at model.model[{fuse_idx}] "
            f"-> P4P3AlignFuse(align={align}, beta_init={beta_init})"
        )
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
    fuse = P4P3AlignFuse(
        channels_per_branch=c,
        align=align,
        beta_init=beta_init,
        flow_max=flow_max,
        refine=refine,
        upsample_mode=upsample_mode,
    )

    for attr in ("i", "f", "type"):
        if hasattr(old, attr):
            setattr(fuse, attr, getattr(old, attr))

    seq[fuse_idx] = fuse

    print(
        f"[enhance241] b2 enabled: patched neck P4->P3 fuse at model.model[{fuse_idx}] "
        f"-> P4P3AlignFuse(align={align}, beta_init={beta_init})"
    )

    return yolo_obj
