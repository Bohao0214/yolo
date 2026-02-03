from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: Optional[int] = None):
        super().__init__()
        pad = k // 2 if p is None else p
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, pad, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class WeightedAdd(nn.Module):
    def __init__(self, n: int, eps: float = 1e-4):
        super().__init__()
        self.w = nn.Parameter(torch.ones(n))
        self.eps = eps

    def forward(self, xs: List[Optional[torch.Tensor]]) -> torch.Tensor:
        parts: List[torch.Tensor] = [x for x in xs if x is not None]
        if len(parts) == 1:
            return parts[0]
        w = F.relu(self.w[: len(parts)])
        w = w / (w.sum() + self.eps)
        out = 0.0
        for i, x in enumerate(parts):
            out = out + w[i] * x
        return out


class SEBlock(nn.Module):
    def __init__(self, channels: int, r: int = 4):
        super().__init__()
        hidden = max(1, channels // r)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, hidden, 1, bias=True)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(hidden, channels, 1, bias=True)
        self.gate = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.pool(x)
        y = self.act(self.fc1(y))
        y = self.gate(self.fc2(y))
        return x * y


class CARAFE(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: Optional[int] = None,
        scale: int = 2,
        kernel: int = 5,
        up_kernel: int = 3,
        comp_channels: int = 64,
    ):
        super().__init__()
        out_ch = int(out_ch or in_ch)
        self.scale = int(scale)
        self.kernel = int(kernel)
        self.up_kernel = int(up_kernel)
        self.comp = ConvBNAct(in_ch, comp_channels, k=1, s=1, p=0)
        self.enc = nn.Conv2d(
            comp_channels,
            self.scale * self.scale * self.up_kernel * self.up_kernel,
            kernel_size=self.kernel,
            padding=self.kernel // 2,
            bias=True,
        )
        self.out_conv = ConvBNAct(in_ch, out_ch, k=1, s=1, p=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        kernel = self.enc(self.comp(x))
        kernel = F.pixel_shuffle(kernel, self.scale)
        kernel = kernel.view(b, self.up_kernel * self.up_kernel, h * self.scale, w * self.scale)
        kernel = F.softmax(kernel, dim=1)

        x = self.out_conv(x)
        unfold = F.unfold(x, kernel_size=self.up_kernel, padding=self.up_kernel // 2)
        unfold = unfold.view(b, c, self.up_kernel * self.up_kernel, h, w)
        unfold = unfold.repeat_interleave(self.scale, dim=3).repeat_interleave(self.scale, dim=4)

        out = (unfold * kernel.unsqueeze(1)).sum(dim=2)
        return out


class CarafeUpsample(nn.Module):
    def __init__(
        self,
        scale: int = 2,
        kernel: int = 5,
        up_kernel: int = 3,
        comp_channels: int = 64,
    ):
        super().__init__()
        self.scale = int(scale)
        self.kernel = int(kernel)
        self.up_kernel = int(up_kernel)
        if hasattr(nn, "LazyConv2d"):
            self.comp = nn.LazyConv2d(comp_channels, kernel_size=1, bias=False)
        else:
            self.comp = None
        self.bn = nn.BatchNorm2d(comp_channels)
        self.act = nn.SiLU()
        self.enc = nn.Conv2d(
            comp_channels,
            self.scale * self.scale * self.up_kernel * self.up_kernel,
            kernel_size=self.kernel,
            padding=self.kernel // 2,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.comp is None:
            return F.interpolate(x, scale_factor=self.scale, mode="nearest")
        b, c, h, w = x.shape
        kernel = self.enc(self.act(self.bn(self.comp(x))))
        kernel = F.pixel_shuffle(kernel, self.scale)
        kernel = kernel.view(b, self.up_kernel * self.up_kernel, h * self.scale, w * self.scale)
        kernel = F.softmax(kernel, dim=1)

        unfold = F.unfold(x, kernel_size=self.up_kernel, padding=self.up_kernel // 2)
        unfold = unfold.view(b, c, self.up_kernel * self.up_kernel, h, w)
        unfold = unfold.repeat_interleave(self.scale, dim=3).repeat_interleave(self.scale, dim=4)
        out = (unfold * kernel.unsqueeze(1)).sum(dim=2)
        return out


def _try_get_dcn() -> Optional[Any]:
    try:
        from torchvision.ops import DeformConv2d  # type: ignore

        return DeformConv2d
    except Exception:
        return None


class DCNConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: Optional[int] = None):
        super().__init__()
        pad = k // 2 if p is None else p
        DeformConv2d = _try_get_dcn()
        if DeformConv2d is None:
            self.fallback = ConvBNAct(in_ch, out_ch, k=k, s=s, p=pad)
            self.offset = None
            self.dcn = None
        else:
            self.fallback = None
            self.offset = nn.Conv2d(in_ch, 2 * k * k, kernel_size=3, stride=s, padding=1)
            self.dcn = DeformConv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=pad, bias=False)
            self.bn = nn.BatchNorm2d(out_ch)
            self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.fallback is not None:
            return self.fallback(x)
        offset = self.offset(x)
        x = self.dcn(x, offset)
        return self.act(self.bn(x))


class BiFPNBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        levels: int,
        use_carafe: bool,
        carafe_cfg: Dict[str, Any],
        dcn_scales: Iterable[int],
        dcn_kernel: int,
    ):
        super().__init__()
        self.levels = int(levels)
        self.up = nn.ModuleList()
        for _ in range(self.levels - 1):
            if use_carafe:
                self.up.append(
                    CARAFE(
                        channels,
                        channels,
                        scale=2,
                        kernel=int(carafe_cfg.get("kernel", 5)),
                        up_kernel=int(carafe_cfg.get("up_kernel", 3)),
                        comp_channels=int(carafe_cfg.get("comp_channels", 64)),
                    )
                )
            else:
                self.up.append(nn.Upsample(scale_factor=2, mode="nearest"))

        self.down = nn.ModuleList()
        for i in range(self.levels - 1):
            if i + 1 in dcn_scales:
                self.down.append(DCNConv(channels, channels, k=dcn_kernel, s=2))
            else:
                self.down.append(ConvBNAct(channels, channels, k=3, s=2))

        self.w1 = nn.ModuleList([WeightedAdd(2) for _ in range(self.levels - 1)])
        self.w2 = nn.ModuleList([WeightedAdd(3) for _ in range(self.levels - 1)])

        self.out_td = nn.ModuleList()
        self.out_bu = nn.ModuleList()
        for i in range(self.levels):
            if i in dcn_scales:
                self.out_td.append(DCNConv(channels, channels, k=dcn_kernel, s=1))
                self.out_bu.append(DCNConv(channels, channels, k=dcn_kernel, s=1))
            else:
                self.out_td.append(ConvBNAct(channels, channels, k=3, s=1))
                self.out_bu.append(ConvBNAct(channels, channels, k=3, s=1))

    def forward(self, feats: List[torch.Tensor]) -> List[torch.Tensor]:
        td: List[torch.Tensor] = [None for _ in range(self.levels)]  # type: ignore
        td[-1] = feats[-1]
        for i in range(self.levels - 2, -1, -1):
            up = self.up[i](td[i + 1])
            td[i] = self.out_td[i](self.w1[i]([feats[i], up]))

        out: List[torch.Tensor] = [None for _ in range(self.levels)]  # type: ignore
        out[0] = td[0]
        for i in range(1, self.levels):
            down = self.down[i - 1](out[i - 1])
            out[i] = self.out_bu[i](self.w2[i - 1]([feats[i], td[i], down]))
        return out


class BiFPN(nn.Module):
    def __init__(
        self,
        in_channels: List[int],
        out_channels: int,
        depth: int,
        use_carafe: bool,
        carafe_cfg: Dict[str, Any],
        dcn_scales: Iterable[int],
        dcn_kernel: int,
    ):
        super().__init__()
        if hasattr(nn, "LazyConv2d"):
            self.in_convs = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.LazyConv2d(out_channels, kernel_size=1, bias=False),
                        nn.BatchNorm2d(out_channels),
                        nn.SiLU(),
                    )
                    for _ in in_channels
                ]
            )
        else:
            self.in_convs = nn.ModuleList(
                [ConvBNAct(ch, out_channels, k=1, s=1, p=0) for ch in in_channels]
            )
        self.blocks = nn.ModuleList(
            [
                BiFPNBlock(
                    out_channels,
                    levels=len(in_channels),
                    use_carafe=use_carafe,
                    carafe_cfg=carafe_cfg,
                    dcn_scales=dcn_scales,
                    dcn_kernel=dcn_kernel,
                )
                for _ in range(max(1, int(depth)))
            ]
        )

    def forward(self, feats: List[torch.Tensor]) -> List[torch.Tensor]:
        x = [conv(f) for conv, f in zip(self.in_convs, feats)]
        for block in self.blocks:
            x = block(x)
        return x


class DyHeadLiteBlock(nn.Module):
    def __init__(self, channels: List[int]):
        super().__init__()
        self.levels = len(channels)
        self.spatial = nn.ModuleList([nn.Conv2d(ch, 1, 1) for ch in channels])
        self.channel = nn.ModuleList([SEBlock(ch) for ch in channels])
        self.scale = nn.Parameter(torch.ones(self.levels, 3))
        self.up_proj = nn.ModuleList()
        self.down_proj = nn.ModuleList()
        for ch in channels:
            if hasattr(nn, "LazyConv2d"):
                bn_cls = getattr(nn, "LazyBatchNorm2d", None)
                bn = bn_cls(ch) if bn_cls is not None else nn.BatchNorm2d(ch)
                proj = nn.Sequential(nn.LazyConv2d(ch, kernel_size=1, bias=False), bn, nn.SiLU())
            else:
                proj = nn.Identity()
            self.up_proj.append(proj)
            if hasattr(nn, "LazyConv2d"):
                bn_cls = getattr(nn, "LazyBatchNorm2d", None)
                bn = bn_cls(ch) if bn_cls is not None else nn.BatchNorm2d(ch)
                proj = nn.Sequential(nn.LazyConv2d(ch, kernel_size=1, bias=False), bn, nn.SiLU())
            else:
                proj = nn.Identity()
            self.down_proj.append(proj)

    def forward(self, feats: List[torch.Tensor]) -> List[torch.Tensor]:
        outs: List[torch.Tensor] = []
        for i, feat in enumerate(feats):
            x = feat * torch.sigmoid(self.spatial[i](feat))
            x = self.channel[i](x)

            up = None
            down = None
            if i < self.levels - 1:
                up = F.interpolate(feats[i + 1], size=feat.shape[-2:], mode="nearest")
                up = self.up_proj[i](up)
            if i > 0:
                down = F.max_pool2d(feats[i - 1], kernel_size=2, stride=2)
                if down.shape[-2:] != feat.shape[-2:]:
                    down = F.interpolate(down, size=feat.shape[-2:], mode="nearest")
                down = self.down_proj[i](down)

            weights = F.relu(self.scale[i])
            if up is None and down is None:
                parts = [x]
                w = weights[:1]
            elif up is None:
                parts = [x, down]
                w = weights[:2]
            elif down is None:
                parts = [x, up]
                w = torch.stack([weights[0], weights[2]])
            else:
                parts = [x, down, up]
                w = weights
            w = w / (w.sum() + 1e-4)
            fused = 0.0
            for j, p in enumerate(parts):
                fused = fused + w[j] * p
            outs.append(fused)
        return outs


class DyHeadLite(nn.Module):
    def __init__(self, channels: List[int], blocks: int = 1):
        super().__init__()
        self.blocks = nn.ModuleList([DyHeadLiteBlock(channels) for _ in range(max(1, int(blocks)))])

    def forward(self, feats: List[torch.Tensor]) -> List[torch.Tensor]:
        x = feats
        for block in self.blocks:
            x = block(x)
        return x


class TaskAlignedHead(nn.Module):
    def __init__(self, channels: List[int], blocks: int = 1):
        super().__init__()
        self.cls_towers = nn.ModuleList()
        self.reg_towers = nn.ModuleList()
        self.align = nn.ModuleList()
        for ch in channels:
            cls_layers = [ConvBNAct(ch, ch, k=3, s=1) for _ in range(max(1, int(blocks)))]
            reg_layers = [ConvBNAct(ch, ch, k=3, s=1) for _ in range(max(1, int(blocks)))]
            self.cls_towers.append(nn.Sequential(*cls_layers))
            self.reg_towers.append(nn.Sequential(*reg_layers))
            self.align.append(nn.Conv2d(ch, ch, 1))

    def forward(self, feats: List[torch.Tensor]) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        reg_feats: List[torch.Tensor] = []
        cls_feats: List[torch.Tensor] = []
        for i, feat in enumerate(feats):
            align = torch.sigmoid(self.align[i](feat))
            cls_feats.append(self.cls_towers[i](feat) * align)
            reg_feats.append(self.reg_towers[i](feat) * align)
        return reg_feats, cls_feats


class NeckHeadEnhancer(nn.Module):
    def __init__(self, cfg: Dict[str, Any], channels: List[int]):
        super().__init__()
        self.cfg = cfg
        self.channels = channels
        self.levels = len(channels)

        self.enable_p2 = bool(cfg.get("enable_p2", False))
        self.p2_source = str(cfg.get("p2_source", "auto")).lower()
        self.p2_select = str(cfg.get("p2_backbone_select", "first")).lower()
        self.p2_pan_fuse = bool(cfg.get("p2_pan_fuse", True))
        self.enable_bifpn = bool(cfg.get("enable_bifpn", False))
        self.enable_carafe = bool(cfg.get("enable_carafe", False))
        self.enable_dcn = bool(cfg.get("enable_dcn", False))
        self.dcn_scales = [int(i) for i in cfg.get("dcn_scales", [])]
        self.dcn_kernel = int(cfg.get("dcn_kernel", 3))
        self._warned_p2_fallback = False
        self.dcn_preh = None
        if self.enable_dcn and not self.enable_p2 and not self.enable_bifpn:
            blocks = []
            for i, ch in enumerate(channels):
                if i in self.dcn_scales:
                    blocks.append(DCNConv(ch, ch, k=self.dcn_kernel, s=1))
                else:
                    blocks.append(nn.Identity())
            self.dcn_preh = nn.ModuleList(blocks)

        carafe_cfg = {
            "kernel": int(cfg.get("carafe_kernel", 5)),
            "up_kernel": int(cfg.get("carafe_up_kernel", 3)),
            "comp_channels": int(cfg.get("carafe_comp_channels", 64)),
        }
        self.carafe_cfg = carafe_cfg

        self.p2_up = None
        self.p2_conv = None
        self.p2_align_backbone = None
        self.p3_down = None
        self.p3_fuse = None
        if self.enable_p2:
            if self.enable_carafe:
                self.p2_up = CARAFE(
                    channels[0],
                    channels[0],
                    scale=2,
                    kernel=carafe_cfg["kernel"],
                    up_kernel=carafe_cfg["up_kernel"],
                    comp_channels=carafe_cfg["comp_channels"],
                )
            else:
                self.p2_up = nn.Upsample(scale_factor=2, mode="nearest")
            if self.p2_source in {"auto", "backbone"}:
                if hasattr(nn, "LazyConv2d"):
                    bn_cls = getattr(nn, "LazyBatchNorm2d", None)
                    bn = bn_cls(channels[0]) if bn_cls is not None else nn.Identity()
                    self.p2_align_backbone = nn.Sequential(
                        nn.LazyConv2d(channels[0], kernel_size=1, bias=False),
                        bn,
                        nn.SiLU(),
                    )
                else:
                    self.p2_align_backbone = None
            if 0 in self.dcn_scales and self.enable_dcn:
                self.p2_conv = DCNConv(channels[0], channels[0], k=self.dcn_kernel, s=1)
            else:
                self.p2_conv = ConvBNAct(channels[0], channels[0], k=3, s=1)
            if self.p2_pan_fuse and not self.enable_bifpn:
                if 1 in self.dcn_scales and self.enable_dcn:
                    self.p3_down = DCNConv(channels[0], channels[0], k=self.dcn_kernel, s=2)
                else:
                    self.p3_down = ConvBNAct(channels[0], channels[0], k=3, s=2)
                self.p3_fuse = WeightedAdd(2)

        self.bifpn = None
        self.bifpn_out_channels = None
        self.bifpn_out = None
        if self.enable_bifpn:
            out_ch = int(cfg.get("bifpn_channels", 0)) or channels[0]
            if out_ch != channels[0]:
                out_ch = channels[0]
            depth = int(cfg.get("bifpn_depth", 1))
            dcn_scales = self.dcn_scales if self.enable_dcn else []
            self.bifpn = BiFPN(
                in_channels=channels,
                out_channels=out_ch,
                depth=depth,
                use_carafe=self.enable_carafe,
                carafe_cfg=carafe_cfg,
                dcn_scales=dcn_scales,
                dcn_kernel=self.dcn_kernel,
            )
            self.bifpn_out_channels = out_ch
            self.bifpn_out = nn.ModuleList(
                [
                    ConvBNAct(out_ch, ch, k=1, s=1, p=0) if ch != out_ch else nn.Identity()
                    for ch in channels
                ]
            )

        head_type = str(cfg.get("head_type", "base")).lower()
        blocks = int(cfg.get("head_blocks", 1))
        if self.bifpn_out_channels is not None:
            head_channels = [int(self.bifpn_out_channels) for _ in channels]
        else:
            head_channels = channels
        self.head_type = head_type
        self.dyhead = None
        self.tood = None
        if head_type == "dyhead":
            self.dyhead = DyHeadLite(head_channels, blocks=blocks)
        elif head_type == "tood":
            self.tood = TaskAlignedHead(head_channels, blocks=blocks)

    def forward(
        self, feats: List[torch.Tensor], backbone_p2: Optional[torch.Tensor] = None
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        x = feats
        if self.dcn_preh is not None:
            x = [blk(xi) for blk, xi in zip(self.dcn_preh, x)]
        if self.enable_p2:
            p3 = x[0]
            use_backbone = self.p2_source in {"auto", "backbone"} and backbone_p2 is not None
            if use_backbone and self.p2_align_backbone is None:
                if backbone_p2.shape[1] != self.channels[0]:
                    use_backbone = False
            if use_backbone:
                p2 = self.p2_align_backbone(backbone_p2) if self.p2_align_backbone else backbone_p2
            else:
                if self.p2_source in {"auto", "backbone"} and not self._warned_p2_fallback:
                    print("[enhance] P2 backbone feature not found; fallback to P3 upsample.")
                    self._warned_p2_fallback = True
                p2 = self.p2_conv(self.p2_up(p3))
            if self.p2_pan_fuse and self.p3_fuse is not None:
                p3 = self.p3_fuse([p3, self.p3_down(p2)])
            x = [p2, p3] + x[1:]
        if self.bifpn is not None:
            x = self.bifpn(x)
        feats = x
        reg_feats = x
        cls_feats = x

        if self.dyhead is not None:
            feats = self.dyhead(feats)
            reg_feats = feats
            cls_feats = feats
        if self.tood is not None:
            reg_feats, cls_feats = self.tood(feats)

        if self.bifpn_out is not None:
            feats = [proj(f) for proj, f in zip(self.bifpn_out, feats)]
            # Avoid triple-pass through BN (feats/reg/cls share same projection).
            reg_feats = feats
            cls_feats = feats

        return feats, reg_feats, cls_feats


def _get_detect_head(model: nn.Module):
    for m in model.modules():
        if m.__class__.__name__ == "Detect":
            return m
    return None


def _first_conv_in_channels(module: nn.Module) -> Optional[int]:
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            return int(m.in_channels)
    return None


def _resolve_channels(detect: nn.Module, idx: int) -> Optional[int]:
    for attr in ("cv2", "cv3"):
        try:
            mod = getattr(detect, attr)[idx]
        except Exception:
            continue
        ch = _first_conv_in_channels(mod)
        if ch is not None:
            return ch
    return None


def _expand_stride(detect: nn.Module) -> None:
    stride = getattr(detect, "stride", None)
    if stride is None:
        return
    if isinstance(stride, (list, tuple)):
        base = [float(s) for s in stride]
        new_stride = [base[0] / 2.0] + base
        detect.stride = new_stride
        return
    if torch.is_tensor(stride):
        base = stride.detach().cpu().tolist()
        new_stride = [base[0] / 2.0] + base
        detect.stride = torch.tensor(new_stride, device=stride.device)


def _register_p2_hook(core: nn.Module, select: str) -> None:
    if getattr(core, "_enhance_p2_hooked", False):
        return

    def pre_hook(module, inputs):
        if not inputs:
            return
        x = inputs[0]
        if torch.is_tensor(x):
            module._enhance_input_hw = (int(x.shape[-2]), int(x.shape[-1]))
            module._enhance_p2 = None

    def fwd_hook(module, inputs, output):
        if not torch.is_tensor(output):
            return
        input_hw = getattr(core, "_enhance_input_hw", None)
        if not input_hw:
            return
        target = (input_hw[0] // 4, input_hw[1] // 4)
        if output.shape[-2:] != target:
            return
        if select == "first":
            if getattr(core, "_enhance_p2", None) is None:
                core._enhance_p2 = output
        else:
            core._enhance_p2 = output

    core.register_forward_pre_hook(pre_hook)
    if hasattr(core, "model") and isinstance(getattr(core, "model"), nn.ModuleList):
        for m in core.model:
            m.register_forward_hook(fwd_hook)
    else:
        for m in core.modules():
            if m is core:
                continue
            m.register_forward_hook(fwd_hook)
    core._enhance_p2_hooked = True


def _replace_upsample_with_carafe(model: nn.Module, cfg: Dict[str, Any]) -> None:
    if getattr(model, "_enhance_carafe_patched", False):
        return

    kernel = int(cfg.get("carafe_kernel", 5))
    up_kernel = int(cfg.get("carafe_up_kernel", 3))
    comp_channels = int(cfg.get("carafe_comp_channels", 64))

    def should_replace(m: nn.Module) -> bool:
        if not isinstance(m, nn.Upsample):
            return False
        scale = m.scale_factor
        if scale is None:
            return False
        if isinstance(scale, (tuple, list)):
            return all(float(s) == 2.0 for s in scale)
        return float(scale) == 2.0

    def replace(module: nn.Module) -> None:
        for name, child in list(module.named_children()):
            if should_replace(child):
                new_mod = CarafeUpsample(
                    scale=2,
                    kernel=kernel,
                    up_kernel=up_kernel,
                    comp_channels=comp_channels,
                )
                # Preserve Ultralytics layer metadata (e.g., .f) so model forward doesn't break.
                for attr in ("f", "i", "type", "np", "args"):
                    if hasattr(child, attr):
                        setattr(new_mod, attr, getattr(child, attr))
                setattr(module, name, new_mod)
            else:
                replace(child)

    replace(model)
    model._enhance_carafe_patched = True


def apply_yolo_enhancements(yolo, cfg: Dict[str, Any]) -> None:
    enhance_cfg = cfg.get("enhance", {}) if isinstance(cfg, dict) else {}
    if not isinstance(enhance_cfg, dict):
        enhance_cfg = {}
    enable_p2 = bool(enhance_cfg.get("enable_p2", False))
    enable_bifpn = bool(enhance_cfg.get("enable_bifpn", False))
    enable_carafe = bool(enhance_cfg.get("enable_carafe", False))
    enable_dcn = bool(enhance_cfg.get("enable_dcn", False))
    head_type = str(enhance_cfg.get("head_type", "base")).lower()
    p2_source = str(enhance_cfg.get("p2_source", "auto")).lower()
    p2_select = str(enhance_cfg.get("p2_backbone_select", "first")).lower()
    replace_neck = bool(enhance_cfg.get("carafe_replace_neck", False))

    if not (enable_p2 or enable_bifpn or enable_carafe or enable_dcn or head_type in {"dyhead", "tood"}):
        return

    model = getattr(yolo, "model", None)
    if model is None:
        return
    if enable_p2 and p2_source in {"auto", "backbone"}:
        _register_p2_hook(model, p2_select)
    if replace_neck and enable_carafe and not enable_bifpn:
        _replace_upsample_with_carafe(model, enhance_cfg)
    detect = _get_detect_head(model)
    if detect is None:
        return

    if getattr(detect, "_enhance_applied", False):
        return

    channels: List[int]
    if hasattr(detect, "ch") and isinstance(getattr(detect, "ch"), (list, tuple)):
        channels = [int(c) for c in detect.ch]
    else:
        channels = []
        for i in range(int(getattr(detect, "nl", 0))):
            ch = _resolve_channels(detect, i)
            if ch is None:
                return
            channels.append(ch)

    if enable_p2:
        if not getattr(detect, "_p2_added", False):
            try:
                detect.cv2.insert(0, copy.deepcopy(detect.cv2[0]))
                detect.cv3.insert(0, copy.deepcopy(detect.cv3[0]))
                detect.nl = len(detect.cv2)
                detect._p2_added = True
                _expand_stride(detect)
                if hasattr(detect, "ch"):
                    detect.ch = [channels[0]] + channels
            except Exception:
                return
        channels = [channels[0]] + channels

    enhancer = NeckHeadEnhancer(enhance_cfg, channels)
    detect.enhancer = enhancer

    if getattr(detect, "_enhance_patched", False):
        detect._enhance_applied = True
        return

    def forward_head(self, x, box_head=None, cls_head=None):
        backbone_p2 = getattr(model, "_enhance_p2", None)
        if hasattr(self, "enhancer"):
            feats, reg_feats, cls_feats = self.enhancer(x, backbone_p2=backbone_p2)
        else:
            feats = x
            reg_feats = x
            cls_feats = x
        if hasattr(model, "_enhance_p2"):
            model._enhance_p2 = None

        if box_head is None or cls_head is None:
            return {}
        bs = feats[0].shape[0]
        boxes = torch.cat(
            [box_head[i](reg_feats[i]).view(bs, 4 * self.reg_max, -1) for i in range(self.nl)],
            dim=-1,
        )
        scores = torch.cat(
            [cls_head[i](cls_feats[i]).view(bs, self.nc, -1) for i in range(self.nl)],
            dim=-1,
        )
        return dict(boxes=boxes, scores=scores, feats=feats)

    detect.forward_head = forward_head.__get__(detect, detect.__class__)
    detect._enhance_patched = True
    detect._enhance_applied = True
