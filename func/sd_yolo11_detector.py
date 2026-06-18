from __future__ import annotations

import sys
import threading
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from ultralytics import YOLO


_CACHE_LOCK = threading.Lock()
_DETECTOR_CACHE: Dict[Tuple[str, str, float, float, int, int], "SDYOLO11Detector"] = {}


def _default_weights_path() -> str:
    local_weight = Path(__file__).resolve().with_name("best.pt")
    if local_weight.exists():
        return str(local_weight)
    return "/home/ubuntu/hpproject/yolo/best.pt"


# -----------------------------
# Inline enhance241 core modules
# -----------------------------


class _DWSeparableConv(torch.nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.dw = torch.nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1, groups=in_ch, bias=True)
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
        s = self.block
        if h % s != 0 or w % s != 0:
            raise ValueError(f"SpatialToDepth needs H/W divisible by {s}, got {(h, w)}")
        x = x.view(b, c, h // s, s, w // s, s)
        x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
        return x.view(b, c * s * s, h // s, w // s)


class SPDConvDownsample(torch.nn.Module):
    def __init__(
        self,
        base_downsample: torch.nn.Module,
        in_ch: int,
        out_ch: int,
        pre_div: int = 4,
        refine: str = "dw",
        alpha_init: float = 0.05,
        alpha_cap: float = 0.5,
    ) -> None:
        super().__init__()
        self.enhance241_a4_base = base_downsample
        self.in_ch = int(in_ch)
        self.out_ch = int(out_ch)
        self.pre_div = int(max(1, pre_div))
        refine_mode = str(refine).lower()

        if self.out_ch > 0 and self.out_ch % self.pre_div == 0:
            pre_ch = self.out_ch // self.pre_div
        else:
            pre_ch = self.out_ch

        if refine_mode == "conv":
            self.enhance241_a4_pre = _Conv3x3(self.in_ch, pre_ch)
        else:
            self.enhance241_a4_pre = _DWSeparableConv(self.in_ch, pre_ch)
        self.enhance241_a4_s2d = _SpaceToDepth(2)
        self.enhance241_a4_post = torch.nn.Conv2d(pre_ch * 4, self.out_ch, kernel_size=1, stride=1, padding=0, bias=True)

        self.alpha_cap = float(max(1e-6, abs(alpha_cap)))
        alpha_init = float(max(-self.alpha_cap * 0.95, min(self.alpha_cap * 0.95, alpha_init)))
        alpha_ratio = alpha_init / self.alpha_cap
        self.enhance241_a4_alpha = torch.nn.Parameter(torch.atanh(torch.tensor(alpha_ratio, dtype=torch.float32)))
        torch.nn.init.zeros_(self.enhance241_a4_post.weight)
        if self.enhance241_a4_post.bias is not None:
            torch.nn.init.zeros_(self.enhance241_a4_post.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y_base = self.enhance241_a4_base(x)
        y_spd = self.enhance241_a4_pre(x)
        y_spd = self.enhance241_a4_s2d(y_spd)
        y_spd = self.enhance241_a4_post(y_spd)
        if y_spd.shape[-2:] != y_base.shape[-2:]:
            y_spd = F.interpolate(y_spd, size=y_base.shape[-2:], mode="nearest")
        alpha = torch.tanh(self.enhance241_a4_alpha.to(dtype=y_base.dtype, device=y_base.device)) * self.alpha_cap
        return y_base + alpha * y_spd


class A4DualDeltaSafe(SPDConvDownsample):
    def __init__(
        self,
        base_downsample: torch.nn.Module,
        in_ch: int,
        out_ch: int,
        a3_pre_div: int = 4,
        a4_refine: str = "dw",
        a4_order: int = 3,
        a4_alpha1_init: float = 0.05,
        a4_alpha1_cap: float = 0.5,
        a4_alpha2_init: float = 0.0,
        a4_alpha2_cap: float = 0.5,
        **_: Any,
    ) -> None:
        super().__init__(
            base_downsample=base_downsample,
            in_ch=in_ch,
            out_ch=out_ch,
            pre_div=a3_pre_div,
            refine=a4_refine,
            alpha_init=a4_alpha1_init,
            alpha_cap=a4_alpha1_cap,
        )
        self.a4_order = int(a4_order)
        self.a4_alpha2_init = float(a4_alpha2_init)
        self.a4_alpha2_cap = float(a4_alpha2_cap)


class CARAFECore(torch.nn.Module):
    def __init__(
        self,
        channels: int,
        scale: int = 2,
        kernel_size: int = 5,
        compress: int = 64,
        chunk_channels: int = 64,
        use_hf: bool = True,
        hf_kernel: int = 3,
        hf_gain_init: float = 0.1,
    ) -> None:
        super().__init__()
        c = int(channels)
        s = max(2, int(scale))
        k = max(3, int(kernel_size))
        if k % 2 == 0:
            k += 1
        cm = max(8, min(int(compress), c))

        self.channels = c
        self.scale = s
        self.kernel_size = k
        self.chunk_channels = max(8, int(chunk_channels))
        self.use_hf = bool(use_hf)
        self.hf_kernel = max(3, int(hf_kernel))
        if self.hf_kernel % 2 == 0:
            self.hf_kernel += 1

        self.comp = torch.nn.Conv2d(c, cm, kernel_size=1, stride=1, padding=0, bias=True)
        self.encoder = torch.nn.Conv2d(cm, (k * k) * (s * s), kernel_size=3, stride=1, padding=1, bias=True)
        self.hf_comp = torch.nn.Conv2d(c, cm, kernel_size=1, stride=1, padding=0, bias=True)
        self.hf_encoder = torch.nn.Conv2d(cm, (k * k) * (s * s), kernel_size=3, stride=1, padding=1, bias=True)
        self.out_proj = torch.nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0, bias=True)
        self.enhance241_b7_hf_gain = torch.nn.Parameter(torch.tensor(float(hf_gain_init), dtype=torch.float32))

        torch.nn.init.zeros_(self.hf_encoder.weight)
        if self.hf_encoder.bias is not None:
            torch.nn.init.zeros_(self.hf_encoder.bias)
        torch.nn.init.zeros_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            torch.nn.init.zeros_(self.out_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        s = self.scale
        k = self.kernel_size
        hs, ws = h * s, w * s

        kernel_logits = self.encoder(self.comp(x))
        hf_gate = torch.tanh(self.enhance241_b7_hf_gain.to(dtype=x.dtype, device=x.device))
        if self.use_hf:
            blur = F.avg_pool2d(x, kernel_size=self.hf_kernel, stride=1, padding=self.hf_kernel // 2, count_include_pad=False)
            hf = x - blur
            hf_logits = self.hf_encoder(self.hf_comp(hf))
            kernel_logits = kernel_logits + hf_gate * hf_logits

        kernel = F.pixel_shuffle(kernel_logits, upscale_factor=s)
        kernel = kernel.float()
        kernel = kernel - kernel.amax(dim=1, keepdim=True)
        kernel = torch.softmax(kernel, dim=1).to(dtype=x.dtype)

        x_up = F.interpolate(x, size=(hs, ws), mode="nearest")
        kflat = kernel.view(b, k * k, hs * ws).unsqueeze(1)
        out = x_up.new_zeros((b, c, hs, ws))

        step = max(1, int(self.chunk_channels))
        for c0 in range(0, c, step):
            c1 = min(c, c0 + step)
            part = x_up[:, c0:c1]
            patches = F.unfold(part, kernel_size=k, dilation=1, padding=k // 2, stride=1)
            patches = patches.view(b, c1 - c0, k * k, hs * ws)
            y_part = (patches * kflat).sum(dim=2).view(b, c1 - c0, hs, ws)
            out[:, c0:c1] = y_part

        return self.out_proj(out)


class CARAFEUpsampleSafe(torch.nn.Module):
    def __init__(
        self,
        base_module: torch.nn.Module,
        channels: int,
        scale: int = 2,
        kernel_size: int = 5,
        compress: int = 64,
        alpha_init: float = 0.05,
        alpha_cap: float = 0.5,
        chunk_channels: int = 64,
        use_hf: bool = True,
        hf_kernel: int = 3,
        hf_gain_init: float = 0.1,
    ) -> None:
        super().__init__()
        self.enhance241_b7_base = base_module
        self.enhance241_b7_carafe = CARAFECore(
            channels=channels,
            scale=scale,
            kernel_size=kernel_size,
            compress=compress,
            chunk_channels=chunk_channels,
            use_hf=use_hf,
            hf_kernel=hf_kernel,
            hf_gain_init=hf_gain_init,
        )
        self.alpha_cap = float(max(1e-6, abs(alpha_cap)))
        alpha_init = float(max(-self.alpha_cap * 0.95, min(self.alpha_cap * 0.95, alpha_init)))
        alpha_ratio = alpha_init / self.alpha_cap
        self.enhance241_b7_alpha = torch.nn.Parameter(torch.atanh(torch.tensor(alpha_ratio, dtype=torch.float32)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y_base = self.enhance241_b7_base(x)
        y_carafe = self.enhance241_b7_carafe(x)
        alpha = torch.tanh(self.enhance241_b7_alpha.to(dtype=y_base.dtype, device=y_base.device)) * self.alpha_cap
        return y_base + alpha * (y_carafe - y_base)


class D11ClsScoreCalib(torch.nn.Module):
    def __init__(
        self,
        base_cls_head: torch.nn.Module,
        temp_init: float = 1.0,
        t_min: float = 0.5,
        t_max: float = 4.0,
        bias_init: float = 0.0,
        head_stride: float = 8.0,
        scale_beta: float = 0.0,
        scale_lambda: float = 32.0,
        scale_threshold: float = 64.0,
        stride_gamma_mul: float = 4.0,
        score_domain: bool = True,
        alpha_init: float = 0.0,
        alpha_cap: float = 0.5,
    ) -> None:
        super().__init__()
        self.enhance241_d11_base_cls = base_cls_head
        self.enhance241_d11_temp = torch.nn.Parameter(torch.tensor(float(temp_init), dtype=torch.float32))
        self.enhance241_d11_bias = torch.nn.Parameter(torch.tensor(float(bias_init), dtype=torch.float32))
        self.t_min = float(t_min)
        self.t_max = float(t_max)
        self.head_stride = float(max(1.0, head_stride))
        self.scale_beta = float(scale_beta)
        self.scale_lambda = float(max(1e-6, scale_lambda))
        self.scale_threshold = float(max(1e-6, scale_threshold))
        self.stride_gamma_mul = float(max(1e-6, stride_gamma_mul))
        self.score_domain = bool(score_domain)
        self.alpha_cap = float(max(1e-6, abs(alpha_cap)))
        alpha_init = float(max(-self.alpha_cap * 0.95, min(self.alpha_cap * 0.95, alpha_init)))
        alpha_ratio = alpha_init / self.alpha_cap
        self.enhance241_d11_alpha_raw = torch.nn.Parameter(torch.atanh(torch.tensor(alpha_ratio, dtype=torch.float32)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.enhance241_d11_base_cls(x)
        t = torch.clamp(self.enhance241_d11_temp, min=self.t_min, max=self.t_max).to(dtype=logits.dtype, device=logits.device)
        b = self.enhance241_d11_bias.to(dtype=logits.dtype, device=logits.device)
        alpha = torch.tanh(self.enhance241_d11_alpha_raw.to(dtype=logits.dtype, device=logits.device)) * self.alpha_cap

        gamma = logits.new_tensor(self.head_stride * self.stride_gamma_mul)
        if float(gamma.item()) < self.scale_threshold:
            scale_bias = logits.new_tensor(self.scale_beta) * torch.exp(-gamma / self.scale_lambda)
        else:
            scale_bias = logits.new_tensor(0.0)

        logits_calib = logits / t + (b + scale_bias)
        if self.score_domain:
            score_base = torch.sigmoid(logits)
            score_calib = torch.sigmoid(logits_calib)
            score_out = score_base + alpha * (score_calib - score_base)
            score_out = torch.clamp(score_out, min=1e-5, max=1.0 - 1e-5)
            return torch.log(score_out / (1.0 - score_out))
        return logits + alpha * (logits_calib - logits)


class D6ScaleAwareCalib(D11ClsScoreCalib):
    def __init__(
        self,
        base_cls_head: torch.nn.Module,
        temp_init: float = 1.0,
        t_min: float = 0.5,
        t_max: float = 4.0,
        bias_init: float = 0.0,
        head_stride: float = 8.0,
        p3_bias: float = 0.2,
        score_domain: bool = True,
        alpha_init: float = 0.05,
        alpha_cap: float = 0.5,
    ) -> None:
        super().__init__(
            base_cls_head=base_cls_head,
            temp_init=temp_init,
            t_min=t_min,
            t_max=t_max,
            bias_init=bias_init,
            head_stride=head_stride,
            scale_beta=0.0,
            scale_lambda=1.0,
            scale_threshold=0.0,
            stride_gamma_mul=1.0,
            score_domain=score_domain,
            alpha_init=alpha_init,
            alpha_cap=alpha_cap,
        )
        self.enhance241_d6_p3_bias = float(p3_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.enhance241_d11_base_cls(x)
        t = torch.clamp(self.enhance241_d11_temp, min=self.t_min, max=self.t_max).to(dtype=logits.dtype, device=logits.device)
        alpha = torch.tanh(self.enhance241_d11_alpha_raw.to(dtype=logits.dtype, device=logits.device)) * self.alpha_cap
        is_p3_stride = abs(float(self.head_stride) - 8.0) < 0.5
        b_stride = logits.new_tensor(float(self.enhance241_d6_p3_bias if is_p3_stride else 0.0))
        logits_calib = logits / t + b_stride

        if self.score_domain:
            score_base = torch.sigmoid(logits)
            score_calib = torch.sigmoid(logits_calib)
            score_out = score_base + alpha * (score_calib - score_base)
            score_out = torch.clamp(score_out, min=1e-5, max=1.0 - 1e-5)
            return torch.log(score_out / (1.0 - score_out))
        return logits + alpha * (logits_calib - logits)


def _ensure_parent_pkg(fullname: str) -> types.ModuleType:
    if fullname in sys.modules:
        return sys.modules[fullname]
    mod = types.ModuleType(fullname)
    mod.__package__ = fullname
    mod.__path__ = []  # namespace-like package marker
    sys.modules[fullname] = mod
    if "." in fullname:
        parent_name, child_name = fullname.rsplit(".", 1)
        parent = _ensure_parent_pkg(parent_name)
        setattr(parent, child_name, mod)
    return mod


def _register_inline_enhance241_modules() -> None:
    _ensure_parent_pkg("third_party")
    _ensure_parent_pkg("third_party.yolo11")
    _ensure_parent_pkg("third_party.yolo11.enhance241")

    specs: List[Tuple[str, Dict[str, Any], List[type]]] = [
        (
            "third_party.yolo11.enhance241.yolo11_241a4",
            {
                "ENHANCE241_AUDIT_KEYS": ["enhance241_a4"],
                "_DWSeparableConv": _DWSeparableConv,
                "_Conv3x3": _Conv3x3,
                "_SpaceToDepth": _SpaceToDepth,
                "SPDConvDownsample": SPDConvDownsample,
                "A4DualDeltaSafe": A4DualDeltaSafe,
            },
            [_DWSeparableConv, _Conv3x3, _SpaceToDepth, SPDConvDownsample, A4DualDeltaSafe],
        ),
        (
            "third_party.yolo11.enhance241.yolo11_241b7",
            {
                "ENHANCE241_AUDIT_KEYS": ["enhance241_b7"],
                "CARAFECore": CARAFECore,
                "CARAFEUpsampleSafe": CARAFEUpsampleSafe,
            },
            [CARAFECore, CARAFEUpsampleSafe],
        ),
        (
            "third_party.yolo11.enhance241.yolo11_241d11",
            {
                "ENHANCE241_AUDIT_KEYS": ["enhance241_d11"],
                "D11ClsScoreCalib": D11ClsScoreCalib,
            },
            [D11ClsScoreCalib],
        ),
        (
            "third_party.yolo11.enhance241.yolo11_241d6",
            {
                "ENHANCE241_AUDIT_KEYS": ["enhance241_d6"],
                "D6ScaleAwareCalib": D6ScaleAwareCalib,
            },
            [D6ScaleAwareCalib],
        ),
        (
            "third_party.yolo11.enhance241.yolo11_241a3",
            {
                "ENHANCE241_AUDIT_KEYS": ["enhance241_a3"],
            },
            [],
        ),
    ]

    for module_name, attrs, classes in specs:
        m = types.ModuleType(module_name)
        m.__package__ = module_name.rsplit(".", 1)[0]
        for cls in classes:
            cls.__module__ = module_name
        for k, v in attrs.items():
            setattr(m, k, v)
        sys.modules[module_name] = m
        parent_name, child_name = module_name.rsplit(".", 1)
        parent = _ensure_parent_pkg(parent_name)
        setattr(parent, child_name, m)


def _ensure_repo_on_path() -> Path:
    func_root = Path(__file__).resolve().parent
    root_str = str(func_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return func_root


def _import_enhance_modules() -> None:
    _register_inline_enhance241_modules()


@dataclass
class Detection:
    class_id: int
    class_name: str
    confidence: float
    xyxy: List[float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": self.confidence,
            "xyxy": self.xyxy,
        }


class SDYOLO11Detector:
    """
    Import-and-run detector wrapper for SD-YOLO11 (a4+b7+d6).

    Initialize once, then call `predict(image_path)` repeatedly.
    """

    def __init__(
        self,
        weights: str | Path,
        *,
        device: str = "0",
        conf: float = 0.25,
        iou: float = 0.7,
        imgsz: int = 640,
        max_det: int = 100,
    ) -> None:
        _ensure_repo_on_path()
        _import_enhance_modules()

        self.weights = str(Path(weights).expanduser().resolve())
        self.device = str(device)
        self.conf = float(conf)
        self.iou = float(iou)
        self.imgsz = int(imgsz)
        self.max_det = int(max_det)

        self.model = YOLO(self.weights)

    def predict(
        self,
        image_path: str | Path,
        *,
        conf: Optional[float] = None,
        iou: Optional[float] = None,
        imgsz: Optional[int] = None,
        max_det: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        image_path = str(Path(image_path).expanduser().resolve())
        results = self.model.predict(
            source=image_path,
            conf=self.conf if conf is None else float(conf),
            iou=self.iou if iou is None else float(iou),
            imgsz=self.imgsz if imgsz is None else int(imgsz),
            max_det=self.max_det if max_det is None else int(max_det),
            device=self.device,
            verbose=False,
            save=False,
        )
        if not results:
            return []

        result = results[0]
        boxes = result.boxes
        names = result.names
        out: List[Dict[str, Any]] = []
        if boxes is None:
            return out

        xyxy = boxes.xyxy.detach().cpu().tolist()
        confs = boxes.conf.detach().cpu().tolist()
        clss = boxes.cls.detach().cpu().tolist()

        for box, score, cls_id in zip(xyxy, confs, clss):
            class_id = int(cls_id)
            class_name = str(names.get(class_id, class_id)) if isinstance(names, dict) else str(class_id)
            out.append(
                Detection(
                    class_id=class_id,
                    class_name=class_name,
                    confidence=float(score),
                    xyxy=[float(v) for v in box],
                ).to_dict()
            )
        return out


def get_detector(
    weights: str | Path = _default_weights_path(),
    *,
    device: str = "0",
    conf: float = 0.25,
    iou: float = 0.7,
    imgsz: int = 640,
    max_det: int = 100,
) -> SDYOLO11Detector:
    """
    Return cached detector instance to avoid repeated initialization.
    """
    key = (
        str(Path(weights).expanduser().resolve()),
        str(device),
        float(conf),
        float(iou),
        int(imgsz),
        int(max_det),
    )
    with _CACHE_LOCK:
        det = _DETECTOR_CACHE.get(key)
        if det is None:
            det = SDYOLO11Detector(
                weights=key[0],
                device=key[1],
                conf=key[2],
                iou=key[3],
                imgsz=key[4],
                max_det=key[5],
            )
            _DETECTOR_CACHE[key] = det
        return det


def detect(
    image_path: str | Path,
    *,
    weights: str | Path = _default_weights_path(),
    device: str = "0",
    conf: float = 0.25,
    iou: float = 0.7,
    imgsz: int = 640,
    max_det: int = 100,
) -> List[Dict[str, Any]]:
    """
    One-line helper: cached init + single-image predict.
    """
    detector = get_detector(
        weights=weights,
        device=device,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        max_det=max_det,
    )
    return detector.predict(image_path)


def clear_detector_cache() -> None:
    with _CACHE_LOCK:
        _DETECTOR_CACHE.clear()
