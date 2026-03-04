from __future__ import annotations

# Path: third_party/yolo11/enhance241/yolo11_241b6.py
# Purpose: enhance241 b6 (DySample-style residual-safe semantic alignment on neck top-down links).

from typing import Any, List

import torch
import torch.nn.functional as F

from .yolo11_241a3 import (
    _bind_module_debug,
    _get_module_recorder,
    _infer_device_dtype,
    _locate_detect,
    _should_capture_delta,
    _tensor_stats,
    _to_float,
    get_check_recorder,
)
from .yolo11_241b7 import (
    CARAFECore,
    _concat_candidates,
    _deep_get,
    _infer_concat_channels,
    _infer_scale_factor,
    _locate_p4_to_p3_fuse,
    _locate_p5_to_p4_fuse,
    _safe_float,
    _safe_int,
)

ENHANCE241_AUDIT_KEYS = ["enhance241_b6"]  # enhance241-audit


class DySampleCore(torch.nn.Module):
    """B7-style CARAFE + high-frequency gain.

    Keeps the historical class name for compatibility; old objects without the new
    CARAFE attrs fall back to the legacy DySample path.
    """

    def __init__(
        self,
        channels: int,
        scale: int = 2,
        offset_max: float = 1.5,
        canvas_mode: str = "bilinear",
        kernel_size: int = 5,
        compress: int = 64,
        chunk_channels: int = 64,
        hf_kernel: int = 3,
        hf_gain_init: float = 0.1,
    ) -> None:
        super().__init__()
        c = int(channels)
        s = max(2, int(scale))
        self.scale = s
        self.offset_max = float(max(0.1, offset_max))
        self.sample_mode = str(canvas_mode).lower()
        self.hf_kernel = max(3, int(hf_kernel))
        if self.hf_kernel % 2 == 0:
            self.hf_kernel += 1
        self.enhance241_b6_offset = torch.nn.Conv2d(c, 2 * s * s, kernel_size=1, stride=1, padding=0, bias=True)
        torch.nn.init.zeros_(self.enhance241_b6_offset.weight)
        if self.enhance241_b6_offset.bias is not None:
            torch.nn.init.zeros_(self.enhance241_b6_offset.bias)
        self.enhance241_b6_carafe = CARAFECore(
            channels=c,
            scale=s,
            kernel_size=int(kernel_size),
            compress=int(compress),
            chunk_channels=int(chunk_channels),
            use_hf=False,
        )
        self.enhance241_b6_hf_gain = torch.nn.Parameter(torch.tensor(float(hf_gain_init), dtype=torch.float32))
        self._enhance241_last_offset_abs_mean = 0.0
        self._enhance241_last_offset_abs_max = 0.0
        self._enhance241_last_edge_abs_mean = 0.0
        self._enhance241_last_hf_gain = 0.0

    def _legacy_forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, h, w = x.shape
        hs, ws = h * self.scale, w * self.scale
        offset = F.pixel_shuffle(self.enhance241_b6_offset(x), upscale_factor=self.scale)
        offset = torch.tanh(offset) * self.offset_max
        self._enhance241_last_offset_abs_mean = _to_float(offset.detach().float().abs().mean().item())
        self._enhance241_last_offset_abs_max = _to_float(offset.detach().float().abs().amax().item())

        theta = torch.tensor(
            [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]],
            device=x.device,
            dtype=torch.float32,
        ).expand(batch, -1, -1)
        base_grid = F.affine_grid(theta, size=(batch, channels, hs, ws), align_corners=False)

        norm_x = offset[:, 0].float() * (2.0 / max(1, w))
        norm_y = offset[:, 1].float() * (2.0 / max(1, h))
        grid = base_grid.clone()
        grid[..., 0] = grid[..., 0] + norm_x
        grid[..., 1] = grid[..., 1] + norm_y
        sample_mode = "nearest" if self.sample_mode == "nearest" else "bilinear"
        sampled = F.grid_sample(
            x.float(),
            grid,
            mode=sample_mode,
            padding_mode="border",
            align_corners=False,
        )
        return sampled.to(dtype=x.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not hasattr(self, "enhance241_b6_carafe"):
            self._enhance241_last_edge_abs_mean = 0.0
            self._enhance241_last_hf_gain = 0.0
            return self._legacy_forward(x)

        carafe = self.enhance241_b6_carafe(x)
        blur = F.avg_pool2d(
            x,
            kernel_size=self.hf_kernel,
            stride=1,
            padding=self.hf_kernel // 2,
            count_include_pad=False,
        )
        edge = x - blur
        self._enhance241_last_edge_abs_mean = _to_float(edge.detach().float().abs().mean().item())
        hf_gain = torch.tanh(self.enhance241_b6_hf_gain.to(dtype=x.dtype, device=x.device))
        self._enhance241_last_hf_gain = _to_float(hf_gain.detach().float().item())
        if self.sample_mode == "nearest":
            edge_up = F.interpolate(edge, size=carafe.shape[-2:], mode="nearest")
        else:
            edge_up = F.interpolate(edge, size=carafe.shape[-2:], mode="bilinear", align_corners=False)
        self._enhance241_last_offset_abs_mean = 0.0
        self._enhance241_last_offset_abs_max = 0.0
        return carafe + hf_gain * edge_up


class B6DySampleSafe(torch.nn.Module):
    """Residual-safe wrapper: out = base_up(x) + alpha * (carafe_hf(x) - base_up(x))."""

    def __init__(
        self,
        base_module: torch.nn.Module,
        channels: int,
        scale: int = 2,
        offset_max: float = 1.5,
        canvas_mode: str = "bilinear",
        kernel_size: int = 5,
        compress: int = 64,
        chunk_channels: int = 64,
        hf_kernel: int = 3,
        hf_gain_init: float = 0.1,
        alpha_init: float = 0.02,
        alpha_cap: float = 0.3,
        tag: str = "",
    ) -> None:
        super().__init__()
        self.enhance241_b6_base = base_module
        self.enhance241_b6_dysample = DySampleCore(
            channels=int(channels),
            scale=int(scale),
            offset_max=float(offset_max),
            canvas_mode=str(canvas_mode),
            kernel_size=int(kernel_size),
            compress=int(compress),
            chunk_channels=int(chunk_channels),
            hf_kernel=int(hf_kernel),
            hf_gain_init=float(hf_gain_init),
        )
        self.tag = str(tag)
        self.alpha_cap = float(max(1e-6, abs(alpha_cap)))
        alpha_init = float(max(-self.alpha_cap * 0.95, min(self.alpha_cap * 0.95, alpha_init)))
        alpha_ratio = alpha_init / self.alpha_cap
        alpha_raw = torch.atanh(torch.tensor(alpha_ratio, dtype=torch.float32))
        self.enhance241_b6_alpha = torch.nn.Parameter(alpha_raw)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        recorder = _get_module_recorder(self)
        prefix = str(getattr(self, "_enhance241_prefix", "b6"))
        step = int(getattr(self, "_enhance241_fwd_step", 0))

        if recorder is not None and step == 0:
            recorder.capture_param_before(self, prefix)

        y_base = self.enhance241_b6_base(x)
        y_dy = self.enhance241_b6_dysample(x)
        if y_dy.shape[-2:] != y_base.shape[-2:]:
            y_dy = F.interpolate(y_dy, size=y_base.shape[-2:], mode="nearest")

        alpha_raw = self.enhance241_b6_alpha.to(dtype=y_base.dtype, device=y_base.device)
        alpha = torch.tanh(alpha_raw) * self.alpha_cap
        delta = y_dy - y_base
        out = y_base + alpha * delta

        if recorder is not None:
            if step == 0:
                recorder.record_module_compare(f"{prefix}.upsample", patched=out, baseline=y_base, input_tensor=x)
                recorder.record_a1_payload(
                    f"{prefix}.gate1",
                    {
                        "tag": self.tag,
                        "alpha": _to_float(alpha.item()),
                        "alpha_cap": _to_float(self.alpha_cap),
                        "offset_abs_mean": _to_float(self.enhance241_b6_dysample._enhance241_last_offset_abs_mean),
                        "offset_abs_max": _to_float(self.enhance241_b6_dysample._enhance241_last_offset_abs_max),
                        "hf_gain": _to_float(self.enhance241_b6_dysample._enhance241_last_hf_gain),
                        "edge_abs_mean": _to_float(self.enhance241_b6_dysample._enhance241_last_edge_abs_mean),
                        "base_stats": _tensor_stats(y_base),
                        "dy_stats": _tensor_stats(y_dy),
                        "delta_stats": _tensor_stats(delta),
                    },
                )
                if out.requires_grad:
                    try:
                        out.register_hook(lambda g, n=f"{prefix}.upsample": recorder.record_output_grad(n, g))
                    except Exception:
                        pass
            if _should_capture_delta(step):
                recorder.capture_param_delta(self, prefix)

        setattr(self, "_enhance241_fwd_step", step + 1)
        return out


def apply(model: Any, cfg: Any) -> Any:
    enable_b6 = bool(_deep_get(cfg, "enhance241", "b6", default=False))
    if not enable_b6:
        return model

    yolo_obj = model
    if hasattr(model, "model") and hasattr(getattr(model, "model"), "model"):
        det_model = getattr(model, "model")
        seq = getattr(det_model, "model", None)
    else:
        det_model = model
        seq = getattr(det_model, "model", None)
    if seq is None:
        raise RuntimeError("enhance241.b6 requires YOLO/DetectionModel-like object with .model sequence.")

    prepatched = [i for i, layer in enumerate(seq) if isinstance(layer, B6DySampleSafe)]
    if prepatched:
        info = {
            "enabled": True,
            "patched_indices": prepatched,
            "patched_count": 0,
            "existing_count": len(prepatched),
            "note": "already_patched",
        }
        setattr(yolo_obj, "_enhance241_b6_info", info)
        return yolo_obj

    p3_concat_idx = _locate_p4_to_p3_fuse(seq)
    p5_concat_idx = _locate_p5_to_p4_fuse(seq, p3_concat_idx)

    offset_max = _safe_float(_deep_get(cfg, "enhance241", "b6_offset_max", default=1.5), 1.5)
    canvas_mode = str(_deep_get(cfg, "enhance241", "b6_canvas_mode", default="bilinear"))
    kernel_size = _safe_int(_deep_get(cfg, "enhance241", "b6_kernel_size", default=5), 5)
    compress = _safe_int(_deep_get(cfg, "enhance241", "b6_compress", default=64), 64)
    chunk_channels = _safe_int(_deep_get(cfg, "enhance241", "b6_chunk_channels", default=64), 64)
    hf_kernel = _safe_int(_deep_get(cfg, "enhance241", "b6_hf_kernel", default=3), 3)
    hf_gain_init = _safe_float(_deep_get(cfg, "enhance241", "b6_hf_gain_init", default=0.1), 0.1)
    alpha_init = _safe_float(_deep_get(cfg, "enhance241", "b6_alpha_init", default=0.02), 0.02)
    alpha_cap = _safe_float(_deep_get(cfg, "enhance241", "b6_alpha_cap", default=0.3), 0.3)

    patched_indices: List[int] = []
    recorder = get_check_recorder("b6", cfg)

    def _patch_one(concat_idx: int, tag: str) -> None:
        up_idx = int(concat_idx) - 1
        if up_idx < 0:
            raise RuntimeError(f"enhance241.b6 invalid upsample index for concat idx={concat_idx}")
        old = seq[up_idx]
        old_name = old.__class__.__name__.lower()
        if "upsample" not in old_name:
            raise RuntimeError(
                f"enhance241.b6 expected upsample module at idx={up_idx} before concat idx={concat_idx}, "
                f"got {old.__class__.__name__}. Concat candidates: {_concat_candidates(seq)}"
            )
        channels = _infer_concat_channels(seq, concat_idx)
        scale = _infer_scale_factor(old)
        mod = B6DySampleSafe(
            base_module=old,
            channels=channels,
            scale=scale,
            offset_max=offset_max,
            canvas_mode=canvas_mode,
            kernel_size=kernel_size,
            compress=compress,
            chunk_channels=chunk_channels,
            hf_kernel=hf_kernel,
            hf_gain_init=hf_gain_init,
            alpha_init=alpha_init,
            alpha_cap=alpha_cap,
            tag=tag,
        )
        for attr in ("i", "f", "type"):
            if hasattr(old, attr):
                setattr(mod, attr, getattr(old, attr))
        device, dtype = _infer_device_dtype(seq, up_idx)
        if device is not None:
            if dtype is not None:
                mod = mod.to(device=device, dtype=dtype)
            else:
                mod = mod.to(device=device)
        seq[up_idx] = mod
        patched_indices.append(up_idx)
        if recorder is not None:
            try:
                _bind_module_debug(mod, recorder, prefix=f"b6.idx{up_idx}.{tag}")
                recorder.register_module_params(mod, f"b6.idx{up_idx}.{tag}")
            except Exception as exc:
                recorder.add_note(f"b6_attach_debug_failed:{up_idx}:{exc}")

    _patch_one(p5_concat_idx, "p5p4")
    _patch_one(p3_concat_idx, "p4p3")

    info = {
        "enabled": True,
        "patched_indices": [int(x) for x in patched_indices],
        "patched_count": len(patched_indices),
        "existing_count": 0,
        "concat_indices": [int(p5_concat_idx), int(p3_concat_idx)],
        "offset_max": _to_float(offset_max),
        "canvas_mode": str(canvas_mode),
        "kernel_size": int(kernel_size),
        "compress": int(compress),
        "chunk_channels": int(chunk_channels),
        "hf_kernel": int(hf_kernel),
        "hf_gain_init": _to_float(hf_gain_init),
        "alpha_init": _to_float(alpha_init),
        "alpha_cap": _to_float(alpha_cap),
        "mode": "carafe_hf_residual_safe",
    }
    setattr(yolo_obj, "_enhance241_b6_info", info)
    print(
        f"[enhance241] b6 enabled: patched upsample indices={patched_indices} "
        f"-> B6DySampleSafe(CARAFE+HF, kernel={kernel_size}, hf_kernel={hf_kernel})"
    )

    if recorder is not None:
        recorder.set_patch_info(info)
        try:
            detect_idx, detect_layer = _locate_detect(seq)
            recorder.attach_detect_hooks(detect_layer, detect_idx)
        except Exception as exc:
            recorder.add_note(f"b6_detect_hook_failed:{exc}")

    return yolo_obj
