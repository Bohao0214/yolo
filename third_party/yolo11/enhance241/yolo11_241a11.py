from __future__ import annotations

# Path: third_party/yolo11/enhance241/yolo11_241a11.py
# Purpose: enhance241 a11 (GAM) patch on backbone P3 stage with safe residual injection.

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from .yolo11_241a3 import (
    _bind_module_debug,
    _get_module_recorder,
    _infer_device_dtype,
    _locate_detect,
    _locate_p4_to_p3_fuse,
    _should_capture_delta,
    _tensor_stats,
    _to_float,
    get_check_recorder,
)

ENHANCE241_AUDIT_KEYS = ["enhance241_a11"]  # enhance241-audit


def _deep_get(mapping: Any, *keys: str, default: Any = None) -> Any:
    cur = mapping
    for key in keys:
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(key)
            continue
        if hasattr(cur, key):
            cur = getattr(cur, key)
            continue
        return default
    return cur if cur is not None else default


def _safe_int(x: Any, default: int) -> int:
    try:
        return int(x)
    except Exception:
        return int(default)


def _safe_float(x: Any, default: float) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _is_rank0() -> bool:
    try:
        from ultralytics.utils import RANK  # type: ignore

        return int(RANK) in {-1, 0}
    except Exception:
        return True


def _append_enhance_check(module_key: str, lines: List[str]) -> None:
    if not lines:
        return
    exp_dir = str(os.environ.get("ENHANCE241_EXP_DIR", "")).strip()
    if not exp_dir:
        return
    path = Path(exp_dir) / "train" / "enhance_check.md"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_text("# enhance241 Gate Checks\n\n", encoding="utf-8")
        with path.open("a", encoding="utf-8") as f:
            f.write(f"## {module_key}\n")
            for line in lines:
                f.write(f"- {line}\n")
            f.write("\n")
    except Exception:
        return


def _infer_layer_out_channels(layer: Any) -> int:
    try:
        cv2 = getattr(layer, "cv2", None)
        if cv2 is not None and hasattr(cv2, "conv") and hasattr(cv2.conv, "out_channels"):
            return int(cv2.conv.out_channels)
    except Exception:
        pass
    try:
        conv = getattr(layer, "conv", None)
        if conv is not None and hasattr(conv, "out_channels"):
            return int(conv.out_channels)
    except Exception:
        pass
    if hasattr(layer, "c2"):
        try:
            return int(getattr(layer, "c2"))
        except Exception:
            pass
    if hasattr(layer, "out_channels"):
        try:
            return int(getattr(layer, "out_channels"))
        except Exception:
            pass
    out_c = None
    try:
        for m in layer.modules():
            if isinstance(m, torch.nn.Conv2d):
                out_c = int(m.out_channels)
    except Exception:
        out_c = None
    if out_c is None:
        raise RuntimeError(f"Unable to infer output channels from layer type={layer.__class__.__name__}")
    return int(out_c)


class _ChannelAttention(torch.nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        c = int(channels)
        hidden = max(4, c // max(1, int(reduction)))
        self.pool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc1 = torch.nn.Conv2d(c, hidden, kernel_size=1, stride=1, padding=0, bias=True)
        self.act = torch.nn.ReLU(inplace=True)
        self.fc2 = torch.nn.Conv2d(hidden, c, kernel_size=1, stride=1, padding=0, bias=True)
        torch.nn.init.zeros_(self.fc2.weight)
        if self.fc2.bias is not None:
            torch.nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.fc2(self.act(self.fc1(self.pool(x))))
        return torch.sigmoid(w)


class _SpatialAttention(torch.nn.Module):
    def __init__(self, channels: int, kernel_size: int = 7) -> None:
        super().__init__()
        c = int(channels)
        k = int(max(3, kernel_size))
        if k % 2 == 0:
            k += 1
        p = k // 2
        self.conv1 = torch.nn.Conv2d(c, c, kernel_size=k, stride=1, padding=p, bias=True)
        self.act = torch.nn.ReLU(inplace=True)
        self.conv2 = torch.nn.Conv2d(c, c, kernel_size=k, stride=1, padding=p, bias=True)
        torch.nn.init.zeros_(self.conv2.weight)
        if self.conv2.bias is not None:
            torch.nn.init.zeros_(self.conv2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.conv2(self.act(self.conv1(x)))
        return torch.sigmoid(w)


class GAMCore(torch.nn.Module):
    """GAM formula:
    F2 = Mc(F1) * F1
    F3 = Ms(F2) * F2
    """

    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7) -> None:
        super().__init__()
        self.channel = _ChannelAttention(channels=channels, reduction=reduction)
        self.spatial = _SpatialAttention(channels=channels, kernel_size=kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f2 = self.channel(x) * x
        f3 = self.spatial(f2) * f2
        return f3


class A11GAMResidual(torch.nn.Module):
    """Safe residual injection:
    y = y_base + alpha * GAM(y_base)
    """

    def __init__(
        self,
        base_module: torch.nn.Module,
        channels: int,
        mc_reduction: int = 16,
        ms_kernel: int = 7,
        alpha_init: float = 0.0,
        alpha_cap: float = 0.5,
    ) -> None:
        super().__init__()
        self.enhance241_a11_base = base_module
        self.enhance241_a11_gam = GAMCore(
            channels=int(channels),
            reduction=int(mc_reduction),
            kernel_size=int(ms_kernel),
        )
        self.alpha_cap = float(max(1e-6, abs(alpha_cap)))
        alpha_init = float(max(-self.alpha_cap * 0.95, min(self.alpha_cap * 0.95, alpha_init)))
        alpha_ratio = alpha_init / self.alpha_cap
        alpha_raw = torch.atanh(torch.tensor(alpha_ratio, dtype=torch.float32))
        self.enhance241_a11_alpha_raw = torch.nn.Parameter(alpha_raw)  # enhance241-audit
        self._check_written = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        recorder = _get_module_recorder(self)
        prefix = str(getattr(self, "_enhance241_prefix", "a11"))
        step = int(getattr(self, "_enhance241_fwd_step", 0))

        if recorder is not None and step == 0:
            recorder.capture_param_before(self, prefix)

        y_base = self.enhance241_a11_base(x)
        y_gam = self.enhance241_a11_gam(y_base)
        alpha_raw = self.enhance241_a11_alpha_raw.to(dtype=y_base.dtype, device=y_base.device)
        alpha = torch.tanh(alpha_raw) * self.alpha_cap
        out = y_base + alpha * y_gam

        if recorder is not None:
            if step == 0:
                recorder.record_module_compare(f"{prefix}.p3_stage", patched=out, baseline=y_base, input_tensor=x)
                v_base = _safe_float(y_base.detach().float().var(unbiased=False).item(), 0.0)
                v_gam = _safe_float(y_gam.detach().float().var(unbiased=False).item(), 0.0)
                v_out = _safe_float(out.detach().float().var(unbiased=False).item(), 0.0)
                cos = _safe_float(
                    F.cosine_similarity(
                        out.detach().float().reshape(out.shape[0], -1),
                        y_base.detach().float().reshape(y_base.shape[0], -1),
                        dim=1,
                    ).mean().item(),
                    0.0,
                )
                gate0 = bool(abs(_to_float(alpha.item())) <= self.alpha_cap + 1e-6)
                gate1 = bool(v_gam > 1e-12)
                gate2 = bool(any(bool(getattr(p, "requires_grad", False)) for p in self.parameters()))
                recorder.record_a1_payload(
                    f"{prefix}.gate0",
                    {
                        "pass": gate0,
                        "alpha": _to_float(alpha.item()),
                        "alpha_cap": _to_float(self.alpha_cap),
                        "cosine_out_vs_base": cos,
                        "var_ratio_out_over_base": v_out / (v_base + 1e-12),
                    },
                )
                recorder.record_a1_payload(
                    f"{prefix}.gate1",
                    {
                        "pass": gate1,
                        "var_gam": _to_float(v_gam),
                        "base_stats": _tensor_stats(y_base),
                        "gam_stats": _tensor_stats(y_gam),
                        "out_stats": _tensor_stats(out),
                    },
                )
                recorder.record_a1_payload(
                    f"{prefix}.gate2",
                    {
                        "pass": gate2,
                        "params_requires_grad": int(sum(1 for p in self.parameters() if p.requires_grad)),
                    },
                )
                recorder.record_scalar_curve(f"{prefix}.alpha", _to_float(alpha.item()), step, max_steps=100)
                if out.requires_grad:
                    try:
                        out.register_hook(lambda g, n=f"{prefix}.p3_stage": recorder.record_output_grad(n, g))
                    except Exception:
                        pass

                if not self._check_written and _is_rank0():
                    self._check_written = True
                    _append_enhance_check(
                        "a11",
                        [
                            f"Gate-0 pass={gate0} alpha={_to_float(alpha.item()):.6f} alpha_cap={_to_float(self.alpha_cap):.6f}",
                            f"Gate-1 pass={gate1} var_gam={_to_float(v_gam):.6e}",
                            f"Gate-2 pass={gate2} params_requires_grad={int(sum(1 for p in self.parameters() if p.requires_grad))}",
                        ],
                    )
            elif step <= 100:
                recorder.record_scalar_curve(f"{prefix}.alpha", _to_float(alpha.item()), step, max_steps=100)
            if _should_capture_delta(step):
                recorder.capture_param_delta(self, prefix)

        setattr(self, "_enhance241_fwd_step", step + 1)
        return out


def _extract_model_seq(model: Any) -> Tuple[Any, Optional[Any]]:
    yolo_obj = model
    if hasattr(model, "model") and hasattr(getattr(model, "model"), "model"):
        det_model = getattr(model, "model")
        seq = getattr(det_model, "model", None)
    else:
        det_model = model
        seq = getattr(det_model, "model", None)
    return yolo_obj, seq


def _normalize_where(raw: Any) -> List[str]:
    if isinstance(raw, str):
        items = [raw]
    elif isinstance(raw, (list, tuple)):
        items = list(raw)
    else:
        items = ["stage_P3_out"]
    out: List[str] = []
    for item in items:
        token = str(item).strip().lower()
        if token:
            out.append(token)
    return out or ["stage_p3_out"]


def apply(model: Any, cfg: Any) -> Any:
    enable_a11 = bool(_deep_get(cfg, "enhance241", "a11", default=False))
    if not enable_a11:
        return model

    yolo_obj, seq = _extract_model_seq(model)
    if seq is None:
        raise RuntimeError("enhance241.a11 requires YOLO/DetectionModel-like object with .model sequence.")

    where = _normalize_where(_deep_get(cfg, "enhance241", "a11_where", default=["stage_P3_out"]))
    if "stage_p3_out" not in where:
        raise RuntimeError(f"enhance241.a11 currently supports only stage_P3_out, got a11_where={where}")

    prepatched = [i for i, layer in enumerate(seq) if isinstance(layer, A11GAMResidual)]
    if prepatched:
        info = {
            "enabled": True,
            "existing_count": len(prepatched),
            "patched_count": 0,
            "patched_indices": [int(i) for i in prepatched],
            "where": where,
            "note": "already_patched",
        }
        setattr(yolo_obj, "_enhance241_a11_info", info)
        recorder = get_check_recorder("a11", cfg, patch_info=info)
        if recorder is not None:
            try:
                idx0 = int(prepatched[0])
                mod0 = seq[idx0]
                _bind_module_debug(mod0, recorder, prefix=f"a11.idx{idx0}.existing")
                recorder.register_module_params(mod0, f"a11.idx{idx0}.existing")
                detect_idx, detect = _locate_detect(seq)
                recorder.attach_detect_hooks(detect, detect_idx)
                recorder.maybe_run_val_separability(yolo_obj, cfg)
            except Exception as exc:
                recorder.add_note(f"a11_prepatched_hook_failed:{exc}")
        return yolo_obj

    _, p3_stage_idx = _locate_p4_to_p3_fuse(seq)
    old = seq[p3_stage_idx]
    channels = _infer_layer_out_channels(old)
    mc_reduction = _safe_int(_deep_get(cfg, "enhance241", "a11_mc_reduction", default=16), 16)
    ms_kernel = _safe_int(_deep_get(cfg, "enhance241", "a11_ms_kernel", default=7), 7)
    alpha_init = _safe_float(_deep_get(cfg, "enhance241", "a11_alpha_init", default=0.0), 0.0)
    alpha_cap = _safe_float(_deep_get(cfg, "enhance241", "a11_alpha_cap", default=0.5), 0.5)

    wrapped = A11GAMResidual(
        base_module=old,
        channels=channels,
        mc_reduction=mc_reduction,
        ms_kernel=ms_kernel,
        alpha_init=alpha_init,
        alpha_cap=alpha_cap,
    )
    for attr in ("i", "f", "type"):
        if hasattr(old, attr):
            setattr(wrapped, attr, getattr(old, attr))
    device, dtype = _infer_device_dtype(seq, p3_stage_idx)
    if device is not None:
        if dtype is not None:
            wrapped = wrapped.to(device=device, dtype=dtype)
        else:
            wrapped = wrapped.to(device=device)
    seq[p3_stage_idx] = wrapped

    info = {
        "enabled": True,
        "existing_count": 0,
        "patched_count": 1,
        "patched_idx": int(p3_stage_idx),
        "where": where,
        "base_type": old.__class__.__name__,
        "new_type": "A11GAMResidual",
        "channels": int(channels),
        "mc_reduction": int(mc_reduction),
        "ms_kernel": int(ms_kernel),
        "mode": "safe_residual",
        "alpha_init": _to_float(alpha_init),
        "alpha_cap": _to_float(alpha_cap),
    }
    setattr(yolo_obj, "_enhance241_a11_info", info)
    print(
        f"[enhance241] a11 enabled: patched backbone stage at model.model[{p3_stage_idx}] "
        f"(orig={old.__class__.__name__}) -> A11GAMResidual(C={channels}, alpha_init={alpha_init:.3f})"
    )
    if _is_rank0():
        _append_enhance_check(
            "a11",
            [
                f"Gate-0 patched_idx={int(p3_stage_idx)} base_type={old.__class__.__name__} channels={int(channels)}",
                f"Gate-1 mode=safe_residual mc_reduction={int(mc_reduction)} ms_kernel={int(ms_kernel)}",
                "Gate-2 waiting first forward/backward stats",
            ],
        )

    recorder = get_check_recorder("a11", cfg, patch_info=info)
    if recorder is not None:
        _bind_module_debug(wrapped, recorder, prefix=f"a11.idx{p3_stage_idx}")
        recorder.register_module_params(wrapped, f"a11.idx{p3_stage_idx}")
        try:
            detect_idx, detect = _locate_detect(seq)
            recorder.attach_detect_hooks(detect, detect_idx)
        except Exception as exc:
            recorder.add_note(f"a11_detect_hook_failed:{exc}")
        try:
            recorder.maybe_run_val_separability(yolo_obj, cfg)
        except Exception as exc:
            recorder.add_note(f"a11_val_check_failed:{exc}")

    return yolo_obj
