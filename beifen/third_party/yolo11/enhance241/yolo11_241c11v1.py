from __future__ import annotations

# Path: third_party/yolo11/enhance241/yolo11_241c11.py
# Purpose: enhance241 c11 (light head-input gate guardrail for P3/P2_new).

from typing import Any, Dict, List, Optional, Tuple

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

ENHANCE241_AUDIT_KEYS = ["enhance241_c11"]  # enhance241-audit


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


def _safe_float(x: Any, default: float) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


class _SEGate(torch.nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        c = int(channels)
        hidden = max(4, c // max(1, int(reduction)))
        self.pool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc1 = torch.nn.Conv2d(c, hidden, kernel_size=1, stride=1, padding=0, bias=True)
        self.act = torch.nn.SiLU(inplace=True)
        self.fc2 = torch.nn.Conv2d(hidden, c, kernel_size=1, stride=1, padding=0, bias=True)
        torch.nn.init.zeros_(self.fc2.weight)
        if self.fc2.bias is not None:
            torch.nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.fc2(self.act(self.fc1(self.pool(x)))))


class _Conv1x1Gate(torch.nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        c = int(channels)
        self.conv = torch.nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0, bias=True)
        torch.nn.init.zeros_(self.conv.weight)
        if self.conv.bias is not None:
            torch.nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.conv(x))


class C11HeadGateInject(torch.nn.Module):
    """Residual gate:
    out = y_base + alpha * (gate(y_base) * y_base)
    """

    def __init__(
        self,
        base_module: torch.nn.Module,
        channels: int,
        mode: str = "se",
        alpha_init: float = 0.0,
        alpha_cap: float = 0.5,
    ) -> None:
        super().__init__()
        self.enhance241_c11_base = base_module
        mode = str(mode).lower()
        if mode in {"reuse_c5", "1x1_gate", "conv1x1"}:
            self.enhance241_c11_gate = _Conv1x1Gate(channels)
            self.mode = "1x1_gate"
        else:
            self.enhance241_c11_gate = _SEGate(channels, reduction=16)
            self.mode = "se"

        self.alpha_cap = float(max(1e-6, abs(alpha_cap)))
        alpha_init = float(max(-self.alpha_cap * 0.95, min(self.alpha_cap * 0.95, alpha_init)))
        alpha_ratio = alpha_init / self.alpha_cap
        alpha_raw = torch.atanh(torch.tensor(alpha_ratio, dtype=torch.float32))
        self.enhance241_c11_alpha_raw = torch.nn.Parameter(alpha_raw)  # enhance241-audit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        recorder = _get_module_recorder(self)
        prefix = str(getattr(self, "_enhance241_prefix", "c11"))
        step = int(getattr(self, "_enhance241_fwd_step", 0))

        if recorder is not None and step == 0:
            recorder.capture_param_before(self, prefix)

        y_base = self.enhance241_c11_base(x)
        gate = self.enhance241_c11_gate(y_base)
        delta = gate * y_base
        alpha_raw = self.enhance241_c11_alpha_raw.to(dtype=y_base.dtype, device=y_base.device)
        alpha = torch.tanh(alpha_raw) * self.alpha_cap
        out = y_base + alpha * delta

        if recorder is not None:
            if step == 0:
                recorder.record_module_compare(f"{prefix}.head_gate", patched=out, baseline=y_base, input_tensor=x)
                recorder.record_a1_payload(
                    f"{prefix}.gate0",
                    {
                        "mode": self.mode,
                        "alpha": _to_float(alpha.item()),
                        "alpha_cap": _to_float(self.alpha_cap),
                        "gate_stats": _tensor_stats(gate),
                        "delta_stats": _tensor_stats(delta),
                        "out_stats": _tensor_stats(out),
                    },
                )
                recorder.record_scalar_curve(f"{prefix}.alpha", _to_float(alpha.item()), step, max_steps=100)
                if out.requires_grad:
                    try:
                        out.register_hook(lambda g, n=f"{prefix}.head_gate": recorder.record_output_grad(n, g))
                    except Exception:
                        pass
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


def _infer_head_channels(detect: Any, head_idx: int) -> int:
    try:
        return int(detect.cv2[head_idx][0].conv.in_channels)  # type: ignore[index]
    except Exception:
        pass
    try:
        branch = detect.cv2[head_idx]  # type: ignore[index]
        for m in branch.modules():
            if isinstance(m, torch.nn.Conv2d):
                return int(m.in_channels)
    except Exception:
        pass
    raise RuntimeError(f"Unable to infer channels for detect head index={head_idx}")


def _parse_apply_to(raw: Any) -> List[str]:
    if isinstance(raw, str):
        items = [raw]
    elif isinstance(raw, (list, tuple)):
        items = list(raw)
    else:
        items = ["P3", "P2_new"]
    out: List[str] = []
    for item in items:
        token = str(item).strip().lower()
        if token:
            out.append(token)
    return out or ["p3", "p2_new"]


def _target_head_indices(detect: Any, apply_to: List[str]) -> List[int]:
    f = getattr(detect, "f", [])
    if isinstance(f, int):
        f_list = [int(f)]
    elif isinstance(f, (list, tuple)):
        f_list = [int(x) for x in f]
    else:
        f_list = []
    nl = int(getattr(detect, "nl", len(f_list)))
    idxs: List[int] = []
    if "p2_new" in apply_to and nl >= 4:
        idxs.append(0)
    if "p3" in apply_to:
        idxs.append(1 if nl >= 4 else 0)
    if "all" in apply_to:
        idxs.extend(list(range(nl)))
    uniq = sorted(set(i for i in idxs if 0 <= i < nl))
    return uniq


def apply(model: Any, cfg: Any) -> Any:
    enable_c11 = bool(_deep_get(cfg, "enhance241", "c11", default=False))
    if not enable_c11:
        return model

    yolo_obj, seq = _extract_model_seq(model)
    if seq is None:
        raise RuntimeError("enhance241.c11 requires YOLO/DetectionModel-like object with .model sequence.")

    detect_idx, detect = _locate_detect(seq)
    f = getattr(detect, "f", [])
    if isinstance(f, int):
        f_list = [int(f)]
    elif isinstance(f, (list, tuple)):
        f_list = [int(x) for x in f]
    else:
        raise RuntimeError(f"enhance241.c11 invalid Detect.f={f}")

    apply_to = _parse_apply_to(_deep_get(cfg, "enhance241", "c11_apply_to", default=["P3", "P2_new"]))
    target_heads = _target_head_indices(detect, apply_to)
    if not target_heads:
        raise RuntimeError(f"enhance241.c11 apply_to={apply_to} matched no valid detect heads.")

    mode = str(_deep_get(cfg, "enhance241", "c11_mode", default="se"))
    alpha_init = _safe_float(_deep_get(cfg, "enhance241", "c11_alpha_init", default=0.0), 0.0)
    alpha_cap = _safe_float(_deep_get(cfg, "enhance241", "c11_alpha_cap", default=0.5), 0.5)

    patched_idxs: List[int] = []
    existing = 0
    recorder = get_check_recorder("c11", cfg)
    for head_idx in target_heads:
        src_idx = int(f_list[head_idx])
        if not (0 <= src_idx < detect_idx):
            raise RuntimeError(f"enhance241.c11 invalid source idx={src_idx} for detect head={head_idx}")
        old = seq[src_idx]
        if isinstance(old, C11HeadGateInject):
            existing += 1
            continue

        channels = _infer_head_channels(detect, head_idx=head_idx)
        wrapped = C11HeadGateInject(
            base_module=old,
            channels=channels,
            mode=mode,
            alpha_init=alpha_init,
            alpha_cap=alpha_cap,
        )
        for attr in ("i", "f", "type"):
            if hasattr(old, attr):
                setattr(wrapped, attr, getattr(old, attr))
        device, dtype = _infer_device_dtype(seq, src_idx)
        if device is not None:
            if dtype is not None:
                wrapped = wrapped.to(device=device, dtype=dtype)
            else:
                wrapped = wrapped.to(device=device)
        seq[src_idx] = wrapped
        patched_idxs.append(int(src_idx))
        if recorder is not None:
            try:
                _bind_module_debug(wrapped, recorder, prefix=f"c11.idx{src_idx}.head{head_idx}")
                recorder.register_module_params(wrapped, f"c11.idx{src_idx}.head{head_idx}")
            except Exception as exc:
                recorder.add_note(f"c11_attach_debug_failed:{src_idx}:{exc}")

    info = {
        "enabled": True,
        "existing_count": int(existing),
        "patched_count": int(len(patched_idxs)),
        "detect_idx": int(detect_idx),
        "target_heads": [int(x) for x in target_heads],
        "patched_indices": [int(x) for x in patched_idxs],
        "mode": str(mode),
        "alpha_init": _to_float(alpha_init),
        "alpha_cap": _to_float(alpha_cap),
    }
    setattr(yolo_obj, "_enhance241_c11_info", info)
    print(
        f"[enhance241] c11 enabled: patched head inputs at indices={patched_idxs or 'none(new)'} "
        f"target_heads={target_heads} mode={mode}"
    )

    if recorder is not None:
        recorder.set_patch_info(info)
        try:
            recorder.attach_detect_hooks(detect, detect_idx)
        except Exception as exc:
            recorder.add_note(f"c11_detect_hook_failed:{exc}")
        try:
            recorder.maybe_run_val_separability(yolo_obj, cfg)
        except Exception as exc:
            recorder.add_note(f"c11_val_check_failed:{exc}")

    return yolo_obj
