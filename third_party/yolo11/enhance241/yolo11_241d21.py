from __future__ import annotations

# Path: third_party/yolo11/enhance241/yolo11_241d21.py
# Purpose: enhance241 d21 patch (cls-only residual-safe score calibration).

from typing import Any, Dict, List, Optional, Tuple

import torch

from .yolo11_241a3 import _bind_module_debug, _get_module_recorder, _locate_detect, _should_capture_delta, _to_float, get_check_recorder
from .yolo11_241d11 import D11ClsScoreCalib

ENHANCE241_AUDIT_KEYS = ["enhance241_d21"]  # enhance241-audit


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


class D21ClsScoreCalib(D11ClsScoreCalib):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        recorder = _get_module_recorder(self)
        prefix = str(getattr(self, "_enhance241_prefix", "d21"))
        step = int(getattr(self, "_enhance241_fwd_step", 0))

        if recorder is not None and step == 0:
            recorder.capture_param_before(self, prefix)

        logits = self.enhance241_d11_base_cls(x)
        t = torch.clamp(self.enhance241_d11_temp, min=self.t_min, max=self.t_max).to(dtype=logits.dtype, device=logits.device)
        b = self.enhance241_d11_bias.to(dtype=logits.dtype, device=logits.device)
        alpha_raw = self.enhance241_d11_alpha_raw.to(dtype=logits.dtype, device=logits.device)
        alpha = torch.tanh(alpha_raw) * self.alpha_cap
        logits_calib = logits / t + b
        out = logits + alpha * (logits_calib - logits)

        if recorder is not None:
            recorder.record_scalar_curve(f"{prefix}.temperature", float(t.item()), step, max_steps=100)
            recorder.record_scalar_curve(f"{prefix}.bias", float(b.item()), step, max_steps=100)
            recorder.record_scalar_curve(f"{prefix}.alpha", float(alpha.item()), step, max_steps=100)
            if step == 0:
                recorder.record_a1_payload(
                    f"{prefix}.cfg",
                    {
                        "temp_init": _to_float(self.enhance241_d11_temp.detach().item()),
                        "temp_min": _to_float(self.t_min),
                        "temp_max": _to_float(self.t_max),
                        "bias_init": _to_float(self.enhance241_d11_bias.detach().item()),
                        "alpha": _to_float(alpha.item()),
                        "alpha_cap": _to_float(self.alpha_cap),
                    },
                )
                if out.requires_grad:
                    try:
                        out.register_hook(lambda g, n=f"{prefix}.logits": recorder.record_output_grad(n, g))
                    except Exception:
                        pass
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


def _cast_like_module(module: torch.nn.Module, wrapped: torch.nn.Module) -> torch.nn.Module:
    try:
        p = next(module.parameters())
        return wrapped.to(device=p.device, dtype=p.dtype)
    except Exception:
        return wrapped


def apply(model: Any, cfg: Any) -> Any:
    enable_d21 = bool(_deep_get(cfg, "enhance241", "d21", default=False))
    if not enable_d21:
        return model

    if any(bool(_deep_get(cfg, "enhance241", k, default=False)) for k in ("d1", "d3", "d5", "d7", "d9", "d11")):
        raise RuntimeError("enhance241.d21 conflicts with d1/d3/d5/d7/d9/d11; enable only one D-class module.")

    yolo_obj, seq = _extract_model_seq(model)
    if seq is None:
        raise RuntimeError("enhance241.d21 requires YOLO/DetectionModel-like object with .model sequence.")

    detect_idx, detect = _locate_detect(seq)
    cls_heads = getattr(detect, "cv3", None)
    if cls_heads is None or not isinstance(cls_heads, torch.nn.ModuleList) or len(cls_heads) < 1:
        raise RuntimeError("enhance241.d21 expects Detect.cv3 ModuleList with at least one head.")

    apply_to = _parse_apply_to(_deep_get(cfg, "enhance241", "d21_apply_to", default=["P3", "P2_new"]))
    target_heads = _target_head_indices(detect, apply_to)
    if not target_heads:
        raise RuntimeError(f"enhance241.d21 apply_to={apply_to} matched no valid detect heads.")

    temp_init = _safe_float(_deep_get(cfg, "enhance241", "d21_temp_init", default=1.0), 1.0)
    t_min = _safe_float(_deep_get(cfg, "enhance241", "d21_temp_min", default=0.7), 0.7)
    t_max = _safe_float(_deep_get(cfg, "enhance241", "d21_temp_max", default=2.0), 2.0)
    bias_init = _safe_float(_deep_get(cfg, "enhance241", "d21_bias_shift_init", default=0.0), 0.0)
    alpha_init = _safe_float(_deep_get(cfg, "enhance241", "d21_alpha_init", default=0.0), 0.0)
    alpha_cap = _safe_float(_deep_get(cfg, "enhance241", "d21_alpha_cap", default=0.5), 0.5)

    patched_heads: List[int] = []
    existing = 0
    recorder = get_check_recorder("d21", cfg)
    for head_idx in target_heads:
        old_cls = cls_heads[head_idx]
        if isinstance(old_cls, D21ClsScoreCalib):
            existing += 1
            continue

        wrapped = D21ClsScoreCalib(
            old_cls,
            temp_init=temp_init,
            t_min=t_min,
            t_max=t_max,
            bias_init=bias_init,
            alpha_init=alpha_init,
            alpha_cap=alpha_cap,
        )
        wrapped = _cast_like_module(old_cls, wrapped)
        cls_heads[head_idx] = wrapped
        patched_heads.append(int(head_idx))

        if recorder is not None:
            try:
                _bind_module_debug(wrapped, recorder, prefix=f"d21.detect{detect_idx}.head{head_idx}")
                recorder.register_module_params(wrapped, f"d21.detect{detect_idx}.head{head_idx}")
            except Exception as exc:
                recorder.add_note(f"d21_attach_debug_failed:{head_idx}:{exc}")

        one2one_cv3 = getattr(detect, "one2one_cv3", None)
        if isinstance(one2one_cv3, torch.nn.ModuleList) and head_idx < len(one2one_cv3):
            old_o2o = one2one_cv3[head_idx]
            if not isinstance(old_o2o, D21ClsScoreCalib):
                wrapped_o2o = D21ClsScoreCalib(
                    old_o2o,
                    temp_init=temp_init,
                    t_min=t_min,
                    t_max=t_max,
                    bias_init=bias_init,
                    alpha_init=alpha_init,
                    alpha_cap=alpha_cap,
                )
                one2one_cv3[head_idx] = _cast_like_module(old_o2o, wrapped_o2o)

    info: Dict[str, Any] = {
        "enabled": True,
        "existing_count": int(existing),
        "patched_count": int(len(patched_heads)),
        "detect_idx": int(detect_idx),
        "patched_heads": [int(x) for x in patched_heads],
        "target_heads": [int(x) for x in target_heads],
        "temp_init": _to_float(temp_init),
        "temp_range": [_to_float(t_min), _to_float(t_max)],
        "bias_init": _to_float(bias_init),
        "alpha_init": _to_float(alpha_init),
        "alpha_cap": _to_float(alpha_cap),
    }
    setattr(yolo_obj, "_enhance241_d21_info", info)
    print(
        f"[enhance241] d21 enabled: patched cls heads={patched_heads or 'none(new)'} "
        f"target_heads={target_heads} temp=[{t_min:.3f},{t_max:.3f}] bias_init={bias_init:.3f}"
    )

    if recorder is not None:
        recorder.set_patch_info(info)
        try:
            recorder.attach_detect_hooks(detect, detect_idx)
        except Exception as exc:
            recorder.add_note(f"d21_detect_hook_failed:{exc}")
        try:
            recorder.maybe_run_val_separability(yolo_obj, cfg)
        except Exception as exc:
            recorder.add_note(f"d21_val_check_failed:{exc}")

    return yolo_obj
