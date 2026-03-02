from __future__ import annotations

# Path: third_party/yolo11/enhance241/yolo11_241a4.py
# Purpose: enhance241 a4 (SPDConvDownsample) module + apply hook + debug checks.

import atexit
import csv
import datetime as dt
import json
import os
import subprocess
import threading
from itertools import count
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

ENHANCE241_AUDIT_KEYS = ["enhance241_a4"]  # enhance241-audit

_IMAGE_SUFFIX = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


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


def _stride_to_int(s: Any) -> Optional[int]:
    try:
        if isinstance(s, (list, tuple)):
            return int(s[0]) if s else None
        if isinstance(s, torch.Tensor):
            if s.numel() == 1:
                return int(s.item())
            return int(s.flatten()[0].item())
        return int(s)
    except Exception:
        return None


def _module_stride(mod: Any) -> Optional[int]:
    if hasattr(mod, "stride"):
        s = _stride_to_int(getattr(mod, "stride"))
        if s is not None:
            return s
    for attr in ("conv", "cv1", "dw"):
        sub = getattr(mod, attr, None)
        if sub is None:
            continue
        conv = getattr(sub, "conv", None) if hasattr(sub, "conv") else sub
        if hasattr(conv, "stride"):
            s = _stride_to_int(getattr(conv, "stride"))
            if s is not None:
                return s
    for layer in mod.modules():
        if isinstance(layer, torch.nn.Conv2d):
            s = _stride_to_int(layer.stride)
            if s is not None:
                return s
    return None


def _find_stride_conv(mod: torch.nn.Module, stride: int = 2) -> Optional[torch.nn.Conv2d]:
    for layer in mod.modules():
        if isinstance(layer, torch.nn.Conv2d):
            s = _stride_to_int(layer.stride)
            if s == stride:
                return layer
    return None


def _find_first_conv(mod: torch.nn.Module) -> Optional[torch.nn.Conv2d]:
    for layer in mod.modules():
        if isinstance(layer, torch.nn.Conv2d):
            return layer
    return None


def _to_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)


def _vector_summary(vec: Optional[torch.Tensor]) -> Dict[str, Any]:
    if vec is None:
        return {"count": 0}
    v = vec.detach().float().reshape(-1)
    n = int(v.numel())
    if n == 0:
        return {"count": 0}
    q = torch.quantile(v, torch.tensor([0.5, 0.9, 0.99], device=v.device))
    return {
        "count": n,
        "min": _to_float(v.min().item()),
        "max": _to_float(v.max().item()),
        "mean": _to_float(v.mean().item()),
        "std": _to_float(v.std(unbiased=False).item()),
        "p50": _to_float(q[0].item()),
        "p90": _to_float(q[1].item()),
        "p99": _to_float(q[2].item()),
    }


def _is_nonzero_map(values: Dict[str, Any], eps: float = 1e-12) -> bool:
    for v in values.values():
        if abs(_to_float(v, 0.0)) > float(eps):
            return True
    return False


def _should_capture_delta(step: int) -> bool:
    # Sparse schedule to survive grad-accumulate setups (e.g., nbs=64, batch=6 => accumulate~11).
    return int(step) in {1, 2, 4, 8, 12, 16, 24, 32, 48, 64}


def _tensor_stats(tensor: Optional[torch.Tensor]) -> Dict[str, Any]:
    if tensor is None:
        return {"shape": None, "count": 0}
    t = tensor.detach()
    x = t.float()
    n = int(x.numel())
    if n == 0:
        return {"shape": list(t.shape), "count": 0}
    nan_mask = torch.isnan(x)
    inf_mask = torch.isinf(x)
    finite = x[~(nan_mask | inf_mask)]
    zero_ratio = _to_float((x == 0).float().mean().item()) if n > 0 else 0.0
    out: Dict[str, Any] = {
        "shape": list(t.shape),
        "count": n,
        "dtype": str(t.dtype),
        "device": str(t.device),
        "nan_count": int(nan_mask.sum().item()),
        "inf_count": int(inf_mask.sum().item()),
        "zero_ratio": zero_ratio,
    }
    if int(finite.numel()) > 0:
        out.update(
            {
                "mean": _to_float(finite.mean().item()),
                "var": _to_float(finite.var(unbiased=False).item()),
                "min": _to_float(finite.min().item()),
                "max": _to_float(finite.max().item()),
                "absmax": _to_float(finite.abs().max().item()),
                "l2norm": _to_float(torch.linalg.vector_norm(finite).item()),
            }
        )
    return out


def record_tensor_stats(name: str, tensor: Optional[torch.Tensor], md_lines: Any) -> Dict[str, Any]:
    """Collect tensor statistics and store into a list(dict-text) or dict container."""

    stats = _tensor_stats(tensor)
    if isinstance(md_lines, dict):
        md_lines[name] = stats
    elif isinstance(md_lines, list):
        md_lines.append(f"- {name}: `{json.dumps(stats, ensure_ascii=False)}`")
    return stats


def _concat_candidates(seq: Any) -> List[str]:
    out: List[str] = []
    for i, layer in enumerate(seq):
        if layer.__class__.__name__ == "Concat":
            out.append(f"idx={i}, f={getattr(layer, 'f', None)}")
    return out


def _f_as_list(v: Any) -> List[int]:
    if isinstance(v, int):
        return [int(v)]
    if isinstance(v, (list, tuple)):
        out: List[int] = []
        for x in v:
            try:
                out.append(int(x))
            except Exception:
                continue
        return out
    return []


def _locate_detect(seq: Any) -> Tuple[int, Any]:
    if not seq:
        raise RuntimeError("Empty model sequence.")
    idx = len(seq) - 1
    det = seq[idx]
    if det.__class__.__name__.lower() == "detect":
        return idx, det
    for i, layer in enumerate(seq):
        if layer.__class__.__name__.lower() == "detect":
            return i, layer
    raise RuntimeError("Detect layer not found.")


def _locate_p4_to_p3_fuse(seq: Any) -> Tuple[int, int]:
    detect_idx, detect = _locate_detect(seq)
    detect_f = _f_as_list(getattr(detect, "f", []))
    if detect_f:
        p3_idx = int(detect_f[0])
        fuse_idx = p3_idx - 1
        if 0 <= fuse_idx < detect_idx:
            f_list = _f_as_list(getattr(seq[fuse_idx], "f", []))
            if len(f_list) >= 2:
                return fuse_idx, int(f_list[1])

    candidates: List[Tuple[int, List[int]]] = []
    for i, layer in enumerate(seq):
        f_list = _f_as_list(getattr(layer, "f", []))
        if layer.__class__.__name__ == "Concat" and len(f_list) == 2 and f_list[0] == -1:
            candidates.append((i, f_list))
    if candidates:
        fuse_idx, f_list = candidates[-1]
        return fuse_idx, int(f_list[1])

    raise RuntimeError(f"Unable to locate P4->P3 fuse. Concat candidates: {_concat_candidates(seq)}")


def _infer_device_dtype(seq: Any, start_idx: int) -> Tuple[Optional[torch.device], Optional[torch.dtype]]:
    for step in range(0, 6):
        for idx in (start_idx + step, start_idx - step):
            if idx < 0 or idx >= len(seq):
                continue
            layer = seq[idx]
            try:
                p = next(layer.parameters())
                return p.device, p.dtype
            except StopIteration:
                continue
            except Exception:
                continue
    return None, None


def find_p3_tensor(model: Any, x: torch.Tensor) -> Optional[torch.Tensor]:
    """Best-effort utility: capture P3 feature entering Detect from a single forward."""

    det_model = None
    seq = None
    if hasattr(model, "model") and hasattr(getattr(model, "model"), "model"):
        det_model = getattr(model, "model")
        seq = getattr(det_model, "model", None)
    elif hasattr(model, "model"):
        det_model = model
        seq = getattr(det_model, "model", None)
    if seq is None:
        return None

    try:
        _, detect = _locate_detect(seq)
    except Exception:
        return None

    captured: Dict[str, torch.Tensor] = {}

    def _pre_hook(_: Any, inputs: Tuple[Any, ...]) -> None:
        if not inputs:
            return
        feats = inputs[0]
        if isinstance(feats, (list, tuple)) and feats:
            t = feats[0]
            if isinstance(t, torch.Tensor):
                captured["p3"] = t.detach()

    h = detect.register_forward_pre_hook(_pre_hook)
    try:
        with torch.no_grad():
            if hasattr(det_model, "forward"):
                det_model(x)
    except Exception:
        pass
    finally:
        try:
            h.remove()
        except Exception:
            pass
    return captured.get("p3")


def _resolve_project_root(cfg: Any) -> Path:
    p = str(_deep_get(cfg, "project_root", default="")).strip()
    if p:
        return Path(p).expanduser().resolve()
    return Path(__file__).resolve().parents[3]


def _resolve_exp_dir(cfg: Any) -> Optional[Path]:
    env_exp = str(os.environ.get("ENHANCE241_EXP_DIR", "")).strip()
    if env_exp:
        p = Path(env_exp).expanduser().resolve()
        if p.exists():
            return p

    project_root = _resolve_project_root(cfg)
    yolo_version = str(_deep_get(cfg, "yolo_version", default="yolo11"))
    exp_name = str(_deep_get(cfg, "exp_name", default="defect241"))
    run_name = str(_deep_get(cfg, "run_name", default="")).strip()

    exp_root = project_root / "experiments" / yolo_version / exp_name
    if run_name:
        exp_root = exp_root / run_name

    candidates: List[Path] = []
    if exp_root.exists():
        try:
            candidates = [p for p in exp_root.glob("exp_*") if p.is_dir()]
        except Exception:
            candidates = []
    if candidates:
        candidates.sort(key=lambda p: p.stat().st_mtime)
        return candidates[-1]

    for key in ("weights", "model"):
        raw = str(_deep_get(cfg, key, default="")).strip()
        if not raw:
            continue
        p = Path(raw).expanduser()
        if not p.is_absolute():
            p = (project_root / p).resolve()
        for par in p.parents:
            if par.name.startswith("exp_"):
                return par
    return None


def maybe_open_md(cfg: Any, exp_dir: Optional[Path], module_key: str = "a4") -> Optional[Path]:
    _ = cfg
    _ = module_key
    if exp_dir is None:
        return None
    try:
        (exp_dir / "train").mkdir(parents=True, exist_ok=True)
    except Exception:
        return None
    return exp_dir / "train" / "enhance241_check.md"


def _git_hash(project_root: Path) -> str:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(project_root), "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        )
        return out.decode("utf-8", errors="ignore").strip()
    except Exception:
        return "unknown"


def _infer_label_path(image_path: Path) -> Path:
    parts = list(image_path.parts)
    img_keys = ("images", "image")
    lbl_keys = ("labels", "label", "lable")

    for img_key in img_keys:
        if img_key in parts:
            idx = parts.index(img_key)
            for lbl_key in lbl_keys:
                parts2 = parts.copy()
                parts2[idx] = lbl_key
                cand = Path(*parts2).with_suffix(".txt")
                if cand.exists():
                    return cand
            parts[idx] = "labels"
            return Path(*parts).with_suffix(".txt")

    for lbl_key in lbl_keys:
        try:
            split = image_path.parent.name
            cand2 = image_path.parent.parent / lbl_key / split / f"{image_path.stem}.txt"
            if cand2.exists():
                return cand2
        except Exception:
            pass

    return image_path.with_suffix(".txt")


def _image_has_label(image_path: Path) -> bool:
    label_path = _infer_label_path(image_path)
    if not label_path.exists():
        return False
    try:
        with label_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    return True
    except Exception:
        return False
    return False


def _resolve_data_yaml(cfg: Any) -> Optional[Path]:
    project_root = _resolve_project_root(cfg)
    data_ref = str(_deep_get(cfg, "data", default="")).strip()
    if not data_ref:
        return None
    p = Path(data_ref).expanduser()
    if not p.is_absolute():
        p = (project_root / p).resolve()
    return p if p.exists() else None


def _resolve_val_source(cfg: Any) -> Optional[Path]:
    data_yaml = _resolve_data_yaml(cfg)
    if data_yaml is None:
        return None

    try:
        import yaml  # type: ignore

        with data_yaml.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except Exception:
        return None

    val_ref = data.get("val", "")
    if isinstance(val_ref, (list, tuple)):
        val_ref = val_ref[0] if val_ref else ""
    val_ref = str(val_ref).strip()
    if not val_ref:
        return None

    data_root_override = str(_deep_get(cfg, "data_root", default="")).strip()
    root_ref = str(data.get("path", "")).strip()

    candidates: List[Path] = []
    v = Path(val_ref).expanduser()
    if v.is_absolute():
        candidates.append(v)
    if data_root_override:
        candidates.append(Path(data_root_override).expanduser() / val_ref)
    if root_ref:
        root = Path(root_ref).expanduser()
        if not root.is_absolute():
            root = (data_yaml.parent / root).resolve()
        candidates.append(root / val_ref)
    candidates.append((data_yaml.parent / val_ref).resolve())
    candidates.append((_resolve_project_root(cfg) / val_ref).resolve())

    for c in candidates:
        if c.exists():
            return c
    return candidates[0] if candidates else None


def _list_source_images(source: Path) -> List[Path]:
    if source.is_file() and source.suffix.lower() == ".txt":
        items: List[Path] = []
        try:
            with source.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    p = Path(line).expanduser()
                    if not p.is_absolute():
                        p = (source.parent / p).resolve()
                    if p.exists() and p.suffix.lower() in _IMAGE_SUFFIX:
                        items.append(p)
        except Exception:
            return []
        return items
    if source.is_dir():
        return sorted([p for p in source.rglob("*") if p.suffix.lower() in _IMAGE_SUFFIX])
    return []


class Enhance241CheckRecorder:
    """Per-run recorder for A1/A2/A3 diagnostics; writes once at process exit."""

    def __init__(self, module_key: str, cfg: Any, exp_dir: Path, md_path: Path) -> None:
        self.module_key = str(module_key)
        self.cfg = cfg
        self.exp_dir = exp_dir
        self.md_path = md_path
        self.project_root = _resolve_project_root(cfg)
        self.git = _git_hash(self.project_root)
        self.created_at = dt.datetime.now().isoformat(timespec="seconds")

        self.lock = threading.Lock()
        self.patch_info: Dict[str, Any] = {}

        self.a1: Dict[str, Any] = {}
        self.a2: Dict[str, Any] = {
            "params": {},
            "grad_l2": {},
            "grad_absmax": {},
            "delta_l2": {},
            "output_grad_l2": {},
        }
        self.a4: Dict[str, Any] = {}
        self.b_specific: Dict[str, Any] = {}

        self.notes: List[str] = []

        self._handles: List[Any] = []
        self._detect_hooked = False
        self._val_done = False
        self._flushed = False
        self._param_before: Dict[str, torch.Tensor] = {}

        atexit.register(self.flush)

    def set_patch_info(self, info: Dict[str, Any]) -> None:
        with self.lock:
            self.patch_info.update(info)

    def add_note(self, text: str) -> None:
        with self.lock:
            self.notes.append(str(text))

    def attach_detect_hooks(self, detect_module: Any, detect_idx: int) -> None:
        with self.lock:
            if self._detect_hooked:
                return
            self.a1["detect_idx"] = int(detect_idx)
            self._detect_hooked = True

        def _pre_hook(mod: Any, inputs: Tuple[Any, ...]) -> None:
            try:
                feats = inputs[0] if inputs else None
                if isinstance(feats, (list, tuple)) and feats:
                    self._record_detect_input(feats, mod)
            except Exception as exc:
                self.add_note(f"detect_pre_hook_error: {exc}")

        def _fwd_hook(_: Any, __: Tuple[Any, ...], output: Any) -> None:
            try:
                self._record_head_output(output)
            except Exception as exc:
                self.add_note(f"detect_fwd_hook_error: {exc}")

        try:
            self._handles.append(detect_module.register_forward_pre_hook(_pre_hook))
            self._handles.append(detect_module.register_forward_hook(_fwd_hook))
        except Exception as exc:
            self.add_note(f"detect_hook_register_failed: {exc}")

    def register_module_params(self, module: torch.nn.Module, prefix: str) -> None:
        for name, p in module.named_parameters():
            full = f"{prefix}.{name}"
            with self.lock:
                self.a2["params"][full] = {
                    "shape": list(p.shape),
                    "numel": int(p.numel()),
                    "requires_grad": bool(p.requires_grad),
                }
            if p.requires_grad:
                try:
                    handle = p.register_hook(lambda grad, n=full: self._record_param_grad(n, grad))
                    self._handles.append(handle)
                except Exception as exc:
                    self.add_note(f"param_hook_failed:{full}:{exc}")

    def _record_param_grad(self, name: str, grad: Optional[torch.Tensor]) -> None:
        if grad is None:
            return
        g = grad.detach().float()
        with self.lock:
            if name not in self.a2["grad_l2"]:
                self.a2["grad_l2"][name] = _to_float(torch.linalg.vector_norm(g).item())
                self.a2["grad_absmax"][name] = _to_float(g.abs().max().item())

    def capture_param_before(self, module: torch.nn.Module, prefix: str) -> None:
        for name, p in module.named_parameters():
            full = f"{prefix}.{name}"
            if full in self._param_before:
                continue
            try:
                self._param_before[full] = p.detach().float().cpu().clone()
            except Exception:
                continue

    def capture_param_delta(self, module: torch.nn.Module, prefix: str) -> None:
        with self.lock:
            # Keep re-sampling until we observe non-zero updates; this avoids false zeros under grad accumulation.
            if self.a2["delta_l2"] and _is_nonzero_map(self.a2["delta_l2"]):
                return
        delta: Dict[str, float] = {}
        for name, p in module.named_parameters():
            full = f"{prefix}.{name}"
            before = self._param_before.get(full)
            if before is None:
                continue
            try:
                now = p.detach().float().cpu()
                d = now - before
                delta[full] = _to_float(torch.linalg.vector_norm(d).item())
            except Exception:
                continue
        with self.lock:
            self.a2["delta_l2"] = delta

    def record_output_grad(self, name: str, grad: Optional[torch.Tensor]) -> None:
        if grad is None:
            return
        g = grad.detach().float()
        with self.lock:
            if name not in self.a2["output_grad_l2"]:
                self.a2["output_grad_l2"][name] = _to_float(torch.linalg.vector_norm(g).item())

    def record_module_compare(
        self,
        name: str,
        patched: Optional[torch.Tensor],
        baseline: Optional[torch.Tensor],
        input_tensor: Optional[torch.Tensor] = None,
    ) -> None:
        entry: Dict[str, Any] = {}
        if input_tensor is not None:
            entry["input"] = _tensor_stats(input_tensor)
        entry["patched"] = _tensor_stats(patched)
        entry["baseline"] = _tensor_stats(baseline)

        if isinstance(patched, torch.Tensor) and isinstance(baseline, torch.Tensor):
            if patched.shape == baseline.shape and patched.numel() > 0:
                p = patched.detach().float().reshape(patched.shape[0], -1)
                b = baseline.detach().float().reshape(baseline.shape[0], -1)
                try:
                    cos = F.cosine_similarity(p, b, dim=1).mean().item()
                except Exception:
                    cos = float("nan")
                entry["patched_vs_baseline"] = {
                    "cosine_mean": _to_float(cos),
                    "var_ratio": _to_float(entry["patched"].get("var", 0.0))
                    / (_to_float(entry["baseline"].get("var", 0.0)) + 1e-12),
                }

        with self.lock:
            self.a1[name] = entry

    def record_b3_weight(self, prefix: str, tag: str, w_norm: torch.Tensor, step: int) -> None:
        if w_norm.numel() < 2:
            return
        key = f"{prefix}:{tag}"
        row = {
            "step": int(step),
            "w_lo": _to_float(w_norm[0].item()),
            "w_hi": _to_float(w_norm[1].item()),
        }
        with self.lock:
            hist = self.b_specific.setdefault("b3_weights", {}).setdefault(key, [])
            if len(hist) < 32:
                hist.append(row)

    def record_b3_feature_relation(self, prefix: str, lo: torch.Tensor, refined: torch.Tensor) -> None:
        if lo.shape != refined.shape:
            return
        l = lo.detach().float().reshape(lo.shape[0], -1)
        r = refined.detach().float().reshape(refined.shape[0], -1)
        try:
            cos = F.cosine_similarity(l, r, dim=1).mean().item()
        except Exception:
            cos = float("nan")
        item = {
            "cosine_mean": _to_float(cos),
            "refined_var": _to_float(refined.detach().float().var(unbiased=False).item()),
            "lo_var": _to_float(lo.detach().float().var(unbiased=False).item()),
            "var_ratio": _to_float(refined.detach().float().var(unbiased=False).item())
            / (_to_float(lo.detach().float().var(unbiased=False).item()) + 1e-12),
        }
        with self.lock:
            self.b_specific.setdefault("b3_feature_relation", {})[prefix] = item

    def record_a1_payload(self, name: str, payload: Dict[str, Any]) -> None:
        with self.lock:
            self.a1[name] = payload

    def record_scalar_curve(self, key: str, value: float, step: int, max_steps: int = 30) -> None:
        if int(step) > int(max_steps):
            return
        row = {"step": int(step), "value": _to_float(value)}
        with self.lock:
            hist = self.b_specific.setdefault("scalar_curves", {}).setdefault(str(key), [])
            if len(hist) < int(max_steps) + 2:
                hist.append(row)

    def record_distribution(self, key: str, values: Optional[torch.Tensor], step: int, max_steps: int = 30) -> None:
        if values is None or int(step) > int(max_steps):
            return
        row = {"step": int(step)}
        row.update(_vector_summary(values))
        with self.lock:
            hist = self.b_specific.setdefault("distributions", {}).setdefault(str(key), [])
            if len(hist) < int(max_steps) + 2:
                hist.append(row)

    def _record_detect_input(self, feats: Any, mod: Any) -> None:
        with self.lock:
            if "detect_input_p3" in self.a1:
                return

        p3 = feats[0] if isinstance(feats, (list, tuple)) and feats else None
        if not isinstance(p3, torch.Tensor):
            return

        all_stats: List[Dict[str, Any]] = []
        head_hw: List[List[int]] = []
        for idx, feat in enumerate(feats):
            if not isinstance(feat, torch.Tensor):
                continue
            st = _tensor_stats(feat)
            all_stats.append({"head_index": int(idx), **st})
            try:
                head_hw.append([int(feat.shape[-2]), int(feat.shape[-1])])
            except Exception:
                pass

        data: Dict[str, Any] = {
            "detect_input_p3": _tensor_stats(p3),
            "head_shapes": [list(x.shape) for x in feats if isinstance(x, torch.Tensor)],
            "head_hw": head_hw,
            "detect_input_all": all_stats,
            "head_strides": _f_as_list(getattr(mod, "stride", [])),
            "head_nl": int(getattr(mod, "nl", len(feats))),
        }
        with self.lock:
            self.a1.update(data)

    def _record_head_output(self, output: Any) -> None:
        with self.lock:
            if "head_conf_step0" in self.a1:
                return

        scores = None
        if isinstance(output, dict):
            scores = output.get("scores")
        elif isinstance(output, tuple) and len(output) >= 2 and isinstance(output[1], dict):
            scores = output[1].get("scores")

        if not isinstance(scores, torch.Tensor) or scores.numel() == 0:
            return

        s = torch.sigmoid(scores.detach().float())
        flat = s.reshape(s.shape[0], -1)
        max_per_img = flat.max(dim=1).values
        k = min(20, int(flat.shape[1]))
        topk_mean = flat.topk(k=k, dim=1).values.mean(dim=1)

        payload = {
            "head_scores": _tensor_stats(scores),
            "head_conf_step0": _vector_summary(max_per_img),
            "head_conf_topk_mean_step0": _vector_summary(topk_mean),
        }
        with self.lock:
            self.a1.update(payload)

    def maybe_run_val_separability(self, yolo_obj: Any, cfg: Any, max_images: int = 200) -> None:
        with self.lock:
            if self._val_done:
                return
            self._val_done = True

        try:
            if not hasattr(yolo_obj, "predict"):
                self.add_note("A3 skipped: model has no predict().")
                return
            source = _resolve_val_source(cfg)
            if source is None:
                self.add_note("A3 skipped: val source unresolved.")
                return

            imgs = _list_source_images(source)
            if not imgs:
                self.add_note(f"A3 skipped: no images under {source}")
                return
            imgs = imgs[: int(max_images)]

            metric_conf = _safe_float(_deep_get(cfg, "metric_conf", default=0.01), 0.01)
            nms_iou = _safe_float(_deep_get(cfg, "nms_iou", default=0.7), 0.7)
            max_det = _safe_int(_deep_get(cfg, "max_det", default=300), 300)
            eval_batch = _safe_int(_deep_get(cfg, "eval_batch", default=1), 1)
            eval_device = str(_deep_get(cfg, "eval_device", default=_deep_get(cfg, "device", default="")))

            results = yolo_obj.predict(
                source=[str(p) for p in imgs],
                conf=float(metric_conf),
                iou=float(nms_iou),
                max_det=int(max_det),
                save=False,
                verbose=False,
                batch=int(eval_batch),
                device=str(eval_device),
            )

            pos_scores: List[float] = []
            neg_scores: List[float] = []
            sample_rows: List[Dict[str, Any]] = []

            for img_path, res in zip(imgs, results):
                has_gt = _image_has_label(img_path)
                max_conf = 0.0
                pred_count = 0
                if getattr(res, "boxes", None) is not None and getattr(res.boxes, "conf", None) is not None:
                    conf_tensor = res.boxes.conf
                    if len(conf_tensor) > 0:
                        pred_count = int(len(conf_tensor))
                        max_conf = float(conf_tensor.max().item())
                if has_gt:
                    pos_scores.append(max_conf)
                else:
                    neg_scores.append(max_conf)
                if len(sample_rows) < 10:
                    sample_rows.append(
                        {
                            "image": str(img_path),
                            "has_gt": bool(has_gt),
                            "pred_count": int(pred_count),
                            "max_conf": float(max_conf),
                        }
                    )

            pos_tensor = torch.tensor(pos_scores, dtype=torch.float32) if pos_scores else None
            neg_tensor = torch.tensor(neg_scores, dtype=torch.float32) if neg_scores else None

            with self.lock:
                self.a4 = {
                    "source": str(source),
                    "sample_count": int(len(imgs)),
                    "metric_conf": float(metric_conf),
                    "nms_iou": float(nms_iou),
                    "max_det": int(max_det),
                    "pos_count": int(len(pos_scores)),
                    "neg_count": int(len(neg_scores)),
                    "pos_max_conf": _vector_summary(pos_tensor),
                    "neg_max_conf": _vector_summary(neg_tensor),
                    "pos_scores_raw": [float(v) for v in pos_scores[:5000]],
                    "neg_scores_raw": [float(v) for v in neg_scores[:5000]],
                    "sample_rows": sample_rows,
                }
        except Exception as exc:
            self.add_note(f"A3 predict failed: {exc}")

    def _write_roc_overlay_artifacts(self) -> Dict[str, Any]:
        pos_scores = self.a4.get("pos_scores_raw", []) if isinstance(self.a4, dict) else []
        neg_scores = self.a4.get("neg_scores_raw", []) if isinstance(self.a4, dict) else []
        if not isinstance(pos_scores, list) or not isinstance(neg_scores, list):
            return {}
        if len(pos_scores) == 0 or len(neg_scores) == 0:
            return {}

        train_dir = self.exp_dir / "train"
        train_dir.mkdir(parents=True, exist_ok=True)
        run_tag = dt.datetime.now().strftime("%Y%m%d%H%M%S")
        module_tag = str(self.module_key)

        roc_rows: List[Dict[str, Any]] = []
        thresholds = [i / 100.0 for i in range(101)]
        n_pos = float(max(1, len(pos_scores)))
        n_neg = float(max(1, len(neg_scores)))
        for thr in thresholds:
            tp = sum(1 for v in pos_scores if float(v) >= thr)
            fp = sum(1 for v in neg_scores if float(v) >= thr)
            recall = float(tp) / n_pos
            fpr = float(fp) / n_neg
            roc_rows.append(
                {
                    "module": module_tag,
                    "run_tag": run_tag,
                    "threshold": float(thr),
                    "recall": float(recall),
                    "fpr": float(fpr),
                }
            )

        roc_csv = train_dir / "roc_overlay.csv"
        new_file = not roc_csv.exists()
        with roc_csv.open("a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["module", "run_tag", "threshold", "recall", "fpr"])
            if new_file:
                writer.writeheader()
            writer.writerows(roc_rows)

        targets = [0.05, 0.10, 0.30]
        key_rows: List[Dict[str, Any]] = []
        for target in targets:
            best = min(roc_rows, key=lambda r: abs(float(r["fpr"]) - float(target)))
            key_rows.append(
                {
                    "module": module_tag,
                    "run_tag": run_tag,
                    "target_fpr": float(target),
                    "threshold": float(best["threshold"]),
                    "fpr": float(best["fpr"]),
                    "recall": float(best["recall"]),
                }
            )

        key_csv = train_dir / "roc_keypoints.csv"
        key_new = not key_csv.exists()
        with key_csv.open("a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["module", "run_tag", "target_fpr", "threshold", "fpr", "recall"])
            if key_new:
                writer.writeheader()
            writer.writerows(key_rows)

        roc_png = train_dir / "roc_overlay.png"
        try:
            import matplotlib.pyplot as plt  # type: ignore

            curves: Dict[str, List[Dict[str, Any]]] = {}
            with roc_csv.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    mod = str(row.get("module", "")).strip()
                    tag = str(row.get("run_tag", "")).strip()
                    if not mod or not tag:
                        continue
                    k = f"{mod}:{tag}"
                    curves.setdefault(k, []).append(row)

            latest_by_module: Dict[str, str] = {}
            for key in curves:
                mod, _, tag = key.partition(":")
                old_tag = latest_by_module.get(mod)
                if old_tag is None or tag > old_tag:
                    latest_by_module[mod] = tag

            fig, ax = plt.subplots(figsize=(7.2, 5.0), dpi=150)
            for mod, tag in sorted(latest_by_module.items()):
                key = f"{mod}:{tag}"
                rows = curves.get(key, [])
                rows_sorted = sorted(rows, key=lambda r: float(r.get("fpr", 0.0)))
                xs = [float(r.get("fpr", 0.0)) for r in rows_sorted]
                ys = [float(r.get("recall", 0.0)) for r in rows_sorted]
                ax.plot(xs, ys, linewidth=1.0, label=f"{mod}@{tag[-6:]}")
            ax.set_xlabel("FPR")
            ax.set_ylabel("Recall")
            ax.set_title("enhance241 ROC overlay")
            ax.grid(True, linewidth=0.4, alpha=0.4)
            if latest_by_module:
                ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(roc_png)
            plt.close(fig)
        except Exception as exc:
            self.add_note(f"roc_overlay_plot_failed:{exc}")

        return {
            "roc_overlay_csv": str(roc_csv),
            "roc_overlay_png": str(roc_png),
            "roc_keypoints_csv": str(key_csv),
            "roc_keypoints": key_rows,
        }

    def _judge_conclusion(self) -> Tuple[str, List[str]]:
        reasons: List[str] = []
        structure = False
        a1_bad = False
        a2_bad = False
        a4_bad = False

        a1_items = [v for v in self.a1.values() if isinstance(v, dict)]
        for item in a1_items:
            patched = item.get("patched") if isinstance(item, dict) else None
            if isinstance(patched, dict):
                if int(patched.get("nan_count", 0)) > 0 or int(patched.get("inf_count", 0)) > 0:
                    structure = True
                    a1_bad = True
                    reasons.append("A1 patched tensor contains NaN/Inf.")
            comp = item.get("patched_vs_baseline") if isinstance(item, dict) else None
            if isinstance(comp, dict):
                vr = _to_float(comp.get("var_ratio", 1.0), 1.0)
                if (vr > 4.0) or (vr < 0.25):
                    structure = True
                    a1_bad = True
                    reasons.append(f"A1 variance ratio abnormal ({vr:.4f}).")

        grad_map = self.a2.get("grad_l2", {}) if isinstance(self.a2, dict) else {}
        delta_map = self.a2.get("delta_l2", {}) if isinstance(self.a2, dict) else {}
        if grad_map and all(_to_float(v, 0.0) == 0.0 for v in grad_map.values()):
            structure = True
            a2_bad = True
            reasons.append("A2 gradients are all zero.")
        if delta_map and all(_to_float(v, 0.0) == 0.0 for v in delta_map.values()):
            if grad_map and _is_nonzero_map(grad_map):
                reasons.append("A2 deltas sampled before optimizer step (gradients are non-zero; check accumulate schedule).")
            else:
                structure = True
                a2_bad = True
                reasons.append("A2 parameter update deltas are all zero.")

        pos = self.a4.get("pos_max_conf", {}) if isinstance(self.a4, dict) else {}
        neg = self.a4.get("neg_max_conf", {}) if isinstance(self.a4, dict) else {}
        if isinstance(pos, dict) and isinstance(neg, dict) and pos.get("count", 0) and neg.get("count", 0):
            if _to_float(pos.get("p90", 0.0)) <= _to_float(neg.get("p90", 0.0)):
                a4_bad = True
                reasons.append("A3 positive/negative max_conf not separable (pos p90 <= neg p90).")
                reasons.append("A3 is a patch-time snapshot on pretrain weights; verify post-train sweep before final judgment.")

        # Treat A3 as supporting evidence; avoid hard-failing solely on pretrain separability.
        if a4_bad and (a1_bad or a2_bad):
            structure = True

        if not reasons:
            reasons.append("No hard structural anomaly detected from A1/A2/A3; likely hyper-parameter mismatch.")

        if structure:
            next_actions = {
                "a4": "Next action: use gradient-safe residual a4 (baseline_downsample + alpha*spd_branch, alpha small non-zero init with bounded tanh, last conv zero-init), then rerun >=10 epochs.",
                "a9": "Next action: keep a9 SE-SAM residual-safe with non-zero alpha warm start, verify Gate-2 delta after first effective optimizer step, then rerun >=10 epochs.",
                "a7": "Next action: keep HorNet delta residual-safe and non-zero alpha init (>=0.03), verify first effective optimizer-step delta>0, then rerun >=10 epochs.",
                "b3": "Next action: use residual-safe b3 (lo + alpha*refine(weighted-hi/lo - lo), alpha init 0, refine last conv zero-init), then rerun >=10 epochs.",
                "b9": "Next action: keep b9 improved_CSP fusion residual-safe, confirm P4->P3 fusion delta is non-zero and low-FPR recall does not regress, then rerun >=10 epochs.",
                "b7": "Next action: keep CARAFE residual-safe with non-zero alpha init and chunked reassembly, then rerun >=10 epochs.",
                "c5": "Next action: use gradient-safe c5 (out=x+alpha*BRA(x), alpha small non-zero init, BRA proj zero-init), then rerun >=10 epochs.",
                "c9": "Next action: use c9 as post-fusion SE-SAM guardrail; compare `full/channel/spatial` modes for FP suppression without low-FPR recall drop.",
                "c7": "Next action: keep MCBAM residual-safe with non-zero alpha init and validate channel/spatial mode separately, then rerun >=10 epochs.",
                "d7": "Next action: keep P3-only d7 with TAL-compatible stride and tuned cls bias shift, then rerun >=10 epochs and inspect low-FPR recall.",
                "d9": "Next action: keep d9 score-calib residual branch conservative (small alpha, zero-init tail), then rerun >=10 epochs and inspect FPR=0.05/0.1 recall.",
            }
            if self.module_key in next_actions:
                reasons.append(next_actions[self.module_key])
            return "结构问题", reasons

        reasons.append("Next action: keep structure, try lr0 down 3x + warmup_epochs=3 then rerun >=10 epochs.")
        return "超参问题", reasons

    def flush(self) -> None:
        with self.lock:
            if self._flushed:
                return
            self._flushed = True

        conclusion, reasons = self._judge_conclusion()
        roc_artifacts = self._write_roc_overlay_artifacts()

        enabled_flags = []
        enh = _deep_get(self.cfg, "enhance241", default={}) or {}
        if isinstance(enh, dict):
            enabled_flags = [k for k, v in enh.items() if isinstance(v, bool) and v]

        params_map = self.a2.get("params", {}) if isinstance(self.a2, dict) else {}
        grad_map = self.a2.get("grad_l2", {}) if isinstance(self.a2, dict) else {}
        delta_map = self.a2.get("delta_l2", {}) if isinstance(self.a2, dict) else {}
        param_total = int(len(params_map)) if isinstance(params_map, dict) else 0
        param_trainable = int(
            sum(
                1
                for v in (params_map.values() if isinstance(params_map, dict) else [])
                if isinstance(v, dict) and bool(v.get("requires_grad", False))
            )
        )
        gate0_ok = bool(param_trainable > 0)
        gate2_ok = bool(
            isinstance(grad_map, dict)
            and isinstance(delta_map, dict)
            and any(abs(_to_float(v, 0.0)) > 0.0 for v in grad_map.values())
            and any(abs(_to_float(v, 0.0)) > 0.0 for v in delta_map.values())
        )

        lines: List[str] = []
        lines.append(f"\n## Run {dt.datetime.now().isoformat(timespec='seconds')}")
        lines.append(f"- module: `{self.module_key}`")
        lines.append(f"- exp_dir: `{self.exp_dir}`")
        lines.append(f"- git_hash: `{self.git}`")
        lines.append(f"- enable_flags: `{enabled_flags}`")
        lines.append("- patch_info:")
        lines.append("```json")
        lines.append(json.dumps(self.patch_info, ensure_ascii=False, indent=2))
        lines.append("```")

        lines.append("### Gate-0 Optimizer Registration")
        lines.append(f"- pass: `{gate0_ok}`")
        lines.append(f"- params_total: `{param_total}`")
        lines.append(f"- params_trainable: `{param_trainable}`")

        lines.append("### Gate-1 Step0 Equivalence (legacy: A1)")
        lines.append("```json")
        lines.append(json.dumps(self.a1, ensure_ascii=False, indent=2))
        lines.append("```")

        lines.append("### Gate-2 Trainability (legacy: A2)")
        lines.append(f"- pass: `{gate2_ok}`")
        lines.append("```json")
        lines.append(json.dumps(self.a2, ensure_ascii=False, indent=2))
        lines.append("```")

        lines.append("### Gate-3 Score Separability Snapshot (legacy: A3)")
        lines.append("```json")
        lines.append(json.dumps(self.a4, ensure_ascii=False, indent=2))
        lines.append("```")

        if self.b_specific:
            lines.append("### B-Specific")
            lines.append("```json")
            lines.append(json.dumps(self.b_specific, ensure_ascii=False, indent=2))
            lines.append("```")

        if roc_artifacts:
            lines.append("### ROC Overlay")
            lines.append(f"- roc_overlay_png: `{roc_artifacts.get('roc_overlay_png', '')}`")
            lines.append(f"- roc_overlay_csv: `{roc_artifacts.get('roc_overlay_csv', '')}`")
            lines.append(f"- roc_keypoints_csv: `{roc_artifacts.get('roc_keypoints_csv', '')}`")
            lines.append("```json")
            lines.append(json.dumps(roc_artifacts.get("roc_keypoints", []), ensure_ascii=False, indent=2))
            lines.append("```")

        if self.notes:
            lines.append("### Notes")
            for n in self.notes:
                lines.append(f"- {n}")

        lines.append("### Gate Legend")
        lines.append("- `Gate-1`: patch 后 step0 的数值稳定性/等价启动检查。")
        lines.append("- `Gate-2`: 参数是否进 optimizer、梯度是否有效、是否观察到参数更新。")
        lines.append("- `Gate-3`: val 快照上正负样本 max_conf 可分性（用于早期诊断，非最终指标结论）。")

        lines.append(f"### 结论: {conclusion}")
        for r in reasons:
            lines.append(f"- {r}")

        try:
            self.md_path.parent.mkdir(parents=True, exist_ok=True)
            with self.md_path.open("a", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
        except Exception:
            pass


_CHECK_RECORDER_LOCK = threading.Lock()
_CHECK_RECORDER_REGISTRY: Dict[str, Enhance241CheckRecorder] = {}
_DEBUG_BIND_LOCK = threading.Lock()
_DEBUG_KEY_COUNTER = count(1)
_DEBUG_RECORDER_BY_KEY: Dict[str, Enhance241CheckRecorder] = {}
_DEBUG_BASELINE_BY_KEY: Dict[str, torch.nn.Module] = {}


def get_check_recorder(module_key: str, cfg: Any, patch_info: Optional[Dict[str, Any]] = None) -> Optional[Enhance241CheckRecorder]:
    exp_dir = _resolve_exp_dir(cfg)
    md_path = maybe_open_md(cfg, exp_dir, module_key=module_key)
    if exp_dir is None or md_path is None:
        return None

    key = f"{module_key}:{md_path}"
    with _CHECK_RECORDER_LOCK:
        rec = _CHECK_RECORDER_REGISTRY.get(key)
        if rec is None:
            rec = Enhance241CheckRecorder(module_key=module_key, cfg=cfg, exp_dir=exp_dir, md_path=md_path)
            _CHECK_RECORDER_REGISTRY[key] = rec
    if patch_info:
        rec.set_patch_info(patch_info)
    return rec


def _new_debug_key(prefix: str) -> str:
    return f"{prefix}_{next(_DEBUG_KEY_COUNTER)}"


def _bind_module_debug(
    module: torch.nn.Module,
    recorder: Optional[Enhance241CheckRecorder],
    prefix: str,
    baseline: Optional[torch.nn.Module] = None,
) -> None:
    if recorder is None:
        return
    with _DEBUG_BIND_LOCK:
        rec_key = str(getattr(module, "_enhance241_check_key", "")).strip()
        if not rec_key:
            rec_key = _new_debug_key("rec")
            setattr(module, "_enhance241_check_key", rec_key)
        _DEBUG_RECORDER_BY_KEY[rec_key] = recorder
        if baseline is not None:
            base_key = _new_debug_key("base")
            _DEBUG_BASELINE_BY_KEY[base_key] = baseline
            setattr(module, "_enhance241_baseline_key", base_key)
    setattr(module, "_enhance241_prefix", str(prefix))
    if not hasattr(module, "_enhance241_fwd_step"):
        setattr(module, "_enhance241_fwd_step", 0)


def _get_module_recorder(module: torch.nn.Module) -> Optional[Enhance241CheckRecorder]:
    key = str(getattr(module, "_enhance241_check_key", "")).strip()
    if not key:
        return None
    return _DEBUG_RECORDER_BY_KEY.get(key)


def _get_module_baseline(module: torch.nn.Module) -> Optional[torch.nn.Module]:
    key = str(getattr(module, "_enhance241_baseline_key", "")).strip()
    if not key:
        return None
    return _DEBUG_BASELINE_BY_KEY.get(key)


class _DWSeparableConv(torch.nn.Module):
    """Depthwise-separable 3x3 + pointwise 1x1 (no BN)."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.dw = torch.nn.Conv2d(
            in_ch, in_ch, kernel_size=3, stride=1, padding=1, groups=in_ch, bias=True
        )
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
        if h % self.block != 0 or w % self.block != 0:
            raise ValueError(f"SpaceToDepth expects H/W divisible by {self.block}, got: {x.shape}")
        bh = h // self.block
        bw = w // self.block
        x = x.view(b, c, bh, self.block, bw, self.block)
        x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
        return x.view(b, c * (self.block ** 2), bh, bw)


class SPDConvDownsample(torch.nn.Module):
    """Residual-safe SPD downsample: out = base_downsample(x) + alpha * spd_branch(x)."""

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
        self.alpha_cap = float(max(1e-6, abs(alpha_cap)))
        refine = str(refine).lower()

        if self.out_ch > 0 and self.out_ch % self.pre_div == 0:
            pre_ch = self.out_ch // self.pre_div
        else:
            pre_ch = self.out_ch

        if refine == "conv":
            self.enhance241_a4_pre = _Conv3x3(self.in_ch, pre_ch)  # enhance241-audit
        else:
            self.enhance241_a4_pre = _DWSeparableConv(self.in_ch, pre_ch)  # enhance241-audit

        self.enhance241_a4_s2d = _SpaceToDepth(2)  # enhance241-audit

        post_in = pre_ch * 4
        self.enhance241_a4_post = torch.nn.Conv2d(
            post_in, self.out_ch, kernel_size=1, stride=1, padding=0, bias=True
        )  # enhance241-audit
        # Safe-start branch: zero-init guarantees y_spd==0 at step0.
        torch.nn.init.zeros_(self.enhance241_a4_post.weight)
        if self.enhance241_a4_post.bias is not None:
            torch.nn.init.zeros_(self.enhance241_a4_post.bias)

        # Use bounded alpha = alpha_cap * tanh(alpha_raw): keeps residual stable while allowing non-zero init.
        alpha_init = float(max(-self.alpha_cap * 0.95, min(self.alpha_cap * 0.95, alpha_init)))
        alpha_ratio = alpha_init / self.alpha_cap
        alpha_raw = torch.atanh(torch.tensor(alpha_ratio, dtype=torch.float32))
        self.enhance241_a4_alpha = torch.nn.Parameter(alpha_raw)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        recorder = _get_module_recorder(self)
        prefix = str(getattr(self, "_enhance241_prefix", "a4"))
        step = int(getattr(self, "_enhance241_fwd_step", 0))

        if recorder is not None and step == 0:
            recorder.capture_param_before(self, prefix)

        x_in = x
        y_base = self.enhance241_a4_base(x)
        y_spd = self.enhance241_a4_pre(x)
        y_spd = self.enhance241_a4_s2d(y_spd)
        y_spd = self.enhance241_a4_post(y_spd)

        if y_spd.shape[-2:] != y_base.shape[-2:]:
            y_spd = torch.nn.functional.interpolate(y_spd, size=y_base.shape[-2:], mode="nearest")

        alpha_raw = self.enhance241_a4_alpha.to(dtype=y_base.dtype, device=y_base.device)
        alpha = torch.tanh(alpha_raw) * self.alpha_cap
        delta = alpha * y_spd
        out = y_base + delta

        if recorder is not None:
            if step == 0:
                recorder.record_module_compare(f"{prefix}.p3_down", patched=out, baseline=y_base, input_tensor=x_in)
                v_base = _to_float(y_base.detach().float().var(unbiased=False).item())
                v_delta = _to_float(delta.detach().float().var(unbiased=False).item())
                cos = _to_float(
                    F.cosine_similarity(
                        out.detach().float().reshape(out.shape[0], -1),
                        y_base.detach().float().reshape(y_base.shape[0], -1),
                        dim=1,
                    ).mean().item()
                )
                recorder.record_a1_payload(
                    f"{prefix}.residual_safe",
                    {
                        "alpha_raw": _to_float(alpha_raw.item()),
                        "alpha": _to_float(alpha.item()),
                        "alpha_cap": _to_float(self.alpha_cap),
                        "var_ratio_alpha_spd_vs_base": v_delta / (v_base + 1e-12),
                        "target_var_ratio": [0.01, 0.3],
                        "y_base": _tensor_stats(y_base),
                        "alpha_y_spd": _tensor_stats(delta),
                        "out": _tensor_stats(out),
                        "out_vs_base_cosine": cos,
                    },
                )
                recorder.record_scalar_curve(f"{prefix}.alpha_raw", _to_float(alpha_raw.item()), step, max_steps=30)
                recorder.record_scalar_curve(f"{prefix}.alpha", _to_float(alpha.item()), step, max_steps=30)
                if out.requires_grad:
                    try:
                        out.register_hook(lambda g, n=f"{prefix}.p3_down": recorder.record_output_grad(n, g))
                    except Exception:
                        pass
            elif step <= 30:
                recorder.record_scalar_curve(f"{prefix}.alpha_raw", _to_float(alpha_raw.item()), step, max_steps=30)
                recorder.record_scalar_curve(f"{prefix}.alpha", _to_float(alpha.item()), step, max_steps=30)
            if _should_capture_delta(step):
                recorder.capture_param_delta(self, prefix)

        setattr(self, "_enhance241_fwd_step", step + 1)
        return out


class A4DualDeltaSafe(SPDConvDownsample):
    """Compatibility wrapper for modules that still import the old a4 class name."""

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


def apply(model: Any, cfg: Any) -> Any:
    """Apply enhance241 a4 SPDConvDownsample at the P3 downsample point."""

    enable_a4 = bool(_deep_get(cfg, "enhance241", "a4", default=False))
    if not enable_a4:
        return model

    yolo_obj = model
    if hasattr(model, "model") and hasattr(getattr(model, "model"), "model"):
        det_model = getattr(model, "model")
        seq = getattr(det_model, "model", None)
    else:
        det_model = model
        seq = getattr(det_model, "model", None)
    if seq is None:
        raise RuntimeError("enhance241.a4 requires an ultralytics YOLO/DetectionModel-like object with a .model.")

    prepatched = [i for i, layer in enumerate(seq) if isinstance(layer, SPDConvDownsample)]
    if prepatched:
        info = {
            "enabled": True,
            "patched_count": 0,
            "existing_count": len(prepatched),
            "existing_indices": prepatched,
            "replaced_count": 0,
        }
        setattr(yolo_obj, "_enhance241_a4_info", info)

        recorder = get_check_recorder("a4", cfg, patch_info=info)
        if recorder is not None:
            try:
                idx0 = int(prepatched[0])
                mod0 = seq[idx0]
                _bind_module_debug(mod0, recorder, prefix=f"a4.idx{idx0}.existing")
                recorder.register_module_params(mod0, f"a4.idx{idx0}.existing")
                detect_idx, detect_layer = _locate_detect(seq)
                recorder.attach_detect_hooks(detect_layer, detect_idx)
                recorder.maybe_run_val_separability(yolo_obj, cfg)
            except Exception as exc:
                recorder.add_note(f"a4_prepatched_hook_failed:{exc}")

        if len(prepatched) == 1:
            return yolo_obj
        raise RuntimeError(f"enhance241.a4 expects exactly one SPDConvDownsample, got {len(prepatched)}")

    _, p3_idx = _locate_p4_to_p3_fuse(seq)
    ds_idx: Optional[int] = None
    for i in range(p3_idx, -1, -1):
        if _module_stride(seq[i]) == 2:
            ds_idx = i
            break
    if ds_idx is None:
        raise RuntimeError("Unable to locate stride=2 downsample before P3 output.")

    old = seq[ds_idx]
    if isinstance(old, SPDConvDownsample):
        return yolo_obj

    stride = _module_stride(old)
    if stride != 2:
        raise RuntimeError(f"Expected stride=2 at idx={ds_idx}, got stride={stride}")

    conv = _find_stride_conv(old, stride=2) or _find_first_conv(old)
    if conv is None:
        raise RuntimeError(f"Unable to infer in/out channels for layer idx={ds_idx} ({old.__class__.__name__}).")

    in_ch = int(conv.in_channels)
    out_ch = int(conv.out_channels)
    pre_div = _safe_int(_deep_get(cfg, "enhance241", "a4_pre_div", default=4), 4)
    refine = str(_deep_get(cfg, "enhance241", "a4_refine", default="dw"))
    alpha_init = _safe_float(
        _deep_get(cfg, "enhance241", "a4_alpha_init", default=_deep_get(cfg, "enhance241", "a4_alpha1_init", default=0.05)),
        0.05,
    )
    alpha_cap = _safe_float(
        _deep_get(cfg, "enhance241", "a4_alpha_cap", default=_deep_get(cfg, "enhance241", "a4_alpha1_cap", default=0.5)),
        0.5,
    )

    fuse = SPDConvDownsample(
        base_downsample=old,
        in_ch=in_ch,
        out_ch=out_ch,
        pre_div=pre_div,
        refine=refine,
        alpha_init=alpha_init,
        alpha_cap=alpha_cap,
    )
    for attr in ("i", "f", "type"):
        if hasattr(old, attr):
            setattr(fuse, attr, getattr(old, attr))

    device, dtype = _infer_device_dtype(seq, ds_idx)
    if device is not None:
        if dtype is not None:
            fuse = fuse.to(device=device, dtype=dtype)
        else:
            fuse = fuse.to(device=device)

    old_params = sum(int(p.numel()) for p in old.parameters())
    new_params = sum(int(p.numel()) for p in fuse.parameters())
    seq[ds_idx] = fuse

    info = {
        "enabled": True,
        "patched_count": 1,
        "replaced_count": 1,
        "existing_count": 0,
        "replaced_idx": ds_idx,
        "p3_index": p3_idx,
        "orig_type": old.__class__.__name__,
        "orig_stride": stride,
        "new_type": "SPDConvDownsample",
        "params_old": int(old_params),
        "params_new": int(new_params),
        "delta_params": int(new_params - old_params),
        "pre_div": int(pre_div),
        "refine": str(refine),
        "mode": "residual_safe",
        "alpha_init": _to_float(alpha_init),
        "alpha_cap": _to_float(alpha_cap),
    }
    setattr(yolo_obj, "_enhance241_a4_info", info)

    recorder = get_check_recorder("a4", cfg, patch_info=info)
    if recorder is not None:
        _bind_module_debug(fuse, recorder, prefix=f"a4.idx{ds_idx}")
        recorder.register_module_params(fuse, f"a4.idx{ds_idx}")

        try:
            detect_idx, detect_layer = _locate_detect(seq)
            recorder.attach_detect_hooks(detect_layer, detect_idx)
        except Exception as exc:
            recorder.add_note(f"a4_detect_hook_failed:{exc}")

        try:
            recorder.maybe_run_val_separability(yolo_obj, cfg)
        except Exception as exc:
            recorder.add_note(f"a4_val_check_failed:{exc}")

    return yolo_obj
