from __future__ import annotations

# Path: third_party/yolo11/enhance241/yolo11_241a3.py
# Purpose: enhance241 a3 (SPDConvDownsample) module + apply hook + debug checks.

import atexit
import datetime as dt
import json
import math
import os
import subprocess
import threading
from itertools import count
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

ENHANCE241_AUDIT_KEYS = ["enhance241_a3"]  # enhance241-audit

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


def _should_capture_delta(step: int) -> bool:
    """Sparse sampling schedule that remains valid with gradient accumulation."""
    return int(step) in {1, 2, 4, 8, 12, 16, 24, 32, 48, 64}


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
    explicit_exp = str(_deep_get(cfg, "enhance241_exp_dir", default="")).strip() or os.environ.get("ENHANCE241_EXP_DIR", "").strip()
    if explicit_exp:
        p = Path(explicit_exp).expanduser()
        if not p.is_absolute():
            p = (_resolve_project_root(cfg) / p).resolve()
        return p.resolve()

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


def maybe_open_md(cfg: Any, exp_dir: Optional[Path], module_key: str = "a3") -> Optional[Path]:
    _ = cfg
    if exp_dir is None:
        return None
    try:
        exp_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        return None
    return exp_dir / f"enhance241_check_{module_key}.md"


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


def _resolve_split_source(cfg: Any, split_key: str) -> Optional[Path]:
    data_yaml = _resolve_data_yaml(cfg)
    if data_yaml is None:
        return None

    try:
        import yaml  # type: ignore

        with data_yaml.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except Exception:
        return None

    split_ref = data.get(split_key, "")
    if isinstance(split_ref, (list, tuple)):
        split_ref = split_ref[0] if split_ref else ""
    split_ref = str(split_ref).strip()
    if not split_ref:
        return None

    data_root_override = str(_deep_get(cfg, "data_root", default="")).strip()
    root_ref = str(data.get("path", "")).strip()

    candidates: List[Path] = []
    p = Path(split_ref).expanduser()
    if p.is_absolute():
        candidates.append(p)
    if data_root_override:
        candidates.append(Path(data_root_override).expanduser() / split_ref)
    if root_ref:
        root = Path(root_ref).expanduser()
        if not root.is_absolute():
            root = (data_yaml.parent / root).resolve()
        candidates.append(root / split_ref)
    candidates.append((data_yaml.parent / split_ref).resolve())
    candidates.append((_resolve_project_root(cfg) / split_ref).resolve())

    for c in candidates:
        if c.exists():
            return c
    return candidates[0] if candidates else None


def _collect_batch_context(cfg: Any) -> Dict[str, Any]:
    batch = max(1, _safe_int(_deep_get(cfg, "batch", default=1), 1))
    workers = max(0, _safe_int(_deep_get(cfg, "workers", default=8), 8))
    epochs = max(1, _safe_int(_deep_get(cfg, "epochs", default=1), 1))
    seed = _safe_int(_deep_get(cfg, "seed", default=0), 0)
    lr0 = _safe_float(_deep_get(cfg, "lr0", default=0.01), 0.01)
    lrf = _safe_float(_deep_get(cfg, "lrf", default=0.1), 0.1)
    warmup_epochs = _safe_float(_deep_get(cfg, "warmup_epochs", default=3.0), 3.0)
    nbs = max(1, _safe_int(_deep_get(cfg, "nbs", default=64), 64))
    accumulate_cfg = _safe_int(_deep_get(cfg, "accumulate", default=0), 0)
    accumulate = accumulate_cfg if accumulate_cfg > 0 else max(1, int(round(float(nbs) / float(batch))))
    effective_batch = int(batch * accumulate)

    train_source = _resolve_split_source(cfg, "train")
    n_train = 0
    if train_source is not None:
        try:
            n_train = int(len(_list_source_images(train_source)))
        except Exception:
            n_train = 0
    steps_per_epoch = int(math.ceil(float(n_train) / float(batch))) if n_train > 0 else 0
    total_optimizer_steps = int(math.ceil(float(steps_per_epoch) / float(accumulate)) * int(epochs)) if steps_per_epoch > 0 else 0

    return {
        "batch": int(batch),
        "workers": int(workers),
        "epochs": int(epochs),
        "seed": int(seed),
        "lr0": float(lr0),
        "lrf": float(lrf),
        "warmup_epochs": float(warmup_epochs),
        "nbs": int(nbs),
        "accumulate": int(accumulate),
        "effective_batch": int(effective_batch),
        "n_train": int(n_train),
        "steps_per_epoch": int(steps_per_epoch),
        "total_optimizer_steps": int(total_optimizer_steps),
        "train_source": str(train_source) if train_source is not None else "",
        "bn_frozen": bool(_deep_get(cfg, "freeze_bn", default=False)),
        "note": "total_optimizer_steps mismatch => cross-run best/ROC is not directly comparable",
    }


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
        self.batch_context = _collect_batch_context(cfg)

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
        self.a3: Dict[str, Any] = {}
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
        delta: Dict[str, float] = {}
        for name, p in module.named_parameters():
            full = f"{prefix}.{name}"
            before = self._param_before.get(full)
            if before is None:
                continue
            try:
                now = p.detach().float().cpu()
                d = now - before
                cur = _to_float(torch.linalg.vector_norm(d).item())
                prev = _to_float(self.a2["delta_l2"].get(full, 0.0), 0.0)
                if abs(cur) >= abs(prev):
                    delta[full] = cur
            except Exception:
                continue
        with self.lock:
            self.a2["delta_l2"].update(delta)

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
                self.add_note("Gate-3 skipped: model has no predict().")
                return
            source = _resolve_val_source(cfg)
            if source is None:
                self.add_note("Gate-3 skipped: val source unresolved.")
                return

            imgs = _list_source_images(source)
            if not imgs:
                self.add_note(f"Gate-3 skipped: no images under {source}")
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
                self.a3 = {
                    "source": str(source),
                    "sample_count": int(len(imgs)),
                    "metric_conf": float(metric_conf),
                    "nms_iou": float(nms_iou),
                    "max_det": int(max_det),
                    "pos_count": int(len(pos_scores)),
                    "neg_count": int(len(neg_scores)),
                    "pos_max_conf": _vector_summary(pos_tensor),
                    "neg_max_conf": _vector_summary(neg_tensor),
                    "sample_rows": sample_rows,
                }
        except Exception as exc:
            self.add_note(f"Gate-3 predict failed: {exc}")

    def _build_gate_summary(self) -> Dict[str, Any]:
        params = self.a2.get("params", {}) if isinstance(self.a2, dict) else {}
        grad_map = self.a2.get("grad_l2", {}) if isinstance(self.a2, dict) else {}
        delta_map = self.a2.get("delta_l2", {}) if isinstance(self.a2, dict) else {}

        trainable = [v for v in params.values() if isinstance(v, dict) and bool(v.get("requires_grad", False))]
        trainable_numel = int(sum(int(v.get("numel", 0)) for v in trainable))
        gate0_ok = bool(trainable_numel > 0)

        cos_vals: List[float] = []
        var_vals: List[float] = []
        has_nan_inf = False
        for item in self.a1.values():
            if not isinstance(item, dict):
                continue
            patched = item.get("patched", {})
            if isinstance(patched, dict):
                has_nan_inf = has_nan_inf or int(patched.get("nan_count", 0)) > 0 or int(patched.get("inf_count", 0)) > 0
            comp = item.get("patched_vs_baseline", {})
            if isinstance(comp, dict):
                cos_vals.append(_to_float(comp.get("cosine_mean", 0.0), 0.0))
                var_vals.append(_to_float(comp.get("var_ratio", 1.0), 1.0))

        gate1_has_compare = bool(cos_vals and var_vals)
        gate1_cos_min = min(cos_vals) if cos_vals else 0.0
        gate1_var_min = min(var_vals) if var_vals else 0.0
        gate1_ok = bool(
            (not has_nan_inf)
            and gate1_has_compare
            and gate1_cos_min >= 0.98
            and gate1_var_min >= 0.5
        )

        grad_vals = [_to_float(v, 0.0) for v in grad_map.values()]
        delta_vals = [_to_float(v, 0.0) for v in delta_map.values()]
        grad_finite = all(math.isfinite(v) for v in grad_vals) if grad_vals else False
        delta_finite = all(math.isfinite(v) for v in delta_vals) if delta_vals else False
        grad_nonzero = int(sum(1 for v in grad_vals if abs(v) > 0.0))
        delta_nonzero = int(sum(1 for v in delta_vals if abs(v) > 0.0))
        gate2_ok = bool(grad_finite and delta_finite and grad_nonzero > 0 and delta_nonzero > 0)

        pos = self.a3.get("pos_max_conf", {}) if isinstance(self.a3, dict) else {}
        neg = self.a3.get("neg_max_conf", {}) if isinstance(self.a3, dict) else {}
        pos_count = int(pos.get("count", 0)) if isinstance(pos, dict) else 0
        neg_count = int(neg.get("count", 0)) if isinstance(neg, dict) else 0
        pos_p90 = _to_float(pos.get("p90", 0.0), 0.0) if isinstance(pos, dict) else 0.0
        neg_p90 = _to_float(neg.get("p90", 0.0), 0.0) if isinstance(neg, dict) else 0.0
        gate3_eval = bool(pos_count > 0 and neg_count > 0)
        gate3_ok = bool(gate3_eval and pos_p90 > neg_p90)

        return {
            "gate0_param_chain": {
                "pass": gate0_ok,
                "param_total": int(len(params)),
                "trainable_param_count": int(len(trainable)),
                "trainable_numel": int(trainable_numel),
                "optimizer_membership_note": "Direct optimizer inspection unavailable in module patch; grad/delta acts as runtime evidence.",
            },
            "gate1_step0_numeric": {
                "pass": gate1_ok,
                "has_step0_compare": bool(gate1_has_compare),
                "cosine_min": float(gate1_cos_min),
                "var_ratio_min": float(gate1_var_min),
                "thresholds": {"cosine_min": 0.98, "var_ratio_min": 0.5},
                "nan_or_inf_seen": bool(has_nan_inf),
            },
            "gate2_learnability": {
                "pass": gate2_ok,
                "grad_count": int(len(grad_vals)),
                "delta_count": int(len(delta_vals)),
                "grad_nonzero_count": int(grad_nonzero),
                "delta_nonzero_count": int(delta_nonzero),
                "grad_finite": bool(grad_finite),
                "delta_finite": bool(delta_finite),
                "delta_sampling_steps": [1, 2, 4, 8, 12, 16, 24, 32, 48, 64],
            },
            "gate3_pos_neg_separability": {
                "evaluated": bool(gate3_eval),
                "pass": bool(gate3_ok),
                "pos_p90": float(pos_p90),
                "neg_p90": float(neg_p90),
                "condition": "pos_p90 > neg_p90",
            },
        }

    def _judge_conclusion(self, gates: Dict[str, Any]) -> Tuple[str, List[str]]:
        reasons: List[str] = []

        gate0 = gates.get("gate0_param_chain", {})
        gate1 = gates.get("gate1_step0_numeric", {})
        gate2 = gates.get("gate2_learnability", {})
        gate3 = gates.get("gate3_pos_neg_separability", {})

        if not bool(gate0.get("pass", False)):
            reasons.append("Gate-0 failed: no trainable module parameter detected.")
        if not bool(gate1.get("pass", False)):
            reasons.append("Gate-1 failed: step0 health check did not satisfy cosine/variance threshold.")
        if not bool(gate2.get("pass", False)):
            reasons.append("Gate-2 failed: gradients/updates are missing or zero in early training steps.")
        if bool(gate3.get("evaluated", False)) and not bool(gate3.get("pass", False)):
            reasons.append("Gate-3 failed: positive/negative max_conf is not separable (pos p90 <= neg p90).")
        if not bool(gate3.get("evaluated", False)):
            reasons.append("Gate-3 not evaluated: val probe set unavailable, check data path.")

        if not reasons:
            reasons.append("All gates pass on current run; if KPI still weak, continue hyper-parameter tuning.")
            reasons.append("Next action: tune lr0/warmup/lrf under fixed total_optimizer_steps.")
            return "检查通过", reasons

        if not bool(gate0.get("pass", False)) or not bool(gate2.get("pass", False)):
            reasons.append("Next action: fix optimizer chain / grad flow before discussing model quality.")
            return "实现问题", reasons

        if not bool(gate1.get("pass", False)) or (bool(gate3.get("evaluated", False)) and not bool(gate3.get("pass", False))):
            if self.module_key == "a3":
                reasons.append(
                    "Next action: keep residual-safe a3 and verify alpha learns from near-zero without destabilizing step0."
                )
            elif self.module_key == "b3":
                reasons.append(
                    "Next action: constrain b3 fusion weights and keep refine branch zero-init for safe startup."
                )
            elif self.module_key == "c7":
                reasons.append(
                    "Next action: check c7 mode/reduction/n settings and ensure gate branch keeps detail on positives."
                )
            else:
                reasons.append("Next action: apply safe residual injection and rerun >=10 epochs.")
            return "结构问题", reasons

        reasons.append("Next action: lock structure and tune lr0/warmup/lrf with fixed optimizer steps.")
        return "超参问题", reasons

    def flush(self) -> None:
        with self.lock:
            if self._flushed:
                return
            self._flushed = True

        gates = self._build_gate_summary()
        conclusion, reasons = self._judge_conclusion(gates)

        enabled_flags = []
        enh = _deep_get(self.cfg, "enhance241", default={}) or {}
        if isinstance(enh, dict):
            enabled_flags = [k for k, v in enh.items() if isinstance(v, bool) and v]

        lines: List[str] = []
        lines.append(f"\n## Run {dt.datetime.now().isoformat(timespec='seconds')}")
        lines.append(f"- module: `{self.module_key}`")
        lines.append(f"- exp_dir: `{self.exp_dir}`")
        lines.append(f"- git_hash: `{self.git}`")
        lines.append(f"- enable_flags: `{enabled_flags}`")
        lines.append("- stage_note: `Gate-0/1/2/3 are check stages, not module IDs (a2/a3 etc).`")
        lines.append("### Run Metadata")
        lines.append("```json")
        lines.append(json.dumps(self.batch_context, ensure_ascii=False, indent=2))
        lines.append("```")
        lines.append("### Gate Summary")
        lines.append("```json")
        lines.append(json.dumps(gates, ensure_ascii=False, indent=2))
        lines.append("```")
        lines.append("- patch_info:")
        lines.append("```json")
        lines.append(json.dumps(self.patch_info, ensure_ascii=False, indent=2))
        lines.append("```")

        lines.append("### Gate-1 Step0 Stats")
        lines.append("```json")
        lines.append(json.dumps(self.a1, ensure_ascii=False, indent=2))
        lines.append("```")

        lines.append("### Gate-0/2 Params Grad Delta")
        lines.append("```json")
        lines.append(json.dumps(self.a2, ensure_ascii=False, indent=2))
        lines.append("```")

        lines.append("### Gate-3 Pos Neg MaxConf")
        lines.append("```json")
        lines.append(json.dumps(self.a3, ensure_ascii=False, indent=2))
        lines.append("```")

        if self.b_specific:
            lines.append("### B-Specific")
            lines.append("```json")
            lines.append(json.dumps(self.b_specific, ensure_ascii=False, indent=2))
            lines.append("```")

        if self.notes:
            lines.append("### Notes")
            for n in self.notes:
                lines.append(f"- {n}")

        lines.append(f"### 结论: {conclusion}")
        for r in reasons:
            lines.append(f"- {r}")

        try:
            self.md_path.parent.mkdir(parents=True, exist_ok=True)
            with self.md_path.open("a", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
            merged_md = self.exp_dir / "train" / "enhance241_check.md"
            merged_md.parent.mkdir(parents=True, exist_ok=True)
            with merged_md.open("a", encoding="utf-8") as f:
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
    ) -> None:
        super().__init__()
        self.enhance241_a3_base = base_downsample
        self.in_ch = int(in_ch)
        self.out_ch = int(out_ch)
        self.pre_div = int(max(1, pre_div))
        refine = str(refine).lower()

        if self.out_ch > 0 and self.out_ch % self.pre_div == 0:
            pre_ch = self.out_ch // self.pre_div
        else:
            pre_ch = self.out_ch

        if refine == "conv":
            self.enhance241_a3_pre = _Conv3x3(self.in_ch, pre_ch)  # enhance241-audit
        else:
            self.enhance241_a3_pre = _DWSeparableConv(self.in_ch, pre_ch)  # enhance241-audit

        self.enhance241_a3_s2d = _SpaceToDepth(2)  # enhance241-audit

        post_in = pre_ch * 4
        self.enhance241_a3_post = torch.nn.Conv2d(
            post_in, self.out_ch, kernel_size=1, stride=1, padding=0, bias=True
        )  # enhance241-audit
        # Safe-start branch: zero-init guarantees y_spd==0 at step0.
        torch.nn.init.zeros_(self.enhance241_a3_post.weight)
        if self.enhance241_a3_post.bias is not None:
            torch.nn.init.zeros_(self.enhance241_a3_post.bias)

        self.enhance241_a3_alpha = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        recorder = _get_module_recorder(self)
        prefix = str(getattr(self, "_enhance241_prefix", "a3"))
        step = int(getattr(self, "_enhance241_fwd_step", 0))

        if recorder is not None and step == 0:
            recorder.capture_param_before(self, prefix)

        x_in = x
        y_base = self.enhance241_a3_base(x)
        y_spd = self.enhance241_a3_pre(x)
        y_spd = self.enhance241_a3_s2d(y_spd)
        y_spd = self.enhance241_a3_post(y_spd)

        if y_spd.shape[-2:] != y_base.shape[-2:]:
            y_spd = torch.nn.functional.interpolate(y_spd, size=y_base.shape[-2:], mode="nearest")

        alpha = self.enhance241_a3_alpha.to(dtype=y_base.dtype, device=y_base.device)
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
                        "alpha": _to_float(alpha.item()),
                        "var_ratio_alpha_spd_vs_base": v_delta / (v_base + 1e-12),
                        "target_var_ratio": [0.01, 0.3],
                        "y_base": _tensor_stats(y_base),
                        "alpha_y_spd": _tensor_stats(delta),
                        "out": _tensor_stats(out),
                        "out_vs_base_cosine": cos,
                    },
                )
                recorder.record_scalar_curve(f"{prefix}.alpha", _to_float(alpha.item()), step, max_steps=30)
                if out.requires_grad:
                    try:
                        out.register_hook(lambda g, n=f"{prefix}.p3_down": recorder.record_output_grad(n, g))
                    except Exception:
                        pass
            elif step <= 30:
                recorder.record_scalar_curve(f"{prefix}.alpha", _to_float(alpha.item()), step, max_steps=30)
            if _should_capture_delta(step):
                recorder.capture_param_delta(self, prefix)

        setattr(self, "_enhance241_fwd_step", step + 1)
        return out


def apply(model: Any, cfg: Any) -> Any:
    """Apply enhance241 a3 SPDConvDownsample at the P3 downsample point."""

    enable_a3 = bool(_deep_get(cfg, "enhance241", "a3", default=False))
    if not enable_a3:
        return model

    yolo_obj = model
    if hasattr(model, "model") and hasattr(getattr(model, "model"), "model"):
        det_model = getattr(model, "model")
        seq = getattr(det_model, "model", None)
    else:
        det_model = model
        seq = getattr(det_model, "model", None)
    if seq is None:
        raise RuntimeError("enhance241.a3 requires an ultralytics YOLO/DetectionModel-like object with a .model.")

    prepatched = [i for i, layer in enumerate(seq) if isinstance(layer, SPDConvDownsample)]
    if prepatched:
        info = {
            "enabled": True,
            "existing_count": len(prepatched),
            "existing_indices": prepatched,
            "replaced_count": 0,
        }
        setattr(yolo_obj, "_enhance241_a3_info", info)

        recorder = get_check_recorder("a3", cfg, patch_info=info)
        if recorder is not None:
            try:
                idx0 = int(prepatched[0])
                mod0 = seq[idx0]
                _bind_module_debug(mod0, recorder, prefix=f"a3.idx{idx0}.existing")
                recorder.register_module_params(mod0, f"a3.idx{idx0}.existing")
                detect_idx, detect_layer = _locate_detect(seq)
                recorder.attach_detect_hooks(detect_layer, detect_idx)
                recorder.maybe_run_val_separability(yolo_obj, cfg)
            except Exception as exc:
                recorder.add_note(f"a3_prepatched_hook_failed:{exc}")

        if len(prepatched) == 1:
            return yolo_obj
        raise RuntimeError(f"enhance241.a3 expects exactly one SPDConvDownsample, got {len(prepatched)}")

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
    pre_div = _safe_int(_deep_get(cfg, "enhance241", "a3_pre_div", default=4), 4)
    refine = str(_deep_get(cfg, "enhance241", "a3_refine", default="dw"))

    fuse = SPDConvDownsample(base_downsample=old, in_ch=in_ch, out_ch=out_ch, pre_div=pre_div, refine=refine)
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
        "alpha_init": 0.0,
    }
    setattr(yolo_obj, "_enhance241_a3_info", info)

    recorder = get_check_recorder("a3", cfg, patch_info=info)
    if recorder is not None:
        _bind_module_debug(fuse, recorder, prefix=f"a3.idx{ds_idx}")
        recorder.register_module_params(fuse, f"a3.idx{ds_idx}")

        try:
            detect_idx, detect_layer = _locate_detect(seq)
            recorder.attach_detect_hooks(detect_layer, detect_idx)
        except Exception as exc:
            recorder.add_note(f"a3_detect_hook_failed:{exc}")

        try:
            recorder.maybe_run_val_separability(yolo_obj, cfg)
        except Exception as exc:
            recorder.add_note(f"a3_val_check_failed:{exc}")

    return yolo_obj
