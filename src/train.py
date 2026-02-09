import locale
import os


def _ensure_utf8_locale() -> None:
    """Avoid UnicodeDecodeError in subprocess(text=True) under ASCII locales.

    NumPy may call `lscpu` (via numpy.testing) while importing SciPy, and if Python's
    preferred encoding is ASCII but `lscpu` output contains non-ASCII characters,
    decoding can crash training at the plotting stage.
    """

    try:
        enc = (locale.getpreferredencoding(False) or "").lower()
        if enc in {"ascii", "ansi_x3.4-1968"}:
            os.environ.setdefault("LANG", "C.UTF-8")
            os.environ.setdefault("LC_ALL", "C.UTF-8")
            locale.setlocale(locale.LC_ALL, "")
    except Exception:
        pass


_ensure_utf8_locale()

import argparse
import atexit
import datetime as dt
import gc
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from data.make_split import create_val_split
from eval import (
    compute_ap,
    compute_auc,
    compute_image_scores,
    compute_image_scores_iou,
    compute_image_level_results,
    compute_threshold_metrics,
    save_image_level_report,
    save_metric_plots,
    save_multi_curve,
    save_multi_curve_grouped,
    save_multi_xy_curves,
    save_multi_xy_curves_grouped,
    save_threshold_table,
    save_threshold_table_multi,
)


def load_yaml(path: Path) -> Dict[str, Any]:
    import yaml  # type: ignore

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML must be a mapping: {path}")
    return data


def resolve_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_data_root(data_yaml: Path) -> Path:
    info = load_yaml(data_yaml)
    root = info.get("path")
    if root:
        return Path(root).resolve()
    return data_yaml.parent.resolve()


def resolve_data_entry(entry: str, data_root: Path) -> Path:
    path = Path(entry)
    if path.is_absolute():
        return path
    return (data_root / path).resolve()


def resolve_pretrained_weight(project_root: Path, weight_name: str) -> Path:
    return (project_root / "models" / "pretrained" / weight_name).resolve()


def normalize_weight_ref(project_root: Path, weight_ref: str) -> str:
    ref = str(weight_ref or "").strip()
    if not ref:
        return ref
    path = Path(ref)
    if path.is_absolute():
        return str(path)
    # Bare "*.pt" names are always mapped to unified pretrained cache dir.
    if len(path.parts) == 1 and path.suffix.lower() == ".pt":
        return str(resolve_pretrained_weight(project_root, path.name))
    return str((project_root / path).resolve())


def configure_ultralytics_weight_cache(project_root: Path) -> None:
    """Route bare Ultralytics asset downloads (e.g. yolo26n.pt) to models/pretrained."""
    try:
        from ultralytics.utils import SETTINGS
        from ultralytics.utils import downloads as ul_downloads
    except Exception:
        return

    weights_dir = (project_root / "models" / "pretrained").resolve()
    weights_dir.mkdir(parents=True, exist_ok=True)

    legacy_amp_weight = (project_root / "yolo26n.pt").resolve()
    target_amp_weight = (weights_dir / "yolo26n.pt").resolve()
    if legacy_amp_weight.exists() and not target_amp_weight.exists():
        try:
            shutil.copy2(legacy_amp_weight, target_amp_weight)
        except Exception:
            pass

    try:
        SETTINGS.update({"weights_dir": str(weights_dir)})
    except Exception:
        try:
            SETTINGS["weights_dir"] = str(weights_dir)
        except Exception:
            pass

    if getattr(ul_downloads, "_weights_dir_patched", False):
        return

    original_attempt_download_asset = ul_downloads.attempt_download_asset

    def _attempt_download_asset_patched(file, *args, **kwargs):
        file_path = Path(str(file).strip().replace("'", ""))
        if not file_path.is_absolute() and len(file_path.parts) == 1:
            file = str(weights_dir / file_path.name)
        return original_attempt_download_asset(file, *args, **kwargs)

    ul_downloads.attempt_download_asset = _attempt_download_asset_patched
    ul_downloads._weights_dir_patched = True


def update_args_yaml(exp_dir: Path, cfg: Dict[str, Any], cfg_path: Path) -> None:
    args_path = exp_dir / "train" / "args.yaml"
    if not args_path.exists():
        return
    try:
        import yaml  # type: ignore
    except Exception:
        return
    with open(args_path, "r", encoding="utf-8") as f:
        args_data = yaml.safe_load(f) or {}
    if not isinstance(args_data, dict):
        return
    merged = dict(args_data)
    merged.update(cfg)
    merged["config_path"] = str(cfg_path)
    with open(args_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(merged, f, sort_keys=False, allow_unicode=True)


def build_data_yaml(
    data_yaml: Path,
    save_dir: Path,
    val_split: float,
    seed: int,
    data_root_override: str,
) -> Tuple[Path, Path]:
    info = load_yaml(data_yaml)
    if data_root_override:
        data_root = Path(data_root_override).resolve()
    else:
        data_root = resolve_data_root(data_yaml)
    train_entry = info.get("train", "images/train")
    val_entry = info.get("val")
    test_entry = info.get("test")

    train_path = resolve_data_entry(str(train_entry), data_root)
    val_path = resolve_data_entry(str(val_entry), data_root) if val_entry else None
    test_path = resolve_data_entry(str(test_entry), data_root) if test_entry else None

    splits_dir = save_dir / "splits"
    train_txt = None
    val_txt = None
    if (val_path is None or not val_path.exists()) and val_split > 0:
        train_txt, val_txt = create_val_split(train_path, val_split, seed, splits_dir)

    yaml_lines = [f"path: {data_root}"]
    if train_txt and val_txt:
        yaml_lines.append(f"train: {train_txt}")
        yaml_lines.append(f"val: {val_txt}")
    else:
        yaml_lines.append(f"train: {train_entry}")
        yaml_lines.append(f"val: {val_entry or train_entry}")
    if test_entry:
        yaml_lines.append(f"test: {test_entry}")
    yaml_lines.append(f"nc: {info.get('nc', 1)}")
    yaml_lines.append(f"names: {info.get('names', ['object'])}")

    save_dir.mkdir(parents=True, exist_ok=True)
    out_yaml = save_dir / "data.yaml"
    with open(out_yaml, "w", encoding="utf-8") as f:
        f.write("\n".join(yaml_lines))
    return out_yaml, data_root


def copy_weights(exp_dir: Path, model_dir: Path) -> None:
    weights_dir = exp_dir / "train" / "weights"
    best = weights_dir / "best.pt"
    last = weights_dir / "last.pt"
    exp_name = exp_dir.name
    if best.exists():
        dst = model_dir / exp_name / "best"
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best, dst / "best.pt")
    if last.exists():
        dst = model_dir / exp_name / "last"
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copy2(last, dst / "last.pt")


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


def _enhance241_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    enh = cfg.get("enhance241", {})
    return enh if isinstance(enh, dict) else {}


def _enhance241_enabled_keys(cfg: Dict[str, Any]) -> List[str]:
    enh = _enhance241_cfg(cfg)
    keys = ["a3", "b1", "b2", "b3", "d1"]
    return [k for k in keys if bool(enh.get(k, False))]


def _extract_model_seq(model: Any) -> Tuple[Any, Any, Optional[Any]]:
    yolo_obj = model
    if hasattr(model, "model") and hasattr(getattr(model, "model"), "model"):
        det_model = getattr(model, "model")
        seq = getattr(det_model, "model", None)
    else:
        det_model = model
        seq = getattr(det_model, "model", None)
    return yolo_obj, det_model, seq


def _collect_concat_candidates(seq: Any) -> List[str]:
    out: List[str] = []
    for i, layer in enumerate(seq or []):
        if layer.__class__.__name__ == "Concat":
            out.append(f"idx={i}, f={getattr(layer, 'f', None)}")
    return out


def _collect_patched(seq: Any) -> List[str]:
    patched_names = {
        "P3ASFFLiteFuse",
        "P4P3GateAlignFuse",
        "P3FuseChain",
        "SPDConvDownsample",
        "NASFPNLiteFuse",
        "P2LiteFuse",
    }
    out: List[str] = []
    for i, layer in enumerate(seq or []):
        if layer.__class__.__name__ in patched_names:
            out.append(f"idx={i}, type={layer.__class__.__name__}")
    return out


def _collect_keyword_hits(state_dict: Dict[str, Any], keyword: str = "enhance241", sample: int = 8) -> Dict[str, Any]:
    keys = [k for k in state_dict.keys() if keyword in k]
    return {"count": int(len(keys)), "sample_keys": keys[: int(sample)]}


def _snapshot_model(model: Any) -> Dict[str, Any]:
    _, det_model, seq = _extract_model_seq(model)
    try:
        state = det_model.state_dict() if hasattr(det_model, "state_dict") else model.state_dict()
        keyword_hits = _collect_keyword_hits(state)
    except Exception:
        keyword_hits = {"count": 0, "sample_keys": []}
    return {
        "concat_candidates": _collect_concat_candidates(seq),
        "patched": _collect_patched(seq),
        "keyword_hits": keyword_hits,
    }


def _collect_trainable_params(model: Any) -> Dict[str, Any]:
    named = list(model.named_parameters()) if hasattr(model, "named_parameters") else []
    total_numel = sum(int(p.numel()) for _, p in named)
    trainable = [(n, p) for n, p in named if bool(getattr(p, "requires_grad", False))]
    trainable_numel = sum(int(p.numel()) for _, p in trainable)
    matched = [(n, p) for n, p in named if "enhance241" in n]
    return {
        "matched": int(len(matched)),
        "trainable": int(len(trainable)),
        "trainable_numel": int(trainable_numel),
        "total_numel": int(total_numel),
    }


def _collect_git_hash(project_root: Path) -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=project_root, stderr=subprocess.DEVNULL)
        return out.decode("utf-8", errors="ignore").strip()
    except Exception:
        return "unknown"


def _load_ckpt_keyword_hits(path: Optional[Path]) -> Dict[str, Any]:
    if path is None or not path.exists():
        return {"count": 0, "sample_keys": [], "source": "missing"}
    try:
        import torch  # type: ignore
    except Exception:
        return {"count": 0, "sample_keys": [], "source": "torch_unavailable"}
    try:
        ckpt = torch.load(path, map_location="cpu")
    except Exception:
        return {"count": 0, "sample_keys": [], "source": "load_failed"}
    state = None
    if isinstance(ckpt, dict):
        if "model" in ckpt:
            model_obj = ckpt["model"]
            if hasattr(model_obj, "state_dict"):
                state = model_obj.state_dict()
            elif isinstance(model_obj, dict):
                state = model_obj
        elif "state_dict" in ckpt:
            state = ckpt["state_dict"]
        else:
            if all(isinstance(k, str) for k in ckpt.keys()):
                state = ckpt
    if not isinstance(state, dict):
        return {"count": 0, "sample_keys": [], "source": "state_dict_missing"}
    hits = _collect_keyword_hits(state)
    hits["source"] = "state_dict"
    return hits


def _collect_enhance241_infos(model: Any) -> Dict[str, Any]:
    info: Dict[str, Any] = {}
    for key, attr in (
        ("a3", "_enhance241_a3_info"),
        ("b3", "_enhance241_b3_info"),
        ("d1", "_enhance241_d1_info"),
    ):
        val = getattr(model, attr, None)
        if val is None and hasattr(model, "model"):
            val = getattr(model.model, attr, None)
        if val is not None:
            info[key] = val
    return info


def _evaluate_enhance241_checks(audit_state: Dict[str, Any]) -> Tuple[str, List[Dict[str, Any]]]:
    enabled = set(audit_state.get("enhance241_enabled", []))
    info = audit_state.get("eval_infos") or audit_state.get("train_infos") or {}
    checks: List[Dict[str, Any]] = []

    if "a3" in enabled:
        a3 = info.get("a3", {}) if isinstance(info, dict) else {}
        replaced = int(a3.get("replaced_count", 0)) + int(a3.get("existing_count", 0))
        ok = replaced == 1
        checks.append({"name": "a3_replace_count", "ok": ok, "detail": f"count={replaced}"})

    if "b3" in enabled:
        b3 = info.get("b3", {}) if isinstance(info, dict) else {}
        patched = int(b3.get("patched_count", 0))
        ok = patched == 2
        checks.append({"name": "b3_patch_count", "ok": ok, "detail": f"count={patched}"})

    if "d1" in enabled:
        d1 = info.get("d1", {}) if isinstance(info, dict) else {}
        heads_after = int(d1.get("detect_heads_after", 0))
        ok = heads_after == 4
        checks.append({"name": "d1_head_count", "ok": ok, "detail": f"heads_after={heads_after}"})

    status = "PASS" if all(c.get("ok", False) for c in checks) else "FAIL"
    return status, checks


def _format_enhance241_audit(audit_state: Dict[str, Any]) -> str:
    timestamp = audit_state.get("timestamp", dt.datetime.now().isoformat(timespec="seconds"))
    status, checks = _evaluate_enhance241_checks(audit_state)
    audit_state["status"] = status
    audit_state["checks"] = checks

    lines: List[str] = []
    lines.append(f"\n## Run {timestamp}")
    lines.append(f"- cmd: {audit_state.get('cmd', '')}")
    lines.append(f"- git: {audit_state.get('git_hash', '')}")
    lines.append(f"- exp_dir: {audit_state.get('exp_dir', '')}")
    lines.append(f"- config: {audit_state.get('config_path', '')}")
    lines.append(f"- data_yaml: {audit_state.get('data_yaml', '')}")
    lines.append(f"- data_root: {audit_state.get('data_root', '')}")
    lines.append(f"- enhance241_enabled: {audit_state.get('enhance241_enabled', [])}")
    lines.append("- enhance241_cfg:")
    lines.append("```json")
    lines.append(json.dumps(audit_state.get("enhance241_cfg", {}), ensure_ascii=False, indent=2))
    lines.append("```")

    pre = audit_state.get("train_pre_patch", {})
    post = audit_state.get("train_post_patch", {})
    lines.append(f"- pre_patch: concat_candidates={pre.get('concat_candidates', [])}")
    lines.append(f"- pre_patch: patched={pre.get('patched', [])}")
    lines.append(f"- pre_patch: keyword_hits={pre.get('keyword_hits', {})}")
    lines.append(f"- post_patch: concat_candidates={post.get('concat_candidates', [])}")
    lines.append(f"- post_patch: patched={post.get('patched', [])}")
    lines.append(f"- post_patch: keyword_hits={post.get('keyword_hits', {})}")

    if audit_state.get("trainable_params") is not None:
        lines.append(f"- trainable_params: {audit_state.get('trainable_params')}")

    infos = audit_state.get("train_infos") or {}
    if "a3" in infos:
        lines.append(f"- a3_info: {infos.get('a3')}")
    if "b3" in infos:
        lines.append(f"- b3_info: {infos.get('b3')}")
    if "d1" in infos:
        lines.append(f"- d1_info: {infos.get('d1')}")

    lines.append(f"- eval_weights: {audit_state.get('eval_weights', '')}")
    lines.append(f"- ckpt_keyword_hits: {audit_state.get('ckpt_keyword_hits', {})}")
    lines.append(f"- threshold_sweep: {audit_state.get('threshold_sweep', {})}")
    lines.append(f"- checks: {checks}")
    lines.append(f"- conclusion: {status}")
    return "\n".join(lines) + "\n"


def _append_enhance241_audit(audit_path: Path, audit_state: Dict[str, Any]) -> None:
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    block = _format_enhance241_audit(audit_state)
    with open(audit_path, "a", encoding="utf-8") as f:
        f.write(block)


def _register_enhance241_audit(audit_path: Path, audit_state: Dict[str, Any]) -> None:
    def _writer() -> None:
        try:
            eval_w = audit_state.get("eval_weights")
            if eval_w:
                audit_state["ckpt_keyword_hits"] = _load_ckpt_keyword_hits(Path(eval_w))
            _append_enhance241_audit(audit_path, audit_state)
        except Exception:
            pass

    atexit.register(_writer)


def _apply_enhance241_patches(model: Any, cfg: Dict[str, Any], stage: str, audit_state: Dict[str, Any]) -> Any:
    pre = _snapshot_model(model)
    if stage == "train":
        audit_state["train_pre_patch"] = pre
    else:
        audit_state["eval_pre_patch"] = pre

    enabled = _enhance241_enabled_keys(cfg)
    if enabled:
        if "b3" in enabled and ("b1" in enabled or "b2" in enabled):
            raise RuntimeError("enhance241.b3 conflicts with b1/b2; enable only one B-class module.")
        from third_party.yolo11.enhance241 import yolo11_241a3, yolo11_241b1, yolo11_241b2, yolo11_241b3, yolo11_241d1

        model = yolo11_241a3.apply(model, cfg)
        model = yolo11_241b1.apply(model, cfg)
        model = yolo11_241b2.apply(model, cfg)
        model = yolo11_241b3.apply(model, cfg)
        model = yolo11_241d1.apply(model, cfg)

    post = _snapshot_model(model)
    if stage == "train":
        audit_state["train_post_patch"] = post
        audit_state["train_infos"] = _collect_enhance241_infos(model)
        audit_state["trainable_params"] = _collect_trainable_params(model)
    else:
        audit_state["eval_post_patch"] = post
        audit_state["eval_infos"] = _collect_enhance241_infos(model)
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLO training runner.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configs/yolo11/*.yaml",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg_path = Path(args.config).resolve()
    cfg = load_yaml(cfg_path)
    root = resolve_root()
    yolo_version = cfg.get("yolo_version", "yolo11")
    exp_name = cfg.get("exp_name", "defect")
    project_root = Path(cfg.get("project_root", root)).resolve()
    exp_root = project_root / "experiments" / yolo_version / exp_name
    run_name = str(cfg.get("run_name", "")).strip()
    timestamp = dt.datetime.now().strftime("%y%m%d%H%M")
    if run_name:
        exp_dir = exp_root / run_name / f"exp_{timestamp}"
    else:
        exp_dir = exp_root / f"exp_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    data_yaml = Path(cfg.get("data", project_root / "configs" / "data" / "defect.yaml")).resolve()
    data_root_override = str(cfg.get("data_root", ""))
    val_split = float(cfg.get("val_split", 0.0))
    seed = int(cfg.get("seed", 42))
    run_data_yaml, data_root = build_data_yaml(
        data_yaml, exp_dir, val_split, seed, data_root_override
    )

    audit_state: Dict[str, Any] = {
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        "cmd": " ".join(sys.argv),
        "git_hash": _collect_git_hash(project_root),
        "exp_dir": str(exp_dir),
        "config_path": str(cfg_path),
        "data_yaml": str(data_yaml),
        "data_root": str(data_root),
        "enhance241_enabled": _enhance241_enabled_keys(cfg),
        "enhance241_cfg": _enhance241_cfg(cfg),
        "threshold_sweep": cfg.get("threshold_sweep", {}),
        "eval_weights": "",
    }
    _register_enhance241_audit(exp_dir / "train" / "enhance241_audit.md", audit_state)

    model_path = normalize_weight_ref(project_root, str(cfg.get("model", "yolo11n.pt")))
    weights_path = normalize_weight_ref(project_root, str(cfg.get("weights", "")))
    epochs = int(cfg.get("epochs", 50))
    imgsz = int(cfg.get("imgsz", 640))
    batch = int(cfg.get("batch", 16))
    device = str(cfg.get("device", ""))
    workers = int(cfg.get("workers", 8))
    conf = float(cfg.get("conf", 0.25))
    patience = int(cfg.get("patience", 50))
    lr0 = float(cfg.get("lr0", 0.01))
    lrf = float(cfg.get("lrf", 0.1))
    warmup_epochs = float(cfg.get("warmup_epochs", 3.0))
    save_train_pic = bool(cfg.get("save_train_pic", False))
    save_val_pic = bool(cfg.get("save_val_pic", True))
    save_test_pic = bool(cfg.get("save_test_pic", True))
    metric_conf = float(cfg.get("metric_conf", 0.001))
    eval_batch = int(cfg.get("eval_batch", 1))
    eval_device = str(cfg.get("eval_device", device))
    nms_iou = float(cfg.get("nms_iou", cfg.get("iou", 0.7)))
    max_det = int(cfg.get("max_det", 300))
    match_iou = float(cfg.get("match_iou", 0.5))
    image_conf = float(cfg.get("image_conf", conf))

    # Persist a minimal, reproducible run meta for traceability.
    try:
        shutil.copy2(cfg_path, exp_dir / "config.yaml")
    except Exception:
        pass
    try:
        meta = {
            "eval_standard": "P2.3.0",
            "config_path": str(cfg_path),
            "postprocess": {
                "conf": conf,
                "conf_threshold": image_conf,
                "nms_iou": nms_iou,
                "max_det": max_det,
            },
            "eval_iou": match_iou,
            "match_iou": match_iou,
        }
        (exp_dir / "config_dump.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except Exception:
        pass

    # One-line evaluation standard log (IoU=交并比, NMS=非极大值抑制)
    print(
        f"[eval_std:P2.3.0] conf_threshold={image_conf:.2f} eval_iou={match_iou:.2f} (IoU=交并比) "
        f"nms_iou={nms_iou:.2f} (NMS=非极大值抑制) max_det={max_det}"
    )

    from ultralytics import YOLO
    configure_ultralytics_weight_cache(project_root)

    mode = str(cfg.get("mode", "train_test")).lower()
    if mode == "test" and not weights_path:
        raise ValueError("mode=test requires weights in config.")

    if mode == "test":
        init_weights = weights_path
    elif mode == "finetune_test" and weights_path:
        init_weights = weights_path
    else:
        init_weights = model_path

    init_path = Path(init_weights)
    if not init_path.exists() and init_path.suffix == ".pt":
        cache_path = resolve_pretrained_weight(project_root, init_path.name)
        if cache_path.exists():
            init_weights = str(cache_path)
        else:
            print(f"Pretrained weight missing: {init_path}")
            print(f"Downloading to: {cache_path}")
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            init_weights = str(cache_path)
    model = YOLO(init_weights)
    model = _apply_enhance241_patches(model, cfg, stage="train", audit_state=audit_state)
    if mode in {"train_test", "finetune_test"}:
        model.train(
            data=str(run_data_yaml),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=device,
            workers=workers,
            patience=patience,
            lr0=lr0,
            lrf=lrf,
            warmup_epochs=warmup_epochs,
            project=str(exp_dir),
            name="train",
            exist_ok=True,
        )
        update_args_yaml(exp_dir, cfg, cfg_path)

    best_weights = exp_dir / "train" / "weights" / "best.pt"
    if mode in {"train_test", "finetune_test"}:
        eval_weights = best_weights if best_weights.exists() else (exp_dir / "train" / "weights" / "last.pt")
        audit_state["eval_weights"] = str(eval_weights)
        # Release training model before evaluation to reduce GPU memory pressure.
        try:
            del model
        except Exception:
            pass
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        gc.collect()
        model = YOLO(str(eval_weights))
        model = _apply_enhance241_patches(model, cfg, stage="eval", audit_state=audit_state)
        status, checks = _evaluate_enhance241_checks(audit_state)
        if status == "FAIL":
            raise RuntimeError(f"enhance241 structure check failed: {checks}")
    elif mode == "test":
        test_weight = Path(weights_path)
        if test_weight.exists():
            eval_weights = test_weight
        else:
            cached = resolve_pretrained_weight(project_root, test_weight.name)
            eval_weights = cached
        audit_state["eval_weights"] = str(eval_weights)
        model = YOLO(str(eval_weights))
        model = _apply_enhance241_patches(model, cfg, stage="eval", audit_state=audit_state)
        status, checks = _evaluate_enhance241_checks(audit_state)
        if status == "FAIL":
            raise RuntimeError(f"enhance241 structure check failed: {checks}")

    # Resolve split sources across one or multiple dataset roots.
    data_info_cfg = load_yaml(data_yaml)
    train_entry = data_info_cfg.get("train", "images/train")
    val_entry = data_info_cfg.get("val", "images/val") or train_entry
    test_entry = data_info_cfg.get("test", "")

    data_root_cfg = cfg.get("data_root", "")
    data_roots: List[Path] = []
    if isinstance(data_root_cfg, (list, tuple)):
        data_roots = [Path(str(p)).resolve() for p in data_root_cfg if str(p).strip()]
    else:
        data_root_str = str(data_root_cfg or "").strip()
        if data_root_str:
            data_roots = [Path(data_root_str).resolve()]
    if not data_roots:
        data_roots = [data_root]

    val_sources: List[Path] = []
    test_sources: List[Path] = []
    for dr in data_roots:
        val_p = resolve_data_entry(str(val_entry), dr)
        if val_p.exists():
            val_sources.append(val_p)
        if test_entry:
            test_p = resolve_data_entry(str(test_entry), dr)
            if test_p.exists():
                test_sources.append(test_p)


    metrics_dir = exp_dir / "metrics"
    sweep_cfg = cfg.get("threshold_sweep", {})
    fix_var = str(sweep_cfg.get("fix", "iou")).lower()
    fix_list = sweep_cfg.get("fix_list", [0.1, 0.2, 0.3, 0.4])
    curve_range = sweep_cfg.get("curve", [0.01, 0.01, 1.00])
    table_range = sweep_cfg.get("table", [0.00, 0.05, 1.00])
    audit_state["threshold_sweep"] = {
        "fix": fix_var,
        "fix_list": fix_list,
        "curve": curve_range,
        "table": table_range,
    }

    if not isinstance(fix_list, (list, tuple)):
        fix_list = [fix_list]

    curve_vals = np.round(
        np.arange(float(curve_range[0]), float(curve_range[2]) + 1e-9, float(curve_range[1])), 2
    )
    table_vals = np.round(
        np.arange(float(table_range[0]), float(table_range[2]) + 1e-9, float(table_range[1])), 2
    )

    def _concat_scores(scores_list: List[np.ndarray]) -> np.ndarray:
        scores_list = [s for s in scores_list if s.size]
        if not scores_list:
            return np.array([], dtype=np.float32)
        return np.concatenate(scores_list, axis=0)

    def _concat_labels(labels_list: List[np.ndarray]) -> np.ndarray:
        labels_list = [l for l in labels_list if l.size]
        if not labels_list:
            return np.array([], dtype=np.int32)
        return np.concatenate(labels_list, axis=0)

    def _compute_scores_all(fn, sources: List[Path], *args: object) -> Tuple[np.ndarray, np.ndarray]:
        labels_all: List[np.ndarray] = []
        scores_all: List[np.ndarray] = []
        for src in sources:
            try:
                lab, sc = fn(model, src, *args)
            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower():
                    raise
                try:
                    import torch

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass
                gc.collect()
                # CPU fallback for this shard
                if fn is compute_image_scores:
                    lab, sc = fn(model, src, *args[:1], 1, "cpu", *args[3:])
                else:
                    lab, sc = fn(model, src, *args[:2], 1, "cpu", *args[4:])
            labels_all.append(lab)
            scores_all.append(sc)
        return _concat_labels(labels_all), _concat_scores(scores_all)

    def eval_visuals(split: str, sources: List[Path]) -> List[Dict[str, object]]:
        if not sources:
            return []
        vis_root = exp_dir / f"{split}_vis"
        all_items: List[Dict[str, object]] = []
        for src in sources:
            try:
                items = compute_image_level_results(
                    model,
                    src,
                    conf_threshold=image_conf,
                    iou_match=match_iou,
                    metric_conf=metric_conf,
                    batch=eval_batch,
                    device=eval_device,
                    nms_iou=nms_iou,
                    max_det=max_det,
                    split=split,
                    vis_root=vis_root,
                    save_visuals=True,
                )
            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower():
                    raise
                try:
                    import torch

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass
                gc.collect()
                items = compute_image_level_results(
                    model,
                    src,
                    conf_threshold=image_conf,
                    iou_match=match_iou,
                    metric_conf=metric_conf,
                    batch=1,
                    device="cpu",
                    nms_iou=nms_iou,
                    max_det=max_det,
                    split=split,
                    vis_root=vis_root,
                    save_visuals=True,
                )
            all_items.extend(items)
        return all_items

    # 1) Per-split error visualizations (kept separate)
    val_items: List[Dict[str, object]] = []
    test_items: List[Dict[str, object]] = []
    if save_val_pic:
        val_items = eval_visuals("val", val_sources)
    if save_test_pic:
        test_items = eval_visuals("test", test_sources)

    # 2) Combined metrics/curves/tables (val+test merged, and multiple roots merged)
    all_sources = val_sources + test_sources
    if all_sources:
        metrics_dir.mkdir(parents=True, exist_ok=True)
        tag = "eval"

        # Save sweep meta
        try:
            sweep_meta = {"fix": fix_var, "fix_list": fix_list, "curve": curve_range, "table": table_range}
            (metrics_dir / f"{tag}_threshold_sweep.json").write_text(
                json.dumps(sweep_meta, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except Exception:
            pass

        # Base ROC/PR vs confidence threshold (reference)
        labels_base, scores_base = _compute_scores_all(
            compute_image_scores, all_sources, metric_conf, eval_batch, eval_device, nms_iou, max_det
        )
        if labels_base.size:
            recall, precision, fpr = compute_threshold_metrics(labels_base, scores_base, curve_vals)
            auroc = compute_auc(fpr, recall)
            ap = compute_ap(recall, precision)
            save_metric_plots(metrics_dir, tag, curve_vals, recall, precision, fpr)
            with open(metrics_dir / f"{tag}_summary.txt", "w", encoding="utf-8") as f:
                f.write(f"image_auroc: {auroc:.6f}\n")
                f.write(f"image_ap: {ap:.6f}\n")

        multi_rows: List[dict] = []
        curve_recall: List[Tuple[str, np.ndarray]] = []
        curve_fpr: List[Tuple[str, np.ndarray]] = []
        roc_curves: List[Tuple[str, np.ndarray, np.ndarray]] = []
        pr_curves: List[Tuple[str, np.ndarray, np.ndarray]] = []

        group_size = 3
        if fix_var == "iou":
            # x-axis: confidence threshold
            var_curve_vals = curve_vals
            var_table_vals = table_vals
            for iou_thr in fix_list:
                labels_i, scores_i = _compute_scores_all(
                    compute_image_scores_iou,
                    all_sources,
                    metric_conf,
                    float(iou_thr),
                    eval_batch,
                    eval_device,
                    nms_iou,
                    max_det,
                )
                if labels_i.size == 0:
                    continue
                rec_i, prec_i, fpr_i = compute_threshold_metrics(labels_i, scores_i, var_curve_vals)
                curve_recall.append((f"iou={float(iou_thr):.2f}", rec_i))
                curve_fpr.append((f"iou={float(iou_thr):.2f}", fpr_i))
                roc_curves.append((f"iou={float(iou_thr):.2f}", fpr_i, rec_i))
                pr_curves.append((f"iou={float(iou_thr):.2f}", rec_i, prec_i))

                # table (coarser thresholds)
                rec_t, _, fpr_t = compute_threshold_metrics(labels_i, scores_i, var_table_vals)
                for thr, rec_v, fpr_v in zip(var_table_vals, rec_t, fpr_t):
                    multi_rows.append(
                        {
                            "fixed_var": "iou",
                            "fixed_value": float(iou_thr),
                            "threshold": float(thr),
                            "recall": float(rec_v),
                            "fpr": float(fpr_v),
                        }
                    )

            # grouped plots: 3 curves per figure
            save_multi_curve_grouped(
                metrics_dir,
                f"{tag}_recall_vs_conf",
                var_curve_vals,
                curve_recall,
                xlabel="conf threshold",
                ylabel="recall",
                title=f"{tag} Recall vs conf (fixed IoU)",
                group_size=group_size,
                group_label="iougroup",
            )
            save_multi_curve_grouped(
                metrics_dir,
                f"{tag}_fpr_vs_conf",
                var_curve_vals,
                curve_fpr,
                xlabel="conf threshold",
                ylabel="fpr",
                title=f"{tag} FPR vs conf (fixed IoU)",
                group_size=group_size,
                group_label="iougroup",
            )
            if roc_curves:
                save_multi_xy_curves_grouped(
                    metrics_dir,
                    f"{tag}_roc",
                    roc_curves,
                    xlabel="FPR",
                    ylabel="TPR/Recall",
                    title=f"{tag} ROC (fixed IoU)",
                    group_size=group_size,
                    group_label="iougroup",
                )
            if pr_curves:
                save_multi_xy_curves_grouped(
                    metrics_dir,
                    f"{tag}_pr",
                    pr_curves,
                    xlabel="Recall",
                    ylabel="Precision",
                    title=f"{tag} PR (fixed IoU)",
                    group_size=group_size,
                    group_label="iougroup",
                )

        elif fix_var == "conf":
            # x-axis: iou_match threshold
            iou_curve_vals = curve_vals
            iou_table_vals = table_vals
            labels_ref: np.ndarray = np.array([], dtype=np.int32)

            # Compute score arrays for every iou we need (curve + table).
            iou_all = sorted({float(x) for x in np.concatenate([iou_curve_vals, iou_table_vals]).tolist()})
            scores_by_iou: Dict[float, np.ndarray] = {}
            for iou_thr in iou_all:
                labels_i, scores_i = _compute_scores_all(
                    compute_image_scores_iou,
                    all_sources,
                    metric_conf,
                    float(iou_thr),
                    eval_batch,
                    eval_device,
                    nms_iou,
                    max_det,
                )
                if labels_ref.size == 0:
                    labels_ref = labels_i
                scores_by_iou[float(iou_thr)] = scores_i

            if labels_ref.size and scores_by_iou:
                for conf_thr in fix_list:
                    thr_arr = np.array([float(conf_thr)], dtype=np.float32)

                    # curve points
                    rec_curve = []
                    fpr_curve = []
                    for iou_thr in iou_curve_vals.tolist():
                        scores_i = scores_by_iou.get(float(iou_thr))
                        if scores_i is None or scores_i.size == 0:
                            rec_curve.append(0.0)
                            fpr_curve.append(0.0)
                            continue
                        rec_i, _, fpr_i = compute_threshold_metrics(labels_ref, scores_i, thr_arr)
                        rec_curve.append(float(rec_i[0]))
                        fpr_curve.append(float(fpr_i[0]))
                    curve_recall.append((f"conf={float(conf_thr):.2f}", np.array(rec_curve, dtype=np.float32)))
                    curve_fpr.append((f"conf={float(conf_thr):.2f}", np.array(fpr_curve, dtype=np.float32)))

                    # table points
                    for iou_thr in iou_table_vals.tolist():
                        scores_i = scores_by_iou.get(float(iou_thr))
                        if scores_i is None or scores_i.size == 0:
                            rec_v = 0.0
                            fpr_v = 0.0
                        else:
                            rec_i, _, fpr_i = compute_threshold_metrics(labels_ref, scores_i, thr_arr)
                            rec_v = float(rec_i[0])
                            fpr_v = float(fpr_i[0])
                        multi_rows.append(
                            {
                                "fixed_var": "conf",
                                "fixed_value": float(conf_thr),
                                "threshold": float(iou_thr),
                                "recall": float(rec_v),
                                "fpr": float(fpr_v),
                            }
                        )

                save_multi_curve_grouped(
                    metrics_dir,
                    f"{tag}_recall_vs_iou",
                    iou_curve_vals,
                    curve_recall,
                    xlabel="iou_match",
                    ylabel="recall",
                    title=f"{tag} Recall vs IoU (fixed conf)",
                    group_size=group_size,
                    group_label="confgroup",
                )
                save_multi_curve_grouped(
                    metrics_dir,
                    f"{tag}_fpr_vs_iou",
                    iou_curve_vals,
                    curve_fpr,
                    xlabel="iou_match",
                    ylabel="fpr",
                    title=f"{tag} FPR vs IoU (fixed conf)",
                    group_size=group_size,
                    group_label="confgroup",
                )

        if multi_rows:
            save_threshold_table_multi(metrics_dir, tag, multi_rows)
        elif labels_base.size:
            save_threshold_table(metrics_dir, tag, labels_base, scores_base, table_vals)

    # 3) Save combined image-level report (split column keeps val/test separable)
    if val_items or test_items:
        meta = {
            "conf_threshold": float(image_conf),
            "iou_match": float(match_iou),
            "nms_iou": float(nms_iou),
            "max_det": int(max_det),
            "metric_conf": float(metric_conf),
            "eval_batch": int(eval_batch),
            "eval_device": str(eval_device),
        }
        save_image_level_report(metrics_dir, "eval", val_items + test_items, meta)

    # Keep weights only under exp_dir/train/weights for baseline workflow.


if __name__ == "__main__":
    main()
