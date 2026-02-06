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
import datetime as dt
import gc
import json
import shutil
import subprocess  # enhance241-audit
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


AUDIT_PATH = Path(
    "/home/ubuntu/hpproject/yolo/experiments/yolo11/defect/exp_base2/train/enhance_audit.md"
)  # enhance241-audit


def _audit_append(text: str) -> None:
    """Append audit text block to enhance_audit.md."""
    # enhance241-audit
    AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(AUDIT_PATH, "a", encoding="utf-8") as f:
        f.write(text.rstrip() + "\n")


def _audit_fail(reason: str) -> None:
    # enhance241-audit
    _audit_append(f"FAIL: {reason}")
    raise RuntimeError(reason)


def _git_commit(project_root: Path) -> str:
    # enhance241-audit
    try:
        out = subprocess.check_output(
            ["git", "-C", str(project_root), "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        )
        return out.decode("utf-8", errors="ignore").strip()
    except Exception:
        return "unknown"


def _get_model_seq(obj: Any) -> Optional[Any]:
    # enhance241-audit
    det = getattr(obj, "model", None)
    if det is None:
        return None
    if hasattr(det, "model"):
        return getattr(det, "model", None)
    return det


def _get_torch_model(obj: Any) -> Optional[Any]:
    # enhance241-audit
    det = getattr(obj, "model", None)
    if det is not None and hasattr(det, "state_dict"):
        return det
    if hasattr(obj, "state_dict"):
        return obj
    return None


def _concat_candidates(seq: Optional[Any]) -> List[str]:
    # enhance241-audit
    if seq is None:
        return []
    out: List[str] = []
    for i, layer in enumerate(seq):
        if layer.__class__.__name__ == "Concat":
            out.append(f"{i}:{getattr(layer, 'f', None)}")
    return out


def _patched_nodes(seq: Optional[Any]) -> List[str]:
    # enhance241-audit
    if seq is None:
        return []
    targets = {"P3ASFFLiteFuse", "P4P3GateAlignFuse", "P3FuseChain"}
    out: List[str] = []
    for i, layer in enumerate(seq):
        if layer.__class__.__name__ in targets:
            out.append(f"{i}:{layer.__class__.__name__}")
    return out


def _state_dict_keys_from_ckpt(ckpt_path: Path) -> List[str]:
    # enhance241-audit
    import torch

    try:
        ckpt = torch.load(str(ckpt_path), map_location="cpu")
    except Exception as exc:
        # PyTorch 2.6+ defaults to weights_only=True, which can reject Ultralytics objects.
        # For local training artifacts we trust, retry with weights_only=False.
        if "Weights only load failed" not in str(exc):
            raise
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    keys: List[str] = []
    if isinstance(ckpt, dict):
        model_obj = ckpt.get("model")
        if model_obj is None:
            model_obj = ckpt.get("ema")
        if model_obj is not None:
            if hasattr(model_obj, "state_dict"):
                keys = list(model_obj.state_dict().keys())
            elif isinstance(model_obj, dict):
                keys = list(model_obj.keys())
        elif isinstance(ckpt.get("state_dict"), dict):
            keys = list(ckpt["state_dict"].keys())
        else:
            keys = list(ckpt.keys())
    elif hasattr(ckpt, "state_dict"):
        keys = list(ckpt.state_dict().keys())
    return keys


def _has_keywords(keys: List[str], keywords: List[str]) -> bool:
    # enhance241-audit
    if not keywords:
        return False
    keyset = "\n".join(keys)
    return any(k in keyset for k in keywords)


def _assert_keywords(keys: List[str], keywords: List[str], context: str) -> None:
    # enhance241-audit
    if not keywords:
        _audit_fail(f"{context}: audit keywords empty")
    keyset = "\n".join(keys)
    missing = [k for k in keywords if k not in keyset]
    if missing:
        _audit_fail(f"{context}: missing audit keys {missing}")


def _trainable_param_stats(model: Any, keywords: List[str]) -> Dict[str, int]:
    # enhance241-audit
    torch_model = _get_torch_model(model)
    if torch_model is None:
        return {"matched": 0, "trainable": 0, "trainable_numel": 0, "total_numel": 0}
    matched = []
    for name, param in torch_model.named_parameters():
        if any(k in name for k in keywords):
            matched.append((name, param))
    trainable = [(n, p) for n, p in matched if p.requires_grad]
    return {
        "matched": len(matched),
        "trainable": len(trainable),
        "trainable_numel": int(sum(p.numel() for _, p in trainable)),
        "total_numel": int(sum(p.numel() for _, p in matched)),
    }


def _ensure_optimizer_contains_keywords(opt: Any, model: Any, keywords: List[str]) -> Dict[str, int]:
    # enhance241-audit
    torch_model = _get_torch_model(model)
    if torch_model is None:
        _audit_fail("optimizer check: no torch model available")
    params = [p for n, p in torch_model.named_parameters() if any(k in n for k in keywords)]
    if not params:
        _audit_fail("optimizer check: no enhance241 params found in model.named_parameters()")
    opt_params = {id(p) for g in opt.param_groups for p in g.get("params", [])}
    missing = [p for p in params if id(p) not in opt_params]
    if missing:
        opt.add_param_group({"params": list(missing)})
        opt_params = {id(p) for g in opt.param_groups for p in g.get("params", [])}
    still_missing = [p for p in params if id(p) not in opt_params]
    if still_missing:
        _audit_fail(f"optimizer check: {len(still_missing)} enhance241 params still missing in optimizer")
    return {"params": len(params), "added": len(missing)}


def _summarize_csv_files(dir_path: Path) -> List[str]:
    # enhance241-audit
    if not dir_path.exists():
        return []
    out: List[str] = []
    for p in sorted(dir_path.glob("*.csv")):
        try:
            with open(p, "r", encoding="utf-8") as f:
                lines = sum(1 for _ in f)
        except Exception:
            lines = -1
        try:
            mtime = dt.datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            mtime = "unknown"
        out.append(f"{p.name} lines={lines} mtime={mtime}")
    return out


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
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    exp_root = project_root / "experiments" / yolo_version / exp_name
    run_name = str(cfg.get("run_name", "")).strip()
    if run_name:
        exp_dir = exp_root / run_name
    else:
        timestamp = dt.datetime.now().strftime("%y%m%d%H%M")
        exp_dir = exp_root / f"exp_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    audit_ts = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cmdline = " ".join(sys.argv) if sys.argv else ""
    _audit_append(
        "\n".join(
            [
                "",
                f"## Run {audit_ts}",
                f"cmd: {cmdline or 'unknown'}",
                f"git: {_git_commit(project_root)}",
                f"exp_dir: {exp_dir}",
                f"config: {cfg_path}",
            ]
        )
    )  # enhance241-audit

    data_yaml = Path(cfg.get("data", project_root / "configs" / "data" / "defect.yaml")).resolve()
    data_root_cfg = cfg.get("data_root", "")
    data_roots: List[Path] = []
    if isinstance(data_root_cfg, (list, tuple)):
        data_roots = [Path(str(p)).resolve() for p in data_root_cfg if str(p).strip()]
    else:
        data_root_str = str(data_root_cfg or "").strip()
        if data_root_str:
            data_roots = [Path(data_root_str).resolve()]

    data_root_override = str(data_roots[0]) if data_roots else str(data_root_cfg or "")
    val_split = float(cfg.get("val_split", 0.0))
    seed = int(cfg.get("seed", 42))
    run_data_yaml, data_root = build_data_yaml(
        data_yaml, exp_dir, val_split, seed, data_root_override
    )
    if not data_roots:
        data_roots = [data_root]


    model_path = str(cfg.get("model", "yolo11n.pt"))
    weights_path = str(cfg.get("weights", ""))
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
    _audit_append(
        "\n".join(
            [
                f"data: {data_yaml}",
                f"model: {model_path}",
                f"weights: {weights_path or ''}",
                f"epochs: {epochs} imgsz: {imgsz} batch: {batch}",
                f"postprocess: conf={image_conf:.4f} nms_iou={nms_iou:.4f} max_det={max_det} match_iou={match_iou:.4f}",
                f"metric_conf: {metric_conf:.4f} eval_batch: {eval_batch} eval_device: {eval_device or device}",
            ]
        )
    )  # enhance241-audit

    enhance241_cfg = cfg.get("enhance241", {}) or {}
    if not isinstance(enhance241_cfg, dict):
        raise ValueError("enhance241 must be a mapping when provided in config.")
    known_flags = {"b1", "b2", "a2", "c1", "d1"}
    enabled_switches = sorted([k for k in known_flags if bool(enhance241_cfg.get(k, False))])
    enable_b1 = bool(enhance241_cfg.get("b1", False))
    enable_b2 = bool(enhance241_cfg.get("b2", False))
    audit_keywords: List[str] = []
    audit_kw_cfg = enhance241_cfg.get("audit_keywords", [])
    if isinstance(audit_kw_cfg, str):
        audit_keywords.extend([k.strip() for k in audit_kw_cfg.split(",") if k.strip()])
    elif isinstance(audit_kw_cfg, (list, tuple)):
        audit_keywords.extend([str(k).strip() for k in audit_kw_cfg if str(k).strip()])
    if enable_b1:
        try:
            from third_party.yolo11.enhance241.yolo11_241b1 import ENHANCE241_AUDIT_KEYS

            audit_keywords.extend(list(ENHANCE241_AUDIT_KEYS))
        except Exception:
            pass
    if enable_b2:
        try:
            from third_party.yolo11.enhance241.yolo11_241b2 import ENHANCE241_AUDIT_KEYS

            audit_keywords.extend(list(ENHANCE241_AUDIT_KEYS))
        except Exception:
            pass
    if (enable_b1 or enable_b2) and not audit_keywords:
        audit_keywords = ["enhance241_b1", "enhance241_b2"]
    audit_keywords = sorted({k for k in audit_keywords if k})
    _audit_append(
        "\n".join(
            [
                f"enhance241_enabled: {enabled_switches}",
                f"enhance241_cfg: {json.dumps(enhance241_cfg, ensure_ascii=False)}",
                f"audit_keywords: {audit_keywords}",
            ]
        )
    )  # enhance241-audit

    # Persist a minimal, reproducible run meta for traceability.
    try:
        shutil.copy2(cfg_path, exp_dir / "config.yaml")
    except Exception:
        pass
    try:
        meta = {
            "eval_standard": "P2.3.0",
            "config_path": str(cfg_path),
            "enhance241_enabled": enabled_switches,
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

    def apply_enhance241(yolo_model: Any) -> Any:
        if not enabled_switches:
            return yolo_model
        unsupported = [
            k
            for k, v in enhance241_cfg.items()
            if k not in known_flags and isinstance(v, bool) and v
        ]
        if unsupported:
            raise RuntimeError(f"Unsupported enhance241 switches enabled: {unsupported}. Only b1/b2 are supported.")
        if enable_b1:
            try:
                from third_party.yolo11.enhance241.yolo11_241b1 import apply as apply_241b1
            except Exception as exc:
                raise RuntimeError(
                    "enhance241.b1 is enabled but module import failed: "
                    "third_party.yolo11.enhance241.yolo11_241b1"
                ) from exc
            yolo_model = apply_241b1(yolo_model, cfg)
        if enable_b2:
            try:
                from third_party.yolo11.enhance241.yolo11_241b2 import apply as apply_241b2
            except Exception as exc:
                raise RuntimeError(
                    "enhance241.b2 is enabled but module import failed: "
                    "third_party.yolo11.enhance241.yolo11_241b2"
                ) from exc
            yolo_model = apply_241b2(yolo_model, cfg)
        return yolo_model

    def _attach_audit_callbacks(yolo_model: Any) -> None:
        # enhance241-audit
        if not (enable_b1 or enable_b2):
            return
        if not hasattr(yolo_model, "add_callback"):
            _audit_fail("enhance241 audit requires YOLO.add_callback but it is unavailable.")
        if getattr(yolo_model, "_enhance241_audit_cb", False):
            return
        setattr(yolo_model, "_enhance241_audit_cb", True)

        def _maybe_patch(trainer: Any, stage: str, require_model: bool) -> None:
            if not hasattr(trainer, "model"):
                msg = f"{stage}: trainer.model unavailable"
                if require_model:
                    _audit_fail(msg)
                _audit_append(msg)
                return
            if getattr(trainer, "_enhance241_patched", False):
                return
            trainer.model = apply_enhance241(trainer.model)
            setattr(trainer, "_enhance241_patched", True)
            seq = _get_model_seq(trainer.model)
            _audit_append(
                f"{stage}_patch: concat_candidates={_concat_candidates(seq)} patched={_patched_nodes(seq)}"
            )
            torch_model = _get_torch_model(trainer.model)
            if torch_model is None:
                _audit_fail(f"{stage}: trainer.model is not a torch module")
            keys = list(torch_model.state_dict().keys())
            _assert_keywords(keys, audit_keywords, f"{stage} trainer.model after patch")

        def on_pretrain_routine_start(trainer: Any) -> None:
            _maybe_patch(trainer, "pretrain_start", require_model=False)

        def on_train_start(trainer: Any) -> None:
            _maybe_patch(trainer, "train_start", require_model=True)
            opt = getattr(trainer, "optimizer", None)
            if opt is None:
                _audit_fail("optimizer missing at train_start")
            res = _ensure_optimizer_contains_keywords(opt, trainer.model, audit_keywords)
            _audit_append(f"optimizer_check: params={res['params']} added={res['added']}")

        yolo_model.add_callback("on_pretrain_routine_start", on_pretrain_routine_start)
        yolo_model.add_callback("on_train_start", on_train_start)

    mode = str(cfg.get("mode", "train_test")).lower()
    _audit_append(f"mode: {mode}")  # enhance241-audit
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
    pre_seq = _get_model_seq(model)
    _audit_append(
        f"pre_patch: concat_candidates={_concat_candidates(pre_seq)} patched={_patched_nodes(pre_seq)}"
    )  # enhance241-audit
    model = apply_enhance241(model)
    post_seq = _get_model_seq(model)
    _audit_append(
        f"post_patch: concat_candidates={_concat_candidates(post_seq)} patched={_patched_nodes(post_seq)}"
    )  # enhance241-audit
    if enable_b1 or enable_b2:
        torch_model = _get_torch_model(model)
        if torch_model is None:
            _audit_fail("post_patch: model is not a torch module")
        _assert_keywords(list(torch_model.state_dict().keys()), audit_keywords, "post_patch model")
        stats = _trainable_param_stats(model, audit_keywords)
        _audit_append(
            f"trainable_params: matched={stats['matched']} trainable={stats['trainable']} "
            f"trainable_numel={stats['trainable_numel']} total_numel={stats['total_numel']}"
        )
    _attach_audit_callbacks(model)  # enhance241-audit
    if mode in {"train_test", "finetune_test"}:
        try:
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
        except UnicodeDecodeError as exc:
            print(
                "WARNING: Training hit UnicodeDecodeError during plotting (likely SciPy->NumPy lscpu decode). "
                "Proceeding with whatever checkpoints were saved. "
                f"Details: {exc}"
            )
        update_args_yaml(exp_dir, cfg, cfg_path)

    best_weights = exp_dir / "train" / "weights" / "best.pt"
    if mode in {"train_test", "finetune_test"}:
        if (enable_b1 or enable_b2) and not best_weights.exists():
            _audit_fail("best.pt missing for enhance241 run; refusing to continue evaluation.")  # enhance241-audit
        eval_weights = best_weights if best_weights.exists() else (exp_dir / "train" / "weights" / "last.pt")
        if not eval_weights.exists():
            _audit_fail(f"eval_weights missing: {eval_weights}")  # enhance241-audit
        _audit_append(f"eval_weights: {eval_weights}")  # enhance241-audit
        if enable_b1 or enable_b2:
            ckpt_keys = _state_dict_keys_from_ckpt(eval_weights)
            _assert_keywords(ckpt_keys, audit_keywords, f"ckpt {eval_weights.name}")
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
        model = apply_enhance241(model)
        if enable_b1 or enable_b2:
            torch_model = _get_torch_model(model)
            if torch_model is None:
                _audit_fail("eval model is not a torch module after reload")
            _assert_keywords(list(torch_model.state_dict().keys()), audit_keywords, "eval model after patch")
    elif mode == "test":
        test_weight = Path(weights_path)
        if test_weight.exists():
            eval_weights = test_weight
        else:
            cached = resolve_pretrained_weight(project_root, test_weight.name)
            eval_weights = cached
        if not eval_weights.exists():
            _audit_fail(f"eval_weights missing: {eval_weights}")  # enhance241-audit
        if enable_b1 or enable_b2:
            ckpt_keys = _state_dict_keys_from_ckpt(eval_weights)
            _assert_keywords(ckpt_keys, audit_keywords, f"ckpt {eval_weights.name}")
        model = YOLO(str(eval_weights))
        model = apply_enhance241(model)
        if enable_b1 or enable_b2:
            torch_model = _get_torch_model(model)
            if torch_model is None:
                _audit_fail("test model is not a torch module after patch")
            _assert_keywords(list(torch_model.state_dict().keys()), audit_keywords, "test model after patch")

    # Resolve split sources across one or multiple dataset roots.
    data_info_cfg = load_yaml(data_yaml)
    train_entry = data_info_cfg.get("train", "images/train")
    val_entry = data_info_cfg.get("val", "images/val") or train_entry
    test_entry = data_info_cfg.get("test", "")

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
    _audit_append(f"threshold_sweep: {json.dumps(sweep_cfg, ensure_ascii=False)}")  # enhance241-audit

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

    csv_summary = _summarize_csv_files(metrics_dir)
    if csv_summary:
        _audit_append("eval_outputs:\n" + "\n".join(csv_summary))  # enhance241-audit
    else:
        _audit_append("eval_outputs: none")  # enhance241-audit
    _audit_append("PASS")  # enhance241-audit

    if mode in {"train_test", "finetune_test"}:
        model_dir = project_root / "models" / exp_name
        copy_weights(exp_dir, model_dir)


if __name__ == "__main__":
    main()
