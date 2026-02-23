import locale
import os
import sys
from pathlib import Path


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

# Ensure repo root is importable when running `python src/train.py`.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import argparse
import atexit
import csv
import datetime as dt
import gc
import hashlib
import json
import shutil
import subprocess
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.data.make_split import create_val_split
from src.eval import (
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

FIXED_AUDIT_PATH = Path(
    "/home/ubuntu/hpproject/yolo/experiments/yolo11/defect/exp_base2/train/audit_enhance241.md"
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


def append_audit(md_path: Path, dict_or_text: Any) -> None:
    md_path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(dict_or_text, str):
        payload = dict_or_text
    else:
        payload = json.dumps(dict_or_text, ensure_ascii=False, indent=2)
    with open(md_path, "a", encoding="utf-8") as f:
        f.write(payload)


def _write_hparam_plan(exp_dir: Path, cfg: Dict[str, Any], cfg_path: Path) -> None:
    plan_path = exp_dir / "train" / "hparam_plan.md"
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# HParam Plan ({dt.datetime.now().isoformat(timespec='seconds')})",
        "",
        "## Baseline Anchor",
        "- epochs: 100",
        "- batch: 10",
        "- patience: 0",
        "- lr0: 0.008",
        "- lrf: 0.12",
        "- warmup_epochs: 2",
        "",
        "## This Run",
        f"- config_path: `{cfg_path}`",
        f"- epochs: `{cfg.get('epochs', 100)}`",
        f"- batch: `{cfg.get('batch', 10)}`",
        f"- workers: `{cfg.get('workers', 8)}`",
        f"- patience: `{cfg.get('patience', 0)}`",
        f"- lr0: `{cfg.get('lr0', 0.008)}`",
        f"- lrf: `{cfg.get('lrf', 0.12)}`",
        f"- warmup_epochs: `{cfg.get('warmup_epochs', 2)}`",
        f"- seed: `{cfg.get('seed', 0)}`",
        "",
        "## Rule",
        "- Keep `total_optimizer_steps` comparable across runs when comparing module deltas.",
    ]
    plan_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _save_roc_curve_csv(
    metrics_dir: Path,
    tag: str,
    thresholds: np.ndarray,
    recall: np.ndarray,
    precision: np.ndarray,
    fpr: np.ndarray,
) -> None:
    metrics_dir.mkdir(parents=True, exist_ok=True)
    out_path = metrics_dir / "roc_curve.csv"
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["tag", "threshold", "recall", "precision", "fpr"])
        for thr, rec, pre, fp in zip(thresholds.tolist(), recall.tolist(), precision.tolist(), fpr.tolist()):
            writer.writerow([str(tag), float(thr), float(rec), float(pre), float(fp)])


def _save_roc_keypoints_md(
    metrics_dir: Path,
    tag: str,
    thresholds: np.ndarray,
    recall: np.ndarray,
    fpr: np.ndarray,
    targets: Tuple[float, ...] = (0.05, 0.10, 0.30),
) -> None:
    metrics_dir.mkdir(parents=True, exist_ok=True)
    lines = [f"# ROC Keypoints ({tag})", "", "| target_fpr | nearest_fpr | recall | threshold |", "|---:|---:|---:|---:|"]
    if fpr.size == 0:
        for t in targets:
            lines.append(f"| {t:.2f} | 0.0000 | 0.0000 | 0.0000 |")
    else:
        for t in targets:
            idx = int(np.argmin(np.abs(fpr - float(t))))
            lines.append(
                f"| {t:.2f} | {float(fpr[idx]):.4f} | {float(recall[idx]):.4f} | {float(thresholds[idx]):.4f} |"
            )
    (metrics_dir / "roc_keypoints.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_result_summary(
    exp_dir: Path,
    *,
    mode: str,
    eval_weights: str,
    auroc: float,
    ap: float,
    labels_count: int,
    sweep_cfg: Dict[str, Any],
    best_last_compare: Optional[Dict[str, Any]] = None,
) -> None:
    out = exp_dir / "train" / "result_summary.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# Result Summary ({dt.datetime.now().isoformat(timespec='seconds')})",
        "",
        f"- mode: `{mode}`",
        f"- eval_weights: `{eval_weights}`",
        f"- image_auroc: `{auroc:.6f}`",
        f"- image_ap: `{ap:.6f}`",
        f"- samples: `{labels_count}`",
        "- key_files:",
        f"  - `{exp_dir / 'metrics' / 'roc_curve.csv'}`",
        f"  - `{exp_dir / 'metrics' / 'roc_keypoints.md'}`",
        f"  - `{exp_dir / 'train' / 'enhance241_audit.md'}`",
        "",
        "## Sweep Config",
        "```json",
        json.dumps(sweep_cfg, ensure_ascii=False, indent=2),
        "```",
    ]
    if best_last_compare is not None:
        lines.extend(
            [
                "",
                "## Best vs Last",
                "```json",
                json.dumps(best_last_compare, ensure_ascii=False, indent=2),
                "```",
            ]
        )
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _collect_env_probe() -> Dict[str, Any]:
    probe: Dict[str, Any] = {}
    probe["pwd"] = str(Path.cwd().resolve())
    probe["python"] = sys.executable
    probe["python_version"] = sys.version.splitlines()[0]
    probe["sys_path_head"] = list(sys.path[:10])

    try:
        import ultralytics  # type: ignore

        probe["ultralytics"] = str(getattr(ultralytics, "__version__", "unknown"))
    except Exception as exc:
        probe["ultralytics"] = f"import_failed: {exc}"

    try:
        import torch  # type: ignore

        probe["torch"] = str(getattr(torch, "__version__", "unknown"))
    except Exception as exc:
        probe["torch"] = f"import_failed: {exc}"

    try:
        import src  # type: ignore

        probe["import_src"] = str(getattr(src, "__file__", "unknown"))
    except Exception as exc:
        probe["import_src"] = f"failed: {exc}"

    try:
        import inspect
        import src.train as src_train  # type: ignore

        probe["import_src_train"] = str(inspect.getsourcefile(src_train) or getattr(src_train, "__file__", "unknown"))
    except Exception as exc:
        probe["import_src_train"] = f"failed: {exc}"

    return probe


def _sha1_file(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _collect_file_hashes(project_root: Path) -> List[Dict[str, str]]:
    targets = [
        project_root / "src" / "train.py",
        project_root / "tools" / "run_yolov11_241.sh",
        project_root / "tools" / "run_yolov11.sh",
        project_root / "configs" / "yolo11" / "enhance241" / "defect241.yaml",
        project_root / "configs" / "yolo11" / "enhance241" / "defect241b2.yaml",
        project_root / "third_party" / "yolo11" / "enhance241" / "yolo11_241a3.py",
        project_root / "third_party" / "yolo11" / "enhance241" / "yolo11_241a5.py",
        project_root / "third_party" / "yolo11" / "enhance241" / "yolo11_241b2.py",
        project_root / "third_party" / "yolo11" / "enhance241" / "yolo11_241b3.py",
        project_root / "third_party" / "yolo11" / "enhance241" / "yolo11_241b5.py",
        project_root / "third_party" / "yolo11" / "enhance241" / "yolo11_241c5.py",
        project_root / "third_party" / "yolo11" / "enhance241" / "yolo11_241d1.py",
        project_root / "third_party" / "yolo11" / "enhance241" / "yolo11_241d3.py",
        project_root / "third_party" / "yolo11" / "enhance241" / "yolo11_241d5.py",
    ]
    out: List[Dict[str, str]] = []
    for p in targets:
        if p.exists():
            out.append({"path": str(p), "sha1": _sha1_file(p)})
        else:
            out.append({"path": str(p), "sha1": "missing"})
    return out


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


def _safe_float(v: Any) -> Optional[float]:
    try:
        return float(v)
    except Exception:
        return None


def append_best_summary_to_results_csv(
    exp_dir: Path,
    *,
    metric_key: str = "metrics/mAP50-95(B)",
) -> Optional[int]:
    """Append one BEST_SUMMARY row to train/results.csv.

    The summary row records which epoch is selected as "best" by a metric proxy.
    It keeps CSV format valid and is easy to inspect when reviewing experiments.
    """

    results_path = exp_dir / "train" / "results.csv"
    if not results_path.exists():
        return None

    with open(results_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = [dict(r) for r in reader]
    if not fieldnames or not rows:
        return None

    marker_col = "best_epoch_note"
    # De-duplicate previously appended summary rows.
    rows = [r for r in rows if not str(r.get(marker_col, "")).startswith("BEST_SUMMARY")]
    if not rows:
        return None

    metric_candidates = [metric_key, "metrics/mAP50-95(B)", "metrics/mAP50(B)", "metrics/recall(B)"]
    best_idx: Optional[int] = None
    best_epoch: Optional[int] = None
    best_metric_name: Optional[str] = None
    best_metric_value = -float("inf")

    for i, r in enumerate(rows):
        epoch_val = _safe_float(r.get("epoch", ""))
        if epoch_val is None:
            continue
        metric_name: Optional[str] = None
        metric_val: Optional[float] = None
        for k in metric_candidates:
            v = _safe_float(r.get(k, ""))
            if v is not None:
                metric_name = k
                metric_val = v
                break
        if metric_name is None or metric_val is None:
            continue
        if best_idx is None or metric_val > best_metric_value:
            best_idx = i
            best_metric_value = float(metric_val)
            best_metric_name = metric_name
            best_epoch = int(round(epoch_val))

    if best_idx is None or best_epoch is None or best_metric_name is None:
        return None

    if marker_col not in fieldnames:
        fieldnames.append(marker_col)
    for r in rows:
        r.setdefault(marker_col, "")

    summary = {k: "" for k in fieldnames}
    summary["epoch"] = str(best_epoch)
    if "time" in fieldnames:
        summary["time"] = rows[best_idx].get("time", "")
    summary[best_metric_name] = rows[best_idx].get(best_metric_name, "")
    summary[marker_col] = f"BEST_SUMMARY metric={best_metric_name}"
    rows.append(summary)

    with open(results_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    return int(best_epoch)


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
    keys = [
        "a3",
        "a5",
        "a7",
        "a9",
        "b1",
        "b2",
        "b3",
        "b5",
        "b7",
        "b9",
        "c5",
        "c7",
        "c9",
        "d1",
        "d3",
        "d5",
        "d7",
        "d9",
    ]
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
        "A5P3Residual",
        "C3HBHorNetSafe",
        "A9SESAMBackboneSafe",
        "NASFPNLiteFuse",
        "B5GFPNFuse",
        "CARAFEUpsampleSafe",
        "B9ImprovedCSPFuseSafe",
        "C5BRAInject",
        "C7MCBAMInject",
        "C9SESAMGuard",
        "P2LiteFuse",
        "P3LogitTemperature",
        "P6Downsample",
        "D9HeadScoreCalib",
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
    _, det_model, _ = _extract_model_seq(model)
    named = list(det_model.named_parameters()) if hasattr(det_model, "named_parameters") else []
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
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        try:
            ckpt = torch.load(path, map_location="cpu")
        except Exception as exc:
            return {"count": 0, "sample_keys": [], "source": f"load_failed:{exc}"}
    except Exception as exc:
        return {"count": 0, "sample_keys": [], "source": f"load_failed:{exc}"}
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
        ("a5", "_enhance241_a5_info"),
        ("a7", "_enhance241_a7_info"),
        ("a9", "_enhance241_a9_info"),
        ("b1", "_enhance241_b1_info"),
        ("b2", "_enhance241_b2_info"),
        ("b3", "_enhance241_b3_info"),
        ("b5", "_enhance241_b5_info"),
        ("b7", "_enhance241_b7_info"),
        ("b9", "_enhance241_b9_info"),
        ("c5", "_enhance241_c5_info"),
        ("c7", "_enhance241_c7_info"),
        ("c9", "_enhance241_c9_info"),
        ("d1", "_enhance241_d1_info"),
        ("d3", "_enhance241_d3_info"),
        ("d5", "_enhance241_d5_info"),
        ("d7", "_enhance241_d7_info"),
        ("d9", "_enhance241_d9_info"),
    ):
        val = getattr(model, attr, None)
        if val is None and hasattr(model, "model"):
            val = getattr(model.model, attr, None)
        if val is not None:
            info[key] = val
    return info


def _evaluate_enhance241_checks(audit_state: Dict[str, Any]) -> Tuple[str, List[Dict[str, Any]]]:
    enabled = set(audit_state.get("enhance241_enabled", []))
    info = (
        audit_state.get("train_runtime_infos")
        or audit_state.get("eval_infos")
        or audit_state.get("train_infos")
        or {}
    )
    checks: List[Dict[str, Any]] = []

    if "a3" in enabled:
        a3 = info.get("a3", {}) if isinstance(info, dict) else {}
        replaced = int(a3.get("replaced_count", 0)) + int(a3.get("existing_count", 0))
        ok = replaced == 1
        checks.append({"name": "a3_replace_count", "ok": ok, "detail": f"count={replaced}"})

    if "a5" in enabled:
        a5 = info.get("a5", {}) if isinstance(info, dict) else {}
        patched = int(a5.get("patched_count", 0)) + int(a5.get("existing_count", 0))
        ok = patched == 1
        checks.append({"name": "a5_patch_count", "ok": ok, "detail": f"count={patched}"})

    if "a7" in enabled:
        a7 = info.get("a7", {}) if isinstance(info, dict) else {}
        patched = int(a7.get("patched_count", 0)) + int(a7.get("existing_count", 0))
        ok = patched == 1
        checks.append({"name": "a7_patch_count", "ok": ok, "detail": f"count={patched}"})

    if "a9" in enabled:
        a9 = info.get("a9", {}) if isinstance(info, dict) else {}
        patched = int(a9.get("patched_count", 0)) + int(a9.get("existing_count", 0))
        ok = patched == 1
        checks.append({"name": "a9_patch_count", "ok": ok, "detail": f"count={patched}"})

    if "b3" in enabled:
        b3 = info.get("b3", {}) if isinstance(info, dict) else {}
        patched = int(b3.get("patched_count", 0))
        ok = patched == 2
        checks.append({"name": "b3_patch_count", "ok": ok, "detail": f"count={patched}"})

    if "b5" in enabled:
        b5 = info.get("b5", {}) if isinstance(info, dict) else {}
        patched = int(b5.get("patched_count", 0)) + int(b5.get("existing_count", 0))
        ok = patched == 2
        checks.append({"name": "b5_patch_count", "ok": ok, "detail": f"count={patched}"})

    if "b7" in enabled:
        b7 = info.get("b7", {}) if isinstance(info, dict) else {}
        patched = int(b7.get("patched_count", 0)) + int(b7.get("existing_count", 0))
        ok = patched >= 1
        checks.append({"name": "b7_patch_count", "ok": ok, "detail": f"count={patched}"})

    if "b9" in enabled:
        b9 = info.get("b9", {}) if isinstance(info, dict) else {}
        patched = int(b9.get("patched_count", 0)) + int(b9.get("existing_count", 0))
        ok = patched == 1
        checks.append({"name": "b9_patch_count", "ok": ok, "detail": f"count={patched}"})

    if "c5" in enabled:
        c5 = info.get("c5", {}) if isinstance(info, dict) else {}
        patched = int(c5.get("patched_count", 0)) + int(c5.get("existing_count", 0))
        ok = patched == 1
        checks.append({"name": "c5_patch_count", "ok": ok, "detail": f"count={patched}"})

    if "c7" in enabled:
        c7 = info.get("c7", {}) if isinstance(info, dict) else {}
        patched = int(c7.get("patched_count", 0)) + int(c7.get("existing_count", 0))
        ok = patched == 1
        checks.append({"name": "c7_patch_count", "ok": ok, "detail": f"count={patched}"})

    if "c9" in enabled:
        c9 = info.get("c9", {}) if isinstance(info, dict) else {}
        patched = int(c9.get("patched_count", 0)) + int(c9.get("existing_count", 0))
        ok = patched == 1
        checks.append({"name": "c9_patch_count", "ok": ok, "detail": f"count={patched}"})

    if "b2" in enabled:
        pre = audit_state.get("train_runtime_pre_patch", {}) or {}
        post = audit_state.get("train_runtime_post_patch", {}) or {}
        pre_patched = len(pre.get("patched", [])) if isinstance(pre, dict) else 0
        post_patched = len(post.get("patched", [])) if isinstance(post, dict) else 0
        ok = post_patched > pre_patched
        checks.append(
            {
                "name": "b2_runtime_patch_applied",
                "ok": ok,
                "detail": f"pre={pre_patched}, post={post_patched}",
            }
        )

    if "d1" in enabled or "d3" in enabled:
        d3 = {}
        if isinstance(info, dict):
            d3 = info.get("d3", {}) or info.get("d1", {})
        patched = int(d3.get("patched_count", 0)) + int(d3.get("existing_count", 0))
        ok = patched == 1
        checks.append(
            {
                "name": "d3_temperature_patch",
                "ok": ok,
                "detail": f"patched_or_existing={patched}",
            }
        )

    if "d5" in enabled:
        d5 = info.get("d5", {}) if isinstance(info, dict) else {}
        heads_after = int(d5.get("detect_heads_after", 0))
        patched = int(d5.get("patched_count", 0)) + int(d5.get("existing_count", 0))
        ok = patched == 1 and heads_after == 4
        checks.append(
            {
                "name": "d5_head_count",
                "ok": ok,
                "detail": f"patched_or_existing={patched}, heads_after={heads_after}",
            }
        )

    if "d7" in enabled:
        d7 = info.get("d7", {}) if isinstance(info, dict) else {}
        heads_after = int(d7.get("detect_heads_after", 0))
        patched = int(d7.get("patched_count", 0)) + int(d7.get("existing_count", 0))
        ok = patched == 1 and heads_after == 1
        checks.append(
            {
                "name": "d7_small_head_only",
                "ok": ok,
                "detail": f"patched_or_existing={patched}, heads_after={heads_after}",
            }
        )

    if "d9" in enabled:
        d9 = info.get("d9", {}) if isinstance(info, dict) else {}
        patched = int(d9.get("patched_count", 0)) + int(d9.get("existing_count", 0))
        ok = patched == 1
        checks.append(
            {
                "name": "d9_score_calib_patch",
                "ok": ok,
                "detail": f"patched_or_existing={patched}",
            }
        )

    status = "PASS" if all(c.get("ok", False) for c in checks) else "FAIL"
    return status, checks


def _append_stage_snapshot(lines: List[str], audit_state: Dict[str, Any], stage: str) -> None:
    pre = audit_state.get(f"{stage}_pre_patch", {}) or {}
    post = audit_state.get(f"{stage}_post_patch", {}) or {}
    if not pre and not post:
        return
    lines.append(f"- {stage}.pre_patch.concat_candidates={pre.get('concat_candidates', [])}")
    lines.append(f"- {stage}.pre_patch.patched={pre.get('patched', [])}")
    lines.append(f"- {stage}.pre_patch.keyword_hits={pre.get('keyword_hits', {})}")
    lines.append(f"- {stage}.post_patch.concat_candidates={post.get('concat_candidates', [])}")
    lines.append(f"- {stage}.post_patch.patched={post.get('patched', [])}")
    lines.append(f"- {stage}.post_patch.keyword_hits={post.get('keyword_hits', {})}")
    if audit_state.get(f"{stage}_trainable_params") is not None:
        lines.append(f"- {stage}.trainable_params={audit_state.get(f'{stage}_trainable_params')}")
    if audit_state.get(f"{stage}_infos") is not None:
        lines.append(f"- {stage}.infos={audit_state.get(f'{stage}_infos')}")


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

    lines.append("### A1 Environment And Import")
    lines.append("```json")
    lines.append(json.dumps(audit_state.get("env_probe", {}), ensure_ascii=False, indent=2))
    lines.append("```")

    lines.append("### A2 SHA1 Snapshot")
    lines.append("```text")
    for row in audit_state.get("file_hashes", []):
        lines.append(f"{row.get('sha1', 'missing')}  {row.get('path', '')}")
    lines.append("```")

    lines.append("### A3 Patch Runtime")
    for stage in ("train", "train_runtime", "eval"):
        _append_stage_snapshot(lines, audit_state, stage)

    lines.append(f"- eval_weights: {audit_state.get('eval_weights', '')}")
    lines.append(f"- ckpt_keyword_hits: {audit_state.get('ckpt_keyword_hits', {})}")
    lines.append(f"- threshold_sweep: {audit_state.get('threshold_sweep', {})}")

    lines.append("### A4 Score Diagnostics")
    lines.append("```json")
    lines.append(json.dumps(audit_state.get("a4_diag", {}), ensure_ascii=False, indent=2))
    lines.append("```")

    lines.append(f"- checks: {checks}")
    lines.append(f"- conclusion: {status}")
    return "\n".join(lines) + "\n"


def _append_enhance241_audit(audit_path: Path, audit_state: Dict[str, Any]) -> None:
    block = _format_enhance241_audit(audit_state)
    append_audit(audit_path, block)


def _register_enhance241_audit(audit_path: Path, audit_state: Dict[str, Any]) -> None:
    def _writer() -> None:
        try:
            eval_w = audit_state.get("eval_weights")
            if eval_w:
                audit_state["ckpt_keyword_hits"] = _load_ckpt_keyword_hits(Path(eval_w))
            _append_enhance241_audit(audit_path, audit_state)
            if FIXED_AUDIT_PATH.resolve() != audit_path.resolve():
                _append_enhance241_audit(FIXED_AUDIT_PATH, audit_state)
        except Exception:
            pass

    atexit.register(_writer)


def _apply_enhance241_patches(model: Any, cfg: Dict[str, Any], stage: str, audit_state: Dict[str, Any]) -> Any:
    stage_key = str(stage).strip().replace(" ", "_")
    pre = _snapshot_model(model)
    audit_state[f"{stage_key}_pre_patch"] = pre
    if stage_key == "train":
        audit_state["train_pre_patch"] = pre
    elif stage_key == "eval":
        audit_state["eval_pre_patch"] = pre

    enabled = _enhance241_enabled_keys(cfg)
    if enabled:
        a_enabled = [k for k in ("a3", "a5", "a7", "a9") if k in enabled]
        if len(a_enabled) > 1:
            raise RuntimeError(f"enhance241 A-class conflict: {a_enabled}; enable only one of a3/a5/a7/a9.")

        b_enabled = [k for k in ("b1", "b2", "b3", "b5", "b7", "b9") if k in enabled]
        if len(b_enabled) > 1:
            raise RuntimeError(f"enhance241 B-class conflict: {b_enabled}; enable only one of b1/b2/b3/b5/b7/b9.")

        c_enabled = [k for k in ("c5", "c7", "c9") if k in enabled]
        if len(c_enabled) > 1:
            raise RuntimeError(f"enhance241 C-class conflict: {c_enabled}; enable only one of c5/c7/c9.")

        d_enabled_norm = []
        for k in enabled:
            if k in ("d1", "d3"):
                d_enabled_norm.append("d3")
            elif k in ("d5", "d7", "d9"):
                d_enabled_norm.append(k)
        d_enabled_norm = sorted(set(d_enabled_norm))
        if len(d_enabled_norm) > 1:
            raise RuntimeError(f"enhance241 D-class conflict: {d_enabled_norm}; enable only one of d3/d5/d7/d9.")
        from third_party.yolo11.enhance241 import (
            yolo11_241a3,
            yolo11_241a5,
            yolo11_241a7,
            yolo11_241a9,
            yolo11_241b1,
            yolo11_241b2,
            yolo11_241b3,
            yolo11_241b5,
            yolo11_241b7,
            yolo11_241b9,
            yolo11_241c5,
            yolo11_241c7,
            yolo11_241c9,
            yolo11_241d3,
            yolo11_241d5,
            yolo11_241d7,
            yolo11_241d9,
        )

        model = yolo11_241a3.apply(model, cfg)
        model = yolo11_241a5.apply(model, cfg)
        model = yolo11_241a7.apply(model, cfg)
        model = yolo11_241a9.apply(model, cfg)
        model = yolo11_241b1.apply(model, cfg)
        model = yolo11_241b2.apply(model, cfg)
        model = yolo11_241b3.apply(model, cfg)
        model = yolo11_241b5.apply(model, cfg)
        model = yolo11_241b7.apply(model, cfg)
        model = yolo11_241b9.apply(model, cfg)
        model = yolo11_241c5.apply(model, cfg)
        model = yolo11_241c7.apply(model, cfg)
        model = yolo11_241c9.apply(model, cfg)
        model = yolo11_241d5.apply(model, cfg)
        model = yolo11_241d7.apply(model, cfg)
        model = yolo11_241d9.apply(model, cfg)
        model = yolo11_241d3.apply(model, cfg)

    post = _snapshot_model(model)
    infos = _collect_enhance241_infos(model)
    trainable = _collect_trainable_params(model)
    audit_state[f"{stage_key}_post_patch"] = post
    audit_state[f"{stage_key}_infos"] = infos
    audit_state[f"{stage_key}_trainable_params"] = trainable

    if stage_key == "train":
        audit_state["train_post_patch"] = post
        audit_state["train_infos"] = infos
        audit_state["trainable_params"] = trainable
    elif stage_key == "train_runtime":
        audit_state["train_runtime_post_patch"] = post
        audit_state["train_runtime_infos"] = infos
    elif stage_key == "eval":
        audit_state["eval_post_patch"] = post
        audit_state["eval_infos"] = infos
    return model


def _build_enhance241_trainer_cls(yolo_model: Any, cfg: Dict[str, Any], audit_state: Dict[str, Any]) -> Optional[Any]:
    enabled = _enhance241_enabled_keys(cfg)
    if not enabled:
        return None

    base_cls = yolo_model._smart_load("trainer")
    cfg_dict = cfg

    class Enhance241Trainer(base_cls):
        def get_model(self, cfg: Any = None, weights: Any = None, verbose: bool = True) -> Any:
            m = super().get_model(cfg=cfg, weights=weights, verbose=verbose)
            return _apply_enhance241_patches(m, cfg_dict, stage="train_runtime", audit_state=audit_state)

    Enhance241Trainer.__name__ = f"Enhance241{base_cls.__name__}"
    return Enhance241Trainer


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
    cfg["enhance241_exp_dir"] = str(exp_dir)
    os.environ["ENHANCE241_EXP_DIR"] = str(exp_dir)

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
        "env_probe": _collect_env_probe(),
        "file_hashes": _collect_file_hashes(project_root),
    }
    _register_enhance241_audit(exp_dir / "train" / "enhance241_audit.md", audit_state)
    _write_hparam_plan(exp_dir, cfg, cfg_path)

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
        train_kwargs = dict(
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
        trainer_cls = _build_enhance241_trainer_cls(model, cfg, audit_state)
        if trainer_cls is not None:
            model.train(trainer=trainer_cls, **train_kwargs)
        else:
            model.train(**train_kwargs)
        update_args_yaml(exp_dir, cfg, cfg_path)
        best_metric_key = str(cfg.get("best_epoch_metric", "metrics/mAP50-95(B)"))
        best_epoch_mark = append_best_summary_to_results_csv(exp_dir, metric_key=best_metric_key)
        if best_epoch_mark is not None:
            print(
                f"[best_marker] results.csv appended BEST_SUMMARY epoch={best_epoch_mark} "
                f"(metric={best_metric_key})"
            )

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
    default_plot_cfg = {
        "roc": {
            "xlim": [0.0, 1.0],
            "ylim": [0.0, 1.0],
            "xtick_step": 0.05,
            "ytick_step": 0.05,
        },
        "curve": {
            "xlim": [0.0, 1.0],
            "ylim": [0.0, 1.0],
            "xtick_step": 0.05,
            "ytick_step": 0.05,
        }
    }
    plot_cfg = sweep_cfg.get("plot_cfg", default_plot_cfg)
    if not isinstance(plot_cfg, dict):
        plot_cfg = default_plot_cfg
    roc_axis_cfg = plot_cfg.get("roc", {}) if isinstance(plot_cfg.get("roc", {}), dict) else {}
    curve_axis_cfg = plot_cfg.get("curve", {}) if isinstance(plot_cfg.get("curve", {}), dict) else {}
    pr_axis_cfg = plot_cfg.get("pr", curve_axis_cfg) if isinstance(plot_cfg.get("pr", curve_axis_cfg), dict) else {}
    audit_state["threshold_sweep"] = {
        "fix": fix_var,
        "fix_list": fix_list,
        "curve": curve_range,
        "table": table_range,
        "plot_cfg": plot_cfg,
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

    def _compute_scores_all(
        fn,
        sources: List[Path],
        *args: object,
        model_obj: Optional[Any] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        active_model = model if model_obj is None else model_obj
        labels_all: List[np.ndarray] = []
        scores_all: List[np.ndarray] = []
        for src in sources:
            try:
                lab, sc = fn(active_model, src, *args)
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
                    lab, sc = fn(active_model, src, *args[:1], 1, "cpu", *args[3:])
                else:
                    lab, sc = fn(active_model, src, *args[:2], 1, "cpu", *args[4:])
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
    run_auroc = 0.0
    run_ap = 0.0
    run_label_count = 0
    run_best_last_compare: Optional[Dict[str, Any]] = None
    if all_sources:
        metrics_dir.mkdir(parents=True, exist_ok=True)
        tag = "eval"

        # Save sweep meta
        try:
            sweep_meta = {
                "fix": fix_var,
                "fix_list": fix_list,
                "curve": curve_range,
                "table": table_range,
                "plot_cfg": plot_cfg,
            }
            (metrics_dir / f"{tag}_threshold_sweep.json").write_text(
                json.dumps(sweep_meta, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except Exception:
            pass

        # Base ROC/PR vs confidence threshold (reference)
        labels_base, scores_base = _compute_scores_all(
            compute_image_scores, all_sources, metric_conf, eval_batch, eval_device, nms_iou, max_det
        )
        run_label_count = int(labels_base.size)
        if labels_base.size:
            recall, precision, fpr = compute_threshold_metrics(labels_base, scores_base, curve_vals)
            auroc = compute_auc(fpr, recall)
            ap = compute_ap(recall, precision)
            run_auroc = float(auroc)
            run_ap = float(ap)
            save_metric_plots(metrics_dir, tag, curve_vals, recall, precision, fpr, plot_cfg=plot_cfg)
            _save_roc_curve_csv(metrics_dir, tag, curve_vals, recall, precision, fpr)
            _save_roc_keypoints_md(metrics_dir, tag, curve_vals, recall, fpr)
            with open(metrics_dir / f"{tag}_summary.txt", "w", encoding="utf-8") as f:
                f.write(f"image_auroc: {auroc:.6f}\n")
                f.write(f"image_ap: {ap:.6f}\n")
        else:
            _save_roc_curve_csv(
                metrics_dir,
                tag,
                np.array([], dtype=np.float32),
                np.array([], dtype=np.float32),
                np.array([], dtype=np.float32),
                np.array([], dtype=np.float32),
            )
            _save_roc_keypoints_md(
                metrics_dir,
                tag,
                np.array([0.0], dtype=np.float32),
                np.array([0.0], dtype=np.float32),
                np.array([0.0], dtype=np.float32),
            )

        if mode in {"train_test", "finetune_test"}:
            best_w = exp_dir / "train" / "weights" / "best.pt"
            last_w = exp_dir / "train" / "weights" / "last.pt"
            eval_w = Path(str(audit_state.get("eval_weights", ""))) if str(audit_state.get("eval_weights", "")).strip() else None
            if best_w.exists() and last_w.exists() and eval_w is not None:
                try:
                    best_is_eval = eval_w.resolve() == best_w.resolve()
                except Exception:
                    best_is_eval = False
                if best_is_eval:
                    model_last = None
                    try:
                        model_last = YOLO(str(last_w))
                        model_last = _apply_enhance241_patches(model_last, cfg, stage="eval_last", audit_state=audit_state)
                        labels_last, scores_last = _compute_scores_all(
                            compute_image_scores,
                            all_sources,
                            metric_conf,
                            eval_batch,
                            eval_device,
                            nms_iou,
                            max_det,
                            model_obj=model_last,
                        )
                        if labels_last.size:
                            rec_last, pre_last, fpr_last = compute_threshold_metrics(labels_last, scores_last, curve_vals)
                            auroc_last = compute_auc(fpr_last, rec_last)
                            ap_last = compute_ap(rec_last, pre_last)

                            def _kp(fprs: np.ndarray, recs: np.ndarray, target: float) -> Dict[str, float]:
                                idx = int(np.argmin(np.abs(fprs - float(target))))
                                return {
                                    "target_fpr": float(target),
                                    "nearest_fpr": float(fprs[idx]),
                                    "recall": float(recs[idx]),
                                    "threshold": float(curve_vals[idx]),
                                }

                            run_best_last_compare = {
                                "best_weight": str(best_w),
                                "last_weight": str(last_w),
                                "best": {
                                    "auroc": float(run_auroc),
                                    "ap": float(run_ap),
                                },
                                "last": {
                                    "auroc": float(auroc_last),
                                    "ap": float(ap_last),
                                },
                                "delta_last_minus_best": {
                                    "auroc": float(auroc_last - run_auroc),
                                    "ap": float(ap_last - run_ap),
                                },
                                "best_keypoints": [_kp(fpr, recall, t) for t in (0.05, 0.10, 0.30)] if labels_base.size else [],
                                "last_keypoints": [_kp(fpr_last, rec_last, t) for t in (0.05, 0.10, 0.30)],
                            }
                        else:
                            run_best_last_compare = {
                                "best_weight": str(best_w),
                                "last_weight": str(last_w),
                                "note": "last weight exists but produced empty score set",
                            }
                    except Exception as exc:
                        run_best_last_compare = {"error": f"best_last_compare_failed: {exc}"}
                    finally:
                        try:
                            if model_last is not None:
                                del model_last
                        except Exception:
                            pass
                        try:
                            import torch

                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                        except Exception:
                            pass
                        gc.collect()

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
                axis_cfg=curve_axis_cfg,
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
                axis_cfg=curve_axis_cfg,
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
                    axis_cfg=roc_axis_cfg,
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
                    axis_cfg=pr_axis_cfg,
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
                    axis_cfg=curve_axis_cfg,
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
                    axis_cfg=curve_axis_cfg,
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

    # A4 diagnostics: fixed assertion point (iou=0.2, threshold=0.3)
    a4_diag: Dict[str, Any] = {}
    if all_sources:
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        gc.collect()

        diag_iou = 0.2
        diag_thr = 0.3
        labels_diag, scores_diag = _compute_scores_all(
            compute_image_scores_iou,
            all_sources,
            metric_conf,
            float(diag_iou),
            eval_batch,
            eval_device,
            nms_iou,
            max_det,
        )
        diag_items: List[Dict[str, object]] = []
        for src in all_sources:
            try:
                diag_items.extend(
                    compute_image_level_results(
                        model,
                        src,
                        conf_threshold=float(diag_thr),
                        iou_match=float(diag_iou),
                        metric_conf=metric_conf,
                        batch=eval_batch,
                        device=eval_device,
                        nms_iou=nms_iou,
                        max_det=max_det,
                        split="diag",
                        vis_root=None,
                        save_visuals=False,
                    )
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
                diag_items.extend(
                    compute_image_level_results(
                        model,
                        src,
                        conf_threshold=float(diag_thr),
                        iou_match=float(diag_iou),
                        metric_conf=metric_conf,
                        batch=1,
                        device="cpu",
                        nms_iou=nms_iou,
                        max_det=max_det,
                        split="diag",
                        vis_root=None,
                        save_visuals=False,
                    )
                )

        first10 = []
        for item in diag_items[:10]:
            first10.append(
                {
                    "image": str(Path(str(item.get("image", ""))).name),
                    "has_gt": bool(item.get("has_gt", False)),
                    "num_preds": int(item.get("num_preds", 0)),
                    "max_score": float(item.get("max_conf", 0.0)),
                    "final_score": float(item.get("max_conf", 0.0)),
                }
            )

        labels_count = int(labels_diag.size)
        positives = int(labels_diag.sum()) if labels_count else 0
        pos_ratio = float(positives / labels_count) if labels_count else 0.0
        if labels_count:
            quantiles = np.quantile(scores_diag, [0.5, 0.9, 0.99]).tolist()
        else:
            quantiles = [0.0, 0.0, 0.0]

        preds_diag = (scores_diag >= float(diag_thr)).astype(np.int32) if labels_count else np.array([], dtype=np.int32)
        tp = int(((preds_diag == 1) & (labels_diag == 1)).sum()) if labels_count else 0
        fp = int(((preds_diag == 1) & (labels_diag == 0)).sum()) if labels_count else 0
        fn = int(((preds_diag == 0) & (labels_diag == 1)).sum()) if labels_count else 0
        tn = int(((preds_diag == 0) & (labels_diag == 0)).sum()) if labels_count else 0

        outcome_counts = {"TP": 0, "FP": 0, "FN": 0, "TN": 0}
        for item in diag_items:
            outcome = str(item.get("outcome", ""))
            if outcome in outcome_counts:
                outcome_counts[outcome] += 1
        confusion_consistent = (
            outcome_counts["TP"] == tp
            and outcome_counts["FP"] == fp
            and outcome_counts["FN"] == fn
            and outcome_counts["TN"] == tn
        )

        a4_diag = {
            "assert_1_first10_samples": first10,
            "assert_2_distribution": {
                "num_samples": labels_count,
                "positive_ratio": pos_ratio,
                "score_quantiles_p50_p90_p99": [float(quantiles[0]), float(quantiles[1]), float(quantiles[2])],
                "labels_all_constant": bool(labels_count > 0 and (labels_diag.max() == labels_diag.min())),
                "scores_all_constant": bool(labels_count > 0 and float(scores_diag.max() - scores_diag.min()) < 1e-12),
            },
            "assert_3_confusion_at_iou0.2_thr0.3": {
                "from_scores": {"TP": tp, "FP": fp, "FN": fn, "TN": tn},
                "from_image_report": outcome_counts,
                "consistent": bool(confusion_consistent),
            },
        }
    audit_state["a4_diag"] = a4_diag

    try:
        _write_result_summary(
            exp_dir,
            mode=str(mode),
            eval_weights=str(audit_state.get("eval_weights", "")),
            auroc=float(run_auroc),
            ap=float(run_ap),
            labels_count=int(run_label_count),
            sweep_cfg=dict(audit_state.get("threshold_sweep", {})),
            best_last_compare=run_best_last_compare,
        )
    except Exception:
        pass

    # Keep weights only under exp_dir/train/weights for baseline workflow.


if __name__ == "__main__":
    main()
