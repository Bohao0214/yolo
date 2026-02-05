from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch


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


def _is_rank0() -> bool:
    try:
        from ultralytics.utils import RANK  # type: ignore

        return int(RANK) in {-1, 0}
    except Exception:
        return True


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _logit(p: float) -> float:
    p = float(p)
    p = max(1e-6, min(p, 1 - 1e-6))
    return float(math.log(p / (1 - p)))


class _DWSeparableConv(torch.nn.Module):
    """Depthwise-separable 3x3 + pointwise 1x1 (no BatchNorm/BN; batch<8 friendly)."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.dw = torch.nn.Conv2d(
            channels, channels, kernel_size=3, stride=1, padding=1, groups=channels, bias=True
        )
        self.act1 = torch.nn.SiLU(inplace=True)
        self.pw = torch.nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.act2 = torch.nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act2(self.pw(self.act1(self.dw(x))))


class P3BiFPNLiteFuse(torch.nn.Module):
    """2.4.1 b1_v2 (recall-focused): BiFPN-lite fusion + detail boost + non-zero gated residual.

    Only patches the neck P4->P3 fusion point.

    Key ideas:
      - BiFPN=Bi-directional Feature Pyramid Network 的归一化加权融合（稳定，避免一路压死另一路）:
          P3_fuse = (w3*P3 + w4*P4_up) / (w3+w4+eps)
      - Local detail boost（HF=high-frequency 高频残差）帮助小目标更易起响应:
          hf = P3 - blur(P3)
          x  = P3_fuse + hf_gain * hf
      - Gated residual that is *non-equivalent from step 0* (forces early non-zero diff):
          P3_out = P3 + gate * Conv(x)
        where gate = sigmoid(g) (per-channel), initialized to ~gate_init (default 0.10).
      - Recall bias for fusion: initialize w3>w4 and optionally warm up w4 over early steps.

    Output keeps concat-compatible shape:
      cat([P4_up, P3_out]) -> 2C channels (downstream neck unchanged).
    """

    def __init__(
        self,
        channels_per_branch: int,
        eps: float = 1e-4,
        upsample_mode: str = "nearest",
        gate_init: float = 0.10,
        w3_init: float = 1.00,
        w4_init: float = 0.50,
        w4_warmup_steps: int = 1000,
        w4_warmup_start: float = 0.20,
        hf_gain_init: float = 0.50,
        mon_every: int = 100,
    ) -> None:
        super().__init__()
        self.c = int(channels_per_branch)
        if self.c <= 0:
            raise ValueError(f"channels_per_branch must be positive, got: {channels_per_branch}")
        self.eps = float(eps)
        self.upsample_mode = str(upsample_mode).lower()
        self.mon_every = int(mon_every)
        self.w4_warmup_steps = int(max(0, w4_warmup_steps))
        self.w4_warmup_start = float(max(0.0, min(1.0, w4_warmup_start)))

        # BiFPN normalized weights (per-channel, non-negative -> normalized)
        w_init = torch.empty(2, self.c, 1, 1, dtype=torch.float32)
        w_init[0].fill_(float(w3_init))
        w_init[1].fill_(float(w4_init))
        self.w = torch.nn.Parameter(w_init)

        # Lightweight refinement on fused feature (depthwise-separable; no BN)
        self.refine = _DWSeparableConv(self.c)

        # Per-channel gate (sigmoid), init to small positive to ensure early non-zero diff.
        self.g = torch.nn.Parameter(torch.full((1, self.c, 1, 1), _logit(float(gate_init)), dtype=torch.float32))

        # High-frequency gain (sigmoid), init to keep P3 details while avoiding over-sharpening.
        self.hf_gain = torch.nn.Parameter(
            torch.full((1, self.c, 1, 1), _logit(float(hf_gain_init)), dtype=torch.float32)
        )

        self._step: int = 0
        self._last_mon: Optional[Dict[str, float]] = None
        self._last_mon_step: int = -1

    def pop_monitor(self) -> Optional[Dict[str, float]]:
        row = self._last_mon
        self._last_mon = None
        return row

    def _w4_factor(self) -> float:
        if not (self.training and torch.is_grad_enabled()) or self.w4_warmup_steps <= 0:
            return 1.0
        t = min(1.0, float(self._step) / float(self.w4_warmup_steps))
        return self.w4_warmup_start + (1.0 - self.w4_warmup_start) * t

    def _norm_w(self) -> torch.Tensor:
        w_pos = torch.nn.functional.softplus(self.w)  # non-negative, smooth gradients
        w3 = w_pos[0]
        w4 = w_pos[1] * float(self._w4_factor())
        denom = w3 + w4 + self.eps
        return torch.stack((w3 / denom, w4 / denom), dim=0)

    def _maybe_align(self, p4_up: torch.Tensor, p3: torch.Tensor) -> torch.Tensor:
        if p4_up.shape[-2:] == p3.shape[-2:]:
            return p4_up
        if self.upsample_mode == "bilinear":
            return torch.nn.functional.interpolate(p4_up, size=p3.shape[-2:], mode="bilinear", align_corners=False)
        return torch.nn.functional.interpolate(p4_up, size=p3.shape[-2:], mode="nearest")

    @staticmethod
    def _p95_abs(diff_abs: torch.Tensor, max_samples: int = 200000) -> torch.Tensor:
        v = diff_abs.reshape(-1)
        if v.numel() > max_samples:
            # Deterministic down-sampling for reproducibility (avoid consuming RNG state).
            step = max(1, int(v.numel() // max_samples))
            v = v[::step][:max_samples]
        return torch.quantile(v, 0.95)

    def forward(self, x: Any) -> torch.Tensor:
        if not isinstance(x, (list, tuple)) or len(x) != 2:
            raise TypeError(
                f"P3BiFPNLiteFuse expects [p4_up, p3], got: {type(x)} len={getattr(x, '__len__', None)}"
            )
        p4_up, p3 = x  # expected order from original Concat(f=[-1, P3])
        p4_up = self._maybe_align(p4_up, p3)
        if p4_up.shape[1] != self.c or p3.shape[1] != self.c:
            raise ValueError(f"Channel mismatch: p4_up={p4_up.shape} p3={p3.shape} expected C={self.c}")

        w_norm = self._norm_w()
        p3_fuse = w_norm[0] * p3 + w_norm[1] * p4_up
        blur = torch.nn.functional.avg_pool2d(p3, kernel_size=3, stride=1, padding=1)
        hf = p3 - blur
        x_in = p3_fuse + torch.sigmoid(self.hf_gain) * hf
        delta = self.refine(x_in)
        gate = torch.sigmoid(self.g)  # non-zero init => non-equivalent from step 0; per-channel
        p3_out = p3 + gate * delta

        # Lightweight monitoring: cache stats, flush by callbacks (no file I/O in forward).
        if self.training and torch.is_grad_enabled():
            if self.mon_every > 0 and self._step % self.mon_every == 0 and self._last_mon_step != self._step:
                with torch.no_grad():
                    diff_abs = (p3_out - p3).abs()
                    row = {
                        "step": float(self._step),
                        "w3_mean": float(w_norm[0].mean().detach().cpu()),
                        "w4_mean": float(w_norm[1].mean().detach().cpu()),
                        "g_mean": float(self.g.mean().detach().cpu()),
                        "g_min": float(self.g.min().detach().cpu()),
                        "g_max": float(self.g.max().detach().cpu()),
                        "gate_mean": float(gate.mean().detach().cpu()),
                        "gate_min": float(gate.min().detach().cpu()),
                        "gate_max": float(gate.max().detach().cpu()),
                        "diff_abs_mean": float(diff_abs.mean().detach().cpu()),
                        "diff_abs_p95": float(self._p95_abs(diff_abs).detach().cpu()),
                    }
                self._last_mon = row
                self._last_mon_step = self._step
            self._step += 1

        return torch.cat((p4_up, p3_out), dim=1)


def _locate_p4_to_p3_fuse(seq: Any) -> Tuple[int, Any]:
    """Return (fuse_idx, detect_layer) for the P4->P3 fusion point."""
    detect = None
    if isinstance(seq, (list, tuple)) and seq:
        detect = seq[-1]
    else:
        try:
            detect = seq[-1]
        except Exception:
            detect = None

    # Preferred: derive from Detect.f = [P3, P4, P5] output indices
    detect_f = getattr(detect, "f", None)
    if isinstance(detect_f, (list, tuple)) and detect_f:
        try:
            p3_out = int(detect_f[0])
            fuse_idx = p3_out - 1
            return fuse_idx, detect
        except Exception:
            pass

    # Fallback: find Concat with f=[-1, 4] (YOLO11 default P4->P3 fuse)
    for i, layer in enumerate(seq):
        if layer.__class__.__name__ == "Concat" and list(getattr(layer, "f", [])) == [-1, 4]:
            return i, detect
    raise RuntimeError("Unable to locate P4->P3 fusion point in model.")


def apply(model: Any, cfg: Any) -> Any:
    """Apply 2.4.1 b1 enhancement (small-scale box recall) to YOLO11 model.

    - If cfg.enhance241.b1 is false: return model unchanged.
    - If true: replace the neck P4->P3 fusion Concat with P3BiFPNLiteFuse (b1_v2).
    """

    enable_b1 = bool(_deep_get(cfg, "enhance241", "b1", default=False))
    if not enable_b1:
        return model

    # Support both ultralytics.YOLO and raw DetectionModel-like inputs.
    yolo_obj = model
    if hasattr(model, "model") and hasattr(getattr(model, "model"), "model"):
        # ultralytics.YOLO: model.model is DetectionModel, and model.model.model is the layer sequence
        det_model = getattr(model, "model")
        seq = getattr(det_model, "model", None)
    else:
        # DetectionModel-like: model.model is the layer sequence
        det_model = model
        seq = getattr(det_model, "model", None)
    if seq is None:
        raise RuntimeError("enhance241.b1 requires an ultralytics YOLO/DetectionModel-like object with a .model sequence.")

    fuse_idx, _ = _locate_p4_to_p3_fuse(seq)
    if fuse_idx < 0 or fuse_idx >= len(seq):
        raise RuntimeError(f"Invalid fuse_idx={fuse_idx} for model length={len(seq)}")

    old = seq[fuse_idx]
    if isinstance(old, P3BiFPNLiteFuse):
        print(f"[enhance241] b1 enabled: already patched at model.model[{fuse_idx}] -> P3BiFPNLiteFuse(C={old.c})")
        return yolo_obj

    if old.__class__.__name__ != "Concat":
        raise RuntimeError(
            f"Expected Concat at P4->P3 fuse idx={fuse_idx}, got {old.__class__.__name__}. Refusing to patch."
        )

    next_layer = seq[fuse_idx + 1] if (fuse_idx + 1) < len(seq) else None
    c_in = None
    try:
        c_in = int(next_layer.cv1.conv.in_channels)  # type: ignore[attr-defined]
    except Exception:
        c_in = None
    if not c_in or c_in % 2 != 0:
        raise RuntimeError(f"Unable to infer concat channels from next layer at idx={fuse_idx+1}. c_in={c_in}")

    enhance_cfg = _deep_get(cfg, "enhance241", default={}) or {}
    if not isinstance(enhance_cfg, dict):
        enhance_cfg = {}
    upsample_mode = str(enhance_cfg.get("b1_upsample", "nearest")).lower()
    gate_init = _safe_float(enhance_cfg.get("b1_gate_init", 0.10), 0.10)
    w3_init = _safe_float(enhance_cfg.get("b1_w3_init", 1.00), 1.00)
    w4_init = _safe_float(enhance_cfg.get("b1_w4_init", 0.50), 0.50)
    w4_warmup_steps = int(_safe_float(enhance_cfg.get("b1_w4_warmup_steps", 1000), 1000))
    w4_warmup_start = _safe_float(enhance_cfg.get("b1_w4_warmup_start", 0.20), 0.20)
    hf_gain_init = _safe_float(enhance_cfg.get("b1_hf_gain_init", 0.50), 0.50)
    mon_every = int(_safe_float(enhance_cfg.get("b1_mon_every", 100), 100))

    c = c_in // 2
    fuse = P3BiFPNLiteFuse(
        channels_per_branch=c,
        eps=1e-4,
        upsample_mode=upsample_mode,
        gate_init=gate_init,
        w3_init=w3_init,
        w4_init=w4_init,
        w4_warmup_steps=w4_warmup_steps,
        w4_warmup_start=w4_warmup_start,
        hf_gain_init=hf_gain_init,
        mon_every=mon_every,
    )

    # Preserve Ultralytics layer meta attributes used by its forward graph.
    for attr in ("i", "f", "type"):
        if hasattr(old, attr):
            setattr(fuse, attr, getattr(old, attr))

    seq[fuse_idx] = fuse

    print(
        f"[enhance241] b1 enabled: patched neck P4->P3 fuse at model.model[{fuse_idx}] "
        f"(Concat f={getattr(old, 'f', None)}) -> P3BiFPNLiteFuse(C={c}, gate_init={gate_init}, up={upsample_mode})"
    )

    # Audit outputs via Ultralytics callbacks (no training script changes).
    if hasattr(yolo_obj, "add_callback") and not getattr(yolo_obj, "_enhance241_b1_callbacks", False):
        setattr(yolo_obj, "_enhance241_b1_callbacks", True)

        train_state: Dict[str, Any] = {"epoch": None, "epochs": None}
        conf_thr = _safe_float(cfg.get("conf", 0.30), 0.30)
        iou_match = _safe_float(cfg.get("match_iou", 0.20), 0.20)
        topk = int(_safe_float(enhance_cfg.get("b1_fn_topk", 20), 20))

        def _write_csv_row(path: Path, row: Dict[str, Any], fieldnames: List[str]) -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            exists = path.exists()
            with open(path, "a", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                if not exists:
                    w.writeheader()
                w.writerow(row)

        def on_train_start(trainer: Any) -> None:
            if not _is_rank0():
                return
            save_dir = Path(getattr(trainer, "save_dir", ""))
            if not save_dir:
                return
            train_state["epochs"] = getattr(trainer, "epochs", None)
            audit = {
                "name": "enhance241_b1_v2",
                "fuse_idx": int(fuse_idx),
                "channels": int(c),
                "gate_init": float(gate_init),
                "upsample_mode": str(upsample_mode),
                "w4_warmup_steps": int(w4_warmup_steps),
                "w4_warmup_start": float(w4_warmup_start),
                "conf_threshold": float(conf_thr),
                "iou_match": float(iou_match),
                "mon_every": int(mon_every),
            }
            (save_dir / "enhance241_b1_v2_audit.json").write_text(
                json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(f"[enhance241] b1 audit: {save_dir / 'enhance241_b1_v2_monitor.csv'}")

        def on_train_epoch_end(trainer: Any) -> None:
            train_state["epoch"] = getattr(trainer, "epoch", None)
            train_state["epochs"] = getattr(trainer, "epochs", train_state.get("epochs"))

        def on_train_batch_end(trainer: Any) -> None:
            if not _is_rank0():
                return
            save_dir = Path(getattr(trainer, "save_dir", ""))
            if not save_dir:
                return
            row = fuse.pop_monitor()
            if not row:
                return
            row_out = dict(row)
            row_out["epoch"] = train_state.get("epoch")
            fieldnames = [
                "epoch",
                "step",
                "w3_mean",
                "w4_mean",
                "g_mean",
                "g_min",
                "g_max",
                "gate_mean",
                "gate_min",
                "gate_max",
                "diff_abs_mean",
                "diff_abs_p95",
            ]
            _write_csv_row(save_dir / "enhance241_b1_v2_monitor.csv", row_out, fieldnames)
            print(
                f"[enhance241] b1 mon step={int(row_out['step'])} "
                f"w3={row_out['w3_mean']:.4f} w4={row_out['w4_mean']:.4f} "
                f"gate_mean={row_out['gate_mean']:.4f} |diff|_mean={row_out['diff_abs_mean']:.6f} "
                f"p95={row_out['diff_abs_p95']:.6f}"
            )

        def on_val_start(validator: Any) -> None:
            save_dir = Path(getattr(validator, "save_dir", ""))
            # Align postprocess params to project defaults when possible (keep enough boxes for recall).
            try:
                metric_conf = _safe_float(cfg.get("metric_conf", 0.01), 0.01)
                validator.args.conf = float(metric_conf)  # keep candidates before conf_threshold filtering below
                validator.args.iou = float(_safe_float(cfg.get("nms_iou", validator.args.iou), validator.args.iou))
                validator.args.max_det = int(_safe_float(cfg.get("max_det", validator.args.max_det), validator.args.max_det))
            except Exception:
                pass

            if not _is_rank0():
                return
            if not save_dir:
                return

            buckets = [(0, 8), (8, 16), (16, 32), (32, 64), (64, 10**9)]
            val_state: Dict[str, Any] = {
                "buckets": buckets,
                "rows": {f"{a}-{b if b < 10**9 else 'inf'}": {"gt": 0, "tp": 0, "fn": 0} for a, b in buckets},
                "candidates": [],
            }
            setattr(validator, "_enhance241_b1_v2_state", val_state)

            if getattr(validator, "_enhance241_b1_v2_wrapped", False):
                return
            setattr(validator, "_enhance241_b1_v2_wrapped", True)

            orig_update = validator.update_metrics

            def _wrap_update_metrics(preds: Any, batch: Any) -> None:
                val_state = getattr(validator, "_enhance241_b1_v2_state", None)
                if not isinstance(val_state, dict):
                    return orig_update(preds, batch)
                buckets = val_state.get("buckets") or []
                if not buckets:
                    return orig_update(preds, batch)
                try:
                    from ultralytics.utils import ops  # type: ignore
                    from ultralytics.utils.metrics import box_iou  # type: ignore
                except Exception:
                    return orig_update(preds, batch)

                # Use CPU tensors here to avoid extra GPU sync/stalls during validation.
                imgsz = batch["img"].shape[2:]  # (h, w)
                scale = torch.tensor([imgsz[1], imgsz[0], imgsz[1], imgsz[0]], device="cpu", dtype=torch.float32)
                conf_keep = float(conf_thr)
                iou_thr = float(iou_match)

                batch_idx = batch["batch_idx"].detach().cpu()
                batch_cls = batch["cls"].detach().cpu()
                batch_bboxes = batch["bboxes"].detach().cpu()

                for si, pred in enumerate(preds):
                    idx = batch_idx == si
                    gt_cls = batch_cls[idx].squeeze(-1)
                    gt_b = batch_bboxes[idx]
                    if gt_cls.numel() == 0:
                        continue
                    gt_xyxy = ops.xywh2xyxy(gt_b) * scale
                    w = (gt_xyxy[:, 2] - gt_xyxy[:, 0]).clamp(min=0)
                    h = (gt_xyxy[:, 3] - gt_xyxy[:, 1]).clamp(min=0)
                    size = torch.minimum(w, h)

                    pb = pred["bboxes"].detach().cpu()
                    pc = pred["conf"].detach().cpu()
                    keep = pc >= conf_keep
                    pb = pb[keep]
                    pc = pc[keep]

                    matched = torch.zeros((gt_xyxy.shape[0],), dtype=torch.bool, device="cpu")
                    if pb.numel():
                        iou = box_iou(gt_xyxy, pb)
                        best = iou.max(dim=1).values
                        matched = best >= iou_thr

                    # bucket update
                    for a, b in buckets:
                        key = f"{a}-{b if b < 10**9 else 'inf'}"
                        m = (size >= a) & (size < b)
                        if not m.any():
                            continue
                        gt_n = int(m.sum().item())
                        tp_n = int((m & matched).sum().item())
                        fn_n = gt_n - tp_n
                        val_state["rows"][key]["gt"] += gt_n
                        val_state["rows"][key]["tp"] += tp_n
                        val_state["rows"][key]["fn"] += fn_n

                    # collect small FN candidates (<32px)
                    small_fn = (~matched) & (size < 32)
                    if small_fn.any() and len(val_state["candidates"]) < max(200, topk * 20):
                        im_file = str(batch["im_file"][si])
                        max_conf = float(pc.max()) if pc.numel() else 0.0
                        min_size = float(size[small_fn].min())
                        # determine bucket label by min_size
                        bucket_label = "unknown"
                        for a, b in buckets:
                            if min_size >= a and min_size < b:
                                bucket_label = f"{a}-{b if b < 10**9 else 'inf'}"
                                break

                        # scale boxes back to original image for visualization
                        try:
                            ori_shape = batch["ori_shape"][si]
                            ratio_pad = batch["ratio_pad"][si]
                            gt_o = ops.scale_boxes(imgsz, gt_xyxy.clone(), ori_shape, ratio_pad).numpy()
                            fn_o = (
                                ops.scale_boxes(imgsz, gt_xyxy[small_fn].clone(), ori_shape, ratio_pad)
                                .numpy()
                            )
                            pb_o = (
                                ops.scale_boxes(imgsz, pb.clone(), ori_shape, ratio_pad).numpy()
                                if pb.numel()
                                else []
                            )
                            pc_o = pc.numpy().tolist() if pc.numel() else []
                        except Exception:
                            gt_o, fn_o, pb_o, pc_o = [], [], [], []

                        val_state["candidates"].append(
                            {
                                "im_file": im_file,
                                "bucket": bucket_label,
                                "min_size": float(min_size),
                                "nms": "unk",
                                "max_conf": float(max_conf),
                                "gt_xyxy": gt_o,
                                "fn_xyxy": fn_o,
                                "pred_xyxy": pb_o,
                                "pred_conf": pc_o,
                            }
                        )

                return orig_update(preds, batch)

            validator.update_metrics = _wrap_update_metrics  # type: ignore[assignment]

        def on_val_end(validator: Any) -> None:
            if not _is_rank0():
                return
            save_dir = Path(getattr(validator, "save_dir", ""))
            if not save_dir:
                return
            val_state = getattr(validator, "_enhance241_b1_v2_state", None)
            if not isinstance(val_state, dict):
                return

            epoch = train_state.get("epoch")
            epoch_s = f"{int(epoch):03d}" if isinstance(epoch, int) else "NA"

            # 3.2 Small-object recall slice table
            out_csv = save_dir / f"enhance241_b1_v2_small_recall_epoch{epoch_s}.csv"
            with open(out_csv, "w", encoding="utf-8", newline="") as f:
                w = csv.writer(f)
                w.writerow(["bucket", "gt_count", "tp_count", "fn_count", "recall", "conf_thr", "iou_match"])
                for key, r in val_state["rows"].items():
                    gt_n = int(r["gt"])
                    tp_n = int(r["tp"])
                    fn_n = int(r["fn"])
                    recall = float(tp_n / gt_n) if gt_n > 0 else 0.0
                    w.writerow([key, gt_n, tp_n, fn_n, f"{recall:.6f}", f"{conf_thr:.3f}", f"{iou_match:.3f}"])
            (save_dir / "enhance241_b1_v2_small_recall_latest.csv").write_text(
                out_csv.read_text(encoding="utf-8"), encoding="utf-8"
            )
            print(f"[enhance241] b1 val: wrote {out_csv}")

            # 3.3 TopK small FN case package (final epoch only by default)
            epochs = train_state.get("epochs")
            is_last = isinstance(epoch, int) and isinstance(epochs, int) and epoch == (epochs - 1)
            if not is_last:
                return

            candidates = list(val_state.get("candidates") or [])
            if not candidates:
                return
            candidates.sort(key=lambda d: (float(d.get("min_size", 1e9)), -float(d.get("max_conf", 0.0))))
            keep = candidates[: max(10, min(30, int(topk)))]

            out_dir = save_dir / f"enhance241_b1_v2_fn_topk_epoch{epoch_s}"
            out_dir.mkdir(parents=True, exist_ok=True)

            try:
                import cv2  # type: ignore
                import numpy as np
            except Exception:
                return

            def _draw(img, boxes, color, labels=None, thickness=2):
                if boxes is None:
                    return
                arr = np.array(boxes, dtype=np.float32)
                if arr.size == 0:
                    return
                for i, (x1, y1, x2, y2) in enumerate(arr.tolist()):
                    p1 = (int(round(x1)), int(round(y1)))
                    p2 = (int(round(x2)), int(round(y2)))
                    cv2.rectangle(img, p1, p2, color, thickness)
                    if labels and i < len(labels) and labels[i]:
                        cv2.putText(
                            img,
                            str(labels[i]),
                            (p1[0], max(0, p1[1] - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            color,
                            1,
                            cv2.LINE_AA,
                        )

            for item in keep:
                im_file = str(item.get("im_file", ""))
                img = cv2.imread(im_file)
                if img is None:
                    continue
                base = img.copy()
                _draw(base, item.get("gt_xyxy"), (0, 255, 0), labels=None, thickness=2)  # GT green
                _draw(base, item.get("fn_xyxy"), (0, 255, 255), labels=None, thickness=3)  # FN yellow
                pred_xyxy = item.get("pred_xyxy", [])
                pred_conf = item.get("pred_conf", [])
                if isinstance(pred_conf, np.ndarray):
                    pred_conf = pred_conf.tolist()
                pred_labels = [f"{float(c):.2f}" for c in pred_conf] if len(pred_conf) else None
                _draw(base, pred_xyxy, (0, 0, 255), labels=pred_labels, thickness=2)  # Pred red

                stem = Path(im_file).stem
                bucket = str(item.get("bucket", "unknown"))
                nms = str(item.get("nms", "unk"))
                mc = float(item.get("max_conf", 0.0))
                name = f"{stem}__bucket={bucket}__nms={nms}__maxconf={mc:.2f}.jpg"
                name = "".join(ch if ch.isalnum() or ch in "._-=+" else "_" for ch in name)
                cv2.imwrite(str(out_dir / name), base)

            print(f"[enhance241] b1 val: saved TopK FN cases to {out_dir}")

        yolo_obj.add_callback("on_train_start", on_train_start)
        yolo_obj.add_callback("on_train_epoch_end", on_train_epoch_end)
        yolo_obj.add_callback("on_train_batch_end", on_train_batch_end)
        yolo_obj.add_callback("on_val_start", on_val_start)
        yolo_obj.add_callback("on_val_end", on_val_end)

    return yolo_obj
