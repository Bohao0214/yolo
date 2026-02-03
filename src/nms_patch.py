from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Dict

import time

import torch


def _resolve_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "max_nms": int(cfg.get("metric_pre_nms_topk", 5000)),
        "max_nms_fallback": int(cfg.get("metric_pre_nms_topk_fallback", 2000)),
        "max_time_img": float(cfg.get("metric_nms_time_limit", 0.05)),
    }


def _apply_nms(
    prediction,
    conf_thres: float,
    iou_thres: float,
    classes,
    agnostic: bool,
    max_det: int,
    nc: int,
    rotated: bool,
    end2end: bool,
    return_idxs: bool,
    max_nms: int,
    max_nms_fallback: int,
    max_time_img: float,
):
    from ultralytics.utils import nms as ul_nms
    from ultralytics.utils.ops import xywh2xyxy
    from ultralytics.utils.metrics import batch_probiou, box_iou

    if isinstance(prediction, (list, tuple)):
        prediction = prediction[0]
    if classes is not None:
        classes = torch.tensor(classes, device=prediction.device)

    if prediction.shape[-1] == 6 or end2end:
        output = [pred[pred[:, 4] > conf_thres][:max_det] for pred in prediction]
        if classes is not None:
            output = [pred[(pred[:, 5:6] == classes).any(1)] for pred in output]
        return output, []

    bs = prediction.shape[0]
    nc = nc or (prediction.shape[1] - 4)
    extra = prediction.shape[1] - nc - 4
    mi = 4 + nc
    xc = prediction[:, 4:mi].amax(1) > conf_thres
    xinds = torch.arange(prediction.shape[-1], device=prediction.device).expand(bs, -1)[..., None]

    prediction = prediction.transpose(-1, -2)
    if not rotated:
        prediction[..., :4] = xywh2xyxy(prediction[..., :4])

    output = [torch.zeros((0, 6 + extra), device=prediction.device)] * bs
    keepi = [torch.zeros((0, 1), device=prediction.device)] * bs
    stats = []

    for xi, (x, xk) in enumerate(zip(prediction, xinds)):
        start = time.time()
        filt = xc[xi]
        x = x[filt]
        if return_idxs:
            xk = xk[filt]

        if not x.shape[0]:
            stats.append({
                "n_pre_nms": 0,
                "n_after_topk": 0,
                "n_post_nms": 0,
                "n_final": 0,
                "nms_timeout": False,
            })
            continue

        box, cls, mask = x.split((4, nc, extra), 1)
        conf, j = cls.max(1, keepdim=True)
        filt = conf.view(-1) > conf_thres
        x = torch.cat((box, conf, j.float(), mask), 1)[filt]
        if return_idxs:
            xk = xk[filt]

        if classes is not None and x.shape[0]:
            filt = (x[:, 5:6] == classes).any(1)
            x = x[filt]
            if return_idxs:
                xk = xk[filt]

        n_pre = int(x.shape[0])
        if not n_pre:
            stats.append({
                "n_pre_nms": 0,
                "n_after_topk": 0,
                "n_post_nms": 0,
                "n_final": 0,
                "nms_timeout": False,
            })
            continue

        # Pre-NMS TopK
        if n_pre > max_nms:
            filt = x[:, 4].argsort(descending=True)[:max_nms]
            x = x[filt]
            if return_idxs:
                xk = xk[filt]
        n_after = int(x.shape[0])

        c = x[:, 5:6] * (0 if agnostic else 7680)
        scores = x[:, 4]
        if rotated:
            boxes = torch.cat((x[:, :2] + c, x[:, 2:4], x[:, -1:]), dim=-1)
            i = ul_nms.TorchNMS.fast_nms(boxes, scores, iou_thres, iou_func=batch_probiou)
        else:
            boxes = x[:, :4] + c
            if "torchvision" in torch.sys.modules:
                import torchvision

                i = torchvision.ops.nms(boxes, scores, iou_thres)
            else:
                i = ul_nms.TorchNMS.nms(boxes, scores, iou_thres)

        if max_nms_fallback > 0 and (time.time() - start) > max_time_img and max_nms_fallback < max_nms:
            # Retry with smaller topk
            filt = x[:, 4].argsort(descending=True)[:max_nms_fallback]
            x = x[filt]
            if return_idxs:
                xk = xk[filt]
            n_after = int(x.shape[0])
            c = x[:, 5:6] * (0 if agnostic else 7680)
            scores = x[:, 4]
            if rotated:
                boxes = torch.cat((x[:, :2] + c, x[:, 2:4], x[:, -1:]), dim=-1)
                i = ul_nms.TorchNMS.fast_nms(boxes, scores, iou_thres, iou_func=batch_probiou)
            else:
                boxes = x[:, :4] + c
                if "torchvision" in torch.sys.modules:
                    import torchvision

                    i = torchvision.ops.nms(boxes, scores, iou_thres)
                else:
                    i = ul_nms.TorchNMS.nms(boxes, scores, iou_thres)

        n_post = int(i.shape[0])
        i = i[:max_det]
        n_final = int(i.shape[0])
        output[xi] = x[i]
        if return_idxs:
            keepi[xi] = xk[i].view(-1)

        stats.append({
            "n_pre_nms": n_pre,
            "n_after_topk": n_after,
            "n_post_nms": n_post,
            "n_final": n_final,
            "nms_timeout": (time.time() - start) > max_time_img,
        })

    return (output, keepi) if return_idxs else output, stats


@contextmanager
def patch_nms(cfg: Dict[str, Any]):
    from ultralytics.utils import nms as ul_nms

    orig = ul_nms.non_max_suppression
    opts = _resolve_cfg(cfg)

    def wrapped(
        prediction,
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
        classes=None,
        agnostic: bool = False,
        multi_label: bool = False,
        labels=(),
        max_det: int = 300,
        nc: int = 0,
        max_time_img: float = 0.05,
        max_nms: int = 30000,
        max_wh: int = 7680,
        rotated: bool = False,
        end2end: bool = False,
        return_idxs: bool = False,
    ):
        max_time = opts.get("max_time_img", max_time_img)
        max_nms_val = opts.get("max_nms", max_nms)
        max_nms_fb = opts.get("max_nms_fallback", 0)
        out, stats = _apply_nms(
            prediction,
            conf_thres,
            iou_thres,
            classes,
            agnostic,
            max_det,
            nc,
            rotated,
            end2end,
            return_idxs,
            max_nms_val,
            max_nms_fb,
            max_time,
        )
        ul_nms._nms_stats = stats
        return out

    ul_nms.non_max_suppression = wrapped
    try:
        yield
    finally:
        ul_nms.non_max_suppression = orig
