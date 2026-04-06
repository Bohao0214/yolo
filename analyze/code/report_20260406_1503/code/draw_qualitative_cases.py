#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

SCRIPT_PATH = Path(__file__).resolve()
REPORT_ROOT = SCRIPT_PATH.parents[1]
PROJECT_ROOT = SCRIPT_PATH.parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPT_PATH.parent) not in sys.path:
    sys.path.insert(0, str(SCRIPT_PATH.parent))

from tools.eval_detection_benchmark import Det, iou_one_to_many, load_ground_truth, load_pred_json
from utils_common import (
    _greedy_match_for_image,
    append_log,
    draw_boxes,
    draw_gt,
    ensure_dir,
    image_highlight_ratio,
    stack_h,
    write_csv,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Draw qualitative comparison cases from existing predictions.")
    p.add_argument("--report-root", type=Path, default=REPORT_ROOT)
    p.add_argument("--max-per-cat", type=int, default=5)
    p.add_argument("--highlight-thr", type=float, default=0.22)
    p.add_argument("--dup-iou", type=float, default=0.7)
    p.add_argument("--score-thr", type=float, default=None)
    return p.parse_args()


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _choose_models(compare_rows: List[dict], metadata: dict):
    baseline = metadata.get("baseline_model")
    best = metadata.get("best_model")

    ok_rows = [r for r in compare_rows if str(r.get("status", "")) == "ok"]
    if not ok_rows:
        return None, None

    if not baseline:
        baseline = ok_rows[0].get("model_name")
    if not best:
        best = sorted(ok_rows, key=lambda r: float(r.get("map50", 0.0) or 0.0), reverse=True)[0].get("model_name")
    return baseline, best


def _title(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 24), (0, 0, 0), -1)
    cv2.putText(out, text, (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def _gt_min_side(gts: List[Det]) -> float:
    if not gts:
        return 1e9
    vals = []
    for g in gts:
        vals.append(min(max(0.0, g.box[2] - g.box[0]), max(0.0, g.box[3] - g.box[1])))
    return float(min(vals)) if vals else 1e9


def _matched_gt_set(gts: List[Det], preds: List[Det], score_thr: float, iou_thr: float) -> set:
    matches, _, _ = _greedy_match_for_image(gts, preds, score_thr=score_thr, iou_thr=iou_thr)
    return {g for g, _ in matches}


def _unmatched_pred_count(gts: List[Det], preds: List[Det], score_thr: float, iou_thr: float) -> int:
    _, _, pred_unmatched = _greedy_match_for_image(gts, preds, score_thr=score_thr, iou_thr=iou_thr)
    cnt = 0
    for i in pred_unmatched:
        if preds[i].score >= score_thr:
            cnt += 1
    return cnt


def _has_duplicate(preds: List[Det], score_thr: float, dup_iou: float) -> bool:
    keep = [p for p in preds if p.score >= score_thr]
    if len(keep) < 2:
        return False
    for i in range(len(keep)):
        for j in range(i + 1, len(keep)):
            if keep[i].label != keep[j].label:
                continue
            iou = float(iou_one_to_many(keep[i].box, np.asarray([keep[j].box], dtype=np.float32))[0])
            if iou >= dup_iou:
                return True
    return False


def _read_csv(path: Path) -> List[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _empty_summary_rows() -> List[dict]:
    return [
        {"category": "small_defect", "n_selected": 0, "n_candidates": 0},
        {"category": "medium_defect", "n_selected": 0, "n_candidates": 0},
        {"category": "highlight_interference", "n_selected": 0, "n_candidates": 0},
        {"category": "baseline_miss_best_hit", "n_selected": 0, "n_candidates": 0},
        {"category": "baseline_fp_best_suppress", "n_selected": 0, "n_candidates": 0},
        {"category": "duplicate_prediction", "n_selected": 0, "n_candidates": 0},
    ]


def main() -> None:
    args = parse_args()
    report_root = args.report_root.expanduser().resolve()
    tables_dir = report_root / "tables"
    figs_dir = report_root / "figures"
    log_path = report_root / "logs" / "draw_qualitative_cases.log"

    ensure_dir(figs_dir / "compare_cases")
    ensure_dir(figs_dir / "fp_cases")
    ensure_dir(figs_dir / "fn_cases")

    metadata = _read_json(tables_dir / "metadata.json")
    compare_rows = _read_csv(tables_dir / "compare_main.csv")

    baseline_model, best_model = _choose_models(compare_rows, metadata)
    if not baseline_model or not best_model:
        append_log(log_path, "no available baseline/best model; skip qualitative draw")
        write_csv(tables_dir / "qualitative_cases.csv", [], ["category", "image_path", "output_png"])
        write_csv(tables_dir / "qualitative_summary.csv", _empty_summary_rows(), ["category", "n_selected", "n_candidates"])
        print("[warn] no successful model rows, qualitative drawing skipped.")
        return

    dataset_root = Path(metadata.get("dataset_root", "")).expanduser().resolve() if metadata.get("dataset_root") else None
    split = metadata.get("split_used", metadata.get("split_requested", "test"))
    if dataset_root is None or not dataset_root.exists():
        append_log(log_path, "dataset_root unresolved in metadata; skip qualitative draw")
        write_csv(tables_dir / "qualitative_cases.csv", [], ["category", "image_path", "output_png"])
        write_csv(tables_dir / "qualitative_summary.csv", _empty_summary_rows(), ["category", "n_selected", "n_candidates"])
        print("[warn] dataset root unresolved, qualitative drawing skipped.")
        return

    try:
        image_paths, gt_map, _ = load_ground_truth(dataset_root=dataset_root, split=split)
    except Exception as e:
        append_log(log_path, f"load_ground_truth failed: {e}")
        write_csv(tables_dir / "qualitative_cases.csv", [], ["category", "image_path", "output_png"])
        write_csv(tables_dir / "qualitative_summary.csv", _empty_summary_rows(), ["category", "n_selected", "n_candidates"])
        print("[warn] load GT failed, qualitative drawing skipped.")
        return

    score_thr = float(args.score_thr) if args.score_thr is not None else float(metadata.get("eval_params", {}).get("score_thr", 0.25))
    obj_iou = float(metadata.get("eval_params", {}).get("obj_iou", 0.5))

    base_pred_path = report_root / "raw_preds" / baseline_model / f"preds_{split}.json"
    best_pred_path = report_root / "raw_preds" / best_model / f"preds_{split}.json"
    if not base_pred_path.exists() or not best_pred_path.exists():
        append_log(log_path, f"missing pred json: base={base_pred_path.exists()} best={best_pred_path.exists()}")
        write_csv(tables_dir / "qualitative_cases.csv", [], ["category", "image_path", "output_png"])
        write_csv(tables_dir / "qualitative_summary.csv", _empty_summary_rows(), ["category", "n_selected", "n_candidates"])
        print("[warn] prediction json missing, qualitative drawing skipped.")
        return

    base_map = load_pred_json(base_pred_path, known_image_paths=image_paths)
    best_map = load_pred_json(best_pred_path, known_image_paths=image_paths)

    categories = OrderedDict(
        [
            ("small_defect", []),
            ("medium_defect", []),
            ("highlight_interference", []),
            ("baseline_miss_best_hit", []),
            ("baseline_fp_best_suppress", []),
            ("duplicate_prediction", []),
        ]
    )

    for ip in image_paths:
        key = str(ip.resolve())
        gts = gt_map.get(key, [])
        pb = base_map.get(key, [])
        pi = best_map.get(key, [])

        # Scale categories
        ms = _gt_min_side(gts)
        if ms < 16.0:
            categories["small_defect"].append(key)
        if 16.0 <= ms < 64.0:
            categories["medium_defect"].append(key)

        # Highlight category
        img = cv2.imread(key)
        if img is not None and image_highlight_ratio(img) >= float(args.highlight_thr):
            categories["highlight_interference"].append(key)

        # Baseline miss -> improved hit
        m_b = _matched_gt_set(gts, pb, score_thr=score_thr, iou_thr=obj_iou)
        m_i = _matched_gt_set(gts, pi, score_thr=score_thr, iou_thr=obj_iou)
        if any((gi not in m_b) and (gi in m_i) for gi in range(len(gts))):
            categories["baseline_miss_best_hit"].append(key)

        # Baseline FP suppressed
        fp_b = _unmatched_pred_count(gts, pb, score_thr=score_thr, iou_thr=obj_iou)
        fp_i = _unmatched_pred_count(gts, pi, score_thr=score_thr, iou_thr=obj_iou)
        if fp_b > fp_i:
            categories["baseline_fp_best_suppress"].append(key)

        # Duplicate
        if _has_duplicate(pb, score_thr=score_thr, dup_iou=float(args.dup_iou)) or _has_duplicate(
            pi, score_thr=score_thr, dup_iou=float(args.dup_iou)
        ):
            categories["duplicate_prediction"].append(key)

    manifest_rows = []
    summary_rows = []

    for cat, imgs in categories.items():
        picked = imgs[: int(args.max_per_cat)]
        summary_rows.append({"category": cat, "n_selected": len(picked), "n_candidates": len(imgs)})

        if cat in {"baseline_miss_best_hit"}:
            out_dir = figs_dir / "fn_cases"
        elif cat in {"baseline_fp_best_suppress", "duplicate_prediction"}:
            out_dir = figs_dir / "fp_cases"
        else:
            out_dir = figs_dir / "compare_cases"
        ensure_dir(out_dir)

        for k, img_path in enumerate(picked, start=1):
            orig = cv2.imread(img_path)
            if orig is None:
                continue
            gts = gt_map.get(img_path, [])
            pb = base_map.get(img_path, [])
            pi = best_map.get(img_path, [])

            panel = stack_h(
                [
                    _title(orig, "ORIG"),
                    _title(draw_gt(orig, gts), "GT"),
                    _title(draw_boxes(orig, pb, score_thr=score_thr, color=(52, 152, 219), label_prefix="B"), f"Baseline: {baseline_model}"),
                    _title(draw_boxes(orig, pi, score_thr=score_thr, color=(231, 76, 60), label_prefix="I"), f"Improved: {best_model}"),
                ]
            )
            out_png = out_dir / f"{cat}_{k:03d}_{Path(img_path).stem}.png"
            cv2.imwrite(str(out_png), panel)
            manifest_rows.append(
                {
                    "category": cat,
                    "image_path": img_path,
                    "output_png": str(out_png),
                    "baseline_model": baseline_model,
                    "improved_model": best_model,
                }
            )

    write_csv(
        tables_dir / "qualitative_cases.csv",
        manifest_rows,
        ["category", "image_path", "output_png", "baseline_model", "improved_model"],
    )
    write_csv(tables_dir / "qualitative_summary.csv", summary_rows, ["category", "n_selected", "n_candidates"])

    append_log(log_path, f"drawn={len(manifest_rows)} baseline={baseline_model} improved={best_model}")
    print(f"[done] qualitative cases -> {tables_dir / 'qualitative_cases.csv'}")


if __name__ == "__main__":
    main()
