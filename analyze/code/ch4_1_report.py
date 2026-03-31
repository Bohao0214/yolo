#!/usr/bin/env python3
"""按论文 4.1.1~4.1.4 组织已有分析产物，输出表格与论述文档。"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
from pathlib import Path
from typing import Dict, List, Tuple

TYPE_ORDER = ["highlight", "edge", "texture_boundary", "particle", "other"]
TYPE_NAME_ZH = {
    "highlight": "高光/强反光",
    "edge": "边缘邻近",
    "texture_boundary": "纹理/阴影边界",
    "particle": "颗粒/小噪点",
    "other": "其他",
}
SCALE_ORDER = ["<16", "16-32", "32-64", ">64"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="按章节 4.1.1~4.1.4 重整分析文档")
    p.add_argument("--image_fn_report_dir", type=str, required=True, help="包含 image_level.csv + fn_diag_summary.csv")
    p.add_argument("--fp_split_report_dir", type=str, required=True, help="包含 p2_3_2a_fp_split_summary.csv")
    p.add_argument("--fp_type_report_dir", type=str, required=True, help="包含 fp_type_metrics.csv 或 p2_3_2b_fp_type_table.csv")
    p.add_argument("--out_root", type=str, default="/home/ubuntu/hpproject/yolo/analyze/result")
    p.add_argument("--report_name", type=str, default="")
    return p.parse_args()


def make_report_dir(out_root: Path, report_name: str) -> Path:
    if report_name:
        report_dir = out_root / report_name
        report_dir.mkdir(parents=True, exist_ok=False)
        return report_dir
    ts = dt.datetime.now().strftime("report_%y%m%d%H%M")
    base = out_root / f"{ts}_ch4_1"
    if not base.exists():
        base.mkdir(parents=True, exist_ok=False)
        return base
    idx = 1
    while True:
        cand = out_root / f"{ts}_ch4_1_{idx:02d}"
        if not cand.exists():
            cand.mkdir(parents=True, exist_ok=False)
            return cand
        idx += 1


def read_csv(path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        lines = [ln for ln in f if ln.strip() and not ln.lstrip().startswith("#")]
    if not lines:
        return rows
    reader = csv.DictReader(lines)
    for r in reader:
        rows.append({k: (v.strip() if isinstance(v, str) else v) for k, v in r.items()})
    return rows


def to_int(x: str) -> int:
    if x is None or x == "":
        return 0
    return int(float(x))


def to_float(x: str) -> float:
    if x is None or x == "":
        return 0.0
    return float(x)


def pct(num: int, den: int) -> float:
    return float(num) / float(den) if den > 0 else 0.0


def pick_row(rows: List[Dict[str, str]], **conds: str) -> Dict[str, str]:
    for r in rows:
        ok = True
        for k, v in conds.items():
            if str(r.get(k, "")) != str(v):
                ok = False
                break
        if ok:
            return r
    raise ValueError(f"missing row with {conds}")


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def load_type_stats(fp_type_report_dir: Path) -> Tuple[Dict[str, int], Dict[str, int]]:
    type_counts = {k: 0 for k in TYPE_ORDER}
    unmatched_type_counts = {k: 0 for k in TYPE_ORDER}

    type_metrics = fp_type_report_dir / "fp_type_metrics.csv"
    type_table = fp_type_report_dir / "p2_3_2b_fp_type_table.csv"

    if type_metrics.exists():
        rows = read_csv(type_metrics)
        for r in rows:
            raw_name = r.get("type", "")
            name = {"edge_bg": "edge", "speckle": "particle"}.get(raw_name, raw_name)
            if name in type_counts:
                type_counts[name] += to_int(r.get("count", "0"))

        cross = fp_type_report_dir / "fp_type_x_dup.csv"
        if cross.exists():
            cross_rows = read_csv(cross)
            for r in cross_rows:
                raw_name = r.get("type", "")
                name = {"edge_bg": "edge", "speckle": "particle"}.get(raw_name, raw_name)
                if name in unmatched_type_counts:
                    unmatched_type_counts[name] += to_int(r.get("unmatched", "0"))
    elif type_table.exists():
        rows = read_csv(type_table)
        all_row = pick_row(rows, source_name="all")
        for k in TYPE_ORDER:
            type_counts[k] = to_int(all_row.get(f"type_{k}", "0"))

        cross = fp_type_report_dir / "p2_3_2b_fp_type_cross.csv"
        if cross.exists():
            cross_rows = read_csv(cross)
            for r in cross_rows:
                if r.get("source_name") == "all" and r.get("fp_tag") == "unmatched":
                    name = r.get("type_name", "")
                    if name in unmatched_type_counts:
                        unmatched_type_counts[name] += to_int(r.get("count", "0"))
    else:
        raise FileNotFoundError("fp_type_report_dir 中未找到 fp_type_metrics.csv 或 p2_3_2b_fp_type_table.csv")

    return type_counts, unmatched_type_counts


def main() -> None:
    args = parse_args()
    image_fn_report_dir = Path(args.image_fn_report_dir)
    fp_split_report_dir = Path(args.fp_split_report_dir)
    fp_type_report_dir = Path(args.fp_type_report_dir)

    image_level_rows = read_csv(image_fn_report_dir / "image_level.csv")
    fn_diag_rows = read_csv(image_fn_report_dir / "fn_diag_summary.csv")
    fp_split_rows = read_csv(fp_split_report_dir / "p2_3_2a_fp_split_summary.csv")

    image_all = pick_row(image_level_rows, source_name="all")
    fn_overall = pick_row(fn_diag_rows, source_name="all", stat_type="overall", bucket="all")
    fp_all = pick_row(fp_split_rows, source_name="all")

    # Table 4-3
    image_total = to_int(image_all["image_total"])
    img_hit = to_int(image_all["img_gt1_pred1"])
    img_fn = to_int(image_all["img_gt1_pred0"])
    img_fp = to_int(image_all["img_gt0_pred1"])
    img_tn = to_int(image_all["img_gt0_pred0"])
    image_recall = to_float(image_all["image_recall"])
    image_fp_rate = to_float(image_all["image_fp_rate"])

    gt_total = to_int(fn_overall["GT"])
    tp_total = to_int(fn_overall["TP"])
    fn_total = to_int(fn_overall["FN"])
    fp_total = to_int(fp_all["FP_total"])
    obj_recall = to_float(fn_overall["Recall"])
    obj_precision = pct(tp_total, tp_total + fp_total)

    table_4_3 = [
        {
            "image_total": image_total,
            "img_hit": img_hit,
            "img_fn": img_fn,
            "img_fp": img_fp,
            "img_tn": img_tn,
            "image_recall": f"{image_recall:.6f}",
            "image_fp_rate": f"{image_fp_rate:.6f}",
            "GT": gt_total,
            "TP": tp_total,
            "FP": fp_total,
            "FN": fn_total,
            "obj_recall": f"{obj_recall:.6f}",
            "obj_precision": f"{obj_precision:.6f}",
        }
    ]

    # Table 4-4
    scale_rows = [r for r in fn_diag_rows if r.get("source_name") == "all" and r.get("stat_type") == "scale"]
    scale_rows = sorted(scale_rows, key=lambda r: SCALE_ORDER.index(r.get("bucket", "")) if r.get("bucket", "") in SCALE_ORDER else 99)
    table_4_4: List[Dict[str, object]] = []
    for r in scale_rows:
        table_4_4.append(
            {
                "scale_bucket": r.get("bucket", ""),
                "GT": to_int(r.get("GT", "0")),
                "TP": to_int(r.get("TP", "0")),
                "FN": to_int(r.get("FN", "0")),
                "recall": f"{to_float(r.get('Recall', '0')):.6f}",
            }
        )

    # Table 4-5
    fn_no = to_int(fn_overall["FN_no_resp"])
    fn_low = to_int(fn_overall["FN_low_score"])
    fn_reg = to_int(fn_overall["FN_reg_poor"])
    fn_post = to_int(fn_overall["FN_postproc"])
    table_4_5 = [
        {"diag_type": "无响应", "count": fn_no, "ratio": f"{pct(fn_no, fn_total):.6f}"},
        {"diag_type": "低分抑制", "count": fn_low, "ratio": f"{pct(fn_low, fn_total):.6f}"},
        {"diag_type": "回归/匹配不足", "count": fn_reg, "ratio": f"{pct(fn_reg, fn_total):.6f}"},
        {"diag_type": "后处理筛除", "count": fn_post, "ratio": f"{pct(fn_post, fn_total):.6f}"},
    ]

    # Table 4-6
    fp_unmatched = to_int(fp_all["FP_unmatched"])
    fp_pred_dup = to_int(fp_all["FP_pred_dup"])
    fp_both = to_int(fp_all.get("FP_both", "0"))
    table_4_6 = [
        {
            "fp_total": fp_total,
            "fp_unmatched": fp_unmatched,
            "fp_pred_dup": fp_pred_dup,
            "fp_both": fp_both,
            "ratio_unmatched": f"{pct(fp_unmatched, fp_total):.6f}",
            "ratio_pred_dup": f"{pct(fp_pred_dup, fp_total):.6f}",
            "ratio_both": f"{pct(fp_both, fp_total):.6f}",
        }
    ]

    # Table 4-7
    type_counts, unmatched_type_counts = load_type_stats(fp_type_report_dir)
    table_4_7: List[Dict[str, object]] = []
    for t in TYPE_ORDER:
        cnt = type_counts[t]
        table_4_7.append(
            {
                "type_name": t,
                "type_name_zh": TYPE_NAME_ZH[t],
                "count": cnt,
                "ratio_in_fp": f"{pct(cnt, fp_total):.6f}",
                "unmatched_count": unmatched_type_counts.get(t, 0),
                "ratio_in_unmatched": f"{pct(unmatched_type_counts.get(t, 0), fp_unmatched):.6f}",
            }
        )

    report_dir = make_report_dir(Path(args.out_root), args.report_name)
    write_csv(report_dir / "table_4_3_baseline_overall.csv", table_4_3, list(table_4_3[0].keys()))
    write_csv(report_dir / "table_4_4_scale_bucket.csv", table_4_4, list(table_4_4[0].keys()))
    write_csv(report_dir / "table_4_5_fn_mechanism.csv", table_4_5, list(table_4_5[0].keys()))
    write_csv(report_dir / "table_4_6_fp_split.csv", table_4_6, list(table_4_6[0].keys()))
    write_csv(report_dir / "table_4_7_fp_type.csv", table_4_7, list(table_4_7[0].keys()))

    tb_16 = next((r for r in table_4_4 if r["scale_bucket"] == "<16"), None)
    tb_16_32 = next((r for r in table_4_4 if r["scale_bucket"] == "16-32"), None)
    tb_32_64 = next((r for r in table_4_4 if r["scale_bucket"] == "32-64"), None)
    tb_64 = next((r for r in table_4_4 if r["scale_bucket"] == ">64"), None)

    tex_row = next((r for r in table_4_7 if r["type_name"] == "texture_boundary"), None)
    tex_cnt = int(tex_row["count"]) if tex_row else 0
    tex_ratio = float(tex_row["ratio_in_fp"]) if tex_row else 0.0
    tex_unm = int(tex_row["unmatched_count"]) if tex_row else 0
    tex_unm_ratio = float(tex_row["ratio_in_unmatched"]) if tex_row else 0.0

    md_lines: List[str] = []
    md_lines.append("# 第4章 4.1 基线模型误差分析（自动整理稿）")
    md_lines.append("")
    md_lines.append(f"- 生成时间：{dt.datetime.now().isoformat(timespec='seconds')}")
    md_lines.append(f"- 图像/漏检来源：`{image_fn_report_dir}`")
    md_lines.append(f"- 误报结构来源：`{fp_split_report_dir}`")
    md_lines.append(f"- 误报类型来源：`{fp_type_report_dir}`")
    md_lines.append("")

    md_lines.append("## 4.1.1 基线性能定量评估")
    md_lines.append(
        f"表4-3给出了基线模型总体检测结果：图像级命中{img_hit}张、漏检{img_fn}张、误报{img_fp}张、真阴性{img_tn}张，共{image_total}张；"
        f"图像级召回率为{image_recall:.3f}，图像级误报率为{image_fp_rate:.3f}。"
        f"目标级统计为 GT={gt_total}、TP={tp_total}、FP={fp_total}、FN={fn_total}，"
        f"目标级召回率为{obj_recall:.3f}，目标级精确率为{obj_precision:.3f}。"
    )
    md_lines.append("- 对应表格：`table_4_3_baseline_overall.csv`")
    md_lines.append("")

    md_lines.append("## 4.1.2 尺度断层主导的召回瓶颈")
    if tb_16 and tb_16_32 and tb_32_64 and tb_64:
        md_lines.append(
            f"按短边尺度分桶统计显示：s<16 召回 {int(tb_16['TP'])}/{int(tb_16['GT'])}={float(tb_16['recall']):.3f}；"
            f"16<=s<32 为 {int(tb_16_32['TP'])}/{int(tb_16_32['GT'])}={float(tb_16_32['recall']):.3f}；"
            f"32<=s<64 为 {int(tb_32_64['TP'])}/{int(tb_32_64['GT'])}={float(tb_32_64['recall']):.3f}；"
            f"s>=64 为 {int(tb_64['TP'])}/{int(tb_64['GT'])}={float(tb_64['recall']):.3f}。"
        )
    else:
        md_lines.append("尺度分桶数据不完整，请检查 fn_diag_summary.csv 是否含 all+scale 行。")
    md_lines.append("- 对应表格：`table_4_4_scale_bucket.csv`")
    md_lines.append("")

    md_lines.append("## 4.1.3 漏检成因的机制性拆解")
    md_lines.append(
        f"对 {fn_total} 个目标级漏检（FN）进行机制拆解：无响应 {fn_no} ({pct(fn_no, fn_total):.3f})，"
        f"低分抑制 {fn_low} ({pct(fn_low, fn_total):.3f})，回归/匹配不足 {fn_reg} ({pct(fn_reg, fn_total):.3f})，"
        f"后处理筛除 {fn_post} ({pct(fn_post, fn_total):.3f})。"
    )
    md_lines.append("- 对应表格：`table_4_5_fn_mechanism.csv`")
    md_lines.append("")

    md_lines.append("## 4.1.4 误报结构与外观捷径分析")
    md_lines.append(
        f"目标级误报共 {fp_total} 个，其中真实误判（unmatched）{fp_unmatched} ({pct(fp_unmatched, fp_total):.3f})，"
        f"预测冗余（pred_dup）{fp_pred_dup} ({pct(fp_pred_dup, fp_total):.3f})，both 为 {fp_both}。"
    )
    md_lines.append(
        f"类型化统计显示：纹理/阴影边界类为 {tex_cnt} ({tex_ratio:.3f})；在真实误判中该类为 {tex_unm} ({tex_unm_ratio:.3f})。"
    )
    md_lines.append("- 对应表格：`table_4_6_fp_split.csv`、`table_4_7_fp_type.csv`")
    md_lines.append("")
    md_lines.append("术语说明：IoU=交并比；NMS=非极大值抑制（重叠框去重）；GT=标注框；FP/TP/FN=误报/命中/漏检。")

    (report_dir / "chapter4_1_analysis.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"[DONE] report_dir: {report_dir}")


if __name__ == "__main__":
    main()
