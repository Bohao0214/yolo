#!/usr/bin/env python3
"""
用法说明（汇总报告生成）：

读取 tables/*.csv 与 tables/*.json，生成 report.md（论文写作版）。
默认报告根目录为：
/home/ubuntu/hpproject/yolo/analyze/code/ch4tool

示例：
conda run -n yolo11 python /home/ubuntu/hpproject/yolo/analyze/code/ch4tool/code/build_report.py \
  --report-root /home/ubuntu/hpproject/yolo/analyze/code/ch4tool
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import List

SCRIPT_PATH = Path(__file__).resolve()
REPORT_ROOT = SCRIPT_PATH.parents[1]
PROJECT_ROOT = SCRIPT_PATH.parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPT_PATH.parent) not in sys.path:
    sys.path.insert(0, str(SCRIPT_PATH.parent))

from utils_common import markdown_table


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build markdown report from generated tables/figures")
    p.add_argument("--report-root", type=Path, default=REPORT_ROOT)
    return p.parse_args()


def _read_csv(path: Path) -> List[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _read_json(path: Path):
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt_pending(pending: List[str]) -> str:
    if not pending:
        return "无"
    return "、".join(sorted(set(str(x) for x in pending if str(x).strip())))


def main() -> None:
    args = parse_args()
    report_root = args.report_root.expanduser().resolve()
    tables = report_root / "tables"
    figures = report_root / "figures"

    compare = _read_csv(tables / "compare_main.csv")
    ablation = _read_csv(tables / "ablation_main.csv")
    scale = _read_csv(tables / "scale_recall.csv")
    fnm = _read_csv(tables / "fn_mechanism.csv")
    fps = _read_csv(tables / "fp_structure.csv")
    q_cases = _read_csv(tables / "qualitative_cases.csv")
    q_sum = _read_csv(tables / "qualitative_summary.csv")
    h_cases = _read_csv(tables / "heatmap_manifest.csv")
    h_method = _read_json(tables / "heatmap_method.json")
    meta = _read_json(tables / "metadata.json")

    models_txt = report_root / "models_to_eval.txt"
    model_lines = []
    if models_txt.exists():
        for ln in models_txt.read_text(encoding="utf-8", errors="ignore").splitlines():
            s = ln.strip()
            if not s or s.startswith("#"):
                continue
            p = Path(s).expanduser().resolve()
            model_lines.append({"model_path": str(p), "exists": "yes" if p.exists() else "no"})

    pending = list(meta.get("pending_user_inputs", []))
    # user requested these to be explicitly marked if unresolved
    required_user_inputs = [
        "dataset_root",
        "data_yaml",
        "eval_split(test/val)",
        "heatmap_sample_list(optional)",
    ]
    if not meta.get("dataset_root"):
        for k in ["dataset_root", "data_yaml", "eval_split(test/val)"]:
            if k not in pending:
                pending.append(k)
    if len(h_cases) == 0 and "heatmap_sample_list(optional)" not in pending:
        pending.append("heatmap_sample_list(optional)")

    ok_rows = [r for r in compare if str(r.get("status", "")) == "ok"]
    miss_rows = [r for r in compare if str(r.get("status", "")) != "ok"]

    missing_items = []
    if len(ok_rows) == 0:
        missing_items.append(
            {
                "item": "对比实验主指标",
                "status": "未稳定得到",
                "reason": "当前机器上模型权重或数据集路径不可用，无法完成统一推理评估。",
                "next": "补充可访问的模型绝对路径与统一 data.yaml/dataset_root 后重跑 run_eval_compare.py。",
            }
        )
    if len(scale) == 0:
        missing_items.append(
            {
                "item": "分尺度 Recall",
                "status": "未产出",
                "reason": "主评估未成功或未获得有效预测。",
                "next": "保证 compare_main.csv 中至少一个模型 status=ok。",
            }
        )
    if len(q_cases) == 0:
        missing_items.append(
            {
                "item": "典型样本对比图",
                "status": "未产出或不足",
                "reason": "缺少可用预测或测试集路径不可解析。",
                "next": "确认 test 集路径与 raw_preds 生成成功后重跑 draw_qualitative_cases.py。",
            }
        )
    if len(h_cases) == 0:
        missing_items.append(
            {
                "item": "热力图/CAM",
                "status": "未产出或不稳定",
                "reason": h_method.get("reason", "模型不可用或方法回退后仍失败"),
                "next": "提供可用模型并固定热力图样本清单，必要时安装 pytorch-grad-cam。",
            }
        )

    report_lines: List[str] = []
    report_lines.append("# 检测实验报告（YOLO系列）")
    report_lines.append("")
    report_lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"报告目录: `{report_root}`")
    report_lines.append("")

    report_lines.append("## 待用户补充")
    report_lines.append("")
    report_lines.append(f"- 待确认项: {_fmt_pending(pending)}")
    report_lines.append(f"- 模型路径填写文件: `{models_txt}`")
    report_lines.append(f"- 数据集覆盖配置: `{report_root / 'dataset_override.txt'}`")
    report_lines.append("")

    report_lines.append("## 1. 实验概况")
    report_lines.append("")
    report_lines.append("- 约束: 不改模型结构、不重训，仅做读取已有模型并统一评估。")
    report_lines.append("- 评估口径: 统一脚本、统一数据划分、统一参数；掩膜转最小外接框时比较的是检测能力而非分割精度。")
    report_lines.append(f"- 指标公式文档: `{report_root / '指标计算说明.md'}`")
    report_lines.append("")

    report_lines.append("## 2. 模型清单与路径")
    report_lines.append("")
    if model_lines:
        report_lines.append(markdown_table(model_lines, ["model_path", "exists"]))
    else:
        report_lines.append("- 未读取到模型路径。")
    report_lines.append("")

    report_lines.append("## 3. 数据集与评估参数")
    report_lines.append("")
    report_lines.append(f"- data.yaml: `{meta.get('data_yaml', '')}`")
    report_lines.append(f"- dataset_root: `{meta.get('dataset_root', '')}`")
    report_lines.append(f"- split(requested/used): `{meta.get('split_requested', '')}` / `{meta.get('split_used', '')}`")
    report_lines.append(f"- split备注: `{meta.get('split_note', '')}`")
    report_lines.append(f"- eval_params: `{json.dumps(meta.get('eval_params', {}), ensure_ascii=False)}`")
    eps = meta.get("eval_param_source", {})
    if eps:
        report_lines.append(f"- eval_param_source: `{json.dumps(eps.get('priority', ''), ensure_ascii=False)}`")
    report_lines.append("")

    report_lines.append("## 4. 对比实验结果")
    report_lines.append("")
    if compare:
        cmp_view = []
        for r in compare:
            cmp_view.append(
                {
                    "model": r.get("model_name", ""),
                    "status": r.get("status", ""),
                    "P": r.get("precision", ""),
                    "R": r.get("recall", ""),
                    "mAP@0.5": r.get("map50", ""),
                    "mAP@0.5:0.95": r.get("map50_95", ""),
                }
            )
        report_lines.append(markdown_table(cmp_view, ["model", "status", "P", "R", "mAP@0.5", "mAP@0.5:0.95"]))
    else:
        report_lines.append("- compare_main.csv 未生成有效行。")
    report_lines.append("")

    report_lines.append("## 5. 消融实验结果")
    report_lines.append("")
    if ablation:
        abl_view = []
        for r in ablation:
            abl_view.append(
                {
                    "variant": r.get("variant_name", ""),
                    "model": r.get("model_name", ""),
                    "status": r.get("status", ""),
                    "mAP@0.5": r.get("map50", ""),
                    "mAP@0.5:0.95": r.get("map50_95", ""),
                }
            )
        report_lines.append(markdown_table(abl_view, ["variant", "model", "status", "mAP@0.5", "mAP@0.5:0.95"]))
    else:
        report_lines.append("- ablation_main.csv 未生成有效行。")
    report_lines.append("")

    report_lines.append("## 6. 分尺度分析")
    report_lines.append("")
    if scale:
        report_lines.append(markdown_table(scale[: min(len(scale), 16)], ["model_name", "scale_bucket", "GT", "TP", "FN", "recall"]))
    else:
        report_lines.append("- scale_recall.csv 为空。")
    report_lines.append("")

    report_lines.append("## 7. 漏检机制分析")
    report_lines.append("")
    if fnm:
        report_lines.append(markdown_table(fnm[: min(len(fnm), 20)], ["model_name", "diag_type", "count"]))
    else:
        report_lines.append("- fn_mechanism.csv 为空。")
    report_lines.append("")

    report_lines.append("## 8. 误报结构分析")
    report_lines.append("")
    if fps:
        report_lines.append(markdown_table(fps[: min(len(fps), 24)], ["model_name", "metric", "count"]))
    else:
        report_lines.append("- fp_structure.csv 为空。")
    report_lines.append("")

    report_lines.append("## 9. 典型样本可视化说明")
    report_lines.append("")
    report_lines.append(f"- 样例清单: `{tables / 'qualitative_cases.csv'}`")
    report_lines.append(f"- 类别统计: `{tables / 'qualitative_summary.csv'}`")
    report_lines.append(f"- 图像目录: `{figures / 'compare_cases'}` / `{figures / 'fp_cases'}` / `{figures / 'fn_cases'}`")
    report_lines.append(f"- 已产出样例数: {len(q_cases)}")
    if q_sum:
        report_lines.append(markdown_table(q_sum, ["category", "n_selected", "n_candidates"]))
    report_lines.append("")

    report_lines.append("## 10. 热力图方法说明")
    report_lines.append("")
    report_lines.append("- 优先级: EigenCAM -> 中间特征响应(mean聚合) -> 框分数密度图")
    report_lines.append(f"- 方法记录: `{tables / 'heatmap_method.json'}`")
    report_lines.append(f"- 结果清单: `{tables / 'heatmap_manifest.csv'}`")
    report_lines.append(f"- 图像目录: `{figures / 'heatmaps'}`")
    report_lines.append(f"- 已产出热力图数量: {len(h_cases)}")
    report_lines.append("")

    report_lines.append("## 11. 未完成项 / 缺失指标说明")
    report_lines.append("")
    if missing_items:
        miss_view = []
        for m in missing_items:
            miss_view.append(
                {
                    "缺失项": m["item"],
                    "状态": m["status"],
                    "原因": m["reason"],
                    "建议补充": m["next"],
                }
            )
        report_lines.append(markdown_table(miss_view, ["缺失项", "状态", "原因", "建议补充"]))
    else:
        report_lines.append("- 当前请求中的核心产物均已生成。")
    report_lines.append("")

    report_lines.append("## 缺失项说明")
    report_lines.append("")
    report_lines.append("- 若某指标或图未稳定得到，已在上一节逐项列出：缺失项、原因、补充建议。")
    report_lines.append("")

    report_lines.append("## 论文可直接引用的文件")
    report_lines.append("")
    report_lines.append(f"- 主对比表: `{tables / 'compare_main.csv'}`")
    report_lines.append(f"- 消融主表: `{tables / 'ablation_main.csv'}`")
    report_lines.append(f"- 分尺度召回: `{tables / 'scale_recall.csv'}`")
    report_lines.append(f"- 漏检机制: `{tables / 'fn_mechanism.csv'}`")
    report_lines.append(f"- 误报结构: `{tables / 'fp_structure.csv'}`")
    report_lines.append(f"- 典型样本图目录: `{figures / 'compare_cases'}`、`{figures / 'fp_cases'}`、`{figures / 'fn_cases'}`")
    report_lines.append(f"- 热力图目录: `{figures / 'heatmaps'}`")

    report_path = report_root / "report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"[done] report -> {report_path}")


if __name__ == "__main__":
    main()
