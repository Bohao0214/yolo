# 检测实验报告（YOLO系列）

生成时间: 2026-04-06 18:09:18
报告目录: `/home/ubuntu/hpproject/yolo/analyze/code/ch4tool`

## 待用户补充

- 待确认项: batch、conf、device、heatmap_sample_list(optional)、imgsz、iou、max_det
- 模型路径填写文件: `/home/ubuntu/hpproject/yolo/analyze/code/ch4tool/models_to_eval.txt`
- 数据集覆盖配置: `/home/ubuntu/hpproject/yolo/analyze/code/ch4tool/dataset_override.txt`

## 1. 实验概况

- 约束: 不改模型结构、不重训，仅做读取已有模型并统一评估。
- 评估口径: 统一脚本、统一数据划分、统一参数；掩膜转最小外接框时比较的是检测能力而非分割精度。
- 指标公式文档: `/home/ubuntu/hpproject/yolo/analyze/code/ch4tool/指标计算说明.md`

## 2. 模型清单与路径

| model_path | exists |
| --- | --- |
| /home/ubuntu/hpproject/yolo/experiments/a4b7d6/datasetm6c/defect241__a4__b7__d6/exp_2603060619/train/weights/best.pt | no |
| /home/ubuntu/hpproject/yolo/experiments/baseline/datasetm6c/exp_2603040206/train/weights/best.pt | no |
| /home/ubuntu/hpproject/yolo/experiments/a3b3d3/datasetm6c/exp_2604050042/train/weights/best.pt | no |
| /home/ubuntu/hpproject/yolo/experiments/a7b7c7d7/datasetm6c/exp_2604050107/train/weights/best.pt | no |
| /home/ubuntu/hpproject/yolo/experiments/yolo11/defect241__a3__c5/exp_2603032345/train/weights/best.pt | no |
| /home/ubuntu/hpproject/yolo/experiments/yolo11/defect241__a6__b6__c6__d6/exp_2603060222/train/weights/best.pt | no |
| /home/ubuntu/hpproject/yolo/experiments/yolo11/defect241__a6__b7__d11/exp_2603060315/train/weights/best.pt | no |
| /home/ubuntu/hpproject/yolo/experiments/yolo11/defect241__a3__c5/exp_2603012344/train/weights/best.pt | no |


## 3. 数据集与评估参数

- data.yaml: `/home/ubuntu/hpproject/yolo/configs/enhance/datasetm6c/defect241.yaml`
- dataset_root: `/home/ubuntu/hpproject/yolo/dataset/yolo/datasetm6c`
- split(requested/used): `test` / `test`
- split备注: ``
- eval_params: `{"imgsz": 640, "conf": 0.3, "iou": 0.6, "max_det": 20, "batch": 4, "device": "0", "score_thr": 0.3, "obj_iou": 0.2}`

## 4. 对比实验结果

| model | status | P | R | mAP@0.5 | mAP@0.5:0.95 |
| --- | --- | --- | --- | --- | --- |
| yolo_a4+b7+d6 | missing_model |  |  |  |  |
| exp_2603040206 | missing_model |  |  |  |  |
| yolo_a3+b3+d3 | missing_model |  |  |  |  |
| yolo_a7+b7+c7+d7 | missing_model |  |  |  |  |
| yolo_a3+c5 | missing_model |  |  |  |  |
| yolo_a6+b6+c6+d6 | missing_model |  |  |  |  |
| yolo_a6+b7+d11 | missing_model |  |  |  |  |
| yolo_a3+c5 | missing_model |  |  |  |  |


## 5. 消融实验结果

| variant | model | status | mAP@0.5 | mAP@0.5:0.95 |
| --- | --- | --- | --- | --- |
| baseline |  | not_found |  |  |
| + 特征提取增强 |  | not_found |  |  |
| + 特征融合增强 |  | not_found |  |  |
| + 分类校准 |  | not_found |  |  |
| 全部模块 |  | not_found |  |  |


## 6. 分尺度分析

- scale_recall.csv 为空。

## 7. 漏检机制分析

- fn_mechanism.csv 为空。

## 8. 误报结构分析

- fp_structure.csv 为空。

## 9. 典型样本可视化说明

- 样例清单: `/home/ubuntu/hpproject/yolo/analyze/code/ch4tool/tables/qualitative_cases.csv`
- 类别统计: `/home/ubuntu/hpproject/yolo/analyze/code/ch4tool/tables/qualitative_summary.csv`
- 图像目录: `/home/ubuntu/hpproject/yolo/analyze/code/ch4tool/figures/compare_cases` / `/home/ubuntu/hpproject/yolo/analyze/code/ch4tool/figures/fp_cases` / `/home/ubuntu/hpproject/yolo/analyze/code/ch4tool/figures/fn_cases`
- 已产出样例数: 0
| category | n_selected | n_candidates |
| --- | --- | --- |
| small_defect | 0 | 0 |
| medium_defect | 0 | 0 |
| highlight_interference | 0 | 0 |
| baseline_miss_best_hit | 0 | 0 |
| baseline_fp_best_suppress | 0 | 0 |
| duplicate_prediction | 0 | 0 |


## 10. 热力图方法说明

- 优先级: EigenCAM -> 中间特征响应(mean聚合) -> 框分数密度图
- 方法记录: `/home/ubuntu/hpproject/yolo/analyze/code/ch4tool/tables/heatmap_method.json`
- 结果清单: `/home/ubuntu/hpproject/yolo/analyze/code/ch4tool/tables/heatmap_manifest.csv`
- 图像目录: `/home/ubuntu/hpproject/yolo/analyze/code/ch4tool/figures/heatmaps`
- 已产出热力图数量: 0

## 11. 未完成项 / 缺失指标说明

| 缺失项 | 状态 | 原因 | 建议补充 |
| --- | --- | --- | --- |
| 对比实验主指标 | 未稳定得到 | 当前机器上模型权重或数据集路径不可用，无法完成统一推理评估。 | 补充可访问的模型绝对路径与统一 data.yaml/dataset_root 后重跑 run_eval_compare.py。 |
| 分尺度 Recall | 未产出 | 主评估未成功或未获得有效预测。 | 保证 compare_main.csv 中至少一个模型 status=ok。 |
| 典型样本对比图 | 未产出或不足 | 缺少可用预测或测试集路径不可解析。 | 确认 test 集路径与 raw_preds 生成成功后重跑 draw_qualitative_cases.py。 |
| 热力图/CAM | 未产出或不稳定 | no successful model rows | 提供可用模型并固定热力图样本清单，必要时安装 pytorch-grad-cam。 |


## 缺失项说明

- 若某指标或图未稳定得到，已在上一节逐项列出：缺失项、原因、补充建议。

## 论文可直接引用的文件

- 主对比表: `/home/ubuntu/hpproject/yolo/analyze/code/ch4tool/tables/compare_main.csv`
- 消融主表: `/home/ubuntu/hpproject/yolo/analyze/code/ch4tool/tables/ablation_main.csv`
- 分尺度召回: `/home/ubuntu/hpproject/yolo/analyze/code/ch4tool/tables/scale_recall.csv`
- 漏检机制: `/home/ubuntu/hpproject/yolo/analyze/code/ch4tool/tables/fn_mechanism.csv`
- 误报结构: `/home/ubuntu/hpproject/yolo/analyze/code/ch4tool/tables/fp_structure.csv`
- 典型样本图目录: `/home/ubuntu/hpproject/yolo/analyze/code/ch4tool/figures/compare_cases`、`/home/ubuntu/hpproject/yolo/analyze/code/ch4tool/figures/fp_cases`、`/home/ubuntu/hpproject/yolo/analyze/code/ch4tool/figures/fn_cases`
- 热力图目录: `/home/ubuntu/hpproject/yolo/analyze/code/ch4tool/figures/heatmaps`