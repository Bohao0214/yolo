# 多模型统一评估说明

## 1. 指标与计算规则
- 目标级匹配：同类别一对一 Hungarian 匹配，IoU>=tp_iou 计为 TP。
- Precision(目标级)：TP/(TP+FP)
- Recall(目标级)：TP/(TP+FN)
- 图像级四格（按“是否有最终预测框”判阳性）：
  - hit_img：有 GT 且最终预测框数量 > 0
  - miss_img：有 GT 且最终预测框数量 = 0（图像级 FN）
  - fp_img：无 GT 且最终预测框数量 > 0
  - tn_img：无 GT 且最终预测框数量 = 0
- 图像级召回率：hit_img/(hit_img+miss_img)
- 图像级误报率：fp_img/(fp_img+tn_img)
- 分尺度 Recall：按 GT 在输入坐标系（letterbox）短边 s=min(w,h) 分桶统计。
- FN 机制（互斥，按顺序判定）：
  1) no_response：best_score_all < score_floor
  2) low_score：score_floor <= best_score_all < conf
  3) regression_or_match_poor：best_score_all>=conf 且 best_metric_conf < tp_iou
  4) postproc_filtered：best_metric_conf>=tp_iou 且 best_metric_final<tp_iou
- FP 结构：
  - pred_dup：未匹配预测中与同类任一 GT 的 IoU>=tp_iou
  - background_fp：其余 FP
- FP 类型（启发式）：highlight / edge / texture_boundary / other。

## 2. 输出文件
- compare_main.csv：主指标 + 目标级总量
- image_level_stats.csv：图像级四格 + 图像级 recall/fpr
- scale_recall.csv：分尺度 GT/TP/FN/Recall
- fn_mechanism.csv：每个 FN 的机制判定与关键中间量
- image_level_fn_images.csv：图像级 FN（miss_img）对应原图路径与导出路径
- target_level_fn_images.csv：目标级 FN（任一 unmatched_gt）对应原图路径与导出路径
- fn_images/<model>/image_level_fn：图像级 FN 可视化图（原图名，叠加 GT/Pred）
- fn_images/<model>/target_level_fn：目标级 FN 可视化图（原图名，叠加 GT/Pred）
- fp_structure.csv：每个 FP 的结构类型与启发式类型
- metadata.json：实际参数、来源、数据与模型映射
- raw_<model>.jsonl：每图原始 GT 与最终预测集合

## 3. 参数来源
- 优先级：手动输入 > fallback 配置文件 > 脚本默认。
- 实际采用值及来源写在 metadata.json 的 infer_params 与 infer_param_source。

## 5. 复现
- 结果目录：`/home/ubuntu/hpproject/yolo/analyze/result/report_2604101149`
- 运行时间：`2026-04-10T11:49:41`