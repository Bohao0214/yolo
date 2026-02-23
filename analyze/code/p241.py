#!/usr/bin/env python3
"""Compatibility launcher for the renamed P2.4.1 experiment script."""

from defect241_hparam_p241 import main


if __name__ == "__main__":
    # 兼容入口用法（推荐直接使用主文件）:
    #
    # 1) 仅查看实验计划（不训练）:
    #    python /home/ubuntu/hpproject/yolo/analyze/code/p241.py \
    #      --dry_run --variants baseline a3c5 --steps 0 1 2 --repeats 1
    #
    # 2) 执行训练+分析:
    #    python /home/ubuntu/hpproject/yolo/analyze/code/p241.py \
    #      --execute --variants baseline a3c5 --steps 0 1 2 --repeats 1 \
    #      --run_name_prefix p241_v1
    #
    # 3) 仅分析历史实验:
    #    python /home/ubuntu/hpproject/yolo/analyze/code/p241.py \
    #      --analyze_only --variants baseline a3c5 --steps 0 1 2 \
    #      --run_name_prefix p241_v1
    #
    # 关键可调参数:
    # - 数据/配置: --baseline_config --enhanced_base_config --out_root
    # - 实验结构: --variants --steps --repeats --run_name_prefix
    # - 训练超参: --epochs_step0/1/2 --patience --warmup_epochs --lr_ref --lrf
    # - 评估口径: --conf_op --match_iou --nms_iou --max_det --metric_conf
    # - 显存保护: --vram_guard --guard_max_gb --safe_batch --safe_workers
    main()
