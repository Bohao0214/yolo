#!/usr/bin/env bash
set -euo pipefail

: <<'DOC'
服务器一键 smoke 脚本（4 数据集，batch=4/6，epoch=2）。

默认行为：
- 训练与检测评估在 GPU `--device 0`
- 图像级评估在 CPU `--image_eval_device cpu`（更稳，避免 OOM）
- 输出落到 /home/ubuntu/hpproject/yolo/experiments/industrial_bs_sweep_YYMMDD_HHMM/

用法：
bash /home/ubuntu/hpproject/yolo/analyze/code/run_industrial_bs_sweep.sh

附加参数（会原样透传给 python 脚本）：
bash /home/ubuntu/hpproject/yolo/analyze/code/run_industrial_bs_sweep.sh --epochs 2 --batches 4 6
DOC

cd /home/ubuntu/hpproject/yolo
conda run -n yolo11 python -u /home/ubuntu/hpproject/yolo/analyze/code/run_industrial_bs_sweep.py \
  --epochs 2 \
  --batches 4 6 \
  --device 0 \
  --image_eval_device cpu \
  --image_eval_batch 1 \
  "$@"
