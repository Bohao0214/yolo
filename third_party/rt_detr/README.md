# RT-DETR (datasetm6c)

训练:

```bash
conda run -n yolo11 python third_party/rt_detr/train.py \
  --dataset-root /home/ubuntu/hpproject/yolo/dataset/yolo/datasetm6c \
  --output-dir /home/ubuntu/hpproject/yolo/experiments/det_bench/datasetm6c/rt_detr \
  --model rtdetr-l.pt \
  --epochs 100 --imgsz 640 --batch 8 --device 0
```

推理并导出 JSON:

```bash
conda run -n yolo11 python third_party/rt_detr/predict.py \
  --weights /home/ubuntu/hpproject/yolo/experiments/det_bench/datasetm6c/rt_detr/best.pt \
  --dataset-root /home/ubuntu/hpproject/yolo/dataset/yolo/datasetm6c \
  --split test \
  --output-json /home/ubuntu/hpproject/yolo/experiments/det_bench/datasetm6c/preds/rt_detr_test.json \
  --imgsz 640 --device 0
```

说明:

- `train.py` 会在输出目录自动生成 `data.yaml`，并把 `train/weights/best.pt` 复制为 `<output-dir>/best.pt`。
- 预测 JSON 与 Faster R-CNN 采用同一结构，可直接进入统一评估脚本。
- 默认关闭 `deterministic`（减少 `grid_sampler_2d_backward_cuda` 警告）；如需强制确定性可加 `--deterministic`。
- 建议优先使用 `.pt` 预训练权重（如 `rtdetr-l.pt`）；`yaml + 不加 --pretrained` 属于从头训练，通常在小数据集上显著变差。
