# Faster R-CNN (datasetm6c)

训练:

```bash
conda run -n yolo11 python third_party/faster_rcnn/train.py \
  --dataset-root /home/ubuntu/hpproject/yolo/dataset/yolo/datasetm6c \
  --output-dir /home/ubuntu/hpproject/yolo/experiments/det_bench/datasetm6c/faster_rcnn \
  --epochs 30 --batch-size 4 --device 0
```

推理并导出 JSON 预测:

```bash
conda run -n yolo11 python third_party/faster_rcnn/predict.py \
  --weights /home/ubuntu/hpproject/yolo/experiments/det_bench/datasetm6c/faster_rcnn/best.pt \
  --dataset-root /home/ubuntu/hpproject/yolo/dataset/yolo/datasetm6c \
  --split test \
  --output-json /home/ubuntu/hpproject/yolo/experiments/det_bench/datasetm6c/preds/faster_rcnn_test.json \
  --device 0
```

说明:

- 标签读取支持 YOLO 框格式（`cls xc yc w h`）和 YOLO 分割格式（`cls x1 y1 ...`，自动转最小外接框）。
- 导出的 JSON 可直接用于 `tools/eval_detection_benchmark.py` 做统一主表对比。
