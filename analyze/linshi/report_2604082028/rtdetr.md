(yolo11) ubuntu@ubuntu-System-Product-Name:~/hpproject/yolo$ cd /home/ubuntu/hpproject/yolo
OUT=experiments/det_bench/datasetm6c/metrics/iou_sweep
mkdir -p "$OUT"

for iou in 0.2 0.3 0.4 0.5 0.6 0.7; do
  conda run -n yolo11 python tools/eval_detection_benchmark.py \
    --dataset-root /home/ubuntu/hpproject/yolo/dataset/yolo/datasetm6c \
    --split test \
    --pred-json "faster_rcnn=/home/ubuntu/hpproject/yolo/experiments/det_bench/datasetm6c/preds/faster_rcnn_test.json" \
    --pred-json "rt_detr=/home/ubuntu/hpproject/yolo/experiments/det_bench/datasetm6c/preds/rt_detr_test.json" \
done--out-json "$OUT/main_table_iou${iou}.json"
model,obj_precision,obj_recall,mAP@0.5,mAP@0.5:0.95,TP,FP,FN
faster_rcnn,0.500000,0.612245,0.362254,0.162878,60,60,38
rt_detr,0.225131,0.877551,0.463736,0.226105,86,296,12
note: 掩膜转最小外接框后得到的是检测能力对比，不是分割精度对比。
[done] csv -> /home/ubuntu/hpproject/yolo/experiments/det_bench/datasetm6c/metrics/iou_sweep/main_table_iou0.2.csv
[done] json -> /home/ubuntu/hpproject/yolo/experiments/det_bench/datasetm6c/metrics/iou_sweep/main_table_iou0.2.json
model,obj_precision,obj_recall,mAP@0.5,mAP@0.5:0.95,TP,FP,FN
faster_rcnn,0.483333,0.591837,0.362254,0.162878,58,62,40
rt_detr,0.222513,0.867347,0.463736,0.226105,85,297,13
note: 掩膜转最小外接框后得到的是检测能力对比，不是分割精度对比。
[done] csv -> /home/ubuntu/hpproject/yolo/experiments/det_bench/datasetm6c/metrics/iou_sweep/main_table_iou0.3.csv
[done] json -> /home/ubuntu/hpproject/yolo/experiments/det_bench/datasetm6c/metrics/iou_sweep/main_table_iou0.3.json
model,obj_precision,obj_recall,mAP@0.5,mAP@0.5:0.95,TP,FP,FN
faster_rcnn,0.458333,0.561224,0.362254,0.162878,55,65,43
rt_detr,0.214660,0.836735,0.463736,0.226105,82,300,16
note: 掩膜转最小外接框后得到的是检测能力对比，不是分割精度对比。
[done] csv -> /home/ubuntu/hpproject/yolo/experiments/det_bench/datasetm6c/metrics/iou_sweep/main_table_iou0.4.csv
[done] json -> /home/ubuntu/hpproject/yolo/experiments/det_bench/datasetm6c/metrics/iou_sweep/main_table_iou0.4.json
model,obj_precision,obj_recall,mAP@0.5,mAP@0.5:0.95,TP,FP,FN
faster_rcnn,0.433333,0.530612,0.362254,0.162878,52,68,46
rt_detr,0.193717,0.755102,0.463736,0.226105,74,308,24
note: 掩膜转最小外接框后得到的是检测能力对比，不是分割精度对比。
[done] csv -> /home/ubuntu/hpproject/yolo/experiments/det_bench/datasetm6c/metrics/iou_sweep/main_table_iou0.5.csv
[done] json -> /home/ubuntu/hpproject/yolo/experiments/det_bench/datasetm6c/metrics/iou_sweep/main_table_iou0.5.json
model,obj_precision,obj_recall,mAP@0.5,mAP@0.5:0.95,TP,FP,FN
faster_rcnn,0.350000,0.428571,0.362254,0.162878,42,78,56
rt_detr,0.170157,0.663265,0.463736,0.226105,65,317,33
note: 掩膜转最小外接框后得到的是检测能力对比，不是分割精度对比。
[done] csv -> /home/ubuntu/hpproject/yolo/experiments/det_bench/datasetm6c/metrics/iou_sweep/main_table_iou0.6.csv
[done] json -> /home/ubuntu/hpproject/yolo/experiments/det_bench/datasetm6c/metrics/iou_sweep/main_table_iou0.6.json
model,obj_precision,obj_recall,mAP@0.5,mAP@0.5:0.95,TP,FP,FN
faster_rcnn,0.266667,0.326531,0.362254,0.162878,32,88,66
rt_detr,0.115183,0.448980,0.463736,0.226105,44,338,54
note: 掩膜转最小外接框后得到的是检测能力对比，不是分割精度对比。
[done] csv -> /home/ubuntu/hpproject/yolo/experiments/det_bench/datasetm6c/metrics/iou_sweep/main_table_iou0.7.csv
[done] json -> /home/ubuntu/hpproject/yolo/experiments/det_bench/datasetm6c/metrics/iou_sweep/main_table_iou0.7.json