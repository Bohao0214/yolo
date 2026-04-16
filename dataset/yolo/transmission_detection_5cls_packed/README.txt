Merged five-class transmission foreign-object dataset

base: /home/wanghua/Dipan_ws/datasets/transmission_detection_4cls_merged
twigs: /home/wanghua/transmission_detection_512/power+line.v1i.yolo26
output: /home/wanghua/Dipan_ws/datasets/transmission_detection_5cls_merged
seed: 42
val_ratio: 0.2

class_names:
- 0: bird_nest
- 1: kite
- 2: balloon
- 3: trash
- 4: tree_branch

total_samples: 4575
train_samples: 3659
val_samples: 916

merged_class_counts:
- 0: 2954
- 1: 508
- 2: 519
- 3: 535
- 4: 59

train_counts:
- 0: 2363
- 1: 406
- 2: 415
- 3: 428
- 4: 47

val_counts:
- 0: 591
- 1: 102
- 2: 104
- 3: 107
- 4: 12

source_counts:
- base: 4516
- twigs:train: 54
- twigs:val: 5
