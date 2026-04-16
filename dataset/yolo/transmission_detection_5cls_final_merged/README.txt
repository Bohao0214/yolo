output_root: /home/wanghua/Dipan_ws/datasets/transmission_detection_5cls_final_merged
packed_root: /home/wanghua/Dipan_ws/datasets/transmission_detection_5cls_packed
augmented_root: /home/wanghua/Dipan_ws/datasets/transmission_objects_augmented_yolo

class_names:
  0: balloon
  1: kite
  2: nets
  3: trash
  4: twigs

source_image_counts:
  packed train=653 val=256
  augmented train=480 val=120

merged_image_counts:
  train: images=1133, labels=1133
    0 balloon: 336 boxes
    1 kite: 316 boxes
    2 nets: 68 boxes
    3 trash: 288 boxes
    4 twigs: 309 boxes
  val: images=376, labels=376
    0 balloon: 123 boxes
    1 kite: 110 boxes
    2 nets: 16 boxes
    3 trash: 100 boxes
    4 twigs: 76 boxes
