# YOLO11 工业缺陷检测工程说明

本仓库是一个基于 YOLO11 的二维工业缺陷检测工程，主要用于 `datasetm6c` 数据集上的 baseline 训练、enhance241 模块组合实验、验证集/测试集指标统计，以及多模型对比分析。

接手时按这个顺序走即可：

```text
准备环境 -> 准备数据集 -> 检查配置 -> 跑 baseline -> 跑模块组合 -> 查看实验产物和结果指标
```

## 1. 工程入口

最重要的文件：

```text
src/train.py
  训练、测试、训练后指标扫描的主入口。

configs/baseline/datasetm6c.yaml
  datasetm6c 的 YOLO11 baseline 配置。

configs/enhance/datasetm6c/defect241.yaml
  enhance241 主配置，包含 A/B/C/D 模块开关与超参。

configs/data/defect.yaml
  数据集类别配置，当前 nc=1，names=["defect"]。

tools/run_yolov11_241.sh
  单个模块或一个模块组合的训练脚本。

tools/run_yolov11_241_module_combo.sh
  多组模块组合批量实验脚本。

analyze/code/p2604_multi_model_eval.py
  多模型统一评估、FN/FP 统计和可视化导出脚本。

third_party/yolo11/enhance241/
  enhance241 各模块实现。
```

常用工作目录：

```text
/home/ubuntu/hpproject/yolo
```

所有命令默认从仓库根目录运行。

## 2. 环境安装

建议环境：

- Ubuntu 20.04 或以上。
- NVIDIA GPU。
- CUDA 驱动需要能支持导出环境中的 PyTorch CUDA 版本。
- 单卡显存建议 12GB 以上；低显存时使用脚本里的 `--vram-guard auto`。

本仓库已把当前可用 Conda 环境导出到：

```text
yolo11.yml
```

该环境来自已有 `yolo11` Conda 环境，核心版本包括：

```text
python=3.9.23
torch==2.7.1+cu118
torchvision==0.22.1+cu118
ultralytics==8.4.6
opencv-python==4.13.0.90
pyyaml==6.0.3
numpy==2.0.2
scipy==1.13.1
```

推荐安装流程：

```bash
cd /home/ubuntu/hpproject/yolo

conda env create -f yolo11.yml
conda activate yolo11

python -c "import torch, ultralytics; print(torch.cuda.is_available(), ultralytics.__version__)"
nvidia-smi
```

如果本机已经存在同名 `yolo11` 环境，用导出文件同步依赖：

```bash
cd /home/ubuntu/hpproject/yolo

conda env update -n yolo11 -f yolo11.yml --prune
conda activate yolo11

python -c "import torch; print('cuda_available=', torch.cuda.is_available())"
```

如果 `torch.cuda.is_available()` 输出 `False`，先检查 NVIDIA 驱动、CUDA 兼容性和当前是否激活了 `yolo11` 环境。训练脚本可以在 yaml 中保持 `device: ""` 自动选卡，也可以写 `device: "0"` 指定第 0 张 GPU。

预训练权重默认放在：

```text
models/pretrained/
├── yolo11m.pt
├── yolo11n.pt
└── yolo26n.pt
```

配置里的 `model: models/pretrained/yolo11m.pt` 会从这里读取。

## 3. 数据集准备

当前主数据集是：

```text
dataset/yolo/datasetm6c/
```

数据集必须按 YOLO 检测格式组织：

```text
dataset/yolo/datasetm6c/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

每张图片对应一个同名 `.txt` 标签文件。例如：

```text
images/train/0001.jpg
labels/train/0001.txt
```

标签格式：

```text
class_id x_center y_center width height
```

坐标是 YOLO 归一化坐标，范围为 `[0, 1]`。当前类别数为 `1`，类别名为 `defect`。

无缺陷图像的约定：

- 对应 label 文件不存在，或者
- 对应 label 文件存在但内容为空。

检查 train/val/test 数量和无缺陷图像数量：

```bash
python - <<'PY'
from pathlib import Path

root = Path("dataset/yolo/datasetm6c")
exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
total = bg_total = 0

for sp in ["train", "val", "test"]:
    img_dir = root / "images" / sp
    lbl_dir = root / "labels" / sp
    imgs = sorted([p for p in img_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts])
    bg = 0
    for im in imgs:
        lp = (lbl_dir / im.relative_to(img_dir)).with_suffix(".txt")
        if not lp.exists() or not lp.read_text(encoding="utf-8", errors="ignore").strip():
            bg += 1
    total += len(imgs)
    bg_total += bg
    print(f"{sp}: total={len(imgs)}, no_defect={bg}")

print(f"ALL: total={total}, no_defect={bg_total}")
PY
```

## 4. 数据配置说明

数据类别配置文件是：

```text
configs/data/defect.yaml
```

里面的关键字段：

```yaml
path: /some/dataset/root
train: images/train
val: images/val
test: images/test
nc: 1
names: ["defect"]
```

注意：当前 `configs/data/defect.yaml` 里的 `path` 是旧机器路径。实际训练时以训练配置里的 `data_root` 为准：

```yaml
data: configs/data/defect.yaml
data_root: /home/ubuntu/hpproject/yolo/dataset/yolo/datasetm6c
```

`src/train.py` 会根据 `data_root` 在每次实验目录里生成真正使用的运行时 data yaml：

```text
experiments/.../train/data.yaml
```

所以排查数据路径时，优先看实验目录中的 `train/data.yaml`。

## 5. 训练配置文件

baseline 配置：

```text
configs/baseline/datasetm6c.yaml
```

enhance241 主配置：

```text
configs/enhance/datasetm6c/defect241.yaml
```

常用字段：

```yaml
yolo_version: yolo11
exp_name: baseline/datasetm6c
run_name: ""
data: configs/data/defect.yaml
data_root: /home/ubuntu/hpproject/yolo/dataset/yolo/datasetm6c
model: models/pretrained/yolo11m.pt
epochs: 150
imgsz: 640
batch: 6
device: ""
workers: 4

conf: 0.25
nms_iou: 0.7
max_det: 100
match_iou: 0.3

mode: train_test
weights: ""
```

字段含义：

- `exp_name`：实验输出根目录，最终落到 `experiments/<exp_name>/exp_YYMMDDHHMM/`。
- `run_name`：可选二级目录，非空时落到 `experiments/<exp_name>/<run_name>/exp_YYMMDDHHMM/`。
- `data`：类别配置 yaml。
- `data_root`：真实数据集根目录，优先级高于 `data` 里的旧 `path`。
- `model`：初始预训练权重。
- `mode`：`train_test` 表示训练后自动做评估；`test` 表示只评估，需要设置 `weights`；`finetune_test` 表示从已有权重继续训练再评估。
- `conf`：推理置信度阈值。
- `nms_iou`：NMS IoU 阈值。
- `match_iou`：GT 和 Pred 匹配时的 IoU 阈值，用于 TP/FP/FN 统计。
- `threshold_sweep`：训练后阈值扫描配置。
- `enhance241`：模块开关和模块超参。

## 6. 从数据集到训练完成的完整流程

### 6.1 准备数据

确认数据目录存在：

```bash
ls dataset/yolo/datasetm6c/images/train
ls dataset/yolo/datasetm6c/images/val
ls dataset/yolo/datasetm6c/images/test
ls dataset/yolo/datasetm6c/labels/train
ls dataset/yolo/datasetm6c/labels/val
ls dataset/yolo/datasetm6c/labels/test
```



### 6.2 先跑 baseline

```bash
cd /home/ubuntu/hpproject/yolo
python src/train.py --config configs/baseline/datasetm6c.yaml
```

训练完成后，终端会打印实验目录。也可以按时间查看：

```bash
find experiments/yolo11-baseline/datasetm6c -maxdepth 1 -type d -name 'exp_*' | sort | tail
```

### 6.3 再跑一个模块组合

推荐用脚本，不手动改 yaml：

```bash
bash tools/run_yolov11_241.sh \
  --base-config configs/enhance/datasetm6c/defect241.yaml \
  --vram-guard auto \
  --safe-batch 6 \
  a4 b7 d6
```

这条命令会基于 `defect241.yaml` 临时打开 `a4`、`b7`、`d6`，并调用 `src/train.py` 训练。

### 6.4 批量跑多个组合

```bash
bash tools/run_yolov11_241_module_combo.sh \
  --base-config configs/enhance/datasetm6c/defect241.yaml \
  --epochs 150 \
  --batch 6 \
  --combos baseline,a4+b7+d6,hmc7,pdd9 \
  --tag datasetm6c_handover
```

只生成临时配置、不训练：

```bash
bash tools/run_yolov11_241_module_combo.sh \
  --base-config configs/enhance/datasetm6c/defect241.yaml \
  --combos baseline,a4+b7+d6 \
  --dry-run
```

### 6.5 查看训练是否完成

每次训练会生成：

```text
experiments/<exp_name>/exp_YYMMDDHHMM/
```

如果 `run_name` 非空，则是：

```text
experiments/<exp_name>/<run_name>/exp_YYMMDDHHMM/
```

训练是否完整，优先看：

```text
train/results.csv
train/weights/best.pt
train/weights/last.pt
train/result_summary.md
metrics/roc_curve.csv
metrics/roc_keypoints.md
```

## 7. 实验产物结构

典型实验目录：

```text
experiments/yolo11-a4b7d6/datasetm6c/exp_YYMMDDHHMM/
├── train/
│   ├── args.yaml
│   ├── config.yaml
│   ├── config_dump.json
│   ├── data.yaml
│   ├── enhance241_audit.md
│   ├── hparam_plan.md
│   ├── result_summary.md
│   ├── results.csv
│   └── weights/
│       ├── best.pt
│       └── last.pt
├── metrics/
│   ├── roc_curve.csv
│   ├── roc_keypoints.md
│   ├── eval_image_level.csv
│   ├── eval_image_level.json
│   └── *_threshold_sweep.json
├── val_vis/
└── test_vis/
```

重点文件说明：

- `train/results.csv`：YOLO 训练过程指标，包含每个 epoch 的验证集 `mAP50`、`mAP50-95`、`precision`、`recall` 等。
- `train/weights/best.pt`：验证集规则下最优权重。
- `train/weights/last.pt`：最后一个 epoch 的权重。
- `train/config.yaml`：本次实验实际使用的训练配置副本。
- `train/data.yaml`：本次实验实际使用的数据 yaml，排查数据路径优先看这个。
- `train/enhance241_audit.md`：模块是否启用、patch 是否执行、可训练参数等审计信息。
- `train/hparam_plan.md`：本次训练的关键超参记录。
- `train/result_summary.md`：训练后评估摘要。
- `metrics/roc_curve.csv`：阈值扫描曲线。
- `metrics/roc_keypoints.md`：常用 FPR 点附近的 recall 和 threshold。
- `metrics/eval_image_level.csv`：val+test 图像级结果明细。
- `val_vis/`、`test_vis/`：验证集和测试集可视化预测图，受 `save_val_pic`、`save_test_pic`、`skip_eval_visuals` 控制。
- 验证集训练指标主要看 `train/results.csv` 中的 `metrics/precision(B)`、`metrics/recall(B)`、`metrics/mAP50(B)`、`metrics/mAP50-95(B)`、`val/box_loss`、`val/cls_loss`、`val/dfl_loss`。
- 图像级漏检、误报和阈值扫描主要看 `metrics/eval_image_level.csv`、`metrics/roc_curve.csv`、`metrics/roc_keypoints.md`。

批量模块脚本还会额外写：

```text
experiments/yolo11-module-combo/<dataset_tag>/exp_<tag>/
├── logs/
└── summary.tsv
```

## 8. enhance241 模块如何开启和关闭

### 8.1 通过脚本开启模块

推荐方式是用脚本传模块名：

```bash
bash tools/run_yolov11_241.sh a3
bash tools/run_yolov11_241.sh a4 b7 d6
bash tools/run_yolov11_241.sh hmc7
```

脚本会基于 `configs/enhance/datasetm6c/defect241.yaml` 生成临时配置，并把对应模块设为 `true`。

批量实验：

```bash
bash tools/run_yolov11_241_module_combo.sh \
  --base-config configs/enhance/datasetm6c/defect241.yaml \
  --combos baseline,a3,b7,d6,a4+b7+d6,hmc7,pdd9
```

### 8.2 通过 yaml 手动开启模块

也可以直接改：

```text
configs/enhance/datasetm6c/defect241.yaml
```

例如开启 `a4+b7+d6`：

```yaml
enhance241:
  a4: true
  b7: true
  d6: true
```

关闭模块就是改回：

```yaml
enhance241:
  a4: false
  b7: false
  d6: false
```

手动改 yaml 后运行：

```bash
python src/train.py --config configs/enhance/datasetm6c/defect241.yaml
```

### 8.3 模块分组规则

模块分为 A/B/C/D 四组：

```text
A：backbone / 特征提取增强
B：neck / 语义融合增强
C：head 前选择、抑噪、门控
D：检测头或分数校准
```

支持的开关：

```text
a3 a4 a5 a6 a7 a9 a11 a21
b1 b2 b3 b5 b6 b7 b9 b11 b21
c4 c5 c6 c7 c9 c11 c21
d1 d3 d5 d6 d7 d9 d11 d21
```

脚本常用别名：

```text
hmc7   = a7+b7+c7+d7
pdd9   = a9+b9+c9+d9
abcd6  = a6+b6+c6+d6
abcd11 = a11+b11+c11+d11
abcd21 = a21+b21+c21+d21
pack21 = a21+b21+c21+d21
b1237  = b1,b2,b3,b7 四个单模块实验
d1579  = d1,d5,d7,d9 四个单模块实验
```

模块实现文件在：

```text
third_party/yolo11/enhance241/
```

每个模块通常提供：

```python
apply(model, cfg)
```

实际 patch 顺序在：

```text
src/train.py::_apply_enhance241_patches()
```

实验后必须看：

```text
<exp_dir>/train/enhance241_audit.md
```

确认模块确实启用、参数进入训练、patch 没有失败。
