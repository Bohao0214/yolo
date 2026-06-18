# SD-YOLO11 检测接口用法

文件：`func/sd_yolo11_detector.py`  
默认权重优先级：  
1) `func/best.pt`（推荐你把权重和本文件夹一起发）  
2) `/home/ubuntu/hpproject/yolo/best.pt`（本机开发路径）

## 0) 环境依赖

先安装依赖（建议在你的 `yolo11` 虚拟环境里）：

```bash
pip install -r /home/ubuntu/hpproject/yolo/func/requirements_func.txt
```

## 1) 最简调用（推荐）

```python
from func import detect

results = detect("/path/to/image.jpg")
print(results)
```

如果你是在 `func/` 目录内部直接跑脚本（如 `python test_batch_infer.py`），
可以改为：

```python
from sd_yolo11_detector import detect
```

返回结果是 `list[dict]`，每个框包含：

- `class_id`
- `class_name`
- `confidence`
- `xyxy`（左上右下像素坐标）

## 2) 避免重复初始化（高频调用）

```python
from func import get_detector

detector = get_detector(
    weights="/home/ubuntu/hpproject/yolo/best.pt",
    device="0",      # GPU0；CPU 可用 "cpu"
    conf=0.25,
    iou=0.7,
    imgsz=640,
    max_det=100,
)

res1 = detector.predict("/path/to/a.jpg")
res2 = detector.predict("/path/to/b.jpg")
```

`get_detector()` 内部有缓存：同一组参数只初始化一次模型。

## 3) 强制清理缓存

```python
from func import clear_detector_cache
clear_detector_cache()
```

## 4) 注意事项

- 已改为单文件内置：`func/sd_yolo11_detector.py` 内部动态注册 a4/b7/d6（含必要依赖）类定义。
- 现在只发 `func` 文件夹即可，不再依赖 `third_party` 子目录。
- 建议把模型权重命名为 `best.pt` 放在 `func/` 目录下，这样对方无需改路径。
- 如果要换权重，直接改 `weights` 参数即可（需与当前结构兼容）。
