# SD-YOLO11 检测接口用法

文件：`func/sd_yolo11_detector.py`  
默认权重：`/home/ubuntu/hpproject/yolo/best.pt`（a4+b7+d6）

## 1) 最简调用（推荐）

```python
from func import detect

results = detect("/path/to/image.jpg")
print(results)
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

- 需要在项目根目录或把项目根目录加入 `PYTHONPATH` 后调用。
- 如果要换权重，直接改 `weights` 参数即可。注意网络结构

