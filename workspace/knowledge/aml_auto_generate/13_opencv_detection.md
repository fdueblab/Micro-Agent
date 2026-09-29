# OpenCV 图像检测算法

## 概述

OpenCV 是最广泛使用的计算机视觉库，提供多种无需训练的检测能力。本文档介绍在单文件实现约束下，使用 OpenCV 预训练模型进行目标检测的正确方法。

## 安装

```bash
pip install opencv-python numpy
```

导入：
```python
import cv2
import numpy as np
```

## 常用检测方法

### 1. Haar 级联分类器（人脸/物体检测）

OpenCV 自带预训练的 Haar 级联模型，无需训练即可使用。

**可用预训练模型**（随 opencv 安装附带）：
- `haarcascade_frontalface_default.xml` — 正面人脸
- `haarcascade_eye.xml` — 眼睛
- `haarcascade_fullbody.xml` — 人体
- `haarcascade_upperbody.xml` — 上半身
- `haarcascade_car.xml` — 车辆（非默认，需单独下载）

```python
import cv2

def detect_objects_haar(image_path: str, cascade_file: str = None) -> list:
    """使用 Haar 级联检测目标。"""
    # 使用默认人脸检测器
    if cascade_file is None:
        cascade_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    
    cascade = cv2.CascadeClassifier(cascade_file)
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法读取图像: {image_path}")
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # detectMultiScale 参数：
    #   scaleFactor: 每次图像缩放比例（1.1 = 缩小 10%）
    #   minNeighbors: 检测框重叠的最小次数（越高越严格）
    #   minSize: 检测框最小尺寸
    objects = cascade.detectMultiScale(
        gray, 
        scaleFactor=1.1, 
        minNeighbors=5, 
        minSize=(30, 30)
    )
    
    results = []
    for (x, y, w, h) in objects:
        results.append({
            "bbox": [int(x), int(y), int(w), int(h)],
            "label": "detected_object"
        })
    return results
```

### 2. HOG + SVM 检测器（行人检测）

OpenCV 内置预训练的 HOG 行人检测器：

```python
import cv2

def detect_pedestrians(image_path: str) -> list:
    """使用 HOG+SVM 检测行人。"""
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法读取图像: {image_path}")
    
    # detectMultiScale 返回 (x, y, w, h) 检测框和置信度
    boxes, weights = hog.detectMultiScale(
        image,
        winStride=(8, 8),
        padding=(4, 4),
        scale=1.05
    )
    
    results = []
    for i, (x, y, w, h) in enumerate(boxes):
        confidence = float(weights[i][0]) if weights is not None else 0.5
        results.append({
            "bbox": [int(x), int(y), int(w), int(h)],
            "label": "person",
            "confidence": max(0.0, min(1.0, confidence))
        })
    return results
```

### 3. 模板匹配

适用于在图像中查找已知模板的位置：

```python
import cv2
import numpy as np

def template_match(source_path: str, template_path: str) -> list:
    """模板匹配，返回匹配位置。"""
    img = cv2.imread(source_path, cv2.IMREAD_GRAYSCALE)
    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None or template is None:
        raise ValueError("无法读取图像")
    
    if template.shape[0] > img.shape[0] or template.shape[1] > img.shape[1]:
        raise ValueError("模板不能大于源图像")
    
    # 六种匹配方法
    methods = [
        cv2.TM_CCOEFF, cv2.TM_CCOEFF_NORMED,
        cv2.TM_CCORR, cv2.TM_CCORR_NORMED,
        cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED
    ]
    
    method = cv2.TM_CCOEFF_NORMED  # 推荐使用归一化方法
    result = cv2.matchTemplate(img, template, method)
    
    # 阈值过滤
    threshold = 0.8
    locations = np.where(result >= threshold)
    
    results = []
    for pt in zip(*locations[::-1]):
        results.append({
            "bbox": [int(pt[0]), int(pt[1]), int(template.shape[1]), int(template.shape[0])],
            "confidence": float(result[pt[1], pt[0]])
        })
    return results
```

### 4. 轮廓检测

基于边缘检测提取对象轮廓：

```python
import cv2

def detect_contours(image_path: str, min_area: int = 500) -> list:
    """检测图像中的轮廓。"""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法读取图像: {image_path}")
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)
    
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    results = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        results.append({
            "bbox": [int(x), int(y), int(w), int(h)],
            "area": float(area)
        })
    return results
```

## main_process 标准接口

检测算法的 main_process 应返回检测框列表：

```python
def main_process(image_path: str = None, image_url: str = None) -> dict:
    """
    主检测函数。
    
    Args:
        image_path: 本地图像路径
        image_url: 图像 URL（需下载）
    
    Returns:
        dict: 包含 detection_results 的结果
    """
    # 处理 URL 输入（下载图像）
    if image_url and not image_path:
        import urllib.request
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
            urllib.request.urlretrieve(image_url, f.name)
            image_path = f.name
    
    if not image_path:
        return {"detection_results": [], "message": "未提供图像"}
    
    # 执行检测
    results = detect_objects_haar(image_path)
    
    return {
        "detection_results": results,
        "total_count": len(results)
    }
```

## 关键注意事项

1. **图像读取检查**：`cv2.imread()` 在文件不存在或格式不支持时返回 `None`，必须检查
2. **颜色通道顺序**：OpenCV 默认 BGR，不是 RGB；如需显示或与其它库交互需转换
3. **坐标格式**：检测框格式为 `(x, y, width, height)`，不是 `(x1, y1, x2, y2)`
4. **置信度归一化**：HOG 检测器的 weights 可能为负值或大于 1，需 clip 到 [0, 1]
5. **尺度因子**：`scaleFactor` 过小会显著增加检测时间，过大可能漏检
6. **图像尺寸**：大图像检测前应缩放，推荐最长边不超过 1280 像素

## 参考文档

- OpenCV 官方文档: https://docs.opencv.org/4.x/d2/d99/tutorial_js_pyramid_display.html
- Cascade Classifier: https://docs.opencv.org/4.x/db/d28/tutorial_cascade_classifier.html
- HOG Descriptor: https://docs.opencv.org/4.x/d5/d33/structcv_1_1HOGDescriptor.html
- Template Matching: https://docs.opencv.org/4.x/d4/dc6/tutorial_py_template_matching.html
