# 算法生成常见陷阱与避坑指南

## 概述

本文档总结了算法模型生成过程中常见的错误和陷阱。生成代码时必须避免这些问题。

## 一、代码结构陷阱

### 1. 文件名不合法导致导入失败

**问题**：文件名包含连字符（`-`）或中文，导致 `import` 失败。

```python
# 错误：文件名含连字符
# 文件名: ZWH-未知领域-001-260728_algorithm.py
import ZWH-未知领域-001-260728_algorithm  # SyntaxError

# 正确：使用下划线
# 文件名: zwh_unknown_domain_001_260728_algorithm.py
import zwh_unknown_domain_001_260728_algorithm  # OK
```

**规则**：Python 模块名只能包含字母、数字和下划线，且不能以数字开头。

### 2. main_process 参数不匹配

**问题**：函数签名与用户指定的输入规格不匹配。

```python
# 用户要求输入: video_url, image_url, structured_data
# 错误：缺少参数或类型错误
def main_process(data: str) -> dict:  # 只有一个参数

# 正确：匹配规格
def main_process(
    video_url: str = None,
    image_url: str = None,
    structured_data: dict = None
) -> dict:
```

### 3. 返回值缺少必需字段

**问题**：返回的字典缺少用户要求的输出字段。

```python
# 用户要求输出: classification_label, confidence_list
# 错误：缺少字段
return {"label": "正常"}  # 字段名不对

# 正确：完整匹配
return {
    "classification_label": ["正常"],
    "confidence_list": [0.9]
}
```

## 二、逻辑正确性陷阱

### 1. 使用 random 作为核心逻辑

**问题**：用 `random.choice()` 或 `random.random()` 作为分类/检测的决策逻辑，结果不可复现且无意义。

```python
# 错误：随机决策
import random
def main_process(data):
    label = random.choice(["正常", "可疑", "高风险"])
    return {"classification_label": [label]}

# 正确：基于规则的决策
def main_process(data: dict) -> dict:
    if data.get("amount", 0) > 10000:
        label = "高风险"
    else:
        label = "正常"
    return {"classification_label": [label]}
```

### 2. 占位符代替实现

**问题**：用注释或 pass 代替真实逻辑。

```python
# 错误：占位符
def main_process(data):
    # TODO: 实现分类逻辑
    pass

# 正确：真实实现
def main_process(data: dict) -> dict:
    score = sum(data.get(k, 0) for k in ["risk_a", "risk_b", "risk_c"])
    label = "高风险" if score > 0.7 else "正常"
    return {"classification_label": [label]}
```

### 3. 条件分支不完整

**问题**：if-elif-else 链缺少 else，导致某些输入无输出。

```python
# 错误：缺少 else，None 输入会返回 None
def classify(score: float) -> str:
    if score > 0.8:
        return "高风险"
    elif score > 0.5:
        return "可疑"
    # 缺少 else

# 正确：有兜底逻辑
def classify(score: float) -> str:
    if score > 0.8:
        return "高风险"
    elif score > 0.5:
        return "可疑"
    else:
        return "正常"
```

### 4. 置信度范围错误

**问题**：置信度不在 [0, 1] 范围内。

```python
# 错误：置信度可能超出范围
confidence = raw_score  # raw_score 可能为负或大于 1

# 正确：归一化到 [0, 1]
confidence = max(0.0, min(1.0, raw_score))
# 或使用 sigmoid 归一化
import math
confidence = 1.0 / (1.0 + math.exp(-raw_score))
```

## 三、数据处理陷阱

### 1. 除零错误

```python
# 错误：可能除以零
avg = sum(values) / len(values)  # values 为空时报错

# 正确：检查空列表
avg = sum(values) / len(values) if values else 0.0
```

### 2. 类型转换错误

```python
# 错误：字符串直接用于数值计算
total = sum(data.values())  # 如果 value 包含字符串会报错

# 正确：过滤并转换类型
numeric_values = [float(v) for v in data.values() if isinstance(v, (int, float))]
total = sum(numeric_values)
```

### 3. None 值未处理

```python
# 错误：未处理 None 输入
def main_process(video_url: str) -> dict:
    features = extract_features(video_url)  # video_url 为 None 时报错

# 正确：默认值 + 空值检查
def main_process(video_url: str = None) -> dict:
    if video_url is None:
        return {"classification_label": ["正常"], "confidence_list": [0.5]}
    features = extract_features(video_url)
```

### 4. 字典 key 不存在

```python
# 错误：直接访问不存在的 key
amount = data["amount"]  # KeyError

# 正确：使用 get 方法带默认值
amount = data.get("amount", 0)
amount = data.get("amount", 0.0)  # 明确指定类型
```

## 四、依赖管理陷阱

### 1. 导入不存在的库

```python
# 错误：导入未安装的库
import torch  # 如果环境没有 torch 会报错

# 正确：使用标准库或常见库
import numpy as np  # numpy 是普遍可用的
```

### 2. 硬编码路径

```python
# 错误：硬编码本地路径
model = load_model("/home/user/models/my_model.pkl")

# 正确：使用相对路径或参数化
import os
model_dir = os.path.dirname(__file__)
model_path = os.path.join(model_dir, "model.pkl")
```

### 3. 不必要的依赖

```python
# 错误：声明了 torch 但只用于生成随机张量
import torch
features = torch.randn(1, 10)  # 完全没必要

# 正确：用标准库替代
import random
features = [random.uniform(0, 1) for _ in range(10)]
```

## 五、可复现性陷阱

### 1. 随机种子未固定

```python
# 错误：每次运行结果不同
import random
label = random.choice(["A", "B", "C"])

# 正确：固定随机种子（如果必须用 random）
import random
random.seed(42)
label = random.choice(["A", "B", "C"])  # 每次运行结果一致
```

### 2. 时间戳导致不确定性

```python
# 错误：基于当前时间做决策
import time
if time.time() % 2 > 1:
    label = "A"
else:
    label = "B"

# 正确：基于输入数据做决策
label = "A" if score > threshold else "B"
```

## 六、输出规范陷阱

### 1. 输出标签不在用户定义的集合内

```python
# 用户定义标签: 正常, 可疑, 高风险, 疑似欺诈

# 错误：输出了未定义的标签
return {"classification_label": ["walking"]}  # "walking" 不在定义集合中

# 正确：只输出用户定义的标签
return {"classification_label": ["高风险"]}
```

### 2. 单标签返回字符串而非列表

```python
# 错误：返回字符串
return {"classification_label": "正常"}  # 应该是列表

# 正确：返回列表
return {"classification_label": ["正常"]}
```

### 3. 置信度与标签数量不匹配

```python
# 错误：3 个标签但只有 2 个置信度
return {
    "classification_label": ["正常", "可疑", "高风险"],
    "confidence_list": [0.9, 0.8]  # 少了一个
}

# 正确：数量一致
return {
    "classification_label": ["正常", "可疑", "高风险"],
    "confidence_list": [0.9, 0.8, 0.7]
}
```

## 自检清单

生成代码后，逐项检查：

- [ ] 文件名是否合法（仅字母、数字、下划线）
- [ ] main_process 参数是否与输入规格匹配
- [ ] 返回值是否包含所有必需的输出字段
- [ ] 是否有 random 作为核心决策逻辑
- [ ] 是否有 pass/TODO/占位符代替实现
- [ ] if-elif-else 是否有兜底 else
- [ ] 置信度是否在 [0, 1] 范围内
- [ ] 是否处理了空输入/None 输入
- [ ] 是否有除零风险
- [ ] 输出标签是否都在用户定义的集合内
- [ ] 置信度列表长度是否与标签列表一致
- [ ] 是否有不必要的依赖
- [ ] 是否有硬编码路径
