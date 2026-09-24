# LLM 生成算法代码：性能与鲁棒性模式规范

本文档针对平台历史评测中发现的三大类质量缺陷（重型资源重复初始化、函数复杂度超标、脏输入崩溃），给出必须采用的正实现模式。平台评测系统会对这三类问题自动检查：重复初始化与复杂度超标会告警，脏输入崩溃（5 类脏输入中 ≥3 类崩溃）直接判不合格。

## 一、重型资源必须使用模块级懒加载单例

### 缺陷场景（历史模型 7/14 命中）

在 main_process 可达的函数体内直接构造重型资源：

```python
# 错误：每次调用 main_process 都重新加载姿态估计模型
def detect_pose_sequence(frames):
    with mp.solutions.pose.Pose() as pose:   # 每次调用加载一次模型！
        ...
```

批量处理 100 行数据 = 加载 100 次模型，单次调用可能耗时数秒，整批处理性能灾难。

### 正确模式：懒加载单例

```python
import mediapipe as mp

_pose_detector = None

def _get_pose_detector():
    """模块级懒加载单例：首次调用时初始化，之后复用。

    Returns:
        mediapipe Pose 检测器实例（进程内唯一）。
    """
    global _pose_detector
    if _pose_detector is None:
        _pose_detector = mp.solutions.pose.Pose()
    return _pose_detector


def detect_pose_sequence(frames):
    """使用单例检测器处理帧序列。"""
    pose = _get_pose_detector()   # 仅首次调用加载
    ...
```

适用对象：mediapipe solutions 系列（Pose/FaceMesh/Hands 等）、cv2.CascadeClassifier、onnxruntime.InferenceSession、transformers.from_pretrained、ultralytics.YOLO、tensorflow/keras 模型加载、任何加载权重文件或下载模型的调用。

注意：不要直接写在模块顶层（`_pose = mp.solutions.pose.Pose()`），那会让 import 就变慢且可能失败；也不要依赖 `functools.lru_cache` 包裹整个检测函数（缓存了输入参数，容易内存膨胀）。

## 二、单函数复杂度控制

### 缺陷场景（历史模型圈复杂度最高 24）

一个函数内堆叠大量 if/elif 分支做规则判定，圈复杂度失控：

```python
# 错误：compute_scores 圈复杂度 24，无法维护
def compute_scores(row):
    if row["a"] > 10:
        if row["b"] < 5:
            score = 1
        elif row["b"] < 10:
            score = 2
        ...（十余个分支）
```

### 正确模式：按职责拆分

- 单函数圈复杂度控制在 15 以内（平台评测红线：超过 15 告警，超过 25 不合格）
- 拆分方法：每个风险因子/特征维度一个独立函数（`_score_amount()`, `_score_time()`...），主函数只做汇总
- 阈值定义为模块级具名常量并注释来源，不要散落在分支里

```python
# 金额阈值来源：用户业务规则描述
_AMOUNT_HIGH_RISK = 80000.0

def _score_amount(amount: float) -> float:
    """金额风险子评分。"""
    ...

def _score_time(is_night: bool) -> float:
    """时间风险子评分。"""
    ...

def compute_scores(row: dict) -> dict:
    """汇总各维度子评分。"""
    return {
        "amount": _score_amount(float(row.get("amount", 0) or 0)),
        "time": _score_time(row.get("is_night", False)),
    }
```

## 三、脏输入必须优雅降级（评测红线）

### 缺陷场景（历史模型 5 类脏输入中 4 类崩溃）

```python
# 错误：脏输入直接抛异常
def main_process(row):
    amount = float(row["amount"])     # {} → KeyError；"abc" → ValueError
    ...
```

真实数据集常含缺失值、类型错乱；逐行处理场景下一次崩溃即整批失败。

### 正确模式：入口统一防御 + 降级契约

main_process 最外层对脏输入统一降级，返回明确结果而非异常：

```python
_DIRTY_INPUT_RESULT = {
    "classification_label": "无法判定",
    "confidence": 0.0,
    "reason": "输入数据缺失或非法",
}


def main_process(row: dict) -> dict:
    """主入口。

    Args:
        row: 一行输入数据。脏输入（空/None/类型错乱/垃圾键）时降级返回。
    """
    if not isinstance(row, dict) or not row:
        return dict(_DIRTY_INPUT_RESULT, reason="输入为空或非字典")
    try:
        amount = float(row.get("amount") or 0)
    except (TypeError, ValueError):
        return dict(_DIRTY_INPUT_RESULT, reason="金额字段类型非法")
    ...
```

降级契约要点：
- 与「无依据兜底」禁令配合：脏输入降级是**显式声明**的无法判定（confidence=0），不是静默默认第一个类别
- 步骤 2b 的测试必须包含脏输入用例（金额传字符串、字段 None、垃圾键），验证不抛异常且返回降级格式
- 平台扰动测试会喂 5 类脏输入：空字典 / None 值 / 错误类型 / 极端数值（1e18）/ 垃圾键——全部要求优雅处理

## 平台自动检查项对照

| 检查 | 判定 |
|---|---|
| 重型资源在 main_process 可达函数体内构造 | warning（应改懒加载单例） |
| 圈复杂度 >15 / >25 | warning / failed |
| 函数 >80 行、嵌套 ≥6 层 | warning |
| 5 类脏输入 ≥3 类崩溃 | failed（不合格，强制回炉） |
| 5 类脏输入 1~2 类崩溃 | warning |
