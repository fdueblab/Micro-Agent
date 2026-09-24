# LLM 生成算法代码的典型缺陷与规避（实证复盘）

## 概述

本文档基于对平台历史生成算法模型的批量评测复盘，总结 LLM 生成算法代码的七类实证缺陷。
每类缺陷均附错误示例、正确写法与平台验证方式。生成代码前必须逐条自查。

## 一、语言混淆级语法错误

**实证**：历史模型中出现 `elif val > 100 || val < 0:`——把 JavaScript 的 `||` 写进 Python。

```python
# 错误（JavaScript 惯性）
if val > 100 || val < 0:   # SyntaxError

# 正确
if val > 100 or val < 0:
```

**规避**：只用 Python 语法；逻辑或/与/非只能用 `or` / `and` / `not`；写完后必须执行
`python -m py_compile` 真实编译验证。平台在算法登记入库前会做编译硬校验，语法错误直接拒绝。

## 二、虚假依赖声明

**实证**：模型头部声明 `flask>=2.3.0`、`flask-restx>=1.1.0`，代码中从未 import。

**规避**：依赖清单必须与实际 import 逐项一致。声明了未使用的依赖会误导用户安装无关环境；
使用了未声明的依赖会导致用户环境缺失。注意 pip 包名与导入名映射：

| pip 包名 | 导入名 |
|---|---|
| opencv-python | cv2 |
| scikit-learn | sklearn |
| pillow | PIL |
| PyYAML | yaml |
| beautifulsoup4 | bs4 |
| python-dateutil | dateutil |
| protobuf | google.protobuf |

## 三、依赖版本区间无上界 + 已废弃 API

**实证**：模型声明 `mediapipe>=0.10.0` 却使用旧版 `mp.solutions.pose` API——该 API 在
mediapipe 后期版本中已被移除，按声明安装最新版必然崩溃。

**规避**：
1. 使用存在 API 变更风险的库（mediapipe/opencv/torch 等），依赖声明必须带版本上界，
   如 `mediapipe>=0.10.0,<0.10.21`
2. 写代码前确认所用 API 在声明的版本区间内真实存在（可 `import` 后 `hasattr` 验证，
   或查该版本官方文档）
3. 平台评测会对所有 import 的属性链做子进程真实探测，函数体内访问不存在的属性
   （如新版 mediapipe 下的 `mp.solutions`）会被直接判不合格

## 四、外部命令行工具依赖

**实证**：模型用 `subprocess.run(["yt-dlp", ...], check=True)` 下载视频——yt-dlp 是
外部二进制工具，未安装时直接抛异常崩溃，且强依赖 YouTube 可访问性。

```python
# 错误：未保护的外部命令
subprocess.run(["yt-dlp", "-x", url], check=True)

# 正确：优先用 pip 可装的 Python 库；确需外部工具时必须降级处理
try:
    subprocess.run(["yt-dlp", "-x", url], check=True, capture_output=True, timeout=120)
except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as e:
    return {"error": f"视频下载失败（需要系统安装 yt-dlp）: {e}"}
```

**规避**：优先选择纯 Python 库实现同等功能；确需外部命令时 try/except 包裹、
限时、失败时返回明确 error 字段，并在 limitations 中说明环境要求。

## 五、吞异常兜底

**实证**：10/14 的历史模型存在 `except Exception: return 默认值`——掩盖错误而非处理错误，
用户拿到"正常"结果时无法区分是真正常还是静默失败。

```python
# 错误：静默吞掉
try:
    score = compute(row)
except Exception:
    return {"classification_label": "正常"}   # 伪装成正常结果

# 正确：记录并显式返回错误
try:
    score = compute(row)
except Exception as e:
    return {"classification_label": "无法判定", "error": f"处理失败: {e}"}
```

**规避**：禁止裸 `except Exception` 后直接返回默认值。异常要么记录进返回值的
error/detail 字段，要么重新抛出。精确捕获具体异常类型，不做全量兜底。

## 六、无依据兜底（恒返回默认类）

**实证**：模型在无法判断时 `confidences[0] = 1.0` 无依据默认第一个类别——在未知输入上
给出"看起来自信实则瞎猜"的结果；数据集评测中这类模型与"多数类基线"持平，被识别为无效模型。

```python
# 错误：无依据默认
if all(v == 0 for v in features.values()):
    return {"classification_label": LABELS[0], "confidence": 1.0}

# 正确：明确无法判定并降低置信度
if all(v == 0 for v in features.values()):
    return {"classification_label": "无法判定", "confidence": 0.1,
            "reason": "输入特征全为空，无法提取有效信号"}
```

**规避**：算法无法判断时必须返回明确的"无法判定"类结果或显著降低 confidence，
禁止静默返回默认类别伪装成正常判断。

## 七、魔数阈值无出处

**实证**：模型中大量 `if val < 0.1: 高风险 / elif val < 0.5: 可疑` 硬编码阈值，
数值无任何出处说明，换一批数据分布就可能失效。

```python
# 错误：魔数
if score < 0.1:
    return "高风险"

# 正确：具名常量 + 来源注释
# 阈值依据：用户需求描述"金额超过1万且夜间交易视为可疑"（2024-09 需求单）
HIGH_RISK_THRESHOLD = 0.1
if score < HIGH_RISK_THRESHOLD:
    return "高风险"
```

**规避**：关键判定阈值定义为具名常量并注释来源（用户需求/参考资料/数据统计三类之一）；
无出处时在 limitations 中说明"阈值为经验估计，建议用户提供标注数据校准"。

## 平台验证机制（生成时自动执行）

| 缺陷 | 验证方式 |
|---|---|
| 一、语法错误 | py_compile 编译 + 登记入库硬校验 |
| 二、虚假依赖 | 依赖声明一致性检查（双向比对） |
| 三、废弃 API | 子进程属性探测（含函数体内属性链） |
| 四、外部命令 | 外部命令依赖检测（未保护即不合格） |
| 五、吞异常 | 吞异常兜底检测 |
| 六、无依据兜底 | 数据集 holdout 评测 + 多数类/均值基线对比 |
| 七、魔数阈值 | 人工复核 + LLM 评委抽检 |

生成流程步骤 5.5 会执行上述全部确定性评测，failed 项必须回到步骤 3 修复后重验。
