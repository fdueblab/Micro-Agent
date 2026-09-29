# 金融欺诈检测与极度不平衡学习

欺诈/可疑交易等金融二分类任务的正样本占比通常 <1%（极端可到 0.01%）。默认训练流程在该分布下完全失效，必须显式处理不平衡。

## 为什么必须处理

- **accuracy 陷阱**：1% 正样本下，全预测负类 accuracy=99% 但召回为 0
- **sklearn 默认参数失效**：逻辑回归/树模型默认按均衡分布构造，决策边界整体偏向负类
- **随机会被"基线对比"抓出**：平台评测用 PR-AUC 与多数类基线对比，不处理不平衡的模型通常不如基线

## 处理方案（按优先级）

### 方案 1：类别权重（首选，无数据泄漏风险）

```python
from sklearn.linear_model import LogisticRegression

# class_weight="balanced" 按频率倒数自动加权
clf = LogisticRegression(class_weight="balanced", max_iter=1000)
clf.fit(X_train, y_train)
```

### 方案 2：代价敏感阈值（训练后调阈值，最贴合业务）

模型输出概率，不直接用 0.5 阈值，而是按业务漏报成本选取：

```python
proba = clf.predict_proba(X_test)[:, 1]
# 业务要求召回 >= 90% 时能接受的最低阈值（在训练/验证集上标定）
threshold = _calibrate_threshold(y_val, proba_val, target_recall=0.90)
label = "fraud" if proba >= threshold else "normal"
# threshold 的标定过程与依据必须写入注释
```

### 方案 3：SMOTE 过采样（仅对训练集，严禁对全量数据）

```python
from imblearn.over_sampling import SMOTE
X_train_res, y_train_res = SMOTE(random_state=42).fit_resample(X_train, y_train)
# 错误示范：SMOTE().fit_resample(X, y) 后再 train_test_split —— 测试集被合成样本污染
```

注意：imblearn 是额外依赖，若依赖受限优先用方案 1/2。

## 评估指标（不平衡场景标准）

| 指标 | 用途 |
|---|---|
| PR-AUC | 主指标，聚焦正类性能 |
| Recall @ Precision=X | 业务口径（如精确率 30% 下的召回） |
| KS 统计量 | 风控行业惯例，正负样本分数分布最大间距 |
| F1 / F-beta | β>1 偏重召回时使用 |
| ~~accuracy~~ | 禁用，无参考价值 |
| ~~ROC-AUC~~ | 慎用，极度不平衡下虚高（1% 正样本时随机模型 ROC-AUC≈0.5 但 PR-AUC≈0.01） |

```python
from sklearn.metrics import average_precision_score, roc_auc_score
pr_auc = average_precision_score(y_test, proba)   # PR-AUC（主指标）
```

## sklearn 可用的轻量模型选型

- **IsolationForest**：无标签异常检测（只有正常样本或无标注时），`contamination` 参数按业务预估欺诈率设置
- **LogisticRegression(class_weight="balanced")**：有标签、需可解释（系数即风险方向）
- **GradientBoosting / RandomForest(class_weight="balanced_subsample")**：追求性能，特征重要性可解释

## 常见错误

1. SMOTE/欠采样在 train_test_split **之前**做（数据泄漏，评测指标虚高）
2. 用 accuracy 汇报性能（业务方会被误导）
3. 阈值硬编码 0.5 不标定（欺诈场景最优阈值通常在 0.01~0.3）
4. 时序数据用随机切分而非按时间切分（见 21 篇金融时序）
