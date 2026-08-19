# scikit-learn 分类算法最佳实践

## 概述

scikit-learn（sklearn）是 Python 最成熟的机器学习库，提供多种分类算法实现。本文档介绍在单文件、无需训练的约束下，如何正确使用 sklearn 的预训练模型或规则进行分类。

## 安装

```bash
pip install scikit-learn numpy
```

## 适用场景

- 结构化数据（表格、字典、数值特征）的分类
- 文本分类（需配合特征提取）
- 不适合：图像分类、视频分类（通常需要深度学习框架）

## 常用分类器对比

| 分类器 | 类名 | 优点 | 缺点 | 适用场景 |
|--------|------|------|------|----------|
| 随机森林 | RandomForestClassifier | 抗过拟合、可解释、无需特征缩放 | 模型较大 | 通用分类、表格数据 |
| SVM | SVC | 小样本表现好、核函数灵活 | 大数据慢、需特征缩放 | 中小样本 |
| 逻辑回归 | LogisticRegression | 简单快速、可解释 | 只能线性分类 | 基线模型、概率输出 |
| 朴素贝叶斯 | GaussianNB/MultinomialNB | 极快、适合文本 | 假设特征独立 | 文本分类、实时场景 |
| KNN | KNeighborsClassifier | 简单直观、无需训练 | 预测慢、维度灾难 | 小数据集 |

## 正确使用模式

### 1. 使用预训练模型（无需训练）

如果模型已预训练并保存为 `.pkl` 或 `.joblib` 文件：

```python
import joblib
import numpy as np

def main_process(features: list) -> dict:
    """使用预训练模型进行分类预测。"""
    model = joblib.load('model.pkl')
    X = np.array(features).reshape(1, -1)
    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0]
    confidence = float(max(probability))
    return {
        "classification_label": [str(prediction)],
        "confidence_list": [confidence]
    }
```

### 2. 规则 + sklearn 特征工程

无需训练时，可用 sklearn 的特征处理工具配合规则：

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import numpy as np

# 使用 TfidfVectorizer 将文本转为特征向量
vectorizer = TfidfVectorizer(max_features=100)
# vectorizer 需要先 fit，或加载预训练的 vectorizer

# 使用 StandardScaler 缩放数值特征
scaler = StandardScaler()
# scaler 需要先 fit
```

### 3. 多标签分类

sklearn 原生支持多标签分类（一个样本属于多个类别）：

```python
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier

# 多标签分类器
base_model = RandomForestClassifier(n_estimators=100, random_state=42)
multi_label_model = MultiOutputClassifier(base_model)
# multi_label_model.fit(X, Y)  # Y 形状为 (n_samples, n_labels)
# predictions = multi_label_model.predict(X_test)  # 返回 0/1 矩阵
```

## 输入输出规范

### 输入处理

sklearn 模型的输入必须是数值型 numpy 数组：

```python
# 正确：将字典转为特征数组
def dict_to_features(data: dict) -> np.ndarray:
    """将结构化数据字典转为特征向量。"""
    numeric_values = [float(v) for v in data.values() if isinstance(v, (int, float))]
    return np.array(numeric_values, dtype=float).reshape(1, -1)

# 正确：处理文本特征
from sklearn.feature_extraction.text import TfidfVectorizer

def text_to_features(text: str) -> np.ndarray:
    """将文本转为 TF-IDF 特征向量。"""
    # 注意：vectorizer 需要预先 fit
    # vectorizer = TfidfVectorizer().fit(corpus)
    # return vectorizer.transform([text]).toarray()
    pass  # 实现取决于预训练 vectorizer
```

### 输出格式

分类算法的标准输出格式：

```python
# 单标签分类
return {
    "classification_label": [predicted_label],  # 单个标签
    "confidence_list": [confidence_score]      # 对应置信度
}

# 多标签分类
return {
    "classification_label": ["标签A", "标签B"],  # 多个标签
    "confidence_list": [0.85, 0.72]             # 各标签置信度
}
```

## 关键注意事项

1. **random_state 必须固定**：`RandomForestClassifier(random_state=42)` 确保结果可复现
2. **predict_proba 并非所有分类器都有**：SVC 需设置 `probability=True` 才有 `predict_proba`
3. **输入维度匹配**：预测时的特征维度必须与训练时一致，使用 `reshape(1, -1)` 处理单样本
4. **标签编码**：如果使用 LabelEncoder，预测后需要 `inverse_transform` 还原原始标签名
5. **NaN/Inf 处理**：sklearn 不接受 NaN 输入，需要 `SimpleImputer` 预处理
6. **类别不平衡**：设置 `class_weight='balanced'` 或使用 `sklearn.utils.resample`

## 参考文档

- scikit-learn 官方文档: https://scikit-learn.org/stable/supervised_learning.html
- RandomForestClassifier: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
- 多标签分类: https://scikit-learn.org/stable/modules/multiclass.html
