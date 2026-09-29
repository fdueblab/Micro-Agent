# 信用评分卡建模模式（WOE/IV 范式）

信用评分是金融风控的经典任务：基于申请人特征预测违约概率，行业有一套标准化建模流程（评分卡范式），输出的分数需可解释、可监管。

## 标准流程（工业界范式）

特征分箱 → WOE 编码 → IV 筛选 → 逻辑回归 → 分数刻度

### 1. 分箱 + WOE 编码

WOE（Weight of Evidence）衡量某分箱内好坏客户的分布差异，同时解决非线性与缺失值（缺失单独一箱）：

```python
def _woe_encode(series: "pd.Series", target: "pd.Series", bins):
    """对单特征分箱并计算每箱 WOE。缺失值单独成箱。"""
    binned = pd.cut(series.fillna(-999), bins, include_lowest=True) if series.dtype != object \
        else series.fillna("missing")
    stats = pd.DataFrame({"bin": binned, "y": target}).groupby("bin")["y"].agg(["sum", "count"])
    bad = stats["sum"]; good = stats["count"] - bad
    # 拉普拉斯平滑防除零
    woe = ((bad + 0.5) / (bad.sum() + 0.5)) / ((good + 0.5) / (good.sum() + 0.5))
    woe = np.log(woe)
    return woe.to_dict()

def _information_value(woe_map, binned, target, total_bad, total_good):
    """IV = Σ (bad% - good%) * WOE。IV<0.02 无预测力，>0.5 需警惕泄漏。"""
```

### 2. IV 特征筛选

| IV 值 | 判断 |
|---|---|
| < 0.02 | 无预测力，剔除 |
| 0.02 ~ 0.1 | 弱 |
| 0.1 ~ 0.3 | 中等 |
| 0.3 ~ 0.5 | 强 |
| > 0.5 | 异常强，**优先怀疑数据泄漏**（如用了贷后才知道的字段） |

### 3. 逻辑回归 + 分数刻度

```python
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(class_weight="balanced", max_iter=1000)
clf.fit(X_woe_train, y_train)  # X 为 WOE 编码后的特征

def _score_from_proba(proba: float, base_score=650, pdo=50) -> int:
    """标准评分刻度：基础分 650，PDO=50（赔率翻倍减 50 分）。"""
    odds = (1 - proba) / max(proba, 1e-9)   # 好坏比
    factor = pdo / np.log(2)
    offset = base_score - factor * np.log(50)  # 20:1 赔率对应基础分
    return int(round(offset + factor * np.log(odds)))
```

## 评估指标（信贷行业标准）

- **KS 统计量**：好坏客户分数分布的最大 separation，行业及格线约 0.3，0.4+ 良好
- **AUC**：≥0.7 可用，≥0.75 良好
- **PSI（群体稳定性）**：上线后监控分数分布漂移，>0.25 需重建模型（生成时可在 holdout 上演示计算）
- 分数段坏账率单调性检查：分数越高违约率必须单调下降

```python
def _ks_statistic(y_true, proba):
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_true, proba)
    return float(np.max(tpr - fpr))
```

## 常见错误

1. 用了**贷后变量**（如逾期天数）预测贷前违约——标签泄漏，IV>0.5 的元凶
2. 缺失值直接 dropna 或填 0（金融数据缺失本身是信号，应单独成箱）
3. 分箱边界用测试集数据标定（泄漏；分箱必须仅基于训练集）
4. 汇报 accuracy 而非 KS/AUC（不平衡场景，见 19 篇）

## 平台适配说明

- pandas/sklearn 均为可用依赖；单文件实现时把分箱边界、WOE 映射表作为**常量表**内嵌（`WOE_TABLES` 字典），`main_process` 查表编码后调用训练好的系数（`COEFFS` 常量）计算分数
- 若用户要求"不训练"约束，则用历史数据预标定的 WOE/系数常量直接打分，并在 model_summary 说明标定数据来源
