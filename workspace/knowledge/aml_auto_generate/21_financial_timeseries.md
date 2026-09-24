# 金融时间序列预测模式（汇率/价格/交易量）

面向汇率预测、价格走势、交易量预测等任务。金融时序的核心纪律是**防时序泄漏**与**战胜朴素基线**——绝大多数时序模型死于这两点。

## 铁律 1：按时间切分，禁止随机切分

随机 train_test_split 会把"未来"放进训练集，指标虚高但实盘失效：

```python
# 错误（泄漏）：
# X_train, X_test = train_test_split(X, y, test_size=0.2, shuffle=True)

# 正确：前 80% 时间训练，后 20% 时间测试
split_idx = int(len(df) * 0.8)
train, test = df.iloc[:split_idx], df.iloc[split_idx:]
```

滚动验证（walk-forward）更稳健：

```python
def _walk_forward_validate(df, window=60, horizon=1):
    """滚动窗口验证：每次用过去 window 天预测下 horizon 天。"""
    errors = []
    for i in range(window, len(df) - horizon, 10):
        train = df.iloc[i - window:i]
        actual = df.iloc[i + horizon - 1]["close"]
        pred = _naive_or_model_forecast(train, horizon)
        errors.append(abs(pred - actual) / actual)
    return float(np.mean(errors))  # MAPE
```

## 铁律 2：必须对比朴素基线（否则没有价值）

金融时序的朴素基线（naive/persistence）很强——"明天价格=今天价格"。模型打不过它就等于没用：

```python
def _naive_baseline(test: "pd.DataFrame") -> float:
    """朴素基线 MAPE：用最后一期值预测全部。"""
    last = test["close"].iloc[0]
    mape = np.mean(np.abs(test["close"] - last) / test["close"]) * 100
    return float(mape)
```

汇报时必须并列：模型 MAPE vs 朴素基线 MAPE。模型略差于朴素基线时，诚实汇报并说明（汇率日频数据接近随机游走，朴素基线极难战胜，这是领域共识而非模型缺陷）。

## 特征工程（时序安全特征）

只能用**预测时点已知**的信息构造特征：

- 滞后特征：`close.shift(1)`、`shift(5)`（滞后阶数必须 ≥ 预测步长）
- 滚动统计：过去 5/20/60 期的均值、波动率、最大回撤
- 日历特征：星期几、是否月末/季末（跨境结算周期效应）
- **禁止**：当期 open/high/low 预测当期 close（当期信息）；全序列均值/标准差（含未来）

## 模型选型（按复杂度递增）

1. **朴素/季节性朴素**：基线必备，也是 no_training 约束下的合理方案
2. **线性回归 + 滞后特征**：可解释，汇率场景常胜过复杂模型
3. **statsmodels ARIMA**：单变量经典（注意依赖声明 `statsmodels`）
4. **sklearn GradientBoosting + 滞后/滚动特征**：多变量场景主力，无需深度学习依赖

## 评估指标

| 指标 | 用途 |
|---|---|
| MAPE | 主指标（比例误差，跨量纲可比） |
| RMSE / MAE | 绝对误差（需注明币种/单位） |
| 方向准确率 | 涨跌方向判对比例（>55% 即有信息量，50% 等于抛硬币） |

## 输出契约

```python
def main_process(data: Dict[str, Any]) -> Dict[str, Any]:
    """输入含历史序列，输出下一期预测。"""
    # 必须校验历史长度 >= 最小窗口，不足时降级返回
    hist = data.get("history", [])
    if len(hist) < 20:
        return {"prediction": None, "confidence": 0.0,
                "reason": "历史序列不足 20 期，无法预测", ...}
```

## 常见错误

1. 随机切分/特征含未来信息（平台时序泄露检查 + 复盘时用"预测日当天能看到什么"自检）
2. 只报模型指标不报朴素基线
3. 用 accuracy 衡量回归任务
4. 缺历史数据时崩溃而非降级（脏输入防御，见 17 篇）
5. 深度学习（LSTM）依赖 torch——单文件交付场景优先轻量方案，确需时声明版本上界
