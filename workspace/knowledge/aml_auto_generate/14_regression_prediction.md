# 回归与预测算法

## 概述

回归分析用于预测连续数值。本文档介绍在单文件、无训练约束下，使用 scikit-learn 和 statsmodels 进行回归预测的正确方法。

## 安装

```bash
pip install scikit-learn numpy statsmodels
```

## 适用场景

| 场景 | 推荐方法 | 库 |
|------|----------|-----|
| 线性关系预测 | 线性回归 | sklearn |
| 带正则化的回归 | Ridge/Lasso | sklearn |
| 非线性关系 | 多项式回归 | sklearn |
| 时间序列预测 | ARIMA/SARIMA | statsmodels |
| 简单规则预测 | 均值/趋势外推 | numpy |

## scikit-learn 回归

### 1. 线性回归

```python
from sklearn.linear_model import LinearRegression
import numpy as np

# 模型必须预先训练并保存，或使用预训练模型
# model = LinearRegression()
# model.fit(X_train, y_train)

# 预测时加载预训练模型
# import joblib
# model = joblib.load('regression_model.pkl')
# prediction = model.predict(X_test)
```

### 2. Ridge 回归（L2 正则化）

```python
from sklearn.linear_model import Ridge

# Ridge 回归通过 alpha 参数控制正则化强度
# alpha=0 等价于普通线性回归
model = Ridge(alpha=1.0, random_state=42)
# model.fit(X_train, y_train)
# prediction = model.predict(X_test)
```

### 3. 多项式回归

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
import numpy as np

# 二次多项式回归
model = Pipeline([
    ('poly', PolynomialFeatures(degree=2)),
    ('linear', LinearRegression())
])
# model.fit(X_train, y_train)
# prediction = model.predict(X_test)
```

## 规则化预测（无需训练）

当不允许训练时，可以使用统计规则进行预测：

### 1. 均值预测

```python
import numpy as np

def predict_by_mean(historical_data: list) -> dict:
    """基于历史均值的简单预测。"""
    if not historical_data:
        return {"predicted_value": 0.0, "method": "mean", "confidence": 0.0}
    
    values = np.array(historical_data, dtype=float)
    mean_val = float(np.mean(values))
    std_val = float(np.std(values))
    
    # 置信度基于数据离散程度：标准差越小，预测越可靠
    if mean_val != 0:
        cv = std_val / abs(mean_val)  # 变异系数
        confidence = max(0.0, min(1.0, 1.0 - cv))
    else:
        confidence = 0.5
    
    return {
        "predicted_value": mean_val,
        "method": "mean",
        "confidence": round(confidence, 3),
        "std_dev": round(std_val, 3)
    }
```

### 2. 移动平均预测

```python
import numpy as np

def predict_by_moving_average(
    historical_data: list, 
    window: int = 5
) -> dict:
    """移动平均预测。"""
    if len(historical_data) < window:
        return predict_by_mean(historical_data)
    
    values = np.array(historical_data, dtype=float)
    recent_values = values[-window:]
    prediction = float(np.mean(recent_values))
    
    # 基于近期波动计算置信度
    recent_std = float(np.std(recent_values))
    if prediction != 0:
        confidence = max(0.0, min(1.0, 1.0 - recent_std / abs(prediction)))
    else:
        confidence = 0.5
    
    return {
        "predicted_value": round(prediction, 3),
        "method": f"moving_average_{window}",
        "confidence": round(confidence, 3),
        "window": window
    }
```

### 3. 线性趋势外推

```python
import numpy as np

def predict_by_trend(historical_data: list, steps_ahead: int = 1) -> dict:
    """基于线性趋势外推预测。"""
    values = np.array(historical_data, dtype=float)
    n = len(values)
    
    if n < 2:
        return predict_by_mean(historical_data)
    
    # 最小二乘法拟合线性趋势
    x = np.arange(n)
    # y = a*x + b
    coefficients = np.polyfit(x, values, 1)
    slope, intercept = coefficients
    
    # 预测未来值
    future_x = n + steps_ahead - 1
    prediction = float(slope * future_x + intercept)
    
    # 计算拟合优度 R²
    predicted = slope * x + intercept
    ss_res = np.sum((values - predicted) ** 2)
    ss_tot = np.sum((values - np.mean(values)) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot != 0 else 0
    
    confidence = max(0.0, min(1.0, float(r_squared)))
    
    return {
        "predicted_value": round(prediction, 3),
        "method": "linear_trend",
        "confidence": round(confidence, 3),
        "trend": "increasing" if slope > 0 else "decreasing" if slope < 0 else "stable",
        "slope": round(float(slope), 6)
    }
```

## statsmodels 时间序列

### ARIMA 模型

ARIMA (AutoRegressive Integrated Moving Average) 是经典时间序列预测方法。

```python
# 需要安装 statsmodels
# pip install statsmodels

import numpy as np

def predict_by_arima(
    historical_data: list,
    order: tuple = (1, 1, 1)
) -> dict:
    """
    ARIMA 预测。
    
    Args:
        historical_data: 历史数据列表
        order: (p, d, q) 参数
            p: AR 阶数（自回归项数）
            d: 差分阶数（使序列平稳）
            q: MA 阶数（移动平均项数）
    
    Returns:
        dict: 预测结果
    """
    try:
        from statsmodels.tsa.arima.model import ARIMA
        import warnings
        warnings.filterwarnings('ignore')
        
        values = np.array(historical_data, dtype=float)
        
        # 拟合 ARIMA 模型
        model = ARIMA(values, order=order)
        fitted = model.fit()
        
        # 预测下一步
        forecast = fitted.forecast(steps=1)
        prediction = float(forecast[0])
        
        # 置信区间
        conf_int = fitted.get_forecast(steps=1).conf_int(alpha=0.05)
        lower = float(conf_int[0, 0])
        upper = float(conf_int[0, 1])
        
        # 区间宽度反映不确定性
        interval_width = upper - lower
        data_range = max(values) - min(values) if len(values) > 1 else 1
        confidence = max(0.0, min(1.0, 1.0 - interval_width / data_range)) if data_range > 0 else 0.5
        
        return {
            "predicted_value": round(prediction, 3),
            "method": f"ARIMA{order}",
            "confidence": round(confidence, 3),
            "confidence_interval": [round(lower, 3), round(upper, 3)]
        }
    except Exception as e:
        # ARIMA 拟合失败时回退到趋势预测
        result = predict_by_trend(historical_data)
        result["method"] = f"arima_fallback_to_trend"
        result["error"] = str(e)
        return result
```

## main_process 标准接口

回归/预测算法的 main_process 应返回预测值和置信度：

```python
def main_process(historical_data: list = None, **kwargs) -> dict:
    """
    主预测函数。
    
    Args:
        historical_data: 历史数据列表，如 [100, 120, 115, 130, 125]
        **kwargs: 其他参数（预测步数等）
    
    Returns:
        dict: 包含预测值和置信度的结果
    """
    if not historical_data:
        return {
            "predicted_value": None,
            "method": "no_data",
            "confidence": 0.0,
            "message": "未提供历史数据，无法预测"
        }
    
    # 使用线性趋势外推
    result = predict_by_trend(historical_data)
    
    return result
```

## 关键注意事项

1. **数据长度**：ARIMA 需要足够长的历史数据（至少 10 个点），数据不足时回退到简单方法
2. **异常值处理**：异常值会严重影响回归结果，应使用中位数或 IQR 过滤
3. **平稳性检查**：时间序列有趋势或季节性时，需先差分（ARIMA 的 d 参数）
4. **预测区间**：预测值必须给出置信区间，单点预测没有意义
5. **外推风险**：线性趋势外推在长期预测中不可靠，预测步数不宜过多
6. **NaN 检查**：历史数据中可能存在 None/NaN，需要清理后再预测

## 参考文档

- scikit-learn 回归: https://scikit-learn.org/stable/modules/linear_model.html
- statsmodels ARIMA: https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMA.html
- 时间序列分析: https://www.statsmodels.org/stable/tsa.html
