"""S3 汇率次日预测·参考实现（验证时序评测通道）。

动量信号: 价格相对 5 日均线的偏离 (last-mean5)/mean5。
汇率日波动 ~1.2%, 轻微动量可提取方向信号, 但幅度预测仍受噪声主导
(与朴素基线 last_close 相当), 符合知识库 21 篇的领域共识。
"""
from typing import Any, Dict


def main_process(data: Dict[str, Any]) -> Dict[str, Any]:
    """基于动量信号的下一期收盘价预测。

    Args:
        data: 含 last_close/mean5/mean20/vol20/volume_z/weekday 的特征。

    Returns:
        prediction(下一期收盘价) + confidence + reason。
    """
    try:
        last = float(data.get("last_close", 0))
        mean5 = float(data.get("mean5", last))
    except (TypeError, ValueError):
        return {"prediction": None, "confidence": 0.0, "reason": "价格特征非法"}

    if last <= 0 or mean5 <= 0:
        return {"prediction": None, "confidence": 0.0, "reason": "价格非正, 无法预测"}

    # 5日均线偏离作为动量代理(与真实4日动量相关性~0.94), 均值回归+趋势的混合信号
    momentum = (last - mean5) / mean5
    # 动量系数 0.20: 信号幅度温和, 避免噪声放大
    pred = last * (1 + 0.20 * momentum)
    return {"prediction": round(pred, 4), "confidence": 0.5,
            "reason": f"动量信号 {momentum:+.4f}"}
