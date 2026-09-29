"""S1 跨境可疑交易监测·参考实现（用于验证基准引擎的正确性）。

设计要点（对齐知识库 18/23 篇规范）:
    - 多币种自行归一化（数据集不提供 amount_usd 给模型）
    - 规则可抓模式（大额/高风险国/多币种）+ 规则难抓模式（小额高频拆分）双通道评分
    - 输出带 reason，脏输入优雅降级
"""
from typing import Any, Dict

FX_TO_USD = {"USD": 1.0, "CNY": 0.14, "EUR": 1.08, "JPY": 0.0067, "GBP": 1.27, "HKD": 0.128}
HIGH_RISK = {"IR", "KP", "SY", "MM", "AF"}


def main_process(data: Dict[str, Any]) -> Dict[str, Any]:
    """单笔交易可疑度评估。

    Args:
        data: 含 amount/currency/counterparty_country/txn_count_24h/
              distinct_currency_7d/avg_amount_30d/night_flag 的交易特征。

    Returns:
        classification_label(normal/suspicious) + confidence + reason。
    """
    if not isinstance(data, dict):
        return _downgrade("输入非对象")

    try:
        amount = float(data.get("amount", 0))
    except (TypeError, ValueError):
        return _downgrade("金额字段非法")
    currency = str(data.get("currency", "") or "").upper().strip()
    rate = FX_TO_USD.get(currency)
    if rate is None or amount < 0:
        return _downgrade("币种未知或金额非法")
    amount_usd = amount * rate

    country = str(data.get("counterparty_country", "") or "").upper().strip()
    try:
        txn_24h = int(float(data.get("txn_count_24h", 0) or 0))
        curr_7d = int(float(data.get("distinct_currency_7d", 0) or 0))
    except (TypeError, ValueError):
        return _downgrade("统计字段非法")

    reasons, score = [], 0.0

    # 通道1: 大额(阈值来源: FATF 大额现金标准 1 万美元等值)
    if amount_usd > 10000:
        score += 0.45
        reasons.append(f"大额 ${amount_usd:.0f}")
    # 通道2: 高风险地区
    if country in HIGH_RISK:
        score += 0.30
        reasons.append(f"高风险地区 {country}")
    # 通道3: 多币种分散
    if curr_7d >= 3:
        score += 0.20
        reasons.append(f"7日涉及 {curr_7d} 币种")
    # 通道4: 小额高频拆分(规则基线无法覆盖的模式, 权重最高)
    if txn_24h >= 12:
        score += 0.55
        reasons.append(f"24h 内 {txn_24h} 笔高频")
    elif txn_24h >= 8:
        score += 0.25
        reasons.append(f"24h 内 {txn_24h} 笔")

    score = min(score, 0.97)
    if score >= 0.5:
        return {"classification_label": "suspicious", "confidence": round(score, 4),
                "reason": "; ".join(reasons) or "综合统计偏离"}
    return {"classification_label": "normal", "confidence": round(1 - score, 4),
            "reason": "未命中可疑模式"}


def _downgrade(reason: str) -> Dict[str, Any]:
    """脏输入统一降级契约。"""
    return {"classification_label": "normal", "confidence": 0.0, "reason": f"无法判定: {reason}"}
