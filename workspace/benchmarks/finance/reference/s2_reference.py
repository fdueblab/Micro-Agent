"""S2 信贷违约预测·参考实现（验证不平衡分类通道，正类 8%）。

评分卡式线性评分: 近期查询次数/逾期历史/负债比/信用卡使用率/收入。
规则基线(高负债+高使用率)只能覆盖约 45% 的违约, 低收入多次查询型
需模型才能识别 —— 体现知识库 19 篇(不平衡学习)的领域要求。
"""
from typing import Any, Dict


def main_process(data: Dict[str, Any]) -> Dict[str, Any]:
    """基于评分卡的违约概率评估。

    Args:
        data: 含 debt_ratio/credit_utilization/inquiries_6m/delinquency_history/
              annual_income/employment_years 等特征。

    Returns:
        classification_label(repay/default) + confidence(违约评分) + reason。
    """
    try:
        debt = float(data.get("debt_ratio", 0))
        util = float(data.get("credit_utilization", 0))
        inq = float(data.get("inquiries_6m", 0))
        delinq = float(data.get("delinquency_history", 0))
        income = float(data.get("annual_income", 0))
    except (TypeError, ValueError):
        # 特征非法: 保守判正常(低置信), 避免脏输入导致崩溃
        return {"classification_label": "repay", "confidence": 0.3,
                "reason": "输入特征非法, 保守放行"}

    score = 0.0
    if inq >= 4:
        score += 0.35  # 近期频繁查询: 规则难覆盖的违约信号
    if delinq >= 1:
        score += 0.25  # 逾期历史
    if debt > 0.55:
        score += 0.20  # 高负债
    if util > 0.80:
        score += 0.20  # 高信用卡使用率
    if 0 < income < 40000:
        score += 0.10  # 低收入

    label = "default" if score >= 0.40 else "repay"
    # confidence = 对输出标签的置信度(repay 时反向), 保证排序语义正确
    confidence = score if label == "default" else 1.0 - score
    return {"classification_label": label,
            "confidence": round(min(confidence, 1.0), 4),
            "reason": f"评分 {score:.2f} (查询{inq:.0f}/逾期{delinq:.0f}/负债{debt:.2f}/使用率{util:.2f})"}
