"""S4 贸易单证审核·参考实现（验证脏值防御通道）。

四类违规: 币种不一致/金额异常(负数或超大)/原产地缺失/超许可证额度。
数据负类中混入 8% 字符串金额脏值 —— 正确行为是解析出数值正常审核,
误判为"金额异常"会产生误报(考察宁缺毋滥原则)。
"""
import re
from typing import Any, Dict


def _parse_amount(v: Any) -> "float | None":
    """解析金额: 数值直接返回; 字符串(如 'USD 1234.56')提取数值; 失败返回 None。"""
    if isinstance(v, (int, float)):
        return float(v)
    m = re.search(r"-?\d+(?:\.\d+)?", str(v))
    return float(m.group()) if m else None


def main_process(data: Dict[str, Any]) -> Dict[str, Any]:
    """单证合规审核: 逐项检查四类违规, 每条结论附带依据。

    Args:
        data: 含 invoice_currency/declared_currency/amount/origin_country/license_limit。

    Returns:
        classification_label(pass/reject) + confidence + reason(违规依据)。
    """
    inv = str(data.get("invoice_currency", "") or "").strip().upper()
    dec = str(data.get("declared_currency", "") or "").strip().upper()
    origin = str(data.get("origin_country", "") or "").strip()
    try:
        limit = float(data.get("license_limit", 0) or 0)
    except (TypeError, ValueError):
        limit = 0.0

    amount = _parse_amount(data.get("amount"))
    if amount is None:
        # 脏值防御: 无法解析时中性置信度放行(宁缺毋滥), 不猜测金额
        return {"classification_label": "pass", "confidence": 0.5,
                "reason": "金额字段无法解析, 低置信放行待人工复核"}

    reasons = []
    if inv and dec and inv != dec:
        reasons.append(f"币种不一致({inv} vs {dec})")
    if amount < 0:
        reasons.append(f"金额为负({amount})")
    if amount > 500000:
        reasons.append(f"金额异常超大({amount})")
    if not origin:
        reasons.append("原产地缺失")
    if limit > 0 and amount > limit:
        reasons.append(f"超许可证额度({amount} > {limit})")

    if reasons:
        return {"classification_label": "reject",
                "confidence": round(min(0.6 + 0.15 * len(reasons), 0.95), 4),
                "reason": "; ".join(reasons)}
    return {"classification_label": "pass", "confidence": 0.8, "reason": "单证合规"}
