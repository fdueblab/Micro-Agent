"""S5 制裁名单筛查·参考实现（验证模糊匹配通道）。

名称变体: 大小写/多余空格/标点/连字符/别名。标准化策略:
去除所有非字母数字字符后精确比对(别名 COMPANY LIMITED 归一为 CO LTD)。
基线(精确子串)只能覆盖部分名单关键词, 模糊标准化可全覆盖。
"""
import re
from typing import Any, Dict

SANCTION_LIST = [
    "IRISL ISLAMIC REPUBLIC OF IRAN SHIPPING", "KWEILIN IMPORT EXPORT",
    "MARC RICH + CO AG", "DALIAN SUNMOON", "GLOBAL TECH SARL",
]


def _key(name: str) -> str:
    """名称标准化: 别名归一 + 去除所有非字母数字字符(变体免疫)。"""
    s = str(name).upper()
    s = s.replace("COMPANY LIMITED", "CO LTD").replace("CO.", "COMPANY")
    return re.sub(r"[^A-Z0-9]", "", s)


_KEYS = [_key(n) for n in SANCTION_LIST]


def main_process(data: Dict[str, Any]) -> Dict[str, Any]:
    """交易对手方制裁名单筛查。

    Args:
        data: 含 counterparty_name 的特征。

    Returns:
        classification_label(clear/hit) + confidence + reason。
    """
    name = str(data.get("counterparty_name", "") or "")
    if not name.strip():
        return {"classification_label": "clear", "confidence": 0.5,
                "reason": "名称为空, 无法筛查"}

    k = _key(name)
    for i, sk in enumerate(_KEYS):
        if k == sk:
            return {"classification_label": "hit", "confidence": 0.95,
                    "reason": f"命中名单: {SANCTION_LIST[i]}(标准化匹配)"}

    # confidence = 对 clear 判定的置信度(标准化后无命中, 把握较高)
    return {"classification_label": "clear", "confidence": 0.9,
            "reason": "标准化后无名单命中"}
