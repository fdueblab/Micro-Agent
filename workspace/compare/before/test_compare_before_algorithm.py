"""
交易流水风险分类算法
用于对交易流水进行风险等级分类，支持正常、可疑、高风险、疑似欺诈四类标签。
"""

import re
from collections import Counter

# 定义分类标签
LABELS = ["正常", "可疑", "高风险", "疑似欺诈"]

# 高风险关键词
HIGH_RISK_KEYWORDS = [
    "转账", "汇款", "充值", "提现", "赌博", "投资", "理财", "贷款", "套现", "刷单"
]

# 可疑关键词
SUSPICIOUS_KEYWORDS = [
    "退款", "返现", "优惠券", "红包", "抽奖", "中奖", "返利", "返点", "佣金", "提成"
]

# 疑似欺诈关键词
FRAUD_KEYWORDS = [
    "诈骗", "盗刷", "冒用", "身份", "验证码", "密码", "账户", "冻结", "异常", "安全"
]

# 异常金额阈值
HIGH_AMOUNT_THRESHOLD = 10000.0
VERY_HIGH_AMOUNT_THRESHOLD = 50000.0

def classify_by_amount(amount: float) -> str:
    """
    根据交易金额初步分类
    
    Args:
        amount: 交易金额
        
    Returns:
        初步分类标签
    """
    if amount >= VERY_HIGH_AMOUNT_THRESHOLD:
        return "高风险"
    elif amount >= HIGH_AMOUNT_THRESHOLD:
        return "可疑"
    else:
        return "正常"

def classify_by_keywords(description: str) -> str:
    """
    根据描述文本中的关键词分类
    
    Args:
        description: 交易描述文本
        
    Returns:
        关键词分类标签
    """
    if not isinstance(description, str):
        return "正常"
    
    desc_lower = description.lower()
    high_count = sum(1 for word in HIGH_RISK_KEYWORDS if word in desc_lower)
    suspicious_count = sum(1 for word in SUSPICIOUS_KEYWORDS if word in desc_lower)
    fraud_count = sum(1 for word in FRAUD_KEYWORDS if word in desc_lower)
    
    # 统计各类型关键词数量
    counts = {
        "高风险": high_count,
        "可疑": suspicious_count,
        "疑似欺诈": fraud_count
    }
    
    # 返回最高频的标签
    max_label = max(counts, key=counts.get)
    if counts[max_label] > 0:
        return max_label
    else:
        return "正常"

def combine_classification(amount_label: str, keyword_label: str) -> dict:
    """
    综合金额和关键词分类结果
    
    Args:
        amount_label: 金额分类标签
        keyword_label: 关键词分类标签
        
    Returns:
        包含最终标签和置信度的字典
    """
    # 置信度计算逻辑
    if amount_label == keyword_label:
        confidence = 0.9
    elif (amount_label == "高风险" and keyword_label == "可疑") or \
         (amount_label == "可疑" and keyword_label == "高风险"):
        confidence = 0.8
    elif amount_label == "正常" and keyword_label != "正常":
        confidence = 0.7
    elif amount_label != "正常" and keyword_label == "正常":
        confidence = 0.7
    else:
        confidence = 0.6
    
    # 最终标签选择：优先考虑关键词分类，其次金额分类
    final_label = keyword_label if keyword_label != "正常" else amount_label
    
    # 确保标签在允许范围内
    if final_label not in LABELS:
        final_label = "正常"
        
    return {
        "label": final_label,
        "confidence": round(confidence, 2)
    }

def main_process(amount: float, description: str) -> dict:
    """
    交易流水风险分类主入口
    
    Args:
        amount: 交易金额
        description: 交易描述文本
        
    Returns:
        包含分类标签和置信度的字典
    """
    # 输入校验
    if not isinstance(amount, (int, float)):
        raise ValueError("金额必须是数字")
    if not isinstance(description, str):
        raise ValueError("描述必须是字符串")
    
    # 分别进行分类
    amount_label = classify_by_amount(amount)
    keyword_label = classify_by_keywords(description)
    
    # 合并结果
    result = combine_classification(amount_label, keyword_label)
    
    return result

if __name__ == "__main__":
    # 示例调用
    print(main_process(100000.0, "用户进行大额转账"))
    print(main_process(50.0, "购买商品"))
    print(main_process(1000.0, "收到中奖通知"))
