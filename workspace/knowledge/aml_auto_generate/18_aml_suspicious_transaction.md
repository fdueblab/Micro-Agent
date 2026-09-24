# 反洗钱与可疑交易监测算法模式

面向跨境支付/银行交易监测场景的算法选型、实现范式与评估方法。适用于 classification（可疑/正常打标签）与 detection（异常评分排序）任务。

## 核心方法论：规则先行 + 模型排序 + 可审计输出

反洗钱场景的行业共识是**混合架构**：先规则与名单硬命中，再对未命中部分用统计/模型评分排序进入复核队列。纯模型方案在该场景不可接受（监管要求决策可解释、可审计）。

```python
def main_process(data: Dict[str, Any]) -> Dict[str, Any]:
    """单笔交易可疑度评估。"""
    # 第一层：名单硬命中（最高优先级，直接可疑）
    if _match_sanction_list(data.get("counterparty_name", "")):
        return _verdict("suspicious", 0.98, "对手方命中制裁名单")

    # 第二层：监管规则硬编码（大额、结构化拆分特征）
    rule_hits = _check_regulatory_rules(data)  # 返回命中的规则描述列表
    if rule_hits:
        return _verdict("suspicious", 0.90, "命中规则: " + "; ".join(rule_hits))

    # 第三层：统计评分（相对同类账户的偏离度）
    score = _statistical_score(data)  # z-score 加权组合
    if score >= 2.5:
        return _verdict("suspicious", min(0.5 + score / 10, 0.89), f"统计偏离度 {score:.2f}")

    return _verdict("normal", 0.7, "未命中名单/规则且统计特征正常")
```

## 典型规则特征（跨境场景）

- **大额规则**：单笔金额超过阈值（如等值 5 万美元）且现金属性
- **结构化拆分**：同对手方短时间内多笔略低于阈值的交易（smurfing 检测：24h 内 N 笔且总额超阈值）
- **地域风险**：对手方国家/地区在高风险名单（FATF 灰名单）
- **行为突变**：交易频率/金额相对该账户历史基线的倍数偏离

## 统计评分特征（无需训练，适配"不训练"约束）

- 金额 z-score（相对全量数据或该账户历史的均值/标准差）
- 时间聚集度：1h/24h/7d 滑动窗口内的交易笔数、金额和
- 对手方集中度：该账户对单一对手方的金额占比
- 跨境属性编码：币种数、涉及国家数、通道类型

## 可审计性要求（监管硬约束）

每个输出必须携带**决策依据**：命中了哪条规则/名单、统计偏离的具体数值。禁止只给标签不给理由：

```python
def _verdict(label: str, confidence: float, reason: str) -> Dict[str, Any]:
    return {
        "classification_label": label,
        "confidence": round(confidence, 4),
        "reason": reason,          # 决策依据，审计必需
        "model_version": "aml_rules_v1",
    }
```

## 评估方法

- **不平衡场景禁用 accuracy**：可疑交易占比通常 <1%，全预测 normal 也有 99% accuracy。使用 PR-AUC、Recall@Precision、KS 统计量
- **业务口径对齐**：误报率（复核队列人力成本）与漏报率（监管处罚风险）的权衡需显式化为阈值参数，并在 limitations 说明
- **基线对比**：与"纯大额阈值规则"对比——模型层必须比简单规则多召回才算有价值

## 常见错误

1. 用 random 生成风险评分（平台反作弊检查直接判不合格）
2. 名单匹配用 `==` 全等而非标准化后的包含/模糊匹配（大小写、空格、别名）
3. 金额比较忘记多币种归一化（见 23 篇金融数据工程）
4. 规则阈值无出处注释（需注明监管依据或数据统计依据）
