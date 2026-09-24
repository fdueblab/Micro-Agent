# 金融数据工程规范（多币种 / 时间戳 / 聚合特征 / 脏值语义）

金融场景的输入数据有领域特有的坑：多币种、时间戳、高基数类别、语义化缺失。本文是所有金融算法的**数据层公共规范**——在需求分析阶段就要检查这些点。

## 多币种归一化（跨境场景必查）

金额比较/聚合前必须统一币种，否则一切阈值和统计全部失真：

```python
# 模块级汇率表（快照日期必须注明，来源注明）
FX_RATES_TO_USD = {
    "USD": 1.0, "CNY": 0.14, "EUR": 1.08, "JPY": 0.0067, "GBP": 1.27, "HKD": 0.128,
}
FX_SNAPSHOT_DATE = "2026-09-01"  # 汇率快照日期：阈值类规则需注明

def _to_usd(amount, currency: str) -> float:
    """金额归一化到美元。币种未知/金额非法返回 None（不得猜测汇率）。"""
    try:
        amt = float(amount)
    except (TypeError, ValueError):
        return None
    rate = FX_RATES_TO_USD.get(str(currency).upper().strip())
    if rate is None or amt < 0:
        return None
    return amt * rate
```

- 大额阈值规则（如 5 万美元申报线）必须基于归一化后金额判断
- 币种识别失败时返回 None 并降级，**禁止用默认汇率猜**
- 汇率是时变量：严格场景应按交易日期取历史汇率（数据提供时才可做，需在 limitations 说明）

## 时间戳处理

```python
def _parse_ts(raw) -> Optional[str]:
    """兼容 ISO/常见格式，失败返回 None。输出统一 'YYYY-MM-DD HH:MM:SS'。"""
    if not raw:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M", "%Y-%m-%d", "%m/%d/%Y"):
        try:
            return datetime.strptime(str(raw).strip(), fmt).strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
            continue
    return None
```

- 时序特征（滑动窗口、间隔）前必须先解析并**校验解析成功率**；成功率低说明格式假设错误
- 涉及跨时区场景（跨境交易）注明以哪个时区为基准

## 对手方/账户聚合特征（反洗钱核心）

单笔交易维度不足，需聚合到账户/对手方维度：

```python
def _aggregate_features(rows: List[Dict]) -> Dict[str, "pd.DataFrame"]:
    """从交易流水构造账户级聚合特征。"""
    df = pd.DataFrame(rows)
    # 交易流水 → 账户维度：笔数、总额、币种数、对手方数、夜间交易占比
    agg = df.groupby("account_id").agg(
        txn_count=("amount_usd", "size"),
        txn_total=("amount_usd", "sum"),
        currency_nunique=("currency", "nunique"),
        counterparty_nunique=("counterparty", "nunique"),
    )
    # 时间特征需解析成功后计算
    return {"account_agg": agg}
```

典型聚合维度：账户、对手方、国家/地区、日/周/月。

## 语义化缺失（金融数据缺失≠随机）

| 缺失字段 | 语义 | 正确处理 |
|---|---|---|
| 收益字段缺失 | 可能=0（未产生收益） | 按业务确认，0 与 None 区分 |
| 风险等级缺失 | 未知 ≠ 低风险 | 单独"未知"类别/箱，禁止填众数 |
| 收入缺失 | 敏感未申报 | 缺失本身是信号，单独成箱（WOE 范式） |
| 币种缺失 | 数据错误 | 降级处理，禁止默认 USD |

## 量纲与极端值

- 金额跨多个数量级时，模型层输入用 `np.log1p` 压缩（阈值规则仍用原值）
- 极端值（10^9 级）先区分"真实大额"与"脏数据"（如负数、非数值、科学计数法字符串）——**降级而非 clip**（clip 会把真实大额交易截断，恰好放过该抓的对象）

## 检查清单（需求分析阶段逐项过）

1. 金额是否多币种？→ 归一化方案
2. 时间字段什么格式？解析成功率？
3. 是否需要账户/对手方聚合？
4. 缺失字段的业务语义是什么？
5. 类别列（国家/币种/渠道）基数多高？高基数做频次编码而非 one-hot

## 常见错误

1. 多币种直接比较金额（阈值规则全错）
2. 币种/汇率未知时猜一个默认值
3. 缺失填均值/众数（破坏"缺失即信号"的语义）
4. 时序字段当字符串排序（"09/10/2026" 与 "2026-09-10" 混排时字典序错误）
5. 真实大额被 clip 截断（该抓的没抓到）
