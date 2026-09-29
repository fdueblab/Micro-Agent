# 跨境贸易合规算法模式（名单筛查 / HS编码 / 规则引擎）

面向海关申报、制裁合规、贸易单证审核等场景。这类任务的行业特点是**规则驱动为主、模糊匹配为核心技术、可追溯为硬约束**，多数需求可用纯规则实现（天然适配平台的 rule_based / no_training 约束）。

## 场景 1：制裁/风险名单筛查

核心技术是**实体名称模糊匹配**（真实名单匹配从不全等：大小写、空格、别名、音译、缩写差异）：

```python
import re

def _normalize_name(name: str) -> str:
    """名称标准化：去标点、压空白、统一大小写、常见别名映射。"""
    s = re.sub(r"[^\w\s]", " ", str(name).upper())
    s = re.sub(r"\s+", " ", s).strip()
    ALIAS = {"CO LTD": "COMPANY LIMITED", "LTD": "LIMITED"}  # 片段级别名
    for k, v in ALIAS.items():
        s = s.replace(k, v)
    return s

def _match_list(name: str, sanction_list: list) -> float:
    """返回匹配分：全等 1.0 / 包含 0.9 / 高重叠 token 0.7+。"""
    n = _normalize_name(name)
    for entry in sanction_list:
        e = _normalize_name(entry)
        if n == e:
            return 1.0
        if n in e or e in n:
            return 0.9
        overlap = len(set(n.split()) & set(e.split())) / max(len(set(e.split())), 1)
        if overlap >= 0.8:
            return 0.7 + 0.2 * (overlap - 0.8)
    return 0.0
```

匹配分超阈值即命中，输出必须携带**命中的名单条目**（审计要求）。

## 场景 2：HS 编码智能归类

根据品名/材质/用途文本推荐海关 HS 编码：

```python
HS_RULES = [
    # (关键词集合(命中任意), HS编码, 品类说明, 税率参考)
    ({"集成电路", "芯片", "半导体"}, "854231", "集成电路", "0%"),
    ({"笔记本电脑", "便携式计算机"}, "847130", "便携式自动数据处理设备", "0%"),
    ({"棉花", "原棉", "皮棉"}, "520100", "未梳棉花", "配额内1%"),
]

def _classify_hs(goods_desc: str) -> Dict[str, Any]:
    """基于品名文本的 HS 归类：关键词规则 + 置信度分层。"""
    text = str(goods_desc)
    for keywords, code, desc, duty in HS_RULES:
        hits = [k for k in keywords if k in text]
        if hits:
            confidence = min(0.6 + 0.15 * len(hits), 0.95)
            return {"hs_code": code, "confidence": confidence,
                    "reason": f"品名命中关键词: {', '.join(hits)}"}
    # 未命中：低置信度兜底（禁止瞎猜编码，申报错误有合规责任）
    return {"hs_code": None, "confidence": 0.0,
            "reason": "品名未命中已知品类规则，需人工归类"}
```

## 场景 3：单证审核规则引擎

报关/结算单证的跨字段一致性校验，规则表驱动：

```python
DOC_RULES = [
    ("币种一致", lambda d: d.get("invoice_currency") == d.get("declared_currency")),
    ("金额区间", lambda d: 0 < float(d.get("amount", 0) or 0) <= 1e7),
    ("原产地必填", lambda d: bool(str(d.get("origin", "") or "").strip())),
]

def _audit_document(doc: Dict[str, Any]) -> Dict[str, Any]:
    failures = [name for name, check in DOC_RULES if not _safe_check(check, doc)]
    passed = len(DOC_RULES) - len(failures)
    return {
        "classification_label": "pass" if not failures else "reject",
        "confidence": passed / len(DOC_RULES),
        "reason": "全部通过" if not failures else f"未通过: {', '.join(failures)}",
    }
```

`_safe_check` 必须捕获字段缺失/类型错误，转为该规则不通过而非崩溃（脏输入防御）。

## 设计约束

1. **规则表与逻辑分离**：`HS_RULES`/`DOC_RULES` 作为模块级常量，便于核对与扩展；每条规则注明来源（监管文件/业务制度）
2. **宁缺毋滥**：名单/归类未命中时返回 None + "需人工处理"，禁止无依据兜底（合规场景错误命中的代价高于漏报的代价由业务权衡，但瞎猜编码是绝对红线）
3. **每条结论带依据**：命中关键词/规则名/名单条目必须出现在 reason 中

## 评估方法

- 用标注样本算精确率/召回率（匹配类任务两者都要看，业务上漏放行 vs 误拦截代价不同）
- 报告**规则覆盖率**：多少比例样本落入"无法判定需人工"——这个比例本身就是业务指标（自动化率）

## 常见错误

1. 名单用 `==` 全等匹配（真实数据几乎永远匹配不上）
2. 未命中时随机选一个编码/强行输出（合规红线，平台无依据兜底禁令直接覆盖）
3. 规则阈值无出处注释
4. 金额/币种字段未做脏值防御（单证数据脏值极普遍）
