"""金融垂域评测基准：合成数据生成器（固定种子，可复现）。

用法:
    python gen_datasets.py            # 在 datasets/ 下生成 5 个想定的 CSV
数据设计原则:
    - 每个想定都内置一个"金融规则/朴素基线"可计算的特征列（amount_usd/debt 组合/last_close 等）
    - 正类除规则可捕获的部分外，还包含规则难以覆盖的模式（如小额高频拆分、名称变体），
      使得优秀模型必须优于基线才算达标
"""
from __future__ import annotations

import csv
import random
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "datasets"
SEED = 20260920

CURRENCIES = ["USD", "CNY", "EUR", "JPY", "GBP", "HKD"]
FX_TO_USD = {"USD": 1.0, "CNY": 0.14, "EUR": 1.08, "JPY": 0.0067, "GBP": 1.27, "HKD": 0.128}
HIGH_RISK = {"IR", "KP", "SY", "MM", "AF"}
NORMAL_COUNTRIES = ["CN", "US", "DE", "JP", "SG", "GB", "AU", "KR", "FR", "CA"]


def _w(v: float) -> float:
    return round(v, 2)


def gen_s1(n: int = 1200, pos_rate: float = 0.04) -> list[dict]:
    """S1 可疑交易：正类 = 大额规则可抓部分(~40%) + 小额高频拆分模式(~60%)。"""
    rows = []
    n_pos = int(n * pos_rate)
    labels = ["suspicious"] * n_pos + ["normal"] * (n - n_pos)
    for i, lab in enumerate(labels):
        if lab == "suspicious" and random.random() < 0.6:
            # 规则难覆盖：小额高频拆分（smurfing）—— 单笔小但 24h 笔数极高
            row = dict(
                txn_id=f"T{i:05d}", amount=_w(random.uniform(800, 4800)),
                currency=random.choice(CURRENCIES), counterparty_country=random.choice(NORMAL_COUNTRIES),
                txn_count_24h=random.randint(14, 40), distinct_currency_7d=random.randint(3, 6),
                avg_amount_30d=_w(random.uniform(200, 900)), night_flag=random.choice([0, 1]),
            )
        elif lab == "suspicious":
            # 规则可抓：大额 + 高风险地区/多币种
            row = dict(
                txn_id=f"T{i:05d}", amount=_w(random.uniform(11000, 90000)),
                currency=random.choice(CURRENCIES), counterparty_country=random.choice(list(HIGH_RISK)),
                txn_count_24h=random.randint(1, 6), distinct_currency_7d=random.randint(2, 5),
                avg_amount_30d=_w(random.uniform(1000, 5000)), night_flag=random.randint(0, 1),
            )
        else:
            row = dict(
                txn_id=f"T{i:05d}", amount=_w(random.uniform(50, 8000)),
                currency=random.choice(CURRENCIES), counterparty_country=random.choice(NORMAL_COUNTRIES),
                txn_count_24h=random.randint(1, 8), distinct_currency_7d=random.randint(1, 2),
                avg_amount_30d=_w(random.uniform(100, 4000)), night_flag=random.choice([0, 1]),
            )
        row["amount_usd"] = _w(row["amount"] * FX_TO_USD[row["currency"]])
        row["label"] = lab
        rows.append(row)
    random.shuffle(rows)
    return rows


def gen_s2(n: int = 1500, pos_rate: float = 0.08) -> list[dict]:
    """S2 信贷违约：正类 = 高负债高使用率(规则可抓~45%) + 低收入多次查询(规则难抓~55%)。"""
    rows = []
    n_pos = int(n * pos_rate)
    labels = ["default"] * n_pos + ["repay"] * (n - n_pos)
    for i, lab in enumerate(labels):
        if lab == "default" and random.random() < 0.55:
            row = dict(
                customer_id=f"C{i:05d}", loan_amount=_w(random.uniform(5000, 30000)),
                annual_income=_w(random.uniform(15000, 45000)),
                debt_ratio=_w(random.uniform(0.15, 0.5)),
                credit_utilization=_w(random.uniform(0.2, 0.75)),
                inquiries_6m=random.randint(5, 12), delinquency_history=random.randint(1, 3),
                employment_years=_w(random.uniform(0.3, 2.0)),
            )
        elif lab == "default":
            row = dict(
                customer_id=f"C{i:05d}", loan_amount=_w(random.uniform(10000, 60000)),
                annual_income=_w(random.uniform(30000, 70000)),
                debt_ratio=_w(random.uniform(0.58, 0.9)),
                credit_utilization=_w(random.uniform(0.82, 1.0)),
                inquiries_6m=random.randint(0, 3), delinquency_history=random.choice([0, 1]),
                employment_years=_w(random.uniform(1.0, 6.0)),
            )
        else:
            row = dict(
                customer_id=f"C{i:05d}", loan_amount=_w(random.uniform(5000, 50000)),
                annual_income=_w(random.uniform(40000, 200000)),
                debt_ratio=_w(random.uniform(0.05, 0.45)),
                credit_utilization=_w(random.uniform(0.05, 0.7)),
                inquiries_6m=random.randint(0, 3), delinquency_history=0,
                employment_years=_w(random.uniform(1.0, 25.0)),
            )
        row["label"] = lab
        rows.append(row)
    random.shuffle(rows)
    return rows


def gen_s3(n: int = 400) -> list[dict]:
    """S3 汇率预测：随机游走 + 轻微动量。每行 = 截至 t-1 的特征 + t 期真值 next_close。"""
    rng = np.random.default_rng(SEED)
    price = 7.15
    prices, volumes = [price], []
    for _ in range(n + 25):
        # 日噪声 1.2%: 对齐汇率真实日波动量级(5%为股票级, 不合理)
        ret = 0.0004 + 0.012 * rng.standard_normal()  # 轻微漂移
        if len(prices) >= 5:
            momentum = (prices[-1] - prices[-5]) / prices[-5]
            # 动量系数 0.20: 信噪比 0.4(信号可提取); 稳定性约束 4c<1(AR单位根)
            ret += 0.20 * momentum
        price = price * (1 + ret)
        prices.append(price)
        volumes.append(abs(rng.standard_normal()) + 1.0)
    rows = []
    for t in range(25, 25 + n):
        window = prices[t - 20:t]
        mean5 = float(np.mean(prices[t - 5:t]))
        mean20 = float(np.mean(window))
        vol20 = float(np.std(window))
        vols = volumes[t - 20:t]
        rows.append(dict(
            date=f"2025-{(t // 30) % 12 + 1:02d}-{t % 30 + 1:02d}",
            last_close=_w(prices[t - 1]), mean5=_w(mean5), mean20=_w(mean20),
            vol20=_w(vol20), volume_z=_w(float(np.mean(vols))),
            weekday=t % 7, next_close=_w(prices[t]),
        ))
    return rows


def gen_s4(n: int = 800, pos_rate: float = 0.15) -> list[dict]:
    """S4 单证审核：违规 = 币种不一致/金额异常/原产地缺失/超额度；部分脏值。"""
    rows = []
    n_pos = int(n * pos_rate)
    labels = ["reject"] * n_pos + ["pass"] * (n - n_pos)
    for i, lab in enumerate(labels):
        cur = random.choice(CURRENCIES)
        if lab == "reject":
            kind = random.choice(["ccy_mismatch", "amount_bad", "origin_missing", "over_limit"])
            if kind == "amount_bad":
                amount = _w(random.choice([-500, 950000]))
            elif kind == "over_limit":
                # 超额度违规: 金额必须显著超过许可证额度(1万), 否则标签与特征矛盾
                amount = _w(random.uniform(11000, 60000))
            else:
                amount = _w(random.uniform(100, 20000))
            row = dict(
                doc_id=f"D{i:05d}", invoice_currency=cur,
                declared_currency=random.choice([c for c in CURRENCIES if c != cur]) if kind == "ccy_mismatch" else cur,
                amount=amount,
                origin_country="" if kind == "origin_missing" else random.choice(NORMAL_COUNTRIES),
                license_limit=_w(10000) if kind == "over_limit" else _w(500000),
            )
            row["amount_usd"] = _w(float(row["amount"]) * FX_TO_USD[row["invoice_currency"]]) if isinstance(row["amount"], (int, float)) and row["amount"] >= 0 else "N/A"
        else:
            amt = _w(random.uniform(100, 40000))
            row = dict(
                doc_id=f"D{i:05d}", invoice_currency=cur, declared_currency=cur,
                amount=amt, origin_country=random.choice(NORMAL_COUNTRIES),
                license_limit=_w(random.uniform(60000, 900000)),
            )
            row["amount_usd"] = _w(amt * FX_TO_USD[cur])
            # 负类中混入 8% 脏值（字符串金额），考察防御性: 误判为金额异常=误报
            if random.random() < 0.08:
                row["amount"] = f"USD {amt}"
        row["label"] = lab
        rows.append(row)
    random.shuffle(rows)
    return rows


SANCTION_LIST = ["IRISL ISLAMIC REPUBLIC OF IRAN SHIPPING", "KWEILIN IMPORT EXPORT",
                 "MARC RICH + CO AG", "DALIAN SUNMOON", "GLOBAL TECH SARL"]


def _variant(name: str, rng: random.Random) -> str:
    """生成名单名称的变体：大小写/标点/空格/别名。"""
    v = name
    c = rng.randint(0, 3)
    if c == 0:
        v = v.lower()
    elif c == 1:
        v = v.replace(" + ", "+").replace("  ", " ")
    elif c == 2:
        v = v.replace("CO LTD", "COMPANY LIMITED").replace("CO.", "COMPANY")
    if rng.random() < 0.3:
        v = f"  {v}  "
    if rng.random() < 0.2:
        v = v.replace(" ", "-", 1)
    return v


def gen_s5(n: int = 300, pos_rate: float = 0.4) -> list[dict]:
    """S5 名单筛查：正类为名单名称变体（模糊匹配才能抓），基线只能抓子串精确形式。"""
    rng = random.Random(SEED + 5)
    clean_pool = ["ALIBABA GROUP HOLDING", "TENCENT TECHNOLOGY", "SAMSUNG ELECTRONICS",
                  "SIEMENS AG", "TOYOTA MOTOR CORP", "HUAWEI TECH", "MICROSOFT CORP",
                  "BYD AUTOMOTIVE", "LENOVO GROUP", "XIAOMI CORP"]
    rows = []
    n_pos = int(n * pos_rate)
    labels = ["hit"] * n_pos + ["clear"] * (n - n_pos)
    for i, lab in enumerate(labels):
        if lab == "hit":
            name = _variant(rng.choice(SANCTION_LIST), rng)
        else:
            name = _variant(rng.choice(clean_pool), rng)
        rows.append(dict(txn_id=f"K{i:05d}", counterparty_name=name, label=lab))
    random.shuffle(rows)
    return rows


def _write(path: Path, rows: list[dict]):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"生成 {path.name}: {len(rows)} 行, 正类率="
          f"{sum(1 for r in rows if r.get('label') not in (None, 'normal', 'repay', 'pass', 'clear')) / len(rows):.1%}")


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    _write(DATA_DIR / "s1_suspicious_txn.csv", gen_s1())
    _write(DATA_DIR / "s2_credit_default.csv", gen_s2())
    _write(DATA_DIR / "s3_fx_forecast.csv", gen_s3())
    _write(DATA_DIR / "s4_doc_audit.csv", gen_s4())
    _write(DATA_DIR / "s5_sanction_screening.csv", gen_s5())


if __name__ == "__main__":
    main()
