"""金融垂域模型重生成 + 基准评测驱动脚本。

走完整生成链路(aml_auto_generate API: Skill 匹配 + RAG 知识库 + Prompt 禁令),
想定与金融垂域评测基准 scenarios.json 对齐, 生成后可用 finance_benchmark 评测。

用法:
    python dev/regen_finance.py --scenario S1     # 触发生成(SSE 进度)
    python dev/regen_finance.py --scenario S2
    python dev/regen_finance.py --inspect         # 查看 aml_generate_result.json 结构
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
API = "http://127.0.0.1:8010/api/agent/aml_auto_generate"
BENCH = ROOT / "workspace" / "benchmarks" / "finance"
RESULT_JSON = ROOT / "workspace" / "temp" / "aml_generate_result.json"

SCENARIOS = {
    "S1": {
        "model_name": "金融垂域-跨境可疑交易监测-重生成",
        "dataset": BENCH / "datasets" / "s1_suspicious_txn.csv",
        "labels": ["normal", "suspicious"],
        "narrative": (
            "跨境支付场景下的可疑交易监测（反洗钱方向）。输入为单笔交易特征"
            "（CSV 数据集已上传，1200 行，含 label 列：normal=正常 / suspicious=可疑，"
            "正类占比约 4%，标签列仅用于自测验证，线上输入不含该列）。字段：\n"
            "- txn_id: 交易流水号\n"
            "- amount: 原币种交易金额（六种币种 USD/CNY/EUR/JPY/GBP/HKD，金额不可直接"
            "跨币种比较，需按汇率归一化到统一口径如 USD；参考汇率 USD=1.0, CNY=0.14, "
            "EUR=1.08, JPY=0.0067, GBP=1.27, HKD=0.128）\n"
            "- currency: 交易币种\n"
            "- counterparty_country: 对手方国家两位代码（IR/KP/SY/MM/AF 为高风险地区）\n"
            "- txn_count_24h: 该账户 24 小时内交易笔数\n"
            "- distinct_currency_7d: 7 天内涉及的币种数\n"
            "- avg_amount_30d: 30 天平均交易金额\n"
            "- night_flag: 是否夜间交易（0/1）\n\n"
            "业务背景：可疑交易有两类模式——(1) 大额交易指向高风险地区或多币种分散；"
            "(2) 小额高频拆分（smurfing：单笔 800-4800 但 24h 笔数 14-40），第二类难以"
            "用简单金额阈值规则发现，请在数据中验证这两类模式并量化。\n\n"
            "输出契约（必须严格遵守）：main_process(data: dict) -> dict，返回必须包含：\n"
            "- classification_label: 字符串，\"normal\" 或 \"suspicious\"\n"
            "- confidence: 0-1 浮点数，对所输出标签的置信度（标量，非列表）\n"
            "- reason: 字符串，判定依据（可审计）"
        ),
    },
    "S2": {
        "model_name": "金融垂域-信贷违约预测-重生成",
        "dataset": BENCH / "datasets" / "s2_credit_default.csv",
        "labels": ["repay", "default"],
        "narrative": (
            "个人信贷违约预测。输入为借款人特征（CSV 数据集已上传，1500 行，含 label 列："
            "repay=正常还款 / default=违约，正类占比约 8%，标签列仅用于自测验证，"
            "线上输入不含该列）。字段：\n"
            "- customer_id: 客户编号\n"
            "- loan_amount: 贷款金额\n"
            "- annual_income: 年收入\n"
            "- debt_ratio: 负债收入比（0-1）\n"
            "- credit_utilization: 信用卡额度使用率（0-1）\n"
            "- inquiries_6m: 近 6 个月征信查询次数\n"
            "- delinquency_history: 历史逾期次数\n"
            "- employment_years: 工作年限\n\n"
            "业务背景：违约客户有两类模式——(1) 高负债比（>0.55）+ 高信用卡使用率（>0.8）；"
            "(2) 低收入 + 近期频繁征信查询（>=5 次）+ 有逾期历史，第二类难以用单变量规则"
            "发现。正类占比仅 8%，属不平衡学习场景，请勿用 accuracy 作为唯一目标，"
            "优先关注少数类召回。\n\n"
            "输出契约（必须严格遵守）：main_process(data: dict) -> dict，返回必须包含：\n"
            "- classification_label: 字符串，\"repay\" 或 \"default\"\n"
            "- confidence: 0-1 浮点数，对所输出标签的置信度（标量，非列表）\n"
            "- reason: 字符串，判定依据（可审计）"
        ),
    },
}


def generate(sid: str) -> bool:
    spec = SCENARIOS[sid]
    if not spec["dataset"].exists():
        print(f"数据集不存在: {spec['dataset']}")
        return False
    with open(spec["dataset"], "rb") as f:
        files = {"dataset_file": (spec["dataset"].name, f, "text/csv")}
        data = {
            "model_name": spec["model_name"],
            "free_narrative": spec["narrative"],
            "industry": "金融",
            "scenario": spec["model_name"],
            "technology": "Python",
            "algorithm_category": "classification",
            "category_params": json.dumps(
                {"labels": spec["labels"], "inputTypes": ["csv"]},
                ensure_ascii=False,
            ),
        }
        print(f"==> 提交生成任务 {sid}: {spec['model_name']}")
        try:
            resp = requests.post(API, data=data, files=files, stream=True, timeout=1500)
        except requests.RequestException as e:
            print(f"请求失败: {e}")
            return False
        if resp.status_code != 200:
            print(f"HTTP {resp.status_code}: {resp.text[:500]}")
            return False
        done = False
        for line in resp.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            try:
                evt = json.loads(line[6:])
            except json.JSONDecodeError:
                continue
            t = evt.get("type", "?")
            if t == "step":
                content = str(evt.get("content", evt))[:120].replace("\n", " ")
                print(f"  [step] {content}")
            elif t in ("done", "error"):
                print(f"  [{t}] {str(evt)[:400]}")
                done = t == "done"
                break
        print("==> SSE 结束" + ("（完成）" if done else "（未收到 done 事件）"))
        return done


def inspect() -> None:
    if not RESULT_JSON.exists():
        print(f"结果文件不存在: {RESULT_JSON}")
        return
    obj = json.loads(RESULT_JSON.read_text(encoding="utf-8"))
    print(json.dumps(obj, ensure_ascii=False, indent=2)[:4000])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=list(SCENARIOS))
    parser.add_argument("--inspect", action="store_true")
    args = parser.parse_args()
    if args.inspect:
        inspect()
        return
    if not args.scenario:
        parser.print_help()
        sys.exit(1)
    ok = generate(args.scenario)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
