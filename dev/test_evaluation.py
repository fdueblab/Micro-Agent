"""评测系统测试脚本：dev/test_evaluation.py

用三类样例验证 micro_agent.evaluation 模块：
1. 作弊代码（random 决策 + 占位注释）→ 应被判不合格（反作弊 failed）
2. 良好规则代码 + 数据集 → 应通过 holdout 评测且优于多数类基线
3. 糊弄代码（忽略输入、恒返回多数类）→ 应在 holdout 上与基线持平/劣于基线
另验证 CLI 端到端调用与 IP 相似度检查。

运行：python dev/test_evaluation.py（在 Micro-Agent 根目录）
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from micro_agent.evaluation.evaluator import EvaluationInput, run_evaluation  # noqa: E402

PASS, FAIL = 0, 0


def check(name: str, cond: bool, extra: str = ""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name} {extra}")


GOOD_CODE = '''"""跨境支付风险分类模型。"""
from typing import Any, Dict

LABELS = ["正常", "可疑", "高风险"]


def _risk_score(amount: float, is_night: bool) -> float:
    score = amount / 10000.0
    if is_night:
        score += 0.5
    return score


def classify_transaction(amount: float, is_night: bool) -> str:
    """根据交易金额与时间判定风险等级。"""
    score = _risk_score(amount, is_night)
    if score >= 1.5:
        return "高风险"
    if score >= 0.8:
        return "可疑"
    return "正常"


def main_process(row: Dict[str, Any]) -> Dict[str, Any]:
    """主入口：输入一行交易数据，输出风险分类结果。

    Args:
        row: 含 amount（金额）、is_night（是否夜间）的字典。

    Returns:
        含 classification_label 的结果字典。
    """
    try:
        amount = float(row.get("amount", 0) or 0)
        is_night = str(row.get("is_night", "")).strip().lower() in ("1", "true", "是", "yes")
    except (TypeError, ValueError):
        amount, is_night = 0.0, False
    label = classify_transaction(amount, is_night)
    return {"classification_label": label, "confidence": 0.9}
'''

CHEAT_CODE = '''"""作弊模型：random 决策 + 占位注释。"""
import random
from typing import Any, Dict


def main_process(row: Dict[str, Any]) -> Dict[str, Any]:
    """主入口。"""
    # 在实际应用中应该接入真实风控引擎
    label = random.choice(["正常", "可疑", "高风险"])
    return {"classification_label": label}
'''

LAZY_CODE = '''"""糊弄模型：无视输入，恒返回多数类。"""
from typing import Any, Dict


def main_process(row: Dict[str, Any]) -> Dict[str, Any]:
    """主入口：恒定输出多数类。"""
    return {"classification_label": "正常"}
'''

HIDDEN_BROKEN_CODE = '''"""隐藏断链：模块导入成功，但函数体内属性访问会崩溃。"""
import json as mp
from typing import Any, Dict


def main_process(row: Dict[str, Any]) -> Dict[str, Any]:
    """主入口。"""
    pose = mp.solutions.pose  # json 没有 solutions 属性
    return {"classification_label": "正常", "detail": str(pose)}
'''

DECL_BROKEN_CODE = '''"""问题模型：虚假依赖 + 未保护 subprocess + 吞异常。
依赖清单:
flask>=2.3.0
numpy>=1.24.0
"""
import subprocess
from typing import Any, Dict

import numpy as np


def main_process(row: Dict[str, Any]) -> Dict[str, Any]:
    """主入口。"""
    subprocess.run(["yt-dlp", "-x", str(row.get("url", ""))], check=True)
    try:
        score = float(np.mean([1.0, 2.0, 3.0]))
    except Exception:
        return {"classification_label": "正常"}
    if score > 1.5:
        return {"classification_label": "可疑"}
    return {"classification_label": "正常"}
'''

TEST_CODE = '''import sys
sys.path.insert(0, '.')
from @MOD@_algorithm import main_process


def test_normal():
    r = main_process({"amount": 500, "is_night": "0"})
    assert r["classification_label"] == "正常", r


def test_high_risk():
    r = main_process({"amount": 90000, "is_night": "1"})
    assert r["classification_label"] == "高风险", r


def test_empty():
    r = main_process({})
    assert isinstance(r, dict), r


test_normal()
test_high_risk()
test_empty()
print("=== 所有测试通过 ===")
'''

SIMILAR_REF = '''"""开源参考实现（用于 IP 相似度测试）：与 GOOD_CODE 高度雷同的版本。"""
from typing import Any, Dict

LABELS = ["正常", "可疑", "高风险"]


def _risk_score(amount: float, is_night: bool) -> float:
    score = amount / 10000.0
    if is_night:
        score += 0.5
    return score


def classify_transaction(amount: float, is_night: bool) -> str:
    """根据交易金额与时间判定风险等级。"""
    score = _risk_score(amount, is_night)
    if score >= 1.5:
        return "高风险"
    if score >= 0.8:
        return "可疑"
    return "正常"


def main_process(row: Dict[str, Any]) -> Dict[str, Any]:
    """主入口：输入一行交易数据，输出风险分类结果。"""
    amount = float(row.get("amount", 0) or 0)
    is_night = str(row.get("is_night", "")).strip().lower() in ("1", "true", "是", "yes")
    label = classify_transaction(amount, is_night)
    return {"classification_label": label, "confidence": 0.9}
'''

PAPER_REF = "本文提出一种基于图神经网络的跨境支付异常检测方法，通过交易图谱建模……（论文正文）"


def make_dataset(path: Path, n: int = 120):
    """构造数据集：金额/夜间与标签强相关，多数类为 正常。"""
    rows = ["amount,is_night,label"]
    for i in range(n):
        if i % 5 == 0:
            rows.append(f"{10000 + i * 900},1,高风险")
        elif i % 5 == 1:
            rows.append(f"{8000 + i * 30},0,可疑")
        else:
            rows.append(f"{100 + i * 5},0,正常")
    path.write_text("\n".join(rows), encoding="utf-8")


async def scenario(name: str, code: str, expect: dict):
    print(f"\n=== 场景：{name} ===")
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        code_path = td / "demo_algorithm.py"
        code_path.write_text(code, encoding="utf-8")
        test_path = td / "demo_test.py"
        test_path.write_text(TEST_CODE.replace("@MOD@", "demo"), encoding="utf-8")
        ds_path = td / "dataset.csv"
        make_dataset(ds_path)

        inp = EvaluationInput(
            code_path=code_path,
            test_path=test_path,
            algorithm_category="classification",
            category_params={"labels": ["正常", "可疑", "高风险"], "constraints": ["no_gpu", "single_file"]},
            constraints=["no_gpu", "single_file"],
            dataset_path=ds_path,
            reference_texts=[("reference.py", SIMILAR_REF), ("paper.txt", PAPER_REF)],
            eval_dir=td / "worker",
        )
        report = await run_evaluation(inp)

        checks = {c["id"]: c for c in report["checks"]}
        print(f"  结论: {report['verdict']} | summary={report['summary']}")
        for cid, exp_status in expect.items():
            got = checks.get(cid, {}).get("status", "<缺失>")
            check(f"{cid} == {exp_status}", got == exp_status,
                  f"实际 {got}: {checks.get(cid, {}).get('details', '')[:120]}")
        return report


async def main():
    # 场景 1：作弊代码
    r1 = await scenario(
        "作弊代码（random 决策 + 占位注释）",
        CHEAT_CODE,
        {
            "anti_random_core": "failed",
            "anti_placeholder": "failed",
            "anti_stub": "passed",
            "ip_code_similarity": "passed",  # 作弊代码与参考实现并不雷同
        },
    )
    check("场景1 结论为不合格", r1["verdict"] == "unqualified", f"实际 {r1['verdict']}")

    # 场景 2：良好代码
    r2 = await scenario(
        "良好规则代码（数据集上应优于基线）",
        GOOD_CODE,
        {
            "anti_random_core": "passed",
            "anti_placeholder": "passed",
            "label_contract": "passed",
            "holdout_evaluation": "passed",
            "agent_test_run": "passed",
            "ip_code_similarity": "warning",  # 与 SIMILAR_REF 高度雷同 → 警告
        },
    )
    h = r2.get("holdout") or {}
    check("场景2 holdout 完成", h.get("status") == "completed")
    check("场景2 优于基线", h.get("verdict") == "beats_baseline", f"实际 {h.get('verdict')}")
    check("场景2 结论非不合格", r2["verdict"] != "unqualified", f"实际 {r2['verdict']}")

    # 场景 3：糊弄代码（无视输入恒返回多数类 → 与基线持平，按设计应为 warning）
    r3 = await scenario(
        "糊弄代码（无视输入恒返回多数类）",
        LAZY_CODE,
        {"holdout_evaluation": "warning"},
    )
    h3 = r3.get("holdout") or {}
    check("场景3 持平基线", h3.get("verdict") == "matches_baseline",
          f"实际 {h3.get('verdict')}")

    # 场景 3b：隐藏断链（导入成功但函数体内属性访问崩溃，属性探测应抓住）
    r3b = await scenario(
        "隐藏断链（函数体内访问不存在属性）",
        HIDDEN_BROKEN_CODE,
        {"import_check": "failed"},
    )
    imp = [c for c in r3b["checks"] if c["id"] == "import_check"][0]
    check("场景3b 探测证据包含 json.solutions",
          any("json.solutions" in e for e in imp.get("evidence", [])),
          f"evidence={imp.get('evidence')}")

    # 场景 3c：虚假依赖声明 + 未保护 subprocess + 吞异常（P0 新增三项检查）
    r3c = await scenario(
        "虚假依赖 + 未保护外部命令 + 吞异常",
        DECL_BROKEN_CODE,
        {
            "dependency_declaration": "failed",
            "external_command": "failed",
            "swallowed_exception": "warning",
        },
    )
    dd = [c for c in r3c["checks"] if c["id"] == "dependency_declaration"][0]
    check("场景3c 虚假依赖证据包含 flask",
          any("flask" in e for e in dd.get("evidence", [])),
          f"evidence={dd.get('evidence')}")
    ec = [c for c in r3c["checks"] if c["id"] == "external_command"][0]
    check("场景3c 外部命令证据含未保护提示",
          any("未用 try/except 保护" in e for e in ec.get("evidence", [])),
          f"evidence={ec.get('evidence')}")

    # 场景 4：CLI 端到端
    print("\n=== 场景：CLI 端到端 ===")
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        code_path = td / "demo_algorithm.py"
        code_path.write_text(GOOD_CODE, encoding="utf-8")
        test_path = td / "demo_test.py"
        test_path.write_text(TEST_CODE.replace("@MOD@", "demo"), encoding="utf-8")
        ds_path = td / "dataset.csv"
        make_dataset(ds_path)
        inputs_path = td / "inputs.json"
        inputs_path.write_text(json.dumps({
            "algorithm_category": "classification",
            "category_params": {"labels": ["正常", "可疑", "高风险"]},
            "constraints": ["no_gpu"],
            "dataset_path": str(ds_path),
        }, ensure_ascii=False), encoding="utf-8")
        out_path = td / "report.json"

        proc = subprocess.run(
            [sys.executable, str(ROOT / "micro_agent" / "evaluation" / "cli.py"),
             "--code", str(code_path), "--test", str(test_path),
             "--inputs", str(inputs_path), "--out", str(out_path)],
            capture_output=True, text=True, timeout=300, cwd=str(td),
        )
        check("CLI 退出码 0/1（非崩溃）", proc.returncode in (0, 1),
              f"rc={proc.returncode}, stderr={proc.stderr[:200]}")
        check("CLI 生成报告文件", out_path.exists())
        if out_path.exists():
            report = json.loads(out_path.read_text(encoding="utf-8"))
            check("CLI 报告含 checks", isinstance(report.get("checks"), list) and report["checks"])
            check("CLI 报告含 holdout", report.get("holdout") is not None)
            check("CLI stdout 含结论", "评测摘要" in proc.stdout, proc.stdout[:200])

    print(f"\n{'=' * 40}\n测试结果: {PASS} 通过, {FAIL} 失败")
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
