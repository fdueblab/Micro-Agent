"""评测编排器：静态检查 + 运行时检查 + IP 相似度，产出完整报告。"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from pathlib import Path

from micro_agent.evaluation.runtime_checks import (
    check_agent_tests,
    check_holdout,
    check_import,
    check_perturbation,
)
from micro_agent.evaluation.schema import EvaluationReport
from micro_agent.evaluation.similarity import run_similarity_checks
from micro_agent.evaluation.static_checks import run_static_checks


@dataclass
class EvaluationInput:
    """一次评测的全部输入。"""

    code_path: Path                       # 被测算法 .py 文件
    test_path: Path | None = None         # Agent 编写的测试文件
    algorithm_category: str = ""          # classification / detection / ...
    category_params: dict = field(default_factory=dict)
    constraints: list[str] = field(default_factory=list)  # no_gpu / single_file / ...
    dataset_path: Path | None = None      # 用户数据集（holdout 评测）
    reference_texts: list[tuple[str, str]] = field(default_factory=list)  # (名称, 文本)
    eval_dir: Path | None = None          # worker 临时目录，默认 temp/ 下自动建


async def run_evaluation(inp: EvaluationInput) -> dict:
    """执行全部评测，返回报告 dict（EvaluationReport.to_dict）。"""
    code_path = Path(inp.code_path)
    if not code_path.exists():
        raise FileNotFoundError(f"被测算法文件不存在: {code_path}")

    eval_dir = inp.eval_dir or code_path.parent / f"eval_{uuid.uuid4().hex[:8]}"
    eval_dir = Path(eval_dir)

    source = code_path.read_text(encoding="utf-8", errors="replace")

    report = EvaluationReport()

    # 1) 静态检查（确定性，不执行代码）
    report.checks.extend(
        run_static_checks(
            source,
            code_path.parent,
            algorithm_category=inp.algorithm_category,
            category_params=inp.category_params,
            constraints=inp.constraints,
        )
    )

    # 2) 运行时检查（隔离子进程真实执行）
    report.checks.append(await check_import(code_path))
    report.checks.append(await check_agent_tests(inp.test_path))
    report.checks.append(await check_perturbation(code_path))
    holdout_check, holdout_payload = await check_holdout(
        code_path,
        inp.dataset_path,
        algorithm_category=inp.algorithm_category,
        labels=[str(l) for l in (inp.category_params.get("labels") or [])],
        eval_dir=eval_dir,
    )
    report.checks.append(holdout_check)
    report.holdout = holdout_payload

    # 3) IP 相似度
    report.checks.extend(
        run_similarity_checks(source, inp.reference_texts or [])
    )

    return report.to_dict()
