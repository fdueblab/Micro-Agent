"""算法模型评测系统：确定性检查（反作弊/约束/契约/数据集/基线/IP）。

设计原则：全部检查由程序确定性执行，不依赖 LLM 自评。
- static_checks  AST 级静态扫描，不执行代码
- runtime_checks 隔离子进程真实执行（导入/测试/holdout+基线）
- similarity     参考资料代码相似度（IP 差异化量化）
- evaluator      编排入口，产出 EvaluationReport
- cli            命令行入口（供 Agent bash 调用与人工使用）
"""

from micro_agent.evaluation.evaluator import EvaluationInput, run_evaluation
from micro_agent.evaluation.schema import (
    CheckResult,
    EvaluationReport,
    STATUS_FAILED,
    STATUS_PASSED,
    STATUS_SKIPPED,
    STATUS_WARNING,
)

__all__ = [
    "EvaluationInput",
    "run_evaluation",
    "CheckResult",
    "EvaluationReport",
    "STATUS_PASSED",
    "STATUS_WARNING",
    "STATUS_FAILED",
    "STATUS_SKIPPED",
]
