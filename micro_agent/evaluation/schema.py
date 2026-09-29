"""评测报告数据结构：CheckResult / EvaluationReport。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime

# 检查状态
STATUS_PASSED = "passed"
STATUS_WARNING = "warning"
STATUS_FAILED = "failed"
STATUS_SKIPPED = "skipped"

# 检查类别（用于报告分组与结论判定）
CATEGORY_BASIC = "basic"          # 基础可用性
CATEGORY_CHEATING = "cheating"    # 反作弊（真实实现）
CATEGORY_CONSTRAINT = "constraint"  # 用户技术约束遵循
CATEGORY_CONTRACT = "contract"    # 输入输出契约
CATEGORY_RUNTIME = "runtime"      # 真实执行验证
CATEGORY_IP = "ip"                # 知识产权/差异化
CATEGORY_QUALITY = "quality"      # 平台代码规范

# 严重级别：出现该类 failed 时整体结论直接判为不合格
_FATAL_CATEGORIES = {CATEGORY_CHEATING, CATEGORY_CONSTRAINT}

# 警告条数达到该阈值时整体结论判为 needs_review
_WARNING_REVIEW_THRESHOLD = 3

VERDICT_QUALIFIED = "qualified"
VERDICT_NEEDS_REVIEW = "needs_review"
VERDICT_UNQUALIFIED = "unqualified"

_STATUS_LABELS = {
    STATUS_PASSED: "通过",
    STATUS_WARNING: "警告",
    STATUS_FAILED: "未通过",
    STATUS_SKIPPED: "跳过",
}


@dataclass
class CheckResult:
    """单项检查结果。"""

    id: str
    name: str
    category: str
    status: str
    details: str = ""
    evidence: list[str] = field(default_factory=list)

    @property
    def status_label(self) -> str:
        return _STATUS_LABELS.get(self.status, self.status)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["status_label"] = self.status_label
        return d


@dataclass
class EvaluationReport:
    """一次算法模型评测的完整报告。"""

    checks: list[CheckResult] = field(default_factory=list)
    holdout: dict | None = None
    evaluator_version: str = "1.0"
    generated_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))

    def summary(self) -> dict:
        counts = {STATUS_PASSED: 0, STATUS_WARNING: 0, STATUS_FAILED: 0, STATUS_SKIPPED: 0}
        for c in self.checks:
            if c.status in counts:
                counts[c.status] += 1
        counts["total"] = len(self.checks)
        return counts

    @property
    def verdict(self) -> str:
        fatal = [c for c in self.checks if c.status == STATUS_FAILED and c.category in _FATAL_CATEGORIES]
        if fatal:
            return VERDICT_UNQUALIFIED
        any_failed = any(c.status == STATUS_FAILED for c in self.checks)
        warning_count = sum(1 for c in self.checks if c.status == STATUS_WARNING)
        if any_failed or warning_count >= _WARNING_REVIEW_THRESHOLD:
            return VERDICT_NEEDS_REVIEW
        return VERDICT_QUALIFIED

    @property
    def verdict_label(self) -> str:
        return {
            VERDICT_QUALIFIED: "合格",
            VERDICT_NEEDS_REVIEW: "需人工复核",
            VERDICT_UNQUALIFIED: "不合格",
        }.get(self.verdict, self.verdict)

    def failed_checks(self) -> list[CheckResult]:
        return [c for c in self.checks if c.status == STATUS_FAILED]

    def warning_checks(self) -> list[CheckResult]:
        return [c for c in self.checks if c.status == STATUS_WARNING]

    def to_dict(self) -> dict:
        return {
            "evaluator_version": self.evaluator_version,
            "generated_at": self.generated_at,
            "verdict": self.verdict,
            "verdict_label": self.verdict_label,
            "summary": self.summary(),
            "checks": [c.to_dict() for c in self.checks],
            "holdout": self.holdout,
        }

    def compact_summary_text(self) -> str:
        """生成紧凑摘要文本（供 CLI stdout / Agent 合并结果用）。"""
        s = self.summary()
        lines = [
            f"评测结论: {self.verdict_label}（通过 {s[STATUS_PASSED]} / 警告 {s[STATUS_WARNING]} / "
            f"未通过 {s[STATUS_FAILED]} / 跳过 {s[STATUS_SKIPPED]}）"
        ]
        for c in self.checks:
            if c.status in (STATUS_FAILED, STATUS_WARNING):
                lines.append(f"[{c.status_label}] {c.name}（{c.id}）: {c.details}")
        if self.holdout and self.holdout.get("status") == "completed":
            m = self.holdout.get("metrics") or {}
            lines.append(f"数据集评测: {m}")
        return "\n".join(lines)
