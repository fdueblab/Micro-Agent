"""IP 差异化检查：生成代码与参考资料的相似度量化。

对代码形态的参考资料（开源代码、.py/.ipynb/.zip 等）做行级相似度比对，
验证 differentiation_summary 中的差异化声明是否属实；
论文等自然语言资料不参与代码相似度计算（语义参考属正常）。
"""

from __future__ import annotations

import difflib

from micro_agent.evaluation.schema import (
    CATEGORY_IP,
    CheckResult,
    STATUS_FAILED,
    STATUS_PASSED,
    STATUS_SKIPPED,
    STATUS_WARNING,
)

# 相似度阈值
_SIM_FAILED = 0.85   # 高度雷同 → 不合格
_SIM_WARNING = 0.65  # 疑似大段借鉴 → 警告

_CODE_MARKERS = ("def ", "import ", "class ", "function ", "return ")


def _looks_like_code(name: str, text: str) -> bool:
    if name.lower().endswith((".py", ".ipynb", ".zip")):
        return True
    return any(marker in text for marker in _CODE_MARKERS)


def _normalize_lines(text: str) -> list[str]:
    """去掉注释行、行内注释与空行，返回归一化代码行。"""
    out: list[str] = []
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#") or s.startswith("//"):
            continue
        if "#" in s:
            idx = s.find("#")
            s = s[:idx].strip()
        if s:
            out.append(s)
    return out


def _similarity(a_lines: list[str], b_lines: list[str]) -> float:
    if not a_lines or not b_lines:
        return 0.0
    matcher = difflib.SequenceMatcher(None, a_lines, b_lines, autojunk=False)
    return matcher.ratio()


def run_similarity_checks(
    code_source: str,
    reference_texts: list[tuple[str, str]],
) -> list[CheckResult]:
    """对每个代码形态参考资料计算相似度，输出一条聚合检查结果。"""
    if not reference_texts:
        return [
            CheckResult(
                id="ip_code_similarity", name="参考资料代码相似度（IP 差异化）",
                category=CATEGORY_IP, status=STATUS_SKIPPED,
                details="未提供参考资料",
            )
        ]

    code_lines = _normalize_lines(code_source)
    per_ref: list[str] = []
    worst = 0.0
    code_ref_count = 0

    for name, text in reference_texts:
        if not text or not text.strip():
            continue
        if not _looks_like_code(name, text):
            per_ref.append(f"{name}: 自然语言资料（论文/网页），不做代码相似度比对")
            continue
        code_ref_count += 1
        ref_lines = _normalize_lines(text)
        sim = _similarity(code_lines, ref_lines)
        worst = max(worst, sim)
        per_ref.append(f"{name}: 相似度 {sim:.2f}")

    if code_ref_count == 0:
        return [
            CheckResult(
                id="ip_code_similarity", name="参考资料代码相似度（IP 差异化）",
                category=CATEGORY_IP, status=STATUS_SKIPPED,
                details="参考资料均不含代码内容，仅作方法参考",
                evidence=per_ref,
            )
        ]

    if worst >= _SIM_FAILED:
        status = STATUS_FAILED
        summary = f"与参考资料最高相似度 {worst:.2f}，存在大段雷同，涉嫌照搬"
    elif worst >= _SIM_WARNING:
        status = STATUS_WARNING
        summary = f"与参考资料最高相似度 {worst:.2f}，存在疑似大段借鉴，请人工复核"
    else:
        status = STATUS_PASSED
        summary = f"与参考资料最高相似度 {worst:.2f}，差异化程度可接受"

    return [
        CheckResult(
            id="ip_code_similarity", name="参考资料代码相似度（IP 差异化）",
            category=CATEGORY_IP, status=status,
            details=summary, evidence=per_ref,
        )
    ]
