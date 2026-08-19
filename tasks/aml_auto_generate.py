"""算法模型想定式开发 — prompt 构建。

v6 架构版本：
- 支持算法类别 (algorithm_category) 与类别特定参数 (category_params)
- 公共参数 + 类别特定参数的分层注入
- 多阶段引导式执行（需求分析 → 技术选型 → 测试设计 → 代码生成 → 真实执行验证）
- 代码规范通过 Skill 注入（algorithm_code_standards）
- 领域知识通过 RAG 预检索 (rag_context) + Agent 内部 RAG 双通道注入
- 强制技术引导：Skill/RAG 包含技术路线时必须优先采用
- 通用负面约束：禁止随机/占位符分类逻辑、禁止虚假实现
- 测试驱动生成：先生成测试用例，再生成代码，最后用测试验证代码正确性
- 真实执行验证：py_compile + import + 功能测试，失败自动重试修复
"""

from __future__ import annotations

from typing import Optional

_CONSTRAINT_LABELS = {
    "no_llm": "禁止使用任何 LLM / 大语言模型 API",
    "no_training": "禁止进行任何模型训练或微调",
    "no_gpu": "算法必须能在纯 CPU 环境下运行",
    "pretrained_only": "只允许使用预训练模型进行推理，不可训练",
    "rule_based": "优先采用纯规则/启发式方法",
    "single_file": "所有代码必须在单个 Python 文件中实现",
}

_CATEGORY_LABELS = {
    "classification": "分类算法",
    "detection": "检测算法",
    "regression": "回归/预测算法",
    "clustering": "聚类算法",
    "generation": "生成算法",
    "recommendation": "推荐算法",
}

_CATEGORY_TASK_HINTS = {
    "classification": "将输入数据映射到预定义类别标签",
    "detection": "在输入数据中定位并识别目标对象或异常",
    "regression": "根据输入特征预测连续数值或趋势",
    "clustering": "对无标签数据进行自动分组和模式发现",
    "generation": "根据输入条件生成新的数据内容",
    "recommendation": "根据用户特征和历史行为推荐相关内容",
}


def _build_category_params_section(category: str, params: dict) -> str:
    """Build prompt section from category-specific parameters."""
    if not params:
        return ""

    lines: list[str] = []
    cat_label = _CATEGORY_LABELS.get(category, category)
    lines.append(f"\n## {cat_label} — 特定参数\n")

    input_types = params.get("inputTypes")
    if input_types:
        lines.append(f"- **输入数据类型**: {', '.join(str(s) for s in input_types)}")

    if category == "classification":
        out = params.get("outputTypes")
        if out:
            lines.append(f"- **输出数据类型**: {', '.join(str(s) for s in out)}")
        labels = params.get("labels")
        if labels:
            lines.append(f"- **分类标签**: {', '.join(str(l) for l in labels)}")
            lines.append(f"  （共 {len(labels)} 个类别，代码中必须包含这些标签的定义）")
        if params.get("multiLabel"):
            lines.append("- **多标签分类**: 是（一个样本可同时属于多个类别）")

    elif category == "detection":
        targets = params.get("targetTypes")
        if targets:
            lines.append(f"- **检测目标类型**: {', '.join(str(s) for s in targets)}")
        out_fmt = params.get("outputFormats")
        if out_fmt:
            lines.append(f"- **输出格式**: {', '.join(str(s) for s in out_fmt)}")
        if params.get("realtime"):
            lines.append("- **实时检测**: 是（需要优化推理速度）")

    elif category == "regression":
        target = params.get("predictionTarget")
        if target:
            lines.append(f"- **预测目标**: {target}")
        granularity = params.get("timeGranularity")
        if granularity:
            lines.append(f"- **时间粒度**: {granularity}")
        metrics = params.get("metrics")
        if metrics:
            lines.append(f"- **评估指标偏好**: {', '.join(str(m) for m in metrics)}")

    elif category == "clustering":
        count = params.get("clusterCount")
        if count is not None:
            if count == 0:
                lines.append("- **聚类数**: 自动确定")
            else:
                lines.append(f"- **期望聚类数**: {count}")
        methods = params.get("methods")
        if methods:
            lines.append(f"- **聚类方法偏好**: {', '.join(str(m) for m in methods)}")
        out_fmt = params.get("outputFormats")
        if out_fmt:
            lines.append(f"- **输出格式**: {', '.join(str(s) for s in out_fmt)}")

    elif category == "generation":
        targets = params.get("targetTypes")
        if targets:
            lines.append(f"- **生成目标类型**: {', '.join(str(s) for s in targets)}")
        gen_count = params.get("generateCount")
        if gen_count:
            lines.append(f"- **单次生成数量**: {gen_count}")
        quality = params.get("qualityPreference")
        if quality:
            lines.append(f"- **质量控制偏好**: {', '.join(str(q) for q in quality)}")

    elif category == "recommendation":
        target = params.get("recommendTarget")
        if target:
            lines.append(f"- **推荐目标**: {target}")
        strategies = params.get("strategies")
        if strategies:
            lines.append(f"- **推荐策略偏好**: {', '.join(str(s) for s in strategies)}")
        top_k = params.get("topK")
        if top_k:
            lines.append(f"- **Top-K 推荐数量**: {top_k}")

    if len(lines) <= 1:
        return ""

    lines.append(
        "\n生成的 `main_process` 函数签名和返回值必须严格匹配上述参数规格。"
    )
    return "\n".join(lines)


def _build_constraints_section(params: dict) -> str:
    """Extract and format constraints from category_params."""
    constraints = params.get("constraints")
    if not constraints:
        return ""

    lines = ["\n## 技术约束（必须严格遵守）\n"]
    for c in constraints:
        c_str = str(c)
        if c_str.startswith("custom: "):
            lines.append(f"- ⚠️ {c_str[8:]}")
        elif c_str in _CONSTRAINT_LABELS:
            lines.append(f"- ⚠️ {_CONSTRAINT_LABELS[c_str]}")
        else:
            lines.append(f"- ⚠️ {c_str}")
    lines.append(
        "\n**违反以上任何一条约束都会导致生成的代码不可用。**"
        "在技术选型时，必须首先排除不满足约束的方案。"
    )
    return "\n".join(lines)


def build_aml_auto_generate_prompt(
    *,
    model_name: str,
    free_narrative: str,
    workspace: str,
    industry: str = "",
    scenario: str = "",
    technology: str = "",
    paper_content: str = "",
    dataset_info: Optional[dict] = None,
    algorithm_category: str = "",
    category_params: Optional[dict] = None,
    rag_context: str = "",
    reference_materials: str = "",
) -> str:
    sections: list[str] = []
    params = category_params or {}

    # ── 基本元信息 ──
    header = f"""你是一个专业的AI算法工程师，需要根据用户的需求生成高质量的算法模型服务代码。

## 任务目标
生成算法模型：**{model_name}**"""

    if algorithm_category:
        cat_label = _CATEGORY_LABELS.get(algorithm_category, algorithm_category)
        cat_hint = _CATEGORY_TASK_HINTS.get(algorithm_category, "")
        header += f"\n\n## 算法类别\n**{cat_label}** — {cat_hint}"

    header += f"\n\n## 用户需求描述\n{free_narrative}"
    sections.append(header)

    if industry:
        sections.append(f"\n## 行业领域\n{industry}")
    if scenario:
        sections.append(f"\n## 应用场景\n{scenario}")
    if technology:
        sections.append(f"\n## 技术方向\n{technology}")
    if paper_content:
        sections.append(f"\n## 想定式描述文件内容\n{paper_content[:3000]}")

    # ── 数据集上下文 ──
    if dataset_info and dataset_info.get("raw_text"):
        ds_section = "\n## 数据集信息\n"
        ds_section += f"- 文件名: {dataset_info.get('file_name', '未知')}\n"
        ds_section += f"- 格式: {dataset_info.get('format', '未知')}\n"
        if dataset_info.get("total_rows"):
            ds_section += f"- 总行数: {dataset_info['total_rows']}\n"
        if dataset_info.get("columns"):
            ds_section += f"- 列名: {', '.join(str(c) for c in dataset_info['columns'])}\n"
        if dataset_info.get("label_distribution"):
            dist = dataset_info["label_distribution"]
            ds_section += "- 标签分布:\n"
            for lbl, cnt in dist.items():
                ds_section += f"  - {lbl}: {cnt} 条\n"
        ds_section += f"\n### 数据样例（前若干行）\n```\n{dataset_info['raw_text'][:2000]}\n```\n"
        ds_section += (
            "\n**重要**：生成的代码必须能够处理上述数据格式。"
            "如果数据中包含 URL，代码应能从这些 URL 获取数据。"
        )
        sections.append(ds_section)

    # ── 领域知识参考（RAG 预检索结果） ──
    if rag_context:
        sections.append(
            "\n## 领域知识参考（从知识库检索）\n\n"
            "以下是与本任务高度相关的领域知识，包含技术路线、库选型、代码范例等。\n"
            "**在步骤 2 技术选型时必须优先参考这些内容。**\n\n"
            f"{rag_context}"
        )

    # ── 用户提供的参考资料 + 差异化/知识产权要求 ──
    if reference_materials:
        sections.append(
            "\n## 用户提供的参考资料\n\n"
            "以下是用户提交的「相关资料」（可能包含论文、专利、开源代码、网址内容或说明），"
            "用于指导本次算法模型的优化方向：\n\n"
            f"{reference_materials[:6000]}\n"
        )
        sections.append(
            "\n## 差异化与知识产权要求（必须严格遵守）\n\n"
            "针对上述参考资料，你必须：\n"
            "1. **参考但不照搬**：可以借鉴其中的思路、方法论与技术路线，"
            "但严禁逐行复制其源码或直接实现受专利保护的具体方案；\n"
            "2. **主动差异化创新**：至少在算法结构、关键步骤、特征工程、"
            "优化策略或工程实现中的若干方面做出实质性改进与区别，避免与参考资料雷同；\n"
            "3. **规避知识产权争议**：不得引入明显侵犯版权/专利的实现；"
            "若参考资料含明确专利点，应采用替代方案绕开；\n"
            "4. **可追溯说明**：在最终结果中明确记录参考了哪些资料、做了哪些差异化处理，"
            "以备知识产权审查（填入下方 JSON 的 references 与 differentiation_summary 字段）。"
        )

    # ── 类别特定参数 ──
    if algorithm_category and params:
        cat_params_text = _build_category_params_section(algorithm_category, params)
        if cat_params_text:
            sections.append(cat_params_text)

    # ── 分类标签硬约束 ──
    labels = params.get("labels")
    if algorithm_category == "classification" and labels:
        label_str = ", ".join(str(l) for l in labels)
        sections.append(
            f"\n## 分类标签硬约束\n\n"
            f"代码中的分类标签 **必须且仅** 包含以下 {len(labels)} 个类别：\n"
            f"**{label_str}**\n\n"
            f"- 不得使用其他任何标签名称（如 walking、running 等无关类别）\n"
            f"- 不得使用通用/占位符类别列表\n"
            f"- 分类函数的返回值必须是上述标签之一"
        )

    # ── 技术约束 ──
    cons_text = _build_constraints_section(params)
    if cons_text:
        sections.append(cons_text)

    # ── 执行步骤 ──
    task_type_hint = ""
    if algorithm_category:
        task_type_hint = f"（{_CATEGORY_LABELS.get(algorithm_category, algorithm_category)}）"

    skill_guidance = ""
    if rag_context:
        skill_guidance = (
            "\n\n**技术选型强制引导**：如果上方「领域知识参考」或系统提示中的 "
            "Skill 技术指导包含了与本任务相关的技术路线、库选型建议或代码范例，"
            "你 **必须** 优先采用其中描述的方案和推荐的第三方库。"
            "不得在有明确技术指导的情况下自行发明替代方案。"
        )

    constraint_enforcement = ""
    constraints = params.get("constraints")
    if constraints:
        constraint_enforcement = (
            "\n- 必须逐条检查上方列出的技术约束，"
            "并为每条约束明确说明当前技术方案如何满足"
        )

    sections.append(f"""
---

## 执行步骤（严格按步骤执行）

### 步骤 1：需求分析与数据理解
分析用户需求、数据集特征（如果提供了数据集）、技术约束。明确：
- 算法的核心任务{task_type_hint}具体要解决什么问题
- 输入数据的格式和获取方式
- 输出结果的格式和内容
- 哪些技术路线被约束排除了

### 步骤 2：技术选型与架构设计
基于需求分析和约束条件，选择具体的技术方案：
- 需要哪些第三方库（列出库名和用途）
- 数据处理流水线的各个阶段
- 核心算法的实现思路
- 哪些函数作为独立可封装的核心 API
- `main_process` 主入口的职责边界{skill_guidance}{constraint_enforcement}

### 步骤 2b：设计测试用例（在生成代码之前）
基于输入输出规格，先设计至少 3 组测试用例并写入测试文件。测试用例是后续验证代码正确性的标准，代码必须全部通过。

**设计要求：**
- 测试 1（正常输入）：构造一个应触发默认/正常结果的输入，验证输出结构和值
- 测试 2（边界/触发输入）：构造一个应触发某个特定结果的输入，验证输出值符合预期
- 测试 3（异常/空输入）：构造空值或异常输入，验证不崩溃且有合理输出

**将测试用例写入** `{workspace}/temp/{{model_name}}_test.py`，格式示例（根据实际规格调整）。

**重要：如果模型名称包含连字符、中文等非法字符，必须先转为合法 Python 模块名（仅小写字母、数字、下划线）。** 例如 `ZWH-未知领域-001` 应转为 `zwh_unknown_domain_001`，代码文件和测试文件都使用转换后的名称。下方模板中的 `{{model_name}}` 占位符需替换为转换后的实际模块名。
```python
import sys
sys.path.insert(0, '.')
from {{model_name}}_algorithm import main_process

def test_normal_input():
    # 测试正常输入应返回默认/正常结果
    # 用基于输入规格的真实测试数据替换下方参数
    result = main_process()
    assert isinstance(result, dict), f'返回类型错误: {{type(result)}}'
    assert 'classification_label' in result, '缺少 classification_label 字段'
    assert all(l in ['正常','可疑','高风险','疑似欺诈'] for l in result['classification_label']), '标签不在允许范围'
    assert all(0 <= c <= 1 for c in result['confidence_list']), '置信度不在 0~1 范围'
    print('test_normal_input 通过:', result)

def test_trigger_input():
    # 构造应触发特定结果的输入
    # 用基于输入规格的、应触发特定结果的测试数据替换下方参数
    result = main_process()
    assert len(result['classification_label']) > 0, '分类标签为空'
    print('test_trigger_input 通过:', result)

def test_empty_input():
    # 空值/异常输入不应崩溃
    result = main_process()
    assert isinstance(result, dict), '空输入返回类型错误'
    print('test_empty_input 通过:', result)

if __name__ == '__main__':
    test_normal_input()
    test_trigger_input()
    test_empty_input()
    print('=== 所有测试通过 ===')
```

**重要：测试数据要基于算法的输入输出规格构造，不能用占位符。**

### 步骤 3：生成算法代码
生成完整 Python 单文件代码，必须严格遵守已加载的 Skill 中的所有要求。
如果系统加载了领域 Skill（如视频处理、姿态估计等），**必须** 参考其中的代码范例和技术路线。
生成的代码必须能通过步骤 2b 中设计的所有测试用例。

### 步骤 4：保存代码文件
使用 bash 工具将代码写入 `{workspace}/temp/{{model_name}}_algorithm.py`（使用转换后的模块名作为文件名）。

### 步骤 5：代码验证（真实执行）
**必须使用 bash 工具真实执行以下验证，不能跳过。如果验证失败，回到步骤 3 修复代码后重新保存并重新验证。**

#### 5.1 语法编译检查
使用 bash 工具执行：
python3 -m py_compile {workspace}/temp/{{model_name}}_algorithm.py

如果编译失败（有语法错误），回到【步骤 3】修复后重新保存。最多重试 3 次。

#### 5.2 导入测试
使用 bash 工具执行：
cd {workspace}/temp && python3 -c "import {{model_name}}_algorithm; print('导入成功')"

如果导入失败（缺少依赖或模块结构错误），回到【步骤 3】修复后重新保存。最多重试 3 次。

#### 5.3 功能测试
执行步骤 2b 中设计的测试文件，验证代码功能正确性：

cd {workspace}/temp && python3 {{model_name}}_test.py

该测试文件包含至少 3 组测试用例（正常输入、触发输入、空输入），使用 assert 断言验证输出结构和语义。

如果测试失败（assert 错误或异常），分析失败原因，回到【步骤 3】修复代码后重新保存，再次执行测试。最多重试 3 次。

#### 5.4 七维质量分析
基于 5.1-5.3 的真实执行结果，对代码逐项分析：
1. 语法编译检查：py_compile 执行结果
2. 导入测试：import 执行结果
3. 功能测试：测试用例执行结果
4. 平台提交规范：是否符合 `main_process` 入口、Google docstring 等要求
5. 接口测试：函数签名和返回值是否匹配输入输出规格
6. 可靠性测试：是否有适当的异常处理和边界情况处理
7. 兼容性测试：依赖库版本是否兼容、是否跨平台

### 步骤 6：保存最终结果
使用 bash 工具将 JSON 写入 `{workspace}/temp/aml_generate_result.json`，格式：
```json
{{{{
    "model_name": "{model_name}",
    "generated_code": "<完整代码>",
    "code_filename": "{{model_name}}_algorithm.py",
    "model_summary": {{{{
        "purpose": "用非技术用户能理解的中文说明：这个算法模型主要帮助用户完成什么任务",
        "input_description": "说明用户需要提供什么数据或材料，不要出现 bash、py_compile 等命令行细节",
        "output_description": "说明模型会输出什么结果，以及结果如何帮助业务判断",
        "usage_scenarios": ["适用业务场景1", "适用业务场景2"],
        "limitations": "说明当前方案可能不完善的地方；若用户描述不完整，必须明确指出需要补充的信息",
        "next_steps": ["建议用户后续补充的数据、规则或评价指标"]
    }}}},
    "test_results": [
        {{{{"name": "语法编译检查", "status": "passed", "details": "py_compile 执行结果..."}}}},
        {{{{"name": "导入测试", "status": "passed", "details": "导入成功..."}}}},
        {{{{"name": "功能测试", "status": "passed", "details": "所有测试用例通过..."}}}},
        {{{{"name": "平台提交规范", "status": "passed", "details": "..."}}}},
        {{{{"name": "接口测试", "status": "passed", "details": "..."}}}},
        {{{{"name": "可靠性测试", "status": "passed", "details": "..."}}}},
        {{{{"name": "兼容性测试", "status": "passed", "details": "..."}}}}
    ],
    "references": [
        {{{{
            "type": "paper",
            "title": "...",
            "summary": "...",
            "source": "用户上传 | RAG知识库 | 网址",
            "what_referenced": "从该资料参考了什么思路/方法",
            "what_added": "在其基础上新增了什么",
            "what_improved": "相比该资料提升/优化了什么",
            "advantages_vs_existing": "相比现有同类算法的特点与优势",
            "ip_considerations": "为规避知识产权争议所做的差异化处理"
        }}}}
    ],
    "differentiation_summary": {{{{
        "overall_strategy": "整体差异化与创新策略概述",
        "key_innovations": ["关键创新点1", "关键创新点2"],
        "improvements": ["相比参考资料/现有方案的提升点1", "提升点2"],
        "advantages": ["对比现有算法的特点与优势1", "优势2"],
        "ip_risk_notes": "知识产权风险规避说明"
    }}}}
}}}}
```
说明：
- 即使用户未提供参考资料，也应基于通用现有算法填写 references（来源标 RAG知识库 或常识）与 differentiation_summary，
  说明本方案参考了什么、新增/提升了什么、对比现有算法有哪些特点与优势。
- differentiation_summary 必须填写，用于向用户清晰展示「参考了…、新增了…、提升了…、对比优势…」。
- model_summary 必须填写，且必须面向不懂技术的用户，避免展示 bash、cat、python3、py_compile、main_process 等命令行或工程实现细节。
- 如果用户需求描述不完整、数据集缺失或参考资料不足，必须在 model_summary.limitations 与 model_summary.next_steps 中用友好语言说明。

### 步骤 7：完成任务
确认 JSON 文件已保存后，调用 terminate 结束任务。

---

## 注意事项
1. 平台规范优先：严格遵守 Skill 中的函数独立性与 Google docstring 要求
2. 代码完整性：不要用省略号或 pass 代替实现
3. 如果 RAG 检索到了参考资料、用户提供了「相关资料」或 Skill 提供了技术指导，在 references 字段中列出，并填写 what_referenced/what_added/what_improved 等子字段
4. 逐步执行，不要跳过任何步骤
5. 如果有技术约束，在步骤 2 中必须逐条说明如何满足
6. 若用户提供了「相关资料」，必须遵守上方「差异化与知识产权要求」，并完整填写 differentiation_summary
7. 面向用户展示的 model_summary 必须通俗、简洁、可操作，不得暴露命令行执行过程或源码写入过程
8. 步骤 5 的代码验证必须真实执行，不能跳过。如果验证失败，必须回到步骤 3 修复代码后重新保存并重新验证。test_results 必须如实记录每次验证的真实结果（passed/failed），不得伪造结果
9. 步骤 2b 的测试用例必须在生成代码之前完成，测试数据要基于输入输出规格构造真实数据，不能用占位符。步骤 5.3 必须执行步骤 2b 生成的测试文件，且所有测试必须通过

## 硬性禁止（违反任一条将导致代码不合格）
1. **禁止**使用 `random.choice()` / `random.randint()` / `random.uniform()` 作为分类、检测或预测的核心决策逻辑
2. **禁止**用注释 "在实际应用中应该..." / "模拟..." / "placeholder" / "demo" 代替真实算法实现
3. **禁止**输出未在用户指定的标签/类别定义中列出的名称
4. **禁止**引入与实际推理逻辑无关的依赖库（如声明了 torch 但仅用于生成随机张量）
5. 每个核心函数都必须包含 **真实的** 数据处理和决策逻辑

现在开始执行任务，请从【步骤 1】开始。
""")

    return "\n".join(sections)
