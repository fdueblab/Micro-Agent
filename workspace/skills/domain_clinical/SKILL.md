# 临床医疗算法生成规范

适用于 `domain=clinical` 的算法想定式开发。先确定临床任务、目标人群、输入指标、单位、预测终点、时间窗、适用与排除条件。资料没有给出的关键参数一律记为缺口。

## 生成方式

- 严格复现：逐项列出原始论文/指南的公式、参数、版本、适用条件与出处，保留原方法，不自动修改公式。无法确认的参数不得猜测。
- 新方案探索：明确哪些内容来自资料，哪些是新假设。新算法与临床有效性分别验证。
- 专利资料仅用于技术检索与来源记录，不等于临床证据。

## 可运行交付

- `main_process` 使用确定的命名参数，仅接受 JSON 基本类型，输出可序列化为 JSON。
- 同步给出 `algorithm_spec.inputs`，字段名称和顺序必须与 `main_process` 参数一致。每项含中文标签、类型、单位、是否必填、可选项和解释。
- 同步给出不含个人信息的 `smoke_input`，并运行它验证入口。测试未实际执行时不得报告“通过”。
- 若需要预训练权重、特殊依赖或模型资产，说明来源与版本；缺少资产时保持草稿，不宣称在线可用。
- 标注成人/儿童等人群边界、单位换算、缺失值、异常范围与不能适用的情形。

## 临床知识来源

优先核对原始指南或论文及其版本。示例可参考 NIDDK 成人 eGFR 公式（https://www.niddk.nih.gov/research-funding/research-programs/kidney-clinical-research-epidemiology/laboratory/glomerular-filtration-rate-equations/adults）、WHO ICD（https://www.who.int/standards/classifications/classification-of-diseases）、LOINC（https://loinc.org/start）。这些资料仅用于对应任务；不得把某一来源自动推广到其他疾病或人群。
