from tasks.aml_auto_generate import build_aml_auto_generate_prompt


def test_clinical_reproduction_keeps_evaluation_and_online_contract():
    prompt = build_aml_auto_generate_prompt(
        model_name="肾功能公式",
        free_narrative="复现论文公式",
        workspace="/workspace",
        reference_materials="原始论文",
        project_root="/app",
        eval_inputs_path="/workspace/eval.json",
        domain="clinical",
        clinical_task="肾功能评估",
        generation_mode="reproduce",
        result_path="/workspace/result-123.json",
    )

    assert "临床任务：肾功能评估。生成模式：reproduce" in prompt
    assert "主动差异化创新" not in prompt
    assert "严格复现、无算法改动" in prompt
    assert "外部确定性评测" in prompt
    assert "/workspace/result-123.json" in prompt
    assert '"algorithm_spec"' in prompt
    assert '"smoke_input"' in prompt


def test_general_generation_retains_reference_guidance():
    prompt = build_aml_auto_generate_prompt(
        model_name="交易分类",
        free_narrative="识别异常交易",
        workspace="/workspace",
        reference_materials="公开论文",
    )

    assert "主动差异化创新" in prompt
    assert "临床任务与生成模式" not in prompt
