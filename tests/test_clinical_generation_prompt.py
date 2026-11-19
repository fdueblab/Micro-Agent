import json

from tasks.aml_auto_generate import build_aml_auto_generate_prompt


def test_clinical_reproduction_uses_task_artifacts_and_online_contract(tmp_path):
    result_path = tmp_path / "run" / "result-123.json"
    prompt = build_aml_auto_generate_prompt(
        model_name="肾功能公式",
        free_narrative="复现论文公式",
        workspace=str(tmp_path),
        paper_content="论文中的公式与分值表",
        reference_materials="原始论文",
        project_root="/app",
        eval_inputs_path=str(tmp_path / "eval.json"),
        domain="clinical",
        clinical_task="肾功能评估",
        generation_mode="reproduce",
        result_path=str(result_path),
    )

    assert (result_path.parent / "paper.txt").read_text(encoding="utf-8") == "论文中的公式与分值表"
    assert json.loads((result_path.parent / "request.json").read_text(encoding="utf-8")) == {
        "model_name": "肾功能公式", "request": "复现论文公式", "references": "原始论文",
    }
    assert "主动差异化创新" not in prompt
    assert "严格复现" in prompt
    assert str(result_path) in prompt
    assert "algorithm_spec" in prompt
    assert "smoke_input" in prompt
    assert "main_process" in prompt
    assert "agent_derived" in prompt


def test_general_generation_retains_reference_guidance():
    prompt = build_aml_auto_generate_prompt(
        model_name="交易分类",
        free_narrative="识别异常交易",
        workspace="/workspace",
        reference_materials="公开论文",
    )

    assert "主动差异化创新" in prompt
    assert "临床任务与生成模式" not in prompt
