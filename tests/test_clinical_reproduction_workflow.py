from tasks.clinical_reproduction import build_clinical_reproduction_prompt


def test_full_paper_is_preserved_and_outputs_are_scoped(tmp_path):
    paper = "a" * 40000 + "FINAL TABLE"
    output = tmp_path / "run" / "result.json"
    prompt = build_clinical_reproduction_prompt(model_name="score", free_narrative="strict",
                                               paper_content=paper, result_path=str(output))
    assert (output.parent / "paper.txt").read_text(encoding="utf-8") == paper
    assert str(output) in prompt
    assert "confidence=0" in prompt
    assert "agent_derived" in prompt
    assert "needs_review" in prompt
    assert "金融" not in prompt
