import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from tasks.clinical_pipeline import ClinicalReproductionAgent, paper_context, add_input_guards
from api.services.clinical_runs import read_json, atomic_json


EVIDENCE = {"model_name": "fixture", "method_type": "score", "population": "adults",
            "outcome": "score", "horizon": "now", "gaps": [],
            "rules": [{"id": "age", "input": "age", "rule": "age >= 60: 1", "source": {
                "page": 1, "location": "Table 1", "quote": "age 60"}}]}
CODE = {"model_name": "fixture", "generated_code": "def main_process(age):\n    return {'total_score': int(age >= 60)}\n",
        "algorithm_spec": {"inputs": [{"name": "age", "type": "integer"}]}, "smoke_input": {"age": 60}}


class FakeLLM:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.calls = []

    async def complete(self, **kwargs):
        self.calls.append(kwargs)
        reply = next(self.replies)
        if isinstance(reply, Exception):
            raise reply
        return SimpleNamespace(content=json.dumps(reply) if isinstance(reply, dict) else reply)


class PipelineTests(unittest.TestCase):
    def test_contract_guards_preserve_score_and_reject_bool_and_bounds(self):
        value = json.loads(json.dumps(CODE))
        value['algorithm_spec']['inputs'][0]['minimum'] = 16
        add_input_guards(value)
        namespace = {}
        # This is the fixed, trusted test fixture above, never LLM source.
        exec(value['generated_code'], namespace)
        self.assertEqual(namespace['main_process'](60), {'total_score': 1})
        for invalid in (True, None, [], 15, 60.0):
            with self.assertRaises(ValueError):
                namespace['main_process'](invalid)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        (self.folder / "paper.txt").write_text("Table 1 age 60", encoding="utf-8")
        atomic_json(self.folder / "request.json", {"request": "score"})

    def run_pipeline(self, llm, **kwargs):
        async def collect():
            return [event async for event in ClinicalReproductionAgent(llm, self.folder, **kwargs).run("unused")]
        return asyncio.run(collect())

    def test_success_has_two_calls_no_shell_and_no_fake_runtime_pass(self):
        llm = FakeLLM([EVIDENCE, CODE])
        llm.model = "openrouter/test"
        events = self.run_pipeline(llm)
        self.assertEqual(len(llm.calls), 2)
        self.assertTrue(all(c["tools"] is None and c["num_retries"] == 0 for c in llm.calls))
        self.assertTrue(all(c["extra_body"]["reasoning"]["max_tokens"] == 1024 for c in llm.calls))
        self.assertEqual(events[-1].type, "done")
        result = read_json(self.folder / "result.json")
        self.assertEqual(result["reproduction"]["status"], "pending_runtime")
        self.assertEqual(result["reproduction"]["verification"]["basis"], "static_contract")

    def test_missing_values_stop_before_codegen(self):
        llm = FakeLLM([dict(EVIDENCE, gaps=["评分表分值不可辨认"])])
        self.run_pipeline(llm)
        self.assertEqual(len(llm.calls), 1)
        self.assertFalse((self.folder / "algorithm.py").exists())
        self.assertEqual(read_json(self.folder / "result.json")["reproduction"]["status"], "needs_review")

    def test_limitations_are_retained_without_blocking_and_provenance_is_saved(self):
        evidence = dict(EVIDENCE, limitations=["预先核对计数"], input_policies=["缺项拒绝计算"])
        self.run_pipeline(FakeLLM([evidence, CODE]))
        result = read_json(self.folder / "result.json")
        self.assertEqual(result["reproduction"]["status"], "pending_runtime")
        self.assertIn("缺项拒绝计算", result["model_summary"]["limitations"])
        self.assertTrue(read_json(self.folder / "evidence_provenance.json")["request_sha256"])

    def test_real_gap_remains_blocking_even_with_limitations(self):
        evidence = dict(EVIDENCE, gaps=["缺少分值"], limitations=["预先核对计数"])
        llm = FakeLLM([evidence])
        self.run_pipeline(llm)
        self.assertEqual(len(llm.calls), 1)
        self.assertFalse((self.folder / "algorithm.py").exists())

    def test_pdf_padding_does_not_reject_readable_paper(self):
        pages = self.folder / "pages"
        pages.mkdir()
        (pages / "page-10.txt").write_text("LAST", encoding="utf-8")
        (pages / "page-2.txt").write_text("FIRST" + " " * 200000 + "END", encoding="utf-8")
        llm = FakeLLM([EVIDENCE, CODE])
        self.run_pipeline(llm)
        context = llm.calls[0]["messages"][1]["content"]
        self.assertIn("FIRST END", context)
        self.assertLess(context.index("FIRST"), context.index("LAST"))
        self.assertEqual(len(llm.calls), 2)

    def test_explicit_target_keeps_neighbors_and_discloses_selection(self):
        pages = self.folder / "pages"
        pages.mkdir()
        for number in range(1, 12):
            text = "Table 2\nScoring rules" if number == 7 else f"page {number}"
            (pages / f"page-{number}.txt").write_text(text, encoding="utf-8")
        context = paper_context(self.folder, {"request": "Implement Table 2"})
        self.assertIn("不是全文", context)
        self.assertIn("[page-6.txt]", context)
        self.assertNotIn("[page-11.txt]", context)
        selection = read_json(self.folder / "context_selection.json")
        self.assertEqual(selection["included_pages"], [1, 2, 6, 7, 8])

    def test_invalid_json_stops_after_two_calls(self):
        llm = FakeLLM(["not json", "still not json"])
        self.run_pipeline(llm, attempts=100)
        self.assertEqual(len(llm.calls), 2)
        self.assertTrue(read_json(self.folder / "result.json")["reproduction"]["gaps"])

    def test_hanging_call_times_out(self):
        class Hanging:
            async def complete(self, **kwargs):
                await asyncio.sleep(60)
        self.run_pipeline(Hanging(), call_timeout=0.01)
        self.assertIn("超时", read_json(self.folder / "result.json")["reproduction"]["gaps"][0])

    def test_restored_evidence_skips_extraction(self):
        atomic_json(self.folder / "evidence.json", EVIDENCE)
        llm = FakeLLM([CODE])
        self.run_pipeline(llm)
        self.assertEqual(len(llm.calls), 1)

    def test_bad_code_has_bounded_repairs(self):
        llm = FakeLLM([EVIDENCE, dict(CODE, generated_code="broken("), dict(CODE, generated_code="broken(")])
        self.run_pipeline(llm)
        self.assertEqual(len(llm.calls), 3)
        self.assertFalse((self.folder / "algorithm.py").exists())

    def test_provider_exception_is_failure_not_success(self):
        events = self.run_pipeline(FakeLLM([ConnectionError("offline")]))
        self.assertEqual(events[-1].type, "error")
        self.assertFalse((self.folder / "result.json").exists())


if __name__ == "__main__":
    unittest.main()
