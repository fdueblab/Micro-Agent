from api.services.clinical_runs import prepare_run, atomic_json, read_json, run_path
import pytest


def test_resume_reuses_only_same_request_evidence(tmp_path):
    first = prepare_run(tmp_path, "request")
    manifest = read_json(first / "manifest.json")
    manifest["status"] = "failed"
    atomic_json(first / "manifest.json", manifest)
    atomic_json(first / "evidence.json", {"rules": []})
    atomic_json(first / "result.json", {"generated_code": "old"})
    same = prepare_run(tmp_path, "request", resume_token=first.name)
    assert (same / "evidence.json").exists()
    assert not (same / "result.json").exists()
    changed = prepare_run(tmp_path, "different", resume_token=first.name)
    assert not (changed / "evidence.json").exists()


def test_resume_rejects_paths(tmp_path):
    with pytest.raises(ValueError):
        run_path(tmp_path, "../outside")
