"""Packaging must not return a stale artifact after an Agent error."""

import json
from pathlib import Path

import pytest

from api.services.files import extract_zip
from api.services.sse import sse_response
from micro_agent.core.schema import AgentEvent
from micro_agent.core.task import TaskContext


@pytest.mark.asyncio
async def test_failed_task_does_not_return_existing_package(tmp_path: Path):
    output = tmp_path / "output"
    output.mkdir()
    for name in ("server.py", "Dockerfile", "docker-compose.yml"):
        (output / name).write_text("placeholder", encoding="utf-8")
    ctx = TaskContext(task_id="failed", status="completed")
    await ctx.add_event(AgentEvent(type="error", step=1, data={"error": "generation failed"}))
    response = await sse_response(ctx, zip_dir=str(output))
    lines = [chunk.decode() if isinstance(chunk, bytes) else chunk async for chunk in response.body_iterator]
    events = [json.loads(line.removeprefix("data: ").strip()) for line in lines if line.startswith("data:")]
    assert events[-1]["error"]
    assert not (events[-1].get("final_results") or {}).get("service_package")


def test_zip_extraction_rejects_parent_traversal(tmp_path: Path):
    import zipfile

    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.writestr("../outside.py", "print('unsafe')")
    with pytest.raises(Exception, match="不安全"):
        extract_zip(archive, tmp_path / "output")
    assert not (tmp_path / "outside.py").exists()
