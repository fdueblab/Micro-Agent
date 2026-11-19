"""Durable clinical artifacts, isolated attempts and bounded resume snapshots."""
import asyncio
import hashlib
import json
import re
import shutil
import uuid
from pathlib import Path


def read_json(path):
    if not path.is_file() or path.stat().st_size > 2 * 1024 * 1024:
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return None


def atomic_json(path, value):
    temporary = path.with_suffix(".pending")
    temporary.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def run_path(workspace, token):
    if not re.fullmatch(r"[a-f0-9]{32}", token or ""):
        raise ValueError("无效的恢复凭据")
    return Path(workspace) / "data" / "clinical-runs" / token


def prepare_run(workspace, request, source=None, resume_token=None):
    fingerprint = hashlib.sha256(request.encode("utf-8") + (source.read_bytes() if source else b"")).hexdigest()
    folder = run_path(workspace, uuid.uuid4().hex)
    folder.mkdir(parents=True)
    reused = []
    if resume_token:
        old = run_path(workspace, resume_token)
        manifest = read_json(old / "manifest.json") or {}
        if manifest.get("fingerprint") == fingerprint and manifest.get("status") not in {"running", "starting"}:
            evidence = read_json(old / "evidence.json")
            if isinstance(evidence, dict) and isinstance(evidence.get("rules"), list):
                for name in ("evidence.json", "reference_cases.json"):
                    value = read_json(old / name)
                    if value is not None:
                        atomic_json(folder / name, value)
                        reused.append(name)
    if source:
        shutil.copyfile(source, folder / ("source" + source.suffix.lower()))
    atomic_json(folder / "manifest.json", {"fingerprint": fingerprint, "status": "starting", "reused": reused})
    return folder


async def retain_outcome(ctx, folder):
    """Independent of the browser's SSE connection; never remove source artifacts."""
    manifest = read_json(folder / "manifest.json") or {}
    manifest.update(status="running", task_id=ctx.task_id)
    atomic_json(folder / "manifest.json", manifest)
    while ctx.status == "running":
        await asyncio.sleep(1)
    manifest["status"] = ctx.status
    result = read_json(folder / "result.json")
    if ctx.status == "completed" and isinstance(result, dict):
        atomic_json(folder / "delivery.json", result)
    elif ctx.status == "completed":
        manifest["status"] = "failed"
    errors = [e.data.get("error") for e in ctx.events if e.type == "error"]
    if errors:
        manifest["error"] = str(errors[-1])[:1000]
    atomic_json(folder / "manifest.json", manifest)
