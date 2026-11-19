import asyncio
import pytest

from micro_agent.core.task import TaskManager, TaskContext
from micro_agent.core.schema import AgentEvent


class StubAgent:
    def __init__(self, events):
        self.events = events

    async def run(self, request):
        for event in self.events:
            yield event


@pytest.mark.parametrize("events,expected", [
    ([AgentEvent(type="error", step=3, data={"error": "connection lost"})], "failed"),
    ([AgentEvent(type="done", step=40, data={"reason": "max_steps"})], "failed"),
    ([AgentEvent(type="done", step=2, data={"reason": "cancelled"})], "cancelled"),
    ([AgentEvent(type="done", step=2, data={"result": "ok"})], "completed"),
    ([], "failed"),
    ([AgentEvent(type="error", step=1, data={"error": "bad"}),
      AgentEvent(type="done", step=2, data={"result": "ok"})], "failed"),
])
def test_terminal_status(events, expected):
    ctx = TaskContext(task_id="test", _agent=StubAgent(events))
    asyncio.run(TaskManager()._execute(ctx, "test"))
    assert ctx.status == expected
