"""Internal deterministic MCP checks used by the authenticated backend."""

import asyncio
import hmac
import os

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel

from micro_agent.tool.mcp.connection import MCPConnectionManager, ServerConfig


router = APIRouter(prefix="/api/internal/mcp", tags=["internal-mcp"])


class CheckRequest(BaseModel):
    server_url: str
    transport: str = "sse"
    tool_name: str | None = None
    arguments: dict | None = None


def _authorize(token: str | None):
    secret = os.environ.get("MCP_INTERNAL_TOKEN", "")
    if not secret:
        raise HTTPException(503, "MCP 内部接口未配置")
    if not token or not hmac.compare_digest(token, secret):
        raise HTTPException(403, "无权调用 MCP 内部接口")


@router.post("/check")
async def check_mcp(req: CheckRequest, x_mcp_internal_token: str | None = Header(default=None)):
    _authorize(x_mcp_internal_token)
    if req.transport not in {"sse", "streamable_http"}:
        raise HTTPException(400, "不支持的 MCP 传输类型")
    try:
        async with MCPConnectionManager() as connection:
            server_id, tools = await asyncio.wait_for(connection.connect(
                ServerConfig(connection_type=req.transport, server_url=req.server_url)
            ), timeout=25)
            result = [{"name": tool.name, "description": tool.description,
                       "inputSchema": tool.parameters} for tool in tools]
            if req.tool_name is None:
                return {"tools": result}
            if req.tool_name not in {tool.name for tool in tools}:
                raise HTTPException(400, "工具不在当前服务清单中")
            session = connection.get_session(server_id)
            called = await asyncio.wait_for(session.call_tool(req.tool_name, req.arguments or {}), timeout=30)
            return {"tools": result, "call": called.model_dump(mode="json")}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(502, f"MCP 服务连接或调用失败: {exc}") from exc
