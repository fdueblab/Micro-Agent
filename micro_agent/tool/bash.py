"""Bash 工具：在持久 shell 会话中执行命令。

从旧版 Micro-Agent/app/tool/bash.py 移植，简化了 Pydantic 依赖。
Windows 支持三档：
  1. msys64 / Git Bash / Cygwin 真实 bash（支持 heredoc、&& 等完整语义）
  2. 兜底 cmd.exe（哨兵换用无特殊字符标记）
  3. 服务端 PATH 注入：python3 重定向到运行服务的同一 Python 解释器
"""

from __future__ import annotations

import asyncio
import os
import sys
from typing import Any, Optional

from loguru import logger

from micro_agent.tool.base import Tool, ToolResult

_IS_WIN = sys.platform == "win32"

_DESCRIPTION = """在终端中执行 bash 命令。
* 长时间命令应在后台运行并重定向输出，如: command = `python3 app.py > server.log 2>&1 &`
* 命令超时后会话会自动重启，需重新执行命令。
"""

# bash 哨兵含 < > 重定向符，cmd.exe 会误解析，改用纯单词
_SENTINEL_BASH = "<<exit>>"
_SENTINEL_CMD = "__CMD_DONE__"


def _find_bash_exe() -> Optional[str]:
    """在常见安装位置查找 Windows 下的真实 bash。"""
    for p in (
        r"C:\msys64\usr\bin\bash.exe",
        r"C:\Program Files\Git\bin\bash.exe",
        r"C:\Program Files (x86)\Git\bin\bash.exe",
        r"C:\cygwin64\bin\bash.exe",
    ):
        if os.path.isfile(p):
            return p
    return None


def _win_to_msys(path: str) -> str:
    """C:\\foo\\bar → /c/foo/bar（供 msys/git bash 的 PATH 使用）。"""
    p = path.replace("\\", "/")
    if len(p) > 1 and p[1] == ":":
        return "/" + p[0].lower() + p[2:]
    return p


class _BashSession:
    """持久化的 shell 会话。通过 sentinel 标记检测命令完成。"""

    def __init__(self, timeout: float = 120.0):
        self._process: Optional[asyncio.subprocess.Process] = None
        self._timeout = timeout
        self._timed_out = False
        self._shell: str = "bash"          # bash | cmd
        self._pending_setup: Optional[str] = None

    @property
    def _sentinel(self) -> str:
        return _SENTINEL_BASH if self._shell == "bash" else _SENTINEL_CMD

    async def start(self) -> None:
        if self._process and self._process.returncode is None:
            return
        self._pending_setup = None
        if not _IS_WIN:
            self._shell = "bash"
            self._process = await asyncio.create_subprocess_shell(
                "/bin/bash",
                preexec_fn=os.setsid,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            self._timed_out = False
            return

        bash_exe = _find_bash_exe()
        if bash_exe:
            # 首选真实 bash：heredoc / cat / && 等完整语义可用
            self._shell = "bash"
            self._process = await asyncio.create_subprocess_exec(
                bash_exe, "--noprofile", "--norc",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            # 注入运行服务的同一 Python：PATH 前置 + python3 函数重定向
            py_dir = os.path.dirname(sys.executable)
            py_scripts = os.path.join(py_dir, "Scripts")
            self._pending_setup = (
                'export PATH="%s:%s:$PATH"\n'
                'python3() { command python "$@"; }'
                % (_win_to_msys(py_dir), _win_to_msys(py_scripts))
            )
        else:
            # 兜底 cmd.exe：无 preexec_fn，哨兵不含特殊字符
            self._shell = "cmd"
            self._process = await asyncio.create_subprocess_shell(
                "cmd.exe /q",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        self._timed_out = False

    async def run(self, command: str) -> ToolResult:
        if not self._process or self._process.returncode is not None:
            return ToolResult(error="bash 会话未启动或已退出，请重试")

        if self._timed_out:
            await self._restart()
            return ToolResult(
                error="上一次命令超时，会话已自动重启，请重新执行命令"
            )

        assert self._process.stdin and self._process.stdout and self._process.stderr

        # 首条命令前执行会话初始化（PATH 注入 / python3 重定向）
        if self._pending_setup:
            command = self._pending_setup + "\n" + command
            self._pending_setup = None

        # 发送前清空 buffer，防止上一条命令残留
        self._process.stdout._buffer.clear()  # type: ignore[attr-defined]
        self._process.stderr._buffer.clear()  # type: ignore[attr-defined]

        sentinel = self._sentinel
        if self._shell == "bash":
            echo_sentinel = f"echo '{sentinel}'"
        else:
            echo_sentinel = f"echo {sentinel}"

        self._process.stdin.write(
            f"{command}\n{echo_sentinel}\n".encode()
        )
        await self._process.stdin.drain()

        try:
            async with asyncio.timeout(self._timeout):
                while True:
                    await asyncio.sleep(0.2)
                    output = self._process.stdout._buffer.decode()  # type: ignore[attr-defined]
                    if sentinel in output:
                        output = output[: output.index(sentinel)]
                        break
        except asyncio.TimeoutError:
            self._timed_out = True
            return ToolResult(error=f"命令超时（{self._timeout}s），会话将自动重启")

        error = self._process.stderr._buffer.decode()  # type: ignore[attr-defined]

        self._process.stdout._buffer.clear()  # type: ignore[attr-defined]
        self._process.stderr._buffer.clear()  # type: ignore[attr-defined]

        output = output.rstrip("\n")
        error = error.rstrip("\n")

        if error and not output:
            return ToolResult(error=error)
        if error:
            return ToolResult(output=f"{output}\n[stderr]: {error}")
        return ToolResult(output=output)

    async def _restart(self) -> None:
        if self._process and self._process.returncode is None:
            self._process.terminate()
        self._process = None
        await self.start()

    def stop(self) -> None:
        if self._process and self._process.returncode is None:
            self._process.terminate()


class Bash(Tool):
    name = "bash"
    description = _DESCRIPTION
    parameters = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "要执行的 bash 命令。",
            },
        },
        "required": ["command"],
    }

    def __init__(self, timeout: float = 120.0):
        self._session = _BashSession(timeout=timeout)

    async def execute(self, **kwargs: Any) -> ToolResult:
        command = kwargs.get("command")
        if not command:
            return ToolResult(error="未提供命令")
        await self._session.start()
        return await self._session.run(command)
