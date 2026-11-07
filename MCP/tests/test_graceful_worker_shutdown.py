from __future__ import annotations

import asyncio
from pathlib import Path

from server import mcp_handler as mcp_handler_module
from server.mcp_handler import MCPHandler


def test_shutdown_observes_configured_workers_as_one_group(monkeypatch) -> None:
    """A larger worker pool cannot multiply the maintenance drain budget."""

    async def scenario() -> None:
        originals = {
            "_runtime_lock": MCPHandler._runtime_lock,
            "_worker_shutdown": MCPHandler._worker_shutdown,
            "_worker_tasks": MCPHandler._worker_tasks,
            "_retention_task": MCPHandler._retention_task,
            "_shared_fastpath_client": MCPHandler._shared_fastpath_client,
        }
        observed: list[tuple[int, float]] = []

        async def idle_worker() -> None:
            await asyncio.Event().wait()

        tasks = [asyncio.create_task(idle_worker()) for _ in range(12)]

        async def observe_group(received, *, timeout):
            received_tasks = tuple(received)
            observed.append((len(received_tasks), timeout))
            await asyncio.gather(*received_tasks, return_exceptions=True)
            return set(received_tasks), set()

        monkeypatch.setattr(mcp_handler_module.asyncio, "wait", observe_group)
        try:
            MCPHandler._runtime_lock = asyncio.Lock()
            MCPHandler._worker_shutdown = asyncio.Event()
            MCPHandler._worker_tasks = tasks
            MCPHandler._retention_task = None
            MCPHandler._shared_fastpath_client = None

            await MCPHandler.stop_background_workers()

            assert observed == [(12, mcp_handler_module.DURABLE_WORKER_SHUTDOWN_WAIT_S)]
            assert MCPHandler._worker_tasks == []
            assert all(task.cancelled() for task in tasks)
        finally:
            await asyncio.gather(*tasks, return_exceptions=True)
            for name, value in originals.items():
                setattr(MCPHandler, name, value)

    asyncio.run(scenario())


def test_lifespan_delegates_worker_shutdown_to_the_grouped_cleanup() -> None:
    source = (Path(__file__).resolve().parents[1] / "server" / "http_server.py").read_text(
        encoding="utf-8"
    )
    start = source.index("cleanup_task.cancel()")
    end = source.index('print("SimpleMem MCP Server stopped")', start)
    shutdown = source[start:end]

    assert "await MCPHandler.stop_background_workers()" in shutdown
    assert "await asyncio.wait_for(asyncio.shield(task), timeout=5.0)" not in shutdown
