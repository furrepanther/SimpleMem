"""Shared fixture for protocol tests that intentionally stub runtime startup."""
import asyncio

import pytest

from server.mcp_handler import MCPHandler


@pytest.fixture
def deliver_ready_message(monkeypatch):
    async def deliver(handler, message):
        monkeypatch.setattr(MCPHandler, "_durable_runtime_ready", True)
        monkeypatch.setattr(MCPHandler, "_service_lock_guard", object())
        monkeypatch.setattr(MCPHandler, "_runtime_event_loop", asyncio.get_running_loop())
        monkeypatch.setattr(MCPHandler, "_runtime_drained", asyncio.Event())
        monkeypatch.setattr(MCPHandler, "_active_runtime_requests", 0)
        return await handler.handle_message(message)
    return deliver
