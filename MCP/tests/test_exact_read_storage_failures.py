"""Exact retrieval and deletion must not turn storage failures into success."""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from server.database.vector_store import MultiTenantVectorStore
from server.mcp_handler import MCPHandler


class BrokenTable:
    def count_rows(self):
        return 1
    def search(self):
        return self
    def where(self, *args):
        return self
    def limit(self, *args):
        return self
    def to_pandas(self):
        raise OSError("synthetic storage read failure")


def broken_store():
    store = object.__new__(MultiTenantVectorStore)
    store._get_table = lambda _: BrokenTable()
    store._delete_records = lambda *_: None
    return store


def test_exact_read_surfaces_storage_failure():
    with pytest.raises(OSError, match="synthetic storage read failure"):
        asyncio.run(broken_store().get_entry("mem_global", "entry-1"))


def test_delete_readback_failure_never_reports_success():
    store = broken_store()
    original = store.get_entry
    calls = 0
    async def controlled(table_name, entry_id):
        nonlocal calls
        calls += 1
        if calls == 1:
            return object()
        return await original(table_name, entry_id)
    store.get_entry = controlled
    result = asyncio.run(store.delete_entry("mem_global", "entry-1"))
    assert result["deleted"] is False
    assert "synthetic storage read failure" in result["error"]


def test_initial_delete_read_failure_does_not_mutate():
    store = broken_store()
    calls = []
    store._delete_records = lambda *args: calls.append(args)
    result = asyncio.run(store.delete_entry("mem_global", "entry-1"))
    assert result["deleted"] is False and result.get("error")
    assert calls == []


def test_exact_read_failure_does_not_fall_back_to_backend():
    handler = object.__new__(MCPHandler)
    handler.user = SimpleNamespace(user_id="global", table_name="mem_global")
    handler.vector_store = broken_store()
    handler._memory_retrieve_fastpath = AsyncMock()
    with pytest.raises(OSError, match="synthetic storage read failure"):
        asyncio.run(handler._memory_retrieve_exact(query="entry-1", top_k=1))
    handler._memory_retrieve_fastpath.assert_not_awaited()


def test_delete_read_failure_does_not_route_to_backend():
    handler = object.__new__(MCPHandler)
    handler.user = SimpleNamespace(user_id="global", table_name="mem_global")
    handler.vector_store = broken_store()
    handler._uses_memory_backend_id = lambda _: False
    handler._memory_backend_base_url = lambda: "http://127.0.0.1:8001"
    handler._memory_mutation_backend_call = AsyncMock()
    with pytest.raises(OSError, match="synthetic storage read failure"):
        asyncio.run(handler._tool_memory_delete({"entry_id": "entry-1"}))
    handler._memory_mutation_backend_call.assert_not_awaited()


def test_jsonrpc_exact_read_failure_does_not_cache_miss_and_recovers():
    import json
    import pandas as pd

    class RecoveringTable(BrokenTable):
        fail = True
        def to_pandas(self):
            if self.fail:
                return super().to_pandas()
            return pd.DataFrame()

    async def exercise():
        MCPHandler._invalidate_memory_caches()
        table = RecoveringTable()
        handler = object.__new__(MCPHandler)
        handler.initialized = True
        handler.user = SimpleNamespace(user_id="global", table_name="mem_global")
        handler.settings = SimpleNamespace(memory_retrieve_max_query_chars=4096, memory_retrieve_max_top_k=10)
        handler.vector_store = broken_store()
        handler.vector_store._get_table = lambda _: table
        handler._memory_retrieve_fastpath = AsyncMock(return_value={"results": []})
        request = {"jsonrpc": "2.0", "id": 91, "method": "tools/call",
                   "params": {"name": "memory_retrieve", "arguments": {"query": "entry-1", "force_exact": True}}}
        failed = json.loads(await handler.handle_message(json.dumps(request)))
        assert failed["id"] == 91 and failed["error"]["code"] == -32603
        assert not MCPHandler._retrieve_cache and not MCPHandler._retrieve_negative_cache
        assert not MCPHandler._retrieve_inflight
        handler._memory_retrieve_fastpath.assert_not_awaited()
        table.fail = False
        request["id"] = 92
        recovered = json.loads(await handler.handle_message(json.dumps(request)))
        assert recovered["id"] == 92 and "error" not in recovered
        assert json.loads(recovered["result"]["content"][0]["text"])["total"] == 0
        handler._memory_retrieve_fastpath.assert_awaited_once()
        MCPHandler._invalidate_memory_caches()
    asyncio.run(exercise())


def test_successful_delete_round_trips_with_real_disposable_lance(tmp_path):
    store = MultiTenantVectorStore(str(tmp_path / "lance"), embedding_dimension=2)
    entry = SimpleNamespace(entry_id="fixture-entry", lossless_restatement="fixture content",
                            keywords=[], timestamp=None, location=None, persons=[], entities=[], topic=None)
    async def exercise():
        assert await store.add_entries("mem_fixture", [entry], [[1.0, 0.0]]) == 1
        assert (await store.get_entry("mem_fixture", "fixture-entry")).entry_id == "fixture-entry"
        result = await store.delete_entry("mem_fixture", "fixture-entry")
        assert result["found"] is True and result["deleted"] is True
        assert await store.get_entry("mem_fixture", "fixture-entry") is None
    asyncio.run(exercise())
