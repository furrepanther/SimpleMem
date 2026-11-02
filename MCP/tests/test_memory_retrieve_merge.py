import asyncio
from types import SimpleNamespace

import pytest

from server.auth.models import MemoryEntry
from server.mcp_handler import MCPHandler


def test_local_retrieval_precedes_and_deduplicates_fastpath_results() -> None:
    local = MemoryEntry(
        entry_id="local-1",
        lossless_restatement="fresh local durable record",
        location="urn:local:fresh",
    )
    merged = MCPHandler._merge_retrieval_results(
        query="fresh local durable record",
        local_entries=[local],
        fastpath={
            "results": [
                {
                    "entry_id": "local-1",
                    "content": "stale backend duplicate",
                    "location": "urn:backend:stale",
                },
                {
                    "entry_id": "backend-1",
                    "content": "shared backend record",
                    "location": "urn:backend:shared",
                },
            ]
        },
        top_k=10,
    )

    assert [row["entry_id"] for row in merged["results"]] == [
        "local-1",
        "backend-1",
    ]
    assert merged["results"][0]["content"] == "fresh local durable record"


def test_exact_retrieval_filters_backend_semantic_substitutes() -> None:
    target_id = "target-memory-id"

    class Store:
        async def get_entry(self, table_name: str, entry_id: str):
            assert table_name == "mem_global"
            assert entry_id == target_id
            return None

    handler = object.__new__(MCPHandler)
    handler.user = SimpleNamespace(user_id="global", table_name="mem_global")
    handler.vector_store = Store()

    async def fastpath(*, query: str, top_k: int, force_exact: bool):
        assert (query, top_k, force_exact) == (target_id, 10, True)
        return {
            "results": [
                {"entry_id": "semantic-substitute", "content": "wrong"},
                {"memory_id": target_id, "content": "exact"},
            ]
        }

    handler._memory_retrieve_fastpath = fastpath
    result = asyncio.run(
        handler._memory_retrieve_exact(query=target_id, top_k=10)
    )

    assert result["total"] == 1
    assert result["results"] == [{"memory_id": target_id, "content": "exact"}]


def test_exact_retrieval_returns_empty_on_backend_miss() -> None:
    target_id = "missing-memory-id"

    class Store:
        async def get_entry(self, table_name: str, entry_id: str):
            return None

    handler = object.__new__(MCPHandler)
    handler.user = SimpleNamespace(user_id="global", table_name="mem_global")
    handler.vector_store = Store()
    handler._memory_retrieve_fastpath = lambda **_: asyncio.sleep(
        0, result={"results": [{"entry_id": "other", "content": "wrong"}]}
    )

    result = asyncio.run(
        handler._memory_retrieve_exact(query=target_id, top_k=10)
    )

    assert result == {"query": target_id, "results": [], "total": 0}


def _retrieval_handler() -> MCPHandler:
    MCPHandler._invalidate_memory_caches()
    handler = object.__new__(MCPHandler)
    handler.user = SimpleNamespace(user_id="global", table_name="mem_global")
    handler.settings = SimpleNamespace(
        memory_retrieve_max_query_chars=4096,
        memory_retrieve_max_top_k=10,
        memory_queue_max_inflight_reads=2,
    )
    return handler


def test_mcp_schema_exposes_the_supported_exact_lookup_option() -> None:
    handler = object.__new__(MCPHandler)
    listing = asyncio.run(handler._handle_tools_list({}))
    tool = next(item for item in listing["tools"] if item["name"] == "memory_retrieve")
    option = tool["inputSchema"]["properties"]["force_exact"]

    assert option["type"] == "boolean"
    assert option["default"] is False


def test_public_exact_lookup_returns_local_entry_without_semantic_search() -> None:
    target_id = "6426df95-a1ab-5d95-adf9-394f12dcc084"
    entry = MemoryEntry(entry_id=target_id, lossless_restatement="exact canary")

    class Store:
        async def get_entry(self, table_name: str, entry_id: str):
            assert (table_name, entry_id) == ("mem_global", target_id)
            return entry

    handler = _retrieval_handler()
    handler.vector_store = Store()

    def reject_semantic_search():
        raise AssertionError("exact lookup entered the semantic retrieval path")

    handler._get_retriever = reject_semantic_search
    result = asyncio.run(
        handler._tool_memory_retrieve({"query": target_id, "force_exact": True})
    )

    assert result["total"] == 1
    assert result["results"][0]["entry_id"] == target_id
    assert result["results"][0]["content"] == "exact canary"


def test_default_lookup_preserves_semantic_merge_order() -> None:
    entry = MemoryEntry(entry_id="local-semantic", lossless_restatement="local fact")

    class Retriever:
        async def retrieve(self, **kwargs):
            assert kwargs["query"] == "semantic query"
            return [entry]

    handler = _retrieval_handler()
    handler._get_retriever = lambda: Retriever()

    async def fastpath(**kwargs):
        assert kwargs["force_exact"] is False
        return {"results": [{"entry_id": "backend-semantic", "content": "backend fact"}]}

    handler._memory_retrieve_fastpath = fastpath
    result = asyncio.run(handler._tool_memory_retrieve({"query": "semantic query"}))

    assert [row["entry_id"] for row in result["results"]] == [
        "local-semantic",
        "backend-semantic",
    ]


def test_retrieve_owner_disconnect_cancels_coalesced_waiter_without_cache() -> None:
    async def exercise() -> None:
        MCPHandler._invalidate_memory_caches()
        MCPHandler._retrieve_inflight.clear()
        MCPHandler._retrieve_inflight_ts.clear()
        handler = _retrieval_handler()
        started = asyncio.Event()

        async def exact(*, query: str, top_k: int) -> dict:
            started.set()
            await asyncio.Event().wait()
            return {"query": query, "results": [], "total": 0}

        handler._memory_retrieve_exact = exact
        owner = asyncio.create_task(
            handler._tool_memory_retrieve(
                {"query": "disconnect fixture", "force_exact": True}
            )
        )
        waiter = None
        try:
            await started.wait()
            waiter = asyncio.create_task(
                handler._tool_memory_retrieve(
                    {"query": "disconnect fixture", "force_exact": True}
                )
            )
            await asyncio.sleep(0)
            owner.cancel()
            outcomes = await asyncio.wait_for(
                asyncio.gather(owner, waiter, return_exceptions=True), timeout=0.5
            )
            assert all(isinstance(item, asyncio.CancelledError) for item in outcomes)
            assert not MCPHandler._retrieve_inflight
            assert not MCPHandler._retrieve_cache
        finally:
            for task in (owner, waiter):
                if task is not None and not task.done():
                    task.cancel()
            await asyncio.gather(
                *(task for task in (owner, waiter) if task is not None),
                return_exceptions=True,
            )
            MCPHandler._invalidate_memory_caches()
            MCPHandler._retrieve_inflight.clear()
            MCPHandler._retrieve_inflight_ts.clear()

    asyncio.run(exercise())


def test_retrieve_read_admission_failure_cleans_inflight_and_can_retry() -> None:
    async def exercise() -> None:
        handler = _retrieval_handler()
        original_active = MCPHandler._active_read_ops

        async def exact(*, query: str, top_k: int) -> dict:
            return {"query": query, "results": [], "total": 0}

        handler._memory_retrieve_exact = exact
        try:
            MCPHandler._active_read_ops = handler.settings.memory_queue_max_inflight_reads
            with pytest.raises(RuntimeError, match="memory_query_overloaded"):
                await handler._tool_memory_retrieve(
                    {"query": "admission fixture", "force_exact": True}
                )
            assert not MCPHandler._retrieve_inflight
            assert not MCPHandler._retrieve_cache

            MCPHandler._active_read_ops = 0
            result = await handler._tool_memory_retrieve(
                {"query": "admission fixture", "force_exact": True}
            )
            assert result == {
                "query": "admission fixture",
                "results": [],
                "total": 0,
            }
        finally:
            MCPHandler._active_read_ops = original_active
            MCPHandler._invalidate_memory_caches()
            MCPHandler._retrieve_inflight.clear()
            MCPHandler._retrieve_inflight_ts.clear()

    asyncio.run(exercise())


@pytest.mark.parametrize("invalid", ["false", "true", 0, 1, None, []])
def test_exact_lookup_rejects_nonboolean_flags(invalid) -> None:
    handler = _retrieval_handler()

    with pytest.raises(ValueError, match="force_exact must be a boolean"):
        asyncio.run(
            handler._tool_memory_retrieve({"query": "target", "force_exact": invalid})
        )
