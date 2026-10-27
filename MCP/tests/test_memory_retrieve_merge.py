import asyncio
from types import SimpleNamespace

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
