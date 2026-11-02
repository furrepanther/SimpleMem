import asyncio
import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import lancedb
import pyarrow as pa

from server.auth.models import MemoryEntry, User
from server.database.vector_store import MultiTenantVectorStore
from server.mcp_handler import MCPHandler


def _entry(entry_id: str, metadata: dict) -> MemoryEntry:
    return MemoryEntry(
        entry_id=entry_id,
        lossless_restatement=f"review evidence {entry_id}",
        keywords=["review", "evidence"],
        topic="review-transport",
        metadata=metadata,
    )


def _temporary_metadata(purge_after_utc: str, **overrides: object) -> dict:
    metadata = {
        "temporary": True,
        "canonical": False,
        "retention_verified": True,
        "record_kind": "temporary_review_evidence",
        "purge_after_utc": purge_after_utc,
        "packet_sha256": "p" * 64,
        "content_sha256": "c" * 64,
    }
    metadata.update(overrides)
    return metadata


def test_metadata_round_trips_through_lance_exact_read(tmp_path) -> None:
    store = MultiTenantVectorStore(str(tmp_path / "lancedb"), embedding_dimension=2)
    metadata = _temporary_metadata("2030-01-01T00:00:00Z", source_tool="review")

    async def exercise() -> MemoryEntry | None:
        assert await store.add_entries("mem_global", [_entry("review-1", metadata)], [[1.0, 0.0]]) == 1
        return await store.get_entry("mem_global", "review-1")

    loaded = asyncio.run(exercise())
    assert loaded is not None
    assert loaded.metadata == metadata
    row = store._get_table("mem_global").to_pandas().iloc[0]
    assert json.loads(row["metadata_json"]) == metadata
    assert row["retention_purge_after_utc"] == "2030-01-01T00:00:00Z"


def test_existing_lance_table_receives_additive_metadata_migration(tmp_path) -> None:
    db_path = str(tmp_path / "lancedb")
    legacy_schema = pa.schema(
        [
            pa.field("entry_id", pa.string()),
            pa.field("lossless_restatement", pa.string()),
            pa.field("keywords", pa.list_(pa.string())),
            pa.field("timestamp", pa.string()),
            pa.field("location", pa.string()),
            pa.field("persons", pa.list_(pa.string())),
            pa.field("entities", pa.list_(pa.string())),
            pa.field("topic", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), 2)),
            pa.field("created_at", pa.string()),
        ]
    )
    legacy = lancedb.connect(db_path)
    legacy.create_table(
        "mem_legacy",
        data=[
            {
                "entry_id": "legacy-1",
                "lossless_restatement": "legacy record",
                "keywords": ["legacy"],
                "timestamp": "",
                "location": "",
                "persons": [],
                "entities": [],
                "topic": "legacy",
                "vector": [0.0, 1.0],
                "created_at": "2026-01-01T00:00:00Z",
            }
        ],
        schema=legacy_schema,
    )

    store = MultiTenantVectorStore(db_path, embedding_dimension=2)
    loaded = asyncio.run(store.get_entry("mem_legacy", "legacy-1"))
    assert loaded is not None
    assert loaded.metadata == {}
    names = store._get_table("mem_legacy").schema.names
    assert "metadata_json" in names
    assert "retention_purge_after_utc" in names


def test_expired_review_cleanup_is_exact_bounded_and_preserves_other_records(tmp_path) -> None:
    store = MultiTenantVectorStore(str(tmp_path / "lancedb"), embedding_dimension=2)
    now = datetime(2026, 11, 2, tzinfo=timezone.utc)
    expired = _temporary_metadata("2026-11-01T00:00:00Z")
    future = _temporary_metadata("2026-11-03T00:00:00Z")
    canonical = _temporary_metadata("2026-11-01T00:00:00Z", canonical=True)

    async def exercise() -> dict:
        entries = [
            _entry("expired", expired),
            _entry("future", future),
            _entry("canonical", canonical),
        ]
        assert await store.add_entries(
            "mem_global", entries, [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]
        ) == 3
        return await store.purge_expired_temporary_review_entries(
            "mem_global",
            now_utc=now.isoformat(),
            limit=2,
        )

    receipt = asyncio.run(exercise())
    assert receipt["selected"] == 1
    assert receipt["deleted"] == [
        {"entry_id": "expired", "purge_after_utc": "2026-11-01T00:00:00Z"}
    ]
    assert asyncio.run(store.get_entry("mem_global", "expired")) is None
    assert asyncio.run(store.get_entry("mem_global", "future")) is not None
    assert asyncio.run(store.get_entry("mem_global", "canonical")) is not None


def test_memory_store_metadata_is_validated_and_included_in_idempotency() -> None:
    handler = object.__new__(MCPHandler)
    handler.user = User(user_id="metadata-user")
    handler.settings = SimpleNamespace(
        memory_write_max_content_chars=4096,
        memory_write_max_metadata_chars=4096,
    )
    first = handler._normalize_memory_store_args(
        {"content": "same", "metadata": {"retention_verified": True, "rank": 1}}
    )
    second = handler._normalize_memory_store_args(
        {"content": "same", "metadata": {"retention_verified": True, "rank": 2}}
    )
    assert first["metadata"]["rank"] == 1
    assert handler._compute_idempotency_key(kind="memory_store", args=first) != handler._compute_idempotency_key(
        kind="memory_store", args=second
    )


def test_durable_enqueue_and_worker_preserve_metadata_end_to_end() -> None:
    class Queue:
        def __init__(self) -> None:
            self.payload: dict | None = None

        def enqueue_or_get(self, **kwargs):
            self.payload = kwargs["payload"]
            return {"accepted": True, "status": "queued", "job_id": "job-metadata"}

    class Client:
        async def create_bounded_semantic_embedding(self, _content: str):
            return [1.0, 0.0]

    class ClientManager:
        def get_client(self, _api_key: str):
            return Client()

    class VectorStore:
        def __init__(self) -> None:
            self.entry: MemoryEntry | None = None

        async def add_entries(self, _table_name: str, entries, _embeddings):
            self.entry = entries[0]
            return 1

    handler = object.__new__(MCPHandler)
    handler.user = User(user_id="durable-metadata")
    handler.settings = SimpleNamespace(
        memory_write_max_content_chars=4096,
        memory_write_max_metadata_chars=4096,
        memory_write_max_payload_bytes=65536,
        memory_queue_max_inflight_writes=10,
        memory_queue_max_global_inflight_writes=10,
        memory_write_job_status_limit=100,
    )
    queue = Queue()
    old_store = MCPHandler._durable_store
    old_runtime = (
        MCPHandler._shared_settings,
        MCPHandler._shared_vector_store,
        MCPHandler._shared_client_manager,
        MCPHandler._shared_user_store,
        MCPHandler._shared_token_manager,
    )
    vector_store = VectorStore()
    metadata = _temporary_metadata("2030-01-01T00:00:00Z", source_tool="review")
    try:
        MCPHandler._durable_store = queue
        queued = asyncio.run(
            handler._enqueue_durable(
                kind="memory_store",
                args={"content": "durable evidence", "metadata": metadata},
            )
        )
        assert queued["accepted"] is True
        assert queue.payload is not None
        assert queue.payload["metadata"] == metadata

        MCPHandler._shared_settings = SimpleNamespace()
        MCPHandler._shared_vector_store = vector_store
        MCPHandler._shared_client_manager = ClientManager()
        MCPHandler._shared_user_store = SimpleNamespace(
            get_user=lambda _user_id: handler.user
        )
        MCPHandler._shared_token_manager = SimpleNamespace(
            decrypt_api_key=lambda _encrypted: "local"
        )
        result = asyncio.run(
            MCPHandler._execute_durable_job(
                {
                    "job_id": "job-metadata",
                    "kind": "memory_store",
                    "payload": queue.payload,
                }
            )
        )
    finally:
        MCPHandler._durable_store = old_store
        (
            MCPHandler._shared_settings,
            MCPHandler._shared_vector_store,
            MCPHandler._shared_client_manager,
            MCPHandler._shared_user_store,
            MCPHandler._shared_token_manager,
        ) = old_runtime

    assert result["entries_created"] == 1
    assert vector_store.entry is not None
    assert vector_store.entry.metadata == metadata


def test_local_exact_result_exposes_persisted_metadata() -> None:
    entry = _entry("exact-review", _temporary_metadata("2030-01-01T00:00:00Z"))
    result = MCPHandler._local_entry_to_retrieval_result(entry, include_metadata=True)
    assert result["entry_id"] == "exact-review"
    assert result["metadata"] == entry.metadata
    assert "metadata" not in MCPHandler._local_entry_to_retrieval_result(entry)


def test_exact_retrieval_includes_metadata_without_changing_normal_result_shape() -> None:
    entry = _entry("exact-review", _temporary_metadata("2030-01-01T00:00:00Z"))

    class VectorStore:
        async def get_entry(self, _table_name: str, entry_id: str):
            return entry if entry_id == entry.entry_id else None

    handler = object.__new__(MCPHandler)
    handler.user = User(user_id="exact-metadata-user")
    handler.vector_store = VectorStore()
    result = asyncio.run(
        handler._memory_retrieve_exact(query=entry.entry_id, top_k=1)
    )

    assert result["total"] == 1
    assert result["results"][0]["metadata"] == entry.metadata


def test_cleanup_tool_caps_the_requested_limit_and_returns_exact_receipt() -> None:
    class VectorStore:
        def __init__(self) -> None:
            self.call: tuple[str, int] | None = None

        async def purge_expired_temporary_review_entries(self, table_name: str, *, limit: int):
            self.call = (table_name, limit)
            return {
                "table_name": table_name,
                "selected": 1,
                "deleted": [{"entry_id": "expired-review"}],
                "skipped": [],
            }

    handler = object.__new__(MCPHandler)
    handler.user = User(user_id="cleanup-user")
    handler.settings = SimpleNamespace(memory_temporary_evidence_purge_batch_size=3)
    handler.vector_store = VectorStore()
    result = asyncio.run(handler._tool_memory_cleanup_expired({"limit": 1000}))

    assert handler.vector_store.call == (handler.user.table_name, 3)
    assert result["scope"] == "expired_temporary_review_evidence_only"
    assert result["deleted"] == [{"entry_id": "expired-review"}]


def test_background_retention_sweep_is_bounded_and_returns_receipts() -> None:
    class VectorStore:
        def __init__(self) -> None:
            self.calls: list[tuple[str, int]] = []

        async def purge_expired_temporary_review_entries(self, table_name: str, *, limit: int):
            self.calls.append((table_name, limit))
            return {
                "table_name": table_name,
                "selected": 1,
                "deleted": [{"entry_id": f"{table_name}-expired"}],
                "skipped": [],
            }

    vector_store = VectorStore()
    old_values = (
        MCPHandler._shared_settings,
        MCPHandler._shared_vector_store,
        MCPHandler._shared_user_store,
    )
    try:
        MCPHandler._shared_settings = SimpleNamespace(
            memory_temporary_evidence_purge_batch_size=1,
            memory_temporary_evidence_purge_max_tables=2,
        )
        MCPHandler._shared_vector_store = vector_store
        MCPHandler._shared_user_store = SimpleNamespace(
            list_users=lambda limit: [
                SimpleNamespace(table_name="mem_one"),
                SimpleNamespace(table_name="mem_two"),
            ][:limit]
        )
        receipt = asyncio.run(MCPHandler._sweep_expired_temporary_review_evidence())
    finally:
        (
            MCPHandler._shared_settings,
            MCPHandler._shared_vector_store,
            MCPHandler._shared_user_store,
        ) = old_values

    assert vector_store.calls == [("mem_one", 1)]
    assert receipt["deleted_count"] == 1
    assert receipt["scope"] == "expired_temporary_review_evidence_only"
