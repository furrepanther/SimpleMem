import asyncio
import json
import sqlite3
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pandas as pd
import pytest

from config.settings import Settings
from server.auth.models import Dialogue, User
from server.core.memory_builder import MemoryBuilder
from server.database.durable_job_store import DurableJobStore
from server.database.vector_store import MultiTenantVectorStore
from server.mcp_handler import MCPHandler
from server.integrations.openrouter import OpenRouterClient


def _claim_abandoned_fixture(store, **kwargs):
    """Simulate claims whose owner has already lost its execution locks."""
    with store.claim_locked_batch(**kwargs) as jobs:
        return jobs


def test_settings_fail_closed_when_jwt_secret_is_empty(tmp_path) -> None:
    with pytest.raises(ValueError, match="JWT_SECRET_KEY"):
        Settings(
            jwt_secret_key="",
            encryption_key="x" * 32,
            data_dir=str(tmp_path / "data"),
            lancedb_path=str(tmp_path / "lancedb"),
            durable_queue_db_path=str(tmp_path / "queue.db"),
        )


@pytest.mark.parametrize(
    ("jwt_secret_key", "encryption_key", "message"),
    [
        (
            "simplemem-secret-key-change-in-production",
            "x" * 32,
            "JWT_SECRET_KEY",
        ),
        (
            "x" * 32,
            "simplemem-encryption-key-32bytes!",
            "ENCRYPTION_KEY",
        ),
    ],
)
def test_settings_rejects_builtin_secrets(
    tmp_path, jwt_secret_key: str, encryption_key: str, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        Settings(
            jwt_secret_key=jwt_secret_key,
            encryption_key=encryption_key,
            data_dir=str(tmp_path / "data"),
            lancedb_path=str(tmp_path / "lancedb"),
            durable_queue_db_path=str(tmp_path / "queue.db"),
        )


def test_cors_wildcard_never_enables_credentials(monkeypatch) -> None:
    monkeypatch.setenv("JWT_SECRET_KEY", "j" * 64)
    monkeypatch.setenv("ENCRYPTION_KEY", "e" * 64)
    from server.http_server import _cors_configuration

    monkeypatch.setenv("CORS_ALLOWED_ORIGINS", "*")
    assert _cors_configuration() == (["*"], False)

    monkeypatch.setenv(
        "CORS_ALLOWED_ORIGINS", "https://memory.example, https://admin.example"
    )
    assert _cors_configuration() == (
        ["https://memory.example", "https://admin.example"],
        True,
    )


def test_durable_queue_idempotency_wins_over_admission_limit(tmp_path) -> None:
    store = DurableJobStore(str(tmp_path / "queue.db"))
    first = store.enqueue_or_get(
        user_id="user-a",
        kind="memory_store",
        idempotency_key="same-request",
        payload={"content": "first"},
        max_inflight_writes=1,
    )
    duplicate = store.enqueue_or_get(
        user_id="user-a",
        kind="memory_store",
        idempotency_key="same-request",
        payload={"content": "different-but-replayed"},
        max_inflight_writes=1,
    )
    rejected = store.enqueue_or_get(
        user_id="user-a",
        kind="memory_store",
        idempotency_key="another-request",
        payload={"content": "over limit"},
        max_inflight_writes=1,
    )

    assert first["accepted"] is True
    assert duplicate["existing"] is True
    assert duplicate["job_id"] == first["job_id"]
    assert rejected["accepted"] is False
    assert rejected["retryable"] is True


def test_durable_queue_recovers_expired_processing_lease(tmp_path) -> None:
    db_path = tmp_path / "queue.db"
    store = DurableJobStore(str(db_path))
    queued = store.enqueue_or_get(
        user_id="user-a",
        kind="memory_store",
        idempotency_key="lease-request",
        payload={"content": "durable"},
        max_inflight_writes=10,
    )
    claimed = _claim_abandoned_fixture(store, worker_id="worker-a", batch_size=1, lease_seconds=5)
    assert claimed and claimed[0]["job_id"] == queued["job_id"]

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "UPDATE durable_jobs SET lease_expires_at_utc = ? WHERE job_id = ?",
            ("2000-01-01T00:00:00+00:00", queued["job_id"]),
        )

    assert store.recover_expired_leases() == 1
    recovered = store.get_job(job_id=queued["job_id"])
    assert recovered is not None
    assert recovered["status"] == "failed_retryable"
    assert recovered["attempt_count"] == 1
    assert recovered["lease_owner"] is None


def test_durable_queue_fences_stale_lease_completion(tmp_path) -> None:
    store = DurableJobStore(str(tmp_path / "queue.db"))
    queued = store.enqueue_or_get(
        user_id="user-a",
        kind="memory_store",
        idempotency_key="fenced-request",
        payload={"content": "durable"},
        max_inflight_writes=10,
    )
    stale = _claim_abandoned_fixture(store, worker_id="worker-a", batch_size=1, lease_seconds=5)[0]
    with sqlite3.connect(store.db_path) as conn:
        conn.execute(
            "UPDATE durable_jobs SET lease_expires_at_utc = ? WHERE job_id = ?",
            ("2000-01-01T00:00:00+00:00", queued["job_id"]),
        )
    assert store.recover_expired_leases(max_retries=6) == 1
    current = _claim_abandoned_fixture(store, worker_id="worker-b", batch_size=1, lease_seconds=5)[0]
    assert store.mark_done(
        job_id=queued["job_id"],
        lease_owner=stale["lease_owner"],
        result={"worker": "stale"},
    ) is False
    assert store.mark_done(
        job_id=queued["job_id"],
        lease_owner=current["lease_owner"],
        result={"worker": "current"},
    ) is True
    assert store.get_job(job_id=queued["job_id"])["result"] == {"worker": "current"}


def test_durable_queue_failed_write_is_retryable_and_reclaimable(tmp_path) -> None:
    db_path = tmp_path / "queue.db"
    store = DurableJobStore(str(db_path))
    queued = store.enqueue_or_get(
        user_id="user-a",
        kind="memory_store",
        idempotency_key="retryable-write",
        payload={"content": "durable"},
        max_inflight_writes=10,
    )
    claimed = _claim_abandoned_fixture(store,
        worker_id="failed-worker", batch_size=1, lease_seconds=5
    )[0]

    failed = store.mark_failed(
        job_id=queued["job_id"],
        lease_owner=claimed["lease_owner"],
        error="backend write failed",
        retryable=True,
        max_retries=3,
        backoff_base_ms=1,
    )
    assert failed["status"] == "failed_retryable"

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "UPDATE durable_jobs SET available_at_utc = ? WHERE job_id = ?",
            ("2000-01-01T00:00:00+00:00", queued["job_id"]),
        )

    retry = _claim_abandoned_fixture(store,
        worker_id="retry-worker", batch_size=1, lease_seconds=5
    )[0]
    assert retry["job_id"] == queued["job_id"]
    assert store.mark_done(
        job_id=retry["job_id"],
        lease_owner=retry["lease_owner"],
        result={"recovered": True},
    )


def test_durable_queue_has_global_admission_limit(tmp_path) -> None:
    store = DurableJobStore(str(tmp_path / "queue.db"))
    first = store.enqueue_or_get(
        user_id="user-a",
        kind="memory_store",
        idempotency_key="global-a",
        payload={"content": "a"},
        max_inflight_writes=10,
        max_global_inflight_writes=1,
    )
    second = store.enqueue_or_get(
        user_id="user-b",
        kind="memory_store",
        idempotency_key="global-b",
        payload={"content": "b"},
        max_inflight_writes=10,
        max_global_inflight_writes=1,
    )
    assert first["accepted"] is True
    assert second["reason"] == "memory_write_global_admission_limit"


def test_durable_queue_claims_are_fair_across_users(tmp_path) -> None:
    store = DurableJobStore(str(tmp_path / "queue.db"))
    for index in range(5):
        store.enqueue_or_get(
            user_id="user-a",
            kind="memory_store",
            idempotency_key=f"fair-a-{index}",
            payload={"content": "a"},
            max_inflight_writes=20,
        )
    store.enqueue_or_get(
        user_id="user-b",
        kind="memory_store",
        idempotency_key="fair-b",
        payload={"content": "b"},
        max_inflight_writes=20,
    )
    claimed = _claim_abandoned_fixture(store,
        worker_id="fair-worker", batch_size=5, lease_seconds=5, max_jobs_per_user=2
    )
    assert {job["user_id"] for job in claimed} == {"user-a", "user-b"}


def test_durable_queue_purge_removes_only_old_terminal_jobs(tmp_path) -> None:
    db_path = tmp_path / "queue.db"
    store = DurableJobStore(str(db_path))
    done = store.enqueue_or_get(
        user_id="user-a",
        kind="memory_store",
        idempotency_key="old-done",
        payload={"content": "done"},
        max_inflight_writes=10,
    )
    pending = store.enqueue_or_get(
        user_id="user-a",
        kind="memory_store",
        idempotency_key="pending",
        payload={"content": "pending"},
        max_inflight_writes=10,
    )
    claimed = _claim_abandoned_fixture(store, worker_id="purge-test", batch_size=2, lease_seconds=5)
    done_claim = next(job for job in claimed if job["job_id"] == done["job_id"])
    assert store.mark_done(
        job_id=done["job_id"],
        lease_owner=done_claim["lease_owner"],
        result={"ok": True},
    )
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "UPDATE durable_jobs SET updated_at_utc = ? WHERE job_id = ?",
            ("2000-01-01T00:00:00+00:00", done["job_id"]),
        )
    assert store.purge_completed_jobs(max_age_hours=1, batch_size=1) == 1
    assert store.get_job(job_id=done["job_id"]) is None
    assert store.get_job(job_id=pending["job_id"]) is not None


def test_durable_queue_retention_boundary_is_explicit(tmp_path) -> None:
    db_path = tmp_path / "queue.db"
    store = DurableJobStore(str(db_path))
    jobs = []
    for key in ("older-than-retention", "newer-than-retention"):
        queued = store.enqueue_or_get(
            user_id="user-a",
            kind="memory_store",
            idempotency_key=key,
            payload={"content": key},
            max_inflight_writes=10,
        )
        claimed = _claim_abandoned_fixture(store,
            worker_id=f"retention-{key}", batch_size=1, lease_seconds=5
        )[0]
        assert store.mark_done(
            job_id=queued["job_id"],
            lease_owner=claimed["lease_owner"],
            result={"ok": True},
        )
        jobs.append(queued["job_id"])

    now = datetime.now(timezone.utc)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "UPDATE durable_jobs SET updated_at_utc = ? WHERE job_id = ?",
            ((now - timedelta(hours=24, seconds=5)).isoformat(), jobs[0]),
        )
        conn.execute(
            "UPDATE durable_jobs SET updated_at_utc = ? WHERE job_id = ?",
            ((now - timedelta(hours=24) + timedelta(seconds=5)).isoformat(), jobs[1]),
        )

    assert store.purge_completed_jobs(max_age_hours=24, batch_size=10) == 1
    assert store.get_job(job_id=jobs[0]) is None
    assert store.get_job(job_id=jobs[1]) is not None


def test_rerank_rejects_malformed_success_response() -> None:
    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"results": "not-a-list"}

    class Client:
        async def post(self, *_args, **_kwargs):
            return Response()

    client = OpenRouterClient(api_key="test", base_url="http://127.0.0.1:1/v1")
    client._get_rerank_client = lambda: Client()

    with pytest.raises(RuntimeError, match="rerank_response_missing_results"):
        asyncio.run(client.rerank(query="q", documents=["a", "b"]))


def test_structured_search_treats_adversarial_person_as_literal() -> None:
    class Table:
        def count_rows(self) -> int:
            return 1

        def to_pandas(self):
            return pd.DataFrame(
                [
                    {
                        "entry_id": "private-1",
                        "lossless_restatement": "Alice's private fact",
                        "keywords": [],
                        "timestamp": "2026-01-01T00:00:00+00:00",
                        "location": "Denver",
                        "persons": ["Alice"],
                        "entities": [],
                        "topic": "private",
                    }
                ]
            )

    store = object.__new__(MultiTenantVectorStore)
    store._get_table = lambda _table_name: Table()
    results = asyncio.run(
        store.structured_search(
            "mem_user_a",
            persons=["Alice')) OR true--"],
            top_k=10,
        )
    )

    assert results == []


def test_vector_store_adapts_embedding_dimensions_deterministically() -> None:
    assert MultiTenantVectorStore._adapt_vector([1, 2, 3], 2) == [1.0, 2.0]
    assert MultiTenantVectorStore._adapt_vector([1], 3) == [1.0, 0.0, 0.0]


def test_lancedb_table_listing_failure_is_not_silent() -> None:
    class BrokenDB:
        def list_tables(self):
            raise OSError("metadata unavailable")

    store = object.__new__(MultiTenantVectorStore)
    store.db = BrokenDB()
    with pytest.raises(RuntimeError, match="table listing failed"):
        store.table_names()


def test_user_table_names_do_not_collide_after_identifier_normalization() -> None:
    dashed = User(user_id="foo-bar")
    underscored = User(user_id="foo_bar")
    assert dashed.table_name != underscored.table_name


def test_automatic_idempotency_includes_write_affecting_metadata() -> None:
    handler = object.__new__(MCPHandler)
    handler.user = User(user_id="user-a")
    first = handler._compute_idempotency_key(
        kind="memory_store",
        args={"content": "same", "tags": ["one"], "confidence": "high"},
    )
    second = handler._compute_idempotency_key(
        kind="memory_store",
        args={"content": "same", "tags": ["two"], "confidence": "low"},
    )
    assert first != second


def test_durable_memory_ids_are_replay_stable_and_distinguish_duplicates() -> None:
    class Client:
        def __init__(self, payload):
            self.payload = payload

        async def chat_completion(self, **_kwargs):
            return self.payload

        def extract_json(self, response):
            return response

    class Store:
        pass

    dialogue = [Dialogue(dialogue_id=1, speaker="user", content="fact")]
    first_payload = {
        "entries": [
            {
                "lossless_restatement": "Alice likes tea",
                "keywords": ["tea"],
                "timestamp": None,
                "location": None,
                "persons": ["Alice"],
                "entities": [],
                "topic": "preference",
                "ignored": "one",
            },
            {
                "lossless_restatement": "Alice likes tea",
                "keywords": ["tea"],
                "timestamp": None,
                "location": None,
                "persons": ["Alice"],
                "entities": [],
                "topic": "preference",
                "ignored": "two",
            },
        ]
    }
    second_payload = {
        "entries": [
            {**first_payload["entries"][0], "ignored": "changed"},
            {**first_payload["entries"][1], "ignored": "changed-again"},
        ]
    }

    async def generate(payload):
        builder = MemoryBuilder(
            Client(payload), Store(), "table", entry_id_namespace="job:test"
        )
        return await builder._generate_memory_entries(dialogue)

    first = asyncio.run(generate(first_payload))
    second = asyncio.run(generate(second_payload))
    assert [entry.entry_id for entry in first] == [
        entry.entry_id for entry in second
    ]
    assert len({entry.entry_id for entry in first}) == 2


def test_memory_extraction_failure_propagates_for_durable_retry() -> None:
    class FailingClient:
        async def chat_completion(self, **_kwargs):
            raise RuntimeError("local inference unavailable")

        def extract_json(self, _response):
            return None

    builder = MemoryBuilder(
        FailingClient(),
        object(),
        "table",
        max_retries=2,
    )
    dialogue = [Dialogue(dialogue_id=1, speaker="user", content="fact")]
    with pytest.raises(RuntimeError, match="memory extraction failed"):
        asyncio.run(builder._generate_memory_entries(dialogue))


def test_malformed_memory_extraction_shape_propagates_for_durable_retry() -> None:
    class MalformedClient:
        async def chat_completion(self, **_kwargs):
            return "{}"

        def extract_json(self, _response):
            return {}

    builder = MemoryBuilder(MalformedClient(), object(), "table", max_retries=2)
    dialogue = [Dialogue(dialogue_id=1, speaker="user", content="fact")]
    with pytest.raises(RuntimeError, match="memory extraction failed"):
        asyncio.run(builder._generate_memory_entries(dialogue))


def test_malformed_memory_extraction_entry_propagates_for_durable_retry() -> None:
    class MalformedEntryClient:
        async def chat_completion(self, **_kwargs):
            return "{\"entries\":[{}]}"

        def extract_json(self, _response):
            return {"entries": [{}]}

    builder = MemoryBuilder(MalformedEntryClient(), object(), "table", max_retries=2)
    dialogue = [Dialogue(dialogue_id=1, speaker="user", content="fact")]
    with pytest.raises(RuntimeError, match="memory extraction failed"):
        asyncio.run(builder._generate_memory_entries(dialogue))


def test_rate_limiter_key_count_is_bounded() -> None:
    from server.http_server import _RateLimiter

    limiter = _RateLimiter(max_requests=100, window_seconds=60, max_keys=100)
    for index in range(250):
        assert limiter.is_allowed(f"client-{index}") is True
    assert len(limiter._counts) <= 100


def test_settings_reject_non_local_inference_endpoint(tmp_path) -> None:
    with pytest.raises(ValueError, match="loopback"):
        Settings(
            jwt_secret_key="j" * 64,
            encryption_key="e" * 64,
            openrouter_base_url="https://example.invalid/v1",
            openrouter_embedding_base_url="http://127.0.0.1:11501/v1",
            openrouter_rerank_base_url="http://127.0.0.1:11502/v1",
            data_dir=str(tmp_path / "data"),
            lancedb_path=str(tmp_path / "lancedb"),
            durable_queue_db_path=str(tmp_path / "queue.db"),
        )


def test_mcp_rejects_cross_user_job_status(tmp_path, monkeypatch) -> None:
    store = DurableJobStore(str(tmp_path / "queue.db"))
    job = store.enqueue_or_get(
        user_id="user-b",
        kind="memory_store",
        idempotency_key="other-user",
        payload={"content": "private"},
        max_inflight_writes=10,
    )
    monkeypatch.setattr(MCPHandler, "_durable_store", store)
    handler = object.__new__(MCPHandler)
    handler.user = User(user_id="user-a", table_name="mem_user_a")

    result = asyncio.run(handler._tool_memory_job_status({"job_id": job["job_id"]}))

    assert result["found"] is False


def test_mcp_unknown_method_is_json_rpc_method_not_found(deliver_ready_message) -> None:
    handler = object.__new__(MCPHandler)
    response = asyncio.run(
        deliver_ready_message(handler,
            json.dumps({"jsonrpc": "2.0", "id": "probe", "method": "unknown"})
        )
    )
    payload = json.loads(response)

    assert payload["error"]["code"] == -32601


def test_mcp_rejects_missing_jsonrpc_version() -> None:
    handler = object.__new__(MCPHandler)
    response = asyncio.run(handler.handle_message('{"id":"missing-version","method":"ping"}'))
    payload = json.loads(response)
    assert payload["error"]["code"] == -32600


def test_mcp_stats_scopes_durable_queue_to_authenticated_user(monkeypatch) -> None:
    class Store:
        def queue_counts(self, user_id=None):
            assert user_id == "user-a"
            return {
                "pending": 1,
                "processing": 0,
                "done": 2,
                "failed_retryable": 0,
                "failed_terminal": 0,
                "oldest_pending_age_s": 1.0,
            }

    monkeypatch.setattr(MCPHandler, "_durable_store", Store())
    monkeypatch.setattr(
        MCPHandler,
        "_write_stats_by_user",
        {"user-a": {"accepted": 1, "completed": 1}},
    )
    handler = object.__new__(MCPHandler)
    handler.user = User(user_id="user-a")
    handler.vector_store = SimpleNamespace(
        get_stats=lambda _table: {"entry_count": 2}
    )
    result = asyncio.run(handler._tool_memory_stats({}))
    assert result["durable_queue"]["pending"] == 1
    assert result["write_queue"]["accepted_total"] == 1


def test_mcp_per_user_stats_are_bounded() -> None:
    original = MCPHandler._write_stats_by_user
    try:
        MCPHandler._write_stats_by_user = {}
        for index in range(MCPHandler._MAX_USER_STATS + 1):
            MCPHandler._user_write_stat(f"user-{index}", "accepted")
        assert len(MCPHandler._write_stats_by_user) == MCPHandler._MAX_USER_STATS
    finally:
        MCPHandler._write_stats_by_user = original


@pytest.mark.parametrize(
    ("message", "error_code"),
    [
        ("[]", -32600),
        ("null", -32600),
        ('{"jsonrpc":"1.0"}', -32600),
        ('{"jsonrpc":"2.0","method":"ping","params":[]}', -32602),
    ],
)
def test_mcp_malformed_request_is_not_reported_as_internal_error(
    message: str, error_code: int
) -> None:
    handler = object.__new__(MCPHandler)
    response = asyncio.run(handler.handle_message(message))
    payload = json.loads(response)

    assert payload["error"]["code"] == error_code


@pytest.mark.parametrize(
    ("tool", "arguments"),
    [
        ("memory_store", {"content": "fixture"}),
        ("memory_add", {"content": "fixture"}),
        ("memory_add_batch", {"dialogues": [{"speaker": "user", "content": "fixture"}]}),
    ],
)
def test_distinct_oversize_idempotency_keys_fail_without_enqueue(monkeypatch, deliver_ready_message, tool, arguments) -> None:
    class ForbiddenStore:
        def enqueue_or_get(self, **_kwargs):
            pytest.fail("oversize idempotency key reached the durable store")

    monkeypatch.setattr(MCPHandler, "_durable_store", ForbiddenStore())
    handler = MCPHandler.__new__(MCPHandler)
    handler.user = User(user_id="fixture")
    handler.settings = SimpleNamespace(
        memory_write_max_content_chars=8192,
        memory_write_max_metadata_chars=2048,
    )
    for suffix in ("A", "B"):
        request = {"jsonrpc": "2.0", "id": 42, "method": "tools/call", "params": {
            "name": tool,
            "arguments": {**arguments, "idempotency_key": "k" * 256 + suffix},
        }}
        response = json.loads(asyncio.run(deliver_ready_message(handler, json.dumps(request))))
        assert response["id"] == 42
        assert response["error"] == {"code": -32602, "message": "idempotency_key exceeds 256 characters"}


def test_idempotency_limit_preserves_valid_keys_and_is_advertised() -> None:
    handler = MCPHandler.__new__(MCPHandler)
    handler.user = User(user_id="fixture")
    assert handler._compute_idempotency_key(kind="memory_store", args={"idempotency_key": "k" * 256}) == "k" * 256
    assert handler._compute_idempotency_key(kind="memory_store", args={"idempotency_key": "  key  "}) == "key"
    assert handler._compute_idempotency_key(kind="memory_store", args={"idempotency_key": " " + "k" * 256 + " "}) == "k" * 256
    assert handler._compute_idempotency_key(kind="memory_store", args={"idempotency_key": " " * 257}) == handler._compute_idempotency_key(kind="memory_store", args={})
    with pytest.raises(ValueError, match="idempotency_key exceeds 256"):
        handler._compute_idempotency_key(kind="memory_store", args={"idempotency_key": " " + "k" * 257 + " "})
    tools = asyncio.run(handler._handle_tools_list({}))["tools"]
    write_tools = {tool["name"]: tool for tool in tools if tool["name"] in {"memory_store", "memory_add", "memory_add_batch"}}
    assert set(write_tools) == {"memory_store", "memory_add", "memory_add_batch"}
    for tool in write_tools.values():
        schema = tool["inputSchema"]["properties"]["idempotency_key"]
        assert "maxLength" not in schema
        assert "256 characters after trimming" in schema["description"]


def test_batch_payload_limit_rejects_oversize_before_durable_enqueue(monkeypatch, deliver_ready_message) -> None:
    class ForbiddenStore:
        def enqueue_or_get(self, **_kwargs):
            pytest.fail("oversize payload reached the durable store")

    monkeypatch.setattr(MCPHandler, "_durable_store", ForbiddenStore())
    handler = MCPHandler.__new__(MCPHandler)
    handler.user = User(user_id="fixture")
    handler.settings = SimpleNamespace(
        memory_write_max_content_chars=8192,
        memory_write_max_batch_content_chars=32768,
        memory_write_max_batch_items=16,
        memory_write_max_payload_bytes=65536,
    )
    request = {"jsonrpc": "2.0", "id": 43, "method": "tools/call", "params": {
        "name": "memory_add_batch",
        "arguments": {"idempotency_key": "bounded", "dialogues": [{"content": "fixture", "extra": "x" * 70000}]},
    }}
    response = json.loads(asyncio.run(deliver_ready_message(handler, json.dumps(request))))
    assert response["id"] == 43
    assert response["error"]["code"] == -32602
    assert "payload exceeds 65536 bytes" in response["error"]["message"]


def test_lease_recovery_retries_unprocessed_rows_after_partial_failure(tmp_path) -> None:
    store = DurableJobStore(str(tmp_path / "lease-recovery.db"))
    for user in ("user-a", "user-b"):
        store.enqueue_or_get(user_id=user, kind="memory_store", idempotency_key=user,
                             payload={"content": user}, max_inflight_writes=10)
    claimed = _claim_abandoned_fixture(store, worker_id="fixture", batch_size=2, lease_seconds=5)
    assert len(claimed) == 2
    conn = store._get_connection()
    conn.execute("UPDATE durable_jobs SET lease_expires_at_utc = '2000-01-01T00:00:00+00:00'")
    rows = conn.execute("SELECT job_id FROM durable_jobs WHERE status = 'processing' AND lease_expires_at_utc < ?", (store._utc_now_iso(),)).fetchall()
    first, second = (str(row["job_id"]) for row in rows)
    conn.create_function("is_fixture_second_job", 1, lambda job_id: job_id == second)
    conn.execute("CREATE TRIGGER inject_recovery_error BEFORE UPDATE ON durable_jobs WHEN is_fixture_second_job(OLD.job_id) AND NEW.status = 'failed_retryable' BEGIN SELECT RAISE(ABORT, 'fixture recovery failure'); END")
    try:
        with pytest.raises(sqlite3.IntegrityError, match="fixture recovery failure"):
            store.recover_expired_leases()
        assert store.get_job(job_id=first)["status"] == "failed_retryable"
        assert store.get_job(job_id=second)["status"] == "processing"
        conn.execute("DROP TRIGGER inject_recovery_error")
        assert store.recover_expired_leases() == 1
        for job_id in (first, second):
            row = store.get_job(job_id=job_id)
            assert row["status"] == "failed_retryable" and row["attempt_count"] == 1
    finally:
        store.close()


def test_user_credential_is_encrypted_before_durable_storage(tmp_path) -> None:
    from server.auth.token_manager import TokenManager
    from server.database.user_store import UserStore

    manager = TokenManager(secret_key="j" * 64, encryption_key="e" * 64)
    cleartext = "private-fixture-key"
    user = User(user_id="fixture")
    user.openrouter_api_key_encrypted = manager.encrypt_api_key(cleartext)
    assert cleartext not in user.openrouter_api_key_encrypted
    store = UserStore(str(tmp_path / "users.db"))
    try:
        store.create_user(user)
        stored = store.get_user(user.user_id)
        assert stored is not None
        assert cleartext not in stored.openrouter_api_key_encrypted
        assert manager.decrypt_api_key(stored.openrouter_api_key_encrypted) == cleartext
    finally:
        store._get_connection().close()


def test_stale_queued_batch_job_is_not_executed_twice(tmp_path, monkeypatch) -> None:
    """A reclaimed job in another worker's batch must not run a second time."""
    from collections import deque

    async def scenario():
        store = DurableJobStore(str(tmp_path / "stale-batch.db"))
        ids = [
            store.enqueue_or_get(
                user_id=user, kind="memory_store", idempotency_key=user,
                payload={"content": user}, max_inflight_writes=10,
            )["job_id"]
            for user in ("fixture-a", "fixture-b")
        ]
        settings = SimpleNamespace(
            memory_write_lease_seconds=90, memory_write_max_retries=6,
            memory_write_batch_max=2, memory_write_batch_max_wait_ms=10,
            memory_write_job_status_limit=100, memory_write_retry_backoff_base_ms=100,
        )
        monkeypatch.setattr(MCPHandler, "_durable_store", store)
        monkeypatch.setattr(MCPHandler, "_shared_settings", settings)
        monkeypatch.setattr(MCPHandler, "_worker_shutdown", asyncio.Event())
        monkeypatch.setattr(MCPHandler, "_lease_recoveries", 0)
        monkeypatch.setattr(MCPHandler, "_write_status_by_job_id", {})
        monkeypatch.setattr(MCPHandler, "_write_status_order", deque())
        monkeypatch.setattr(MCPHandler, "_write_stats", dict(MCPHandler._write_stats))
        monkeypatch.setattr(MCPHandler, "_write_stats_by_user", {})
        monkeypatch.setattr(MCPHandler, "_invalidate_memory_caches", classmethod(lambda cls: None))
        first_started, release_first = asyncio.Event(), asyncio.Event()
        second_started, duplicate = asyncio.Event(), asyncio.Event()
        seen = []
        stale_id = None

        async def execute_fixture(cls, job):
            seen.append(job["job_id"])
            if len(seen) == 1:
                first_started.set()
                await release_first.wait()
            elif job["job_id"] == stale_id:
                (second_started if seen.count(stale_id) == 1 else duplicate).set()
            return {"fixture": True}

        monkeypatch.setattr(MCPHandler, "_execute_durable_job", classmethod(execute_fixture))
        async def until(predicate):
            while not predicate():
                await asyncio.sleep(0.01)

        worker_one = asyncio.create_task(MCPHandler._durable_worker_loop(1))
        worker_two = None
        try:
            await asyncio.wait_for(first_started.wait(), 5)
            stale_id = next(job_id for job_id in ids if job_id != seen[0])
            assert store.get_job(job_id=stale_id)["status"] == "processing"
            with sqlite3.connect(store.db_path) as conn:
                conn.execute(
                    "UPDATE durable_jobs SET lease_expires_at_utc = ? WHERE job_id = ?",
                    ("2000-01-01T00:00:00+00:00", stale_id),
                )
            assert store.recover_expired_leases() == 0
            worker_two = asyncio.create_task(MCPHandler._durable_worker_loop(2))
            await asyncio.sleep(0.05)
            assert not second_started.is_set()
            assert store.get_job(job_id=stale_id)["attempt_count"] == 0
            release_first.set()
            await asyncio.wait_for(second_started.wait(), 5)
            await asyncio.wait_for(until(lambda: store.get_job(job_id=stale_id)["status"] == "done"), 5)
            assert not duplicate.is_set() and seen.count(stale_id) == 1
        finally:
            MCPHandler._worker_shutdown.set()
            release_first.set()
            for task in [worker_one, worker_two]:
                if task is not None:
                    task.cancel()
            await asyncio.gather(*(task for task in [worker_one, worker_two] if task is not None), return_exceptions=True)
            store.close()

    asyncio.run(scenario())


def test_lease_recovery_process_crash_preserves_fencing(tmp_path) -> None:
    """A crash between row recoveries preserves each job and fences old owners."""
    import os
    import subprocess
    import sys

    db = tmp_path / "recovery-crash.db"
    store = DurableJobStore(str(db))
    job_ids = [
        store.enqueue_or_get(
            user_id=user, kind="memory_store", idempotency_key=user,
            payload={"content": user}, max_inflight_writes=10,
        )["job_id"]
        for user in ("fixture-a", "fixture-b")
    ]
    assert len(_claim_abandoned_fixture(store, worker_id="old-worker", batch_size=2, lease_seconds=5)) == 2
    conn = store._get_connection()
    conn.execute("UPDATE durable_jobs SET lease_expires_at_utc = '2000-01-01T00:00:00+00:00'")
    rows = conn.execute(
        "SELECT job_id FROM durable_jobs WHERE status = 'processing' AND lease_expires_at_utc < ?",
        (store._utc_now_iso(),),
    ).fetchall()
    first, second = (str(row["job_id"]) for row in rows)
    conn.execute(
        "CREATE TRIGGER fixture_crash BEFORE UPDATE ON durable_jobs "
        "WHEN NEW.status = 'failed_retryable' AND crash_on_job(OLD.job_id) "
        "BEGIN SELECT 1; END"
    )
    child = """
import os, sys
from server.database.durable_job_store import DurableJobStore
store = DurableJobStore(sys.argv[1])
def crash_on_job(job_id):
    if job_id == sys.argv[2]:
        os._exit(73)
    return 0
store._get_connection().create_function('crash_on_job', 1, crash_on_job)
store.recover_expired_leases()
raise SystemExit(99)
"""
    env = dict(os.environ)
    try:
        proc = subprocess.run([sys.executable, "-c", child, str(db), second], env=env, capture_output=True, text=True)
        assert proc.returncode == 73, proc.stderr
        assert store.get_job(job_id=first)["status"] == "failed_retryable"
        assert store.get_job(job_id=second)["status"] == "processing"
        conn.execute("DROP TRIGGER fixture_crash")
        assert store.recover_expired_leases() == 1
        claimed = _claim_abandoned_fixture(store, worker_id="new-worker", batch_size=2, lease_seconds=5)
        assert {job["job_id"] for job in claimed} == set(job_ids)
        for job_id in job_ids:
            assert not store.mark_done(job_id=job_id, lease_owner="old-worker", result={"stale": True})
            assert store.mark_done(job_id=job_id, lease_owner="new-worker", result={"fresh": True})
            assert not store.mark_done(job_id=job_id, lease_owner="new-worker", result={"duplicate": True})
            assert store.get_job(job_id=job_id)["result"] == {"fresh": True}
    finally:
        store.close()


def test_pre_execution_fence_rejects_a_reassigned_job(tmp_path, monkeypatch) -> None:
    store = DurableJobStore(str(tmp_path / "pre-execution-fence.db"))
    try:
        job_id = store.enqueue_or_get(user_id="fixture", kind="memory_store", idempotency_key="key", payload={"content": "fixture"}, max_inflight_writes=10)["job_id"]
        stale = _claim_abandoned_fixture(store, worker_id="old", batch_size=1, lease_seconds=5)[0]
        store._get_connection().execute("UPDATE durable_jobs SET lease_expires_at_utc = '2000-01-01T00:00:00+00:00' WHERE job_id = ?", (job_id,))
        assert store.recover_expired_leases() == 1
        assert _claim_abandoned_fixture(store, worker_id="new", batch_size=1, lease_seconds=5)
        called = []
        async def execute_fixture(cls, job):
            called.append(job)
        monkeypatch.setattr(MCPHandler, "_execute_durable_job", classmethod(execute_fixture))
        with pytest.raises(RuntimeError, match="lease_lost_before_execution"):
            asyncio.run(MCPHandler._execute_durable_job_with_lease(store=store, job=stale, lease_seconds=5))
        assert called == []
    finally:
        store.close()
