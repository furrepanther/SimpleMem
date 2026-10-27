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
    claimed = store.claim_batch(worker_id="worker-a", batch_size=1, lease_seconds=5)
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
    stale = store.claim_batch(worker_id="worker-a", batch_size=1, lease_seconds=5)[0]
    with sqlite3.connect(store.db_path) as conn:
        conn.execute(
            "UPDATE durable_jobs SET lease_expires_at_utc = ? WHERE job_id = ?",
            ("2000-01-01T00:00:00+00:00", queued["job_id"]),
        )
    assert store.recover_expired_leases(max_retries=6) == 1
    current = store.claim_batch(worker_id="worker-b", batch_size=1, lease_seconds=5)[0]
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
    claimed = store.claim_batch(
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

    retry = store.claim_batch(
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
    claimed = store.claim_batch(
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
    claimed = store.claim_batch(worker_id="purge-test", batch_size=2, lease_seconds=5)
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
        claimed = store.claim_batch(
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


def test_mcp_unknown_method_is_json_rpc_method_not_found() -> None:
    handler = object.__new__(MCPHandler)
    response = asyncio.run(
        handler.handle_message(
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
