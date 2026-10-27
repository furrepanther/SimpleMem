import asyncio
import threading
import time
from types import SimpleNamespace

import pandas as pd
import pytest

from server.database.vector_store import MultiTenantVectorStore


class RecordingTable:
    def __init__(self) -> None:
        self.active_writers = 0
        self.max_active_writers = 0
        self.records: list[dict] = []
        self.lock = threading.Lock()

    def add(self, records: list[dict]) -> None:
        with self.lock:
            self.active_writers += 1
            self.max_active_writers = max(
                self.max_active_writers, self.active_writers
            )
        time.sleep(0.02)
        self.records.extend(records)
        with self.lock:
            self.active_writers -= 1

    def merge_insert(self, _on: str):
        return self

    def when_not_matched_insert_all(self):
        return self

    def execute(self, records: list[dict]) -> None:
        with self.lock:
            self.active_writers += 1
            self.max_active_writers = max(
                self.max_active_writers, self.active_writers
            )
        time.sleep(0.02)
        existing = {row["entry_id"] for row in self.records}
        self.records.extend(row for row in records if row["entry_id"] not in existing)
        with self.lock:
            self.active_writers -= 1


def _store_for(table: RecordingTable) -> MultiTenantVectorStore:
    store = object.__new__(MultiTenantVectorStore)
    store.embedding_dimension = 2
    store._tables = {"mem_global": table}
    store._table_dimensions = {"mem_global": 2}
    store._table_write_locks = {}
    store._table_write_locks_guard = threading.Lock()
    return store


def test_concurrent_adds_serialize_lance_mutations() -> None:
    table = RecordingTable()
    store = _store_for(table)

    async def append(index: int) -> None:
        entry = SimpleNamespace(
            entry_id=f"entry-{index}",
            lossless_restatement=f"record-{index}",
            keywords=[],
            timestamp=None,
            location=None,
            persons=[],
            entities=[],
            topic=None,
        )
        assert await store.add_entries("mem_global", [entry], [[1.0, 0.0]]) == 1

    async def exercise() -> None:
        await asyncio.gather(*(append(index) for index in range(8)))

    asyncio.run(exercise())

    assert table.max_active_writers == 1
    assert len(table.records) == 8


def test_concurrent_writes_keep_tenant_records_separate() -> None:
    table_a = RecordingTable()
    table_b = RecordingTable()
    store = _store_for(table_a)
    store._tables = {"mem_a": table_a, "mem_b": table_b}
    store._table_dimensions = {"mem_a": 2, "mem_b": 2}

    async def append(table_name: str, index: int) -> None:
        entry = SimpleNamespace(
            entry_id=f"{table_name}-entry-{index}",
            lossless_restatement=f"{table_name}-record-{index}",
            keywords=[],
            timestamp=None,
            location=None,
            persons=[],
            entities=[],
            topic=None,
        )
        assert await store.add_entries(table_name, [entry], [[1.0, 0.0]]) == 1

    async def exercise() -> None:
        await asyncio.gather(
            *(append(table_name, index) for table_name in ("mem_a", "mem_b") for index in range(8))
        )

    asyncio.run(exercise())

    assert table_a.max_active_writers == 1
    assert table_b.max_active_writers == 1
    assert {row["entry_id"] for row in table_a.records} == {
        f"mem_a-entry-{index}" for index in range(8)
    }
    assert {row["entry_id"] for row in table_b.records} == {
        f"mem_b-entry-{index}" for index in range(8)
    }


def test_replayed_entry_id_is_inserted_only_once(tmp_path) -> None:
    store = MultiTenantVectorStore(str(tmp_path / "lancedb"), embedding_dimension=2)
    entry = SimpleNamespace(
        entry_id="stable-entry",
        lossless_restatement="replayed record",
        keywords=[],
        timestamp=None,
        location=None,
        persons=[],
        entities=[],
        topic=None,
    )

    async def exercise() -> None:
        assert await store.add_entries("mem_replay", [entry], [[1.0, 0.0]]) == 1
        assert await store.add_entries("mem_replay", [entry], [[1.0, 0.0]]) == 1

    asyncio.run(exercise())
    assert store._get_table("mem_replay").count_rows() == 1


def test_replayed_entry_id_with_changed_metadata_stays_single_version(tmp_path) -> None:
    store = MultiTenantVectorStore(str(tmp_path / "lancedb"), embedding_dimension=2)
    first = SimpleNamespace(
        entry_id="stable-metadata-entry",
        lossless_restatement="original record",
        keywords=["original"],
        timestamp=None,
        location=None,
        persons=[],
        entities=[],
        topic="original",
    )
    replay = SimpleNamespace(
        entry_id="stable-metadata-entry",
        lossless_restatement="replayed record with changed metadata",
        keywords=["changed"],
        timestamp=None,
        location=None,
        persons=[],
        entities=[],
        topic="changed",
    )

    async def exercise() -> None:
        assert await store.add_entries("mem_replay_metadata", [first], [[1.0, 0.0]]) == 1
        assert await store.add_entries("mem_replay_metadata", [replay], [[0.0, 1.0]]) == 1

    asyncio.run(exercise())
    table = store._get_table("mem_replay_metadata")
    assert table.count_rows() == 1
    assert table.to_pandas().iloc[0]["lossless_restatement"] == "original record"


def test_missing_merge_insert_capability_fails_closed() -> None:
    class AddOnlyTable:
        def add(self, _records):
            raise AssertionError("append fallback must not be used")

    store = _store_for(AddOnlyTable())
    with pytest.raises(RuntimeError, match="merge_insert idempotency"):
        store._append_records("mem_global", [{"entry_id": "entry-1"}])


def test_get_entry_pushes_exact_id_filter_to_lance() -> None:
    class ExactIdQuery:
        def __init__(self) -> None:
            self.where_clause: str | None = None
            self.limit_value: int | None = None

        def where(self, clause: str):
            self.where_clause = clause
            return self

        def limit(self, value: int):
            self.limit_value = value
            return self

        def to_pandas(self):
            return pd.DataFrame(
                [
                    {
                        "entry_id": "exact'id",
                        "lossless_restatement": "exact record",
                        "keywords": ["exact"],
                        "timestamp": None,
                        "location": None,
                        "persons": [],
                        "entities": [],
                        "topic": None,
                    }
                ]
            )

    class ExactIdTable:
        def __init__(self) -> None:
            self.query = ExactIdQuery()

        def count_rows(self) -> int:
            return 100_000

        def search(self):
            return self.query

        def to_pandas(self):
            raise AssertionError("get_entry must not materialize the whole Lance table")

    table = ExactIdTable()
    store = _store_for(table)

    async def exercise() -> None:
        entry = await store.get_entry("mem_global", "exact'id")
        assert entry is not None
        assert entry.entry_id == "exact'id"

    asyncio.run(exercise())
    assert table.query.where_clause == "entry_id = 'exact''id'"
    assert table.query.limit_value == 1
