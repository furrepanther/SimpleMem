"""Update receipts require the requested values to survive real Lance readback."""
import asyncio
from types import SimpleNamespace

import pytest
from server.database.vector_store import MultiTenantVectorStore


def entry():
    return SimpleNamespace(entry_id="fixture-id", lossless_restatement="original", keywords=[],
                           timestamp=None, location=None, persons=[], entities=[], topic=None)


@pytest.mark.parametrize("mode", ["content", "vector", "metadata"])
def test_concurrent_replacement_cannot_produce_false_success(tmp_path, mode):
    store = MultiTenantVectorStore(str(tmp_path / "lance"), embedding_dimension=2)
    async def exercise():
        await store.add_entries("mem_fixture", [entry()], [[1.0, 0.0]])
        table = store._get_table("mem_fixture")
        original = store._row_to_record(table.to_pandas().iloc[0], table_dim=2)
        update = store._update_records
        def concurrent_replacement(table_name, where, values):
            store._delete_records(table_name, where)
            update(table_name, where, values)
            store._append_records(table_name, [original])
        store._update_records = concurrent_replacement
        fields = {"lossless_restatement": "changed"} if mode == "content" else ({"keywords": ["changed"]} if mode == "metadata" else {})
        vector = [0.0, 1.0] if mode == "vector" else None
        result = await store.update_entry("mem_fixture", "fixture-id", fields, embedding=vector)
        assert result["updated"] is False
        assert result["error"] == "table_update_verification_mismatch"
        assert (await store.get_entry("mem_fixture", "fixture-id")).lossless_restatement == "original"
    asyncio.run(exercise())


def test_successful_content_metadata_and_vector_update_is_verified(tmp_path):
    store = MultiTenantVectorStore(str(tmp_path / "lance"), embedding_dimension=2)
    async def exercise():
        await store.add_entries("mem_fixture", [entry()], [[1.0, 0.0]])
        result = await store.update_entry("mem_fixture", "fixture-id",
            {"lossless_restatement": "changed", "keywords": ["verified"]}, embedding=[0.3, 0.7])
        assert result["updated"] is True
        updated = await store.get_entry("mem_fixture", "fixture-id")
        assert updated.lossless_restatement == "changed" and updated.keywords == ["verified"]
    asyncio.run(exercise())
