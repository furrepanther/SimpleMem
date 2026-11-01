import asyncio
import warnings
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from server.auth.models import MemoryEntry, User
from server.core.answer_generator import AnswerGenerator
from server.core.retriever import Retriever
from server.database.vector_store import MultiTenantVectorStore
from server.mcp_handler import MCPHandler


def make_handler(generator):
    MCPHandler._invalidate_memory_caches()
    handler = object.__new__(MCPHandler)
    handler.user = SimpleNamespace(user_id="query-fixture", table_name="mem_fixture")
    handler.settings = SimpleNamespace(
        memory_query_max_question_chars=4096,
        memory_query_default_top_k=3,
        memory_query_max_top_k=5,
        memory_query_synthesis_timeout_ms=250,
        enable_reflection=False,
        enable_planning=False,
        memory_queue_max_inflight_reads=2,
    )

    async def retrieve(**kwargs):
        return [MemoryEntry(entry_id="fixture", lossless_restatement="A local fact.")]

    handler._get_retriever = lambda: SimpleNamespace(retrieve=retrieve)
    handler._get_answer_generator = lambda: generator
    return handler


def test_query_inflight_keeps_pending_owner_past_age_threshold():
    async def exercise():
        first = asyncio.get_running_loop().create_future()
        second = asyncio.get_running_loop().create_future()
        inflight, timestamps = {}, {}
        with patch("server.mcp_handler.time.monotonic", side_effect=[1000.0, 1301.0]):
            MCPHandler._add_inflight(inflight, timestamps, "first", first, preserve_pending=True)
            MCPHandler._add_inflight(inflight, timestamps, "second", second, preserve_pending=True)
        assert not first.done() and inflight["first"] is first

    asyncio.run(exercise())


def test_query_inflight_capacity_rejects_instead_of_evicting_pending_owner():
    async def exercise():
        first = asyncio.get_running_loop().create_future()
        second = asyncio.get_running_loop().create_future()
        inflight, timestamps = {}, {}
        with patch.object(MCPHandler, "_MAX_INFLIGHT", 1):
            MCPHandler._add_inflight(inflight, timestamps, "first", first, preserve_pending=True)
            with pytest.raises(RuntimeError, match="memory_query_overloaded"):
                MCPHandler._add_inflight(inflight, timestamps, "second", second, preserve_pending=True)
        assert inflight == {"first": first} and not first.done()

    asyncio.run(exercise())


def test_inflight_default_preserves_retrieve_legacy_expiration():
    async def exercise():
        first = asyncio.get_running_loop().create_future()
        second = asyncio.get_running_loop().create_future()
        inflight, timestamps = {}, {}
        with patch("server.mcp_handler.time.monotonic", side_effect=[1000.0, 1301.0]):
            MCPHandler._add_inflight(inflight, timestamps, "first", first)
            MCPHandler._add_inflight(inflight, timestamps, "second", second)
        assert "first" not in inflight and not first.done()

    asyncio.run(exercise())


def test_query_inflight_reclaims_completed_entries_at_capacity():
    async def exercise():
        finished = asyncio.get_running_loop().create_future()
        finished.set_result("done")
        next_request = asyncio.get_running_loop().create_future()
        inflight, timestamps = {}, {}
        with patch.object(MCPHandler, "_MAX_INFLIGHT", 1):
            MCPHandler._add_inflight(inflight, timestamps, "finished", finished, preserve_pending=True)
            MCPHandler._add_inflight(inflight, timestamps, "next", next_request, preserve_pending=True)
        assert inflight == {"next": next_request}

    asyncio.run(exercise())


def test_query_coalesces_pending_request_after_inflight_age_threshold():
    async def exercise():
        release = asyncio.Event()
        started = asyncio.Event()
        calls = []

        async def generate_answer(query, contexts):
            calls.append(query)
            if len(calls) == 2:
                started.set()
            await release.wait()
            return {"answer": query, "confidence": "high"}

        handler = make_handler(SimpleNamespace(generate_answer=generate_answer))
        with patch.object(MCPHandler, "_INFLIGHT_TTL_S", -1):
            first = asyncio.create_task(handler._tool_memory_query({"question": "first"}))
            await asyncio.sleep(0)
            other = asyncio.create_task(handler._tool_memory_query({"question": "other"}))
            await started.wait()
            waiter = asyncio.create_task(handler._tool_memory_query({"question": "first"}))
            await asyncio.sleep(0)
            release.set()
            results = await asyncio.gather(first, other, waiter)
        assert calls.count("first") == 1 and len(calls) == 2
        assert results[0]["answer"] == results[2]["answer"] == "first"
        assert not MCPHandler._query_inflight

    asyncio.run(exercise())


def test_query_does_not_cancel_synthesis_after_former_timeout():
    async def exercise():
        started = asyncio.Event()
        release = asyncio.Event()

        async def generate_answer(**kwargs):
            started.set()
            await release.wait()
            return {"answer": "Completed answer.", "reasoning": "fixture", "confidence": "high"}

        handler = make_handler(SimpleNamespace(generate_answer=generate_answer))
        task = asyncio.create_task(handler._tool_memory_query({"question": "slow fixture"}))
        await started.wait()
        await asyncio.sleep(0.35)  # Exceeds the prior fixture setting; never terminates a job.
        assert not task.done(), "duration alone completed or cancelled synthesis"
        release.set()
        result = await task
        assert result["answer"] == "Completed answer."
        assert result["confidence"] == "high"

    asyncio.run(exercise())


def test_query_owner_disconnect_cancels_coalesced_waiter_without_cache():
    async def exercise():
        started = asyncio.Event()

        async def generate_answer(**kwargs):
            started.set()
            await asyncio.Event().wait()

        handler = make_handler(SimpleNamespace(generate_answer=generate_answer))
        owner = asyncio.create_task(handler._tool_memory_query({"question": "disconnect fixture"}))
        await started.wait()
        waiter = asyncio.create_task(handler._tool_memory_query({"question": "disconnect fixture"}))
        await asyncio.sleep(0)
        owner.cancel()  # Simulate an explicit caller disconnect, not an elapsed-time limit.
        outcomes = await asyncio.gather(owner, waiter, return_exceptions=True)
        assert all(isinstance(item, asyncio.CancelledError) for item in outcomes)
        assert not MCPHandler._query_inflight
        assert not MCPHandler._query_cache

    asyncio.run(exercise())


def test_query_propagates_generator_failure_without_cache_or_unhandled_future():
    async def exercise():
        import gc

        events = []
        loop = asyncio.get_running_loop()
        old_handler = loop.get_exception_handler()
        loop.set_exception_handler(lambda loop, context: events.append(context))

        async def generate_answer(**kwargs):
            raise RuntimeError("fixture upstream failure")

        try:
            handler = make_handler(SimpleNamespace(generate_answer=generate_answer))
            with pytest.raises(RuntimeError, match="fixture upstream failure"):
                await handler._tool_memory_query({"question": "failure fixture"})
            gc.collect()
            await asyncio.sleep(0)
            assert not MCPHandler._query_cache
            assert not MCPHandler._query_inflight
            assert not events
        finally:
            loop.set_exception_handler(old_handler)

    asyncio.run(exercise())


def test_query_read_admission_failure_cleans_inflight_and_can_retry():
    async def exercise():
        async def generate_answer(**kwargs):
            return {"answer": "Recovered.", "confidence": "high"}

        handler = make_handler(SimpleNamespace(generate_answer=generate_answer))
        original_active = MCPHandler._active_read_ops
        MCPHandler._active_read_ops = handler.settings.memory_queue_max_inflight_reads
        try:
            with pytest.raises(RuntimeError, match="memory_query_overloaded"):
                await handler._tool_memory_query({"question": "admission fixture"})
            assert not MCPHandler._query_inflight
            MCPHandler._active_read_ops = 0
            result = await handler._tool_memory_query({"question": "admission fixture"})
            assert result["answer"] == "Recovered."
        finally:
            MCPHandler._active_read_ops = original_active
            MCPHandler._invalidate_memory_caches()

    asyncio.run(exercise())


def test_query_cancelled_during_admission_releases_waiter_without_exiting_gate():
    async def exercise():
        handler = make_handler(None)
        entered = asyncio.Event()
        exits = []

        async def enter():
            entered.set()
            await asyncio.Event().wait()

        async def exit_gate():
            exits.append(True)

        handler._enter_read_gate = enter
        handler._exit_read_gate = exit_gate
        owner = asyncio.create_task(handler._tool_memory_query({"question": "admission cancellation"}))
        await entered.wait()
        waiter = asyncio.create_task(handler._tool_memory_query({"question": "admission cancellation"}))
        await asyncio.sleep(0)
        owner.cancel()
        outcomes = await asyncio.gather(owner, waiter, return_exceptions=True)
        assert all(isinstance(item, asyncio.CancelledError) for item in outcomes)
        assert not MCPHandler._query_inflight
        assert not exits

    asyncio.run(exercise())


def test_query_surfaces_real_answer_generator_upstream_failure():
    async def exercise():
        class FailingClient:
            def __init__(self):
                self.calls = 0

            async def chat_completion(self, **kwargs):
                self.calls += 1
                raise RuntimeError("fixture upstream failure")

        client = FailingClient()
        handler = make_handler(AnswerGenerator(client))
        with pytest.raises(RuntimeError, match="answer generation failed"):
            await handler._tool_memory_query({"question": "real generator failure"})
        assert client.calls == 3
        assert not MCPHandler._query_cache and not MCPHandler._query_inflight

    asyncio.run(exercise())


def test_query_preserves_a_legitimate_answer_matching_error_text():
    async def exercise():
        async def generate_answer(**kwargs):
            return {
                "answer": "An error occurred while generating the answer.",
                "reasoning": "The source document quotes that sentence.",
                "confidence": "high",
            }

        handler = make_handler(SimpleNamespace(generate_answer=generate_answer))
        result = await handler._tool_memory_query({"question": "quoted error text"})
        assert result["answer"] == "An error occurred while generating the answer."
        assert result["confidence"] == "high"

    asyncio.run(exercise())


def test_query_uses_local_lancedb_hit_and_empty_controls(tmp_path):
    async def exercise():
        class FakeEmbedding:
            async def create_single_embedding(self, *args, **kwargs):
                return [1.0, 0.0]

        async def generate_answer(query, contexts):
            assert [item.entry_id for item in contexts] == ["fixture"]
            return {"answer": contexts[0].lossless_restatement, "confidence": "high"}

        store = MultiTenantVectorStore(str(tmp_path / "lancedb"), embedding_dimension=2)
        entry = MemoryEntry(entry_id="fixture", lossless_restatement="A local fact.")
        assert await store.add_entries("mem_hit", [entry], [[1.0, 0.0]]) == 1
        retriever = Retriever(FakeEmbedding(), store, "mem_hit", enable_planning=False, enable_reflection=False, enable_reranking=False)
        handler = make_handler(SimpleNamespace(generate_answer=generate_answer))
        handler._get_retriever = lambda: retriever
        hit = await handler._tool_memory_query({"question": "hit fixture"})
        assert hit["answer"] == "A local fact." and hit["contexts_used"] == 1

        empty_retriever = Retriever(FakeEmbedding(), store, "mem_empty", enable_planning=False, enable_reflection=False, enable_reranking=False)
        handler._get_retriever = lambda: empty_retriever
        handler._get_answer_generator = lambda: AnswerGenerator(FakeEmbedding())
        empty = await handler._tool_memory_query({"question": "empty fixture"})
        assert empty["contexts_used"] == 0 and empty["confidence"] == "low"
        assert empty["answer"] == "I don't have any relevant memories to answer this question."

    asyncio.run(exercise())


def test_user_defaults_preserve_naive_utc_and_legacy_roundtrip_without_warning():
    legacy = {
        "user_id": "legacy-fixture",
        "openrouter_api_key_encrypted": "",
        "table_name": "mem_legacy",
        "created_at": "2026-10-01T12:00:00",
        "last_active": "2026-10-01T12:01:00",
        "token_version": 0,
    }
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        created = User(user_id="new-fixture")
    assert created.created_at.tzinfo is None and created.last_active.tzinfo is None
    assert datetime.fromisoformat(created.to_dict()["created_at"]).tzinfo is None
    assert User.from_dict(legacy).to_dict() == legacy
