"""Real generation errors must not become successful cached memory answers."""
import asyncio

import pytest

from server.core.answer_generator import AnswerGenerator
from server.mcp_handler import MCPHandler
from test_query_liveness_and_user_time import make_handler


class ParsedReplyClient:
    def __init__(self, data):
        self.data = data
        self.requests = []

    async def chat_completion(self, **kwargs):
        self.requests.append(kwargs)
        return "synthetic inference response"

    def extract_json(self, response):
        return self.data


@pytest.mark.parametrize("data", [None, {}, {"confidence": "high"}, {"answer": ""},
                                 {"answer": "  "}, {"answer": None}, {"answer": 5}, ["answer"]])
def test_invalid_real_generator_answers_fail_without_cache_and_same_query_recovers(data):
    async def exercise():
        client = ParsedReplyClient(data)
        handler = make_handler(AnswerGenerator(client))
        with pytest.raises(RuntimeError, match="invalid JSON answer"):
            await handler._tool_memory_query({"question": "answer contract fixture"})
        assert len(client.requests) == 3
        assert client.requests[0] == client.requests[1] == client.requests[2]
        assert not MCPHandler._query_cache and not MCPHandler._query_inflight
        client.data = {"answer": "Recovered fact.", "confidence": "high"}
        result = await handler._tool_memory_query({"question": "answer contract fixture"})
        assert result["answer"] == "Recovered fact." and len(client.requests) == 4
        assert len(MCPHandler._query_cache) == 1
        MCPHandler._invalidate_memory_caches()
    asyncio.run(exercise())


def test_valid_model_text_matching_the_old_error_sentinel_remains_a_real_answer():
    async def exercise():
        client = ParsedReplyClient({"answer": "An error occurred while generating the answer.",
                                    "reasoning": "Error: is part of this quoted historical fact.",
                                    "confidence": "high"})
        handler = make_handler(AnswerGenerator(client))
        result = await handler._tool_memory_query({"question": "quote historical fact"})
        assert result["answer"] == client.data["answer"]
        assert result["reasoning"] == client.data["reasoning"] and len(client.requests) == 1
        MCPHandler._invalidate_memory_caches()
    asyncio.run(exercise())


def test_invalid_generation_failure_reaches_coalesced_waiters_without_orphan_errors():
    async def exercise():
        started, release = asyncio.Event(), asyncio.Event()
        class HeldClient(ParsedReplyClient):
            async def chat_completion(self, **kwargs):
                started.set()
                await release.wait()
                return await super().chat_completion(**kwargs)
        client = HeldClient(None)
        handler = make_handler(AnswerGenerator(client))
        loop = asyncio.get_running_loop()
        errors = []
        previous = loop.get_exception_handler()
        loop.set_exception_handler(lambda loop, context: errors.append(context))
        try:
            owner = asyncio.create_task(handler._tool_memory_query({"question": "coalesced invalid fixture"}))
            await started.wait()
            waiter = asyncio.create_task(handler._tool_memory_query({"question": "coalesced invalid fixture"}))
            await asyncio.sleep(0)
            release.set()
            results = await asyncio.gather(owner, waiter, return_exceptions=True)
            assert all(isinstance(result, RuntimeError) for result in results)
            assert len(client.requests) == 3
            assert not MCPHandler._query_cache and not MCPHandler._query_inflight
            await asyncio.sleep(0)
            assert not errors
        finally:
            loop.set_exception_handler(previous)
            MCPHandler._invalidate_memory_caches()
    asyncio.run(exercise())
