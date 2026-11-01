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


def test_actual_transport_failure_releases_read_permit_and_is_not_success_cached():
    import httpx
    from server.integrations.openrouter import OpenRouterClient
    async def exercise():
        requests = []
        def failed_read(request):
            requests.append(request)
            raise httpx.ReadTimeout("synthetic failed transport", request=request)
        client = OpenRouterClient(api_key="fixture", base_url="http://127.0.0.1:9/v1",
                                  llm_model="granite-4.0-h-1b-Q4_K_S:latest")
        configured = client._get_chat_client()
        timeout = configured.timeout
        assert timeout.read == 120.0
        await configured.aclose()
        client._chat_client = httpx.AsyncClient(base_url=client.base_url,
                                               transport=httpx.MockTransport(failed_read), timeout=timeout)
        handler = make_handler(AnswerGenerator(client))
        before = MCPHandler._active_read_ops
        try:
            with pytest.raises(httpx.ReadTimeout, match="synthetic failed transport"):
                await handler._tool_memory_query({"question": "transport failure fixture"})
            assert len(requests) == 3 and MCPHandler._active_read_ops == before
            assert not MCPHandler._query_inflight and not MCPHandler._query_cache
        finally:
            await client.close()
            MCPHandler._invalidate_memory_caches()
    asyncio.run(exercise())


def test_real_generator_explicit_cancellation_releases_owner_waiter_and_read_permit():
    async def exercise():
        started = asyncio.Event()
        class HeldClient(ParsedReplyClient):
            async def chat_completion(self, **kwargs):
                self.requests.append(kwargs)
                started.set()
                await asyncio.Event().wait()
        client = HeldClient(None)
        handler = make_handler(AnswerGenerator(client))
        before = MCPHandler._active_read_ops
        owner = asyncio.create_task(handler._tool_memory_query({"question": "explicit cancellation fixture"}))
        await started.wait()
        waiter = asyncio.create_task(handler._tool_memory_query({"question": "explicit cancellation fixture"}))
        await asyncio.sleep(0)
        owner.cancel()
        results = await asyncio.gather(owner, waiter, return_exceptions=True)
        assert all(isinstance(result, asyncio.CancelledError) for result in results)
        assert len(client.requests) == 1 and MCPHandler._active_read_ops == before
        assert not MCPHandler._query_inflight and not MCPHandler._query_cache
        MCPHandler._invalidate_memory_caches()
    asyncio.run(exercise())


def test_generation_failure_through_actual_jsonrpc_retains_id_and_same_query_recovers(deliver_ready_message):
    import json
    async def exercise():
        client = ParsedReplyClient(None)
        handler = make_handler(AnswerGenerator(client))
        handler.initialized = True
        request = {"jsonrpc": "2.0", "id": 47, "method": "tools/call",
                   "params": {"name": "memory_query", "arguments": {"question": "wire error fixture"}}}
        result = json.loads(await deliver_ready_message(handler, json.dumps(request)))
        assert result["id"] == 47 and result["error"]["code"] == -32603
        assert not MCPHandler._query_inflight and not MCPHandler._query_cache
        client.data = {"answer": "wire recovered", "confidence": "high"}
        request["id"] = 48
        recovered = json.loads(await deliver_ready_message(handler, json.dumps(request)))
        assert recovered["id"] == 48 and "error" not in recovered
        body = json.loads(recovered["result"]["content"][0]["text"])
        assert body["answer"] == "wire recovered"
        MCPHandler._invalidate_memory_caches()
    asyncio.run(exercise())
