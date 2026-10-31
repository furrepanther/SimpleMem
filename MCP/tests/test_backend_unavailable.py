import asyncio
import gc
import json
from types import SimpleNamespace

import httpx
import pytest

from server.auth.models import MemoryEntry
from server.mcp_handler import MCPHandler


def make_handler(client, local=None, enabled=True):
    MCPHandler._invalidate_memory_caches()
    handler = object.__new__(MCPHandler)
    handler.user = SimpleNamespace(user_id="global", table_name="mem_global")
    handler.settings = SimpleNamespace(
        memory_retrieve_backend_url="http://backend.invalid/search" if enabled else "",
        memory_retrieve_backend_global_only=True,
        memory_retrieve_max_query_chars=4096,
        memory_retrieve_max_top_k=10,
    )
    async def get_client(timeout_s):
        return client
    async def get_entry(*args):
        return local
    async def retrieve(**kwargs):
        return [local] if local else []
    handler._get_fastpath_client = get_client
    handler.vector_store = SimpleNamespace(get_entry=get_entry)
    handler._get_retriever = lambda: SimpleNamespace(retrieve=retrieve)
    return handler


def backend_response(case, request):
    if case == "unreachable":
        raise httpx.ConnectError("isolated unavailable", request=request)
    if case == "pending503":
        return httpx.Response(503, json={"detail": {"error": "semantic_refresh_pending", "retryable": True, "gated": True}})
    if case == "error503":
        return httpx.Response(503, json={"detail": "busy"})
    if case == "invalid_json":
        return httpx.Response(200, content=b"not json")
    if case == "not_object":
        return httpx.Response(200, json=[])
    if case == "missing_results":
        return httpx.Response(200, json={})
    if case == "pending200":
        return httpx.Response(200, json={"results": [], "search_mode": "semantic_refresh_pending", "gated": True})
    if case == "gated200":
        return httpx.Response(200, json={"results": [], "gated": True})
    if case == "degraded_empty200":
        return httpx.Response(200, json={"results": [], "status": "degraded"})
    if case == "degraded_partial200":
        return httpx.Response(200, json={"results": [{"entry_id": "partial", "content": "incomplete backend context"}], "status": "degraded"})
    return httpx.Response(200, json={"results": [], "total_available": 0})


@pytest.mark.parametrize("case", ["pending503", "error503", "unreachable", "invalid_json", "not_object", "missing_results", "pending200", "gated200", "degraded_empty200", "degraded_partial200"])
@pytest.mark.parametrize("force_exact", [False, True])
def test_backend_failure_is_retryable_tool_error_never_cached_as_empty(case, force_exact):
    async def run():
        calls = []
        response_case = [case]
        def respond(request):
            calls.append(request)
            return backend_response(response_case[0], request)
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            handler = make_handler(client)
            params = {"name": "memory_retrieve", "arguments": {"query": "owned-absent-id", "force_exact": force_exact}}
            for attempt in range(2):
                result = await handler._handle_tools_call(params)
                assert result["isError"] is True
                payload = json.loads(result["content"][0]["text"])
                expected = "semantic_refresh_pending" if case in {"pending503", "pending200"} else "memory_backend_unavailable"
                assert payload == {"error": expected, "retryable": True, "gated": True}
                assert not MCPHandler._retrieve_cache and not MCPHandler._retrieve_negative_cache
                assert not MCPHandler._retrieve_inflight
            assert len(calls) == 2
            response_case[0] = "healthy_empty"
            recovered = await handler._handle_tools_call(params)
            assert not recovered.get("isError")
            assert json.loads(recovered["content"][0]["text"])["total"] == 0
            assert MCPHandler._retrieve_cache and MCPHandler._retrieve_negative_cache
    asyncio.run(run())


def test_backend_outage_cannot_turn_local_semantic_partial_result_into_success():
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: backend_response("pending503", r))) as client:
            handler = make_handler(client, MemoryEntry(entry_id="local", lossless_restatement="local only fact"))
            result = await handler._handle_tools_call({"name": "memory_retrieve", "arguments": {"query": "local fact"}})
            assert result["isError"] and "local only fact" not in json.dumps(result)
            assert not MCPHandler._retrieve_cache and not MCPHandler._retrieve_negative_cache
    asyncio.run(run())


@pytest.mark.parametrize("enabled", [True, False])
def test_local_exact_hit_does_not_depend_on_backend(enabled):
    async def run():
        def forbidden(request):
            raise AssertionError("local exact hit must not call backend")
        async with httpx.AsyncClient(transport=httpx.MockTransport(forbidden)) as client:
            handler = make_handler(client, MemoryEntry(entry_id="local", lossless_restatement="durable local fact"), enabled)
            result = await handler._handle_tools_call({"name": "memory_retrieve", "arguments": {"query": "local", "force_exact": True}})
            assert not result.get("isError")
            assert json.loads(result["content"][0]["text"])["results"][0]["entry_id"] == "local"
    asyncio.run(run())


def test_explicitly_unconfigured_backend_preserves_local_semantic_retrieval():
    async def run():
        async with httpx.AsyncClient() as client:
            handler = make_handler(client, MemoryEntry(entry_id="local", lossless_restatement="local fact"), False)
            result = await handler._handle_tools_call({"name": "memory_retrieve", "arguments": {"query": "local fact"}})
            assert not result.get("isError")
            assert json.loads(result["content"][0]["text"])["total"] == 1
    asyncio.run(run())


def test_coalesced_callers_all_receive_pending_and_recover_without_empty_cache():
    async def run():
        entered = asyncio.Event()
        release = asyncio.Event()
        calls = []
        async def respond(request):
            calls.append(request)
            entered.set()
            await release.wait()
            return backend_response("pending503", request)
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            handler = make_handler(client)
            params = {"name": "memory_retrieve", "arguments": {"query": "same coalesced query"}}
            tasks = [asyncio.create_task(handler._handle_tools_call(params)) for _ in range(4)]
            await entered.wait()
            await asyncio.sleep(0)
            release.set()
            results = await asyncio.gather(*tasks)
            assert len(calls) == 1
            assert all(r.get("isError") and json.loads(r["content"][0]["text"])["retryable"] for r in results)
            assert not MCPHandler._retrieve_cache and not MCPHandler._retrieve_negative_cache
            assert not MCPHandler._retrieve_inflight
    asyncio.run(run())


def test_actual_jsonrpc_pending_and_recovery_leave_no_unhandled_owner_future():
    async def run():
        state = ["pending503"]
        unhandled = []
        loop = asyncio.get_running_loop()
        original = loop.get_exception_handler()
        loop.set_exception_handler(lambda loop, context: unhandled.append(context))
        try:
            async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: backend_response(state[0], r))) as client:
                handler = make_handler(client)
                handler.initialized = True
                request = {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "memory_retrieve", "arguments": {"query": "actual wire error and recovery"}}}
                pending = json.loads(await handler.handle_message(json.dumps(request)))
                assert pending["result"]["isError"]
                assert json.loads(pending["result"]["content"][0]["text"])["retryable"]
                state[0] = "healthy_empty"
                request["id"] = 2
                recovered = json.loads(await handler.handle_message(json.dumps(request)))
                assert not recovered["result"].get("isError")
                assert json.loads(recovered["result"]["content"][0]["text"])["total"] == 0
                gc.collect()
                await asyncio.sleep(0)
                assert not unhandled
        finally:
            loop.set_exception_handler(original)
    asyncio.run(run())
