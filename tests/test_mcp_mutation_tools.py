from __future__ import annotations

import ast
from pathlib import Path


SOURCE = Path(__file__).resolve().parents[1] / "MCP/server/mcp_handler.py"


def _source_text() -> str:
    return SOURCE.read_text(encoding="utf-8")


def _tool_schemas() -> dict[str, dict]:
    tree = ast.parse(_source_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "_handle_tools_list":
            for child in ast.walk(node):
                if isinstance(child, ast.Return):
                    payload = ast.literal_eval(child.value)
                    return {
                        tool["name"]: tool["inputSchema"]
                        for tool in payload["tools"]
                    }
    raise AssertionError("_handle_tools_list return payload not found")


def test_targeted_mutation_tools_replace_clear_tool() -> None:
    schemas = _tool_schemas()

    assert "memory_clear" not in schemas
    assert "memory_delete" in schemas
    assert "memory_update" in schemas
    assert "memory_clear" not in _source_text()


def test_mutation_tools_accept_entry_id_or_memory_id() -> None:
    schemas = _tool_schemas()

    delete_schema = schemas["memory_delete"]
    update_schema = schemas["memory_update"]

    assert delete_schema["required"] == []
    assert update_schema["required"] == []
    assert {"entry_id", "memory_id"} <= set(delete_schema["properties"])
    assert {"entry_id", "memory_id", "content", "text"} <= set(
        update_schema["properties"]
    )


def test_cache_invalidation_uses_generation_guard() -> None:
    text = _source_text()

    assert "_memory_cache_generation: int = 0" in text
    assert "cls._memory_cache_generation += 1" in text
    assert "cache_generation = MCPHandler._memory_cache_generation" in text
    assert (
        "if cache_generation == MCPHandler._memory_cache_generation:" in text
    )
    assert 'result.get("queued")' in text


def test_backend_mutation_routing_handles_non_local_fastpath_ids() -> None:
    text = _source_text()

    assert "route_backend = self._uses_memory_backend_id(entry_id)" in text
    assert "await self.vector_store.get_entry(self.user.table_name, entry_id)" in text
    assert "if route_backend:" in text
