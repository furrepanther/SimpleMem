from __future__ import annotations

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from MCP.config.settings import Settings
from MCP.server.core.retriever import Retriever
from MCP.server.database.vector_store import MultiTenantVectorStore
from MCP.server.integrations.openrouter import OpenRouterClient


def test_settings_default_rerank_cap_is_registry_aligned(monkeypatch) -> None:
    monkeypatch.setenv("JWT_SECRET_KEY", "test-jwt-secret")
    monkeypatch.setenv("ENCRYPTION_KEY", "test-encryption-key")
    monkeypatch.setenv("OPENROUTER_RERANK_BASE_URL", "http://127.0.0.1:11502/v1")
    settings = Settings()

    assert settings.rerank_candidate_cap == 32
    assert settings.rerank_document_max_chars == 1200
    assert settings.openrouter_rerank_base_url == "http://127.0.0.1:11502/v1"


def test_retriever_defaults_match_settings_cap() -> None:
    retriever = Retriever(object(), object(), "memory")

    assert retriever.rerank_candidate_cap == 32
    assert retriever.rerank_document_max_chars == 1200


def test_vector_store_exposes_table_names_compatibility(tmp_path) -> None:
    store = MultiTenantVectorStore(str(tmp_path / "lancedb"), embedding_dimension=8)

    assert store.table_names() == []


def test_rerank_uses_dedicated_rerank_base_url(monkeypatch) -> None:
    client = OpenRouterClient(
        api_key="test-key",
        base_url="http://127.0.0.1:11437/v1",
        embedding_base_url="http://127.0.0.1:11501/v1",
        rerank_base_url="http://127.0.0.1:11502/v1",
        llm_model="granite",
        embedding_model="bge-m3",
        reranker_model="bge-reranker",
    )

    class _FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {"results": [{"index": 0, "score": 1.0}]}

    class _FakeClient:
        def __init__(self, base_url: str) -> None:
            self.base_url = base_url

        async def post(self, path: str, json: dict[str, object]) -> _FakeResponse:
            assert path == "/rerank"
            assert self.base_url == "http://127.0.0.1:11502/v1"
            return _FakeResponse()

    monkeypatch.setattr(client, "_get_rerank_client", lambda: _FakeClient(client.rerank_base_url))

    def _boom() -> None:
        raise AssertionError("embed client should not be used for rerank")

    monkeypatch.setattr(client, "_get_embed_client", _boom)

    ranked = asyncio.run(client.rerank("query", ["doc"]))

    assert ranked == [{"index": 0, "score": 1.0}]
