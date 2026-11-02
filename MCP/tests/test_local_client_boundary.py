from unittest.mock import patch
from pathlib import Path

import pytest

from config.settings import Settings
from server.integrations.openrouter import OpenRouterClient, OpenRouterClientManager


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("factory,kwargs", [
    (OpenRouterClient, {"api_key": "fixture"}),
    (OpenRouterClientManager, {}),
])
def test_base_endpoint_is_required_before_client_creation(factory, kwargs):
    with patch("httpx.AsyncClient") as http_client:
        with pytest.raises(TypeError):
            factory(**kwargs)
        http_client.assert_not_called()


@pytest.mark.parametrize("factory,kwargs", [
    (OpenRouterClient, {"api_key": "fixture"}),
    (OpenRouterClientManager, {}),
])
@pytest.mark.parametrize("name,value", [
    ("base_url", "https://inference.invalid/v1"),
    ("base_url", ""),
    ("base_url", "http://localhost.inference.invalid/v1"),
    ("base_url", "http://127.0.0.1.inference.invalid/v1"),
    ("base_url", "http://localhost@inference.invalid/v1"),
    ("embedding_base_url", "https://inference.invalid/v1"),
    ("embedding_base_url", ""),
    ("rerank_base_url", "https://inference.invalid/v1"),
    ("rerank_base_url", ""),
])
def test_nonlocal_or_empty_explicit_endpoint_is_rejected(factory, kwargs, name, value):
    endpoints = {"base_url": "http://127.0.0.1:11437/v1", name: value}
    with patch("httpx.AsyncClient") as http_client:
        with pytest.raises(ValueError, match=name):
            factory(**kwargs, **endpoints)
        http_client.assert_not_called()


def test_explicit_local_endpoints_and_fallback_are_preserved():
    base = "http://127.0.0.1:11437/v1"
    embed = "http://localhost:8002/v1"
    rerank = "http://[::1]:8003/v1"
    manager = OpenRouterClientManager(base, embed, rerank)
    client = manager.get_client("fixture")
    assert (client.base_url, client.embedding_base_url, client.rerank_base_url) == (base, embed, rerank)

    fallback = OpenRouterClient("fixture", base, embed)
    assert (fallback.base_url, fallback.embedding_base_url, fallback.rerank_base_url) == (base, embed, embed)
    all_fallback = OpenRouterClientManager(base).get_client("fixture")
    assert (all_fallback.base_url, all_fallback.embedding_base_url, all_fallback.rerank_base_url) == (base, base, base)


def test_unset_llm_model_falls_back_to_canonical_granite(monkeypatch, tmp_path):
    monkeypatch.delenv("LLM_MODEL", raising=False)
    base = "http://127.0.0.1:11437/v1"
    embed = "http://127.0.0.1:11501/v1"
    rerank = "http://127.0.0.1:11502/v1"
    settings = Settings(
        jwt_secret_key="j" * 64,
        encryption_key="e" * 64,
        data_dir=str(tmp_path / "data"),
        lancedb_path=str(tmp_path / "lancedb"),
        durable_queue_db_path=str(tmp_path / "queue.db"),
        openrouter_base_url=base,
        openrouter_embedding_base_url=embed,
        openrouter_rerank_base_url=rerank,
    )
    assert settings.llm_model == "granite-4.0-h-1b-Q4_K_S:latest"
    assert OpenRouterClient("fixture", base).llm_model == settings.llm_model
    assert OpenRouterClientManager(base).llm_model == settings.llm_model


def test_registration_ui_never_requests_a_cloud_provider_key():
    index = (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
    app = (ROOT / "frontend" / "app.js").read_text(encoding="utf-8")

    assert "Local Service Credential" in index
    assert "Do not enter a cloud-provider key." in index
    assert "Register Another Credential" in index
    assert "OpenRouter API key" not in index
    assert "OpenRouter API key" not in app
    assert "local service credential" in app
    assert "openrouter.ai" not in index
    assert "openrouter.ai" not in app
