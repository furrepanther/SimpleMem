from unittest.mock import patch

import pytest

from server.integrations.openrouter import OpenRouterClient, OpenRouterClientManager


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
