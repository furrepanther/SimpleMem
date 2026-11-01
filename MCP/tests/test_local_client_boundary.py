from unittest.mock import patch
import os
from pathlib import Path
import subprocess
import sys

import pytest

from config.settings import _is_local_endpoint
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


def test_client_import_does_not_import_settings():
    mcp_root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, PYTHONPATH=str(mcp_root), PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run(
        [sys.executable, "-c", "import sys; import server.integrations.openrouter; assert 'config.settings' not in sys.modules"],
        cwd=mcp_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("endpoint", [
    "http://127.0.0.1:11437/v1", "https://localhost/v1", "http://[::1]:8000/v1",
    "http://example.com/v1", "ftp://localhost/v1", "", "http://localhost@example.com/v1",
])
def test_client_loopback_rule_matches_settings(endpoint):
    if _is_local_endpoint(endpoint):
        assert OpenRouterClientManager(endpoint).base_url == endpoint
    else:
        with pytest.raises(ValueError):
            OpenRouterClientManager(endpoint)
