"""Exercise deployed auth endpoint functions with fake keys and disposable SQLite."""

import ast
import asyncio
import json
import secrets
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import pytest
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.testclient import TestClient
from pydantic import BaseModel
from server.auth.models import User
from server.auth.token_manager import TokenManager
from server.database.user_store import UserStore

SOURCE = Path(__file__).resolve().parents[1] / "server/http_server.py"
NAMES = {
    "RotateApiKeyRequest",
    "verify_bearer_token",
    "_invalidate_user_sessions",
    "_read_request_body_limited",
    "verify_token",
    "refresh_token",
    "rotate_api_key",
    "revoke_tokens",
}


@pytest.fixture
def auth_app(tmp_path):
    store = UserStore(str(tmp_path / "users.sqlite3"))
    manager = TokenManager(secrets.token_urlsafe(48), secrets.token_urlsafe(48))
    user = User(
        user_id="global",
        openrouter_api_key_encrypted=manager.encrypt_api_key("synthetic-local-key"),
    )
    store.create_user(user)

    class LocalOnlyClient:
        def __init__(self, **kwargs):
            self.key = kwargs["api_key"]

        async def verify_api_key(self):
            return (self.key != "synthetic-invalid", "synthetic rejection")

        async def close(self):
            pass

    settings = SimpleNamespace(
        openrouter_base_url="http://127.0.0.1:9/v1",
        openrouter_embedding_base_url="http://127.0.0.1:9/v1",
        openrouter_rerank_base_url="http://127.0.0.1:9/v1",
        llm_model="synthetic",
        embedding_model="synthetic",
        reranker_model="synthetic",
    )
    sessions = {
        "old-global": SimpleNamespace(user_id="global"),
        "other": SimpleNamespace(user_id="other"),
    }
    handlers = {"global": object(), "other": object()}
    scope = {
        "Optional": Optional,
        "User": User,
        "BaseModel": BaseModel,
        "Header": Header,
        "Request": Request,
        "HTTPException": HTTPException,
        "asyncio": asyncio,
        "json": json,
        "user_store": store,
        "token_manager": manager,
        "OpenRouterClient": LocalOnlyClient,
        "settings": settings,
        "_sessions": sessions,
        "_mcp_handlers": handlers,
        "_session_lock": asyncio.Lock(),
    }
    tree = ast.parse(SOURCE.read_text())
    nodes = [
        n
        for n in tree.body
        if isinstance(n, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        and n.name in NAMES
    ]
    assert {n.name for n in nodes} == NAMES
    for node in nodes:
        node.decorator_list = []
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
            str(SOURCE),
            "exec",
        ),
        scope,
    )
    app = FastAPI()
    for route, func, methods in [
        ("/api/auth/rotate", "rotate_api_key", ["POST"]),
        ("/api/auth/verify", "verify_token", ["GET"]),
        ("/api/auth/refresh", "refresh_token", ["POST"]),
        ("/api/auth/revoke", "revoke_tokens", ["POST"]),
    ]:
        app.add_api_route(route, scope[func], methods=methods)
    with TestClient(app, raise_server_exceptions=False) as client:
        yield SimpleNamespace(
            client=client,
            store=store,
            manager=manager,
            user=user,
            sessions=sessions,
            handlers=handlers,
            scope=scope,
        )


def auth(token):
    return {"Authorization": "Bearer " + token}


def test_real_rotation_invalidates_old_tokens_and_sessions_and_issues_new(auth_app):
    f = auth_app
    old = f.manager.generate_token(f.user)
    pre = f.client.post("/api/auth/refresh", headers=auth(old))
    assert pre.status_code == 200
    refreshed = pre.json()["token"]
    response = f.client.post(
        "/api/auth/rotate",
        headers=auth(old),
        json={"openrouter_api_key": "synthetic-local-key"},
    )
    assert response.status_code == 200
    new = response.json()["token"]
    assert f.client.get("/api/auth/verify", headers=auth(old)).status_code == 401
    assert f.client.get("/api/auth/verify", headers=auth(refreshed)).status_code == 401
    assert f.client.get("/api/auth/verify", headers=auth(new)).status_code == 200
    assert f.store.get_user("global").token_version == 1
    assert (
        f.manager.decrypt_api_key(
            f.store.get_user("global").openrouter_api_key_encrypted
        )
        == "synthetic-local-key"
    )
    assert set(f.sessions) == {"other"} and set(f.handlers) == {"other"}


def test_failed_provider_validation_does_not_revoke_or_update(auth_app):
    f = auth_app
    old = f.manager.generate_token(f.user)
    response = f.client.post(
        "/api/auth/rotate",
        headers=auth(old),
        json={"openrouter_api_key": "synthetic-invalid"},
    )
    assert response.status_code == 400
    assert f.client.get("/api/auth/verify", headers=auth(old)).status_code == 200
    assert f.store.get_user("global").token_version == 0
    assert set(f.sessions) == {"old-global", "other"}


def test_lost_response_after_commit_requires_privileged_out_of_band_mint(
    auth_app, monkeypatch
):
    f = auth_app
    old = f.manager.generate_token(f.user)
    original = f.manager.generate_token
    with monkeypatch.context() as patch:

        def fail_after_commit(_user):
            raise RuntimeError("synthetic lost response after commit")

        patch.setattr(f.manager, "generate_token", fail_after_commit)
        response = f.client.post(
            "/api/auth/rotate",
            headers=auth(old),
            json={"openrouter_api_key": "synthetic-local-key"},
        )
    assert response.status_code == 500
    assert f.store.get_user("global").token_version == 1
    assert f.client.get("/api/auth/verify", headers=auth(old)).status_code == 401
    assert f.client.post("/api/auth/refresh", headers=auth(old)).status_code == 401
    assert (
        f.client.post(
            "/api/auth/rotate",
            headers=auth(old),
            json={"openrouter_api_key": "synthetic-local-key"},
        ).status_code
        == 401
    )
    recovered = original(f.store.get_user("global"))
    assert f.client.get("/api/auth/verify", headers=auth(recovered)).status_code == 200


def test_concurrent_rotations_preserve_first_replacement(auth_app, monkeypatch):
    import concurrent.futures
    import threading

    f = auth_app
    old = f.manager.generate_token(f.user)
    both_validated = threading.Barrier(2)
    release_second = threading.Event()

    class RacingClient:
        def __init__(self, **kwargs):
            self.key = kwargs["api_key"]

        async def verify_api_key(self):
            await asyncio.to_thread(both_validated.wait, 5)
            if self.key == "synthetic-second":
                released = await asyncio.to_thread(release_second.wait, 5)
                if not released:
                    raise RuntimeError("fixture release missing")
            return True, ""

        async def close(self):
            pass

    monkeypatch.setitem(f.scope, "OpenRouterClient", RacingClient)

    def rotate(key):
        return f.client.post(
            "/api/auth/rotate", headers=auth(old), json={"openrouter_api_key": key}
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(rotate, "synthetic-first")
        second = pool.submit(rotate, "synthetic-second")
        try:
            first_response = first.result(timeout=10)
            assert first_response.status_code == 200
            replacement = first_response.json()["token"]
            assert (
                f.client.get("/api/auth/verify", headers=auth(replacement)).status_code
                == 200
            )
        finally:
            release_second.set()
        second_response = second.result(timeout=10)
    assert second_response.status_code == 409
    assert (
        f.client.get("/api/auth/verify", headers=auth(replacement)).status_code == 200
    )
    assert f.store.get_user("global").token_version == 1


def test_revoke_wins_against_already_authorized_rotation(auth_app, monkeypatch):
    import concurrent.futures
    import threading

    f = auth_app
    old = f.manager.generate_token(f.user)
    authorized = threading.Event()
    release = threading.Event()

    class PausedClient:
        def __init__(self, **kwargs):
            pass

        async def verify_api_key(self):
            authorized.set()
            if not await asyncio.to_thread(release.wait, 5):
                raise RuntimeError("fixture release missing")
            return True, ""

        async def close(self):
            pass

    monkeypatch.setitem(f.scope, "OpenRouterClient", PausedClient)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(
            f.client.post,
            "/api/auth/rotate",
            headers=auth(old),
            json={"openrouter_api_key": "synthetic-new"},
        )
        try:
            assert authorized.wait(5)
            revoke = f.client.post("/api/auth/revoke", headers=auth(old))
            assert revoke.status_code == 200
        finally:
            release.set()
        rotation = pending.result(timeout=10)
    assert rotation.status_code == 409
    assert f.store.get_user("global").token_version == 1
    assert (
        f.manager.decrypt_api_key(
            f.store.get_user("global").openrouter_api_key_encrypted
        )
        == "synthetic-local-key"
    )
    assert f.client.get("/api/auth/verify", headers=auth(old)).status_code == 401


def test_database_failure_rolls_back_and_releases_write_lock(auth_app):
    import sqlite3

    f = auth_app
    old = f.manager.generate_token(f.user)
    with sqlite3.connect(f.store.db_path) as c:
        c.execute(
            "CREATE TRIGGER synthetic_reject_update BEFORE UPDATE ON users BEGIN SELECT RAISE(ABORT, 'synthetic failure'); END"
        )
    response = f.client.post(
        "/api/auth/rotate",
        headers=auth(old),
        json={"openrouter_api_key": "synthetic-local-key"},
    )
    assert response.status_code == 500
    assert f.client.get("/api/auth/verify", headers=auth(old)).status_code == 200
    with sqlite3.connect(f.store.db_path, timeout=1) as c:
        c.execute("BEGIN IMMEDIATE")
        c.execute("DROP TRIGGER synthetic_reject_update")
    assert f.store.get_user("global").token_version == 0


def test_stale_store_version_cannot_revoke_or_rotate(auth_app):
    f = auth_app
    assert f.store.revoke_tokens("global", expected_token_version=0) is True
    assert f.store.revoke_tokens("global", expected_token_version=0) is False
    assert (
        f.store.rotate_api_key(
            "global", "synthetic-ciphertext", expected_token_version=0
        )
        is False
    )
    assert f.store.get_user("global").token_version == 1


def test_concurrent_revocations_have_exactly_one_winner(auth_app):
    import concurrent.futures
    import threading

    f = auth_app
    barrier = threading.Barrier(2)

    def revoke():
        barrier.wait(5)
        return f.store.revoke_tokens("global", expected_token_version=0)

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: revoke(), range(2)))
    assert sorted(results) == [False, True]
    assert f.store.get_user("global").token_version == 1


def test_rotation_never_mints_for_a_later_version(auth_app, monkeypatch):
    f = auth_app
    old = f.manager.generate_token(f.user)
    original = f.store.get_user

    def change_after_rotation(user_id):
        current = original(user_id)
        if current is not None and current.token_version == 1:
            assert f.store.revoke_tokens(user_id, expected_token_version=1)
            return original(user_id)
        return current

    monkeypatch.setattr(f.store, "get_user", change_after_rotation)
    response = f.client.post(
        "/api/auth/rotate",
        headers=auth(old),
        json={"openrouter_api_key": "synthetic-local-key"},
    )
    assert response.status_code == 409
    assert "token" not in response.json()
    assert f.store.get_user("global").token_version == 2
    assert set(f.sessions) == {"other"} and set(f.handlers) == {"other"}


def test_missing_user_during_rotation_reports_not_found(auth_app, monkeypatch):
    f = auth_app
    old = f.manager.generate_token(f.user)
    original = f.store.rotate_api_key

    def delete_before_rotation(user_id, encrypted_api_key, *, expected_token_version):
        assert f.store.delete_user(user_id)
        return original(
            user_id, encrypted_api_key, expected_token_version=expected_token_version
        )

    monkeypatch.setattr(f.store, "rotate_api_key", delete_before_rotation)
    response = f.client.post(
        "/api/auth/rotate",
        headers=auth(old),
        json={"openrouter_api_key": "synthetic-local-key"},
    )
    assert response.status_code == 404
    assert "token" not in response.json()


def test_rotation_wins_against_already_authorized_revoke(auth_app, monkeypatch):
    import concurrent.futures
    import threading

    f = auth_app
    old = f.manager.generate_token(f.user)
    authorized = threading.Event()
    release = threading.Event()
    original_revoke = f.store.revoke_tokens

    def paused_revoke(user_id, *, expected_token_version):
        authorized.set()
        if not release.wait(5):
            raise RuntimeError("fixture release missing")
        return original_revoke(user_id, expected_token_version=expected_token_version)

    monkeypatch.setattr(f.store, "revoke_tokens", paused_revoke)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(f.client.post, "/api/auth/revoke", headers=auth(old))
        try:
            assert authorized.wait(5)
            rotation = f.client.post(
                "/api/auth/rotate",
                headers=auth(old),
                json={"openrouter_api_key": "synthetic-new"},
            )
            assert rotation.status_code == 200
        finally:
            release.set()
        revoked = pending.result(timeout=10)
    assert revoked.status_code == 409
    assert (
        f.client.get(
            "/api/auth/verify", headers=auth(rotation.json()["token"])
        ).status_code
        == 200
    )
    assert f.store.get_user("global").token_version == 1
