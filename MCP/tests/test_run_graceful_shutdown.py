from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path


RUNNER = Path(__file__).resolve().parents[1] / "run.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location("simplemem_run_for_test", RUNNER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_explicit_shutdown_has_a_bounded_maintenance_drain(monkeypatch) -> None:
    """A deliberate quiesce cannot wait indefinitely on an MCP stream."""

    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    monkeypatch.setitem(
        sys.modules,
        "uvicorn",
        types.SimpleNamespace(run=lambda *args, **kwargs: calls.append((args, kwargs))),
    )
    monkeypatch.setattr(sys, "argv", ["run.py", "--host", "127.0.0.1", "--port", "8123"])

    runner = _load_runner()
    runner.main()

    assert calls == [
        (
            ("server.http_server:app",),
            {
                "host": "127.0.0.1",
                "port": 8123,
                "reload": False,
                "timeout_graceful_shutdown": runner.MAINTENANCE_GRACEFUL_SHUTDOWN_S,
            },
        )
    ]
    assert runner.MAINTENANCE_GRACEFUL_SHUTDOWN_S == 15
