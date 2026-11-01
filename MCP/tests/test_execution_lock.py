import os
from pathlib import Path
import sqlite3
import subprocess
import sys

import pytest

from server.database.execution_lock import JobExecutionLocks


def _queue(tmp_path):
    path = tmp_path / "queue.sqlite"
    connection = sqlite3.connect(path, isolation_level=None)
    return path, connection


def test_same_job_excludes_independent_descriptions_and_other_jobs_coexist(tmp_path):
    path, first = _queue(tmp_path)
    second = sqlite3.connect(path, isolation_level=None)
    try:
        left = JobExecutionLocks(str(path), first)
        right = JobExecutionLocks(str(path), second)
        with left.try_job("job-a") as acquired:
            assert acquired
            with right.try_job("job-a") as competing:
                assert not competing
            with right.try_job("job-b") as other:
                assert other
        with right.try_job("job-a") as acquired:
            assert acquired
    finally:
        first.close()
        second.close()


def test_process_exit_releases_job_lock(tmp_path):
    path, connection = _queue(tmp_path)
    locks = JobExecutionLocks(str(path), connection)
    child = """
import os, sqlite3, sys
from server.database.execution_lock import JobExecutionLocks
connection = sqlite3.connect(sys.argv[1], isolation_level=None)
locks = JobExecutionLocks(sys.argv[1], connection)
with locks.try_job('job-a') as acquired:
    assert acquired
    print('locked', flush=True)
    sys.stdin.readline()
    os._exit(73)
"""
    proc = subprocess.Popen([sys.executable, "-c", child, str(path)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        assert proc.stdout.readline().strip() == "locked"
        with locks.try_job("job-a") as acquired:
            assert not acquired
        proc.stdin.write("release\n")
        proc.stdin.flush()
        proc.stdin.close()
        assert proc.wait() == 73
        with locks.try_job("job-a") as acquired:
            assert acquired
    finally:
        if proc.poll() is None:
            if proc.stdin and not proc.stdin.closed:
                proc.stdin.close()
            proc.wait()
        connection.close()


def test_replaced_lock_file_fails_closed_for_old_and_new_instances(tmp_path):
    path, connection = _queue(tmp_path)
    locks = JobExecutionLocks(str(path), connection)
    replacement = tmp_path / "replacement"
    try:
        with locks.try_job("job-a") as acquired:
            assert acquired
            os.rename(locks.path, replacement)
            locks.path.write_bytes(replacement.read_bytes())
            locks.path.chmod(0o600)
            with pytest.raises(RuntimeError, match="replaced"):
                with locks.try_job("job-a"):
                    pass
            with pytest.raises(RuntimeError, match="replaced"):
                JobExecutionLocks(str(path), connection)
    finally:
        connection.close()


def test_refuses_symlink_lock_file(tmp_path):
    path, connection = _queue(tmp_path)
    other = tmp_path / "other"
    other.write_text("fixture")
    Path(str(path) + ".execution-locks").symlink_to(other)
    try:
        with pytest.raises(OSError):
            JobExecutionLocks(str(path), connection)
    finally:
        connection.close()


def test_refuses_hardlinked_queue(tmp_path):
    path, connection = _queue(tmp_path)
    os.link(path, tmp_path / "alias.sqlite")
    try:
        with pytest.raises(RuntimeError, match="hard links"):
            JobExecutionLocks(str(path), connection)
    finally:
        connection.close()


def test_refuses_unverified_abi(tmp_path, monkeypatch):
    path, connection = _queue(tmp_path)
    monkeypatch.setattr("server.database.execution_lock.platform.machine", lambda: "unverified")
    try:
        with pytest.raises(RuntimeError, match="verified Linux"):
            JobExecutionLocks(str(path), connection)
    finally:
        connection.close()


def test_child_exec_does_not_inherit_job_lock(tmp_path):
    path, connection = _queue(tmp_path)
    locks = JobExecutionLocks(str(path), connection)
    child = "import sys; print('ready', flush=True); sys.stdin.readline()"
    proc = None
    try:
        with locks.try_job("job-a") as acquired:
            assert acquired
            proc = subprocess.Popen([sys.executable, "-c", child], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, close_fds=False)
            assert proc.stdout.readline().strip() == "ready"
        # If the subprocess inherited the lock-bearing description, this would fail.
        with locks.try_job("job-a") as acquired:
            assert acquired
    finally:
        if proc is not None:
            if proc.stdin and not proc.stdin.closed:
                proc.stdin.close()
            proc.wait()
        connection.close()


def test_recovery_does_not_reassign_or_terminalize_live_locked_job(tmp_path):
    from server.database.durable_job_store import DurableJobStore

    db = str(tmp_path / "queue.sqlite")
    owner = DurableJobStore(db)
    recovery = DurableJobStore(db)
    try:
        job_id = owner.enqueue_or_get(user_id="fixture", kind="memory_store", idempotency_key="k", payload={"content": "fixture"}, max_inflight_writes=10)["job_id"]
        assert owner.claim_batch(worker_id="owner", batch_size=1, lease_seconds=5)[0]["job_id"] == job_id
        owner._get_connection().execute("UPDATE durable_jobs SET lease_expires_at_utc = '2000-01-01T00:00:00+00:00' WHERE job_id = ?", (job_id,))
        with owner.execution_locks.try_job(job_id) as acquired:
            assert acquired
            assert recovery.recover_expired_leases(max_retries=1) == 0
            row = recovery.get_job(job_id=job_id)
            assert row["status"] == "processing"
            assert row["lease_owner"] == "owner"
            assert row["attempt_count"] == 0
            assert recovery.claim_batch(worker_id="other", batch_size=1, lease_seconds=5) == []
        assert recovery.recover_expired_leases(max_retries=2) == 1
        row = recovery.get_job(job_id=job_id)
        assert row["status"] == "failed_retryable" and row["attempt_count"] == 1
        assert recovery.claim_batch(worker_id="other", batch_size=1, lease_seconds=5)[0]["job_id"] == job_id
        assert not owner.mark_done(job_id=job_id, lease_owner="owner", result={"stale": True})
        assert recovery.mark_done(job_id=job_id, lease_owner="other", result={"ok": True})
    finally:
        owner.close()
        recovery.close()


def test_concurrent_first_initialization_pins_one_lock_identity(tmp_path):
    path, connection = _queue(tmp_path)
    connection.close()
    child = """
import json, sqlite3, sys
from server.database.execution_lock import JobExecutionLocks
connection = sqlite3.connect(sys.argv[1], timeout=10, isolation_level=None)
print('ready', flush=True)
sys.stdin.readline()
locks = JobExecutionLocks(sys.argv[1], connection)
print(json.dumps({'inode': locks.inode, 'token': locks.token}), flush=True)
connection.close()
"""
    procs = [subprocess.Popen([sys.executable, "-c", child, str(path)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) for _ in range(2)]
    try:
        for proc in procs:
            assert proc.stdout.readline().strip() == "ready"
        for proc in procs:
            proc.stdin.write("start\n")
            proc.stdin.flush()
            proc.stdin.close()
        outputs = [proc.stdout.readline().strip() for proc in procs]
        for proc in procs:
            assert proc.wait() == 0, proc.stderr.read()
        assert outputs[0] and outputs[0] == outputs[1]
    finally:
        for proc in procs:
            if proc.poll() is None:
                if proc.stdin and not proc.stdin.closed:
                    proc.stdin.close()
                proc.wait()


def test_refuses_shared_or_hardlinked_lock_file(tmp_path):
    path, connection = _queue(tmp_path)
    locks = JobExecutionLocks(str(path), connection)
    try:
        locks.path.chmod(0o644)
        with pytest.raises(RuntimeError, match="unsafe"):
            with locks.try_job("job-a"):
                pass
        locks.path.chmod(0o600)
        os.link(locks.path, tmp_path / "lock-alias")
        with pytest.raises(RuntimeError, match="unsafe"):
            with locks.try_job("job-a"):
                pass
    finally:
        connection.close()


def test_locked_batch_keeps_all_claimed_jobs_safe_until_context_exit(tmp_path):
    from server.database.durable_job_store import DurableJobStore

    db = str(tmp_path / "queue.sqlite")
    owner = DurableJobStore(db)
    recovery = DurableJobStore(db)
    try:
        ids = [owner.enqueue_or_get(user_id=user, kind="memory_store", idempotency_key=user, payload={"content": user}, max_inflight_writes=10)["job_id"] for user in ("a", "b")]
        with owner.claim_locked_batch(worker_id="owner", batch_size=2, lease_seconds=5) as jobs:
            assert {job["job_id"] for job in jobs} == set(ids)
            owner._get_connection().execute("UPDATE durable_jobs SET lease_expires_at_utc = '2000-01-01T00:00:00+00:00'")
            assert recovery.recover_expired_leases(max_retries=1) == 0
            assert all(recovery.get_job(job_id=job_id)["attempt_count"] == 0 for job_id in ids)
            with recovery.claim_locked_batch(worker_id="other", batch_size=2, lease_seconds=5) as other:
                assert other == []
            assert owner.mark_done(job_id=ids[0], lease_owner="owner", result={"ok": True})
            assert recovery.recover_expired_leases(max_retries=1) == 0
        assert recovery.recover_expired_leases(max_retries=2) == 1
        assert recovery.get_job(job_id=ids[0])["status"] == "done"
        assert recovery.get_job(job_id=ids[1])["status"] == "failed_retryable"
        with recovery.claim_locked_batch(worker_id="other", batch_size=2, lease_seconds=5) as claimed:
            assert [job["job_id"] for job in claimed] == [ids[1]]
            assert not owner.mark_done(job_id=ids[1], lease_owner="owner", result={"stale": True})
            assert recovery.mark_done(job_id=ids[1], lease_owner="other", result={"ok": True})
    finally:
        owner.close()
        recovery.close()


def test_locked_claim_rechecks_candidate_generation_after_lock(tmp_path, monkeypatch):
    from contextlib import contextmanager
    from server.database.durable_job_store import DurableJobStore

    db = str(tmp_path / "queue.sqlite")
    owner = DurableJobStore(db)
    concurrent = DurableJobStore(db)
    try:
        job_id = owner.enqueue_or_get(user_id="fixture", kind="memory_store", idempotency_key="k", payload={"content": "fixture"}, max_inflight_writes=10)["job_id"]
        original = owner.execution_locks.try_job
        @contextmanager
        def changed_before_claim(candidate):
            with original(candidate) as acquired:
                if acquired:
                    concurrent._get_connection().execute("UPDATE durable_jobs SET attempt_count = attempt_count + 1 WHERE job_id = ?", (candidate,))
                yield acquired
        monkeypatch.setattr(owner.execution_locks, "try_job", changed_before_claim)
        with owner.claim_locked_batch(worker_id="owner", batch_size=1, lease_seconds=5) as jobs:
            assert jobs == []
        row = concurrent.get_job(job_id=job_id)
        assert row["status"] == "pending" and row["attempt_count"] == 1
    finally:
        owner.close()
        concurrent.close()


@pytest.mark.parametrize("blocked_phase", ["open_table", "append"])
def test_cancelled_write_holds_job_lock_until_mutation_thread_exits(tmp_path, blocked_phase):
    import asyncio
    import threading
    from server.auth.models import MemoryEntry
    from server.database.vector_store import MultiTenantVectorStore

    async def scenario():
        path, connection = _queue(tmp_path)
        locks = JobExecutionLocks(str(path), connection)
        store = MultiTenantVectorStore.__new__(MultiTenantVectorStore)
        started = asyncio.Event()
        release = threading.Event()
        loop = asyncio.get_running_loop()
        def wait_in_thread():
            loop.call_soon_threadsafe(started.set)
            release.wait()
        def get_table(_name):
            if blocked_phase == "open_table":
                wait_in_thread()
            return object()
        def append(_name, _records):
            if blocked_phase == "append":
                wait_in_thread()
        store._get_table = get_table
        store._get_table_dimension = lambda *_args: 4
        store._append_records = append
        async def write():
            with locks.try_job("job-a") as acquired:
                assert acquired
                await store.add_entries("fixture", [MemoryEntry(entry_id="e", lossless_restatement="fixture")], [[0.1, 0.2, 0.3, 0.4]])
        task = asyncio.create_task(write())
        try:
            await asyncio.wait_for(started.wait(), 5)
            task.cancel()
            await asyncio.sleep(0)
            task.cancel()
            await asyncio.sleep(0.02)
            assert not task.done()
            with locks.try_job("job-a") as competing:
                assert not competing
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, 5)
            with locks.try_job("job-a") as acquired:
                assert acquired
        finally:
            release.set()
            if not task.done():
                await asyncio.gather(task, return_exceptions=True)
            connection.close()
    asyncio.run(scenario())


def test_cross_process_live_batch_is_not_recovered_until_owner_exits(tmp_path):
    from server.database.durable_job_store import DurableJobStore

    db = str(tmp_path / "queue.sqlite")
    recovery = DurableJobStore(db)
    job_id = recovery.enqueue_or_get(user_id="fixture", kind="memory_store", idempotency_key="key", payload={"content": "fixture"}, max_inflight_writes=10)["job_id"]
    child = """
import os, sys
from server.database.durable_job_store import DurableJobStore
store = DurableJobStore(sys.argv[1])
with store.claim_locked_batch(worker_id='child-owner', batch_size=1, lease_seconds=5) as jobs:
    assert len(jobs) == 1
    print(jobs[0]['job_id'], flush=True)
    sys.stdin.readline()
    os._exit(73)
"""
    proc = subprocess.Popen([sys.executable, "-c", child, db], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        assert proc.stdout.readline().strip() == job_id
        recovery._get_connection().execute("UPDATE durable_jobs SET lease_expires_at_utc = '2000-01-01T00:00:00+00:00' WHERE job_id = ?", (job_id,))
        assert recovery.recover_expired_leases(max_retries=1) == 0
        assert recovery.get_job(job_id=job_id)["status"] == "processing"
        proc.stdin.write("exit\n")
        proc.stdin.flush()
        proc.stdin.close()
        assert proc.wait() == 73, proc.stderr.read()
        assert recovery.recover_expired_leases(max_retries=2) == 1
        with recovery.claim_locked_batch(worker_id="new-owner", batch_size=1, lease_seconds=5) as jobs:
            assert [job["job_id"] for job in jobs] == [job_id]
            assert not recovery.mark_done(job_id=job_id, lease_owner="child-owner", result={"stale": True})
            assert recovery.mark_done(job_id=job_id, lease_owner="new-owner", result={"ok": True})
    finally:
        if proc.poll() is None:
            if proc.stdin and not proc.stdin.closed:
                proc.stdin.close()
            proc.wait()
        recovery.close()


def test_running_worker_retains_expired_lease_until_execution_finishes(tmp_path, monkeypatch):
    import asyncio
    from collections import deque
    from types import SimpleNamespace
    from server.database.durable_job_store import DurableJobStore
    from server.mcp_handler import MCPHandler

    async def scenario():
        store = DurableJobStore(str(tmp_path / "queue.sqlite"))
        job_id = store.enqueue_or_get(user_id="fixture", kind="memory_store", idempotency_key="key", payload={"content": "fixture"}, max_inflight_writes=10)["job_id"]
        settings = SimpleNamespace(memory_write_lease_seconds=90, memory_write_max_retries=1, memory_write_batch_max=1, memory_write_batch_max_wait_ms=10, memory_write_job_status_limit=100, memory_write_retry_backoff_base_ms=100)
        for name, value in (("_durable_store", store), ("_shared_settings", settings), ("_worker_shutdown", asyncio.Event()), ("_lease_recoveries", 0), ("_write_status_by_job_id", {}), ("_write_status_order", deque()), ("_write_stats", dict(MCPHandler._write_stats)), ("_write_stats_by_user", {})):
            monkeypatch.setattr(MCPHandler, name, value)
        monkeypatch.setattr(MCPHandler, "_invalidate_memory_caches", classmethod(lambda cls: None))
        started, release = asyncio.Event(), asyncio.Event()
        executions = []
        async def execute_fixture(cls, job):
            executions.append(job["job_id"])
            started.set()
            await release.wait()
            return {"fixture": True}
        monkeypatch.setattr(MCPHandler, "_execute_durable_job", classmethod(execute_fixture))
        worker = asyncio.create_task(MCPHandler._durable_worker_loop(1))
        try:
            await asyncio.wait_for(started.wait(), 5)
            store._get_connection().execute("UPDATE durable_jobs SET lease_expires_at_utc = '2000-01-01T00:00:00+00:00' WHERE job_id = ?", (job_id,))
            assert store.recover_expired_leases(max_retries=1) == 0
            row = store.get_job(job_id=job_id)
            assert row["status"] == "processing" and row["attempt_count"] == 0
            release.set()
            async def completed():
                while store.get_job(job_id=job_id)["status"] != "done":
                    await asyncio.sleep(0.01)
            await asyncio.wait_for(completed(), 5)
            assert executions == [job_id]
        finally:
            MCPHandler._worker_shutdown.set()
            release.set()
            worker.cancel()
            await asyncio.gather(worker, return_exceptions=True)
            store.close()
    asyncio.run(scenario())


def test_directory_fsync_failure_rolls_back_lock_identity(tmp_path, monkeypatch):
    import stat
    path, connection = _queue(tmp_path)
    actual_fsync = os.fsync
    def failing_directory_fsync(fd):
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            raise OSError("fixture directory fsync failure")
        return actual_fsync(fd)
    try:
        with monkeypatch.context() as scoped:
            scoped.setattr("server.database.execution_lock.os.fsync", failing_directory_fsync)
            with pytest.raises(OSError, match="directory fsync failure"):
                JobExecutionLocks(str(path), connection)
        assert connection.execute("SELECT COUNT(*) FROM durable_execution_lock_identity").fetchone()[0] == 0
        locks = JobExecutionLocks(str(path), connection)
        with locks.try_job("fixture") as acquired:
            assert acquired
    finally:
        connection.close()
