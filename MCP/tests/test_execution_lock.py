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


def _claim_abandoned_fixture(store, **kwargs):
    """Simulate claims whose owner has already lost its execution locks."""
    with store.claim_locked_batch(**kwargs) as jobs:
        return jobs


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
        assert _claim_abandoned_fixture(owner, worker_id="owner", batch_size=1, lease_seconds=5)[0]["job_id"] == job_id
        owner._get_connection().execute("UPDATE durable_jobs SET lease_expires_at_utc = '2000-01-01T00:00:00+00:00' WHERE job_id = ?", (job_id,))
        with owner.execution_locks.try_job(job_id) as acquired:
            assert acquired
            assert recovery.recover_expired_leases(max_retries=1) == 0
            row = recovery.get_job(job_id=job_id)
            assert row["status"] == "processing"
            assert row["lease_owner"] == "owner"
            assert row["attempt_count"] == 0
            assert _claim_abandoned_fixture(recovery, worker_id="other", batch_size=1, lease_seconds=5) == []
        assert recovery.recover_expired_leases(max_retries=2) == 1
        row = recovery.get_job(job_id=job_id)
        assert row["status"] == "failed_retryable" and row["attempt_count"] == 1
        assert _claim_abandoned_fixture(recovery, worker_id="other", batch_size=1, lease_seconds=5)[0]["job_id"] == job_id
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
                await task
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


def test_checkpoint_is_first_write_wins_owner_fenced_and_deleted_with_job(tmp_path):
    from server.database.durable_job_store import DurableJobStore

    store = DurableJobStore(str(tmp_path / "queue.sqlite"))
    try:
        job_id = store.enqueue_or_get(user_id="fixture", kind="memory_add", idempotency_key="key", payload={"content": "fixture", "table_name": "fixture"}, max_inflight_writes=10)["job_id"]
        with store.claim_locked_batch(worker_id="owner", batch_size=1, lease_seconds=5) as jobs:
            assert len(jobs) == 1
            assert store.load_checkpoint(job_id=job_id, lease_owner="owner", window_index=0, input_hash="fixture-hash") is None
            assert store.save_checkpoint(job_id=job_id, lease_owner="owner", window_index=0, input_hash="fixture-hash", entries=[{"id": "first"}]) == [{"id": "first"}]
            assert store.save_checkpoint(job_id=job_id, lease_owner="owner", window_index=0, input_hash="fixture-hash", entries=[{"id": "different"}]) == [{"id": "first"}]
            assert store.load_checkpoint(job_id=job_id, lease_owner="owner", window_index=0, input_hash="fixture-hash") == [{"id": "first"}]
            with pytest.raises(RuntimeError, match="checkpoint_input_mismatch"):
                store.load_checkpoint(job_id=job_id, lease_owner="owner", window_index=0, input_hash="different")
            with pytest.raises(RuntimeError, match="checkpoint_input_mismatch"):
                store.save_checkpoint(job_id=job_id, lease_owner="owner", window_index=0, input_hash="different", entries=[])
            with pytest.raises(RuntimeError, match="lease_lost"):
                store.save_checkpoint(job_id=job_id, lease_owner="other", window_index=1, input_hash="fixture-hash", entries=[])
            with pytest.raises(RuntimeError, match="lease_lost"):
                store.load_checkpoint(job_id=job_id, lease_owner="other", window_index=0, input_hash="fixture-hash")
            assert store.mark_done(job_id=job_id, lease_owner="owner", result={"ok": True})
        store._get_connection().execute("UPDATE durable_jobs SET updated_at_utc = '2000-01-01T00:00:00+00:00' WHERE job_id = ?", (job_id,))
        assert store.purge_completed_jobs(max_age_hours=24) == 1
        assert store._get_connection().execute("SELECT COUNT(*) FROM durable_job_checkpoints WHERE job_id = ?", (job_id,)).fetchone()[0] == 0
    finally:
        store.close()


def test_generated_window_replays_after_process_crash_without_duplicate_lance_rows(tmp_path):
    import asyncio
    from server.core.memory_builder import MemoryBuilder
    from server.database.durable_job_store import DurableJobStore
    from server.database.vector_store import MultiTenantVectorStore

    db = str(tmp_path / "queue.sqlite")
    lance = str(tmp_path / "lance")
    store = DurableJobStore(db)
    job_id = store.enqueue_or_get(user_id="fixture", kind="memory_add_batch", idempotency_key="key", payload={"content": "fixture", "table_name": "fixture"}, max_inflight_writes=10)["job_id"]
    child = """
import asyncio, os, sys
from server.core.memory_builder import MemoryBuilder
from server.database.durable_job_store import DurableJobStore
from server.database.vector_store import MultiTenantVectorStore
class Client:
    def __init__(self): self.calls = 0
    async def chat_completion(self, **kwargs):
        self.calls += 1
        if self.calls == 2: os._exit(73)
        return 'first durable fact'
    def extract_json(self, response): return {'entries': [{'lossless_restatement': response}]}
    async def create_embedding(self, texts): return [[0.1, 0.2, 0.3, 0.4] for _ in texts]
async def run():
    store = DurableJobStore(sys.argv[1])
    vector = MultiTenantVectorStore(db_path=sys.argv[2], embedding_dimension=4)
    with store.claim_locked_batch(worker_id='old-owner', batch_size=1, lease_seconds=5) as jobs:
        job = jobs[0]
        builder = MemoryBuilder(llm_client=Client(), vector_store=vector, table_name='fixture', window_size=1, entry_id_namespace='simplemem:fixture:'+job['job_id'], checkpoint_loader=lambda index, input_hash: store.load_checkpoint(job_id=job['job_id'], lease_owner='old-owner', window_index=index, input_hash=input_hash), checkpoint_saver=lambda index, input_hash, entries: store.save_checkpoint(job_id=job['job_id'], lease_owner='old-owner', window_index=index, input_hash=input_hash, entries=entries), checkpoint_register=lambda ids: store.record_job_entries(job_id=job['job_id'], lease_owner='old-owner', table_name='fixture', entry_ids=ids))
        await builder.add_dialogues([{'speaker': 'u', 'content': 'first', 'timestamp': '2026-01-01T00:00:00Z'}, {'speaker': 'u', 'content': 'second', 'timestamp': '2026-01-01T00:00:00Z'}])
asyncio.run(run())
raise SystemExit(99)
"""
    try:
        proc = subprocess.run([sys.executable, "-c", child, db, lance], capture_output=True, text=True)
        assert proc.returncode == 73, proc.stderr
        assert store._get_connection().execute("SELECT COUNT(*) FROM durable_job_checkpoints WHERE job_id = ?", (job_id,)).fetchone()[0] == 1
        store._get_connection().execute("UPDATE durable_jobs SET lease_expires_at_utc = '2000-01-01T00:00:00+00:00' WHERE job_id = ?", (job_id,))
        assert store.recover_expired_leases() == 1
        class Client:
            calls = 0
            async def chat_completion(self, **kwargs):
                self.calls += 1
                return "second durable fact"
            def extract_json(self, response):
                return {"entries": [{"lossless_restatement": response}]}
            async def create_embedding(self, texts):
                return [[0.1, 0.2, 0.3, 0.4] for _ in texts]
        client = Client()
        vector = MultiTenantVectorStore(db_path=lance, embedding_dimension=4)
        with store.claim_locked_batch(worker_id="new-owner", batch_size=1, lease_seconds=5) as jobs:
            assert [job["job_id"] for job in jobs] == [job_id]
            builder = MemoryBuilder(llm_client=client, vector_store=vector, table_name="fixture", window_size=1, entry_id_namespace="simplemem:fixture:"+job_id, checkpoint_loader=lambda index, input_hash: store.load_checkpoint(job_id=job_id, lease_owner="new-owner", window_index=index, input_hash=input_hash), checkpoint_saver=lambda index, input_hash, entries: store.save_checkpoint(job_id=job_id, lease_owner="new-owner", window_index=index, input_hash=input_hash, entries=entries), checkpoint_register=lambda ids: store.record_job_entries(job_id=job_id, lease_owner="new-owner", table_name="fixture", entry_ids=ids))
            asyncio.run(builder.add_dialogues([{"speaker": "u", "content": "first", "timestamp": "2026-01-01T00:00:00Z"}, {"speaker": "u", "content": "second", "timestamp": "2026-01-01T00:00:00Z"}]))
            assert client.calls == 1
            rows = vector._get_table("fixture").to_pandas()
            assert len(rows) == 2
            assert set(rows["lossless_restatement"].tolist()) == {"first durable fact", "second durable fact"}
            assert store.mark_done(job_id=job_id, lease_owner="new-owner", result={"ok": True})
    finally:
        store.close()


def test_checkpoint_integrity_failure_quarantines_without_retention_loss(tmp_path, monkeypatch):
    import asyncio
    from collections import deque
    from types import SimpleNamespace
    from server.database.durable_job_store import DurableJobStore
    from server.mcp_handler import MCPHandler

    async def scenario():
        store = DurableJobStore(str(tmp_path / "queue.sqlite"))
        job_id = store.enqueue_or_get(user_id="fixture", kind="memory_add", idempotency_key="key", payload={"content": "fixture", "table_name": "fixture"}, max_inflight_writes=10)["job_id"]
        with store.claim_locked_batch(worker_id="previous", batch_size=1, lease_seconds=5) as jobs:
            assert jobs
            store.save_checkpoint(job_id=job_id, lease_owner="previous", window_index=0, input_hash="original", entries=[])
            result = store.mark_failed(job_id=job_id, lease_owner="previous", error="fixture restart", retryable=True, max_retries=6, backoff_base_ms=100)
            assert result["status"] == "failed_retryable"
        store._get_connection().execute("UPDATE durable_jobs SET available_at_utc = '2000-01-01T00:00:00+00:00' WHERE job_id = ?", (job_id,))
        settings = SimpleNamespace(memory_write_lease_seconds=90, memory_write_max_retries=1, memory_write_batch_max=1, memory_write_batch_max_wait_ms=10, memory_write_job_status_limit=100, memory_write_retry_backoff_base_ms=100)
        for name, value in (("_durable_store", store), ("_shared_settings", settings), ("_worker_shutdown", asyncio.Event()), ("_lease_recoveries", 0), ("_write_status_by_job_id", {}), ("_write_status_order", deque()), ("_write_stats", dict(MCPHandler._write_stats)), ("_write_stats_by_user", {})):
            monkeypatch.setattr(MCPHandler, name, value)
        async def execute_fixture(cls, job):
            store.load_checkpoint(job_id=job_id, lease_owner=job["lease_owner"], window_index=0, input_hash="changed")
            pytest.fail("mismatched checkpoint must not execute")
        monkeypatch.setattr(MCPHandler, "_execute_durable_job", classmethod(execute_fixture))
        worker = asyncio.create_task(MCPHandler._durable_worker_loop(1))
        try:
            async def quarantined():
                while store.get_job(job_id=job_id)["status"] != "quarantined":
                    await asyncio.sleep(0.01)
            await asyncio.wait_for(quarantined(), 5)
            assert store.queue_counts()["quarantined"] == 1
            store._get_connection().execute("UPDATE durable_jobs SET updated_at_utc = '2000-01-01T00:00:00+00:00' WHERE job_id = ?", (job_id,))
            assert store.purge_completed_jobs(max_age_hours=24) == 0
            assert store._get_connection().execute("SELECT COUNT(*) FROM durable_job_checkpoints WHERE job_id = ?", (job_id,)).fetchone()[0] == 1
            assert not store.quarantine_checkpoint(job_id=job_id, lease_owner="previous", error="stale")
        finally:
            MCPHandler._worker_shutdown.set()
            worker.cancel()
            await asyncio.gather(worker, return_exceptions=True)
            store.close()
    asyncio.run(scenario())


def test_locked_head_does_not_starve_another_ready_job(tmp_path):
    from server.database.durable_job_store import DurableJobStore
    store = DurableJobStore(str(tmp_path / "queue.sqlite"))
    try:
        ids = [store.enqueue_or_get(user_id=user, kind="memory_store", idempotency_key=user, payload={"content": user}, max_inflight_writes=10)["job_id"] for user in ("a", "b")]
        with store.execution_locks.try_job(ids[0]) as held:
            assert held
            with store.claim_locked_batch(worker_id="worker", batch_size=1, lease_seconds=5) as jobs:
                assert [job["job_id"] for job in jobs] == [ids[1]]
                assert store.mark_done(job_id=ids[1], lease_owner="worker", result={"ok": True})
            assert store.get_job(job_id=ids[0])["status"] == "pending"
        with store.claim_locked_batch(worker_id="worker", batch_size=1, lease_seconds=5) as jobs:
            assert [job["job_id"] for job in jobs] == [ids[0]]
    finally:
        store.close()


def test_bounded_scan_reaches_ready_job_beyond_multiple_locked_pages(tmp_path):
    from contextlib import ExitStack
    from server.database.durable_job_store import DurableJobStore
    store = DurableJobStore(str(tmp_path / "queue.sqlite"))
    try:
        ids = [store.enqueue_or_get(user_id=f"u-{i}", kind="memory_store", idempotency_key=str(i), payload={"content": "fixture"}, max_inflight_writes=10)["job_id"] for i in range(130)]
        with ExitStack() as holders:
            assert all(holders.enter_context(store.execution_locks.try_job(job_id)) for job_id in ids[:-1])
            found = []
            for _ in range(3):
                with store.claim_locked_batch(worker_id="worker", batch_size=1, lease_seconds=5) as jobs:
                    if jobs:
                        found.extend(job["job_id"] for job in jobs)
                        assert store.mark_done(job_id=jobs[0]["job_id"], lease_owner="worker", result={"ok": True})
                        break
            assert found == [ids[-1]]
            assert all(store.get_job(job_id=job_id)["status"] == "pending" for job_id in ids[:-1])
    finally:
        store.close()


def test_legacy_unlocked_claim_fails_without_mutating_queue(tmp_path):
    from server.database.durable_job_store import DurableJobStore
    store = DurableJobStore(str(tmp_path / "queue.sqlite"))
    try:
        job_id = store.enqueue_or_get(user_id="fixture", kind="memory_store", idempotency_key="key", payload={"content": "fixture"}, max_inflight_writes=10)["job_id"]
        with store.execution_locks.try_job(job_id) as held:
            assert held
            with pytest.raises(RuntimeError, match="unlocked job claims are disabled"):
                store.claim_batch(worker_id="legacy", batch_size=1, lease_seconds=5)
        assert store.get_job(job_id=job_id)["status"] == "pending"
    finally:
        store.close()


def test_shutdown_waits_for_live_worker_and_keeps_its_lock(tmp_path, monkeypatch):
    import asyncio
    from collections import deque
    from types import SimpleNamespace
    from server.database.durable_job_store import DurableJobStore
    from server.mcp_handler import MCPHandler

    async def scenario():
        store = DurableJobStore(str(tmp_path / "queue.sqlite"))
        job_id = store.enqueue_or_get(user_id="fixture", kind="memory_store", idempotency_key="key", payload={"content": "fixture"}, max_inflight_writes=10)["job_id"]
        settings = SimpleNamespace(memory_write_lease_seconds=90, memory_write_max_retries=1, memory_write_batch_max=1, memory_write_batch_max_wait_ms=10, memory_write_job_status_limit=100, memory_write_retry_backoff_base_ms=100)
        for name, value in (("_runtime_lock", asyncio.Lock()), ("_durable_store", store), ("_service_lock_guard", None), ("_shared_settings", settings), ("_worker_shutdown", asyncio.Event()), ("_lease_recoveries", 0), ("_write_status_by_job_id", {}), ("_write_status_order", deque()), ("_write_stats", dict(MCPHandler._write_stats)), ("_write_stats_by_user", {}), ("_retention_task", None), ("_shared_fastpath_client", None)):
            monkeypatch.setattr(MCPHandler, name, value)
        monkeypatch.setattr(MCPHandler, "_invalidate_memory_caches", classmethod(lambda cls: None))
        started, release = asyncio.Event(), asyncio.Event()
        async def execute_fixture(cls, job):
            started.set()
            await release.wait()
            return {"fixture": True}
        monkeypatch.setattr(MCPHandler, "_execute_durable_job", classmethod(execute_fixture))
        worker = asyncio.create_task(MCPHandler._durable_worker_loop(1))
        monkeypatch.setattr(MCPHandler, "_worker_tasks", [worker])
        stop = None
        try:
            await asyncio.wait_for(started.wait(), 5)
            stop = asyncio.create_task(MCPHandler.stop_background_workers())
            await asyncio.sleep(0.02)
            assert not stop.done() and not worker.cancelled()
            store._get_connection().execute("UPDATE durable_jobs SET lease_expires_at_utc = '2000-01-01T00:00:00+00:00' WHERE job_id = ?", (job_id,))
            assert store.recover_expired_leases(max_retries=1) == 0
            release.set()
            await stop
            assert store.get_job(job_id=job_id)["status"] == "done"
            assert MCPHandler._worker_tasks == []
        finally:
            MCPHandler._worker_shutdown.set()
            release.set()
            if stop is not None:
                await asyncio.gather(stop, return_exceptions=True)
            await asyncio.gather(worker, return_exceptions=True)
            store.close()
    asyncio.run(scenario())


def test_user_cancellation_preserves_live_job_then_deletes_its_checkpoint(tmp_path):
    from server.database.durable_job_store import DurableJobStore
    store = DurableJobStore(str(tmp_path / "queue.sqlite"))
    try:
        ids = [store.enqueue_or_get(user_id=user, kind="memory_add", idempotency_key=user, payload={"content": user}, max_inflight_writes=10)["job_id"] for user in ("a", "b")]
        with store.claim_locked_batch(worker_id="worker", batch_size=2, lease_seconds=5) as jobs:
            assert {job["job_id"] for job in jobs} == set(ids)
            for job in jobs:
                store.save_checkpoint(job_id=job["job_id"], lease_owner="worker", window_index=0, input_hash="fixture", entries=[])
            with pytest.raises(RuntimeError, match="user_has_active_durable_job"):
                store.cancel_jobs_for_user("a")
            assert store.get_job(job_id=ids[0])["status"] == "processing"
            assert store._get_connection().execute("SELECT COUNT(*) FROM durable_job_checkpoints WHERE job_id = ?", (ids[0],)).fetchone()[0] == 1
        assert store.cancel_jobs_for_user("a") == 1
        assert store.get_job(job_id=ids[0]) is None
        assert store._get_connection().execute("SELECT COUNT(*) FROM durable_job_checkpoints WHERE job_id = ?", (ids[0],)).fetchone()[0] == 0
        assert store.get_job(job_id=ids[1])["status"] == "processing"
        assert store._get_connection().execute("SELECT COUNT(*) FROM durable_job_checkpoints WHERE job_id = ?", (ids[1],)).fetchone()[0] == 1
    finally:
        store.close()


def test_service_singleton_is_held_until_workers_shutdown(tmp_path, monkeypatch):
    import asyncio
    from types import SimpleNamespace
    from server.database.durable_job_store import DurableJobStore
    from server.mcp_handler import MCPHandler

    async def scenario():
        store = DurableJobStore(str(tmp_path / "queue.sqlite"))
        second = DurableJobStore(store.db_path)
        settings = SimpleNamespace(memory_write_workers=1, memory_write_lease_seconds=90, memory_write_max_retries=6, memory_write_batch_max=1, memory_write_batch_max_wait_ms=10, memory_write_purge_interval_s=300, memory_write_retention_hours=24)
        for name, value in (("_runtime_lock", None), ("_durable_store", store), ("_service_lock_guard", None), ("_worker_shutdown", None), ("_worker_tasks", []), ("_retention_task", None), ("_lease_recoveries", 0), ("_shared_fastpath_client", None), ("_shared_settings", None), ("_shared_vector_store", None), ("_shared_client_manager", None), ("_shared_user_store", None), ("_shared_token_manager", None)):
            monkeypatch.setattr(MCPHandler, name, value)
        try:
            await MCPHandler.start_background_workers(settings=settings, vector_store=object(), client_manager=object())
            with second.execution_locks.try_service() as acquired:
                assert not acquired
            with second.execution_locks.try_job("unrelated-job") as acquired:
                assert acquired
            await MCPHandler.stop_background_workers()
            with second.execution_locks.try_service() as acquired:
                assert acquired
        finally:
            if MCPHandler._service_lock_guard is not None:
                await MCPHandler.stop_background_workers()
            store.close()
            second.close()
    asyncio.run(scenario())


def test_actual_worker_crash_replays_checkpoint_without_duplicate_lance_entries(tmp_path, monkeypatch):
    import asyncio
    from collections import deque
    from types import SimpleNamespace
    from server.database.durable_job_store import DurableJobStore
    from server.database.vector_store import MultiTenantVectorStore
    from server.mcp_handler import MCPHandler

    db = str(tmp_path / "queue.sqlite")
    lance = str(tmp_path / "lance")
    store = DurableJobStore(db)
    dialogues = [{"speaker": "u", "content": "first", "timestamp": "2026-01-01T00:00:00Z"}, {"speaker": "u", "content": "second", "timestamp": "2026-01-01T00:00:00Z"}]
    job_id = store.enqueue_or_get(user_id="fixture", kind="memory_add_batch", idempotency_key="key", payload={"user_id": "fixture", "table_name": "fixture", "dialogues": dialogues}, max_inflight_writes=10)["job_id"]
    child = """
import asyncio, os, sys
from types import SimpleNamespace
from server.database.durable_job_store import DurableJobStore
from server.database.vector_store import MultiTenantVectorStore
from server.mcp_handler import MCPHandler
class Client:
    def __init__(self): self.calls = 0
    async def chat_completion(self, **kwargs):
        self.calls += 1
        if self.calls == 2: os._exit(73)
        return 'first durable fact'
    def extract_json(self, response): return {'entries': [{'lossless_restatement': response}]}
    async def create_embedding(self, texts): return [[0.1, 0.2, 0.3, 0.4] for _ in texts]
class Clients:
    def get_client(self, key): return Client()
class Users:
    def get_user(self, user_id): return SimpleNamespace(openrouter_api_key_encrypted='fixture')
class Tokens:
    def decrypt_api_key(self, value): return 'fixture'
async def run():
    c=MCPHandler
    c._durable_store=DurableJobStore(sys.argv[1])
    c._shared_settings=SimpleNamespace(memory_write_lease_seconds=90, memory_write_max_retries=6, memory_write_batch_max=1, memory_write_batch_max_wait_ms=10, memory_write_job_status_limit=100, memory_write_retry_backoff_base_ms=100, window_size=1, overlap_size=0, llm_temperature=0.1, llm_max_retries=1)
    c._shared_vector_store=MultiTenantVectorStore(db_path=sys.argv[2], embedding_dimension=4)
    c._shared_client_manager=Clients(); c._shared_user_store=Users(); c._shared_token_manager=Tokens(); c._worker_shutdown=asyncio.Event()
    await c._durable_worker_loop(1)
asyncio.run(run())
raise SystemExit(99)
"""
    try:
        proc = subprocess.run([sys.executable, "-c", child, db, lance], capture_output=True, text=True)
        assert proc.returncode == 73, proc.stderr
        assert store._get_connection().execute("SELECT COUNT(*) FROM durable_job_checkpoints WHERE job_id = ?", (job_id,)).fetchone()[0] == 1
        store._get_connection().execute("UPDATE durable_jobs SET lease_expires_at_utc = '2000-01-01T00:00:00+00:00' WHERE job_id = ?", (job_id,))
        assert store.recover_expired_leases() == 1
        class Client:
            calls = 0
            async def chat_completion(self, **kwargs):
                self.calls += 1
                return "second durable fact"
            def extract_json(self, response):
                return {"entries": [{"lossless_restatement": response}]}
            async def create_embedding(self, texts):
                return [[0.1, 0.2, 0.3, 0.4] for _ in texts]
        client = Client()
        class Clients:
            def get_client(self, key): return client
        class Users:
            def get_user(self, user_id): return SimpleNamespace(openrouter_api_key_encrypted="fixture")
        class Tokens:
            def decrypt_api_key(self, value): return "fixture"
        vector = MultiTenantVectorStore(db_path=lance, embedding_dimension=4)
        settings = SimpleNamespace(memory_write_lease_seconds=90, memory_write_max_retries=6, memory_write_batch_max=1, memory_write_batch_max_wait_ms=10, memory_write_job_status_limit=100, memory_write_retry_backoff_base_ms=100, window_size=1, overlap_size=0, llm_temperature=0.1, llm_max_retries=1)
        async def resume():
            for name, value in (("_durable_store", store), ("_shared_settings", settings), ("_shared_vector_store", vector), ("_shared_client_manager", Clients()), ("_shared_user_store", Users()), ("_shared_token_manager", Tokens()), ("_worker_shutdown", asyncio.Event()), ("_lease_recoveries", 0), ("_write_status_by_job_id", {}), ("_write_status_order", deque()), ("_write_stats", dict(MCPHandler._write_stats)), ("_write_stats_by_user", {})):
                monkeypatch.setattr(MCPHandler, name, value)
            monkeypatch.setattr(MCPHandler, "_invalidate_memory_caches", classmethod(lambda cls: None))
            worker = asyncio.create_task(MCPHandler._durable_worker_loop(2))
            try:
                async def completed():
                    while store.get_job(job_id=job_id)["status"] != "done":
                        await asyncio.sleep(0.01)
                await asyncio.wait_for(completed(), 5)
            finally:
                MCPHandler._worker_shutdown.set()
                await worker
        asyncio.run(resume())
        assert client.calls == 1
        rows = vector._get_table("fixture").to_pandas()
        assert len(rows) == 2
        assert set(rows["lossless_restatement"].tolist()) == {"first durable fact", "second durable fact"}
        assert store._get_connection().execute("SELECT COUNT(*) FROM durable_job_checkpoints WHERE job_id = ?", (job_id,)).fetchone()[0] == 2
    finally:
        store.close()


def test_existing_queue_upgrades_without_changing_accepted_job(tmp_path):
    from server.database.durable_job_store import DurableJobStore
    db = str(tmp_path / "legacy.sqlite")
    conn = sqlite3.connect(db)
    conn.execute("""CREATE TABLE durable_jobs (
        job_id TEXT PRIMARY KEY, user_id TEXT NOT NULL, kind TEXT NOT NULL,
        idempotency_key TEXT NOT NULL, payload_json TEXT NOT NULL, status TEXT NOT NULL,
        attempt_count INTEGER NOT NULL DEFAULT 0, available_at_utc TEXT NOT NULL,
        lease_owner TEXT, lease_expires_at_utc TEXT, last_error TEXT, result_json TEXT,
        created_at_utc TEXT NOT NULL, updated_at_utc TEXT NOT NULL)""")
    conn.execute("INSERT INTO durable_jobs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", ("legacy-job", "fixture", "memory_store", "key", '{"content":"fixture"}', "pending", 0, "2000-01-01T00:00:00+00:00", None, None, None, None, "2000-01-01T00:00:00+00:00", "2000-01-01T00:00:00+00:00"))
    conn.commit()
    conn.close()
    store = DurableJobStore(db)
    try:
        with store.claim_locked_batch(worker_id="worker", batch_size=1, lease_seconds=5) as jobs:
            assert len(jobs) == 1 and jobs[0]["job_id"] == "legacy-job"
            assert jobs[0]["payload"] == {"content": "fixture"}
            assert store.mark_done(job_id="legacy-job", lease_owner="worker", result={"ok": True})
        assert store.get_job(job_id="legacy-job")["status"] == "done"
    finally:
        store.close()


def test_changed_window_input_never_reuses_checkpoint_or_calls_model(tmp_path):
    import asyncio
    from server.core.memory_builder import MemoryBuilder
    from server.database.durable_job_store import CheckpointIntegrityError, DurableJobStore
    from server.database.vector_store import MultiTenantVectorStore

    class Client:
        calls = 0
        async def chat_completion(self, **kwargs):
            self.calls += 1
            return "durable fact"
        def extract_json(self, response):
            return {"entries": [{"lossless_restatement": response}]}
        async def create_embedding(self, texts):
            return [[0.1, 0.2, 0.3, 0.4] for _ in texts]
    store = DurableJobStore(str(tmp_path / "queue.sqlite"))
    job_id = store.enqueue_or_get(user_id="fixture", kind="memory_add", idempotency_key="key", payload={"content": "fixture", "table_name": "fixture"}, max_inflight_writes=10)["job_id"]
    vector = MultiTenantVectorStore(db_path=str(tmp_path / "lance"), embedding_dimension=4)
    client = Client()
    try:
        with store.claim_locked_batch(worker_id="worker", batch_size=1, lease_seconds=5) as jobs:
            assert jobs
            def builder():
                return MemoryBuilder(llm_client=client, vector_store=vector, table_name="fixture", entry_id_namespace="simplemem:fixture:"+job_id, checkpoint_loader=lambda index, input_hash: store.load_checkpoint(job_id=job_id, lease_owner="worker", window_index=index, input_hash=input_hash), checkpoint_saver=lambda index, input_hash, entries: store.save_checkpoint(job_id=job_id, lease_owner="worker", window_index=index, input_hash=input_hash, entries=entries), checkpoint_register=lambda ids: store.record_job_entries(job_id=job_id, lease_owner="worker", table_name="fixture", entry_ids=ids))
            asyncio.run(builder().add_dialogue("u", "original", timestamp="2026-01-01T00:00:00Z"))
            with pytest.raises(CheckpointIntegrityError, match="checkpoint_input_mismatch"):
                asyncio.run(builder().add_dialogue("u", "changed", timestamp="2026-01-01T00:00:00Z"))
            assert client.calls == 1
            assert len(vector._get_table("fixture").to_pandas()) == 1
    finally:
        store.close()


def test_delete_waits_for_exact_replayable_entry_then_succeeds(tmp_path):
    import asyncio
    from server.core.memory_builder import MemoryBuilder
    from server.database.durable_job_store import DurableJobStore
    from server.database.vector_store import MultiTenantVectorStore

    class Client:
        calls = 0
        async def chat_completion(self, **kwargs):
            self.calls += 1
            return "fixture fact"
        def extract_json(self, response): return {"entries": [{"lossless_restatement": response}]}
        async def create_embedding(self, texts): return [[0.1, 0.2, 0.3, 0.4] for _ in texts]
    async def scenario():
        store = DurableJobStore(str(tmp_path / "queue.sqlite"))
        vector = MultiTenantVectorStore(db_path=str(tmp_path / "lance"), embedding_dimension=4)
        vector.set_pending_replay_checker(store.has_pending_entry)
        client = Client()
        job_id = store.enqueue_or_get(user_id="fixture", kind="memory_add", idempotency_key="key", payload={"content": "fixture", "table_name": "fixture"}, max_inflight_writes=10)["job_id"]
        def builder(owner):
            return MemoryBuilder(llm_client=client, vector_store=vector, table_name="fixture", entry_id_namespace="simplemem:fixture:"+job_id, checkpoint_loader=lambda index, h: store.load_checkpoint(job_id=job_id, lease_owner=owner, window_index=index, input_hash=h), checkpoint_saver=lambda index, h, entries: store.save_checkpoint(job_id=job_id, lease_owner=owner, window_index=index, input_hash=h, entries=entries), checkpoint_register=lambda ids: store.record_job_entries(job_id=job_id, lease_owner=owner, table_name="fixture", entry_ids=ids))
        try:
            with store.claim_locked_batch(worker_id="old", batch_size=1, lease_seconds=5) as jobs:
                assert jobs
                await builder("old").add_dialogue("u", "fixture", timestamp="2026-01-01T00:00:00Z")
                ids = vector._get_table("fixture").to_pandas()["entry_id"].tolist()
                assert len(ids) == 1
                result = await vector.delete_entry("fixture", ids[0])
                assert result["deleted"] is False and result["retryable"] is True
                assert result["error"] == "durable_write_not_final"
            store._get_connection().execute("UPDATE durable_jobs SET lease_expires_at_utc = '2000-01-01T00:00:00+00:00' WHERE job_id = ?", (job_id,))
            assert store.recover_expired_leases() == 1
            with store.claim_locked_batch(worker_id="new", batch_size=1, lease_seconds=5) as jobs:
                assert jobs
                await builder("new").add_dialogue("u", "fixture", timestamp="2026-01-01T00:00:00Z")
                assert client.calls == 1
                assert not (await vector.delete_entry("fixture", ids[0]))["deleted"]
                assert store.mark_done(job_id=job_id, lease_owner="new", result={"ok": True})
            # Another pending job for this table does not block an unrelated ID.
            store.enqueue_or_get(user_id="fixture", kind="memory_store", idempotency_key="other", payload={"content": "other", "table_name": "fixture"}, max_inflight_writes=10)
            result = await vector.delete_entry("fixture", ids[0])
            assert result["deleted"] is True
            assert await vector.get_entry("fixture", ids[0]) is None
        finally:
            store.close()
    asyncio.run(scenario())


def test_delete_without_replay_guard_fails_closed(tmp_path):
    import asyncio
    from server.auth.models import MemoryEntry
    from server.database.vector_store import MultiTenantVectorStore
    async def scenario():
        vector = MultiTenantVectorStore(db_path=str(tmp_path / "lance"), embedding_dimension=4)
        await vector.add_entries("fixture", [MemoryEntry(entry_id="id", lossless_restatement="fixture")], [[0.1, 0.2, 0.3, 0.4]])
        result = await vector.delete_entry("fixture", "id")
        assert result["deleted"] is False and result["error"] == "durable_replay_guard_unavailable"
        assert await vector.get_entry("fixture", "id") is not None
    asyncio.run(scenario())


def test_raw_worker_registers_replay_guard_before_memory_write(tmp_path, monkeypatch):
    import asyncio
    from types import SimpleNamespace
    from server.database.durable_job_store import DurableJobStore
    from server.database.vector_store import MultiTenantVectorStore
    from server.mcp_handler import MCPHandler

    class Client:
        async def create_bounded_semantic_embedding(self, text): return [0.1, 0.2, 0.3, 0.4]
    class Clients:
        def get_client(self, key): return Client()
    class Users:
        def get_user(self, user_id): return SimpleNamespace(openrouter_api_key_encrypted="fixture")
    class Tokens:
        def decrypt_api_key(self, value): return "fixture"
    async def scenario():
        store = DurableJobStore(str(tmp_path / "queue.sqlite"))
        vector = MultiTenantVectorStore(db_path=str(tmp_path / "lance"), embedding_dimension=4)
        vector.set_pending_replay_checker(store.has_pending_entry)
        job_id = store.enqueue_or_get(user_id="fixture", kind="memory_store", idempotency_key="key", payload={"user_id": "fixture", "table_name": "fixture", "content": "raw fixture"}, max_inflight_writes=10)["job_id"]
        for name, value in (("_durable_store", store), ("_shared_settings", object()), ("_shared_vector_store", vector), ("_shared_client_manager", Clients()), ("_shared_user_store", Users()), ("_shared_token_manager", Tokens())):
            monkeypatch.setattr(MCPHandler, name, value)
        try:
            with store.claim_locked_batch(worker_id="worker", batch_size=1, lease_seconds=5) as jobs:
                assert len(jobs) == 1
                result = await MCPHandler._execute_durable_job(jobs[0])
                assert result["added"]
                ids = vector._get_table("fixture").to_pandas()["entry_id"].tolist()
                assert len(ids) == 1
                rejected = await vector.delete_entry("fixture", ids[0])
                assert rejected["error"] == "durable_write_not_final" and rejected["retryable"]
                assert store.mark_done(job_id=job_id, lease_owner="worker", result=result)
            assert (await vector.delete_entry("fixture", ids[0]))["deleted"]
        finally:
            store.close()
    asyncio.run(scenario())


@pytest.mark.parametrize("method", ["_tool_memory_store", "_tool_memory_add", "_tool_memory_add_batch"])
def test_quarantine_is_reported_without_looking_like_processing(tmp_path, monkeypatch, method):
    import asyncio
    from types import SimpleNamespace
    from server.auth.models import User
    from server.database.durable_job_store import DurableJobStore
    from server.mcp_handler import MCPHandler

    store = DurableJobStore(str(tmp_path / "queue.sqlite"))
    job_id = store.enqueue_or_get(user_id="fixture", kind="memory_store", idempotency_key="key", payload={"content": "fixture"}, max_inflight_writes=10)["job_id"]
    store._get_connection().execute("UPDATE durable_jobs SET status='quarantined', last_error='checkpoint mismatch' WHERE job_id = ?", (job_id,))
    monkeypatch.setattr(MCPHandler, "_durable_store", store)
    handler = MCPHandler.__new__(MCPHandler)
    handler.user = User(user_id="fixture")
    handler.settings = SimpleNamespace(memory_add_async_default=False, memory_sync_wait_timeout_ms=100, memory_write_max_content_chars=8192, memory_write_max_metadata_chars=4096)
    async def queued(**kwargs):
        return {"accepted": True, "status": "quarantined", "job_id": job_id, "idempotency_key": "key", "existing": True, "last_error": "checkpoint mismatch"}
    monkeypatch.setattr(handler, "_enqueue_durable", queued)
    try:
        result = asyncio.run(getattr(handler, method)({"content": "fixture", "sync": True}))
        assert result["status"] == "quarantined"
        assert result["error"] == "checkpoint mismatch" and result["retryable"] is False
        assert store.queue_counts(user_id="fixture")["quarantined"] == 1
    finally:
        store.close()


def test_health_distinguishes_quarantined_jobs_from_unavailable_queue(tmp_path):
    import ast
    import asyncio
    import logging
    from datetime import datetime, timezone
    from types import SimpleNamespace
    from typing import Any
    from server.database.durable_job_store import DurableJobStore

    source = (Path(__file__).resolve().parents[1] / "server/http_server.py").read_text()
    tree = ast.parse(source)
    function = next(node for node in tree.body if isinstance(node, ast.AsyncFunctionDef) and node.name == "health_check")
    function.decorator_list = []
    store = DurableJobStore(str(tmp_path / "queue.sqlite"))
    job_id = store.enqueue_or_get(user_id="fixture", kind="memory_store", idempotency_key="key", payload={"content": "fixture"}, max_inflight_writes=10)["job_id"]
    store._get_connection().execute("UPDATE durable_jobs SET status='quarantined' WHERE job_id = ?", (job_id,))
    handler = SimpleNamespace(_durable_store=store)
    scope = {"MCPHandler": handler, "user_store": SimpleNamespace(count_users=lambda: 1), "vector_store": SimpleNamespace(table_names=lambda: []), "datetime": datetime, "timezone": timezone, "logger": logging.getLogger(__name__), "Any": Any}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "health_check_fixture", "exec"), scope)
    try:
        health = asyncio.run(scope["health_check"]())
        assert health["status"] == "healthy" and health["checks"]["queue_db"] is True
        assert health["queue_state"] == "attention_required" and health["queue"]["quarantined"] == 1
        handler._durable_store = None
        health = asyncio.run(scope["health_check"]())
        assert health["status"] == "degraded" and health["checks"]["queue_db"] is False
        assert health["queue_state"] == "unavailable"
    finally:
        store.close()
