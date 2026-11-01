"""
Durable SQLite WAL-backed queue for SimpleMem write jobs.

This queue is the source of truth for accepted write requests.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
import uuid
from contextlib import ExitStack, contextmanager
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from .execution_lock import JobExecutionLocks


class CheckpointIntegrityError(RuntimeError):
    """A durable checkpoint cannot safely be replayed without inspection."""


_CLAIM_CANDIDATES_SQL = """
SELECT job_id, user_id, kind, idempotency_key, payload_json, status, attempt_count,
       available_at_utc, created_at_utc, updated_at_utc
FROM (
    SELECT job_id, user_id, kind, idempotency_key, payload_json, status,
           attempt_count, available_at_utc, created_at_utc, updated_at_utc,
           ROW_NUMBER() OVER (
               PARTITION BY user_id ORDER BY created_at_utc ASC, job_id ASC
           ) AS user_rank
    FROM durable_jobs
    WHERE status IN ('pending', 'failed_retryable') AND available_at_utc <= ?
)
WHERE user_rank <= ?
  AND (? IS NULL OR created_at_utc > ? OR (created_at_utc = ? AND job_id > ?))
ORDER BY created_at_utc ASC, job_id ASC LIMIT ?
"""


class DurableJobStore:
    """SQLite WAL durable queue with idempotent enqueue + leasing."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_lock = threading.Lock()
        self._local = threading.local()
        self._claim_scan_cursor: tuple[str, str] | None = None
        parent = os.path.dirname(db_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        self._init_db()
        self.execution_locks = JobExecutionLocks(db_path, self._get_connection())

    @staticmethod
    def _utc_now() -> datetime:
        return datetime.now(timezone.utc)

    @staticmethod
    def _utc_now_iso() -> str:
        return DurableJobStore._utc_now().isoformat()

    def _get_connection(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            try:
                conn.execute("SELECT 1")
                return conn
            except Exception:
                try:
                    conn.close()
                except Exception:
                    pass
                conn = None
        conn = sqlite3.connect(self.db_path, timeout=30.0, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA synchronous=FULL")
        conn.execute("PRAGMA busy_timeout=30000")
        self._local.conn = conn
        return conn

    def close(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
            self._local.conn = None

    @contextmanager
    def _connection(self):
        conn = self._get_connection()
        yield conn

    def _init_db(self) -> None:
        with self._init_lock:
            with self._connection() as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS durable_jobs (
                        job_id TEXT PRIMARY KEY,
                        user_id TEXT NOT NULL,
                        kind TEXT NOT NULL,
                        idempotency_key TEXT NOT NULL,
                        payload_json TEXT NOT NULL,
                        status TEXT NOT NULL,
                        attempt_count INTEGER NOT NULL DEFAULT 0,
                        available_at_utc TEXT NOT NULL,
                        lease_owner TEXT,
                        lease_expires_at_utc TEXT,
                        last_error TEXT,
                        result_json TEXT,
                        created_at_utc TEXT NOT NULL,
                        updated_at_utc TEXT NOT NULL
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS durable_job_checkpoints (
                        job_id TEXT NOT NULL,
                        window_index INTEGER NOT NULL,
                        input_hash TEXT NOT NULL,
                        entries_json TEXT NOT NULL,
                        PRIMARY KEY(job_id, window_index),
                        FOREIGN KEY(job_id) REFERENCES durable_jobs(job_id) ON DELETE CASCADE
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS durable_job_entries (
                        job_id TEXT NOT NULL,
                        table_name TEXT NOT NULL,
                        entry_id TEXT NOT NULL,
                        PRIMARY KEY(job_id, table_name, entry_id),
                        FOREIGN KEY(job_id) REFERENCES durable_jobs(job_id) ON DELETE CASCADE
                    )
                    """
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_durable_job_entry_lookup ON durable_job_entries(table_name, entry_id)"
                )
                conn.execute(
                    """
                    CREATE UNIQUE INDEX IF NOT EXISTS idx_durable_jobs_idempotency
                    ON durable_jobs(user_id, kind, idempotency_key)
                    """
                )
                conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_durable_jobs_status_available
                    ON durable_jobs(status, available_at_utc, created_at_utc)
                    """
                )
                conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_durable_jobs_lease
                    ON durable_jobs(lease_expires_at_utc)
                    """
                )

    def enqueue_or_get(
        self,
        *,
        user_id: str,
        kind: str,
        idempotency_key: str,
        payload: dict[str, Any],
        max_inflight_writes: int,
        max_global_inflight_writes: int | None = None,
    ) -> dict[str, Any]:
        now = self._utc_now()
        now_iso = now.isoformat()
        payload_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        with self._connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                # Idempotency wins over admission limits.
                row = conn.execute(
                    """
                    SELECT job_id, status, attempt_count, created_at_utc, updated_at_utc, result_json, last_error
                    FROM durable_jobs
                    WHERE user_id = ? AND kind = ? AND idempotency_key = ?
                    """,
                    (user_id, kind, idempotency_key),
                ).fetchone()
                if row:
                    result_obj = None
                    raw = row["result_json"]
                    if raw:
                        try:
                            result_obj = json.loads(raw)
                        except json.JSONDecodeError:
                            result_obj = {"decode_error": True}
                    conn.execute("COMMIT")
                    return {
                        "accepted": True,
                        "durable": True,
                        "status": str(row["status"]),
                        "job_id": str(row["job_id"]),
                        "idempotency_key": idempotency_key,
                        "attempt_count": int(row["attempt_count"]),
                        "created_at_utc": row["created_at_utc"],
                        "updated_at_utc": row["updated_at_utc"],
                        "result": result_obj,
                        "last_error": row["last_error"],
                        "existing": True,
                    }

                inflight_row = conn.execute(
                    """
                    SELECT COUNT(*) AS c
                    FROM durable_jobs
                    WHERE user_id = ?
                      AND status IN ('pending', 'processing', 'failed_retryable')
                    """,
                    (user_id,),
                ).fetchone()
                inflight = int(inflight_row["c"] if inflight_row else 0)
                global_limit = max_global_inflight_writes
                if global_limit is not None:
                    global_row = conn.execute(
                        """
                        SELECT COUNT(*) AS c
                        FROM durable_jobs
                        WHERE status IN ('pending', 'processing', 'failed_retryable')
                        """
                    ).fetchone()
                    global_inflight = int(global_row["c"] if global_row else 0)
                    if global_inflight >= max(1, int(global_limit)):
                        conn.execute("ROLLBACK")
                        return {
                            "accepted": False,
                            "durable": False,
                            "status": "rejected_overloaded",
                            "reason": "memory_write_global_admission_limit",
                            "retryable": True,
                            "inflight": global_inflight,
                            "limit": max(1, int(global_limit)),
                        }
                if inflight >= max(1, int(max_inflight_writes)):
                    conn.execute("ROLLBACK")
                    return {
                        "accepted": False,
                        "durable": False,
                        "status": "rejected_overloaded",
                        "reason": "memory_write_admission_limit",
                        "retryable": True,
                        "inflight": inflight,
                        "limit": max(1, int(max_inflight_writes)),
                    }

                job_id = f"dq-{uuid.uuid4()}"
                conn.execute(
                    """
                    INSERT INTO durable_jobs(
                        job_id, user_id, kind, idempotency_key, payload_json, status,
                        attempt_count, available_at_utc, lease_owner, lease_expires_at_utc,
                        last_error, result_json, created_at_utc, updated_at_utc
                    )
                    VALUES (?, ?, ?, ?, ?, 'pending', 0, ?, NULL, NULL, NULL, NULL, ?, ?)
                    """,
                    (
                        job_id,
                        user_id,
                        kind,
                        idempotency_key,
                        payload_json,
                        now_iso,
                        now_iso,
                        now_iso,
                    ),
                )
                conn.execute("COMMIT")
                return {
                    "accepted": True,
                    "durable": True,
                    "status": "queued",
                    "job_id": job_id,
                    "idempotency_key": idempotency_key,
                    "attempt_count": 0,
                    "created_at_utc": now_iso,
                    "updated_at_utc": now_iso,
                    "existing": False,
                }
            except Exception:
                conn.execute("ROLLBACK")
                raise

    def load_checkpoint(
        self, *, job_id: str, lease_owner: str, window_index: int, input_hash: str
    ) -> list[dict[str, Any]] | None:
        """Load a generated window only while this worker owns the job."""
        with self._connection() as conn:
            row = conn.execute(
                """
                SELECT c.input_hash, c.entries_json FROM durable_jobs j
                LEFT JOIN durable_job_checkpoints c
                  ON c.job_id = j.job_id AND c.window_index = ?
                WHERE j.job_id = ? AND j.status = 'processing' AND j.lease_owner = ?
                """,
                (window_index, job_id, lease_owner),
            ).fetchone()
            if row is None:
                raise RuntimeError("durable_job_lease_lost_before_checkpoint")
            if row["entries_json"] is None:
                return None
            if row["input_hash"] != input_hash:
                raise CheckpointIntegrityError("durable_checkpoint_input_mismatch")
            try:
                entries = json.loads(row["entries_json"])
            except json.JSONDecodeError as exc:
                raise CheckpointIntegrityError("invalid durable job checkpoint JSON") from exc
            if not isinstance(entries, list) or any(not isinstance(item, dict) for item in entries):
                raise CheckpointIntegrityError("invalid durable job checkpoint")
            return entries

    def save_checkpoint(
        self, *, job_id: str, lease_owner: str, window_index: int, input_hash: str, entries: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Persist the first generated window before any corresponding Lance write."""
        payload = json.dumps(entries, ensure_ascii=False, separators=(",", ":"))
        with self._connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                owner = conn.execute(
                    "SELECT 1 FROM durable_jobs WHERE job_id = ? AND status = 'processing' AND lease_owner = ?",
                    (job_id, lease_owner),
                ).fetchone()
                if owner is None:
                    raise RuntimeError("durable_job_lease_lost_before_checkpoint")
                conn.execute(
                    "INSERT OR IGNORE INTO durable_job_checkpoints(job_id, window_index, input_hash, entries_json) VALUES (?, ?, ?, ?)",
                    (job_id, window_index, input_hash, payload),
                )
                row = conn.execute(
                    "SELECT input_hash, entries_json FROM durable_job_checkpoints WHERE job_id = ? AND window_index = ?",
                    (job_id, window_index),
                ).fetchone()
                if row["input_hash"] != input_hash:
                    raise CheckpointIntegrityError("durable_checkpoint_input_mismatch")
                try:
                    result = json.loads(row["entries_json"])
                except json.JSONDecodeError as exc:
                    raise CheckpointIntegrityError("invalid durable job checkpoint JSON") from exc
                if not isinstance(result, list) or any(not isinstance(item, dict) for item in result):
                    raise CheckpointIntegrityError("invalid durable job checkpoint")
                conn.execute("COMMIT")
                return result
            except BaseException:
                conn.execute("ROLLBACK")
                raise

    def record_job_entries(
        self, *, job_id: str, lease_owner: str, table_name: str, entry_ids: list[str]
    ) -> None:
        """Record replayable IDs before any corresponding physical write."""
        if any(not isinstance(entry_id, str) or not entry_id for entry_id in entry_ids):
            raise CheckpointIntegrityError("invalid durable job entry identity")
        with self._connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                row = conn.execute(
                    "SELECT payload_json FROM durable_jobs WHERE job_id = ? AND status = 'processing' AND lease_owner = ?",
                    (job_id, lease_owner),
                ).fetchone()
                if row is None:
                    raise RuntimeError("durable_job_lease_lost_before_entry_registration")
                try:
                    payload = json.loads(row["payload_json"])
                except json.JSONDecodeError as exc:
                    raise CheckpointIntegrityError("invalid durable job payload JSON") from exc
                if not isinstance(payload, dict) or payload.get("table_name") != table_name:
                    raise CheckpointIntegrityError("durable_job_table_identity_mismatch")
                conn.executemany(
                    "INSERT OR IGNORE INTO durable_job_entries(job_id, table_name, entry_id) VALUES (?, ?, ?)",
                    [(job_id, table_name, entry_id) for entry_id in entry_ids],
                )
                conn.execute("COMMIT")
            except BaseException:
                conn.execute("ROLLBACK")
                raise

    def has_pending_entry(self, table_name: str, entry_id: str) -> bool:
        """Return whether an unfinished durable job can replay this exact ID."""
        with self._connection() as conn:
            row = conn.execute(
                """
                SELECT 1 FROM durable_job_entries e
                JOIN durable_jobs j ON j.job_id = e.job_id
                WHERE e.table_name = ? AND e.entry_id = ?
                  AND j.status IN ('pending', 'processing', 'failed_retryable', 'quarantined')
                LIMIT 1
                """,
                (table_name, entry_id),
            ).fetchone()
            return row is not None

    def claim_batch(self, **_kwargs):
        """Reject legacy claims that cannot retain the required execution locks."""
        raise RuntimeError("unlocked job claims are disabled; use claim_locked_batch")

    @contextmanager
    def claim_locked_batch(
        self,
        *,
        worker_id: str,
        batch_size: int,
        lease_seconds: int,
        max_jobs_per_user: int = 4,
    ):
        """Lock candidates before publishing claims; hold locks until the caller exits."""
        now = self._utc_now()
        now_iso = now.isoformat()
        take = max(1, int(batch_size))
        with self._connection() as conn:
            # Always revisit the oldest page and scan one additional page past
            # a stable keyset cursor, so locked heads cannot starve later jobs.
            page_size = max(64, take)
            user_limit = max(1, int(max_jobs_per_user))
            def page(cursor):
                created, job_id = cursor if cursor is not None else (None, None)
                return conn.execute(
                    _CLAIM_CANDIDATES_SQL,
                    (now_iso, user_limit, created, created, created, job_id, page_size),
                ).fetchall()
            with ExitStack() as held_locks:
                locked = []
                seen = set()
                front = page(None)
                for row in front:
                    seen.add(str(row["job_id"]))
                    if held_locks.enter_context(self.execution_locks.try_job(str(row["job_id"]))):
                        locked.append(row)
                        if len(locked) >= take:
                            break
                if len(locked) < take and front:
                    cursor = self._claim_scan_cursor or (str(front[-1]["created_at_utc"]), str(front[-1]["job_id"]))
                    following = page(cursor)
                    examined = None
                    for row in following:
                        examined = (str(row["created_at_utc"]), str(row["job_id"]))
                        if str(row["job_id"]) in seen:
                            continue
                        if held_locks.enter_context(self.execution_locks.try_job(str(row["job_id"]))):
                            locked.append(row)
                            if len(locked) >= take:
                                break
                    if not following or (examined == (str(following[-1]["created_at_utc"]), str(following[-1]["job_id"])) and len(following) < page_size):
                        self._claim_scan_cursor = None
                    elif examined is not None:
                        self._claim_scan_cursor = examined
                if not locked:
                    yield []
                    return
                # No blocking job-lock acquisition occurs inside this transaction.
                now = self._utc_now()
                now_iso = now.isoformat()
                lease_expires_iso = (now + timedelta(seconds=max(5, int(lease_seconds)))).isoformat()
                conn.execute("BEGIN IMMEDIATE")
                try:
                    out = []
                    for row in locked:
                        job_id = str(row["job_id"])
                        cur = conn.execute(
                            """
                            UPDATE durable_jobs
                            SET status = 'processing', lease_owner = ?,
                                lease_expires_at_utc = ?, updated_at_utc = ?
                            WHERE job_id = ? AND status = ? AND available_at_utc = ?
                              AND updated_at_utc = ? AND attempt_count = ?
                              AND available_at_utc <= ?
                            """,
                            (worker_id, lease_expires_iso, now_iso, job_id,
                             str(row["status"]), str(row["available_at_utc"]),
                             str(row["updated_at_utc"]), int(row["attempt_count"]), now_iso),
                        )
                        if cur.rowcount != 1:
                            continue
                        payload_raw = str(row["payload_json"])
                        try:
                            payload = json.loads(payload_raw)
                        except json.JSONDecodeError:
                            payload = {"_corrupt_payload": True, "_raw": payload_raw}
                        out.append({
                            "job_id": job_id,
                            "user_id": str(row["user_id"]),
                            "kind": str(row["kind"]),
                            "idempotency_key": str(row["idempotency_key"]),
                            "payload": payload,
                            "status": str(row["status"]),
                            "attempt_count": int(row["attempt_count"]),
                            "lease_owner": worker_id,
                            "available_at_utc": str(row["available_at_utc"]),
                            "created_at_utc": str(row["created_at_utc"]),
                            "updated_at_utc": str(row["updated_at_utc"]),
                        })
                    conn.execute("COMMIT")
                except BaseException:
                    conn.execute("ROLLBACK")
                    raise
                yield out

    def mark_done(
        self, *, job_id: str, lease_owner: str, result: dict[str, Any]
    ) -> bool:
        now_iso = self._utc_now_iso()
        result_json = json.dumps(result, ensure_ascii=False, separators=(",", ":"))
        with self._connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                cur = conn.execute(
                    """
                    UPDATE durable_jobs
                    SET status = 'done',
                        lease_owner = NULL,
                        lease_expires_at_utc = NULL,
                        last_error = NULL,
                        result_json = ?,
                        updated_at_utc = ?
                    WHERE job_id = ?
                      AND lease_owner = ?
                      AND status = 'processing'
                    """,
                    (result_json, now_iso, job_id, lease_owner),
                )
                conn.execute("COMMIT")
                return cur.rowcount == 1
            except Exception:
                conn.execute("ROLLBACK")
                raise

    def mark_failed(
        self,
        *,
        job_id: str,
        lease_owner: str,
        error: str,
        retryable: bool,
        max_retries: int,
        backoff_base_ms: int,
    ) -> dict[str, Any]:
        now = self._utc_now()
        now_iso = now.isoformat()
        with self._connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                row = conn.execute(
                    """
                    SELECT attempt_count FROM durable_jobs
                    WHERE job_id = ? AND lease_owner = ? AND status = 'processing'
                    """,
                    (job_id, lease_owner),
                ).fetchone()
                if not row:
                    conn.execute("ROLLBACK")
                    return {"status": "lease_lost"}
                attempt_count = int(row["attempt_count"]) + 1

                if retryable and attempt_count < max(1, int(max_retries)):
                    exp = max(1, attempt_count - 1)
                    delay_ms = int(max(100, int(backoff_base_ms)) * (2**exp))
                    # jitter in-process based on time + hash without random module dependency
                    jitter = (
                        abs(hash(f"{job_id}:{attempt_count}:{time.time_ns()}")) % 200
                    )
                    retry_at = (
                        now + timedelta(milliseconds=delay_ms + jitter)
                    ).isoformat()
                    conn.execute(
                        """
                        UPDATE durable_jobs
                        SET status = 'failed_retryable',
                            attempt_count = ?,
                            available_at_utc = ?,
                            lease_owner = NULL,
                            lease_expires_at_utc = NULL,
                            last_error = ?,
                            updated_at_utc = ?
                        WHERE job_id = ?
                          AND lease_owner = ?
                          AND status = 'processing'
                        """,
                        (
                            attempt_count,
                            retry_at,
                            error[:5000],
                            now_iso,
                            job_id,
                            lease_owner,
                        ),
                    )
                    conn.execute("COMMIT")
                    return {
                        "status": "failed_retryable",
                        "attempt_count": attempt_count,
                        "retry_at_utc": retry_at,
                    }

                conn.execute(
                    """
                    UPDATE durable_jobs
                    SET status = 'failed_terminal',
                        attempt_count = ?,
                        lease_owner = NULL,
                        lease_expires_at_utc = NULL,
                        last_error = ?,
                        updated_at_utc = ?
                    WHERE job_id = ?
                      AND lease_owner = ?
                      AND status = 'processing'
                    """,
                    (attempt_count, error[:5000], now_iso, job_id, lease_owner),
                )
                conn.execute("COMMIT")
                return {"status": "failed_terminal", "attempt_count": attempt_count}
            except Exception:
                conn.execute("ROLLBACK")
                raise

    def quarantine_checkpoint(self, *, job_id: str, lease_owner: str, error: str) -> bool:
        """Preserve unsafe-to-replay jobs outside automatic retries and retention."""
        with self._connection() as conn:
            cur = conn.execute(
                """
                UPDATE durable_jobs
                SET status = 'quarantined', lease_owner = NULL,
                    lease_expires_at_utc = NULL, last_error = ?, updated_at_utc = ?
                WHERE job_id = ? AND status = 'processing' AND lease_owner = ?
                """,
                (error[:5000], self._utc_now_iso(), job_id, lease_owner),
            )
            return cur.rowcount == 1

    def get_job(self, *, job_id: str) -> dict[str, Any] | None:
        with self._connection() as conn:
            row = conn.execute(
                """
                SELECT job_id, user_id, kind, idempotency_key, status, attempt_count, available_at_utc,
                       lease_owner, lease_expires_at_utc, last_error, result_json, created_at_utc, updated_at_utc
                FROM durable_jobs
                WHERE job_id = ?
                """,
                (job_id,),
            ).fetchone()
            if not row:
                return None
            result_obj = None
            raw_result = row["result_json"]
            if raw_result and isinstance(raw_result, str):
                try:
                    result_obj = json.loads(raw_result)
                except json.JSONDecodeError:
                    result_obj = {"decode_error": True}
            return {
                "job_id": str(row["job_id"]),
                "user_id": str(row["user_id"]),
                "kind": str(row["kind"]),
                "idempotency_key": str(row["idempotency_key"]),
                "status": str(row["status"]),
                "attempt_count": int(row["attempt_count"]),
                "available_at_utc": row["available_at_utc"],
                "lease_owner": row["lease_owner"],
                "lease_expires_at_utc": row["lease_expires_at_utc"],
                "last_error": row["last_error"],
                "result": result_obj,
                "created_at_utc": row["created_at_utc"],
                "updated_at_utc": row["updated_at_utc"],
            }

    def renew_lease(
        self, *, job_id: str, lease_owner: str, lease_seconds: int
    ) -> bool:
        now = self._utc_now()
        now_iso = now.isoformat()
        lease_expires_iso = (
            now + timedelta(seconds=max(5, int(lease_seconds)))
        ).isoformat()
        with self._connection() as conn:
            cur = conn.execute(
                """
                UPDATE durable_jobs
                SET lease_expires_at_utc = ?, updated_at_utc = ?
                WHERE job_id = ? AND lease_owner = ? AND status = 'processing'
                """,
                (lease_expires_iso, now_iso, job_id, lease_owner),
            )
            return cur.rowcount == 1

    def recover_expired_leases(self, *, max_retries: int = 6) -> int:
        now_iso = self._utc_now_iso()
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT job_id, attempt_count, lease_owner, lease_expires_at_utc
                FROM durable_jobs
                WHERE status = 'processing'
                  AND lease_expires_at_utc IS NOT NULL
                  AND lease_expires_at_utc < ?
                """,
                (now_iso,),
            ).fetchall()
            recovered = 0
            for row in rows:
                job_id = str(row["job_id"])
                attempt_count = int(row["attempt_count"]) + 1
                terminal = attempt_count >= max(1, int(max_retries))
                status = "failed_terminal" if terminal else "failed_retryable"
                # Expiry invites a liveness check; it does not prove that the
                # owner or its in-flight side effect has stopped.
                with self.execution_locks.try_job(job_id) as acquired:
                    if not acquired:
                        continue
                    cur = conn.execute(
                        """
                        UPDATE durable_jobs
                        SET status = ?,
                            attempt_count = ?,
                            available_at_utc = ?,
                            lease_owner = NULL,
                            lease_expires_at_utc = NULL,
                            last_error = ?,
                            updated_at_utc = ?
                        WHERE job_id = ?
                          AND status = 'processing'
                          AND lease_owner = ?
                          AND lease_expires_at_utc = ?
                        """,
                        (
                            status,
                            attempt_count,
                            now_iso,
                            "lease expired before completion",
                            now_iso,
                            job_id,
                            str(row["lease_owner"]),
                            str(row["lease_expires_at_utc"]),
                        ),
                    )
                    recovered += int(cur.rowcount or 0)
            return recovered

    def queue_counts(self, user_id: Optional[str] = None) -> dict[str, Any]:
        with self._connection() as conn:
            if user_id:
                rows = conn.execute(
                    """
                    SELECT status, COUNT(*) AS c
                    FROM durable_jobs
                    WHERE user_id = ?
                    GROUP BY status
                    """,
                    (user_id,),
                ).fetchall()
                oldest = conn.execute(
                    """
                    SELECT MIN(created_at_utc) AS oldest
                    FROM durable_jobs
                    WHERE status IN ('pending', 'failed_retryable', 'processing')
                      AND user_id = ?
                    """,
                    (user_id,),
                ).fetchone()
            else:
                rows = conn.execute(
                    """
                    SELECT status, COUNT(*) AS c
                    FROM durable_jobs
                    GROUP BY status
                    """
                ).fetchall()
                oldest = conn.execute(
                    """
                    SELECT MIN(created_at_utc) AS oldest
                    FROM durable_jobs
                    WHERE status IN ('pending', 'failed_retryable', 'processing')
                    """
                ).fetchone()
            counts = {str(r["status"]): int(r["c"]) for r in rows}
            oldest_pending_age_s = 0.0
            if oldest and oldest["oldest"]:
                try:
                    created = datetime.fromisoformat(str(oldest["oldest"]))
                    oldest_pending_age_s = max(
                        0.0, (self._utc_now() - created).total_seconds()
                    )
                except Exception:
                    oldest_pending_age_s = 0.0
            return {
                "pending": counts.get("pending", 0),
                "processing": counts.get("processing", 0),
                "done": counts.get("done", 0),
                "failed_retryable": counts.get("failed_retryable", 0),
                "failed_terminal": counts.get("failed_terminal", 0),
                "quarantined": counts.get("quarantined", 0),
                "oldest_pending_age_s": round(oldest_pending_age_s, 3),
            }

    def purge_completed_jobs(
        self, max_age_hours: int = 24, batch_size: int = 1000
    ) -> int:
        cutoff = (self._utc_now() - timedelta(hours=max_age_hours)).isoformat()
        limit = max(1, int(batch_size))
        with self._connection() as conn:
            cur = conn.execute(
                """
                DELETE FROM durable_jobs
                WHERE rowid IN (
                    SELECT rowid FROM durable_jobs
                    WHERE status IN ('done', 'failed_terminal')
                      AND updated_at_utc IS NOT NULL
                      AND updated_at_utc < ?
                    ORDER BY updated_at_utc ASC
                    LIMIT ?
                )
                """,
                (cutoff, limit),
            )
            conn.commit()
            return int(cur.rowcount or 0)

    def cancel_jobs_for_user(self, user_id: str) -> int:
        """Delete a user's unfinished jobs only after their effects are idle.

        Callers must revoke new submissions before invoking this operation.
        Quarantined jobs are included only because this is an explicit user
        cancellation; ordinary retention never deletes them.
        """
        active = ("pending", "failed_retryable", "processing", "quarantined")
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT job_id FROM durable_jobs WHERE user_id = ? AND status IN (?, ?, ?, ?) ORDER BY job_id",
                (user_id, *active),
            ).fetchall()
            expected = {str(row["job_id"]) for row in rows}
            with ExitStack() as locks:
                for job_id in sorted(expected):
                    if not locks.enter_context(self.execution_locks.try_job(job_id)):
                        raise RuntimeError("user_has_active_durable_job")
                conn.execute("BEGIN IMMEDIATE")
                try:
                    current = {
                        str(row["job_id"])
                        for row in conn.execute(
                            "SELECT job_id FROM durable_jobs WHERE user_id = ? AND status IN (?, ?, ?, ?)",
                            (user_id, *active),
                        ).fetchall()
                    }
                    if current != expected:
                        raise RuntimeError("user_durable_jobs_changed_during_cancellation")
                    cur = conn.execute(
                        "DELETE FROM durable_jobs WHERE user_id = ? AND status IN (?, ?, ?, ?)",
                        (user_id, *active),
                    )
                    conn.execute("COMMIT")
                    return int(cur.rowcount or 0)
                except BaseException:
                    conn.execute("ROLLBACK")
                    raise
