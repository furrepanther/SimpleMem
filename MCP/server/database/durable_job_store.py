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
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Any, Optional


class DurableJobStore:
    """SQLite WAL durable queue with idempotent enqueue + leasing."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_lock = threading.Lock()
        self._local = threading.local()
        parent = os.path.dirname(db_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        self._init_db()

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

    def claim_batch(
        self,
        *,
        worker_id: str,
        batch_size: int,
        lease_seconds: int,
        max_jobs_per_user: int = 4,
    ) -> list[dict[str, Any]]:
        now = self._utc_now()
        now_iso = now.isoformat()
        lease_expires_iso = (
            now + timedelta(seconds=max(5, int(lease_seconds)))
        ).isoformat()
        take = max(1, int(batch_size))

        with self._connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                rows = conn.execute(
                    """
                    SELECT job_id, user_id, kind, idempotency_key, payload_json, status, attempt_count,
                           available_at_utc, created_at_utc, updated_at_utc
                    FROM (
                        SELECT job_id, user_id, kind, idempotency_key, payload_json, status,
                               attempt_count, available_at_utc, created_at_utc, updated_at_utc,
                               ROW_NUMBER() OVER (
                                   PARTITION BY user_id
                                   ORDER BY created_at_utc ASC, job_id ASC
                               ) AS user_rank
                        FROM durable_jobs
                        WHERE status IN ('pending', 'failed_retryable')
                          AND available_at_utc <= ?
                    )
                    WHERE user_rank <= ?
                    ORDER BY created_at_utc ASC, job_id ASC
                    LIMIT ?
                    """,
                    (now_iso, max(1, int(max_jobs_per_user)), take),
                ).fetchall()
                if not rows:
                    conn.execute("COMMIT")
                    return []

                out: list[dict[str, Any]] = []
                for row in rows:
                    job_id = str(row["job_id"])
                    conn.execute(
                        """
                        UPDATE durable_jobs
                        SET status = 'processing',
                            lease_owner = ?,
                            lease_expires_at_utc = ?,
                            updated_at_utc = ?
                        WHERE job_id = ?
                        """,
                        (worker_id, lease_expires_iso, now_iso, job_id),
                    )
                    payload_raw = str(row["payload_json"])
                    try:
                        payload = json.loads(payload_raw)
                    except json.JSONDecodeError:
                        payload = {"_corrupt_payload": True, "_raw": payload_raw}
                    out.append(
                        {
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
                        }
                    )
                conn.execute("COMMIT")
                return out
            except Exception:
                conn.execute("ROLLBACK")
                raise

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
        with self._connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                cur = conn.execute(
                    """
                    DELETE FROM durable_jobs
                    WHERE user_id = ?
                      AND status IN ('pending', 'failed_retryable', 'processing')
                    """,
                    (user_id,),
                )
                conn.execute("COMMIT")
                return int(cur.rowcount or 0)
            except Exception:
                conn.execute("ROLLBACK")
                raise
