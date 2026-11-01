"""Stable host-local durable-job exclusion with Linux OFD byte-range locks.

The queue and lock file must live on one verified local filesystem.  All
workers and recovery processes must participate in this protocol; never unlink
or replace the lock file while any process can use the queue.
"""

from __future__ import annotations

from contextlib import contextmanager
import ctypes
import errno
import fcntl
import hashlib
import os
from pathlib import Path
import platform
import sqlite3
import stat
import sys
from typing import Iterator
import uuid


class _Flock(ctypes.Structure):
    _fields_ = [
        ("l_type", ctypes.c_short),
        ("l_whence", ctypes.c_short),
        ("l_start", ctypes.c_longlong),
        ("l_len", ctypes.c_longlong),
        ("l_pid", ctypes.c_int),
    ]


def _check_abi() -> None:
    fields = ("l_type", "l_whence", "l_start", "l_len", "l_pid")
    if (
        sys.platform != "linux"
        or platform.machine() != "x86_64"
        or not hasattr(fcntl, "F_OFD_SETLK")
        or ctypes.sizeof(_Flock) != 32
        or tuple(getattr(_Flock, name).offset for name in fields) != (0, 2, 8, 16, 24)
    ):
        raise RuntimeError("durable job locks require the verified Linux x86_64 OFD ABI")


def _job_offset(job_id: str) -> int:
    if not job_id:
        raise ValueError("job_id is required for an execution lock")
    # Offset zero is reserved; collisions conservatively serialize different jobs.
    return 1 + int.from_bytes(hashlib.sha256(job_id.encode("utf-8")).digest()[:8], "big") % (2**63 - 2)


def _checked_file(fd: int, path: Path) -> os.stat_result:
    info = os.fstat(fd)
    named = os.stat(path, follow_symlinks=False)
    if (
        not stat.S_ISREG(info.st_mode)
        or info.st_nlink != 1
        or info.st_uid != os.geteuid()
        or stat.S_IMODE(info.st_mode) & 0o077
        or (info.st_dev, info.st_ino) != (named.st_dev, named.st_ino)
    ):
        raise RuntimeError("unsafe durable execution lock file")
    return info


class JobExecutionLocks:
    """Pin one lock inode to the queue database and try independent job locks."""

    def __init__(self, db_path: str, connection: sqlite3.Connection):
        _check_abi()
        database = Path(db_path).resolve(strict=True)
        db_info = database.stat()
        if not stat.S_ISREG(db_info.st_mode) or db_info.st_nlink != 1:
            raise RuntimeError("durable queue must be a regular file without hard links")
        self.path = database.with_name(database.name + ".execution-locks")
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT | os.O_CLOEXEC | os.O_NOFOLLOW, 0o600)
        try:
            info = _checked_file(fd, self.path)
            connection.execute(
                "CREATE TABLE IF NOT EXISTS durable_execution_lock_identity "
                "(singleton INTEGER PRIMARY KEY CHECK (singleton = 1), inode INTEGER NOT NULL, token TEXT NOT NULL)"
            )
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = connection.execute(
                    "SELECT inode, token FROM durable_execution_lock_identity WHERE singleton = 1"
                ).fetchone()
                token = os.pread(fd, 128, 0).decode("ascii")
                if row is None:
                    if not token:
                        token = uuid.uuid4().hex
                        os.pwrite(fd, token.encode("ascii"), 0)
                        os.fsync(fd)
                    if len(token) != 32 or any(ch not in "0123456789abcdef" for ch in token):
                        raise RuntimeError("invalid durable execution lock identity")
                    connection.execute(
                        "INSERT INTO durable_execution_lock_identity(singleton, inode, token) VALUES (1, ?, ?)",
                        (info.st_ino, token),
                    )
                elif int(row[0]) != info.st_ino or str(row[1]) != token:
                    raise RuntimeError("durable execution lock file was replaced")
                connection.execute("COMMIT")
            except BaseException:
                connection.execute("ROLLBACK")
                raise
            self.inode = info.st_ino
            self.token = token
        finally:
            os.close(fd)

    @contextmanager
    def try_job(self, job_id: str) -> Iterator[bool]:
        fd = os.open(self.path, os.O_RDWR | os.O_CLOEXEC | os.O_NOFOLLOW)
        try:
            info = _checked_file(fd, self.path)
            if info.st_ino != self.inode or os.pread(fd, 128, 0).decode("ascii") != self.token:
                raise RuntimeError("durable execution lock file was replaced")
            request = _Flock(fcntl.F_WRLCK, os.SEEK_SET, _job_offset(job_id), 1, 0)
            try:
                fcntl.fcntl(fd, fcntl.F_OFD_SETLK, bytes(request))
                acquired = True
            except OSError as exc:
                if exc.errno not in (errno.EACCES, errno.EAGAIN):
                    raise
                acquired = False
            yield acquired
        finally:
            os.close(fd)
