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
