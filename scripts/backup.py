#!/usr/bin/env python3
"""SimpleMem backup utility."""

import argparse
import os
import shutil
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path


def backup_sqlite(db_path: Path, backup_dir: Path) -> Path:
    backup_path = (
        backup_dir
        / f"{db_path.stem}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.db"
    )
    backup_dir.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    backup_conn = sqlite3.connect(str(backup_path))
    conn.backup(backup_conn)
    backup_conn.close()
    conn.close()
    return backup_path


def cleanup_old_backups(backup_dir: Path, max_age_days: int = 7):
    cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)
    for f in backup_dir.iterdir():
        if f.is_file() and f.stat().st_mtime < cutoff.timestamp():
            f.unlink()


def main():
    parser = argparse.ArgumentParser(description="SimpleMem backup utility")
    parser.add_argument("--data-dir", default="./data", help="Data directory")
    parser.add_argument("--backup-dir", default="./backups", help="Backup directory")
    parser.add_argument(
        "--max-age-days", type=int, default=7, help="Delete backups older than N days"
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    backup_dir = Path(args.backup_dir) / datetime.now(timezone.utc).strftime("%Y%m%d")

    backed_up = []
    for db_file in data_dir.rglob("*.db"):
        path = backup_sqlite(db_file, backup_dir)
        backed_up.append(str(path))
        print(f"Backed up: {db_file} -> {path}")

    for lancedb_dir in data_dir.rglob("lancedb"):
        if lancedb_dir.is_dir():
            dest = backup_dir / lancedb_dir.name
            shutil.copytree(lancedb_dir, dest, dirs_exist_ok=True)
            backed_up.append(str(dest))
            print(f"Backed up: {lancedb_dir} -> {dest}")

    cleanup_old_backups(Path(args.backup_dir), args.max_age_days)
    print(f"Backup complete: {len(backed_up)} items")


if __name__ == "__main__":
    main()
