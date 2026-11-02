"""
Multi-tenant vector store for SimpleMem MCP Server
Uses LanceDB for vector storage with per-user table isolation
"""

import os
import asyncio
import json
import threading
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone

import lancedb
import pyarrow as pa

from ..auth.models import MemoryEntry


# LanceDB schema for memory entries
def get_memory_schema(embedding_dimension: int = 2560) -> pa.Schema:
    """Get PyArrow schema for memory entries"""
    return pa.schema(
        [
            pa.field("entry_id", pa.string()),
            pa.field("lossless_restatement", pa.string()),
            pa.field("keywords", pa.list_(pa.string())),
            pa.field("timestamp", pa.string()),
            pa.field("location", pa.string()),
            pa.field("persons", pa.list_(pa.string())),
            pa.field("entities", pa.list_(pa.string())),
            pa.field("topic", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), embedding_dimension)),
            pa.field("created_at", pa.string()),
            pa.field("metadata_json", pa.string()),
            pa.field("retention_purge_after_utc", pa.string()),
        ]
    )


class MultiTenantVectorStore:
    """
    Multi-tenant vector storage with per-user table isolation.
    Each user gets their own LanceDB table for complete data isolation.
    """

    def __init__(
        self,
        db_path: str = "./data/lancedb",
        embedding_dimension: int = 2560,
    ):
        self.db_path = db_path
        self.embedding_dimension = embedding_dimension
        os.makedirs(db_path, exist_ok=True)

        # Connect to LanceDB
        self.db = lancedb.connect(db_path)

        # Cache for opened tables
        self._tables: Dict[str, Any] = {}
        # Cache effective vector dimension per table (schema-derived when available)
        self._table_dimensions: Dict[str, int] = {}
        # Lance manifests are not safe for overlapping local mutations.  The
        # durable queue deliberately has multiple embedding workers, so all
        # physical writes for one table must converge through this lock.
        self._table_write_locks: Dict[str, threading.RLock] = {}
        self._table_write_locks_guard = threading.Lock()

    def _table_write_lock(self, table_name: str) -> threading.RLock:
        with self._table_write_locks_guard:
            lock = self._table_write_locks.get(table_name)
            if lock is None:
                lock = threading.RLock()
                self._table_write_locks[table_name] = lock
            return lock

    def _infer_table_dimension(self, table: Any) -> int:
        """Infer vector dimension from table schema; fall back to configured dimension."""
        try:
            schema = table.schema
            if hasattr(schema, "names") and "vector" in schema.names:
                vector_field = schema.field("vector")
                vector_type = getattr(vector_field, "type", None)
                size = getattr(vector_type, "list_size", None)
                if isinstance(size, int) and size > 0:
                    return size
        except Exception:
            pass
        return self.embedding_dimension

    def _get_table_dimension(self, table_name: str, table: Any) -> int:
        """Get cached dimension for table, deriving from schema if needed."""
        if table_name in self._table_dimensions:
            return self._table_dimensions[table_name]
        dim = self._infer_table_dimension(table)
        self._table_dimensions[table_name] = dim
        return dim

    @staticmethod
    def _adapt_vector(vector: List[float], target_dim: int) -> List[float]:
        """
        Adapt vector length to table schema:
        - truncate if source is longer
        - zero-pad if source is shorter
        """
        if target_dim <= 0:
            return [float(v) for v in vector]
        out = [float(v) for v in vector[:target_dim]]
        missing = target_dim - len(out)
        if missing > 0:
            out.extend([0.0] * missing)
        return out

    @staticmethod
    def _list_or_empty(value: Any) -> List[Any]:
        """Safely normalize DB row values that may be arrays/lists/None."""
        if value is None:
            return []
        try:
            return list(value)
        except Exception:
            return []

    @staticmethod
    def _escape_filter_value(value: str) -> str:
        """Escape a value for a LanceDB SQL-style string filter."""
        return str(value).replace("'", "''")

    @staticmethod
    def _text_or_empty(value: Any) -> str:
        if value is None:
            return ""
        return str(value)

    @staticmethod
    def _row_value(row: Any, key: str, default: Any = None) -> Any:
        getter = getattr(row, "get", None)
        if callable(getter):
            return getter(key, default)
        try:
            return row[key]
        except (KeyError, TypeError, IndexError):
            return default

    @staticmethod
    def _metadata_from_json(value: Any) -> dict[str, Any]:
        if not isinstance(value, str) or not value.strip():
            return {}
        try:
            decoded = json.loads(value)
        except (TypeError, ValueError):
            return {}
        return dict(decoded) if isinstance(decoded, dict) else {}

    @staticmethod
    def _metadata_to_json(value: Any) -> str:
        if value is None:
            value = {}
        if not isinstance(value, dict):
            raise ValueError("memory entry metadata must be an object")
        try:
            return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        except (TypeError, ValueError) as exc:
            raise ValueError("memory entry metadata must be JSON serializable") from exc

    @staticmethod
    def _canonical_utc(value: Any) -> str:
        raw = str(value or "").strip()
        if not raw:
            return ""
        try:
            parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            return ""
        if parsed.tzinfo is None:
            return ""
        return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")

    @classmethod
    def _temporary_review_purge_after(cls, metadata: Any) -> str | None:
        """Return an expiry only for explicitly non-canonical review evidence."""
        if not isinstance(metadata, dict):
            return None
        if (
            metadata.get("temporary") is not True
            or metadata.get("canonical") is not False
            or metadata.get("retention_verified") is not True
            or metadata.get("record_kind") != "temporary_review_evidence"
        ):
            return None
        expiry = cls._canonical_utc(metadata.get("purge_after_utc"))
        return expiry or None

    def _row_to_entry(self, row: Any) -> MemoryEntry:
        return MemoryEntry(
            entry_id=str(row["entry_id"]),
            lossless_restatement=str(row["lossless_restatement"] or ""),
            keywords=self._list_or_empty(row["keywords"]),
            timestamp=row["timestamp"] if row["timestamp"] else None,
            location=row["location"] if row["location"] else None,
            persons=self._list_or_empty(row["persons"]),
            entities=self._list_or_empty(row["entities"]),
            topic=row["topic"] if row["topic"] else None,
            metadata=self._metadata_from_json(self._row_value(row, "metadata_json")),
        )

    def _row_to_record(
        self,
        row: Any,
        *,
        table_dim: int,
        updates: dict[str, Any] | None = None,
        embedding: List[float] | None = None,
    ) -> dict[str, Any]:
        updates = updates or {}
        vector = (
            self._adapt_vector(embedding, table_dim)
            if embedding is not None
            else self._adapt_vector(self._list_or_empty(row["vector"]), table_dim)
        )
        return {
            "entry_id": str(row["entry_id"]),
            "lossless_restatement": self._text_or_empty(
                updates.get("lossless_restatement", row["lossless_restatement"] or "")
            ),
            "keywords": self._list_or_empty(updates.get("keywords", row["keywords"])),
            "timestamp": self._text_or_empty(
                updates.get("timestamp", row["timestamp"] or "")
            ),
            "location": self._text_or_empty(
                updates.get("location", row["location"] or "")
            ),
            "persons": self._list_or_empty(updates.get("persons", row["persons"])),
            "entities": self._list_or_empty(updates.get("entities", row["entities"])),
            "topic": self._text_or_empty(updates.get("topic", row["topic"] or "")),
            "vector": vector,
            "created_at": self._text_or_empty(row["created_at"] or ""),
            "metadata_json": self._text_or_empty(
                self._row_value(row, "metadata_json", "")
            ),
            "retention_purge_after_utc": self._text_or_empty(
                self._row_value(row, "retention_purge_after_utc", "")
            ),
        }

    def _ensure_metadata_columns(self, table_name: str, table: Any) -> Any:
        """Apply only the additive metadata migration required by this release."""
        schema = getattr(table, "schema", None)
        names = getattr(schema, "names", None)
        if names is None:
            # Lightweight test doubles need not emulate the complete Lance API.
            return table
        missing = [
            field
            for field in (
                pa.field("metadata_json", pa.string()),
                pa.field("retention_purge_after_utc", pa.string()),
            )
            if field.name not in names
        ]
        if not missing:
            return table
        add_columns = getattr(table, "add_columns", None)
        if not callable(add_columns):
            raise RuntimeError(
                "LanceDB table does not support required additive metadata migration"
            )
        try:
            add_columns(missing)
            migrated = self.db.open_table(table_name)
        except Exception as exc:
            raise RuntimeError(
                f"LanceDB metadata migration failed for table {table_name}"
            ) from exc
        self._tables[table_name] = migrated
        self._table_dimensions.pop(table_name, None)
        return migrated

    def _get_table(self, table_name: str) -> Any:
        """Get or create a user's table"""
        with self._table_write_lock(table_name):
            if table_name not in self._tables:
                if table_name in self.table_names():
                    self._tables[table_name] = self.db.open_table(table_name)
                else:
                    # Create new table with schema
                    schema = get_memory_schema(self.embedding_dimension)
                    self._tables[table_name] = self.db.create_table(
                        table_name,
                        schema=schema,
                    )
            self._tables[table_name] = self._ensure_metadata_columns(
                table_name, self._tables[table_name]
            )
            self._table_dimensions[table_name] = self._infer_table_dimension(
                self._tables[table_name]
            )
            return self._tables[table_name]

    def _append_records(self, table_name: str, records: List[dict[str, Any]]) -> None:
        """Append records under the table's single local Lance mutation lane."""
        with self._table_write_lock(table_name):
            table = self._get_table(table_name)
            merge_insert = getattr(table, "merge_insert", None)
            if callable(merge_insert):
                # Durable jobs may be replayed after a lease loss.  Make the
                # physical mutation idempotent on the stable entry identifier.
                (
                    merge_insert("entry_id")
                    .when_not_matched_insert_all()
                    .execute(records)
                )
            else:
                raise RuntimeError(
                    "LanceDB table does not support required merge_insert idempotency"
                )

    def _update_records(
        self, table_name: str, where: str, values: dict[str, Any]
    ) -> None:
        """Update records under the table's single local Lance mutation lane."""
        with self._table_write_lock(table_name):
            self._get_table(table_name).update(where=where, values=values)

    def _delete_records(self, table_name: str, where: str) -> None:
        """Delete records under the table's single local Lance mutation lane."""
        with self._table_write_lock(table_name):
            self._get_table(table_name).delete(where)

    def table_names(self) -> List[str]:
        """
        Return current LanceDB table names.

        LanceDB has renamed this API over time, so we keep a small
        compatibility shim here and prefer the newer list_tables() when
        available.
        """
        try:
            list_tables = getattr(self.db, "list_tables", None)
            if callable(list_tables):
                table_listing = list_tables()
                table_names = getattr(table_listing, "tables", None)
                if table_names is not None:
                    return [str(name) for name in table_names]
                if isinstance(table_listing, dict) and "tables" in table_listing:
                    return [str(name) for name in table_listing["tables"]]
                return [str(name) for name in list(table_listing)]
            return [str(name) for name in self.db.table_names()]
        except Exception as exc:
            raise RuntimeError("LanceDB table listing failed") from exc

    async def add_entries(
        self,
        table_name: str,
        entries: List[MemoryEntry],
        embeddings: List[List[float]],
    ) -> int:
        """
        Add memory entries to a user's table

        Args:
            table_name: User's table name
            entries: List of MemoryEntry objects
            embeddings: List of embedding vectors

        Returns:
            Number of entries added
        """
        if len(entries) != len(embeddings):
            raise ValueError("Number of entries must match number of embeddings")

        if not entries:
            return 0

        table = await asyncio.to_thread(self._get_table, table_name)
        table_dim = self._get_table_dimension(table_name, table)
        created_at = datetime.now(timezone.utc).isoformat()

        # Build records
        records = []
        for entry, embedding in zip(entries, embeddings):
            metadata = getattr(entry, "metadata", {})
            records.append(
                {
                    "entry_id": entry.entry_id,
                    "lossless_restatement": entry.lossless_restatement,
                    "keywords": entry.keywords or [],
                    "timestamp": entry.timestamp or "",
                    "location": entry.location or "",
                    "persons": entry.persons or [],
                    "entities": entry.entities or [],
                    "topic": entry.topic or "",
                    "vector": self._adapt_vector(embedding, table_dim),
                    "created_at": created_at,
                    "metadata_json": self._metadata_to_json(metadata),
                    "retention_purge_after_utc": self._temporary_review_purge_after(
                        metadata
                    ),
                }
            )

        # A durable worker may finish embedding concurrently with another
        # worker.  Serialize the manifest-changing append, not the GPU work.
        await asyncio.to_thread(self._append_records, table_name, records)
        return len(records)

    async def semantic_search(
        self,
        table_name: str,
        query_embedding: List[float],
        top_k: int = 25,
    ) -> List[MemoryEntry]:
        """
        Perform semantic search using vector similarity

        Args:
            table_name: User's table name
            query_embedding: Query embedding vector
            top_k: Number of results to return

        Returns:
            List of matching MemoryEntry objects
        """
        table = await asyncio.to_thread(self._get_table, table_name)
        table_dim = self._get_table_dimension(table_name, table)

        try:
            # Check if table has data
            if await asyncio.to_thread(table.count_rows) == 0:
                return []
            query_vec = self._adapt_vector(query_embedding, table_dim)
            results = await asyncio.to_thread(
                lambda: table.search(query_vec).limit(top_k).to_pandas()
            )

            return [self._row_to_entry(row) for _, row in results.iterrows()]

        except Exception as e:
            print(f"Semantic search error: {e}")
            return []

    async def keyword_search(
        self,
        table_name: str,
        keywords: List[str],
        top_k: int = 5,
    ) -> List[MemoryEntry]:
        """
        Perform keyword-based search (BM25-style matching)

        Args:
            table_name: User's table name
            keywords: List of keywords to match
            top_k: Number of results to return

        Returns:
            List of matching MemoryEntry objects
        """
        table = await asyncio.to_thread(self._get_table, table_name)

        try:
            if await asyncio.to_thread(table.count_rows) == 0:
                return []

            # Load all entries for keyword matching
            df = await asyncio.to_thread(table.to_pandas)

            # Score each entry
            scores = []
            for idx, row in df.iterrows():
                score = 0
                row_keywords = self._list_or_empty(row["keywords"])
                entry_keywords = set(str(k).lower() for k in row_keywords)
                entry_text = row["lossless_restatement"].lower()

                for kw in keywords:
                    kw_lower = kw.lower()
                    # Keyword list match: 2 points
                    if kw_lower in entry_keywords:
                        score += 2
                    # Text match: 1 point
                    if kw_lower in entry_text:
                        score += 1

                scores.append((idx, score))

            # Sort by score and get top-k
            scores.sort(key=lambda x: x[1], reverse=True)
            top_indices = [idx for idx, score in scores[:top_k] if score > 0]

            return [self._row_to_entry(df.iloc[idx]) for idx in top_indices]

        except Exception as e:
            print(f"Keyword search error: {e}")
            return []

    async def structured_search(
        self,
        table_name: str,
        persons: Optional[List[str]] = None,
        location: Optional[str] = None,
        entities: Optional[List[str]] = None,
        timestamp_start: Optional[str] = None,
        timestamp_end: Optional[str] = None,
        top_k: int = 5,
    ) -> List[MemoryEntry]:
        """
        Perform structured/metadata-based search

        Args:
            table_name: User's table name
            persons: Filter by person names
            location: Filter by location
            entities: Filter by entities
            timestamp_start: Start of timestamp range
            timestamp_end: End of timestamp range
            top_k: Number of results to return

        Returns:
            List of matching MemoryEntry objects
        """
        table = await asyncio.to_thread(self._get_table, table_name)

        try:
            if await asyncio.to_thread(table.count_rows) == 0:
                return []

            df = await asyncio.to_thread(table.to_pandas)

            # Apply filters
            mask = [True] * len(df)

            if persons:
                persons_lower = set(p.lower() for p in persons)
                for i, row in df.iterrows():
                    row_persons = set(
                        str(p).lower() for p in self._list_or_empty(row["persons"])
                    )
                    if not persons_lower.intersection(row_persons):
                        mask[i] = False

            if location:
                location_lower = location.lower()
                for i, row in df.iterrows():
                    if mask[i] and row["location"]:
                        if location_lower not in row["location"].lower():
                            mask[i] = False
                    elif mask[i]:
                        mask[i] = False

            if entities:
                entities_lower = set(e.lower() for e in entities)
                for i, row in df.iterrows():
                    if mask[i]:
                        row_entities = set(
                            str(e).lower() for e in self._list_or_empty(row["entities"])
                        )
                        if not entities_lower.intersection(row_entities):
                            mask[i] = False

            if timestamp_start:
                for i, row in df.iterrows():
                    if mask[i] and row["timestamp"]:
                        if row["timestamp"] < timestamp_start:
                            mask[i] = False

            if timestamp_end:
                for i, row in df.iterrows():
                    if mask[i] and row["timestamp"]:
                        if row["timestamp"] > timestamp_end:
                            mask[i] = False

            # Get filtered results
            filtered_df = df[[m for m in mask]][:top_k]

            return [self._row_to_entry(row) for _, row in filtered_df.iterrows()]

        except Exception as e:
            print(f"Structured search error: {e}")
            return []

    async def get_all_entries(self, table_name: str) -> List[MemoryEntry]:
        """Get all entries from a user's table"""
        table = await asyncio.to_thread(self._get_table, table_name)

        try:
            if await asyncio.to_thread(table.count_rows) == 0:
                return []

            df = await asyncio.to_thread(table.to_pandas)
            return [self._row_to_entry(row) for _, row in df.iterrows()]

        except Exception as e:
            print(f"Get all entries error: {e}")
            return []

    async def get_entry(self, table_name: str, entry_id: str) -> MemoryEntry | None:
        """Get a single entry by ID from a user's table."""
        entry_id = str(entry_id or "").strip()
        if not entry_id:
            return None
        table = await asyncio.to_thread(self._get_table, table_name)
        try:
            if await asyncio.to_thread(table.count_rows) == 0:
                return None
            where = f"entry_id = '{self._escape_filter_value(entry_id)}'"
            matches = await asyncio.to_thread(
                lambda: table.search().where(where).limit(1).to_pandas()
            )
            if matches.empty:
                return None
            return self._row_to_entry(matches.iloc[0])
        except Exception as e:
            print(f"Get entry error: {e}")
            return None

    def _purge_expired_temporary_review_entries(
        self,
        table_name: str,
        *,
        now_utc: str,
        limit: int,
    ) -> dict[str, Any]:
        """Delete only expired, verified, non-canonical review evidence by ID."""
        cutoff = self._canonical_utc(now_utc)
        if not cutoff:
            raise ValueError("temporary evidence cleanup requires timezone-aware now_utc")
        cap = max(1, min(256, int(limit)))
        with self._table_write_lock(table_name):
            table = self._get_table(table_name)
            where = (
                "retention_purge_after_utc > '' AND retention_purge_after_utc <= "
                f"'{self._escape_filter_value(cutoff)}'"
            )
            try:
                matches = table.search().where(where).limit(cap).to_pandas()
            except Exception as exc:
                raise RuntimeError(
                    f"LanceDB temporary-evidence retention query failed for table {table_name}"
                ) from exc

            receipt: dict[str, Any] = {
                "table_name": table_name,
                "cutoff_utc": cutoff,
                "selected": int(len(matches.index)),
                "deleted": [],
                "skipped": [],
            }
            for _, row in matches.iterrows():
                entry_id = str(self._row_value(row, "entry_id", "") or "").strip()
                metadata = self._metadata_from_json(
                    self._row_value(row, "metadata_json")
                )
                stored_expiry = self._canonical_utc(
                    self._row_value(row, "retention_purge_after_utc")
                )
                proven_expiry = self._temporary_review_purge_after(metadata)
                if (
                    not entry_id
                    or not stored_expiry
                    or proven_expiry != stored_expiry
                    or stored_expiry > cutoff
                ):
                    receipt["skipped"].append(
                        {"entry_id": entry_id, "reason": "retention_proof_mismatch"}
                    )
                    continue
                entry_where = f"entry_id = '{self._escape_filter_value(entry_id)}'"
                try:
                    table.delete(entry_where)
                    remaining = table.search().where(entry_where).limit(1).to_pandas()
                except Exception as exc:
                    receipt["skipped"].append(
                        {"entry_id": entry_id, "reason": f"delete_failed:{type(exc).__name__}"}
                    )
                    continue
                if not remaining.empty:
                    receipt["skipped"].append(
                        {"entry_id": entry_id, "reason": "delete_not_confirmed"}
                    )
                    continue
                receipt["deleted"].append(
                    {"entry_id": entry_id, "purge_after_utc": stored_expiry}
                )
            return receipt

    async def purge_expired_temporary_review_entries(
        self,
        table_name: str,
        *,
        now_utc: str | None = None,
        limit: int = 32,
    ) -> dict[str, Any]:
        """Run a bounded exact-ID cleanup without materializing the full table."""
        if now_utc is None:
            now_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        return await asyncio.to_thread(
            self._purge_expired_temporary_review_entries,
            table_name,
            now_utc=now_utc,
            limit=limit,
        )

    async def delete_entry(self, table_name: str, entry_id: str) -> dict[str, Any]:
        """Delete one entry by ID from a user's table."""
        entry_id = str(entry_id or "").strip()
        if not entry_id:
            return {"found": False, "deleted": False, "error": "entry_id_required"}
        table = await asyncio.to_thread(self._get_table, table_name)
        try:
            if await asyncio.to_thread(table.count_rows) == 0:
                return {"found": False, "deleted": False, "entry_id": entry_id}
            existing = await self.get_entry(table_name, entry_id)
            if existing is None:
                return {"found": False, "deleted": False, "entry_id": entry_id}
            where = f"entry_id = '{self._escape_filter_value(entry_id)}'"
            await asyncio.to_thread(self._delete_records, table_name, where)
            deleted = await self.get_entry(table_name, entry_id) is None
            return {"found": True, "deleted": deleted, "entry_id": entry_id}
        except Exception as e:
            print(f"Delete entry error: {e}")
            return {
                "found": True,
                "deleted": False,
                "entry_id": entry_id,
                "error": str(e),
            }

    async def update_entry(
        self,
        table_name: str,
        entry_id: str,
        updates: dict[str, Any],
        *,
        embedding: List[float] | None = None,
    ) -> dict[str, Any]:
        """Update one entry by ID in a user's table."""
        entry_id = str(entry_id or "").strip()
        if not entry_id:
            return {"found": False, "updated": False, "error": "entry_id_required"}
        allowed = {
            "lossless_restatement",
            "keywords",
            "timestamp",
            "location",
            "persons",
            "entities",
            "topic",
        }
        updates = {k: v for k, v in dict(updates or {}).items() if k in allowed}
        if not updates and embedding is None:
            return {"found": False, "updated": False, "error": "updates_required"}

        table = await asyncio.to_thread(self._get_table, table_name)
        try:
            if await asyncio.to_thread(table.count_rows) == 0:
                return {"found": False, "updated": False, "entry_id": entry_id}
            df = await asyncio.to_thread(table.to_pandas)
            matches = df[df["entry_id"] == entry_id]
            if matches.empty:
                return {"found": False, "updated": False, "entry_id": entry_id}

            table_dim = self._get_table_dimension(table_name, table)
            row = matches.iloc[0]
            record = self._row_to_record(
                row,
                table_dim=table_dim,
                updates=updates,
                embedding=embedding,
            )
            where = f"entry_id = '{self._escape_filter_value(entry_id)}'"

            update_values = {k: record[k] for k in allowed if k in updates}
            if embedding is not None:
                update_values["vector"] = record["vector"]
            try:
                await asyncio.to_thread(
                    self._update_records, table_name, where, update_values
                )
                updated = await self.get_entry(table_name, entry_id)
                if updated is not None:
                    return {
                        "found": True,
                        "updated": True,
                        "entry_id": entry_id,
                        "method": "update",
                    }
            except Exception:
                return {
                    "found": True,
                    "updated": False,
                    "entry_id": entry_id,
                    "error": "table_update_failed",
                }
            return {
                "found": True,
                "updated": False,
                "entry_id": entry_id,
                "error": "table_update_not_applied",
            }
        except Exception as e:
            print(f"Update entry error: {e}")
            return {
                "found": True,
                "updated": False,
                "entry_id": entry_id,
                "error": str(e),
            }

    async def count_entries(self, table_name: str) -> int:
        """Count entries in a user's table"""
        table = await asyncio.to_thread(self._get_table, table_name)
        try:
            return int(await asyncio.to_thread(table.count_rows))
        except Exception:
            return 0

    async def clear_table(self, table_name: str) -> bool:
        """Clear all entries from a user's table"""
        try:
            if table_name in self._tables:
                del self._tables[table_name]

            if table_name in self.table_names():
                self.db.drop_table(table_name)

            # Recreate empty table
            self._get_table(table_name)
            return True

        except Exception as e:
            print(f"Clear table error: {e}")
            return False

    async def delete_table(self, table_name: str) -> bool:
        """Completely delete a user's table"""
        try:
            if table_name in self._tables:
                del self._tables[table_name]

            if table_name in self.table_names():
                self.db.drop_table(table_name)

            return True

        except Exception as e:
            print(f"Delete table error: {e}")
            return False

    def get_stats(self, table_name: str) -> Dict[str, Any]:
        """Get statistics for a user's table"""
        try:
            table = self._get_table(table_name)
            count = table.count_rows()

            return {
                "table_name": table_name,
                "entry_count": count,
                "embedding_dimension": self.embedding_dimension,
            }
        except Exception as e:
            return {
                "table_name": table_name,
                "entry_count": 0,
                "embedding_dimension": self.embedding_dimension,
                "error": str(e),
            }
