"""
MCP Protocol Handler - JSON-RPC 2.0 over SSE.

Implements the Model Context Protocol for remote clients.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import time
import uuid
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

from .auth.models import User, MemoryEntry
from .core.answer_generator import AnswerGenerator
from .core.memory_builder import MemoryBuilder
from .core.retriever import Retriever
from .database.durable_job_store import DurableJobStore
from .database.vector_store import MultiTenantVectorStore

# Type alias for client manager (OpenRouter-compatible client manager)
ClientManager = object  # Duck-typed: OpenRouter-compatible client manager


@dataclass
class JsonRpcRequest:
    jsonrpc: str
    method: str
    id: Optional[int | str]
    params: Optional[dict] = None


@dataclass
class JsonRpcResponse:
    jsonrpc: str = "2.0"
    id: Optional[int | str] = None
    result: Optional[Any] = None
    error: Optional[dict] = None

    def to_dict(self):
        d = {"jsonrpc": self.jsonrpc, "id": self.id}
        if self.error:
            d["error"] = self.error
        else:
            d["result"] = self.result
        return d


MCP_VERSION = "2025-03-26"
SERVER_NAME = "simplemem"
SERVER_VERSION = "1.0.0"


class MCPHandler:
    """Handles MCP protocol messages for a specific user session."""

    _runtime_lock: Optional[asyncio.Lock] = None
    _read_gate_lock: Optional[asyncio.Lock] = None
    _active_read_ops: int = 0
    _read_rejects: int = 0

    _durable_store: Optional[DurableJobStore] = None
    _shared_settings: Optional[Any] = None
    _shared_vector_store: Optional[MultiTenantVectorStore] = None
    _shared_client_manager: Optional[ClientManager] = None
    _shared_fastpath_client: Optional[Any] = None
    _shared_user_store: Optional[Any] = None
    _shared_token_manager: Optional[Any] = None
    _worker_shutdown: Optional[asyncio.Event] = None
    _worker_tasks: list[asyncio.Task] = []
    _lease_recoveries: int = 0
    _retrieve_cache: dict[str, tuple[float, dict[str, Any]]] = {}
    _retrieve_cache_order: deque[str] = deque()
    _MAX_INFLIGHT = 1000
    _INFLIGHT_TTL_S = 300.0
    _retrieve_inflight: dict[str, asyncio.Future] = {}
    _retrieve_inflight_ts: dict[str, float] = {}
    _retrieve_negative_cache: dict[str, tuple[float, dict[str, Any]]] = {}
    _retrieve_negative_cache_order: deque[str] = deque()
    _query_cache: dict[str, tuple[float, dict[str, Any]]] = {}
    _query_cache_order: deque[str] = deque()
    _query_inflight: dict[str, asyncio.Future] = {}
    _query_inflight_ts: dict[str, float] = {}

    _write_stats: dict[str, int] = {
        "accepted": 0,
        "completed": 0,
        "failed": 0,
        "retryable_failures": 0,
        "rejected_admission": 0,
    }
    _write_status_by_job_id: dict[str, dict[str, Any]] = {}
    _write_status_order: deque[str] = deque()

    @classmethod
    def _add_inflight(
        cls,
        inflight_dict: dict[str, asyncio.Future],
        ts_dict: dict[str, float],
        key: str,
        future: asyncio.Future,
    ) -> None:
        now = time.monotonic()
        expired = [k for k, t in ts_dict.items() if now - t > cls._INFLIGHT_TTL_S]
        for k in expired:
            inflight_dict.pop(k, None)
            ts_dict.pop(k, None)
        if len(inflight_dict) >= cls._MAX_INFLIGHT:
            oldest_key = next(iter(inflight_dict))
            inflight_dict.pop(oldest_key, None)
            ts_dict.pop(oldest_key, None)
        inflight_dict[key] = future
        ts_dict[key] = now

    @classmethod
    def _remove_inflight(
        cls,
        inflight_dict: dict[str, asyncio.Future],
        ts_dict: dict[str, float],
        key: str,
        owner_future: asyncio.Future,
    ) -> None:
        current = inflight_dict.get(key)
        if current is owner_future:
            inflight_dict.pop(key, None)
            ts_dict.pop(key, None)

    def __init__(
        self,
        user: User,
        api_key: str,
        vector_store: MultiTenantVectorStore,
        client_manager: ClientManager,
        settings: Any,
    ):
        self.user = user
        self.api_key = api_key
        self.vector_store = vector_store
        self.client_manager = client_manager
        self.settings = settings
        self.initialized = False

        self._retriever: Optional[Retriever] = None
        self._answer_generator: Optional[AnswerGenerator] = None

        # Ensure class-level shared runtime is initialized once.
        if MCPHandler._shared_vector_store is None:
            MCPHandler._shared_vector_store = vector_store
        if MCPHandler._shared_client_manager is None:
            MCPHandler._shared_client_manager = client_manager
        if MCPHandler._shared_settings is None:
            MCPHandler._shared_settings = settings
        if MCPHandler._durable_store is None:
            MCPHandler._durable_store = DurableJobStore(settings.durable_queue_db_path)
            MCPHandler._lease_recoveries += (
                MCPHandler._durable_store.recover_expired_leases()
            )

    @classmethod
    async def start_background_workers(
        cls,
        *,
        settings: Any,
        vector_store: MultiTenantVectorStore,
        client_manager: ClientManager,
        user_store: Any = None,
        token_manager: Any = None,
    ) -> None:
        if cls._runtime_lock is None:
            cls._runtime_lock = asyncio.Lock()
        async with cls._runtime_lock:
            cls._shared_settings = settings
            cls._shared_vector_store = vector_store
            cls._shared_client_manager = client_manager
            if user_store is not None:
                cls._shared_user_store = user_store
            if token_manager is not None:
                cls._shared_token_manager = token_manager
            if cls._durable_store is None:
                cls._durable_store = DurableJobStore(settings.durable_queue_db_path)
            cls._lease_recoveries += cls._durable_store.recover_expired_leases()
            if cls._worker_shutdown is None:
                cls._worker_shutdown = asyncio.Event()
            cls._worker_shutdown.clear()
            target = max(1, int(settings.memory_write_workers))
            while len(cls._worker_tasks) < target:
                idx = len(cls._worker_tasks) + 1
                task = asyncio.create_task(
                    cls._durable_worker_loop(idx),
                    name=f"simplemem-durable-worker-{idx}",
                )
                cls._worker_tasks.append(task)

    @classmethod
    async def stop_background_workers(cls) -> None:
        if cls._runtime_lock is None:
            cls._runtime_lock = asyncio.Lock()
        async with cls._runtime_lock:
            if cls._worker_shutdown is not None:
                cls._worker_shutdown.set()
            if cls._shared_fastpath_client is not None:
                try:
                    await cls._shared_fastpath_client.aclose()
                except Exception:
                    pass
                cls._shared_fastpath_client = None
            for task in cls._worker_tasks:
                task.cancel()
            if cls._worker_tasks:
                await asyncio.gather(*cls._worker_tasks, return_exceptions=True)
            cls._worker_tasks = []

    @classmethod
    async def _durable_worker_loop(cls, worker_idx: int) -> None:
        worker_id = f"w{worker_idx}-{uuid.uuid4().hex[:8]}"
        while True:
            if cls._worker_shutdown is not None and cls._worker_shutdown.is_set():
                return
            settings = cls._shared_settings
            store = cls._durable_store
            if settings is None or store is None:
                await asyncio.sleep(0.2)
                continue

            try:
                jobs = store.claim_batch(
                    worker_id=worker_id,
                    batch_size=max(1, int(settings.memory_write_batch_max)),
                    lease_seconds=max(10, int(settings.memory_write_lease_seconds)),
                )
                if not jobs:
                    await asyncio.sleep(
                        max(
                            0.005, int(settings.memory_write_batch_max_wait_ms) / 1000.0
                        )
                    )
                    continue

                for job in jobs:
                    job_id = str(job["job_id"])
                    cls._write_status_by_job_id[job_id] = {
                        "job_id": job_id,
                        "kind": str(job["kind"]),
                        "status": "processing",
                        "started_at_utc": datetime.now(timezone.utc).isoformat(),
                        "worker": worker_id,
                    }
                    cls._write_status_order.append(job_id)
                    cls._trim_write_status(
                        getattr(settings, "memory_write_job_status_limit", 2000)
                    )
                    try:
                        result = await cls._execute_durable_job(job)
                        store.mark_done(job_id=job_id, result=result)
                        cls._write_stats["completed"] += 1
                        cls._write_status_by_job_id[job_id] = {
                            "job_id": job_id,
                            "kind": str(job["kind"]),
                            "status": "done",
                            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                            "worker": worker_id,
                            "result": result,
                        }
                    except Exception as exc:
                        status = store.mark_failed(
                            job_id=job_id,
                            error=str(exc),
                            retryable=True,
                            max_retries=max(1, int(settings.memory_write_max_retries)),
                            backoff_base_ms=max(
                                50, int(settings.memory_write_retry_backoff_base_ms)
                            ),
                        )
                        if status.get("status") == "failed_retryable":
                            cls._write_stats["retryable_failures"] += 1
                        else:
                            cls._write_stats["failed"] += 1
                        cls._write_status_by_job_id[job_id] = {
                            "job_id": job_id,
                            "kind": str(job["kind"]),
                            "status": status.get("status", "failed_terminal"),
                            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                            "worker": worker_id,
                            "error": str(exc),
                            "attempt_count": status.get("attempt_count"),
                        }
            except asyncio.CancelledError:
                return
            except Exception as _worker_err:
                logging.getLogger(__name__).warning(
                    "durable_worker_loop_error: %s", _worker_err, exc_info=True
                )
                await asyncio.sleep(0.2)

    @classmethod
    async def _execute_durable_job(cls, job: dict[str, Any]) -> dict[str, Any]:
        settings = cls._shared_settings
        vector_store = cls._shared_vector_store
        client_manager = cls._shared_client_manager
        if settings is None or vector_store is None or client_manager is None:
            raise RuntimeError("durable_worker_not_initialized")

        payload = job.get("payload", {})
        if not isinstance(payload, dict):
            raise RuntimeError("invalid_payload")
        kind = str(job.get("kind", ""))
        user_id = str(payload.get("user_id", "")).strip()
        table_name = str(payload.get("table_name", "")).strip()
        if not table_name:
            raise RuntimeError("missing_table_name")

        user_store = cls._shared_user_store
        token_manager = cls._shared_token_manager
        if user_store is not None and token_manager is not None and user_id:
            user_row = user_store.get_user(user_id)
            if user_row is None:
                raise RuntimeError(f"user_not_found:{user_id}")
            api_key = token_manager.decrypt_api_key(
                user_row.openrouter_api_key_encrypted
            )
        elif user_store is None:
            raise RuntimeError("user_store_not_initialized")
        elif not user_id:
            raise RuntimeError("user_id_missing_in_payload")
        else:
            raise RuntimeError("token_manager_not_initialized")

        llm_client = client_manager.get_client(api_key)

        if kind == "memory_store":
            entry = cls._build_raw_memory_entry(payload)
            embedding = await llm_client.create_single_embedding(
                entry.lossless_restatement
            )
            added = await vector_store.add_entries(
                table_name,
                [entry],
                [embedding],
            )
            return {
                "added": True,
                "processed": True,
                "entries_created": added,
                "message": "Stored exact raw memory entry",
            }

        builder = MemoryBuilder(
            llm_client=llm_client,
            vector_store=vector_store,
            table_name=table_name,
            window_size=settings.window_size,
            overlap_size=settings.overlap_size,
            temperature=settings.llm_temperature,
            max_retries=settings.llm_max_retries,
        )

        if kind == "memory_add":
            speaker = str(payload.get("speaker", "")).strip()
            content = str(payload.get("content", "")).strip()
            if not speaker or not content:
                raise RuntimeError("invalid_memory_add_payload")
            return await builder.add_dialogue(
                speaker=speaker,
                content=content,
                timestamp=payload.get("timestamp"),
            )

        if kind == "memory_add_batch":
            dialogues = payload.get("dialogues")
            if not isinstance(dialogues, list) or not dialogues:
                raise RuntimeError("invalid_memory_add_batch_payload")
            return await builder.add_dialogues(dialogues=dialogues)

        raise RuntimeError(f"unsupported_job_kind:{kind}")

    @classmethod
    def _trim_write_status(cls, limit: int) -> None:
        cap = max(100, int(limit))
        while len(cls._write_status_order) > cap:
            old = cls._write_status_order.popleft()
            cls._write_status_by_job_id.pop(old, None)

    def _get_retriever(self) -> Retriever:
        if not self._retriever:
            self._retriever = Retriever(
                llm_client=self.client_manager.get_client(self.api_key),
                vector_store=self.vector_store,
                table_name=self.user.table_name,
                semantic_top_k=self.settings.semantic_top_k,
                keyword_top_k=self.settings.keyword_top_k,
                enable_planning=self.settings.enable_planning,
                enable_reflection=self.settings.enable_reflection,
                enable_reranking=getattr(self.settings, "enable_reranking", True),
                rerank_candidate_cap=getattr(
                    self.settings, "rerank_candidate_cap", 100
                ),
                rerank_document_max_chars=getattr(
                    self.settings, "rerank_document_max_chars", 1500
                ),
                max_reflection_rounds=self.settings.max_reflection_rounds,
                temperature=self.settings.llm_temperature,
            )
        return self._retriever

    def _get_answer_generator(self) -> AnswerGenerator:
        if not self._answer_generator:
            self._answer_generator = AnswerGenerator(
                llm_client=self.client_manager.get_client(self.api_key),
                temperature=self.settings.llm_temperature,
                max_context_entries=self.settings.memory_query_max_context_entries,
                max_context_entry_chars=self.settings.memory_query_max_context_entry_chars,
                max_context_total_chars=self.settings.memory_query_max_context_total_chars,
                max_answer_tokens=self.settings.memory_query_answer_max_tokens,
            )
        return self._answer_generator

    async def _get_fastpath_client(self, timeout_s: float):
        client = MCPHandler._shared_fastpath_client
        if client is not None:
            return client
        import httpx

        client = httpx.AsyncClient(timeout=max(0.1, timeout_s))
        MCPHandler._shared_fastpath_client = client
        return client

    async def _memory_retrieve_fastpath(
        self,
        *,
        query: str,
        top_k: int,
    ) -> dict[str, Any] | None:
        backend_url = str(
            getattr(self.settings, "memory_retrieve_backend_url", "") or ""
        ).strip()
        if not backend_url:
            return None
        if (
            getattr(self.settings, "memory_retrieve_backend_global_only", True)
            and self.user.user_id != "global"
        ):
            return None

        metadata_keys = (
            "persons",
            "entities",
            "location",
            "topic",
            "smu_topic",
            "timestamp",
            "smu_timestamp",
            "created_at",
            "created_at_utc",
        )

        def _iter_rows(rows: list[Any]) -> list[dict[str, Any]]:
            flat: list[dict[str, Any]] = []
            for row in rows:
                if not isinstance(row, dict):
                    continue
                nested = row.get("results")
                if isinstance(nested, list):
                    for child in nested:
                        if isinstance(child, dict):
                            flat.append(child)
                    continue
                flat.append(row)
            return flat

        def _row_to_result(row: dict[str, Any]) -> dict[str, Any]:
            metadata = row.get("metadata")
            if not isinstance(metadata, dict):
                metadata = {}
            smu_json = metadata.get("smu_json")
            smu_payload: dict[str, Any] = {}
            if isinstance(smu_json, str) and smu_json.strip():
                try:
                    parsed = json.loads(smu_json)
                    if isinstance(parsed, dict):
                        smu_payload = parsed
                except Exception:
                    smu_payload = {}
            topic = (
                metadata.get("smu_topic")
                or metadata.get("topic")
                or smu_payload.get("topic")
                or row.get("source")
            )
            timestamp = (
                metadata.get("timestamp")
                or metadata.get("smu_timestamp")
                or row.get("created_at_utc")
                or metadata.get("created_at_utc")
                or row.get("created_at")
            )
            persons = metadata.get("persons")
            if not isinstance(persons, list):
                persons = (
                    smu_payload.get("persons")
                    if isinstance(smu_payload.get("persons"), list)
                    else []
                )
            entities = metadata.get("entities")
            if not isinstance(entities, list):
                entities = (
                    smu_payload.get("entities")
                    if isinstance(smu_payload.get("entities"), list)
                    else []
                )
            content = (
                row.get("content")
                or row.get("memory")
                or smu_payload.get("lossless_restatement")
                or ""
            )
            return {
                "content": str(content or ""),
                "timestamp": timestamp,
                "location": metadata.get("location"),
                "persons": persons,
                "entities": entities,
                "topic": topic,
                "source": str(row.get("source", "") or ""),
                "metadata_keys": [k for k in metadata_keys if k in metadata],
            }

        try:
            timeout_s = (
                float(
                    getattr(self.settings, "memory_retrieve_backend_timeout_ms", 3000)
                )
                / 1000.0
            )
            client = await self._get_fastpath_client(timeout_s)
            response = await client.post(
                backend_url,
                json={
                    "query": query,
                    "limit": top_k,
                    "user_id": self.user.user_id,
                },
            )
            response.raise_for_status()
            payload = response.json()
        except Exception:
            return None

        rows = payload.get("results")
        if not isinstance(rows, list):
            return None
        flat_rows = _iter_rows(rows)

        return {
            "query": query,
            "results": [_row_to_result(row) for row in flat_rows[:top_k]],
            "total": int(
                payload.get("total_available", len(flat_rows)) or len(flat_rows)
            ),
        }

    @staticmethod
    def _normalize_retrieve_query(query: str) -> str:
        return re.sub(r"\s+", " ", str(query or "").strip().lower())

    @classmethod
    def _retrieve_negative_cache_get(
        cls,
        *,
        key: str,
        ttl_ms: int,
    ) -> dict[str, Any] | None:
        row = cls._retrieve_negative_cache.get(key)
        if row is None:
            return None
        cached_at, payload = row
        if (time.time() - cached_at) * 1000.0 > max(100, ttl_ms):
            cls._retrieve_negative_cache.pop(key, None)
            return None
        return dict(payload)

    @classmethod
    def _retrieve_negative_cache_put(
        cls,
        *,
        key: str,
        payload: dict[str, Any],
        max_entries: int,
    ) -> None:
        cls._retrieve_negative_cache[key] = (time.time(), dict(payload))
        cls._retrieve_negative_cache_order.append(key)
        limit = max(32, max_entries)
        while len(cls._retrieve_negative_cache_order) > limit:
            victim = cls._retrieve_negative_cache_order.popleft()
            cls._retrieve_negative_cache.pop(victim, None)

    @classmethod
    def _retrieve_cache_get(
        cls,
        *,
        key: str,
        ttl_ms: int,
    ) -> dict[str, Any] | None:
        row = cls._retrieve_cache.get(key)
        if row is None:
            return None
        cached_at, payload = row
        if (time.time() - cached_at) * 1000.0 > max(100, ttl_ms):
            cls._retrieve_cache.pop(key, None)
            return None
        return dict(payload)

    @classmethod
    def _retrieve_cache_put(
        cls,
        *,
        key: str,
        payload: dict[str, Any],
        max_entries: int,
    ) -> None:
        cls._retrieve_cache[key] = (time.time(), dict(payload))
        cls._retrieve_cache_order.append(key)
        limit = max(32, max_entries)
        while len(cls._retrieve_cache_order) > limit:
            victim = cls._retrieve_cache_order.popleft()
            cls._retrieve_cache.pop(victim, None)

    @classmethod
    def _query_cache_get(
        cls,
        *,
        key: str,
        ttl_ms: int,
    ) -> dict[str, Any] | None:
        row = cls._query_cache.get(key)
        if row is None:
            return None
        cached_at, payload = row
        if (time.time() - cached_at) * 1000.0 > max(100, ttl_ms):
            cls._query_cache.pop(key, None)
            return None
        return dict(payload)

    @classmethod
    def _query_cache_put(
        cls,
        *,
        key: str,
        payload: dict[str, Any],
        max_entries: int,
    ) -> None:
        cls._query_cache[key] = (time.time(), dict(payload))
        cls._query_cache_order.append(key)
        limit = max(32, max_entries)
        while len(cls._query_cache_order) > limit:
            victim = cls._query_cache_order.popleft()
            cls._query_cache.pop(victim, None)

    @staticmethod
    def _derive_keywords_for_memory_store(
        args: dict[str, Any], content: str
    ) -> list[str]:
        keywords: list[str] = []
        for key in ("category", "component", "source", "status", "confidence"):
            value = str(args.get(key, "") or "").strip().lower()
            if value:
                keywords.append(value)

        tags = args.get("tags", [])
        if isinstance(tags, list):
            for tag in tags:
                text = str(tag).strip().lower()
                if text:
                    keywords.append(text)

        keywords.extend(re.findall(r"[A-Za-z0-9_:/.-]{3,}", content.lower()))

        seen: set[str] = set()
        deduped: list[str] = []
        for keyword in keywords:
            if keyword not in seen:
                seen.add(keyword)
                deduped.append(keyword)
        return deduped[:64]

    @classmethod
    def _build_raw_memory_entry(cls, args: dict[str, Any]) -> MemoryEntry:
        content = str(args.get("content", args.get("text", ""))).strip()
        if not content:
            raise RuntimeError("invalid_memory_store_payload")

        component = str(args.get("component", "") or "").strip()
        category = str(args.get("category", "") or "").strip()
        source = str(args.get("source", "") or "").strip()
        topic = component or category or source or "memory_store"
        pointer = str(args.get("pointer", "") or "").strip() or None
        timestamp = (
            str(args.get("timestamp", args.get("timestamp_utc", "")) or "").strip()
            or None
        )

        return MemoryEntry(
            lossless_restatement=content,
            keywords=cls._derive_keywords_for_memory_store(args, content),
            timestamp=timestamp,
            location=pointer,
            persons=[],
            entities=[],
            topic=topic,
        )

    def _compute_idempotency_key(self, *, kind: str, args: dict[str, Any]) -> str:
        provided = str(args.get("idempotency_key", "")).strip()
        if provided:
            return provided[:256]
        if kind == "memory_store":
            core = {
                "user_id": self.user.user_id,
                "kind": kind,
                "content": str(args.get("content", args.get("text", ""))).strip(),
                "timestamp": str(
                    args.get("timestamp", args.get("timestamp_utc", "")) or ""
                ).strip(),
                "category": str(args.get("category", "") or "").strip(),
                "component": str(args.get("component", "") or "").strip(),
                "source": str(args.get("source", "") or "").strip(),
                "pointer": str(args.get("pointer", "") or "").strip(),
                "status": str(args.get("status", "") or "").strip(),
            }
        elif kind == "memory_add":
            content = str(args.get("content", args.get("text", ""))).strip()
            core = {
                "user_id": self.user.user_id,
                "kind": kind,
                "speaker": str(args.get("speaker", args.get("role", "user"))).strip(),
                "content": content,
                "timestamp": str(args.get("timestamp", "") or "").strip(),
            }
        else:
            core = {
                "user_id": self.user.user_id,
                "kind": kind,
                "dialogues": args.get("dialogues", []),
            }
        raw = json.dumps(
            core, sort_keys=True, ensure_ascii=False, separators=(",", ":")
        )
        return hashlib.sha256(raw.encode("utf-8", errors="replace")).hexdigest()

    async def _enqueue_durable(
        self, *, kind: str, args: dict[str, Any]
    ) -> dict[str, Any]:
        store = MCPHandler._durable_store
        if store is None:
            raise RuntimeError("durable_queue_unavailable")
        idem = self._compute_idempotency_key(kind=kind, args=args)

        if kind == "memory_store":
            content = str(args.get("content", args.get("text", ""))).strip()
            if not content:
                raise ValueError("memory_store requires content or text")
            payload = {
                "content": content,
                "timestamp": args.get("timestamp", args.get("timestamp_utc")),
                "category": args.get("category"),
                "component": args.get("component"),
                "confidence": args.get("confidence"),
                "source": args.get("source"),
                "tags": args.get("tags", []),
                "pointer": args.get("pointer"),
                "status": args.get("status"),
                "table_name": self.user.table_name,
                "user_id": self.user.user_id,
            }
        elif kind == "memory_add":
            speaker = (
                str(args.get("speaker", args.get("role", "user"))).strip() or "user"
            )
            content = str(args.get("content", args.get("text", ""))).strip()
            if not content:
                raise ValueError("memory_add requires content or text")
            payload = {
                "speaker": speaker,
                "content": content,
                "timestamp": args.get("timestamp"),
                "table_name": self.user.table_name,
                "user_id": self.user.user_id,
            }
        else:
            payload = {
                "dialogues": args["dialogues"],
                "table_name": self.user.table_name,
                "user_id": self.user.user_id,
            }

        out = store.enqueue_or_get(
            user_id=self.user.user_id,
            kind=kind,
            idempotency_key=idem,
            payload=payload,
            max_inflight_writes=max(
                10, int(self.settings.memory_queue_max_inflight_writes)
            ),
        )
        if out.get("accepted"):
            MCPHandler._write_stats["accepted"] += 1
            job_id = str(out.get("job_id", ""))
            if job_id:
                MCPHandler._write_status_by_job_id[job_id] = {
                    "job_id": job_id,
                    "kind": kind,
                    "status": str(out.get("status", "queued")),
                    "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                    "idempotency_key": idem,
                }
                MCPHandler._write_status_order.append(job_id)
                MCPHandler._trim_write_status(
                    getattr(self.settings, "memory_write_job_status_limit", 2000)
                )
        else:
            MCPHandler._write_stats["rejected_admission"] += 1
        return out

    async def _wait_for_job_completion(
        self, *, job_id: str, timeout_ms: int
    ) -> dict[str, Any]:
        store = MCPHandler._durable_store
        if store is None:
            return {"status": "unknown"}
        timeout_s = max(0.05, int(timeout_ms) / 1000.0)
        start = asyncio.get_running_loop().time()
        while True:
            job = store.get_job(job_id=job_id)
            if not job:
                return {"status": "missing"}
            status = str(job.get("status", "unknown"))
            if status in {"done", "failed_terminal"}:
                return job
            if (asyncio.get_running_loop().time() - start) >= timeout_s:
                return job
            await asyncio.sleep(0.05)

    async def _enter_read_gate(self) -> None:
        if MCPHandler._read_gate_lock is None:
            MCPHandler._read_gate_lock = asyncio.Lock()
        async with MCPHandler._read_gate_lock:
            if MCPHandler._active_read_ops >= max(
                1, int(self.settings.memory_queue_max_inflight_reads)
            ):
                MCPHandler._read_rejects += 1
                raise RuntimeError(
                    "memory_query_overloaded: read admission limit reached"
                )
            MCPHandler._active_read_ops += 1

    async def _exit_read_gate(self) -> None:
        if MCPHandler._read_gate_lock is None:
            return
        async with MCPHandler._read_gate_lock:
            MCPHandler._active_read_ops = max(0, MCPHandler._active_read_ops - 1)

    async def handle_message(self, message: str) -> str:
        try:
            data = json.loads(message)
            request = JsonRpcRequest(
                jsonrpc=data.get("jsonrpc", "2.0"),
                method=data.get("method", ""),
                id=data.get("id"),
                params=data.get("params", {}),
            )
            response = await self._dispatch(request)
            return json.dumps(response.to_dict(), ensure_ascii=False)
        except json.JSONDecodeError as exc:
            return json.dumps(
                JsonRpcResponse(
                    error={"code": -32700, "message": f"Parse error: {exc}"}
                ).to_dict()
            )
        except Exception as exc:
            return json.dumps(
                JsonRpcResponse(
                    error={"code": -32603, "message": f"Internal error: {exc}"}
                ).to_dict()
            )

    async def _dispatch(self, request: JsonRpcRequest) -> JsonRpcResponse:
        handlers = {
            "initialize": self._handle_initialize,
            "initialized": self._handle_initialized,
            "ping": self._handle_ping,
            "tools/list": self._handle_tools_list,
            "tools/call": self._handle_tools_call,
            "resources/list": self._handle_resources_list,
            "resources/read": self._handle_resources_read,
        }
        handler = handlers.get(request.method)
        if not handler:
            return JsonRpcResponse(
                id=request.id,
                error={
                    "code": -32601,
                    "message": f"Method not found: {request.method}",
                },
            )
        try:
            result = await handler(request.params or {})
            return JsonRpcResponse(id=request.id, result=result)
        except Exception as exc:
            return JsonRpcResponse(
                id=request.id, error={"code": -32603, "message": str(exc)}
            )

    async def _handle_initialize(self, params: dict) -> dict:
        self.initialized = True
        return {
            "protocolVersion": MCP_VERSION,
            "capabilities": {"tools": {}, "resources": {}},
            "serverInfo": {
                "name": SERVER_NAME,
                "version": SERVER_VERSION,
                "description": "SimpleMem durable memory system with idempotent queued writes and adaptive retrieval.",
            },
            "instructions": (
                "Use memory_store for durable single-entry writes (memory_add is a legacy alias). "
                "Use memory_add_batch for durable batch writes. "
                "Writes are accepted only after durable enqueue and return a job_id. "
                "Use memory_job_status for completion state."
            ),
        }

    async def _handle_initialized(self, params: dict) -> dict:
        return {}

    async def _handle_ping(self, params: dict) -> dict:
        return {}

    async def _handle_tools_list(self, params: dict) -> dict:
        return {
            "tools": [
                {
                    "name": "memory_store",
                    "description": "Canonical durable write entrypoint for storing one memory entry.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "user_id": {
                                "type": "string",
                                "description": "Optional; ignored in authenticated mode",
                            },
                            "speaker": {
                                "type": "string",
                                "description": "Optional explicit speaker/agent label",
                            },
                            "role": {
                                "type": "string",
                                "description": "Alias of speaker",
                            },
                            "content": {
                                "type": "string",
                                "description": "Durable memory content",
                            },
                            "text": {
                                "type": "string",
                                "description": "Alias of content for compatibility",
                            },
                            "timestamp": {
                                "type": "string",
                                "description": "ISO timestamp",
                            },
                            "timestamp_utc": {
                                "type": "string",
                                "description": "Alias of timestamp",
                            },
                            "category": {"type": "string"},
                            "component": {"type": "string"},
                            "confidence": {"type": "string"},
                            "source": {"type": "string"},
                            "tags": {"type": "array", "items": {"type": "string"}},
                            "pointer": {"type": "string"},
                            "status": {"type": "string"},
                            "metadata": {
                                "type": "object",
                                "description": "Optional metadata passthrough",
                            },
                            "sync": {
                                "type": "boolean",
                                "description": "Wait briefly for completion after durable accept",
                            },
                            "idempotency_key": {
                                "type": "string",
                                "description": "Optional idempotency key",
                            },
                        },
                        "required": ["content"],
                    },
                },
                {
                    "name": "memory_add",
                    "description": "Legacy alias of memory_store. Durably enqueue a single dialogue for memory extraction and indexing.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "user_id": {
                                "type": "string",
                                "description": "Optional; ignored in authenticated mode",
                            },
                            "speaker": {
                                "type": "string",
                                "description": "Speaker name (defaults to 'user')",
                            },
                            "role": {
                                "type": "string",
                                "description": "Alias of speaker",
                            },
                            "content": {
                                "type": "string",
                                "description": "Dialogue content",
                            },
                            "text": {
                                "type": "string",
                                "description": "Alias of content for compatibility",
                            },
                            "metadata": {
                                "type": "object",
                                "description": "Optional metadata passthrough",
                            },
                            "timestamp": {
                                "type": "string",
                                "description": "ISO timestamp",
                            },
                            "sync": {
                                "type": "boolean",
                                "description": "Wait briefly for completion after durable accept",
                            },
                            "idempotency_key": {
                                "type": "string",
                                "description": "Optional idempotency key",
                            },
                        },
                        "required": [],
                    },
                },
                {
                    "name": "memory_add_batch",
                    "description": "Durably enqueue a batch of dialogues for background processing.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "dialogues": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "speaker": {"type": "string"},
                                        "content": {"type": "string"},
                                        "timestamp": {"type": "string"},
                                    },
                                    "required": ["speaker", "content"],
                                },
                            },
                            "sync": {
                                "type": "boolean",
                                "description": "Wait briefly for completion after durable accept",
                            },
                            "idempotency_key": {
                                "type": "string",
                                "description": "Optional idempotency key",
                            },
                        },
                        "required": ["dialogues"],
                    },
                },
                {
                    "name": "memory_job_status",
                    "description": "Return status for a previously accepted durable write job.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {"job_id": {"type": "string"}},
                        "required": ["job_id"],
                    },
                },
                {
                    "name": "memory_query",
                    "description": "Query stored memories and synthesize answer.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "user_id": {
                                "type": "string",
                                "description": "Optional; ignored in authenticated mode",
                            },
                            "question": {"type": "string", "description": "Query text"},
                            "query": {
                                "type": "string",
                                "description": "Alias of question for compatibility",
                            },
                            "enable_reflection": {"type": "boolean"},
                            "enable_planning": {"type": "boolean"},
                            "limit": {
                                "type": "integer",
                                "description": "Alias of top_k",
                            },
                            "top_k": {"type": "integer"},
                        },
                        "required": [],
                    },
                },
                {
                    "name": "memory_retrieve",
                    "description": "Retrieve relevant memory entries without synthesis.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "user_id": {
                                "type": "string",
                                "description": "Optional; ignored in authenticated mode",
                            },
                            "query": {"type": "string"},
                            "limit": {
                                "type": "integer",
                                "description": "Alias of top_k",
                            },
                            "top_k": {"type": "integer"},
                        },
                        "required": ["query"],
                    },
                },
                {
                    "name": "memory_clear",
                    "description": "Clear all memories for this user.",
                    "inputSchema": {"type": "object", "properties": {}},
                },
                {
                    "name": "memory_stats",
                    "description": "Return memory and queue statistics.",
                    "inputSchema": {"type": "object", "properties": {}},
                },
            ]
        }

    async def _handle_tools_call(self, params: dict) -> dict:
        name = params.get("name", "")
        arguments = params.get("arguments", {})
        handlers = {
            "memory_store": self._tool_memory_store,
            "memory_add": self._tool_memory_add,
            "memory_add_batch": self._tool_memory_add_batch,
            "memory_job_status": self._tool_memory_job_status,
            "memory_query": self._tool_memory_query,
            "memory_retrieve": self._tool_memory_retrieve,
            "memory_clear": self._tool_memory_clear,
            "memory_stats": self._tool_memory_stats,
        }
        handler = handlers.get(name)
        if not handler:
            raise ValueError(f"Unknown tool: {name}")
        result = await handler(arguments)
        return {
            "content": [
                {
                    "type": "text",
                    "text": json.dumps(
                        result, ensure_ascii=False, separators=(",", ":")
                    ),
                }
            ]
        }

    def _normalize_memory_store_args(self, args: dict[str, Any]) -> dict[str, Any]:
        content = str(args.get("content", args.get("text", ""))).strip()
        if not content:
            raise ValueError("memory_store requires content or text")

        speaker = (
            str(args.get("speaker", "")).strip()
            or str(args.get("role", "")).strip()
            or str(args.get("agent", "")).strip()
            or str(args.get("source", "")).strip()
            or "runner"
        )

        normalized = dict(args)
        normalized["speaker"] = speaker
        normalized["content"] = content

        timestamp = str(
            args.get("timestamp", args.get("timestamp_utc", "")) or ""
        ).strip()
        if timestamp:
            normalized["timestamp"] = timestamp

        return normalized

    async def _tool_memory_store(self, args: dict) -> dict:
        normalized = self._normalize_memory_store_args(args)
        queued = await self._enqueue_durable(kind="memory_store", args=normalized)
        if not queued.get("accepted"):
            return {
                "accepted": False,
                "durable": False,
                "status": "rejected_overloaded",
                "reason": queued.get("reason", "memory_write_admission_limit"),
                "retryable": bool(queued.get("retryable", True)),
            }

        out = {
            "accepted": True,
            "durable": True,
            "status": str(queued.get("status", "queued")),
            "job_id": str(queued.get("job_id", "")),
            "idempotency_key": str(queued.get("idempotency_key", "")),
            "existing": bool(queued.get("existing", False)),
        }

        force_sync = bool(args.get("sync", False)) or (
            not bool(self.settings.memory_add_async_default)
        )
        if not force_sync:
            return out

        waited = await self._wait_for_job_completion(
            job_id=out["job_id"],
            timeout_ms=max(50, int(self.settings.memory_sync_wait_timeout_ms)),
        )
        status = str(waited.get("status", "processing"))
        if status == "done":
            out["status"] = "done"
            out["result"] = waited.get("result")
        elif status == "failed_terminal":
            out["status"] = "failed_terminal"
            out["error"] = waited.get("last_error")
        else:
            out["status"] = "accepted_processing"
        return out

    async def _tool_memory_add(self, args: dict) -> dict:
        queued = await self._enqueue_durable(kind="memory_add", args=args)
        if not queued.get("accepted"):
            return {
                "accepted": False,
                "durable": False,
                "status": "rejected_overloaded",
                "reason": queued.get("reason", "memory_write_admission_limit"),
                "retryable": bool(queued.get("retryable", True)),
            }

        out = {
            "accepted": True,
            "durable": True,
            "status": str(queued.get("status", "queued")),
            "job_id": str(queued.get("job_id", "")),
            "idempotency_key": str(queued.get("idempotency_key", "")),
            "existing": bool(queued.get("existing", False)),
        }

        force_sync = bool(args.get("sync", False)) or (
            not bool(self.settings.memory_add_async_default)
        )
        if not force_sync:
            return out

        waited = await self._wait_for_job_completion(
            job_id=out["job_id"],
            timeout_ms=max(50, int(self.settings.memory_sync_wait_timeout_ms)),
        )
        status = str(waited.get("status", "processing"))
        if status == "done":
            out["status"] = "done"
            out["result"] = waited.get("result")
        elif status == "failed_terminal":
            out["status"] = "failed_terminal"
            out["error"] = waited.get("last_error")
        else:
            out["status"] = "accepted_processing"
        return out

    async def _tool_memory_add_batch(self, args: dict) -> dict:
        queued = await self._enqueue_durable(kind="memory_add_batch", args=args)
        if not queued.get("accepted"):
            return {
                "accepted": False,
                "durable": False,
                "status": "rejected_overloaded",
                "reason": queued.get("reason", "memory_write_admission_limit"),
                "retryable": bool(queued.get("retryable", True)),
            }

        out = {
            "accepted": True,
            "durable": True,
            "status": str(queued.get("status", "queued")),
            "job_id": str(queued.get("job_id", "")),
            "idempotency_key": str(queued.get("idempotency_key", "")),
            "existing": bool(queued.get("existing", False)),
        }

        force_sync = bool(args.get("sync", False)) or (
            not bool(self.settings.memory_add_async_default)
        )
        if not force_sync:
            return out

        waited = await self._wait_for_job_completion(
            job_id=out["job_id"],
            timeout_ms=max(50, int(self.settings.memory_sync_wait_timeout_ms)),
        )
        status = str(waited.get("status", "processing"))
        if status == "done":
            out["status"] = "done"
            out["result"] = waited.get("result")
        elif status == "failed_terminal":
            out["status"] = "failed_terminal"
            out["error"] = waited.get("last_error")
        else:
            out["status"] = "accepted_processing"
        return out

    async def _tool_memory_job_status(self, args: dict) -> dict:
        job_id = str(args.get("job_id", "")).strip()
        if not job_id:
            return {"found": False, "error": "job_id_required"}
        store = MCPHandler._durable_store
        if store is None:
            return {"found": False, "error": "durable_queue_unavailable"}
        row = store.get_job(job_id=job_id)
        if not row:
            return {"found": False, "job_id": job_id}
        if row.get("user_id") != self.user.user_id:
            return {"found": False, "job_id": job_id}
        return {"found": True, **row}

    async def _tool_memory_query(self, args: dict) -> dict:
        question = str(args.get("question", args.get("query", ""))).strip()
        if not question:
            raise ValueError("memory_query requires question or query")
        requested_top_k = args.get(
            "top_k", args.get("limit", self.settings.memory_query_default_top_k)
        )
        try:
            requested_top_k = int(requested_top_k)
        except Exception:
            requested_top_k = self.settings.memory_query_default_top_k
        top_k = max(1, min(requested_top_k, self.settings.memory_query_max_top_k))
        enable_reflection = bool(
            args.get("enable_reflection", self.settings.enable_reflection)
        )
        enable_planning = bool(
            args.get("enable_planning", self.settings.enable_planning)
        )
        cache_key = (
            f"{self.user.user_id}|{top_k}|{int(enable_reflection)}|"
            f"{int(enable_planning)}|{self._normalize_retrieve_query(question)}"
        )
        cache_ttl_ms = int(
            getattr(self.settings, "memory_query_result_cache_ttl_ms", 4000)
        )
        cache_limit = int(
            getattr(self.settings, "memory_query_result_cache_entries", 256)
        )
        cached = self._query_cache_get(key=cache_key, ttl_ms=cache_ttl_ms)
        if cached is not None:
            return cached
        inflight = MCPHandler._query_inflight.get(cache_key)
        if inflight is not None:
            return dict(await asyncio.shield(inflight))
        loop = asyncio.get_running_loop()
        owner_future: asyncio.Future = loop.create_future()
        MCPHandler._add_inflight(
            MCPHandler._query_inflight,
            MCPHandler._query_inflight_ts,
            cache_key,
            owner_future,
        )

        await self._enter_read_gate()
        try:
            generator = self._get_answer_generator()
            retriever = self._get_retriever()
            contexts = await retriever.retrieve(
                query=question,
                enable_reflection=enable_reflection,
                enable_planning=enable_planning,
                top_k=top_k,
            )
            contexts = contexts[:top_k]

            synthesis_timeout_s = max(
                0.25,
                float(getattr(self.settings, "memory_query_synthesis_timeout_ms", 4500))
                / 1000.0,
            )
            try:
                answer_result = await asyncio.wait_for(
                    generator.generate_answer(query=question, contexts=contexts),
                    timeout=synthesis_timeout_s,
                )
            except asyncio.TimeoutError:
                snippets = [
                    str(e.lossless_restatement or "").strip()
                    for e in contexts[:3]
                    if str(e.lossless_restatement or "").strip()
                ]
                fallback = (
                    " | ".join(snippets)
                    if snippets
                    else "Timed out while synthesizing answer."
                )
                answer_result = {
                    "answer": fallback[:700],
                    "reasoning": f"Fallback summary from memory_retrieve contexts after synthesis timeout ({synthesis_timeout_s:.2f}s).",
                    "confidence": "low",
                }
            answer_text = str(answer_result.get("answer", ""))
            reasoning_text = str(answer_result.get("reasoning", ""))
            if "error occurred while generating the answer" in answer_text.lower():
                snippets = [
                    str(e.lossless_restatement or "").strip()
                    for e in contexts[:3]
                    if str(e.lossless_restatement or "").strip()
                ]
                if snippets:
                    fallback = " | ".join(snippets)
                    answer_result = {
                        "answer": fallback[:700],
                        "reasoning": f"Fallback summary from memory_retrieve contexts. Original synthesis failure: {reasoning_text[:220]}",
                        "confidence": "low",
                    }

            result = {
                "question": question,
                "answer": answer_result.get("answer", ""),
                "reasoning": answer_result.get("reasoning", ""),
                "confidence": answer_result.get("confidence", "low"),
                "contexts_used": len(contexts),
                "top_k": top_k,
            }
            self._query_cache_put(
                key=cache_key,
                payload=result,
                max_entries=cache_limit,
            )
            owner_future.set_result(dict(result))
            return result
        except Exception as exc:
            owner_future.set_exception(exc)
            raise
        finally:
            current = MCPHandler._query_inflight.get(cache_key)
            if current is owner_future:
                MCPHandler._remove_inflight(
                    MCPHandler._query_inflight,
                    MCPHandler._query_inflight_ts,
                    cache_key,
                    owner_future,
                )
            await self._exit_read_gate()

    async def _tool_memory_retrieve(self, args: dict) -> dict:
        query = str(args.get("query", "")).strip()
        if not query:
            raise ValueError("memory_retrieve requires query")
        top_k = int(args.get("top_k", args.get("limit", 10)))
        retrieve_cache_ttl_ms = int(
            getattr(self.settings, "memory_retrieve_result_cache_ttl_ms", 2000)
        )
        retrieve_cache_limit = int(
            getattr(self.settings, "memory_retrieve_result_cache_entries", 512)
        )
        negative_cache_ttl_ms = int(
            getattr(self.settings, "memory_retrieve_negative_cache_ttl_ms", 3000)
        )
        negative_cache_limit = int(
            getattr(self.settings, "memory_retrieve_negative_cache_entries", 512)
        )
        cache_key = (
            f"{self.user.user_id}|{top_k}|{self._normalize_retrieve_query(query)}"
        )
        cached_result = self._retrieve_cache_get(
            key=cache_key, ttl_ms=retrieve_cache_ttl_ms
        )
        if cached_result is not None:
            return cached_result
        cached_empty = self._retrieve_negative_cache_get(
            key=cache_key, ttl_ms=negative_cache_ttl_ms
        )
        if cached_empty is not None:
            return cached_empty
        inflight = MCPHandler._retrieve_inflight.get(cache_key)
        if inflight is not None:
            return dict(await asyncio.shield(inflight))
        loop = asyncio.get_running_loop()
        owner_future: asyncio.Future = loop.create_future()
        MCPHandler._add_inflight(
            MCPHandler._retrieve_inflight,
            MCPHandler._retrieve_inflight_ts,
            cache_key,
            owner_future,
        )
        fastpath = await self._memory_retrieve_fastpath(query=query, top_k=top_k)
        try:
            if fastpath is not None and int(fastpath.get("total", 0) or 0) > 0:
                result = fastpath
            else:
                retriever = self._get_retriever()
                entries = await retriever.retrieve(
                    query=query,
                    enable_reflection=False,
                    enable_planning=False,
                    top_k=top_k,
                )
                result = {
                    "query": query,
                    "results": [
                        {
                            "content": e.lossless_restatement,
                            "timestamp": e.timestamp,
                            "location": e.location,
                            "persons": e.persons,
                            "entities": e.entities,
                            "topic": e.topic,
                        }
                        for e in entries[:top_k]
                    ],
                    "total": len(entries),
                }
                if int(result.get("total", 0) or 0) <= 0:
                    self._retrieve_negative_cache_put(
                        key=cache_key,
                        payload=result,
                        max_entries=negative_cache_limit,
                    )
            self._retrieve_cache_put(
                key=cache_key,
                payload=result,
                max_entries=retrieve_cache_limit,
            )
            owner_future.set_result(dict(result))
            return result
        except Exception as exc:
            owner_future.set_exception(exc)
            raise
        finally:
            current = MCPHandler._retrieve_inflight.get(cache_key)
            if current is owner_future:
                MCPHandler._remove_inflight(
                    MCPHandler._retrieve_inflight,
                    MCPHandler._retrieve_inflight_ts,
                    cache_key,
                    owner_future,
                )

    async def _tool_memory_clear(self, args: dict) -> dict:
        store = MCPHandler._durable_store
        if store is not None:
            counts = store.queue_counts(user_id=self.user.user_id)
            pending = int(counts.get("pending", 0))
            processing = int(counts.get("processing", 0))
            if pending > 0 or processing > 0:
                return {
                    "success": False,
                    "message": f"Cannot clear: {pending} pending and {processing} processing jobs exist. Wait for pending operations to complete.",
                }
        success = await self.vector_store.clear_table(self.user.table_name)
        return {
            "success": success,
            "message": "All memories cleared" if success else "Failed",
        }

    async def _tool_memory_stats(self, args: dict) -> dict:
        stats = self.vector_store.get_stats(self.user.table_name)
        store = MCPHandler._durable_store
        durable = store.queue_counts() if store is not None else {}
        pending = int(durable.get("pending", 0))
        processing = int(durable.get("processing", 0))
        done = int(durable.get("done", 0))
        failed_retryable = int(durable.get("failed_retryable", 0))
        failed_terminal = int(durable.get("failed_terminal", 0))
        oldest_pending_age_s = durable.get("oldest_pending_age_s")
        retry_rate_window = MCPHandler._write_stats["retryable_failures"] / max(
            1, MCPHandler._write_stats["accepted"]
        )
        write_queue = {
            **durable,
            "workers": len(MCPHandler._worker_tasks),
            "accepted_total": MCPHandler._write_stats["accepted"],
            "completed_total": MCPHandler._write_stats["completed"],
            "failed_total": MCPHandler._write_stats["failed"],
            "retryable_failures_total": MCPHandler._write_stats["retryable_failures"],
            "rejected_admission_total": MCPHandler._write_stats["rejected_admission"],
            "active_read_ops": MCPHandler._active_read_ops,
            "lease_recoveries_total": MCPHandler._lease_recoveries,
        }
        return {
            "user_id": self.user.user_id,
            "entry_count": stats.get("entry_count", 0),
            "durable_queue": {
                "pending": pending,
                "processing": processing,
                "done": done,
                "failed_retryable": failed_retryable,
                "failed_terminal": failed_terminal,
                "oldest_pending_age_s": oldest_pending_age_s,
                "retry_rate_window": retry_rate_window,
                "lease_timeout_recoveries": MCPHandler._lease_recoveries,
                "queue_admission_rejects": MCPHandler._write_stats[
                    "rejected_admission"
                ],
            },
            "durable_workers_running": len(MCPHandler._worker_tasks),
            "admission_rejects_read": MCPHandler._read_rejects,
            "admission_rejects_write": MCPHandler._write_stats["rejected_admission"],
            "write_queue": write_queue,
        }

    async def _handle_resources_list(self, params: dict) -> dict:
        return {
            "resources": [
                {
                    "uri": f"memory://{self.user.user_id}/stats",
                    "name": "Memory Statistics",
                    "description": "Statistics about your memory store",
                    "mimeType": "application/json",
                },
                {
                    "uri": f"memory://{self.user.user_id}/all",
                    "name": "All Memories",
                    "description": "All stored memory entries",
                    "mimeType": "application/json",
                },
            ]
        }

    async def _handle_resources_read(self, params: dict) -> dict:
        uri = params.get("uri", "")
        if uri.endswith("/stats"):
            content = json.dumps(
                self.vector_store.get_stats(self.user.table_name), ensure_ascii=False
            )
        elif uri.endswith("/all"):
            entries = await self.vector_store.get_all_entries(self.user.table_name)
            content = json.dumps(
                {"entries": [e.to_dict() for e in entries], "total": len(entries)},
                ensure_ascii=False,
            )
        else:
            raise ValueError(f"Unknown resource: {uri}")
        return {
            "contents": [{"uri": uri, "mimeType": "application/json", "text": content}]
        }
