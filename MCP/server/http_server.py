"""
HTTP Server for SimpleMem MCP

Provides:
- User registration and authentication (/api/auth/*)
- MCP over Streamable HTTP (2025-03-26 spec) (/mcp)
- Legacy MCP over SSE for backwards compatibility (/mcp/sse)
- Static frontend for configuration (/)

Streamable HTTP Transport:
- Single endpoint /mcp supporting POST, GET, DELETE
- Authentication via Authorization: Bearer <token> header
- Session management via Mcp-Session-Id header
- Supports both JSON and SSE response formats
"""

import asyncio
import json
import logging
import os
import re
import secrets
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

from fastapi import FastAPI, HTTPException, Query, Request, Header, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from pydantic import BaseModel

from .auth.token_manager import TokenManager
from .auth.models import User
from .database.user_store import UserStore
from .database.vector_store import MultiTenantVectorStore
from .integrations.openrouter import OpenRouterClient, OpenRouterClientManager
from .mcp_handler import MCPHandler

import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import get_settings


logger = logging.getLogger(__name__)


# === Pydantic Models ===


class RegisterRequest(BaseModel):
    openrouter_api_key: str
    user_id: Optional[str] = None


class RotateApiKeyRequest(BaseModel):
    openrouter_api_key: str


class RegisterResponse(BaseModel):
    success: bool
    token: Optional[str] = None
    user_id: Optional[str] = None
    mcp_endpoint: Optional[str] = None
    error: Optional[str] = None


# === Session Management ===


@dataclass
class MCPSession:
    """Represents an active MCP session"""

    session_id: str
    user_id: str
    user: User
    api_key: str
    handler: MCPHandler
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_active: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    initialized: bool = False
    # Server-to-client message queue for GET requests
    message_queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    # Track active SSE streams
    active_streams: set = field(default_factory=set)
    # Event ID counter for resumability
    event_counter: int = 0

    def next_event_id(self) -> str:
        """Generate next SSE event ID"""
        self.event_counter += 1
        return f"{self.session_id}-{self.event_counter}"

    def touch(self):
        """Update last active timestamp"""
        self.last_active = datetime.now(timezone.utc)


# Session expiry time (30 minutes of inactivity)
SESSION_EXPIRY_MINUTES = 30


# === Global Instances ===

settings = get_settings()
user_store = UserStore(settings.user_db_path)
vector_store = MultiTenantVectorStore(
    settings.lancedb_path, settings.embedding_dimension
)
token_manager = TokenManager(
    secret_key=settings.jwt_secret_key,
    encryption_key=settings.encryption_key,
    expiration_days=settings.jwt_expiration_days,
)

# Initialize client manager against the active OpenRouter-compatible endpoints.
client_manager = OpenRouterClientManager(
    base_url=settings.openrouter_base_url,
    embedding_base_url=settings.openrouter_embedding_base_url,
    rerank_base_url=settings.openrouter_rerank_base_url,
    llm_model=settings.llm_model,
    embedding_model=settings.embedding_model,
    reranker_model=settings.reranker_model,
)

# Store active MCP handlers per user (legacy)
_mcp_handlers: dict[str, MCPHandler] = {}

# Store active sessions by session_id
_sessions: dict[str, MCPSession] = {}

# Lock for session operations
_session_lock = asyncio.Lock()

# Throttle user activity writes to avoid sync-write amplification on MCP hot paths.
LAST_ACTIVE_UPDATE_INTERVAL_S = max(
    0,
    int(os.getenv("SIMPLEMEM_LAST_ACTIVE_UPDATE_INTERVAL_S", "300")),
)
_last_active_write_ts: dict[str, float] = {}
_last_active_lock = asyncio.Lock()


# === Session Helper Functions ===


async def cleanup_expired_sessions():
    """Remove expired sessions"""
    async with _session_lock:
        now = datetime.now(timezone.utc)
        expired = [
            sid
            for sid, session in _sessions.items()
            if (
                now
                - (
                    session.last_active
                    if session.last_active.tzinfo is not None
                    else session.last_active.replace(tzinfo=timezone.utc)
                )
            ).total_seconds()
            > SESSION_EXPIRY_MINUTES * 60
        ]
        for sid in expired:
            del _sessions[sid]
        if expired:
            print(f"Cleaned up {len(expired)} expired sessions")

        active_user_ids = set()
        for sess in _sessions.values():
            if hasattr(sess, "user_id") and sess.user_id:
                active_user_ids.add(sess.user_id)
        orphaned = [uid for uid in _mcp_handlers if uid not in active_user_ids]
        for uid in orphaned:
            _mcp_handlers.pop(uid, None)


async def session_cleanup_task():
    """Background task to clean up expired sessions"""
    while True:
        await asyncio.sleep(60)  # Check every minute
        await cleanup_expired_sessions()


def generate_session_id() -> str:
    """Generate a cryptographically secure session ID"""
    return secrets.token_urlsafe(32)


async def get_or_create_session(
    user: User, api_key: str, session_id: Optional[str] = None
) -> MCPSession:
    """Get existing session or create new one"""
    async with _session_lock:
        if session_id and session_id in _sessions:
            session = _sessions[session_id]
            session.touch()
            return session

        # Create new session
        new_session_id = generate_session_id()
        handler = MCPHandler(
            user=user,
            api_key=api_key,
            vector_store=vector_store,
            client_manager=client_manager,
            settings=settings,
        )
        session = MCPSession(
            session_id=new_session_id,
            user_id=user.user_id,
            user=user,
            api_key=api_key,
            handler=handler,
        )
        _sessions[new_session_id] = session
        return session


async def get_session(session_id: str) -> Optional[MCPSession]:
    """Get session by ID"""
    async with _session_lock:
        session = _sessions.get(session_id)
        if session:
            session.touch()
        return session


async def delete_session(session_id: str) -> bool:
    """Delete a session"""
    async with _session_lock:
        if session_id in _sessions:
            del _sessions[session_id]
            return True
        return False


# === Authentication Helper ===


async def verify_bearer_token(authorization: Optional[str]) -> tuple[User, str]:
    """
    Verify Bearer token from Authorization header.
    Returns (user, api_key) tuple.
    Raises HTTPException on failure.
    """
    if not authorization:
        raise HTTPException(
            status_code=401,
            detail="Missing Authorization header",
            headers={"WWW-Authenticate": "Bearer"},
        )

    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(
            status_code=401,
            detail="Invalid Authorization header format. Use: Bearer <token>",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = parts[1]
    is_valid, payload, error = token_manager.verify_token(token)
    if not is_valid:
        raise HTTPException(
            status_code=401,
            detail=f"Invalid token: {error}",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user = await asyncio.to_thread(user_store.get_user, payload.user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    if int(payload.token_version) != int(user.token_version):
        raise HTTPException(status_code=401, detail="Token has been revoked")

    api_key = token_manager.decrypt_api_key(user.openrouter_api_key_encrypted)
    return user, api_key


async def _invalidate_user_sessions(user_id: str) -> None:
    async with _session_lock:
        for session_id, session in list(_sessions.items()):
            if session.user_id == user_id:
                _sessions.pop(session_id, None)
    # Legacy /mcp/message handlers cache the decrypted provider key.  Drop
    # that object too, otherwise credential rotation would not take effect.
    _mcp_handlers.pop(user_id, None)


async def maybe_update_last_active(user_id: str) -> None:
    """Rate-limit last_active writes to reduce hot-path DB overhead."""
    if LAST_ACTIVE_UPDATE_INTERVAL_S <= 0:
        return
    now = time.monotonic()
    should_write = False
    async with _last_active_lock:
        last = _last_active_write_ts.get(user_id, 0.0)
        if (now - last) >= float(LAST_ACTIVE_UPDATE_INTERVAL_S):
            _last_active_write_ts[user_id] = now
            should_write = True
    if should_write:
        try:
            await asyncio.to_thread(user_store.update_last_active, user_id)
        except Exception:
            # Activity refresh is best-effort, but a failed write must remain
            # visible so operators do not mistake it for a healthy request path.
            logger.warning(
                "last_active_update_failed user_id=%s",
                user_id,
                exc_info=True,
            )


# === Lifecycle ===


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for the app"""
    global _default_executor
    loop = asyncio.get_running_loop()
    max_workers = max(
        2,
        int(os.getenv("ANTIGRAVITY_SIMPLEMEM_TO_THREAD_MAX_WORKERS", "4")),
    )
    _default_executor = ThreadPoolExecutor(
        max_workers=max_workers,
        thread_name_prefix="simplemem-to-thread",
    )
    loop.set_default_executor(_default_executor)
    print("SimpleMem MCP Server started")
    print(f"  - LLM Model: {settings.llm_model}")
    print(f"  - Embedding Model: {settings.embedding_model}")
    print(f"  - Reranker Model: {settings.reranker_model}")
    print(f"  - Window Size: {settings.window_size}")
    print("  - Transport: Streamable HTTP (2025-03-26)")
    await MCPHandler.start_background_workers(
        settings=settings,
        vector_store=vector_store,
        client_manager=client_manager,
        user_store=user_store,
        token_manager=token_manager,
    )

    # Start session cleanup task
    cleanup_task = asyncio.create_task(session_cleanup_task())

    try:
        yield
    finally:
        cleanup_task.cancel()
        try:
            await cleanup_task
        except asyncio.CancelledError:
            pass

        # Do not cancel accepted jobs or their storage threads on elapsed time.
        await MCPHandler.stop_background_workers()
        if _default_executor is not None:
            _default_executor.shutdown(wait=True, cancel_futures=False)
            _default_executor = None
        print("SimpleMem MCP Server stopped")


# === Rate Limiter ===


class _RateLimiter:
    def __init__(self, max_requests=100, window_seconds=60, max_keys=10000):
        self._max = max_requests
        self._window = window_seconds
        self._max_keys = max(100, int(max_keys))
        self._counts: dict[str, list[float]] = {}

    def is_allowed(self, key: str) -> bool:
        import time

        now = time.time()
        cutoff = now - self._window
        timestamps = self._counts.get(key)
        if timestamps is None:
            if len(self._counts) >= self._max_keys:
                oldest_key = min(
                    self._counts,
                    key=lambda candidate: max(self._counts[candidate], default=0.0),
                )
                self._counts.pop(oldest_key, None)
            timestamps = []
            self._counts[key] = timestamps
        else:
            timestamps = [t for t in timestamps if t > cutoff]
            self._counts[key] = timestamps
        if len(self._counts) > self._max_keys:
            oldest_key = min(
                self._counts,
                key=lambda candidate: max(self._counts[candidate], default=0.0),
            )
            if oldest_key != key:
                self._counts.pop(oldest_key, None)
        if len(self._counts[key]) >= self._max:
            return False
        self._counts[key].append(now)
        return True


_request_limiter = _RateLimiter(max_requests=100, window_seconds=60)
_default_executor: ThreadPoolExecutor | None = None


async def _read_request_body_limited(request: Request, max_bytes: int) -> bytes:
    """Read a request incrementally so oversized bodies are rejected early."""
    limit = max(1, int(max_bytes))
    chunks: list[bytes] = []
    size = 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > limit:
            raise HTTPException(
                status_code=413,
                detail=f"Request body too large (max {limit} bytes)",
            )
        chunks.append(chunk)
    return b"".join(chunks)


# === FastAPI App ===

app = FastAPI(
    title="SimpleMem MCP Server",
    description="Multi-tenant Memory Service for LLM Agents",
    version="1.0.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    client_ip = request.client.host if request.client else "unknown"
    if not _request_limiter.is_allowed(client_ip):
        return JSONResponse(status_code=429, content={"error": "Too many requests"})
    response = await call_next(request)
    return response


def _cors_configuration() -> tuple[list[str], bool]:
    """Resolve CORS origins without allowing wildcard credentials."""
    raw = os.getenv("CORS_ALLOWED_ORIGINS", os.getenv("CORS_ORIGINS", "")).strip()
    origins = [origin.strip() for origin in raw.split(",") if origin.strip()]
    if not origins:
        origins = ["http://localhost:3000"]
    return origins, "*" not in origins


cors_origins, allow_credentials = _cors_configuration()
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=allow_credentials,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Authorization", "Content-Type", "Accept", "Mcp-Session-Id"],
)


# === Authentication Endpoints ===


@app.post("/api/auth/register", response_model=RegisterResponse)
async def register(request: Request):
    """Register a new user with API key (local endpoints may ignore it)."""
    try:
        body = await _read_request_body_limited(request, 1 * 1024 * 1024)
        reg = RegisterRequest(**json.loads(body))
        api_key = reg.openrouter_api_key
        requested_user_id = (reg.user_id or "").strip()

        # Validate the active OpenRouter-compatible endpoint. Local endpoints
        # only require reachability; cloud endpoints still require a valid key.
        client = OpenRouterClient(
            api_key=api_key,
            base_url=settings.openrouter_base_url,
            embedding_base_url=settings.openrouter_embedding_base_url,
            rerank_base_url=settings.openrouter_rerank_base_url,
            llm_model=settings.llm_model,
            embedding_model=settings.embedding_model,
            reranker_model=settings.reranker_model,
        )
        is_valid, error = await client.verify_api_key()
        await client.close()

        if not is_valid:
            return RegisterResponse(
                success=False,
                error=f"Invalid OpenRouter-compatible endpoint or API key: {error}",
            )

        # Resolve target user:
        # - when user_id is provided, upsert that specific user
        # - otherwise preserve prior behavior (new random user)
        encrypted_api_key = token_manager.encrypt_api_key(api_key)
        if requested_user_id:
            existing_user = user_store.get_user(requested_user_id)
            if existing_user:
                raise HTTPException(
                    status_code=409,
                    detail="User already exists; registration cannot rotate credentials",
                )
            else:
                if requested_user_id == "global":
                    raise HTTPException(
                        status_code=403,
                        detail="Reserved user_id cannot be registered",
                    )
                if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", requested_user_id):
                    raise HTTPException(
                        status_code=400,
                        detail="user_id must contain only letters, digits, underscore, or hyphen",
                    )
                user = User(user_id=requested_user_id)
                user.openrouter_api_key_encrypted = encrypted_api_key
                user_store.create_user(user)
        else:
            user = User()
            user.openrouter_api_key_encrypted = encrypted_api_key
            user_store.create_user(user)

        # Generate token
        token = token_manager.generate_token(user)

        # Base URL for MCP endpoint (can be overridden via env)
        base_url = os.getenv("MCP_BASE_URL", "")
        mcp_endpoint = f"{base_url}/mcp" if base_url else "/mcp"

        return RegisterResponse(
            success=True,
            token=token,
            user_id=user.user_id,
            mcp_endpoint=mcp_endpoint,  # Streamable HTTP endpoint
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("registration_failed")
        raise HTTPException(
            status_code=500,
            detail="Registration failed. Please try again.",
        ) from exc


@app.get("/api/auth/verify")
async def verify_token(token: str = Header(..., alias="Authorization")):
    """Verify token is valid"""
    if token.startswith("Bearer "):
        token = token[7:]
    is_valid, payload, error = token_manager.verify_token(token)
    if not is_valid:
        raise HTTPException(status_code=401, detail=error)

    user = user_store.get_user(payload.user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    if int(payload.token_version) != int(user.token_version):
        raise HTTPException(status_code=401, detail="Token has been revoked")

    return {
        "valid": True,
        "user_id": user.user_id,
    }


@app.post("/api/auth/refresh")
async def refresh_token(token: str = Header(..., alias="Authorization")):
    """Refresh authentication token"""
    if token.startswith("Bearer "):
        token = token[7:]
    is_valid, payload, error = token_manager.verify_token(token)
    if not is_valid:
        raise HTTPException(status_code=401, detail=error)

    user = user_store.get_user(payload.user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    if int(payload.token_version) != int(user.token_version):
        raise HTTPException(status_code=401, detail="Token has been revoked")

    new_token = token_manager.generate_token(user)
    return {
        "success": True,
        "token": new_token,
    }


@app.post("/api/auth/revoke")
async def revoke_tokens(authorization: Optional[str] = Header(None)):
    """Revoke all existing tokens and sessions for the authenticated user."""
    user, _ = await verify_bearer_token(authorization)
    revoked = await asyncio.to_thread(user_store.revoke_tokens, user.user_id)
    if not revoked:
        raise HTTPException(status_code=404, detail="User not found")
    await _invalidate_user_sessions(user.user_id)
    return {"success": True, "user_id": user.user_id}


@app.post("/api/auth/rotate")
async def rotate_api_key(
    request: Request,
    authorization: Optional[str] = Header(None),
):
    """Rotate the user's encrypted provider key and invalidate old sessions."""
    user, _ = await verify_bearer_token(authorization)
    try:
        body = await _read_request_body_limited(request, 1 * 1024 * 1024)
        rotation = RotateApiKeyRequest(**json.loads(body))
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {exc}") from exc

    client = OpenRouterClient(
        api_key=rotation.openrouter_api_key,
        base_url=settings.openrouter_base_url,
        embedding_base_url=settings.openrouter_embedding_base_url,
        rerank_base_url=settings.openrouter_rerank_base_url,
        llm_model=settings.llm_model,
        embedding_model=settings.embedding_model,
        reranker_model=settings.reranker_model,
    )
    try:
        is_valid, error = await client.verify_api_key()
    finally:
        await client.close()
    if not is_valid:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid OpenRouter-compatible endpoint or API key: {error}",
        )

    encrypted_api_key = token_manager.encrypt_api_key(rotation.openrouter_api_key)
    rotated = await asyncio.to_thread(
        user_store.rotate_api_key, user.user_id, encrypted_api_key
    )
    if not rotated:
        raise HTTPException(status_code=404, detail="User not found")
    updated_user = await asyncio.to_thread(user_store.get_user, user.user_id)
    if not updated_user:
        raise HTTPException(status_code=404, detail="User not found")
    await _invalidate_user_sessions(user.user_id)
    return {
        "success": True,
        "user_id": updated_user.user_id,
        "token": token_manager.generate_token(updated_user),
    }


# === Health & Info ===


@app.get("/api/health")
async def health_check():
    checks: dict[str, bool] = {"healthy": True, "status": True}
    try:
        user_store.count_users()
        checks["users_db"] = True
    except Exception:
        logger.exception("health_check_component_failed component=users_db")
        checks["users_db"] = False
        checks["healthy"] = False
    try:
        vector_store.table_names()
        checks["lancedb"] = True
    except Exception:
        logger.exception("health_check_component_failed component=lancedb")
        checks["lancedb"] = False
        checks["healthy"] = False
    try:
        store = MCPHandler._durable_store
        if store is not None:
            store.queue_counts()
            checks["queue_db"] = True
    except Exception:
        logger.exception("health_check_component_failed component=queue_db")
        checks["queue_db"] = False
        checks["healthy"] = False
    if not checks["healthy"]:
        return {
            "status": "degraded",
            "checks": checks,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "version": "1.0.0",
        }
    return {
        "status": "healthy",
        "checks": checks,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": "1.0.0",
    }


@app.get("/api/server/info")
async def server_info():
    """Get server information"""
    return {
        "name": "SimpleMem MCP Server",
        "version": "1.0.0",
        "protocol_version": "2025-03-26",
        "transport": "Streamable HTTP",
        "reranker_model": settings.reranker_model,
        "llm_model": settings.llm_model,
        "embedding_model": settings.embedding_model,
        "window_size": settings.window_size,
        "active_sessions": len(_sessions),
        "total_users": user_store.count_users(),
    }


# === MCP Protocol Endpoints (Streamable HTTP - 2025-03-26 spec) ===


def _accepts_transport(request: Request, *mime_types: str) -> bool:
    """Treat generic clients as compatible while still honoring explicit Accept headers."""
    accept = request.headers.get("accept", "").strip().lower()
    if not accept or accept == "*/*":
        return True
    return any(mime_type in accept for mime_type in mime_types)


def _get_mcp_handler(user: User, api_key: str) -> MCPHandler:
    """Get or create MCP handler for user (legacy)"""
    handler = _mcp_handlers.get(user.user_id)
    if handler is None or handler.api_key != api_key:
        _mcp_handlers[user.user_id] = MCPHandler(
            user=user,
            api_key=api_key,
            vector_store=vector_store,
            client_manager=client_manager,
            settings=settings,
        )
    return _mcp_handlers[user.user_id]


def _is_initialize_request(data: dict | list) -> bool:
    """Check if the message is an initialize request"""
    if isinstance(data, list):
        return any(
            isinstance(item, dict) and item.get("method") == "initialize"
            for item in data
        )
    return isinstance(data, dict) and data.get("method") == "initialize"


def _is_notification_or_response_only(data: dict | list) -> bool:
    """Check if message contains only notifications or responses (no requests)"""
    items = data if isinstance(data, list) else [data]
    for item in items:
        if not isinstance(item, dict):
            return False
        # Has 'method' but no 'id' -> notification
        # Has 'result' or 'error' -> response
        # Has both 'method' and 'id' -> request
        if "method" in item and "id" in item:
            return False  # This is a request
        if "method" not in item and "result" not in item and "error" not in item:
            return False
    return True


@app.post("/mcp")
async def mcp_post_endpoint(
    request: Request,
    authorization: Optional[str] = Header(None),
    mcp_session_id: Optional[str] = Header(None, alias="Mcp-Session-Id"),
):
    """
    Streamable HTTP POST endpoint (MCP 2025-03-26 spec).

    Handles JSON-RPC 2.0 messages from clients.

    Headers:
    - Authorization: Bearer <token> (required)
    - Accept: application/json, text/event-stream (required)
    - Mcp-Session-Id: <session-id> (required after initialization)

    Request body: one JSON-RPC request, notification, or response object.

    Response:
    - For notifications/responses only: 202 Accepted
    - For requests: JSON response or SSE stream
    """
    # Validate Accept header
    if not _accepts_transport(request, "application/json", "text/event-stream"):
        raise HTTPException(
            status_code=406,
            detail="Accept header must include application/json or text/event-stream",
        )

    # Authenticate
    user, api_key = await verify_bearer_token(authorization)
    await maybe_update_last_active(user.user_id)

    # Parse request body
    try:
        body = await _read_request_body_limited(request, 10 * 1024 * 1024)
        data = json.loads(body.decode("utf-8"))
    except json.JSONDecodeError as e:
        return JSONResponse(
            status_code=400,
            content={
                "jsonrpc": "2.0",
                "error": {"code": -32700, "message": f"Parse error: {e}"},
            },
        )

    # Handle initialization (creates new session)
    if _is_initialize_request(data):
        session = await get_or_create_session(user, api_key)
        session.initialized = True

        # Process the initialize request
        response_str = await session.handler.handle_message(json.dumps(data))
        response_data = json.loads(response_str)

        # Return with Mcp-Session-Id header
        return JSONResponse(
            content=response_data,
            headers={"Mcp-Session-Id": session.session_id},
        )

    # For non-initialization requests, session ID is required
    if not mcp_session_id:
        raise HTTPException(
            status_code=400,
            detail="Mcp-Session-Id header required for non-initialization requests",
        )

    # Get existing session
    session = await get_session(mcp_session_id)
    if not session:
        raise HTTPException(
            status_code=404,
            detail="Session expired, re-initialize required",
        )

    # Verify session belongs to this user
    if session.user_id != user.user_id:
        raise HTTPException(
            status_code=403, detail="Session does not belong to this user"
        )

    # If only notifications or responses, return 202 Accepted
    if _is_notification_or_response_only(data):
        # Still process them (e.g., initialized notification)
        await session.handler.handle_message(json.dumps(data))
        return Response(status_code=202)

    # Process request(s) and return response
    # For now, we return JSON. SSE streaming can be added for long-running tools.
    response_str = await session.handler.handle_message(json.dumps(data))
    response_data = json.loads(response_str)

    return JSONResponse(
        content=response_data,
        headers={"Mcp-Session-Id": session.session_id},
    )


@app.get("/mcp")
async def mcp_get_endpoint(
    request: Request,
    authorization: Optional[str] = Header(None),
    mcp_session_id: Optional[str] = Header(None, alias="Mcp-Session-Id"),
    last_event_id: Optional[str] = Header(None, alias="Last-Event-ID"),
):
    """
    Streamable HTTP GET endpoint for server-to-client SSE stream.

    Used for server-initiated messages (notifications, requests to client).

    Headers:
    - Authorization: Bearer <token> (required)
    - Accept: text/event-stream (required)
    - Mcp-Session-Id: <session-id> (required)
    - Last-Event-ID: <event-id> (optional, for resumability)
    """
    # Validate Accept header
    if not _accepts_transport(request, "text/event-stream"):
        raise HTTPException(
            status_code=406,
            detail="Accept header must include text/event-stream",
        )

    # Authenticate
    user, api_key = await verify_bearer_token(authorization)

    # Session ID required for GET
    if not mcp_session_id:
        raise HTTPException(
            status_code=400,
            detail="Mcp-Session-Id header required",
        )

    # Get session
    session = await get_session(mcp_session_id)
    if not session:
        raise HTTPException(
            status_code=404,
            detail="Session not found or expired",
        )

    # Verify session belongs to this user
    if session.user_id != user.user_id:
        raise HTTPException(
            status_code=403, detail="Session does not belong to this user"
        )

    # Generate unique stream ID
    stream_id = secrets.token_urlsafe(16)
    session.active_streams.add(stream_id)

    async def event_generator():
        """Generate SSE events for server-to-client messages"""
        try:
            # Send initial keepalive
            yield ": keepalive\n\n"

            while True:
                try:
                    # Wait for messages with timeout for keepalive
                    message = await asyncio.wait_for(
                        session.message_queue.get(),
                        timeout=15.0,
                    )

                    # Format as SSE event
                    event_id = session.next_event_id()
                    yield f"id: {event_id}\n"
                    yield "event: message\n"
                    yield f"data: {json.dumps(message)}\n\n"

                except asyncio.TimeoutError:
                    # Send keepalive comment
                    yield ": keepalive\n\n"

        finally:
            # Remove stream from active set
            session.active_streams.discard(stream_id)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "Mcp-Session-Id": session.session_id,
        },
    )


@app.delete("/mcp")
async def mcp_delete_endpoint(
    authorization: Optional[str] = Header(None),
    mcp_session_id: Optional[str] = Header(None, alias="Mcp-Session-Id"),
):
    """
    Terminate an MCP session.

    Headers:
    - Authorization: Bearer <token> (required)
    - Mcp-Session-Id: <session-id> (required)
    """
    # Authenticate
    user, api_key = await verify_bearer_token(authorization)

    if not mcp_session_id:
        raise HTTPException(status_code=400, detail="Mcp-Session-Id header required")

    # Get session to verify ownership
    session = await get_session(mcp_session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    if session.user_id != user.user_id:
        raise HTTPException(
            status_code=403, detail="Session does not belong to this user"
        )

    # Delete session
    await delete_session(mcp_session_id)
    return Response(status_code=204)


# === Legacy MCP Endpoints (HTTP+SSE - 2024-11-05 spec) ===
# Kept for backwards compatibility with older clients


@app.get("/mcp/sse")
async def mcp_sse_endpoint_legacy(
    request: Request,
    authorization: Optional[str] = Header(None),
):
    """
    Legacy MCP over Server-Sent Events (SSE) endpoint.

    DEPRECATED: Use Streamable HTTP at /mcp instead.

    Authentication via Authorization: Bearer <token> header only.
    Query parameter auth removed due to token leakage in logs/headers.
    """
    if not authorization:
        raise HTTPException(
            status_code=401,
            detail="Authorization header required. Use: Bearer <token>",
            headers={"WWW-Authenticate": "Bearer"},
        )
    user, api_key = await verify_bearer_token(authorization)

    # Create session for legacy client
    session = await get_or_create_session(user, api_key)

    # Get base URL
    base_url = os.getenv("MCP_BASE_URL", "")
    message_endpoint = f"{base_url}/mcp/message" if base_url else "/mcp/message"

    async def event_generator():
        """Generate SSE events"""
        # Send endpoint info as first event (legacy format)
        endpoint_url = f"{message_endpoint}?session_id={session.session_id}"
        yield f"event: endpoint\ndata: {endpoint_url}\n\n"

        # Send initial keepalive
        yield ": keepalive\n\n"

        # Keep connection alive
        while True:
            await asyncio.sleep(15)
            yield ": keepalive\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "X-Content-Type-Options": "nosniff",
        },
    )


@app.post("/mcp/message")
async def mcp_message_endpoint_legacy(
    request: Request,
    session_id: Optional[str] = Query(None, description="Session ID"),
    authorization: Optional[str] = Header(None),
    mcp_session_id: Optional[str] = Header(None, alias="Mcp-Session-Id"),
):
    """
    Legacy MCP message endpoint.

    DEPRECATED: Use Streamable HTTP POST to /mcp instead.

    Authentication via Authorization: Bearer <token> header only.
    Query parameter auth removed due to token leakage in logs/headers.
    """
    if not authorization:
        raise HTTPException(
            status_code=401,
            detail="Authorization header required. Use: Bearer <token>",
            headers={"WWW-Authenticate": "Bearer"},
        )
    user, api_key = await verify_bearer_token(authorization)

    # Get session ID from header or query param
    sid = mcp_session_id or session_id
    if sid:
        session = await get_session(sid)
        if session and session.user_id == user.user_id:
            handler = session.handler
        else:
            handler = _get_mcp_handler(user, api_key)
    else:
        handler = _get_mcp_handler(user, api_key)

    await maybe_update_last_active(user.user_id)

    # Process message
    body = await _read_request_body_limited(request, 10 * 1024 * 1024)
    raw_message = body.decode("utf-8")
    response = await handler.handle_message(raw_message)
    try:
        parsed_message = json.loads(raw_message)
    except (UnicodeDecodeError, json.JSONDecodeError):
        parsed_message = None
    if isinstance(parsed_message, (dict, list)) and _is_notification_or_response_only(
        parsed_message
    ):
        return Response(status_code=202)

    return json.loads(response)


# === Static Files (Frontend) ===

frontend_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
if os.path.exists(frontend_path):
    app.mount("/static", StaticFiles(directory=frontend_path), name="static")


@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve frontend HTML"""
    index_path = os.path.join(frontend_path, "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(
        content="<h1>SimpleMem MCP Server</h1><p>Frontend not found.</p>"
    )


# === Entry Point ===


def run_server(host: str = "127.0.0.1", port: int = 8000):
    """Run the HTTP server"""
    import uvicorn

    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("SimpleMem MCP must bind to a loopback host")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()
