"""
Data models for authentication
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional
import hashlib
import uuid


def _naive_utc_now() -> datetime:
    """Preserve the existing naive-UTC serialization contract."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


@dataclass
class User:
    """User model for multi-tenant isolation"""

    user_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    openrouter_api_key_encrypted: str = ""
    table_name: str = ""
    created_at: datetime = field(default_factory=_naive_utc_now)
    last_active: datetime = field(default_factory=_naive_utc_now)
    token_version: int = 0

    def __post_init__(self):
        if not self.table_name:
            # Generate unique table name for this user
            safe_id = self.user_id.replace("-", "_")
            if safe_id != self.user_id:
                safe_id = f"{safe_id}_{hashlib.sha256(self.user_id.encode()).hexdigest()[:12]}"
            self.table_name = f"mem_{safe_id}"

    def to_dict(self) -> dict:
        return {
            "user_id": self.user_id,
            "openrouter_api_key_encrypted": self.openrouter_api_key_encrypted,
            "table_name": self.table_name,
            "created_at": self.created_at.isoformat(),
            "last_active": self.last_active.isoformat(),
            "token_version": self.token_version,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "User":
        return cls(
            user_id=data["user_id"],
            openrouter_api_key_encrypted=data["openrouter_api_key_encrypted"],
            table_name=data["table_name"],
            created_at=datetime.fromisoformat(data["created_at"]),
            last_active=datetime.fromisoformat(data["last_active"]),
            token_version=int(data.get("token_version", 0)),
        )


@dataclass
class TokenPayload:
    """JWT token payload structure"""

    user_id: str
    table_name: str
    created_at: str
    exp: int  # Expiration timestamp
    token_version: int = 0

    def to_dict(self) -> dict:
        return {
            "user_id": self.user_id,
            "table_name": self.table_name,
            "created_at": self.created_at,
            "exp": self.exp,
            "token_version": self.token_version,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TokenPayload":
        return cls(
            user_id=data["user_id"],
            table_name=data["table_name"],
            created_at=data["created_at"],
            exp=data["exp"],
            token_version=int(data.get("token_version", 0)),
        )


@dataclass
class MemoryEntry:
    """Atomic memory entry - self-contained fact"""

    entry_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    lossless_restatement: str = ""
    keywords: list = field(default_factory=list)
    timestamp: Optional[str] = None
    location: Optional[str] = None
    persons: list = field(default_factory=list)
    entities: list = field(default_factory=list)
    topic: Optional[str] = None
    # Source-owned metadata is retained separately from retrieval/reranking
    # fields so exact identity reads can prove lifecycle state without changing
    # ranking inputs.
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "entry_id": self.entry_id,
            "lossless_restatement": self.lossless_restatement,
            "keywords": self.keywords,
            "timestamp": self.timestamp,
            "location": self.location,
            "persons": self.persons,
            "entities": self.entities,
            "topic": self.topic,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MemoryEntry":
        return cls(
            entry_id=data.get("entry_id", str(uuid.uuid4())),
            lossless_restatement=data.get("lossless_restatement", ""),
            keywords=data.get("keywords", []),
            timestamp=data.get("timestamp"),
            location=data.get("location"),
            persons=data.get("persons", []),
            entities=data.get("entities", []),
            topic=data.get("topic"),
            metadata=(
                dict(data.get("metadata", {}))
                if isinstance(data.get("metadata", {}), dict)
                else {}
            ),
        )


@dataclass
class Dialogue:
    """Input dialogue model"""

    dialogue_id: int
    speaker: str
    content: str
    timestamp: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "dialogue_id": self.dialogue_id,
            "speaker": self.speaker,
            "content": self.content,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Dialogue":
        return cls(
            dialogue_id=data.get("dialogue_id", 0),
            speaker=data.get("speaker", ""),
            content=data.get("content", ""),
            timestamp=data.get("timestamp"),
        )
