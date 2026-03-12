"""
Settings configuration for SimpleMem MCP Server
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from functools import lru_cache


def _load_env_file():
    """Load .env file manually if it exists"""
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ.setdefault(key.strip(), value.strip())


_load_env_file()


@dataclass
class Settings:
    """Application settings"""

    # Server Configuration
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False

    # JWT Configuration
    jwt_secret_key: str = field(default_factory=lambda: os.getenv(
        "JWT_SECRET_KEY",
        "simplemem-secret-key-change-in-production"
    ))
    jwt_algorithm: str = "HS256"
    jwt_expiration_days: int = 30

    # Encryption for API Keys
    encryption_key: str = field(default_factory=lambda: os.getenv(
        "ENCRYPTION_KEY",
        "simplemem-encryption-key-32bytes!"  # Must be 32 bytes for AES-256
    ))

    # Database Paths
    data_dir: str = field(default_factory=lambda: os.getenv(
        "DATA_DIR",
        "./data"
    ))
    lancedb_path: str = field(default_factory=lambda: os.getenv(
        "LANCEDB_PATH",
        "./data/lancedb"
    ))
    user_db_path: str = field(default_factory=lambda: os.getenv(
        "USER_DB_PATH",
        "./data/users.db"
    ))

    # LLM Provider Configuration
    llm_provider: str = field(default_factory=lambda: os.getenv(
        "LLM_PROVIDER",
        "openrouter"  # Options: "openrouter", "ollama"
    ))

    # OpenRouter Configuration (used when llm_provider is "openrouter")
    openrouter_base_url: str = field(default_factory=lambda: os.getenv(
        "OPENROUTER_BASE_URL",
        "https://openrouter.ai/api/v1"
    ))
    openrouter_embedding_base_url: str = field(default_factory=lambda: os.getenv(
        "OPENROUTER_EMBED_BASE_URL",
        os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    ))

    # Ollama Configuration (used when llm_provider is "ollama")
    ollama_base_url: str = field(default_factory=lambda: os.getenv(
        "OLLAMA_BASE_URL",
        "http://localhost:11434/v1"
    ))
    ollama_embedding_base_url: str = field(default_factory=lambda: os.getenv(
        "OLLAMA_EMBED_BASE_URL",
        os.getenv("OLLAMA_EMBEDDING_BASE_URL", os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"))
    ))

    # Common LLM Configuration
    llm_model: str = field(default_factory=lambda: os.getenv(
        "LLM_MODEL",
        "openai/gpt-4.1-mini" # "qwen3:4b-instruct" Default model name
    ))
    embedding_model: str = field(default_factory=lambda: os.getenv(
        "EMBEDDING_MODEL",
        "qwen3-embedding:4b"  # Default embedding model
    ))
    embedding_dimension: int = field(default_factory=lambda: int(os.getenv(
        "EMBEDDING_DIMENSION",
        "2560"  # Default: 2560 for qwen3-embedding:4b, 768 for nomic-embed-text
    )))

    # Memory Building Configuration
    window_size: int = 20
    overlap_size: int = 2

    # Retrieval Configuration
    semantic_top_k: int = 25
    keyword_top_k: int = 5
    enable_planning: bool = field(default_factory=lambda: os.getenv(
        "ENABLE_PLANNING",
        "0",
    ).strip().lower() in {"1", "true", "yes", "on"})
    enable_reflection: bool = field(default_factory=lambda: os.getenv(
        "ENABLE_REFLECTION",
        "0",
    ).strip().lower() in {"1", "true", "yes", "on"})
    max_reflection_rounds: int = 2

    # LLM Configuration
    llm_temperature: float = 0.1
    llm_max_retries: int = field(default_factory=lambda: int(os.getenv(
        "LLM_MAX_RETRIES",
        "3",
    )))
    use_streaming: bool = True

    # memory_query guardrails (prevent oversized synthesis payloads / 413s)
    memory_query_default_top_k: int = field(default_factory=lambda: int(os.getenv(
        "MEMORY_QUERY_DEFAULT_TOP_K",
        "8",
    )))
    memory_query_max_top_k: int = field(default_factory=lambda: int(os.getenv(
        "MEMORY_QUERY_MAX_TOP_K",
        "16",
    )))
    memory_query_max_context_entries: int = field(default_factory=lambda: int(os.getenv(
        "MEMORY_QUERY_MAX_CONTEXT_ENTRIES",
        "10",
    )))
    memory_query_max_context_entry_chars: int = field(default_factory=lambda: int(os.getenv(
        "MEMORY_QUERY_MAX_CONTEXT_ENTRY_CHARS",
        "320",
    )))
    memory_query_max_context_total_chars: int = field(default_factory=lambda: int(os.getenv(
        "MEMORY_QUERY_MAX_CONTEXT_TOTAL_CHARS",
        "3500",
    )))

    # Async write path for memory_add / memory_add_batch
    memory_add_async_default: bool = field(default_factory=lambda: os.getenv(
        "MEMORY_ADD_ASYNC_DEFAULT",
        "1",
    ).strip().lower() in {"1", "true", "yes", "on"})
    memory_write_queue_size: int = field(default_factory=lambda: int(os.getenv(
        "MEMORY_WRITE_QUEUE_SIZE",
        "512",
    )))
    memory_write_workers: int = field(default_factory=lambda: int(os.getenv(
        "MEMORY_WRITE_WORKERS",
        "6",
    )))
    memory_write_batch_max: int = field(default_factory=lambda: int(os.getenv(
        "MEMORY_WRITE_BATCH_MAX",
        "24",
    )))
    memory_write_batch_max_wait_ms: int = field(default_factory=lambda: int(os.getenv(
        "MEMORY_WRITE_BATCH_MAX_WAIT_MS",
        "40",
    )))
    memory_write_lease_seconds: int = field(default_factory=lambda: int(os.getenv(
        "MEMORY_WRITE_LEASE_SECONDS",
        "90",
    )))
    memory_write_max_retries: int = field(default_factory=lambda: int(os.getenv(
        "MEMORY_WRITE_MAX_RETRIES",
        "6",
    )))
    memory_write_retry_backoff_base_ms: int = field(default_factory=lambda: int(os.getenv(
        "MEMORY_WRITE_RETRY_BACKOFF_BASE_MS",
        "250",
    )))
    memory_queue_max_inflight_writes: int = field(default_factory=lambda: int(os.getenv(
        "MEMORY_QUEUE_MAX_INFLIGHT_WRITES",
        "20000",
    )))
    memory_queue_max_inflight_reads: int = field(default_factory=lambda: int(os.getenv(
        "MEMORY_QUEUE_MAX_INFLIGHT_READS",
        "48",
    )))
    memory_write_job_status_limit: int = field(default_factory=lambda: int(os.getenv(
        "MEMORY_WRITE_JOB_STATUS_LIMIT",
        "2000",
    )))
    memory_sync_wait_timeout_ms: int = field(default_factory=lambda: int(os.getenv(
        "MEMORY_SYNC_WAIT_TIMEOUT_MS",
        "1200",
    )))
    durable_queue_db_path: str = field(default_factory=lambda: os.getenv(
        "DURABLE_QUEUE_DB_PATH",
        "./data/durable_jobs.db",
    ))

    def __post_init__(self):
        """Ensure directories exist; use absolute paths so cwd and permissions are predictable."""
        data_dir = os.path.abspath(os.path.expanduser(self.data_dir))
        lancedb_path = os.path.abspath(os.path.expanduser(self.lancedb_path))
        self.data_dir = data_dir
        self.lancedb_path = lancedb_path
        self.durable_queue_db_path = os.path.abspath(os.path.expanduser(self.durable_queue_db_path))
        try:
            os.makedirs(self.data_dir, exist_ok=True)
            os.makedirs(self.lancedb_path, exist_ok=True)
            os.makedirs(os.path.dirname(self.durable_queue_db_path), exist_ok=True)
        except PermissionError as e:
            raise PermissionError(
                f"Cannot create data dir(s): {e}. "
                "In Docker, use a named volume for data (see docker-compose.yml) or ensure the mounted dir is writable by the container user."
            ) from e
        self.llm_max_retries = max(1, int(self.llm_max_retries))
        self.memory_write_queue_size = max(8, int(self.memory_write_queue_size))
        self.memory_write_workers = max(1, int(self.memory_write_workers))
        self.memory_write_batch_max = max(1, int(self.memory_write_batch_max))
        self.memory_write_batch_max_wait_ms = max(5, int(self.memory_write_batch_max_wait_ms))
        self.memory_write_lease_seconds = max(10, int(self.memory_write_lease_seconds))
        self.memory_write_max_retries = max(1, int(self.memory_write_max_retries))
        self.memory_write_retry_backoff_base_ms = max(50, int(self.memory_write_retry_backoff_base_ms))
        self.memory_queue_max_inflight_writes = max(10, int(self.memory_queue_max_inflight_writes))
        self.memory_queue_max_inflight_reads = max(1, int(self.memory_queue_max_inflight_reads))
        self.memory_write_job_status_limit = max(100, int(self.memory_write_job_status_limit))
        self.memory_sync_wait_timeout_ms = max(50, int(self.memory_sync_wait_timeout_ms))
        self.memory_query_default_top_k = max(1, int(self.memory_query_default_top_k))
        self.memory_query_max_top_k = max(self.memory_query_default_top_k, int(self.memory_query_max_top_k))
        self.memory_query_max_context_entries = max(1, int(self.memory_query_max_context_entries))
        self.memory_query_max_context_entry_chars = max(80, int(self.memory_query_max_context_entry_chars))
        self.memory_query_max_context_total_chars = max(
            self.memory_query_max_context_entry_chars,
            int(self.memory_query_max_context_total_chars),
        )


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()
