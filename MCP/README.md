# SimpleMem MCP Server

**Production-Ready Memory Service for LLM Agents via Model Context Protocol (MCP)**

SimpleMem MCP Server is a local-only long-term memory service for LLM agents, implementing the **Streamable HTTP** transport (MCP 2025-03-26 spec). It enables MCP-compatible clients to store, retrieve, and query conversational memories without direct cloud-provider inference.

## Features

- **Semantic Lossless Compression**: Converts dialogues into atomic, self-contained facts
- **Coreference Resolution**: Automatically replaces pronouns (he/she/it) with actual names
- **Temporal Anchoring**: Converts relative times (tomorrow, next week) to absolute timestamps
- **Hybrid Retrieval**: Semantic search + keyword matching + metadata filtering
- **Intelligent Planning**: Automatic query decomposition and reflection for complex queries
- **Multi-tenant Isolation**: Per-user data tables with token authentication
- **Local-only Inference**: Chat, embedding, and reranking use configured loopback endpoints only
- **Production Optimized**: Faster response times compared to the academic reference implementation

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    SimpleMem MCP Server                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              HTTP Server (FastAPI)                        │  │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────────────┐  │  │
│  │  │  Web UI    │  │  REST API  │  │  MCP Streamable    │  │  │
│  │  │  (/)       │  │  (/api/*)  │  │  HTTP (/mcp)       │  │  │
│  │  └────────────┘  └────────────┘  └────────────────────┘  │  │
│  └──────────────────────────────────────────────────────────┘  │
│                              │                                  │
│                              ▼                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                  Token Authentication                     │  │
│  │            (JWT + AES-256 Encrypted API Keys)            │  │
│  └──────────────────────────────────────────────────────────┘  │
│                              │                                  │
│         ┌────────────────────┼────────────────────┐            │
│         ▼                    ▼                    ▼            │
│  ┌────────────┐       ┌────────────┐       ┌────────────┐     │
│  │  User A    │       │  User B    │       │  User C    │     │
│  │  Table     │       │  Table     │       │  Table     │     │
│  └────────────┘       └────────────┘       └────────────┘     │
│  └─────────────────── LanceDB ──────────────────────────┘     │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │          OpenAI-compatible Local Inference                │  │
│  │  LLM: Granite 4.0 H 1B       Embed: BAAI/bge-m3           │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start

### Production Deployment (Local-only)

This production fork never calls a cloud provider directly. It uses loopback
endpoints for Granite chat inference, BGE-M3 embeddings, and the BGE reranker.
The canonical model root is `/var/lib/models/llama`; model-server ownership and
lifecycle remain outside SimpleMem.

### Self-Hosting

#### 1. Install Dependencies

```bash
cd MCP
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

#### 2. Configure Environment Variables (Optional)

```bash
# Production environment recommended settings
export JWT_SECRET_KEY="your-secure-random-secret-key"
export ENCRYPTION_KEY="your-32-byte-encryption-key!!"
export LLM_MODEL="granite-4.0-h-1b-Q4_K_S:latest"
export OPENROUTER_BASE_URL="http://127.0.0.1:11437/v1"
export OPENROUTER_EMBED_BASE_URL="http://127.0.0.1:11501/v1"
export OPENROUTER_RERANK_BASE_URL="http://127.0.0.1:11502/v1"
```

#### 3. Start the Server

```bash
python run.py
```

Output:
```
============================================================
  SimpleMem MCP Server
  Multi-tenant Memory Service for LLM Agents
============================================================

  Web UI:     http://localhost:8000/
  REST API:   http://localhost:8000/api/
  MCP:        http://localhost:8000/mcp

------------------------------------------------------------
```

## MCP Protocol

### Protocol Information

| Item | Value |
|------|-------|
| Protocol Version | 2025-03-26 |
| Transport | Streamable HTTP |
| Message Format | JSON-RPC 2.0 |
| Authentication | Bearer Token |

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/mcp` | POST | Send JSON-RPC messages (requests, notifications) |
| `/mcp` | GET | Server-to-client SSE stream |
| `/mcp` | DELETE | Terminate session |

### Authentication

All MCP requests require a Bearer token in the Authorization header:

```
Authorization: Bearer <your-token>
```

After initialization, include the session ID header:

```
Mcp-Session-Id: <session-id>
```

## MCP Tools

| Tool | Description |
|------|-------------|
| `memory_store` | Canonical exact raw durable write entrypoint for one memory entry |
| `memory_add` | Legacy compatibility alias for dialogue/extraction-style writes |
| `memory_add_batch` | Add multiple dialogues at once |
| `memory_query` | Query memories and generate AI-synthesized answers (with planning + hybrid retrieval + reflection) |
| `memory_retrieve` | Retrieve relevant memory entries (returns raw data) |
| `memory_delete` | Delete one exact memory by ID and return its receipt |
| `memory_update` | Update one exact memory by ID |
| `memory_cleanup_expired` | Sweep only expired, verified, non-canonical temporary review evidence and return exact-ID receipts |
| `memory_stats` | Get memory statistics |

## Client Configuration

Add to your MCP JSON settings:

```json
{
  "mcpServers": {
    "simplemem": {
      "url": "http://127.0.0.1:8000/mcp",
      "headers": {
        "Authorization": "Bearer YOUR_TOKEN"
      }
    }
  }
}
```


## How It Works

### Write Flow (Dialogue -> Memory)

```
Dialogue Input                  Processing                     Memory Storage
───────────────────────────────────────────────────────────────────────────

"I'll meet Bob            ┌─────────────────┐
 at Starbucks             │ LLM Processing  │
 tomorrow at 3pm"      ──▶│                 │ ──────────────▶  Atomic Fact
                          └─────────────────┘
                                                      │
                                                      ▼
                                              ┌─────────────────────────┐
                                              │ Atomic Fact:            │
                                              │ "User will meet Bob at  │
                                              │  Starbucks on           │
                                              │  2025-01-15 at 15:00"   │
                                              │                         │
                                              │ persons: [User, Bob]    │
                                              │ location: Starbucks     │
                                              │ timestamp: 2025-01-15   │
                                              │ topic: Meeting          │
                                              └───────────┬─────────────┘
                                                          │
                                                          ▼
                                              ┌─────────────────────────┐
                                              │      Embedding          │
                                              │   (BAAI/bge-m3)                    │
                                              └───────────┬─────────────┘
                                                          │
                                                          ▼
                                              ┌─────────────────────────┐
                                              │   LanceDB Vector Store  │
                                              └─────────────────────────┘
```

### Read Flow (Query -> Answer)

```
User Question: "When am I meeting Bob?"
                │
                ▼
┌───────────────────────────────┐
│  1. Query Complexity Analysis │
│     - Type: Temporal query    │
│     - Entity: Bob             │
│     - Complexity: 0.3 (simple)│
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│  2. Generate Search Queries   │
│     → "Bob meeting time"      │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│  3. Hybrid Retrieval          │
│     - Semantic (vector)       │
│     - Keyword (BM25)          │
│     - Metadata (persons)      │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│  4. Answer Generation         │
│     Context + Question → LLM  │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│  Response:                    │
│  {                            │
│    "answer": "15 January 2025 │
│              at 3:00 PM at    │
│              Starbucks",      │
│    "confidence": "high",      │
│    "contexts_used": 1         │
│  }                            │
└───────────────────────────────┘
```

## Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `window_size` | 20 | Number of dialogues per processing batch |
| `semantic_top_k` | 25 | Semantic search result count |
| `keyword_top_k` | 5 | Keyword search result count |
| `enable_planning` | true | Enable query planning |
| `enable_reflection` | true | Enable reflection iteration |
| `max_reflection_rounds` | 2 | Maximum reflection rounds |
| `llm_model` | granite-4.0-h-1b-Q4_K_S:latest | Loopback chat model |
| `embedding_model` | BAAI/bge-m3 | Embedding model |

### Temporary review-evidence lifecycle

`memory_store` preserves a JSON-object `metadata` field on exact local reads.
Metadata is not used as a retrieval or reranking input. A record is eligible for
automatic retention cleanup only when all of these fields are present and exact:
`temporary=true`, `canonical=false`, `retention_verified=true`,
`record_kind=temporary_review_evidence`, and a timezone-aware
`purge_after_utc` value.

The existing retention worker runs at `MEMORY_WRITE_PURGE_INTERVAL_S` (default
`300`) and uses bounded, exact-ID deletes. This is a data-retention schedule,
not a liveness deadline or process-termination mechanism. It never deletes
canonical memories, and it verifies each deletion before recording a receipt.

| Environment variable | Default | Description |
|---|---:|---|
| `MEMORY_TEMPORARY_EVIDENCE_PURGE_BATCH_SIZE` | `32` | Maximum expired review-evidence records inspected in one retention pass; clamped to 1–256. |
| `MEMORY_TEMPORARY_EVIDENCE_PURGE_MAX_TABLES` | `128` | Maximum user tables considered in one retention pass; clamped to 1–1000. |

## Development

```bash
# Development mode (auto-reload)
python run.py --reload

# Specify port
python run.py --port 3000

# View help
python run.py --help
```

## License

MIT License

## Note

Built upon SimpleMem research implementation, refactored and optimized for production deployment with multi-tenant support, faster processing, and comprehensive user isolation.
