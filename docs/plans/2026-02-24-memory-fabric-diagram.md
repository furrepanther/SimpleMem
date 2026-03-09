# SimpleMem Memory Fabric Diagram

This diagram shows the integration shape with a **careful splice point** (adapter/router layer), not blind wiring.

```mermaid
flowchart TB
    U[User / Agent Runtime<br/>Claude, Cursor, MCP, HTTP] --> O[CrossMemOrchestrator<br/>cross/orchestrator.py]

    O --> SM[SessionManager<br/>cross/session_manager.py]
    O --> CI[ContextInjector<br/>cross/context_injector.py]

    %% Write path
    SM --> EC[Event Collector + Summary<br/>cross/collectors.py + SQLite]
    SM --> MB[SimpleMem MemoryBuilder<br/>core/memory_builder.py]
    MB --> SP[Memory Fabric Splice Point<br/>NEW: adapter/router layer]

    SP --> LWS[Local Write Sink<br/>CrossSessionVectorStore + SQLite<br/>cross/storage_lancedb.py + cross/storage_sqlite.py]
    SP --> M0W[Mem0 Write Adapter<br/>NEW: mem0 backend adapter]

    %% Read path
    Q[Question / Prompt] --> HR[HybridRetriever<br/>core/hybrid_retriever.py]
    HR --> SPQ[Memory Fabric Query Router<br/>NEW: same splice layer]

    SPQ --> LRS[Local Read Sources<br/>semantic + keyword + structured]
    SPQ --> M0R[Mem0 Search Adapter<br/>NEW: mem0 backend adapter]
    SPQ --> XCI[Cross-Session Context<br/>ContextInjector bundle]

    LRS --> MR[Merge + Dedup + Rerank]
    M0R --> MR
    XCI --> MR

    MR --> CB[Context Budget Packer<br/>token caps + priority tiers]
    CB --> AG[AnswerGenerator<br/>core/answer_generator.py]
    AG --> RESP[Final Response]

    %% Observability
    O --> OBS[Metrics + Traces<br/>TTFT, retrieval p95, token budget hit rate]
    SP --> OBS
    SPQ --> OBS
```

## Splice Contract (tap-in point)

- **Input (write):** normalized `MemoryEntry` list from `MemoryBuilder`.
- **Output (write):** fan-out to local store and optional Mem0 store.
- **Input (read):** query + retrieval intent from `HybridRetriever`.
- **Output (read):** unified candidate memories with provenance + scores.

## Why this is a careful splice

- Existing SimpleMem modules remain primary; no core replacement.
- New layer is isolated and reversible.
- Local path remains authoritative fallback if Mem0 is down.
- All merges are explicit: dedupe, scoring normalization, provenance tracking.

