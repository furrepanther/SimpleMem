# Supplemental adversarial review — `3839fd9`

Scope: metadata persistence and expiry cleanup for temporary SimpleMem review
evidence. This is not a promotion decision and does not close the
source-complete review gate.

## Immutable inputs

- Base: `e56827270cd13e699f178043c07bfa72036145ca`
- Candidate: `3839fd91ced8022088c7bc57a632d3e8cd8d36c1`
- Local review packet SHA-256:
  `50b353f4594d44ece45025133adc639acbb7d07c7985888e8a82cb7d69e32d3f`
- Adversarial checker result SHA-256:
  `33de90f159064733381f8789e19252db7647d8551478ff6313024ec7750cab9c`
- Grumpy result SHA-256:
  `bbe405750f81e310741de3f3262dfdbd3edca0c3dc2837301f9108dd297523e5`

Both reviewer invocations used only the local router at
`http://127.0.0.1:11888/v1`; their route evidence reported the
`primus-inference-router` adapter, `cloud_used=false`,
`direct_backend_used=false`, no truncation, and no timeout.

## Automated verification

- Focused lifecycle, retrieval, and vector-store tests: `29 passed`.
- Full current SimpleMem MCP suite: `138 passed`.
- `git diff --check`: passed.

The tests cover metadata round trip, legacy LanceDB additive-column migration,
expired/future/canonical cleanup discrimination, idempotency, durable
queue-to-worker propagation, exact-only metadata exposure, on-demand bounded
cleanup, and bounded background sweep receipts.

## SDLC and Grumpy outcomes

The adversarial checker reported no code finding, but returned
`insufficient_evidence` because the packet was deliberately supplemental
(`complete_packet=false`). Grumpy also returned `insufficient_evidence`; its
coverage was wrapper-scaffolded for the same reason. Neither result is a clean
gate or a production authorization.

## Solid Snake evidence ledger

| Field | Evidence |
|---|---|
| Observed failure | `memory_store` accepted metadata but dropped it before the durable payload and raw LanceDB entry; local exact retrieval also omitted it. This made retention proof and review receipt verification impossible. |
| Root-cause repair | `metadata_json` and `retention_purge_after_utc` are additive LanceDB fields. Metadata now crosses normalization, idempotency, durable queue, raw-entry creation, exact readback, and exact retention cleanup. |
| Hidden-path containment | Ineligible records store a null expiry; the cleanup query excludes blank values, then recomputes the metadata proof before every exact-ID delete and read-back confirmation. |
| Ranking containment | Semantic retrieval still uses the same vectors, candidate counts, reranking, and result shape. Metadata is returned only on `force_exact` local reads. |
| Concurrency containment | Schema migration, append, update, and delete use the existing per-table reentrant mutation lock. No second worker, timer service, process replacement, or liveness deadline is introduced. |
| Residual risk | Live SimpleMem is still older than this candidate. Production activation must quiesce the old writer first, apply the source pin under the approved handoff, and prove a live exact write/read/delete receipt before running the final source-complete review. |

## Required next proof

1. Review and promote this candidate only through the existing maintenance
   handoff and liveness fence; do not run it beside the live writer.
2. On the candidate, perform one controlled temporary-record lifecycle:
   durable write, terminal receipt, exact metadata readback, exact delete, and
   confirmed absence.
3. Run the source-complete SDLC-plus-Grumpy review with its verified temporary
   evidence pointer, then release the pointer only through an exact delete
   receipt or its verified expiry sweep.
