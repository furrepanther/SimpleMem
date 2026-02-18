# Changelog

## [Unreleased]

### Added
- **Window Overlap Support**: `OVERLAP_SIZE` config parameter now actually retains dialogues between sliding windows for context continuity at window boundaries. Previously defined in config but unused in window advancement logic.
- **Online Semantic Synthesis (Stage 2)**: New `core/synthesis.py` module implementing Section 3.2 of the paper. Consolidates related memory fragments during write phase by detecting semantically similar entries (cosine similarity) and merging them via LLM. Controlled by `ENABLE_SYNTHESIS` flag (default: off).
- **VectorStore.find_similar_entries()**: Searches existing entries by vector cosine similarity with configurable threshold. Returns ranked matches with similarity scores.
- **VectorStore.delete_entry()**: Removes entries by ID, enabling synthesis merge-and-replace workflow.
- New config parameters: `ENABLE_SYNTHESIS`, `SYNTHESIS_SIMILARITY_THRESHOLD`, `SYNTHESIS_MAX_CANDIDATES`
- Unit tests for overlap behavior (`tests/test_overlap.py`)
- Unit tests for synthesis logic (`tests/test_synthesis.py`)
- Additional vector store tests for new methods (`tests/test_vector_store.py`)

### Fixed
- **OVERLAP_SIZE window advancement**: Buffer now advances by `window_size - overlap_size` instead of `window_size`, ensuring overlap dialogues are retained across windows for both sequential and parallel processing paths.

### Changed
- `MemoryBuilder.__init__` now accepts `overlap_size` parameter with validation (must be non-negative and less than window_size)
- `SimpleMemSystem` and `create_system()` now expose `window_size` and `overlap_size` parameters
- Import ordering in `core/memory_builder.py` follows stdlib-first convention
- Removed unused `asyncio` and `functools` imports from `core/memory_builder.py`
