"""
Online Semantic Synthesis (Section 3.2)

Consolidates related memory fragments during the write phase to eliminate
redundancy. For each new entry, searches the existing store for semantically
similar entries and merges them via LLM when similarity exceeds a threshold.

Controlled by config flags:
  ENABLE_SYNTHESIS (default False)
  SYNTHESIS_SIMILARITY_THRESHOLD (default 0.85)
  SYNTHESIS_MAX_CANDIDATES (default 10)
"""
from dataclasses import dataclass, field
from typing import List, Optional

import config
from database.vector_store import VectorStore
from models.memory_entry import MemoryEntry
from utils.embedding import EmbeddingModel
from utils.llm_client import LLMClient


@dataclass
class SynthesisResult:
    """Outcome of a synthesis pass over a batch of new entries."""
    entries_to_store: List[MemoryEntry] = field(default_factory=list)
    entries_to_delete: List[str] = field(default_factory=list)
    merged_count: int = 0
    passthrough_count: int = 0


class SemanticSynthesizer:
    """
    Stage 2: Online Semantic Synthesis

    For each new memory entry, searches the vector store for existing entries
    with high semantic similarity. When a match is found, merges the two
    entries via LLM to produce a single, consolidated entry that preserves
    all information from both sources.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        vector_store: VectorStore,
        embedding_model: EmbeddingModel = None,
        similarity_threshold: float = None,
        max_candidates: int = None,
    ):
        self.llm_client = llm_client
        self.vector_store = vector_store
        self.embedding_model = embedding_model or vector_store.embedding_model
        self.similarity_threshold = (
            similarity_threshold
            if similarity_threshold is not None
            else getattr(config, 'SYNTHESIS_SIMILARITY_THRESHOLD', 0.85)
        )
        self.max_candidates = (
            max_candidates
            if max_candidates is not None
            else getattr(config, 'SYNTHESIS_MAX_CANDIDATES', 10)
        )

    def synthesize(self, new_entries: List[MemoryEntry]) -> SynthesisResult:
        """
        Main entry point. For each new entry, check if a similar entry
        already exists in the store. If so, merge them via LLM.

        Returns a SynthesisResult with:
          - entries_to_store: entries to add (merged or original)
          - entries_to_delete: entry_ids of old entries that were merged
          - merged_count / passthrough_count
        """
        result = SynthesisResult()

        if not new_entries:
            return result

        # Batch-encode all new entries
        texts = [e.lossless_restatement for e in new_entries]
        vectors = self.embedding_model.encode_documents(texts)

        for entry, vector in zip(new_entries, vectors):
            similar = self.vector_store.find_similar_entries(
                query_vector=vector.tolist(),
                top_k=self.max_candidates,
                similarity_threshold=self.similarity_threshold,
            )

            if not similar:
                # No similar existing entry — pass through as-is
                result.entries_to_store.append(entry)
                result.passthrough_count += 1
                continue

            # Take the most similar existing entry
            best_match = max(similar, key=lambda s: s["cosine_similarity"])
            existing_entry = best_match["entry"]

            merged = self._merge_entries_via_llm(existing_entry, entry)
            if merged is not None:
                result.entries_to_store.append(merged)
                result.entries_to_delete.append(existing_entry.entry_id)
                result.merged_count += 1
                print(
                    f"[Synthesis] Merged new entry with existing {existing_entry.entry_id[:8]}... "
                    f"(similarity={best_match['cosine_similarity']:.3f})"
                )
            else:
                # Merge failed — store new entry as-is, keep existing
                result.entries_to_store.append(entry)
                result.passthrough_count += 1

        return result

    def _merge_entries_via_llm(
        self, existing: MemoryEntry, new: MemoryEntry
    ) -> Optional[MemoryEntry]:
        """
        Merge two semantically similar entries via LLM.

        The merged entry reuses the existing entry's entry_id for continuity.
        Retries up to 2 times on failure; returns None if all attempts fail.
        """
        prompt = f"""You are merging two memory entries that describe similar or overlapping information.
Combine them into a SINGLE memory entry that preserves ALL information from both.

[Existing Entry]
- Restatement: {existing.lossless_restatement}
- Keywords: {existing.keywords}
- Timestamp: {existing.timestamp}
- Location: {existing.location}
- Persons: {existing.persons}
- Entities: {existing.entities}
- Topic: {existing.topic}

[New Entry]
- Restatement: {new.lossless_restatement}
- Keywords: {new.keywords}
- Timestamp: {new.timestamp}
- Location: {new.location}
- Persons: {new.persons}
- Entities: {new.entities}
- Topic: {new.topic}

[Requirements]
1. The merged lossless_restatement must be a SINGLE complete sentence preserving ALL facts from both entries.
2. Combine keywords, persons, entities (deduplicate).
3. Keep the more specific timestamp, location, and topic.
4. Do NOT lose any information.

Return ONLY a JSON object:
```json
{{
  "lossless_restatement": "merged complete sentence",
  "keywords": ["keyword1", "keyword2"],
  "timestamp": "YYYY-MM-DDTHH:MM:SS or null",
  "location": "location or null",
  "persons": ["name1", "name2"],
  "entities": ["entity1", "entity2"],
  "topic": "topic phrase"
}}
```"""

        messages = [
            {
                "role": "system",
                "content": "You merge memory entries into a single consolidated entry. Output valid JSON only.",
            },
            {"role": "user", "content": prompt},
        ]

        max_retries = 2
        for attempt in range(max_retries):
            try:
                response_format = None
                if hasattr(config, 'USE_JSON_FORMAT') and config.USE_JSON_FORMAT:
                    response_format = {"type": "json_object"}

                response = self.llm_client.chat_completion(
                    messages, temperature=0.1, response_format=response_format
                )
                data = self.llm_client.extract_json(response)

                merged = MemoryEntry(
                    entry_id=existing.entry_id,  # reuse existing ID
                    lossless_restatement=data["lossless_restatement"],
                    keywords=data.get("keywords", []),
                    timestamp=data.get("timestamp"),
                    location=data.get("location"),
                    persons=data.get("persons", []),
                    entities=data.get("entities", []),
                    topic=data.get("topic"),
                )
                return merged

            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"[Synthesis] Merge attempt {attempt + 1} failed: {e}. Retrying...")
                else:
                    print(f"[Synthesis] All merge attempts failed: {e}. Storing new entry as-is.")

        return None
