"""
Retriever - Stage 3: Adaptive Query-Aware Retrieval

Performs intelligent retrieval through:
- Query complexity analysis
- Multi-query planning
- Hybrid search (semantic + lexical + symbolic)
- Reflection-based iterative refinement

Refactored: Removed parallel processing for simplicity and stability.
"""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass

from ..auth.models import MemoryEntry
from ..database.vector_store import MultiTenantVectorStore

# Type alias for LLM client (duck-typed OpenRouter-compatible client).
LLMClient = object


@dataclass
class RetrievalPlan:
    """Query analysis and retrieval plan"""
    question_type: str
    key_entities: List[str]
    required_info: List[Dict[str, Any]]
    relationships: List[str]
    minimal_queries_needed: int
    complexity_score: float  # 0-1


class Retriever:
    """
    Adaptive retriever with intelligent query planning.
    Sequential processing for stability.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        vector_store: MultiTenantVectorStore,
        table_name: str,
        semantic_top_k: int = 25,
        keyword_top_k: int = 5,
        enable_planning: bool = True,
        enable_reflection: bool = True,
        enable_reranking: bool = True,
        rerank_candidate_cap: int = 100,
        rerank_document_max_chars: int = 1500,
        max_reflection_rounds: int = 2,
        temperature: float = 0.1,
    ):
        self.client = llm_client
        self.vector_store = vector_store
        self.table_name = table_name
        self.semantic_top_k = semantic_top_k
        self.keyword_top_k = keyword_top_k
        self.enable_planning = enable_planning
        self.enable_reflection = enable_reflection
        self.enable_reranking = enable_reranking
        self.rerank_candidate_cap = max(2, int(rerank_candidate_cap))
        self.rerank_document_max_chars = max(256, int(rerank_document_max_chars))
        self.max_reflection_rounds = max_reflection_rounds
        self.temperature = temperature

    def _truncate_rerank_document(self, text: str) -> str:
        text = str(text or "").strip()
        if len(text) <= self.rerank_document_max_chars:
            return text
        head = text[: self.rerank_document_max_chars].rstrip()
        cut = head.rfind(" ")
        if cut >= max(128, self.rerank_document_max_chars // 2):
            head = head[:cut].rstrip()
        return head

    async def retrieve(
        self,
        query: str,
        enable_reflection: Optional[bool] = None,
        enable_planning: Optional[bool] = None,
        top_k: Optional[int] = None,
    ) -> List[MemoryEntry]:
        """
        Retrieve relevant memory entries for a query

        Args:
            query: User's question
            enable_reflection: Override reflection setting

        Returns:
            List of relevant MemoryEntry objects
        """
        use_reflection = (
            enable_reflection
            if enable_reflection is not None
            else self.enable_reflection
        )

        use_planning = (
            enable_planning
            if enable_planning is not None
            else self.enable_planning
        )

        semantic_top_k = self.semantic_top_k
        if top_k is not None:
            try:
                semantic_top_k = max(1, int(top_k))
            except Exception:
                semantic_top_k = self.semantic_top_k

        if use_planning:
            return await self._retrieve_with_planning(query, use_reflection, semantic_top_k)

        results = await self._simple_retrieve(query, semantic_top_k)
        return await self._maybe_rerank(query, results)

    async def _simple_retrieve(self, query: str, semantic_top_k: int) -> List[MemoryEntry]:
        """Simple semantic search without planning"""
        query_embedding = await self.client.create_single_embedding(
            query,
            task="query",
            instruction=self._build_query_instruction(query),
        )
        return await self.vector_store.semantic_search(
            self.table_name,
            query_embedding,
            top_k=semantic_top_k,
        )

    async def _retrieve_with_planning(
        self,
        query: str,
        enable_reflection: bool,
        semantic_top_k: int,
    ) -> List[MemoryEntry]:
        """Retrieve with intelligent planning and optional reflection"""

        # Step 1: Analyze information requirements
        plan = await self._analyze_information_requirements(query)

        # Step 2: Generate targeted queries
        search_queries = await self._generate_targeted_queries(query, plan)

        # Step 3: Execute searches sequentially
        all_results = await self._execute_searches(search_queries, semantic_top_k)

        # Step 4: Merge and deduplicate
        merged_results = self._merge_and_deduplicate(all_results)

        # Step 5: Optional reflection
        if enable_reflection and plan.complexity_score > 0.5:
            merged_results = await self._retrieve_with_reflection(
                query,
                merged_results,
                plan,
                semantic_top_k,
            )

        return await self._maybe_rerank(
            query,
            merged_results,
            instruction=self._build_query_instruction(query, plan.question_type),
        )

    async def _maybe_rerank(
        self,
        query: str,
        results: List[MemoryEntry],
        instruction: Optional[str] = None,
    ) -> List[MemoryEntry]:
        """Apply local reranking when the active client supports it."""
        if not self.enable_reranking or len(results) < 2:
            return results

        rerank_fn = getattr(self.client, "rerank", None)
        if not callable(rerank_fn):
            return results

        documents: list[str] = []
        result_indexes: list[int] = []
        for idx, entry in enumerate(results):
            text = self._truncate_rerank_document(entry.lossless_restatement or "")
            if not text:
                continue
            documents.append(text)
            result_indexes.append(idx)

        if len(documents) < 2:
            return results

        rerank_cap = min(len(documents), self.rerank_candidate_cap)
        try:
            ranked = await rerank_fn(
                query=query,
                documents=documents,
                top_n=rerank_cap,
                instruction=instruction or self._build_query_instruction(query),
            )
        except TypeError:
            try:
                ranked = await rerank_fn(query=query, documents=documents, top_n=rerank_cap)
            except Exception as exc:
                print(f"Rerank error: {exc}")
                return results
        except Exception as exc:
            print(f"Rerank error: {exc}")
            return results

        if not ranked:
            return results

        ordered: list[MemoryEntry] = []
        seen_result_indexes: set[int] = set()
        for row in ranked:
            if not isinstance(row, dict):
                continue
            row_index = row.get("index")
            try:
                doc_index = int(row_index)
            except Exception:
                continue
            if doc_index < 0 or doc_index >= len(result_indexes):
                continue
            original_index = result_indexes[doc_index]
            if original_index in seen_result_indexes:
                continue
            seen_result_indexes.add(original_index)
            ordered.append(results[original_index])

        if not ordered:
            return results

        for idx, entry in enumerate(results):
            if idx not in seen_result_indexes:
                ordered.append(entry)

        return ordered

    async def _analyze_information_requirements(
        self,
        query: str,
    ) -> RetrievalPlan:
        """Analyze query to determine information requirements"""

        prompt = f"""Analyze the following question and determine retrieval requirements.

Question: {query}

Analyze:
1. What type of question is this? (factual, temporal, relational, comparative, etc.)
2. What key entities/events need to be identified?
3. What information types are required? (with priority: high/medium/low)
4. What relationships need to be established?
5. How many minimal search queries are needed? (1-4)
6. Complexity score (0.0-1.0): simple facts=0.2, multi-hop=0.6, complex reasoning=0.8+

Return JSON:
{{
  "question_type": "type",
  "key_entities": ["entity1", "entity2"],
  "required_info": [
    {{"type": "info_type", "priority": "high/medium/low"}}
  ],
  "relationships": ["relationship1"],
  "minimal_queries_needed": 1-4,
  "complexity_score": 0.0-1.0
}}

Return ONLY valid JSON."""

        messages = [
            {"role": "system", "content": "You are a query analysis expert."},
            {"role": "user", "content": prompt},
        ]

        try:
            response = await self.client.chat_completion(
                messages=messages,
                temperature=self.temperature,
            )

            data = self.client.extract_json(response)
            if data:
                return RetrievalPlan(
                    question_type=data.get("question_type", "factual"),
                    key_entities=data.get("key_entities", []),
                    required_info=data.get("required_info", []),
                    relationships=data.get("relationships", []),
                    minimal_queries_needed=min(data.get("minimal_queries_needed", 1), 4),
                    complexity_score=min(max(data.get("complexity_score", 0.5), 0.0), 1.0),
                )
        except Exception as e:
            print(f"Query analysis error: {e}")

        # Default plan
        return RetrievalPlan(
            question_type="factual",
            key_entities=[],
            required_info=[],
            relationships=[],
            minimal_queries_needed=1,
            complexity_score=0.5,
        )

    async def _generate_targeted_queries(
        self,
        original_query: str,
        plan: RetrievalPlan,
    ) -> List[str]:
        """Generate targeted search queries based on analysis"""

        if plan.minimal_queries_needed <= 1:
            return [original_query]

        prompt = f"""Based on the analysis, generate {plan.minimal_queries_needed} targeted search queries.

Original Question: {original_query}

Analysis:
- Question Type: {plan.question_type}
- Key Entities: {plan.key_entities}
- Required Information: {plan.required_info}
- Relationships: {plan.relationships}

Requirements:
1. Generate {plan.minimal_queries_needed} distinct queries
2. Each query should target specific information
3. Together they should cover all required information
4. Keep queries concise and focused

Return JSON:
{{
  "queries": ["query1", "query2", ...]
}}

Return ONLY valid JSON."""

        messages = [
            {"role": "system", "content": "You are a search query generator."},
            {"role": "user", "content": prompt},
        ]

        try:
            response = await self.client.chat_completion(
                messages=messages,
                temperature=self.temperature,
            )

            data = self.client.extract_json(response)
            if data and "queries" in data:
                queries = data["queries"][:4]  # Max 4 queries
                if queries:
                    return queries
        except Exception as e:
            print(f"Query generation error: {e}")

        return [original_query]

    async def _execute_searches(
        self,
        queries: List[str],
        semantic_top_k: int,
    ) -> List[List[MemoryEntry]]:
        """Execute searches sequentially"""
        all_results = []

        for query in queries:
            # Semantic search
            query_embedding = await self.client.create_single_embedding(
                query,
                task="query",
                instruction=self._build_query_instruction(query),
            )
            semantic_results = await self.vector_store.semantic_search(
                self.table_name,
                query_embedding,
                top_k=semantic_top_k,
            )
            all_results.append(semantic_results)

            # Keyword search
            keywords = self._extract_keywords(query)
            if keywords:
                keyword_results = await self.vector_store.keyword_search(
                    self.table_name,
                    keywords,
                    top_k=self.keyword_top_k,
                )
                all_results.append(keyword_results)

        return all_results

    def _extract_keywords(self, query: str) -> List[str]:
        """Extract keywords from query for lexical search"""
        # Simple keyword extraction
        stop_words = {
            "a", "an", "the", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "must", "shall",
            "can", "need", "dare", "ought", "used", "to", "of", "in",
            "for", "on", "with", "at", "by", "from", "as", "into",
            "through", "during", "before", "after", "above", "below",
            "between", "under", "again", "further", "then", "once",
            "here", "there", "when", "where", "why", "how", "all",
            "each", "few", "more", "most", "other", "some", "such",
            "no", "nor", "not", "only", "own", "same", "so", "than",
            "too", "very", "just", "and", "but", "if", "or", "because",
            "until", "while", "what", "which", "who", "whom", "this",
            "that", "these", "those", "am", "i", "me", "my", "myself",
            "we", "our", "ours", "ourselves", "you", "your", "yours",
            "yourself", "yourselves", "he", "him", "his", "himself",
            "she", "her", "hers", "herself", "it", "its", "itself",
            "they", "them", "their", "theirs", "themselves",
        }

        words = query.lower().split()
        keywords = [
            word.strip(".,!?;:'\"()[]{}") for word in words
            if word.lower() not in stop_words and len(word) > 2
        ]

        return keywords[:10]  # Max 10 keywords

    def _merge_and_deduplicate(
        self,
        results_lists: List[List[MemoryEntry]],
    ) -> List[MemoryEntry]:
        """Merge and deduplicate results from multiple searches"""
        seen_ids = set()
        merged = []

        for results in results_lists:
            for entry in results:
                if entry.entry_id not in seen_ids:
                    seen_ids.add(entry.entry_id)
                    merged.append(entry)

        return merged

    async def _retrieve_with_reflection(
        self,
        query: str,
        initial_results: List[MemoryEntry],
        plan: RetrievalPlan,
        semantic_top_k: int,
    ) -> List[MemoryEntry]:
        """Iterative refinement through reflection"""

        current_results = initial_results

        for round_num in range(self.max_reflection_rounds):
            # Check completeness
            is_complete, missing_info = await self._check_completeness(
                query,
                current_results,
                plan,
            )

            if is_complete:
                break

            # Generate additional queries for missing info
            additional_queries = await self._generate_missing_info_queries(
                query,
                missing_info,
            )

            if not additional_queries:
                break

            # Execute additional searches
            additional_results = await self._execute_searches(additional_queries, semantic_top_k)

            # Merge with existing results
            all_results = [current_results] + additional_results
            current_results = self._merge_and_deduplicate(all_results)

        return current_results

    async def _check_completeness(
        self,
        query: str,
        results: List[MemoryEntry],
        plan: RetrievalPlan,
    ) -> tuple[bool, List[str]]:
        """Check if retrieved results are sufficient"""

        if not results:
            return False, ["No results found"]

        # Format results for analysis
        results_text = "\n".join([
            f"- {entry.lossless_restatement}"
            for entry in results[:20]  # Limit for prompt size
        ])

        prompt = f"""Analyze if the retrieved information is sufficient to answer the question.

Question: {query}

Required Information:
{[info.get("type", "") for info in plan.required_info]}

Retrieved Information:
{results_text}

Determine:
1. Is the information sufficient to answer the question? (yes/no)
2. If no, what specific information is missing?

Return JSON:
{{
  "is_complete": true/false,
  "missing_info": ["missing1", "missing2"] or []
}}

Return ONLY valid JSON."""

        messages = [
            {"role": "system", "content": "You are an information completeness analyst."},
            {"role": "user", "content": prompt},
        ]

        try:
            response = await self.client.chat_completion(
                messages=messages,
                temperature=self.temperature,
            )

            data = self.client.extract_json(response)
            if data:
                return (
                    data.get("is_complete", True),
                    data.get("missing_info", []),
                )
        except Exception as e:
            print(f"Completeness check error: {e}")

        return True, []

    async def _generate_missing_info_queries(
        self,
        original_query: str,
        missing_info: List[str],
    ) -> List[str]:
        """Generate queries to find missing information"""

        if not missing_info:
            return []

        prompt = f"""Generate search queries to find the missing information.

Original Question: {original_query}

Missing Information:
{missing_info}

Generate 1-2 targeted search queries to find this missing information.

Return JSON:
{{
  "queries": ["query1", "query2"]
}}

Return ONLY valid JSON."""

        messages = [
            {"role": "system", "content": "You are a search query generator."},
            {"role": "user", "content": prompt},
        ]

        try:
            response = await self.client.chat_completion(
                messages=messages,
                temperature=self.temperature,
            )

            data = self.client.extract_json(response)
            if data and "queries" in data:
                return data["queries"][:2]
        except Exception as e:
            print(f"Missing info query generation error: {e}")

        return []

    async def hybrid_retrieve(
        self,
        query: str,
        persons: Optional[List[str]] = None,
        location: Optional[str] = None,
        entities: Optional[List[str]] = None,
        timestamp_start: Optional[str] = None,
        timestamp_end: Optional[str] = None,
    ) -> List[MemoryEntry]:
        """
        Hybrid retrieval combining semantic, lexical, and structured search

        Args:
            query: Search query
            persons: Filter by person names
            location: Filter by location
            entities: Filter by entities
            timestamp_start: Start of timestamp range
            timestamp_end: End of timestamp range

        Returns:
            List of relevant MemoryEntry objects
        """
        all_results = []

        # Semantic search
        query_embedding = await self.client.create_single_embedding(
            query,
            task="query",
            instruction=self._build_query_instruction(query),
        )
        semantic_results = await self.vector_store.semantic_search(
            self.table_name,
            query_embedding,
            top_k=self.semantic_top_k,
        )
        all_results.append(semantic_results)

        # Keyword search
        keywords = self._extract_keywords(query)
        if keywords:
            keyword_results = await self.vector_store.keyword_search(
                self.table_name,
                keywords,
                top_k=self.keyword_top_k,
            )
            all_results.append(keyword_results)

        # Structured search (if filters provided)
        if any([persons, location, entities, timestamp_start, timestamp_end]):
            structured_results = await self.vector_store.structured_search(
                self.table_name,
                persons=persons,
                location=location,
                entities=entities,
                timestamp_start=timestamp_start,
                timestamp_end=timestamp_end,
                top_k=self.keyword_top_k,
            )
            # Prepend structured results (higher priority)
            all_results.insert(0, structured_results)

        merged = self._merge_and_deduplicate(all_results)
        return await self._maybe_rerank(query, merged)

    def _build_query_instruction(self, query: str, question_type: Optional[str] = None) -> str:
        """Build a Qwen3-style query-side instruction in English."""
        q = (query or "").strip().lower()
        qt = (question_type or "").strip().lower()

        if qt in {"temporal", "time", "temporal_reasoning"}:
            return (
                "Given a memory retrieval query, retrieve documents that contain the relevant event details "
                "and absolute time information needed to answer the question."
            )
        if qt in {"comparative", "comparison"}:
            return (
                "Given a memory retrieval query, retrieve documents that support direct comparison and the "
                "key distinguishing details."
            )
        if qt in {"relational", "relationship"}:
            return (
                "Given a memory retrieval query, retrieve documents that directly establish the relevant "
                "relationship, association, or identity."
            )
        if qt in {"factual", "fact", "simple"} and q:
            return (
                "Given a memory retrieval query, retrieve documents that directly help answer the question. "
                "Prefer exact facts, strong evidence, and high-signal context over tangential similarity."
            )

        if not q:
            return "Given a memory retrieval query, retrieve relevant passages that directly help answer the query."
        if any(token in q for token in ("when", "date", "time", "timeline", "deadline", "before", "after")):
            return (
                "Given a memory retrieval query, retrieve documents that contain the relevant event details "
                "and absolute time information needed to answer the question."
            )
        if any(token in q for token in ("compare", "comparison", "versus", "vs", "difference", "better", "worse")):
            return (
                "Given a memory retrieval query, retrieve documents that support direct comparison and the "
                "key distinguishing details."
            )
        if any(token in q for token in ("who", "whose", "with", "relationship", "connected", "related")):
            return (
                "Given a memory retrieval query, retrieve documents that directly establish the relevant "
                "relationship, association, or identity."
            )
        return (
            "Given a memory retrieval query, retrieve documents that directly help answer the question. "
            "Prefer exact facts, strong evidence, and high-signal context over tangential similarity."
        )
