"""
Answer Generator - Final Synthesis Module

Synthesizes answers from retrieved memory contexts using LLM.
"""

from typing import List, Optional

from ..auth.models import MemoryEntry

# Type alias for LLM client (duck-typed OpenRouter-compatible client).
LLMClient = object  # Duck-typed: OpenRouter-compatible client


class AnswerGenerator:
    """
    Generates answers from retrieved memory contexts.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        temperature: float = 0.1,
        max_context_entries: int = 10,
        max_context_entry_chars: int = 320,
        max_context_total_chars: int = 3500,
        max_answer_tokens: int = 64,
    ):
        self.client = llm_client
        self.temperature = temperature
        self.max_context_entries = max(1, int(max_context_entries))
        self.max_context_entry_chars = max(80, int(max_context_entry_chars))
        self.max_context_total_chars = max(self.max_context_entry_chars, int(max_context_total_chars))
        self.max_answer_tokens = max(24, int(max_answer_tokens))

    async def generate_answer(
        self,
        query: str,
        contexts: List[MemoryEntry],
    ) -> dict:
        """
        Generate an answer from retrieved contexts

        Args:
            query: User's question
            contexts: Retrieved MemoryEntry objects

        Returns:
            Dict with answer and reasoning
        """
        if not contexts:
            return {
                "answer": "I don't have any relevant memories to answer this question.",
                "reasoning": "No relevant context was found in the memory store.",
                "confidence": "low",
            }

        # Format contexts
        context_str = self._format_contexts(contexts)

        # Build prompt
        prompt = self._build_answer_prompt(query, context_str)

        messages = [
            {
                "role": "system",
                "content": (
                    "Answer only from the provided context. "
                    "Return minified JSON with keys answer and confidence. "
                    "If the context is insufficient, answer exactly "
                    '{"answer":"insufficient context","confidence":"low"}.'
                ),
            },
            {"role": "user", "content": prompt},
        ]

        # Retry mechanism
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = await self.client.chat_completion(
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_answer_tokens,
                    response_format={"type": "json_object"},
                )

                data = self.client.extract_json(response)
                if data:
                    return {
                        "answer": data.get("answer", "Unable to generate answer."),
                        "reasoning": data.get("reasoning", ""),
                        "confidence": data.get("confidence", "medium"),
                    }

            except Exception as e:
                print(f"Answer generation attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    return {
                        "answer": "An error occurred while generating the answer.",
                        "reasoning": f"Error: {str(e)}",
                        "confidence": "low",
                    }

        return {
            "answer": "Unable to generate answer after multiple attempts.",
            "reasoning": "JSON parsing failed.",
            "confidence": "low",
        }

    def _truncate_text(self, text: str, limit: int) -> str:
        if len(text) <= limit:
            return text
        return text[: max(0, limit - 16)] + " ...[truncated]"

    def _format_contexts(self, contexts: List[MemoryEntry]) -> str:
        """Format memory entries into readable context"""
        formatted = []
        total_chars = 0

        for i, entry in enumerate(contexts[: self.max_context_entries], 1):
            statement = self._truncate_text(entry.lossless_restatement or "", self.max_context_entry_chars)
            parts = [f"[{i}] {statement}"]

            metadata = []
            if entry.timestamp:
                metadata.append(f"t={entry.timestamp}")
            if entry.location:
                metadata.append(f"loc={entry.location}")
            if entry.persons:
                metadata.append(f"p={','.join(entry.persons)}")
            if entry.entities:
                metadata.append(f"e={','.join(entry.entities)}")
            if entry.topic:
                metadata.append(f"topic={entry.topic}")

            if metadata:
                parts.append(f" ({'; '.join(metadata)})")

            block = "".join(parts)
            if total_chars + len(block) > self.max_context_total_chars:
                remaining = self.max_context_total_chars - total_chars
                if remaining > 120:
                    formatted.append(self._truncate_text(block, remaining))
                formatted.append("...[context budget reached]")
                break
            formatted.append(block)
            total_chars += len(block)

        return "\n\n".join(formatted)

    def _build_answer_prompt(self, query: str, context_str: str) -> str:
        """Build the answer generation prompt"""
        return (
            f"Q:{query}\n"
            f"CTX:\n{context_str}\n"
            'Return only minified JSON {"answer":"...","confidence":"high|medium|low"}.'
        )

    async def generate_summary(
        self,
        entries: List[MemoryEntry],
        topic: Optional[str] = None,
    ) -> str:
        """
        Generate a summary of memory entries

        Args:
            entries: MemoryEntry objects to summarize
            topic: Optional topic focus

        Returns:
            Summary text
        """
        if not entries:
            return "No memories to summarize."

        # Format entries
        entries_text = "\n".join([
            f"- {entry.lossless_restatement}"
            for entry in entries[:50]
        ])

        topic_str = f" about {topic}" if topic else ""

        prompt = f"""Summarize the following memories{topic_str}:

{entries_text}

Provide a concise summary (2-4 sentences) that captures the key information.

Return ONLY the summary text, no JSON or formatting."""

        messages = [
            {"role": "system", "content": "You are a helpful summarization assistant."},
            {"role": "user", "content": prompt},
        ]

        try:
            response = await self.client.chat_completion(
                messages=messages,
                temperature=self.temperature,
            )
            return response.strip()
        except Exception as e:
            return f"Error generating summary: {e}"
