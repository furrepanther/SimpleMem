"""
Ollama SDK integration for LLM and Embedding services
"""

import json
import asyncio
import random
import re
import time
from typing import List, Dict, Any, Optional


class OllamaClient:
    """
    Ollama API client for LLM and Embedding operations.
    Uses Ollama's OpenAI-compatible API endpoints.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434/v1",
        embedding_base_url: Optional[str] = None,
        llm_model: str = "qwen2.5:7b",
        embedding_model: str = "nomic-embed-text",
    ):
        self.base_url = base_url.rstrip("/")
        self.embedding_base_url = (embedding_base_url or base_url).rstrip("/")
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        self._retry_delays_s = [0.25, 0.5, 1.0, 2.0, 4.0]
        self._breaker_open_until: dict[str, float] = {"chat": 0.0, "embed": 0.0}
        self._breaker_429_streak: dict[str, int] = {"chat": 0, "embed": 0}
        self._breaker_threshold = 3
        self._breaker_cooldown_s = 8.0

        # Lazy import to avoid issues if not installed
        self._chat_client = None
        self._embed_client = None

    def _ensure_lane_open(self, lane: str) -> None:
        now = time.monotonic()
        open_until = float(self._breaker_open_until.get(lane, 0.0))
        if now < open_until:
            raise RuntimeError(f"circuit_open:{lane}:retry_after_s={round(open_until - now, 2)}")

    def _note_success(self, lane: str) -> None:
        self._breaker_429_streak[lane] = 0
        self._breaker_open_until[lane] = 0.0

    def _note_429(self, lane: str) -> None:
        streak = int(self._breaker_429_streak.get(lane, 0)) + 1
        self._breaker_429_streak[lane] = streak
        if streak >= self._breaker_threshold:
            self._breaker_open_until[lane] = time.monotonic() + self._breaker_cooldown_s

    async def _post_with_retry(self, *, lane: str, client: Any, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        self._ensure_lane_open(lane)
        last_exc: Exception | None = None
        for idx, delay in enumerate(self._retry_delays_s, start=1):
            try:
                resp = await client.post(path, json=payload)
                if resp.status_code == 429:
                    self._note_429(lane)
                    jitter = random.uniform(0.0, 0.25)
                    await asyncio.sleep(delay + jitter)
                    continue
                resp.raise_for_status()
                self._note_success(lane)
                return resp.json()
            except Exception as exc:
                last_exc = exc
                msg = str(exc).lower()
                retryable = "429" in msg or "503" in msg or "502" in msg or "timeout" in msg or "connection" in msg
                if not retryable or idx >= len(self._retry_delays_s):
                    break
                jitter = random.uniform(0.0, 0.25)
                await asyncio.sleep(delay + jitter)
                continue
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("request_failed_without_exception")

    def _get_chat_client(self):
        """Get or create chat client."""
        if self._chat_client is None:
            import httpx
            self._chat_client = httpx.AsyncClient(
                base_url=self.base_url,
                headers={
                    "Content-Type": "application/json",
                },
                timeout=120.0,
            )
        return self._chat_client

    def _get_embed_client(self):
        """Get or create embedding client."""
        if self._embed_client is None:
            import httpx
            self._embed_client = httpx.AsyncClient(
                base_url=self.embedding_base_url,
                headers={
                    "Content-Type": "application/json",
                },
                timeout=120.0,
            )
        return self._embed_client

    async def close(self):
        """Close underlying clients."""
        if self._chat_client:
            await self._chat_client.aclose()
            self._chat_client = None
        if self._embed_client:
            await self._embed_client.aclose()
            self._embed_client = None

    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict] = None,
        stream: bool = False,
    ) -> str:
        """
        Call LLM for chat completion using Ollama's OpenAI-compatible API

        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens in response
            response_format: Optional response format (e.g., {"type": "json_object"})
            stream: Whether to stream the response

        Returns:
            Generated text content
        """
        client = self._get_chat_client()

        payload = {
            "model": self.llm_model,
            "messages": messages,
            "temperature": temperature,
        }

        if max_tokens:
            payload["max_tokens"] = max_tokens

        # Note: Ollama may not support response_format in the same way
        # We include it for compatibility, but it may be ignored
        if response_format:
            payload["response_format"] = response_format

        if stream:
            return await self._stream_completion(payload)

        data = await self._post_with_retry(
            lane="chat",
            client=client,
            path="/chat/completions",
            payload=payload,
        )
        return data["choices"][0]["message"]["content"]

    async def _stream_completion(self, payload: Dict) -> str:
        """Stream chat completion and return full content"""
        client = self._get_chat_client()
        payload["stream"] = True

        content_parts = []

        async with client.stream("POST", "/chat/completions", json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str.strip() == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                        delta = data["choices"][0].get("delta", {})
                        if "content" in delta:
                            content_parts.append(delta["content"])
                    except json.JSONDecodeError:
                        continue

        return "".join(content_parts)

    async def create_embedding(
        self,
        texts: List[str],
    ) -> List[List[float]]:
        """
        Create embeddings for texts using Ollama's embedding API

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        client = self._get_embed_client()

        payload = {"model": self.embedding_model, "input": texts}
        data = await self._post_with_retry(
            lane="embed",
            client=client,
            path="/embeddings",
            payload=payload,
        )
        rows = data.get("data", [])
        if not isinstance(rows, list):
            raise RuntimeError("invalid_embedding_response")
        out: list[list[float]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            emb = row.get("embedding")
            if isinstance(emb, list):
                out.append(emb)
        if len(out) == len(texts):
            return out
        # Fallback to per-item if server returned unexpected shape.
        fallback: list[list[float]] = []
        for text in texts:
            row_data = await self._post_with_retry(
                lane="embed",
                client=client,
                path="/embeddings",
                payload={"model": self.embedding_model, "input": text},
            )
            fallback.append(row_data["data"][0]["embedding"])
        return fallback

    async def create_single_embedding(self, text: str) -> List[float]:
        """Create embedding for a single text"""
        embeddings = await self.create_embedding([text])
        return embeddings[0]

    async def verify_api_key(self) -> tuple[bool, Optional[str]]:
        """
        Verify that Ollama is accessible

        Ollama doesn't use API keys, so we just check connectivity

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            client = self._get_chat_client()
            # Try to list models to verify connectivity
            response = await client.get("/models")
            if response.status_code == 200:
                return True, None
            else:
                return False, f"Ollama API error: {response.status_code}"
        except Exception as e:
            return False, f"Connection error: {str(e)}"

    def extract_json(self, text: str) -> Any:
        """
        Extract JSON from LLM response text with robust parsing

        Args:
            text: Raw text that may contain JSON

        Returns:
            Parsed JSON object
        """
        if not text:
            return None

        # Strategy 1: Direct JSON parsing
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            pass

        # Strategy 2: Extract from ```json ``` blocks
        json_block_pattern = r"```json\s*([\s\S]*?)\s*```"
        matches = re.findall(json_block_pattern, text, re.IGNORECASE)
        if matches:
            try:
                return json.loads(matches[0].strip())
            except json.JSONDecodeError:
                pass

        # Strategy 3: Extract from generic ``` ``` blocks
        generic_block_pattern = r"```\s*([\s\S]*?)\s*```"
        matches = re.findall(generic_block_pattern, text)
        if matches:
            for match in matches:
                try:
                    return json.loads(match.strip())
                except json.JSONDecodeError:
                    continue

        # Strategy 4: Find balanced JSON object/array
        # Find first { or [
        start_obj = text.find("{")
        start_arr = text.find("[")

        if start_obj == -1 and start_arr == -1:
            return None

        if start_arr == -1 or (start_obj != -1 and start_obj < start_arr):
            # Try to find matching }
            json_str = self._extract_balanced_braces(text[start_obj:], "{", "}")
        else:
            # Try to find matching ]
            json_str = self._extract_balanced_braces(text[start_arr:], "[", "]")

        if json_str:
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass

        # Strategy 5: Clean and retry
        cleaned = self._clean_json_string(text)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        return None

    def _extract_balanced_braces(self, text: str, open_char: str, close_char: str) -> Optional[str]:
        """Extract a balanced brace-enclosed string"""
        if not text or text[0] != open_char:
            return None

        count = 0
        in_string = False
        escape_next = False

        for i, char in enumerate(text):
            if escape_next:
                escape_next = False
                continue

            if char == "\\":
                escape_next = True
                continue

            if char == '"' and not escape_next:
                in_string = not in_string
                continue

            if in_string:
                continue

            if char == open_char:
                count += 1
            elif char == close_char:
                count -= 1
                if count == 0:
                    return text[: i + 1]

        return None

    def _clean_json_string(self, text: str) -> str:
        """Clean common JSON issues from LLM output"""
        # Remove common prefixes
        prefixes = [
            "Here's the JSON:",
            "Here is the JSON:",
            "JSON output:",
            "Output:",
            "Result:",
        ]
        cleaned = text.strip()
        for prefix in prefixes:
            if cleaned.lower().startswith(prefix.lower()):
                cleaned = cleaned[len(prefix):].strip()

        # Remove trailing commas before } or ]
        cleaned = re.sub(r",\s*([\}\]])", r"\1", cleaned)

        # Remove single-line comments
        cleaned = re.sub(r"//.*$", "", cleaned, flags=re.MULTILINE)

        return cleaned


class OllamaClientManager:
    """
    Manages Ollama client instances.
    Since Ollama doesn't use API keys, this is simpler than OpenRouter.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434/v1",
        embedding_base_url: Optional[str] = None,
        llm_model: str = "qwen2.5:7b",
        embedding_model: str = "nomic-embed-text",
    ):
        self.base_url = base_url
        self.embedding_base_url = embedding_base_url or base_url
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        self._client: Optional[OllamaClient] = None

    def get_client(self, api_key: str = None) -> OllamaClient:
        """
        Get or create an Ollama client

        Args:
            api_key: Ignored for Ollama (kept for interface compatibility)

        Returns:
            OllamaClient instance
        """
        if self._client is None:
            self._client = OllamaClient(
                base_url=self.base_url,
                embedding_base_url=self.embedding_base_url,
                llm_model=self.llm_model,
                embedding_model=self.embedding_model,
            )

        return self._client

    async def close_all(self):
        """Close all client connections"""
        if self._client:
            await self._client.close()
            self._client = None

    async def remove_client(self, api_key: str = None):
        """Remove and close the client (kept for interface compatibility)"""
        await self.close_all()
