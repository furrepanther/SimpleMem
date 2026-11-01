"""
OpenRouter SDK integration for LLM and Embedding services
"""

import json
import logging
import math
import re
from urllib.parse import urlparse
from typing import List, Dict, Any, Optional, AsyncGenerator


logger = logging.getLogger(__name__)


def _local_endpoints(
    base_url: str, embedding_base_url: Optional[str], rerank_base_url: Optional[str]
) -> tuple[str, str, str]:
    embedding_url = base_url if embedding_base_url is None else embedding_base_url
    rerank_url = embedding_url if rerank_base_url is None else rerank_base_url
    for name, endpoint in (
        ("base_url", base_url),
        ("embedding_base_url", embedding_url),
        ("rerank_base_url", rerank_url),
    ):
        # Importing config.settings also loads the environment file, so mirror
        # its loopback rule here without introducing that import-time side effect.
        parsed = urlparse(endpoint.strip()) if isinstance(endpoint, str) else None
        if (
            parsed is None
            or parsed.scheme not in {"http", "https"}
            or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
        ):
            raise ValueError(f"{name} must use a loopback HTTP(S) URL")
    return base_url, embedding_url, rerank_url


class OpenRouterClient:
    """
    OpenRouter API client for LLM and Embedding operations.
    Each instance is bound to a specific user's API key.
    """

    # BGE-M3 rejects the measured 3,002-token record because its physical
    # batch is 2,048 tokens.  Cap chunks at 384 Unicode code points: UTF-8
    # consumes at most four bytes per code point, so this remains below the
    # limit even for byte-tokenized, non-ASCII text.  Keep each request to one
    # chunk so the server never aggregates otherwise-safe chunks into an
    # oversized physical batch.
    _MAX_EMBEDDING_CHARS_PER_INPUT = 384

    def __init__(
        self,
        api_key: str,
        base_url: str,
        embedding_base_url: Optional[str] = None,
        rerank_base_url: Optional[str] = None,
        llm_model: str = "openai/gpt-4.1-mini",
        embedding_model: str = "BAAI/bge-m3",
        reranker_model: str = "BAAI/bge-reranker-v2-m3",
    ):
        base_url, embedding_url, rerank_url = _local_endpoints(
            base_url, embedding_base_url, rerank_base_url
        )
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.embedding_base_url = embedding_url.rstrip("/")
        self.rerank_base_url = rerank_url.rstrip("/")
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        self.reranker_model = reranker_model

        # Lazy import to avoid issues if not installed
        self._chat_client = None
        self._embed_client = None
        self._rerank_client = None

    @staticmethod
    def _is_local_base_url(base_url: str) -> bool:
        host = (urlparse(base_url or "").hostname or "").lower()
        return host in {"127.0.0.1", "localhost", "::1"}

    def _build_headers(self, base_url: Optional[str] = None) -> Dict[str, str]:
        headers: Dict[str, str] = {
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        # OpenRouter-specific headers are not required for local gateways.
        if not self._is_local_base_url(base_url or self.base_url):
            headers["HTTP-Referer"] = "https://simplemem.app"
            headers["X-Title"] = "SimpleMem MCP Server"
        return headers

    def _get_chat_client(self):
        """Get or create the chat HTTP client."""
        if self._chat_client is None:
            import httpx

            self._chat_client = httpx.AsyncClient(
                base_url=self.base_url,
                headers=self._build_headers(self.base_url),
                timeout=120.0,
            )
        return self._chat_client

    def _get_embed_client(self):
        """Get or create the embedding HTTP client."""
        if self._embed_client is None:
            import httpx

            self._embed_client = httpx.AsyncClient(
                base_url=self.embedding_base_url,
                headers=self._build_headers(self.embedding_base_url),
                timeout=120.0,
            )
        return self._embed_client

    def _get_rerank_client(self):
        """Get or create the rerank HTTP client."""
        if self._rerank_client is None:
            import httpx

            self._rerank_client = httpx.AsyncClient(
                base_url=self.rerank_base_url,
                headers=self._build_headers(self.rerank_base_url),
                timeout=120.0,
            )
        return self._rerank_client

    async def close(self):
        """Close the HTTP client"""
        if self._chat_client:
            await self._chat_client.aclose()
            self._chat_client = None
        if self._embed_client:
            await self._embed_client.aclose()
            self._embed_client = None
        if self._rerank_client:
            await self._rerank_client.aclose()
            self._rerank_client = None

    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict] = None,
        stream: bool = False,
    ) -> str:
        """
        Call LLM for chat completion

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

        if response_format:
            payload["response_format"] = response_format

        if stream:
            return await self._stream_completion(payload)

        response = await client.post("/chat/completions", json=payload)
        response.raise_for_status()
        data = response.json()

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
        *,
        task: Optional[str] = None,
        instruction: Optional[str] = None,
    ) -> List[List[float]]:
        """
        Create embeddings for texts

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        client = self._get_embed_client()

        payload = {
            "model": self.embedding_model,
            "input": texts,
        }
        if task and self._is_local_base_url(self.embedding_base_url):
            payload["task"] = task
        if instruction and self._is_local_base_url(self.embedding_base_url):
            payload["instruction"] = instruction

        response = await client.post("/embeddings", json=payload)
        response.raise_for_status()
        data = response.json()

        # Sort by index to ensure correct order
        embeddings_data = sorted(data["data"], key=lambda x: x["index"])
        return [item["embedding"] for item in embeddings_data]

    async def create_single_embedding(
        self,
        text: str,
        *,
        task: Optional[str] = None,
        instruction: Optional[str] = None,
    ) -> List[float]:
        """Create embedding for a single text"""
        embeddings = await self.create_embedding(
            [text], task=task, instruction=instruction
        )
        return embeddings[0]

    @classmethod
    def _split_embedding_text(cls, text: str) -> List[str]:
        """Split an oversized semantic projection without changing source storage."""
        raw = str(text or "").strip()
        if not raw or len(raw) <= cls._MAX_EMBEDDING_CHARS_PER_INPUT:
            return [raw]

        chunks: List[str] = []
        for paragraph in (part.strip() for part in re.split(r"\n\s*\n", raw)):
            if not paragraph:
                continue
            remainder = paragraph
            while len(remainder) > cls._MAX_EMBEDDING_CHARS_PER_INPUT:
                cut = remainder.rfind(" ", 0, cls._MAX_EMBEDDING_CHARS_PER_INPUT)
                if cut < cls._MAX_EMBEDDING_CHARS_PER_INPUT // 2:
                    cut = cls._MAX_EMBEDDING_CHARS_PER_INPUT
                chunk = remainder[:cut].strip()
                if chunk:
                    chunks.append(chunk)
                remainder = remainder[cut:].lstrip()
            if remainder:
                chunks.append(remainder)
        return chunks or [raw]

    async def create_bounded_semantic_embedding(
        self,
        text: str,
        *,
        task: Optional[str] = None,
        instruction: Optional[str] = None,
    ) -> List[float]:
        """Embed a complete record through bounded chunks and normalized pooling.

        The caller retains the unmodified source record.  Only its vector
        projection is chunked, which prevents a long durable memory write from
        becoming one oversized BGE request.
        """
        chunks = self._split_embedding_text(text)
        if len(chunks) == 1:
            return await self.create_single_embedding(
                chunks[0], task=task, instruction=instruction
            )

        vectors: List[List[float]] = []
        for chunk in chunks:
            # Deliberately do not batch multiple chunks: BGE's physical batch
            # limit applies to the aggregate request, not only each element.
            vectors.append(
                await self.create_single_embedding(
                    chunk, task=task, instruction=instruction
                )
            )

        dimensions = len(vectors[0]) if vectors else 0
        if dimensions == 0 or any(len(vector) != dimensions for vector in vectors):
            raise RuntimeError("embedding_chunk_projection_dimension_mismatch")

        weights = [max(1, len(chunk)) for chunk in chunks]
        total_weight = float(sum(weights))
        pooled = [
            sum(float(vector[index]) * weight for vector, weight in zip(vectors, weights))
            / total_weight
            for index in range(dimensions)
        ]
        norm = math.sqrt(sum(value * value for value in pooled))
        if norm > 0.0:
            pooled = [value / norm for value in pooled]
        return pooled

    async def rerank(
        self,
        query: str,
        documents: List[str],
        top_n: Optional[int] = None,
        instruction: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Rerank documents for a query using the local reranker endpoint.
        """
        if not documents:
            return []

        client = self._get_rerank_client()
        payload: Dict[str, Any] = {
            "model": self.reranker_model,
            "query": query,
            "documents": documents,
        }
        if top_n is not None:
            payload["top_n"] = max(1, int(top_n))
        if instruction and self._is_local_base_url(self.rerank_base_url):
            payload["instruction"] = instruction

        response = await client.post("/rerank", json=payload)
        response.raise_for_status()
        data = response.json()
        results = data.get("results", data.get("data", []))
        if not isinstance(results, list):
            logger.error("rerank_invalid_response reason=missing_results")
            raise RuntimeError("rerank_response_missing_results")
        out: list[dict[str, Any]] = []
        for row in results:
            if not isinstance(row, dict):
                logger.error("rerank_invalid_response reason=invalid_result")
                raise RuntimeError("rerank_response_contains_invalid_result")
            out.append(row)
        if not out:
            logger.error("rerank_invalid_response reason=empty_results")
            raise RuntimeError("rerank_response_empty")
        return out

    async def verify_api_key(self) -> tuple[bool, Optional[str]]:
        """
        Verify that the API key is valid OpenRouter key

        Returns:
            Tuple of (is_valid, error_message)
        """
        if self._is_local_base_url(self.base_url):
            # Local gateway mode: verify endpoint reachability, not remote key semantics.
            try:
                client = self._get_chat_client()
                # Bound local verify so /api/auth/register always responds
                # promptly even if the local gateway (127.0.0.1:11437) stalls
                # (RWO-20261024T081301Z-repair-memfab-token-refresh-endpoint-e5f522).
                response = await client.get("/models", timeout=8.0)
                if response.status_code == 200:
                    return True, None
                return False, f"Local gateway error: {response.status_code}"
            except Exception as e:
                return False, f"Connection error: {str(e)}"

        # Cloud mode: enforce OpenRouter key semantics.
        if not self.api_key or not self.api_key.startswith("sk-or-"):
            return (
                False,
                "Invalid key format. OpenRouter API keys start with 'sk-or-'. Get yours at openrouter.ai/keys",
            )

        try:
            client = self._get_chat_client()
            # Use /auth/key endpoint to verify the key
            response = await client.get("/auth/key")
            if response.status_code == 200:
                data = response.json()
                # Check if key data is returned (valid key)
                if data.get("data"):
                    return True, None
                return False, "Invalid API key"
            elif response.status_code == 401:
                return False, "Invalid or expired API key"
            elif response.status_code == 403:
                return False, "API key access denied"
            else:
                return False, f"API error: {response.status_code}"
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

    def _extract_balanced_braces(
        self, text: str, open_char: str, close_char: str
    ) -> Optional[str]:
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
                cleaned = cleaned[len(prefix) :].strip()

        # Remove trailing commas before } or ]
        cleaned = re.sub(r",\s*([\}\]])", r"\1", cleaned)

        # Remove single-line comments
        cleaned = re.sub(r"//.*$", "", cleaned, flags=re.MULTILINE)

        return cleaned


class OpenRouterClientManager:
    """
    Manages OpenRouter client instances for multiple users.
    Provides client pooling and lifecycle management with LRU eviction.
    """

    _MAX_CACHED_CLIENTS = 128

    def __init__(
        self,
        base_url: str,
        embedding_base_url: Optional[str] = None,
        rerank_base_url: Optional[str] = None,
        llm_model: str = "openai/gpt-4.1-mini",
        embedding_model: str = "BAAI/bge-m3",
        reranker_model: str = "BAAI/bge-reranker-v2-m3",
    ):
        base_url, embedding_url, rerank_url = _local_endpoints(
            base_url, embedding_base_url, rerank_base_url
        )
        self.base_url = base_url
        self.embedding_base_url = embedding_url
        self.rerank_base_url = rerank_url
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        self.reranker_model = reranker_model
        self._clients: Dict[str, OpenRouterClient] = {}
        self._client_order: List[str] = []

    def _evict_oldest(self) -> None:
        import asyncio

        while len(self._clients) > self._MAX_CACHED_CLIENTS and self._client_order:
            oldest_key = self._client_order.pop(0)
            client = self._clients.pop(oldest_key, None)
            if client is not None:
                try:
                    loop = asyncio.get_running_loop()
                    loop.create_task(client.close())
                except RuntimeError:
                    pass

    def get_client(self, api_key: str) -> OpenRouterClient:
        import hashlib

        key_hash = hashlib.sha256(api_key.encode()).hexdigest()[:16]

        if key_hash in self._clients:
            self._client_order.remove(key_hash)
            self._client_order.append(key_hash)
            return self._clients[key_hash]

        self._clients[key_hash] = OpenRouterClient(
            api_key=api_key,
            base_url=self.base_url,
            embedding_base_url=self.embedding_base_url,
            rerank_base_url=self.rerank_base_url,
            llm_model=self.llm_model,
            embedding_model=self.embedding_model,
            reranker_model=self.reranker_model,
        )
        self._client_order.append(key_hash)
        self._evict_oldest()

        return self._clients[key_hash]

    async def close_all(self):
        """Close all client connections"""
        for client in self._clients.values():
            await client.close()
        self._clients.clear()
        self._client_order.clear()

    async def remove_client(self, api_key: str):
        """Remove and close a specific client"""
        import hashlib

        key_hash = hashlib.sha256(api_key.encode()).hexdigest()[:16]

        if key_hash in self._clients:
            await self._clients[key_hash].close()
            del self._clients[key_hash]
            self._client_order.remove(key_hash)
