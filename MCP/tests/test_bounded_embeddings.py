import math

import pytest

from server.integrations.openrouter import OpenRouterClient


class RecordingEmbeddingClient(OpenRouterClient):
    def __init__(self) -> None:
        super().__init__(api_key="", embedding_base_url="http://127.0.0.1:11501/v1")
        self.requests: list[str] = []

    async def create_single_embedding(self, text: str, **_kwargs: object) -> list[float]:
        self.requests.append(text)
        return [float(len(text)), 1.0]


@pytest.mark.asyncio
async def test_bounded_semantic_embedding_chunks_and_normalizes() -> None:
    client = RecordingEmbeddingClient()
    source = "x" * 3002

    vector = await client.create_bounded_semantic_embedding(source)

    assert len(client.requests) == math.ceil(
        len(source) / client._MAX_EMBEDDING_CHARS_PER_INPUT
    )
    assert all(len(chunk) <= 1024 for chunk in client.requests)
    assert math.isclose(math.sqrt(sum(value * value for value in vector)), 1.0)
