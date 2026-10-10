from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import httpx


BGE_QUERY_PREFIX = (
    "Represent this sentence for searching relevant passages: "
)


class EmbeddingProvider(Protocol):
    def embed_documents(
        self,
        texts: Sequence[str],
    ) -> list[list[float]]:
        ...

    def embed_query(self, text: str) -> list[float]:
        ...


class LlamaCppEmbeddingClient:
    """OpenAI-compatible embedding client for llama-server."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8081",
        model: str = "bge-small-en-v1.5",
        *,
        timeout: float = 180.0,
        query_prefix: str = BGE_QUERY_PREFIX,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.query_prefix = query_prefix

    def embed_documents(
        self,
        texts: Sequence[str],
    ) -> list[list[float]]:
        prepared = [self._validate_text(text) for text in texts]
        if not prepared:
            return []
        return self._request(prepared)

    def embed_query(self, text: str) -> list[float]:
        query = self._validate_text(text)
        if self.query_prefix:
            query = f"{self.query_prefix}{query}"
        return self._request([query])[0]

    @staticmethod
    def _validate_text(text: str) -> str:
        if not isinstance(text, str):
            raise TypeError("embedding text must be a string")
        text = text.strip()
        if not text:
            raise ValueError("embedding text cannot be empty")
        return text

    def _request(self, texts: list[str]) -> list[list[float]]:
        url = f"{self.base_url}/v1/embeddings"

        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(
                    url,
                    json={
                        "model": self.model,
                        "input": texts,
                        "encoding_format": "float",
                    },
                )
                response.raise_for_status()
        except httpx.HTTPError as exc:
            raise RuntimeError(
                "embedding request failed; ensure the llama.cpp embedding "
                f"server is running at {self.base_url}: {exc}"
            ) from exc

        try:
            payload = response.json()
            items = payload["data"]
        except (ValueError, KeyError, TypeError) as exc:
            raise RuntimeError(
                "embedding server returned an invalid response"
            ) from exc

        if not isinstance(items, list) or len(items) != len(texts):
            raise RuntimeError(
                "embedding server returned an unexpected number of vectors"
            )

        try:
            ordered = sorted(items, key=lambda item: int(item["index"]))
            vectors = [
                [float(value) for value in item["embedding"]]
                for item in ordered
            ]
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                "embedding server returned malformed vectors"
            ) from exc

        if not vectors or not vectors[0]:
            raise RuntimeError("embedding server returned an empty vector")

        dimensions = len(vectors[0])
        if any(len(vector) != dimensions for vector in vectors):
            raise RuntimeError(
                "embedding server returned inconsistent vector dimensions"
            )

        return vectors
