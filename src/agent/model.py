from __future__ import annotations
from typing import Any
import httpx
from dataclasses import dataclass

@dataclass(frozen=True)
class GenerationConfig:
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 20
    min_p: float = 0.0
    repeat_penalty: float = 1.0
    reasoning: str = "auto"
    reasoning_budget: int = 16384
    reasoning_budget_message: str = (
        "<system>Reasoning token-limit reached. "
        "Maintain identity and conversation context. "
        "Answer from existing conclusions.</system>"
    )

class LlamaCppClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        generation: GenerationConfig | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        timeout: httpx.Timeout | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.generation = generation or GenerationConfig()
        self.transport = transport
        self.timeout = timeout or httpx.Timeout(
            connect=10.0,
            read=None,
            write=30.0,
            pool=10.0,
        )

    async def generate(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        min_p: float | None = None,
        repeat_penalty: float | None = None,
    ) -> dict[str, Any]:
        async with httpx.AsyncClient(
            transport=self.transport,
            base_url=self.base_url,
            timeout=self.timeout,
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                json={
                    "model": self.model,
                    "messages": messages,
                    "tools": tools,
                    "stream": False,
                    "temperature": (
                        temperature
                        if temperature is not None
                        else self.generation.temperature
                    ),
                    "top_p": (
                        top_p
                        if top_p is not None
                        else self.generation.top_p
                    ),
                    "top_k": (
                        top_k
                        if top_k is not None
                        else self.generation.top_k
                    ),
                    "min_p": (
                        min_p
                        if min_p is not None
                        else self.generation.min_p
                    ),
                    "repeat_penalty": (
                        repeat_penalty
                        if repeat_penalty is not None
                        else self.generation.repeat_penalty
                    ),
                    "reasoning": self.generation.reasoning,
                    "thinking_budget_tokens": self.generation.reasoning_budget,
                    "reasoning_budget_message": self.generation.reasoning_budget_message,
                },
            )

            response.raise_for_status()

            data = response.json()
            if "usage" in data:
                data["usage"] = {
                    "prompt_tokens": data["usage"].get("prompt_tokens", 0),
                    "completion_tokens": data["usage"].get("completion_tokens", 0),
                    "total_tokens": data["usage"].get("total_tokens", 0),
                }
            return data
