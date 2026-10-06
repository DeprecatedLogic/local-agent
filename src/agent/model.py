from __future__ import annotations

import json
from dataclasses import dataclass, replace
from typing import Any

import httpx

from agent.events import AgentEvent, StreamHandler


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

    def with_reasoning_budget(self, reasoning_budget: int) -> "LlamaCppClient":
        """Create a logical client for the same backend with a smaller budget."""
        if reasoning_budget < 1:
            raise ValueError("reasoning_budget must be at least 1")

        return LlamaCppClient(
            base_url=self.base_url,
            model=self.model,
            generation=replace(
                self.generation,
                reasoning_budget=reasoning_budget,
            ),
            transport=self.transport,
            timeout=self.timeout,
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
        on_stream: StreamHandler | None = None,
    ) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "messages": messages,
            "tools": tools,
            "stream": True,
            "stream_options": {
                "include_usage": True,
            },
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
            "thinking_budget_tokens": (
                self.generation.reasoning_budget
            ),
            "reasoning_budget_message": (
                self.generation.reasoning_budget_message
            ),
        }

        async with httpx.AsyncClient(
            transport=self.transport,
            base_url=self.base_url,
            timeout=self.timeout,
        ) as client:
            async with client.stream(
                "POST",
                "/v1/chat/completions",
                json=payload,
            ) as response:
                response.raise_for_status()

                content = ""
                reasoning_content = ""
                tool_calls: list[dict[str, Any]] = []
                usage = None
                finish_reason = None

                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue

                    raw = line[5:].lstrip()

                    if not raw:
                        continue

                    if raw == "[DONE]":
                        break

                    chunk = json.loads(raw)

                    raw_usage = chunk.get("usage")
                    if raw_usage is not None:
                        usage = {
                            "prompt_tokens": raw_usage.get(
                                "prompt_tokens",
                                0,
                            ),
                            "completion_tokens": raw_usage.get(
                                "completion_tokens",
                                0,
                            ),
                            "total_tokens": raw_usage.get(
                                "total_tokens",
                                0,
                            ),
                        }

                    choices = chunk.get("choices", [])

                    if not choices:
                        continue

                    choice = choices[0]
                    delta = choice.get("delta", {})

                    if choice.get("finish_reason") is not None:
                        finish_reason = choice["finish_reason"]

                    reasoning_delta = (
                        delta.get("reasoning_content")
                        or ""
                    )

                    if reasoning_delta:
                        reasoning_content += reasoning_delta

                        if on_stream is not None:
                            on_stream(
                                AgentEvent(
                                    kind="reasoning",
                                    text=reasoning_delta,
                                )
                            )

                    content_delta = delta.get("content") or ""

                    if content_delta:
                        content += content_delta

                        if on_stream is not None:
                            on_stream(
                                AgentEvent(
                                    kind="content",
                                    text=content_delta,
                                )
                            )

                    for tool_delta in delta.get("tool_calls", []):
                        index = tool_delta.get("index", 0)

                        while len(tool_calls) <= index:
                            tool_calls.append(
                                {
                                    "id": "",
                                    "type": "function",
                                    "function": {
                                        "name": "",
                                        "arguments": "",
                                    },
                                }
                            )

                        current = tool_calls[index]

                        if tool_delta.get("id"):
                            current["id"] = tool_delta["id"]

                        if tool_delta.get("type"):
                            current["type"] = tool_delta["type"]

                        function = tool_delta.get("function", {})

                        if function.get("name"):
                            current["function"]["name"] += function["name"]

                        if function.get("arguments"):
                            current["function"]["arguments"] += (
                                function["arguments"]
                            )

        message: dict[str, Any] = {
            "role": "assistant",
            "content": content or None,
        }

        if reasoning_content:
            message["reasoning_content"] = reasoning_content

        if tool_calls:
            message["tool_calls"] = tool_calls

        result: dict[str, Any] = {
            "choices": [
                {
                    "finish_reason": finish_reason,
                    "message": message,
                }
            ],
        }

        if usage is not None:
            result["usage"] = usage

        return result
