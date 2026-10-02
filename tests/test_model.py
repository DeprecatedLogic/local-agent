import json

import httpx
import pytest

from agent.events import AgentEvent
from agent.model import GenerationConfig, LlamaCppClient

@pytest.mark.anyio
async def test_llama_cpp_client_generates_response():
    requests = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)

        return httpx.Response(
            200,
            text=(
                'data: {"choices":[{"delta":'
                '{"content":"Hello"},'
                '"finish_reason":"stop"}]}\n\n'
                "data: [DONE]\n\n"
            ),
        )

    transport = httpx.MockTransport(handler)

    client = LlamaCppClient(
        base_url="http://localhost:8080",
        model="test-model",
        transport=transport,
    )

    messages = [
        {
            "role": "user",
            "content": "Hello",
        }
    ]

    tools = [
        {
            "type": "function",
            "function": {
                "name": "test_tool",
                "description": "A test tool.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                },
            },
        }
    ]

    result = await client.generate(
        messages=messages,
        tools=tools,
    )

    assert result["choices"][0]["message"]["content"] == "Hello"

    assert len(requests) == 1

    request = requests[0]

    assert request.method == "POST"
    assert request.url == "http://localhost:8080/v1/chat/completions"

    body = json.loads(request.content)

    assert body["model"] == "test-model"
    assert body["messages"] == messages
    assert body["tools"] == tools


@pytest.mark.anyio
async def test_llama_cpp_client_raises_on_http_error():
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            500,
            json={"error": "Internal server error"},
        )

    transport = httpx.MockTransport(handler)

    client = LlamaCppClient(
        base_url="http://localhost:8080",
        model="test-model",
        transport=transport,
    )

    with pytest.raises(httpx.HTTPStatusError):
        await client.generate(
            messages=[{
                "role": "user",
                "content": "Hello",
            }],
            tools=[],
        )


@pytest.mark.anyio
async def test_llama_cpp_client_raises_on_invalid_json():
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            text=(
                "data: this is not json\n\n"
                "data: [DONE]\n\n"
            )
        )

    transport = httpx.MockTransport(handler)

    client = LlamaCppClient(
        base_url="http://localhost:8080",
        model="test-model",
        transport=transport,
    )

    with pytest.raises(ValueError):
        await client.generate(
            messages=[{
                "role": "user",
                "content": "Hello",
            }],
            tools=[],
        )


@pytest.mark.anyio
async def test_llama_cpp_client_normalizes_base_url():
    requests = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)

        return httpx.Response(
            200,
            json={"content": "ok"},
        )

    client = LlamaCppClient(
        base_url="http://localhost:8080///",
        model="test-model",
        transport=httpx.MockTransport(handler),
    )

    await client.generate(
        messages=[],
        tools=[],
    )

    assert requests[0].url == "http://localhost:8080/v1/chat/completions"


@pytest.mark.anyio
async def test_llama_cpp_client_enables_streaming():
    requests = []

    async def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        requests.append(request)

        return httpx.Response(
            200,
            text="data: [DONE]\n\n",
        )

    client = LlamaCppClient(
        base_url="http://localhost:8080",
        model="test-model",
        transport=httpx.MockTransport(
            handler
        ),
    )

    await client.generate(
        messages=[],
        tools=[],
    )

    body = json.loads(
        requests[0].content
    )

    assert body["stream"] is True

    assert body["stream_options"] == {
        "include_usage": True,
    }


@pytest.mark.anyio
async def test_llama_cpp_client_configures_temperature():
    requests = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)

        return httpx.Response(
            200,
            json={
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": "Hello",
                    }
                }]
            },
        )

    client = LlamaCppClient(
        base_url="http://localhost:8080",
        model="test-model",
        generation=GenerationConfig(temperature=0.2),
        transport=httpx.MockTransport(handler),
    )

    await client.generate(
        messages=[],
        tools=[],
    )

    body = json.loads(requests[0].content)

    assert body["temperature"] == 0.2


@pytest.mark.anyio
async def test_llama_cpp_client_accepts_generation_config():
    requests = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)

        return httpx.Response(
            200,
            json={
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": "Hello",
                    }
                }]
            },
        )

    config = GenerationConfig(
        temperature=0.7,
        top_p=0.9,
        top_k=32,
        repeat_penalty=1.1,
        #reasoning="auto",
        #reasoning_budget=2048,
        #reasoning_budget_message="test message",
    )

    client = LlamaCppClient(
        base_url="http://localhost:8080",
        model="test-model",
        generation=config,
        transport=httpx.MockTransport(handler),
    )

    await client.generate(
        messages=[],
        tools=[],
    )

    body = json.loads(requests[0].content)

    assert body["temperature"] == 0.7
    assert body["top_p"] == 0.9
    assert body["top_k"] == 32
    assert body["repeat_penalty"] == 1.1
    """
    assert body["reasoning"] == "auto"
    assert body["reasoning_budget"] == 2048
    assert body["reasoning_budget_message"] == "test message"
    """


@pytest.mark.anyio
async def test_llama_cpp_client_uses_default_generation_config():
    requests = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)

        return httpx.Response(
            200,
            json={
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": "Hello",
                    }
                }]
            },
        )

    client = LlamaCppClient(
        base_url="http://localhost:8080",
        model="test-model",
        transport=httpx.MockTransport(handler),
    )

    await client.generate(messages=[], tools=[])

    body = json.loads(requests[0].content)

    assert body["temperature"] == client.generation.temperature
    assert body["top_p"] == client.generation.top_p
    assert body["top_k"] == client.generation.top_k
    assert body["repeat_penalty"] == client.generation.repeat_penalty
    """
    assert body["reasoning"] == "auto"
    assert body["reasoning_budget"] == 4096
    assert body["reasoning_budget_message"] == (
        "<system>Reasoning token-limit reached. "
        "Maintain identity and conversation context. "
        "Answer from existing conclusions.</system>"
    )
    """


@pytest.mark.anyio
async def test_llama_cpp_client_configures_timeout():
    custom_timeout = httpx.Timeout(42.0)

    client = LlamaCppClient(
        base_url="http://localhost:8080",
        model="test-model",
        timeout=custom_timeout,
    )

    assert client.timeout is custom_timeout

    client = LlamaCppClient(
        base_url="http://localhost:8080",
        model="test-model",
    )

    assert client.timeout.connect == 10.0
    assert client.timeout.read is None
    assert client.timeout.write == 30.0
    assert client.timeout.pool == 10.0

@pytest.mark.anyio
async def test_model_streams_content():
    events = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)

        assert payload["stream"] is True

        body = "\n\n".join(
            [
                'data: {"choices":[{"delta":{"content":"Hel"}}]}',
                'data: {"choices":[{"delta":{"content":"lo"},'
                '"finish_reason":"stop"}]}',
                (
                    'data: {"choices":[],"usage":{'
                    '"prompt_tokens":10,'
                    '"completion_tokens":2,'
                    '"total_tokens":12}}'
                ),
                "data: [DONE]",
                "",
            ]
        )

        return httpx.Response(
            200,
            text=body,
        )

    client = LlamaCppClient(
        base_url="http://test",
        model="test-model",
        transport=httpx.MockTransport(handler),
    )

    response = await client.generate(
        messages=[
            {
                "role": "user",
                "content": "hello",
            }
        ],
        tools=[],
        on_stream=events.append,
    )

    assert events == [
        AgentEvent(
            kind="content",
            text="Hel",
        ),
        AgentEvent(
            kind="content",
            text="lo",
        ),
    ]

    message = response["choices"][0]["message"]

    assert message["content"] == "Hello"
    assert response["choices"][0]["finish_reason"] == "stop"
    assert response["usage"] == {
        "prompt_tokens": 10,
        "completion_tokens": 2,
        "total_tokens": 12,
    }

@pytest.mark.anyio
async def test_model_streams_reasoning_separately():
    events = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = "\n\n".join(
            [
                (
                    'data: {"choices":[{"delta":{'
                    '"reasoning_content":"I should "}}]}'
                ),
                (
                    'data: {"choices":[{"delta":{'
                    '"reasoning_content":"inspect this."}}]}'
                ),
                (
                    'data: {"choices":[{"delta":{'
                    '"content":"The answer "}}]}'
                ),
                (
                    'data: {"choices":[{"delta":{'
                    '"content":"is 42."},'
                    '"finish_reason":"stop"}]}'
                ),
                "data: [DONE]",
                "",
            ]
        )

        return httpx.Response(
            200,
            text=body,
        )

    client = LlamaCppClient(
        base_url="http://test",
        model="test-model",
        transport=httpx.MockTransport(handler),
    )

    response = await client.generate(
        messages=[],
        tools=[],
        on_stream=events.append,
    )

    assert events == [
        AgentEvent(
            kind="reasoning",
            text="I should ",
        ),
        AgentEvent(
            kind="reasoning",
            text="inspect this.",
        ),
        AgentEvent(
            kind="content",
            text="The answer ",
        ),
        AgentEvent(
            kind="content",
            text="is 42.",
        ),
    ]

    message = response["choices"][0]["message"]

    assert (
        message["reasoning_content"]
        == "I should inspect this."
    )

    assert message["content"] == "The answer is 42."

@pytest.mark.anyio
async def test_model_accumulates_streamed_tool_call_fragments():
    def handler(request: httpx.Request) -> httpx.Response:
        body = "\n\n".join(
            [
                (
                    'data: {"choices":[{"delta":{"tool_calls":['
                    '{"index":0,"id":"call_","type":"function",'
                    '"function":{"name":"read_",'
                    '"arguments":"{\\"path\\":"}}]}}]}'
                ),
                (
                    'data: {"choices":[{"delta":{"tool_calls":['
                    '{"index":0,"id":"123",'
                    '"function":{"name":"file",'
                    '"arguments":"\\"README.md\\"}"}}]},'
                    '"finish_reason":"tool_calls"}]}'
                ),
                "data: [DONE]",
                "",
            ]
        )

        return httpx.Response(
            200,
            text=body,
        )

    client = LlamaCppClient(
        base_url="http://test",
        model="test-model",
        transport=httpx.MockTransport(handler),
    )

    response = await client.generate(
        messages=[],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "parameters": {},
                },
            }
        ],
    )

    choice = response["choices"][0]
    tool_call = choice["message"]["tool_calls"][0]

    assert choice["finish_reason"] == "tool_calls"

    assert tool_call == {
        "id": "123",
        "type": "function",
        "function": {
            "name": "read_file",
            "arguments": '{"path":"README.md"}',
        },
    }

@pytest.mark.anyio
async def test_model_streaming_does_not_require_handler():
    def handler(request: httpx.Request) -> httpx.Response:
        body = "\n\n".join(
            [
                (
                    'data: {"choices":[{"delta":{'
                    '"content":"Hello"}}]}'
                ),
                (
                    'data: {"choices":[{"delta":{'
                    '"content":" world"},'
                    '"finish_reason":"stop"}]}'
                ),
                "data: [DONE]",
                "",
            ]
        )

        return httpx.Response(
            200,
            text=body,
        )

    client = LlamaCppClient(
        base_url="http://test",
        model="test-model",
        transport=httpx.MockTransport(handler),
    )

    response = await client.generate(
        messages=[],
        tools=[],
    )

    assert (
        response["choices"][0]["message"]["content"]
        == "Hello world"
    )
