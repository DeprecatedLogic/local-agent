import httpx
import pytest
import json
from agent.model import GenerationConfig, LlamaCppClient


@pytest.mark.anyio
async def test_llama_cpp_client_generates_response():
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
            content=b"this is not json",
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
async def test_llama_cpp_client_disables_streaming():
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

    await client.generate(
        messages=[{
            "role": "user",
            "content": "Hello",
        }],
        tools=[],
    )

    body = json.loads(requests[0].content)

    assert body["stream"] is False


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
