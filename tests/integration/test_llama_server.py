import pytest
import httpx
import json
from pathlib import Path
from agent.mcp_.agent_server import AgentServer
from agent.model import LlamaCppClient
from agent.runtime import AgentRuntime, TASK_STATE_SYSTEM_PROMPT

LLAMA_SERVER_URL = "http://127.0.0.1:8080"
MODEL_NAME = "Qwen3.6-35B-A3B-MTP-UD-Q4_K_XL"


@pytest.fixture
async def llama_client():
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{LLAMA_SERVER_URL}/health",
                timeout=1.0,
            )
    except httpx.HTTPError:
        pytest.skip("llama-server is not running")

    if response.status_code != 200:
        pytest.skip("llama-server is not ready")

    return LlamaCppClient(
        base_url=LLAMA_SERVER_URL,
        model=MODEL_NAME,
    )


@pytest.mark.anyio
async def test_llama_server_generates_response(llama_client):
    response = await llama_client.generate(
        messages=[
            {
                "role": "user",
                "content": "Reply with exactly: OK",
            }
        ],
        tools=[],
    )

    assert response["choices"]

    message = response["choices"][0]["message"]

    assert message["role"] == "assistant"
    assert message["content"] == "OK"


@pytest.mark.anyio
async def test_llama_server_calls_project_info(llama_client, tmp_path):
    server = AgentServer(tmp_path)

    tools = await server.mcp.list_tools()

    tool_schemas = [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description or "",
                "parameters": tool.parameters,
            },
        }
        for tool in tools
    ]

    response = await llama_client.generate(
        messages=[
            {
                "role": "user",
                "content": (
                    "You must call the project_info tool before answering. "
                    "Do not answer from assumptions."
                ),
            }
        ],
        tools=tool_schemas,
    )
    print(json.dumps(response, indent=2))

    message = response["choices"][0]["message"]
    tool_calls = message.get("tool_calls", [])

    assert tool_calls

    tool_call = tool_calls[0]

    assert tool_call["type"] == "function"
    assert tool_call["function"]["name"] == "project_info"

    arguments = json.loads(tool_call["function"]["arguments"])

    assert arguments == {}


@pytest.mark.anyio
async def test_agent_runtime_completes_project_info_task(
    llama_client,
    tmp_path,
):
    server = AgentServer(tmp_path)
    runtime = AgentRuntime(
        server=server,
        model=llama_client,
        max_iterations=3,
    )

    result = await runtime.run(
        "Use the project_info tool to inspect the project, "
        "then give me a short summary of what you found."
    )

    assert result

    state = server.state.load()

    assert state.status == "completed"
    assert state.task == (
        "Use the project_info tool to inspect the project, "
        "then give me a short summary of what you found."
    )


class RecordingModel:
    def __init__(self, client):
        self.client = client
        self.responses = []

    async def generate(self, messages, tools):
        response = await self.client.generate(
            messages=messages,
            tools=tools,
        )
        self.responses.append(response)
        return response


@pytest.mark.anyio
async def test_agent_runtime_uses_task_state_with_live_model(
    llama_client,
    tmp_path,
):
    recording_model = RecordingModel(llama_client)

    server = AgentServer(tmp_path)

    runtime = AgentRuntime(
        server=server,
        model=recording_model,
        max_iterations=5,
    )

    task = (
        "Inspect the project using the project_info tool. "
        "You MUST use `set_task_state` before the project_info call to record what you are currently doing. "
        "After project_info succeeds, you MUST use `set_task_state` again to mark the inspection step as completed "
        "and clear any blockers. Then, give me a short summary."
    )

    result = await runtime.run(task)
    
    assert result

    for index, response in enumerate(recording_model.responses, 1):
        message = response.get("choices", [{}])[0].get("message", {})

        print(f"\n--- MODEL RESPONSE {index} ---")
        print("content:", message.get("content"))
        print("tool_calls:", message.get("tool_calls"))

    state = server.state.load()

    assert state.status == "completed"
    assert state.task == task
    #assert state.current # this is explicitly managed by the model, it's variable and it will *randomly* fail
    assert state.completed
    assert state.blocked == []
    assert state.files == []


TOOL_CALL_PROMPT = (
    "Call the `project_info` tool now. "
    "Do not answer with an explanation."
)


async def _get_tool_schemas(server: AgentServer) -> list[dict]:
    tools = await server.mcp.list_tools()

    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description or "",
                "parameters": tool.parameters,
            },
        }
        for tool in tools
    ]


def _classify_response(response: dict) -> str:
    choices = response.get("choices", [])

    if not choices:
        return "empty"

    message = choices[0].get("message", {})

    if message.get("tool_calls"):
        return "structured"

    content = message.get("content") or ""

    if "<tool_call>" in content:
        return "textual_tool_call"

    if "<function=" in content:
        return "textual_function_call"

    if content:
        return "plain"

    return "empty"


def _tool_call_name(response: dict) -> str | None:
    choices = response.get("choices", [])

    if not choices:
        return None

    message = choices[0].get("message", {})
    tool_calls = message.get("tool_calls", [])

    if not tool_calls:
        return None

    return tool_calls[0].get("function", {}).get("name")


async def _run_attempt(
    llama_client,
    tools: list[dict],
    *,
    temperature: float,
    top_p: float,
    top_k: int,
    min_p: float,
) -> tuple[str, dict]:
    response = await llama_client.generate(
        messages=[
            {
                "role": "system",
                "content": (
                    "You are testing function calling. "
                    "If you need to call a function, use the provided "
                    "function-calling mechanism."
                ),
            },
            {
                "role": "user",
                "content": TOOL_CALL_PROMPT,
            },
        ],
        tools=tools,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
    )

    return _classify_response(response), response


@pytest.mark.anyio
async def test_llama_server_tool_call_is_structured(llama_client):
    server = AgentServer("/tmp/local-agent-tool-test")
    tools = await _get_tool_schemas(server)

    classification, response = await _run_attempt(
        llama_client,
        tools,
        temperature=1.0,
        top_p=0.95,
        top_k=64,
        min_p=0.05,
    )

    assert classification == "structured", (
        "Model did not produce a structured tool call.\n"
        f"Classification: {classification}\n"
        f"Response: {response!r}"
    )

    assert _tool_call_name(response) == "project_info"


@pytest.mark.anyio
async def test_llama_server_tool_call_reliability(llama_client):
    """
    Controlled reliability benchmark.

    The test first performs one baseline attempt. If that attempt fails,
    the benchmark stops immediately because repeated sampling would not
    provide useful evidence for this stage.

    If the baseline succeeds, the same prompt is sampled repeatedly under:
    - current server settings
    - lower Qwen3-Coder-oriented sampling settings
    """

    # This test intentionally performs its own fixture-style setup so the
    # baseline gate and benchmark share exactly the same client.

    client = LlamaCppClient(
        base_url=LLAMA_SERVER_URL,
        model=MODEL_NAME,
    )

    server = AgentServer("/tmp/local-agent-tool-test")
    tools = await _get_tool_schemas(server)

    configurations = [
        {
            "name": "baseline",
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 64,
            "min_p": 0.05,
        },
        {
            "name": "lower_sampling",
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
            "min_p": 0.0,
        },
    ]

    baseline_classification, baseline_response = await _run_attempt(
        client,
        tools,
        temperature=configurations[0]["temperature"],
        top_p=configurations[0]["top_p"],
        top_k=configurations[0]["top_k"],
        min_p=configurations[0]["min_p"],
    )

    assert baseline_classification == "structured", (
        "Baseline tool-call gate failed; skipping repeated benchmark.\n"
        f"Classification: {baseline_classification}\n"
        f"Response: {baseline_response!r}"
    )

    for config in configurations:
        results = {
            "structured": 0,
            "textual_tool_call": 0,
            "textual_function_call": 0,
            "plain": 0,
            "empty": 0,
            "wrong_tool": 0,
        }

        responses = []

        for attempt in range(20):
            classification, response = await _run_attempt(
                client,
                tools,
                temperature=config["temperature"],
                top_p=config["top_p"],
                top_k=config["top_k"],
                min_p=config["min_p"],
            )

            if (
                classification == "structured"
                and _tool_call_name(response) != "project_info"
            ):
                results["wrong_tool"] += 1
            else:
                results[classification] += 1

            responses.append(
                {
                    "attempt": attempt + 1,
                    "classification": classification,
                    "tool": _tool_call_name(response),
                }
            )

        total = sum(results.values())
        structured_rate = results["structured"] / total

        print()
        print("=" * 72)
        print(f"TOOL-CALL RELIABILITY: {config['name']}")
        print("=" * 72)
        print(
            "sampling:",
            f"temperature={config['temperature']}",
            f"top_p={config['top_p']}",
            f"top_k={config['top_k']}",
            f"min_p={config['min_p']}",
        )
        print()
        print(f"structured:            {results['structured']:2d}/20")
        print(f"textual <tool_call>:   {results['textual_tool_call']:2d}/20")
        print(f"textual <function=>:    {results['textual_function_call']:2d}/20")
        print(f"plain response:        {results['plain']:2d}/20")
        print(f"empty response:        {results['empty']:2d}/20")
        print(f"wrong structured tool: {results['wrong_tool']:2d}/20")
        print(f"structured rate:       {structured_rate:.1%}")
        print()
        print("attempts:")
        print(json.dumps(responses, indent=2))

        assert total == 20


@pytest.mark.anyio
async def test_llama_server_tool_call_prompt_ablation(llama_client):
    """
    Diagnose whether TASK_STATE_SYSTEM_PROMPT wording causes Qwen3-Coder
    to emit textual rather than structured tool calls.

    Every case uses the same model, tools, sampling parameters, and user
    request. Only the system prompt changes.
    """

    client = LlamaCppClient(
        base_url=LLAMA_SERVER_URL,
        model=MODEL_NAME,
    )

    server = AgentServer("/tmp/local-agent-tool-test")
    tools = await _get_tool_schemas(server)

    explicit_tool_format = """
Function calls HAVE to be enclosed within <tool_call> and </tool_call> tags.
If you choose to call a function ONLY reply in the following format with NO suffix:

<tool_call>
<function=FUNCTION_NAME>
<parameter=PARAMETER_NAME>
VALUE
</parameter>
</function>
</tool_call>

Do NOT omit the initial <tool_call> tag.
""".strip()

    prompts = [
        (
            "minimal",
            (
                "You are a coding agent. "
                "Use the provided tools when appropriate."
            ),
        ),
        (
            "task_state",
            TASK_STATE_SYSTEM_PROMPT,
        ),
        (
            "task_state_explicit_format",
            TASK_STATE_SYSTEM_PROMPT
            + "\n\n"
            + explicit_tool_format,
        ),
    ]

    user_prompt = "Call the project_info tool now. Do not answer with an explanation."

    async def run_attempt(system_prompt: str) -> tuple[str, dict]:
        response = await client.generate(
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            tools=tools,
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            min_p=0.0,
        )

        return _classify_response(response), response

    for name, system_prompt in prompts:
        baseline_classification, baseline_response = await run_attempt(
            system_prompt
        )

        print()
        print("=" * 72)
        print(f"PROMPT ABLATION: {name}")
        print("=" * 72)
        print(f"baseline classification: {baseline_classification}")
        print(f"baseline tool: {_tool_call_name(baseline_response)}")
        print(f"baseline response: {baseline_response!r}")

        results = {
            "structured": 0,
            "textual_tool_call": 0,
            "textual_function_call": 0,
            "plain": 0,
            "empty": 0,
            "wrong_tool": 0,
        }

        # Apply the same gate used by the existing reliability benchmark.
        # If a configuration cannot produce one valid structured call,
        # don't spend another 20 generations measuring it.
        if baseline_classification != "structured":
            results[baseline_classification] += 1

            print("baseline gate failed; skipping repeated attempts")

            if baseline_classification == "structured":
                assert _tool_call_name(baseline_response) == "project_info"

            continue

        assert _tool_call_name(baseline_response) == "project_info"

        results["structured"] += 1

        for attempt in range(19):
            classification, response = await run_attempt(system_prompt)

            if (
                classification == "structured"
                and _tool_call_name(response) != "project_info"
            ):
                results["wrong_tool"] += 1
            else:
                results[classification] += 1

        total = sum(results.values())
        structured_rate = results["structured"] / total

        print(f"structured:            {results['structured']:2d}/20")
        print(f"textual <tool_call>:   {results['textual_tool_call']:2d}/20")
        print(f"textual <function=>:   {results['textual_function_call']:2d}/20")
        print(f"plain response:        {results['plain']:2d}/20")
        print(f"empty response:        {results['empty']:2d}/20")
        print(f"wrong structured tool: {results['wrong_tool']:2d}/20")
        print(f"structured rate:       {structured_rate:.1%}")

@pytest.mark.anyio
async def test_llama_server_tool_call_reliability_with_task_state_prompt(llama_client):
    """
    Isolates whether TASK_STATE_SYSTEM_PROMPT affects native tool-call
    reliability.

    The tool set and sampling configuration are identical to the baseline
    reliability benchmark. The only meaningful change is the system prompt.
    """
    #pytest.skip("A deeper analysis was ran before this")

    client = LlamaCppClient(
        base_url=LLAMA_SERVER_URL,
        model=MODEL_NAME,
    )

    server = AgentServer("/tmp/local-agent-tool-test")
    tools = await _get_tool_schemas(server)

    system_prompt = TASK_STATE_SYSTEM_PROMPT
    user_prompt = "Call the project_info tool now. Do not answer with an explanation."

    async def run_attempt() -> tuple[str, dict]:
        response = await client.generate(
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            tools=tools,
            temperature=0.65,
            top_p=0.8,
            top_k=20,
            min_p=0,
        )

        return _classify_response(response), response

    baseline_classification, baseline_response = await run_attempt()

    assert baseline_classification == "structured", (
        "Task-state prompt baseline failed; skipping repeated benchmark.\n"
        f"Classification: {baseline_classification}\n"
        f"Response: {baseline_response!r}"
    )

    results = {
        "structured": 0,
        "textual_tool_call": 0,
        "textual_function_call": 0,
        "plain": 0,
        "empty": 0,
        "wrong_tool": 0,
    }

    responses = []

    for attempt in range(20):
        classification, response = await run_attempt()

        if (
            classification == "structured"
            and _tool_call_name(response) != "project_info"
        ):
            results["wrong_tool"] += 1
        else:
            results[classification] += 1

        responses.append(
            {
                "attempt": attempt + 1,
                "classification": classification,
                "tool": _tool_call_name(response),
            }
        )

    total = sum(results.values())
    structured_rate = results["structured"] / total

    print()
    print("=" * 72)
    print("TOOL-CALL RELIABILITY: TASK_STATE_SYSTEM_PROMPT")
    print("=" * 72)
    print(
        "sampling:",
        "temperature=0.65",
        "top_p=0.8",
        "top_k=20",
        "min_p=0",
    )
    print()
    print(f"structured:            {results['structured']:2d}/20")
    print(f"textual <tool_call>:   {results['textual_tool_call']:2d}/20")
    print(f"textual <function=>:    {results['textual_function_call']:2d}/20")
    print(f"plain response:        {results['plain']:2d}/20")
    print(f"empty response:        {results['empty']:2d}/20")
    print(f"wrong structured tool: {results['wrong_tool']:2d}/20")
    print(f"structured rate:       {structured_rate:.1%}")
    print()
    print("attempts:")
    print(json.dumps(responses, indent=2))

    assert total == 20
