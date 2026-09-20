import pytest
import asyncio
import json
from unittest.mock import AsyncMock, patch, MagicMock
from agent.model import LlamaCppClient
from agent.runtime import AgentRuntime, AgentResponse
from agent.mcp_.agent_server import AgentServer
from pathlib import Path


@pytest.fixture
def mock_client():
    return LlamaCppClient(base_url="http://127.0.0.1:8080", model="test-model")


@pytest.fixture
def mock_server():
    server = MagicMock(spec=AgentServer)
    server.mcp = AsyncMock()
    server.mcp.list_tools = AsyncMock(return_value=[])
    server.state = MagicMock()
    server.state.load = MagicMock(return_value=MagicMock(files=[]))
    server.state.save = MagicMock()
    return server


class TestModelUsageExtraction:
    @pytest.mark.anyio
    async def test_usage_data_extracted_from_response(self, mock_client):
        mock_response = {
            "choices": [{"message": {"content": "Hello"}}],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
            }
        }
        with patch.object(mock_client, "generate", new_callable=AsyncMock) as mock_gen:
            mock_gen.return_value = mock_response
            result = await mock_client.generate(messages=[{"role": "user", "content": "hi"}], tools=[])
            
            assert "usage" in result
            assert result["usage"]["prompt_tokens"] == 10
            assert result["usage"]["completion_tokens"] == 20
            assert result["usage"]["total_tokens"] == 30

    @pytest.mark.anyio
    async def test_missing_usage_returns_clean_dict(self, mock_client):
        mock_response = {"choices": [{"message": {"content": "No stats here"}}]}
        with patch.object(mock_client, "generate", new_callable=AsyncMock) as mock_gen:
            mock_gen.return_value = mock_response
            result = await mock_client.generate(messages=[{"role": "user", "content": "hi"}], tools=[])
            
            assert "usage" not in result


class TestRuntimeParseResponse:
    def test_text_response_attaches_usage(self, mock_server):
        runtime = AgentRuntime(server=mock_server, model=AsyncMock(), max_iterations=1)
        response_data = {
            "choices": [{"message": {"content": "I did it!"}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 15, "total_tokens": 20}
        }
        agent_resp = AgentRuntime._parse_response(response_data)
        
        assert agent_resp.content == "I did it!"
        assert agent_resp.usage is not None
        assert agent_resp.usage["total_tokens"] == 20

    def test_tool_call_response_attaches_usage(self, mock_server):
        runtime = AgentRuntime(server=mock_server, model=AsyncMock(), max_iterations=1)
        response_data = {
            "choices": [{"message": {"tool_calls": [{"id": "1", "type": "function", "function": {"name": "test", "arguments": "{}"}}]}}],
            "usage": {"prompt_tokens": 8, "completion_tokens": 12, "total_tokens": 20}
        }
        agent_resp = AgentRuntime._parse_response(response_data)
        
        assert agent_resp.tool_calls is not None
        assert len(agent_resp.tool_calls) == 1
        assert agent_resp.usage is not None
        assert agent_resp.usage["prompt_tokens"] == 8

    def test_no_choices_returns_empty_usage(self, mock_server):
        runtime = AgentRuntime(server=mock_server, model=AsyncMock(), max_iterations=1)
        response_data = {"choices": []}
        agent_resp = AgentRuntime._parse_response(response_data)
        
        assert agent_resp.content == ""
        assert agent_resp.usage is None


class TestUsageInjectionInRunLoop:
    @pytest.mark.anyio
    async def test_stats_injected_into_messages(self, mock_server):
        runtime = AgentRuntime(server=mock_server, model=AsyncMock(), max_iterations=2)
        
        # Iteration 1 returns a tool call to keep the loop going, iteration 2 completes
        mock_llm_response = {
            "choices": [{"message": {"content": "Task done"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 25, "total_tokens": 35}
        }
        runtime.model.generate = AsyncMock(return_value=mock_llm_response)
        runtime._get_tools = AsyncMock(return_value=[])
        
        await runtime.run("Test task")
        
        # Get the messages passed to model.generate on the call
        call_kwargs = runtime.model.generate.call_args.kwargs
        messages = call_kwargs["messages"]

        system_messages = [msg for msg in messages if msg.get("role") == "system"]
        stats_msg = [msg for msg in system_messages if "[Session Stats:" in msg.get("content", "")]
        
        assert stats_msg != []
        assert "Prompt=10" in stats_msg[0]["content"]
        assert "Completion=25" in stats_msg[0]["content"]
        assert "Total=35" in stats_msg[0]["content"]
