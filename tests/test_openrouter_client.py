"""Tests for OpenRouter client with witness tracking."""

import os
import pytest

# Skip all tests if no API key
pytestmark = pytest.mark.skipif(
    not os.environ.get("OPENROUTER_API_KEY"),
    reason="OPENROUTER_API_KEY not set"
)


def test_client_initialization():
    """Test client can be created with API key."""
    from modelforecast.clients.openrouter import OpenRouterClient

    client = OpenRouterClient()
    assert client.api_key is not None
    assert client.report.total_calls == 0


def test_chat_completion():
    """Test basic chat completion."""
    from modelforecast.clients.openrouter import OpenRouterClient

    client = OpenRouterClient()

    response = client.chat.completions.create(
        model="meta-llama/llama-3.2-3b-instruct",
        messages=[{"role": "user", "content": "Say 'hello'"}],
        max_tokens=10,
        temperature=0.1,
    )

    assert response.choices[0].message.content is not None
    assert response.usage.prompt_tokens > 0

    # Check witness tracking
    assert client.report.total_calls == 1
    assert client.report.successful_calls == 1


def test_tool_calling():
    """Test tool calling with Claude."""
    from modelforecast.clients.openrouter import OpenRouterClient

    client = OpenRouterClient()

    tools = [{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                },
                "required": ["location"],
            },
        },
    }]

    response = client.chat.completions.create(
        model="anthropic/claude-3.5-haiku",
        messages=[{"role": "user", "content": "What's the weather in Tokyo?"}],
        tools=tools,
        temperature=0.1,
    )

    choice = response.choices[0]
    assert choice.message.tool_calls is not None
    assert len(choice.message.tool_calls) > 0
    assert choice.message.tool_calls[0].function.name == "get_weather"


def test_embeddings():
    """Test embedding generation."""
    from modelforecast.clients.openrouter import OpenRouterClient

    client = OpenRouterClient()

    response = client.embeddings.create(
        model="qwen/qwen3-embedding-4b",
        input=["Hello world", "Test text"],
    )

    assert len(response.data) == 2
    assert len(response.data[0].embedding) > 0


def test_cost_tracking():
    """Test cost tracking for paid models."""
    from modelforecast.clients.openrouter import OpenRouterClient

    client = OpenRouterClient()

    # Use a paid model
    client.chat.completions.create(
        model="meta-llama/llama-3.2-3b-instruct",
        messages=[{"role": "user", "content": "Hi"}],
        max_tokens=5,
    )

    # Check cost was tracked
    assert client.report.total_calls == 1
    call = client.report.calls[0]
    assert call.input_tokens > 0
    assert call.total_cost > 0  # Paid model should have cost


def test_report_generation():
    """Test report generation."""
    from modelforecast.clients.openrouter import OpenRouterClient

    client = OpenRouterClient()

    client.chat.completions.create(
        model="meta-llama/llama-3.2-3b-instruct",
        messages=[{"role": "user", "content": "Hi"}],
        max_tokens=5,
    )

    report = client.get_report()
    assert report.total_calls == 1

    # Check markdown generation
    md = report.to_markdown()
    assert "OpenRouter Usage Report" in md
    assert "Total Calls" in md


def test_probe_compatibility():
    """Test client is compatible with T0 probe interface."""
    from modelforecast.clients.openrouter import OpenRouterClient
    from modelforecast.probes.t0_invoke import T0InvokeProbe

    client = OpenRouterClient()
    probe = T0InvokeProbe()

    # Run probe with paid model that supports tools
    result = probe.run("anthropic/claude-3.5-haiku", client)

    assert result.success is True
    assert result.tool_called is True
    assert result.tool_name == "search"

    # Verify witness tracked the call
    assert client.report.total_calls == 1


def test_error_tracking():
    """Test that errors are properly tracked."""
    from modelforecast.clients.openrouter import OpenRouterClient

    client = OpenRouterClient()

    # Try with invalid model
    try:
        client.chat.completions.create(
            model="invalid/model-that-does-not-exist",
            messages=[{"role": "user", "content": "Hi"}],
        )
    except RuntimeError:
        pass  # Expected

    # Error should be tracked
    assert client.report.total_calls == 1
    assert client.report.successful_calls == 0
    assert client.report.calls[0].error is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
