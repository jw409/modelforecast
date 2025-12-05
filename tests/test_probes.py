"""Unit tests for probe implementations."""

import pytest
from unittest.mock import MagicMock, patch

from modelforecast.probes.base import ProbeResult
from modelforecast.probes.t0_invoke import T0InvokeProbe
from modelforecast.probes.t1_schema import T1SchemaProbe
from modelforecast.probes.t2_selection import T2SelectionProbe


class TestProbeResult:
    """Test ProbeResult dataclass."""

    def test_probe_result_dataclass(self):
        """ProbeResult stores trial data correctly."""
        result = ProbeResult(
            success=True,
            tool_called=True,
            tool_name="search",
            parameters={"query": "test"},
            raw_response={"test": "data"},
            latency_ms=150,
            error=None,
        )
        assert result.success is True
        assert result.tool_called is True
        assert result.tool_name == "search"
        assert result.latency_ms == 150

    def test_probe_result_failure(self):
        """ProbeResult handles failure case."""
        result = ProbeResult(
            success=False,
            tool_called=False,
            tool_name=None,
            parameters=None,
            raw_response={},
            latency_ms=0,
            error="Connection timeout",
        )
        assert result.success is False
        assert result.error == "Connection timeout"

    def test_probe_result_repr(self):
        """ProbeResult has compact repr."""
        result = ProbeResult(
            success=True,
            tool_called=True,
            tool_name="search",
            parameters={"query": "auth"},
            raw_response={},
            latency_ms=250,
        )
        repr_str = repr(result)
        assert "PASS" in repr_str
        assert "search" in repr_str
        assert "250ms" in repr_str


class TestT0InvokeProbe:
    """Test T0 (basic tool invocation) probe."""

    def test_probe_attributes(self):
        """T0 probe has correct attributes."""
        probe = T0InvokeProbe()
        assert probe.level == 0
        assert probe.name == "Basic Tool Calling"
        assert "search" in probe.prompt.lower()
        assert len(probe.tools) == 1

    def test_str_representation(self):
        """T0 probe has readable string representation."""
        probe = T0InvokeProbe()
        assert "Level 0" in str(probe)
        assert "Basic Tool Calling" in str(probe)


class TestT1SchemaProbe:
    """Test T1 (schema compliance) probe."""

    def test_probe_attributes(self):
        """T1 probe has correct attributes."""
        probe = T1SchemaProbe()
        assert probe.level == 1


class TestT2SelectionProbe:
    """Test T2 (tool selection) probe."""

    def test_probe_attributes(self):
        """T2 probe has correct attributes."""
        probe = T2SelectionProbe()
        assert probe.level == 2


class TestMockedProbeRun:
    """Test probe run with mocked OpenAI client."""

    def test_t0_success_with_tool_call(self):
        """T0 correctly identifies successful tool call."""
        probe = T0InvokeProbe()

        # Create mock response
        mock_response = MagicMock()
        mock_response.model_dump.return_value = {"id": "test"}
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.tool_calls = [MagicMock()]
        mock_response.choices[0].message.tool_calls[0].function.name = "search"
        mock_response.choices[0].message.tool_calls[0].function.arguments = '{"query": "authentication"}'

        # Create mock client
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response

        result = probe.run("test-model", mock_client)

        assert result.success is True
        assert result.tool_called is True
        assert result.tool_name == "search"
        assert result.parameters == {"query": "authentication"}

    def test_t0_failure_no_tool_call(self):
        """T0 correctly identifies missing tool call."""
        probe = T0InvokeProbe()

        # Create mock response with no tool calls
        mock_response = MagicMock()
        mock_response.model_dump.return_value = {"id": "test"}
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.tool_calls = None
        mock_response.choices[0].message.content = "The weather is nice."

        # Create mock client
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response

        result = probe.run("test-model", mock_client)

        assert result.success is False
        assert result.tool_called is False
        assert result.tool_name is None

    def test_t0_handles_exception(self):
        """T0 gracefully handles API exceptions."""
        probe = T0InvokeProbe()

        # Create mock client that raises
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("API Error")

        result = probe.run("test-model", mock_client)

        assert result.success is False
        assert result.error == "API Error"


class TestIntegration:
    """Integration tests (require API key)."""

    @pytest.mark.skipif(
        not pytest.importorskip("os").getenv("OPENROUTER_API_KEY"),
        reason="OPENROUTER_API_KEY not set"
    )
    def test_t0_real_model(self):
        """Run T0 against a real model (requires API key)."""
        # This is an integration test - skip in CI without key
        pass
