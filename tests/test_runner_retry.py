"""Tests verifying ProbeRunner uses SDK built-in max_retries (not a manual loop)."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import openai
import pytest


@pytest.fixture
def api_key(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")


# ---------------------------------------------------------------------------
# Test 1: OpenAI constructor receives max_retries from ProbeRunner.__init__
# ---------------------------------------------------------------------------


def test_openai_constructor_receives_max_retries(tmp_path, api_key):
    """ProbeRunner must forward max_retries to OpenAI(max_retries=...)."""
    with patch("modelforecast.runner.OpenAI") as mock_openai:
        mock_openai.return_value = MagicMock()
        from modelforecast.runner import ProbeRunner

        ProbeRunner(
            output_dir=tmp_path,
            models=["test-model/x"],
            skip_validation=True,
            max_retries=3,
        )

    # Verify OpenAI was called with max_retries=3
    call_kwargs = mock_openai.call_args.kwargs
    assert "max_retries" in call_kwargs, (
        f"OpenAI() was not called with max_retries. kwargs were: {call_kwargs}"
    )
    assert call_kwargs["max_retries"] == 3, (
        f"Expected max_retries=3, got {call_kwargs['max_retries']}"
    )


# ---------------------------------------------------------------------------
# Test 2: Manual retry loop is gone from runner.py source
# ---------------------------------------------------------------------------


def test_no_manual_retry_loop_in_source():
    """runner.py must not contain 'for attempt in range' (manual retry loop)."""
    runner_path = Path(__file__).parent.parent / "src" / "modelforecast" / "runner.py"
    source = runner_path.read_text()

    assert "for attempt in range" not in source, (
        "Manual retry loop ('for attempt in range') still present in runner.py — "
        "should be removed in favour of SDK max_retries"
    )


# ---------------------------------------------------------------------------
# Test 3: run_level handles probe.run() raising RateLimitError without crash
# ---------------------------------------------------------------------------


def test_run_level_handles_rate_limit_error(tmp_path, monkeypatch):
    """run_level should return a failed ProbeResult when RateLimitError is raised."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    with patch("modelforecast.runner.OpenAI") as mock_openai:
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        # Stub rate limiter so we don't sleep
        with patch("modelforecast.runner.RateLimiter") as mock_rl:
            mock_rl.return_value.acquire = MagicMock()

            from modelforecast.runner import ProbeRunner

            runner = ProbeRunner(
                output_dir=tmp_path,
                models=["test-model/x"],
                skip_validation=True,
                max_retries=3,
            )

    # Patch probe to raise RateLimitError on .run()
    mock_probe = MagicMock()
    mock_probe.prompt = "test prompt"
    mock_probe.tools = []
    mock_probe.__str__ = lambda _: "MockProbe"

    # Build a minimal RateLimitError
    rate_err = openai.RateLimitError(
        message="rate limited",
        response=MagicMock(status_code=429, headers={}),
        body={"error": {"message": "rate limited"}},
    )
    mock_probe.run.side_effect = rate_err

    runner.probes = {0: mock_probe}

    # Also stub provenance tracker to avoid needing real files
    with patch("modelforecast.runner.ProvenanceTracker") as mock_tracker:
        mock_tracker_instance = MagicMock()
        mock_tracker.return_value = mock_tracker_instance
        mock_tracker_instance.create_trial_record.return_value = {}
        mock_tracker_instance.create_result.return_value = {
            "summary": {"rate": 0.0},
        }

        with patch("modelforecast.runner.write_individual_result"):
            result = runner.run_level("test-model/x", level=0, trials=1)

    # Result should be a dict (not an exception)
    assert result is not None
    assert isinstance(result, dict)
