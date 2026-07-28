"""Tests for the curated model roster and tool-capability filtering."""

from unittest.mock import patch

import pytest

MOCK_MODELS = {
    "vendor/tool-model": {
        "id": "vendor/tool-model",
        "supported_parameters": ["tools", "temperature"],
    },
    "vendor/no-tool": {
        "id": "vendor/no-tool",
        "supported_parameters": ["temperature"],
    },
}


def test_default_roster_requires_live_tool_support(monkeypatch, tmp_path):
    """ProbeRunner validates every default roster entry before probing."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake-key")

    with (
        patch("modelforecast.models.DEFAULT_ROSTER", ("vendor/tool-model",)),
        patch("modelforecast.models.get_available_models", return_value=MOCK_MODELS),
        patch("openai.OpenAI"),
    ):
        from modelforecast.runner import ProbeRunner

        runner = ProbeRunner(output_dir=tmp_path, models=None)

    assert runner.models == ["vendor/tool-model"]


def test_stale_default_roster_fails_fast(monkeypatch, tmp_path):
    """A missing or non-tool roster entry aborts before a sweep starts."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake-key")

    with (
        patch(
            "modelforecast.models.DEFAULT_ROSTER",
            ("vendor/tool-model", "vendor/no-tool", "vendor/missing"),
        ),
        patch("modelforecast.models.get_available_models", return_value=MOCK_MODELS),
        patch("openai.OpenAI"),
        pytest.raises(ValueError, match="Default roster is stale"),
    ):
        from modelforecast.runner import ProbeRunner

        ProbeRunner(output_dir=tmp_path, models=None)


def test_explicit_models_validated_before_run(monkeypatch, tmp_path):
    """ProbeRunner with explicit models warns for invalid and keeps valid ones."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake-key")

    with (
        patch("modelforecast.models.get_available_models", return_value=MOCK_MODELS),
        patch("openai.OpenAI"),
    ):
        from modelforecast.runner import ProbeRunner

        runner = ProbeRunner(
            output_dir=tmp_path,
            models=["vendor/tool-model", "vendor/nonexistent"],
        )

    assert "vendor/tool-model" in runner.models
    assert "vendor/nonexistent" not in runner.models


def test_skip_validation_bypasses_check(monkeypatch, tmp_path):
    """ProbeRunner with skip_validation=True accepts any model IDs without validation."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake-key")

    with (
        patch("modelforecast.models.get_available_models", return_value=MOCK_MODELS),
        patch("openai.OpenAI"),
    ):
        from modelforecast.runner import ProbeRunner

        runner = ProbeRunner(
            output_dir=tmp_path,
            models=["fake/model"],
            skip_validation=True,
        )

    assert runner.models == ["fake/model"]


def test_get_models_tools_only_is_not_tier_filtered():
    """Tool discovery includes both zero-cost and paid endpoints."""
    models = {
        "vendor/paid": {"supported_parameters": ["tools"], "pricing": {"prompt": "0.1"}},
        "vendor/zero-cost:free": {
            "supported_parameters": ["tools"],
            "pricing": {"prompt": "0"},
        },
        "vendor/no-tools": {"supported_parameters": []},
    }

    with patch("modelforecast.models.get_available_models", return_value=models):
        from modelforecast.models import get_models

        assert get_models(tools_only=True) == ["vendor/paid", "vendor/zero-cost:free"]


def test_validate_roster_accepts_an_explicit_empty_roster():
    """An explicit empty cohort must not silently fall back to the default."""
    with patch("modelforecast.models.get_available_models", return_value=MOCK_MODELS):
        from modelforecast.models import validate_roster

        assert validate_roster([]) == {}
