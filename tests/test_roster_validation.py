"""Tests for model roster validation and tools_only filtering."""

import pytest
from unittest.mock import patch, MagicMock


MOCK_MODELS = {
    "vendor/tool-model:free": {
        "id": "vendor/tool-model:free",
        "supported_parameters": ["tools", "temperature"],
    },
    "vendor/no-tool:free": {
        "id": "vendor/no-tool:free",
        "supported_parameters": ["temperature"],
    },
}


def test_auto_discovery_uses_tools_only(monkeypatch, tmp_path):
    """ProbeRunner with models=None auto-discovers only tool-capable free models."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake-key")

    with patch("modelforecast.models.get_available_models", return_value=MOCK_MODELS), \
         patch("openai.OpenAI"):
        from modelforecast.runner import ProbeRunner
        runner = ProbeRunner(output_dir=tmp_path, models=None)

    # Only the tool-capable model should be included
    assert "vendor/tool-model:free" in runner.models
    assert "vendor/no-tool:free" not in runner.models


def test_explicit_models_validated_before_run(monkeypatch, tmp_path):
    """ProbeRunner with explicit models warns for invalid and keeps valid ones."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake-key")

    with patch("modelforecast.models.get_available_models", return_value=MOCK_MODELS), \
         patch("openai.OpenAI"):
        from modelforecast.runner import ProbeRunner
        # Pass both a valid model and a model not in MOCK_MODELS
        runner = ProbeRunner(
            output_dir=tmp_path,
            models=["vendor/tool-model:free", "vendor/nonexistent:free"],
        )

    # Valid model kept, nonexistent model filtered out with a warning
    assert "vendor/tool-model:free" in runner.models
    assert "vendor/nonexistent:free" not in runner.models


def test_skip_validation_bypasses_check(monkeypatch, tmp_path):
    """ProbeRunner with skip_validation=True accepts any model IDs without validation."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake-key")

    with patch("modelforecast.models.get_available_models", return_value=MOCK_MODELS), \
         patch("openai.OpenAI"):
        from modelforecast.runner import ProbeRunner
        runner = ProbeRunner(
            output_dir=tmp_path,
            models=["fake:free"],
            skip_validation=True,
        )

    assert runner.models == ["fake:free"]
