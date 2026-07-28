"""Tests for explicit paid-sweep confirmation."""

import sys
from unittest.mock import patch

from modelforecast.__main__ import main


def test_sweep_requires_spend_confirmation(monkeypatch, capsys):
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake-key")
    monkeypatch.setattr(sys, "argv", ["modelforecast", "sweep", "--trials", "1"])

    with (
        patch("modelforecast.__main__.ProbeRunner") as runner,
        patch("modelforecast.__main__.SweepOrchestrator") as orchestrator,
    ):
        result = main()

    assert result == 2
    assert "may use paid OpenRouter endpoints" in capsys.readouterr().out
    runner.assert_not_called()
    orchestrator.assert_not_called()


def test_roster_validation_does_not_require_spend_confirmation(monkeypatch, capsys):
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake-key")
    monkeypatch.setattr(
        sys,
        "argv",
        ["modelforecast", "sweep", "--validate-roster"],
    )

    with patch(
        "modelforecast.__main__.validate_roster",
        return_value={"vendor/model": (True, "live with tool support")},
    ):
        result = main()

    assert result == 0
    assert "1/1 roster models ready" in capsys.readouterr().out
