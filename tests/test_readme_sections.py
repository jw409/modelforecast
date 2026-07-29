"""Tests for generated README benchmark summaries."""

from modelforecast.output.readme_sections import build_quick_answer_section


def test_quick_answer_is_dated_and_not_a_deployment_recommendation():
    summary = {
        "vendor/model": {
            "grade": "A",
            "level_results": {0: {"summary": {"rate": 1.0}}},
            "failure_modes": [],
        }
    }
    metadata = {
        "sweep_date": "2026-07-28",
        "model_count": 9,
        "trials_per_level": 10,
    }

    section = build_quick_answer_section(summary, metadata)

    assert "Best model in the 2026-07-28 sweep" in section
    assert "from a 9-model cohort" in section
    assert "Validate current availability and rerun before deployment" in section
    assert "> Use " not in section
