"""CI reproduction verification for submitted results."""

import json
from pathlib import Path
from typing import Any


def verify_results(
    claimed_file: Path,
    tolerance: float = 0.15,
    min_agreement: float = 0.70,
) -> bool:
    """Verify claimed results by re-running probes.

    Args:
        claimed_file: Path to submitted results JSON
        tolerance: Maximum acceptable difference in success rates (default 15%)
        min_agreement: Minimum required agreement rate (default 70%)

    Returns:
        True if verification passed, False otherwise

    Note:
        This is a stub implementation. Full implementation requires:
        1. Loading claimed results
        2. Re-running probes for the claimed model/level
        3. Comparing success rates within tolerance
        4. Checking if agreement meets threshold
    """
    # Load claimed results
    with open(claimed_file) as f:
        claimed = json.load(f)

    # TODO: In full implementation:
    # 1. Extract metadata: model, level, claimed_rate, claimed_trials from claimed dict
    # 2. Import and use ProbeRunner to re-run probes
    # 3. Compare results statistically
    # 4. Check for agreement within tolerance
    #
    # For now, this is a placeholder that always returns True
    # Real implementation would look like:
    #
    # model = claimed["probes"]["model"]
    # level = claimed["probes"]["level"]
    # claimed_rate = claimed["summary"]["rate"]
    # claimed_trials = claimed["summary"]["trials"]
    #
    # from modelforecast.runner import ProbeRunner
    # runner = ProbeRunner(output_dir=Path("/tmp/verify"))
    # verification_results = runner.run_level(model, level, trials=claimed_trials)
    # verification_rate = verification_results["summary"]["rate"]
    #
    # rate_diff = abs(claimed_rate - verification_rate)
    # return rate_diff <= tolerance

    # Avoid unused variable warnings in stub
    _ = claimed
    return True


def calculate_agreement(
    results1: list[dict[str, Any]],
    results2: list[dict[str, Any]],
) -> float:
    """Calculate agreement rate between two sets of trial results.

    Args:
        results1: First set of trial results
        results2: Second set of trial results

    Returns:
        Agreement rate (0.0 to 1.0)

    Note:
        Agreement is measured by comparing success/failure outcomes.
        Both result sets must have the same number of trials.
    """
    if len(results1) != len(results2):
        raise ValueError("Result sets must have same number of trials")

    if not results1:
        return 1.0  # Perfect agreement on empty sets

    agreements = 0
    for r1, r2 in zip(results1, results2):
        # Compare success outcomes
        success1 = r1.get("tool_called") and r1.get("schema_valid")
        success2 = r2.get("tool_called") and r2.get("schema_valid")

        if success1 == success2:
            agreements += 1

    return agreements / len(results1)
