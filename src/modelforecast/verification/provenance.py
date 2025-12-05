"""Cryptographic provenance tracking for benchmark results."""

import hashlib
import json
import platform
import secrets
import sys
from datetime import datetime, timezone
from typing import Any


def generate_submission_id() -> str:
    """Generate a unique submission ID with prefix."""
    random_suffix = secrets.token_hex(6)
    return f"sub_{random_suffix}"


def hash_content(content: str) -> str:
    """Generate SHA256 hash of content."""
    return f"sha256:{hashlib.sha256(content.encode()).hexdigest()}"


class ProvenanceTracker:
    """Tracks cryptographic provenance of benchmark results."""

    def __init__(self, contributor: str | None = None):
        """Initialize provenance tracker.

        Args:
            contributor: GitHub username of contributor (defaults to "unknown")
        """
        self.contributor = contributor or "unknown"
        self.submission_id = generate_submission_id()
        self.timestamp = datetime.now(timezone.utc).isoformat()
        self._environment = self._capture_environment()

    def _capture_environment(self) -> dict[str, Any]:
        """Capture environment metadata."""
        import openai

        env_data = {
            "python_version": sys.version.split()[0],
            "openai_sdk_version": openai.__version__,
            "os": f"{platform.system()} {platform.release()}",
        }

        # Hash environment for comparison
        env_string = json.dumps(env_data, sort_keys=True)
        env_data["env_hash"] = hash_content(env_string)

        return env_data

    def create_trial_record(
        self,
        prompt: str,
        response: str,
        tool_called: bool,
        schema_valid: bool,
        latency_ms: int,
        openrouter_request_id: str | None = None,
        *,
        # New: Full request/response for schema-on-read analysis
        request_data: dict[str, Any] | None = None,
        response_data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create a trial record with provenance data.

        Args:
            prompt: The prompt sent to the model
            response: The model's response (string repr for hashing)
            tool_called: Whether a tool call was made
            schema_valid: Whether the tool call schema is valid
            latency_ms: Response latency in milliseconds
            openrouter_request_id: OpenRouter request ID from headers
            request_data: Full request payload (model, messages, tools, settings)
            response_data: Full API response (usage, choices, etc.)

        Returns:
            Trial record dictionary with full data for schema-on-read analysis
        """
        record = {
            "openrouter_request_id": openrouter_request_id,
            "prompt_hash": hash_content(prompt),
            "response_hash": hash_content(response),
            "tool_called": tool_called,
            "schema_valid": schema_valid,
            "latency_ms": latency_ms,
        }

        # Store full data for schema-on-read efficiency analysis
        if request_data:
            record["request"] = request_data
        if response_data:
            record["response"] = response_data

        return record

    def create_result(
        self,
        model: str,
        level: int,
        trials: list[dict[str, Any]],
        successes: int,
        wilson_ci: tuple[float, float],
    ) -> dict[str, Any]:
        """Create a complete result with provenance.

        Args:
            model: Model identifier
            level: Probe level (0-4)
            trials: List of trial records
            successes: Number of successful trials
            wilson_ci: Wilson confidence interval (lower, upper)

        Returns:
            Complete result dictionary with provenance
        """
        return {
            "submission_id": self.submission_id,
            "timestamp": self.timestamp,
            "contributor": self.contributor,
            "environment": self._environment,
            "probes": {
                "model": model,
                "level": level,
                "trials": trials,
            },
            "summary": {
                "successes": successes,
                "trials": len(trials),
                "rate": successes / len(trials) if trials else 0.0,
                "wilson_ci_95": list(wilson_ci),
            },
        }
