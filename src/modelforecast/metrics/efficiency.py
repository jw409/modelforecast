"""Efficiency metrics for schema-on-read analysis of benchmark results.

These functions analyze the full request/response data stored per trial
to compute efficiency metrics that differentiate models beyond pass/fail.
"""

from dataclasses import dataclass
from statistics import mean, stdev
from typing import Any


@dataclass
class EfficiencyMetrics:
    """Computed efficiency metrics for a model's trials."""

    # Token metrics
    avg_completion_tokens: float
    avg_prompt_tokens: float
    token_efficiency: float  # 0-1, higher = fewer tokens for same task

    # Strictness metrics (no fluff)
    strictness_score: float  # 0-1, 1 = pure tool call, no text
    avg_content_length: float  # chars of non-tool response text

    # Latency metrics
    avg_latency_ms: float
    p95_latency_ms: float
    latency_jitter: float  # std/mean, lower = more predictable

    # Composite score (weighted)
    agent_efficiency_score: float  # AES - the tiebreaker


def compute_token_efficiency(trial: dict[str, Any]) -> dict[str, Any]:
    """Extract token usage from a trial's response data.

    Args:
        trial: Trial record with 'response' containing full API response

    Returns:
        Dict with prompt_tokens, completion_tokens, or None values if missing
    """
    response = trial.get("response", {})
    usage = response.get("usage", {})

    return {
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
    }


def compute_strictness(trial: dict[str, Any]) -> dict[str, Any]:
    """Analyze response strictness - did model add fluff text?

    A "strict" response has:
    - tool_calls populated
    - content empty or whitespace only

    A "chatty" response has:
    - tool_calls populated
    - content with "Sure! I'd be happy to..." type text

    Args:
        trial: Trial record with 'response' containing full API response

    Returns:
        Dict with strictness metrics
    """
    response = trial.get("response", {})
    choices = response.get("choices", [])

    if not choices:
        return {"strict": False, "content_length": 0, "has_tool_call": False}

    message = choices[0].get("message", {})
    content = message.get("content") or ""
    tool_calls = message.get("tool_calls") or []

    content_stripped = content.strip()
    content_length = len(content_stripped)
    has_tool_call = len(tool_calls) > 0

    # Strict = tool call made AND no substantive content
    strict = has_tool_call and content_length == 0

    return {
        "strict": strict,
        "content_length": content_length,
        "has_tool_call": has_tool_call,
        "content_preview": content_stripped[:100] if content_stripped else None,
    }


def compute_latency_stats(trials: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute latency statistics across trials.

    Args:
        trials: List of trial records with 'latency_ms'

    Returns:
        Dict with avg, p95, std, jitter metrics
    """
    latencies = [t.get("latency_ms", 0) for t in trials if t.get("latency_ms")]

    if not latencies:
        return {
            "avg_ms": 0,
            "p95_ms": 0,
            "std_ms": 0,
            "jitter": 0,
        }

    avg_ms = mean(latencies)
    std_ms = stdev(latencies) if len(latencies) > 1 else 0

    # P95 - 95th percentile
    sorted_latencies = sorted(latencies)
    p95_index = int(len(sorted_latencies) * 0.95)
    p95_ms = sorted_latencies[min(p95_index, len(sorted_latencies) - 1)]

    # Jitter = coefficient of variation (std/mean)
    jitter = std_ms / avg_ms if avg_ms > 0 else 0

    return {
        "avg_ms": avg_ms,
        "p95_ms": p95_ms,
        "std_ms": std_ms,
        "jitter": jitter,
    }


def compute_efficiency_metrics(trials: list[dict[str, Any]]) -> EfficiencyMetrics:
    """Compute all efficiency metrics for a set of trials.

    This is the main entry point for schema-on-read analysis.

    Args:
        trials: List of trial records with 'request', 'response', 'latency_ms'

    Returns:
        EfficiencyMetrics dataclass with all computed values
    """
    # Token metrics
    token_data = [compute_token_efficiency(t) for t in trials]
    completion_tokens = [
        t["completion_tokens"] for t in token_data if t["completion_tokens"]
    ]
    prompt_tokens = [t["prompt_tokens"] for t in token_data if t["prompt_tokens"]]

    avg_completion = mean(completion_tokens) if completion_tokens else 0
    avg_prompt = mean(prompt_tokens) if prompt_tokens else 0

    # Token efficiency: lower completion tokens = higher efficiency
    # Normalize: assume 20 tokens is "ideal" for a simple tool call
    IDEAL_COMPLETION_TOKENS = 20
    token_efficiency = min(1.0, IDEAL_COMPLETION_TOKENS / avg_completion) if avg_completion > 0 else 0

    # Strictness metrics
    strictness_data = [compute_strictness(t) for t in trials]
    strict_count = sum(1 for s in strictness_data if s["strict"])
    strictness_score = strict_count / len(trials) if trials else 0
    content_lengths = [s["content_length"] for s in strictness_data]
    avg_content_length = mean(content_lengths) if content_lengths else 0

    # Latency metrics
    latency_stats = compute_latency_stats(trials)

    # Latency score: faster = better, normalize to 0-1
    # Assume 1000ms is "bad", 500ms is "good"
    latency_score = max(0, 1 - (latency_stats["avg_ms"] / 2000))

    # Composite Agent Efficiency Score (AES)
    # Weights from Gemini's recommendation: 40% tokens, 40% latency, 20% strictness
    aes = (
        0.4 * token_efficiency +
        0.4 * latency_score +
        0.2 * strictness_score
    )

    return EfficiencyMetrics(
        avg_completion_tokens=avg_completion,
        avg_prompt_tokens=avg_prompt,
        token_efficiency=token_efficiency,
        strictness_score=strictness_score,
        avg_content_length=avg_content_length,
        avg_latency_ms=latency_stats["avg_ms"],
        p95_latency_ms=latency_stats["p95_ms"],
        latency_jitter=latency_stats["jitter"],
        agent_efficiency_score=aes,
    )


def analyze_result_file(result: dict[str, Any]) -> dict[str, Any]:
    """Analyze a complete result file and add efficiency metrics.

    This reads the stored trial data and computes efficiency metrics,
    adding them to the result without modifying the original structure.

    Args:
        result: Complete result dict from JSON file

    Returns:
        Result dict with added 'efficiency' section
    """
    trials = result.get("probes", {}).get("trials", [])

    # Check if trials have response data (new format)
    has_response_data = any(t.get("response") for t in trials)

    if not has_response_data:
        # Old format - can only compute latency stats
        latency_stats = compute_latency_stats(trials)
        return {
            **result,
            "efficiency": {
                "has_full_data": False,
                "latency": latency_stats,
            },
        }

    # New format - compute full efficiency metrics
    metrics = compute_efficiency_metrics(trials)

    return {
        **result,
        "efficiency": {
            "has_full_data": True,
            "token_efficiency": metrics.token_efficiency,
            "avg_completion_tokens": metrics.avg_completion_tokens,
            "strictness_score": metrics.strictness_score,
            "avg_content_length": metrics.avg_content_length,
            "latency": {
                "avg_ms": metrics.avg_latency_ms,
                "p95_ms": metrics.p95_latency_ms,
                "jitter": metrics.latency_jitter,
            },
            "agent_efficiency_score": metrics.agent_efficiency_score,
        },
    }
