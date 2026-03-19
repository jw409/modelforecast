"""Per-model token bucket rate limiter with jitter."""

import random
import time


class RateLimiter:
    """Per-model token bucket with jitter.

    Enforces a minimum interval between API calls to the same model.
    Uses jitter to avoid thundering-herd on rate limit resets.

    Args:
        calls_per_minute: Target call rate (default 8 — conservative for 20rpm free tier)
        jitter_range: Max random seconds added after wait (default 0.5)
    """

    def __init__(self, calls_per_minute: int = 8, jitter_range: float = 0.5) -> None:
        self._last_call: dict[str, float] = {}
        self._min_interval = 60.0 / calls_per_minute
        self._jitter_range = jitter_range

    def acquire(self, model: str) -> None:
        """Block until the rate limit interval has passed for this model.

        Args:
            model: Model identifier used as per-model bucket key
        """
        now = time.time()
        last = self._last_call.get(model, 0.0)
        wait = self._min_interval - (now - last)
        if wait > 0:
            jitter = random.uniform(0, self._jitter_range)
            time.sleep(wait + jitter)
        self._last_call[model] = time.time()
