#!/usr/bin/env python3
"""Generate tweet-ready text with compelling hooks."""

from pathlib import Path
import json
import sys

HOOKS = {
    "trap": """The 3-Trial Trap:

8 LLMs passed 100% on 3 trials.
Then failed on 10.

Only 2 of 25 "free tool-calling" models actually work.

github.com/jw409/modelforecast

#LLM #AI #OpenRouter""",

    "lie": """"Supports tool calling" is a lie.

We tested 25 free models.
14 failed 100% of trials.
8 passed quick tests, failed extended.

Only 2 actually work.

github.com/jw409/modelforecast""",

    "paid": """Free model matches $15/M Claude.

KAT Coder Pro: 100% tool-calling, 1.3s latency, $0
Claude Sonnet: 100% tool-calling, 1.8s latency, $15/M

23 other "free" models? Broken.

github.com/jw409/modelforecast""",

    "multi": """Why your AI agent loop breaks:

Grok 4.1: 100% single-shot, 0% multi-turn
Most models: Can't continue after tool results

The free tier that ACTUALLY works for agents:
→ KAT Coder Pro (100% L0-L4)

github.com/jw409/modelforecast""",

    "stats": """Free LLM Tool-Calling Reality Check:

✅ 2 models work (100%)
⚠️ 9 unreliable (20-60%)
❌ 14 completely broken (0%)

Don't waste compute. Check the forecast.

github.com/jw409/modelforecast""",
}


def main():
    if len(sys.argv) > 1 and sys.argv[1] in HOOKS:
        hook = sys.argv[1]
    else:
        hook = "trap"  # Default to the proven winner

    tweet = HOOKS[hook]

    print("=" * 50)
    print(f"HOOK: {hook.upper()} ({len(tweet)} chars)")
    print("=" * 50)
    print(tweet)
    print("=" * 50)

    if len(sys.argv) == 1:
        print("\nAvailable hooks: " + ", ".join(HOOKS.keys()))
        print("Usage: uv run python scripts/generate_tweet.py [hook]")


if __name__ == "__main__":
    main()
