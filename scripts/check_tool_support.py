#!/usr/bin/env python3
"""Check which free models support tool calling on OpenRouter.

Usage:
    uv run python scripts/check_tool_support.py

Output:
    Lists all free models with their tool support status.
"""

import os
import sys

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from modelforecast.models import get_tool_support_matrix


def main():
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: Set OPENROUTER_API_KEY environment variable")
        sys.exit(1)

    print("Fetching model capabilities from OpenRouter...\n")

    matrix = get_tool_support_matrix(api_key)

    supports = [m for m, v in matrix.items() if v]
    not_supports = [m for m, v in matrix.items() if not v]

    print(f"=== SUPPORTS TOOLS ({len(supports)}) ===")
    for model in sorted(supports):
        print(f"  ✓ {model}")

    print(f"\n=== DOES NOT SUPPORT TOOLS ({len(not_supports)}) ===")
    for model in sorted(not_supports):
        print(f"  ✗ {model}")

    print(f"\n--- Summary ---")
    print(f"Total free models: {len(matrix)}")
    print(f"Support tools: {len(supports)} ({100*len(supports)/len(matrix):.0f}%)")
    print(f"No tool support: {len(not_supports)} ({100*len(not_supports)/len(matrix):.0f}%)")


if __name__ == "__main__":
    main()
