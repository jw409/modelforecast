#!/usr/bin/env python3
"""Generate tweet-ready text from latest results."""

from pathlib import Path
import json

def main():
    results_dir = Path(__file__).parent.parent / "results"

    # Count results
    l0_files = list(results_dir.glob("*_level_0.json"))
    total_models = len(l0_files)

    # Find 100% models
    winners = []
    for f in l0_files:
        data = json.loads(f.read_text())
        trials = data.get("probes", {}).get("trials", [])
        if trials and all(t.get("tool_called") for t in trials):
            model = data["probes"]["model"].split("/")[-1].replace(":free", "")
            winners.append(model)

    tweet = f"""🔬 Free LLM Tool-Calling: Only {len(winners)}/{total_models} work

✅ {', '.join(winners[:2])}
❌ 13 models claim tools but fail 100%

The 3-Trial Trap: 8 passed quick tests, failed extended.

github.com/jw409/modelforecast

#LLM #OpenRouter"""

    print("=" * 50)
    print("TWEET ({} chars):".format(len(tweet)))
    print("=" * 50)
    print(tweet)
    print("=" * 50)

if __name__ == "__main__":
    main()
