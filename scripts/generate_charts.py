#!/usr/bin/env python3
"""Generate charts for ModelForecast README.

Charts generated:
1. Reliability vs Latency scatter (which free models are fast AND reliable?)
2. Success rate bar chart with confidence intervals
3. The "graveyard" - models that claim tool support but fail
"""

import json
import os
from pathlib import Path
from statistics import mean, stdev

# Try matplotlib, fall back to ASCII if not available
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("matplotlib not installed - generating ASCII charts")


RESULTS_DIR = Path(__file__).parent.parent / "results"
CHARTS_DIR = Path(__file__).parent.parent / "charts"

# Import efficiency metrics
try:
    from modelforecast.metrics import compute_efficiency_metrics
    HAS_METRICS = True
except ImportError:
    HAS_METRICS = False


def load_all_results():
    """Load all JSON result files."""
    results = {}
    for f in RESULTS_DIR.glob("*.json"):
        if f.name.startswith(".") or "comparison" in f.name:
            continue
        try:
            data = json.loads(f.read_text())
            model = data.get("probes", {}).get("model", f.stem)
            level = data.get("probes", {}).get("level", 0)

            # Extract latencies
            trials = data.get("probes", {}).get("trials", [])
            latencies = [t.get("latency_ms", 0) for t in trials if t.get("latency_ms")]

            key = f"{model}__L{level}"
            results[key] = {
                "model": model,
                "level": level,
                "rate": data.get("summary", {}).get("rate", 0),
                "trials": data.get("summary", {}).get("trials", 0),
                "successes": data.get("summary", {}).get("successes", 0),
                "ci_low": data.get("summary", {}).get("wilson_ci_95", [0, 1])[0],
                "ci_high": data.get("summary", {}).get("wilson_ci_95", [0, 1])[1],
                "latencies": latencies,
                "avg_latency_ms": mean(latencies) if latencies else 0,
                "latency_std": stdev(latencies) if len(latencies) > 1 else 0,
            }
        except Exception as e:
            print(f"Error loading {f}: {e}")
    return results


def generate_reliability_latency_scatter(results, output_path):
    """Scatter plot: X=avg latency, Y=success rate, size=trials."""
    if not HAS_MPL:
        print("\n=== RELIABILITY vs LATENCY (ASCII) ===")
        print("Model                          | Rate | Latency(ms)")
        print("-" * 55)
        for key, data in sorted(results.items(), key=lambda x: -x[1]["rate"]):
            if data["level"] == 0:  # Only T0 for simplicity
                short_name = data["model"].split("/")[-1][:25]
                print(f"{short_name:30} | {data['rate']*100:4.0f}% | {data['avg_latency_ms']:7.0f}")
        return

    # Filter to T0 only for main chart
    l0_data = {k: v for k, v in results.items() if v["level"] == 0 and v["avg_latency_ms"] > 0}

    fig, ax = plt.subplots(figsize=(12, 8))

    # Color by verdict
    colors = []
    for k, v in l0_data.items():
        if v["rate"] >= 0.9:
            colors.append("#2ecc71")  # Green - production ready
        elif v["rate"] >= 0.5:
            colors.append("#f39c12")  # Orange - unreliable
        else:
            colors.append("#e74c3c")  # Red - broken

    x = [v["avg_latency_ms"] / 1000 for v in l0_data.values()]  # Convert to seconds
    y = [v["rate"] * 100 for v in l0_data.values()]
    sizes = [v["trials"] * 20 for v in l0_data.values()]

    scatter = ax.scatter(x, y, c=colors, s=sizes, alpha=0.7, edgecolors='black', linewidth=0.5)

    # Add labels for notable models
    for key, data in l0_data.items():
        if data["rate"] >= 0.9 or data["rate"] <= 0.1:
            short_name = data["model"].split("/")[-1].replace(":free", "")
            ax.annotate(short_name,
                       (data["avg_latency_ms"]/1000, data["rate"]*100),
                       textcoords="offset points", xytext=(5, 5), fontsize=8)

    ax.set_xlabel("Average Latency (seconds)", fontsize=12)
    ax.set_ylabel("Success Rate (%)", fontsize=12)
    ax.set_title("Free Model Tool-Calling: Reliability vs Speed\n(T0 Basic Tool Invocation)", fontsize=14)

    # Add quadrant lines
    ax.axhline(y=90, color='gray', linestyle='--', alpha=0.5, label='90% threshold')
    ax.axvline(x=3, color='gray', linestyle='--', alpha=0.5, label='3s threshold')

    # Legend
    green_patch = mpatches.Patch(color='#2ecc71', label='Production Ready (≥90%)')
    orange_patch = mpatches.Patch(color='#f39c12', label='Unreliable (50-89%)')
    red_patch = mpatches.Patch(color='#e74c3c', label='Broken (<50%)')
    ax.legend(handles=[green_patch, orange_patch, red_patch], loc='lower right')

    ax.set_ylim(-5, 105)
    ax.set_xlim(0, max(x) * 1.1 if x else 10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def generate_success_bar_chart(results, output_path):
    """Bar chart with CI error bars."""
    if not HAS_MPL:
        print("\n=== SUCCESS RATES WITH CI (ASCII) ===")
        return

    # Filter to T0
    l0_data = [(k, v) for k, v in results.items() if v["level"] == 0]
    l0_data.sort(key=lambda x: -x[1]["rate"])

    fig, ax = plt.subplots(figsize=(14, 8))

    names = [d[1]["model"].split("/")[-1].replace(":free", "") for d in l0_data]
    rates = [d[1]["rate"] * 100 for d in l0_data]
    ci_low = [(d[1]["rate"] - d[1]["ci_low"]) * 100 for d in l0_data]
    ci_high = [(d[1]["ci_high"] - d[1]["rate"]) * 100 for d in l0_data]

    colors = ['#2ecc71' if r >= 90 else '#f39c12' if r >= 50 else '#e74c3c' for r in rates]

    bars = ax.bar(range(len(names)), rates, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax.errorbar(range(len(names)), rates, yerr=[ci_low, ci_high],
                fmt='none', color='black', capsize=3, alpha=0.7)

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel("Success Rate (%)", fontsize=12)
    ax.set_title("Free Model Tool-Calling Success Rates (T0)\nwith 95% Wilson Confidence Intervals", fontsize=14)
    ax.set_ylim(0, 110)

    # Add 90% line
    ax.axhline(y=90, color='gray', linestyle='--', alpha=0.5, linewidth=2)
    ax.text(len(names)-1, 92, "Production threshold", ha='right', fontsize=9, color='gray')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def generate_multi_level_comparison(results, output_path):
    """Compare models across T0-T2/A1/R0 capability dimensions."""
    if not HAS_MPL:
        return

    # Find models with multiple levels
    models_with_levels = {}
    for key, data in results.items():
        model = data["model"]
        level = data["level"]
        if model not in models_with_levels:
            models_with_levels[model] = {}
        models_with_levels[model][level] = data["rate"] * 100

    # Filter to models with at least 3 levels
    multi_level = {k: v for k, v in models_with_levels.items() if len(v) >= 3}

    if not multi_level:
        print("No models with multi-level data yet")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    levels = [0, 1, 2, 3, 4]
    level_names = ["T0\nInvoke", "T1\nSchema", "T2\nSelection", "A1\nLinear", "R0\nAbstain"]

    x = range(len(levels))
    width = 0.8 / len(multi_level)

    colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6', '#f39c12']

    for i, (model, level_data) in enumerate(multi_level.items()):
        rates = [level_data.get(l, 0) for l in levels]
        short_name = model.split("/")[-1].replace(":free", "")
        offset = (i - len(multi_level)/2 + 0.5) * width
        ax.bar([xi + offset for xi in x], rates, width, label=short_name,
               color=colors[i % len(colors)], alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(level_names)
    ax.set_ylabel("Success Rate (%)")
    ax.set_title("Capability Dimension Comparison\n(A1 Agency is the differentiator)")
    ax.legend(loc='upper right')
    ax.set_ylim(0, 110)
    ax.axhline(y=90, color='gray', linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def generate_ascii_summary():
    """Generate ASCII art for README (no image dependencies)."""
    results = load_all_results()

    # Quick summary
    l0_results = {k: v for k, v in results.items() if v["level"] == 0}
    production_ready = [k for k, v in l0_results.items() if v["rate"] >= 0.9]
    unreliable = [k for k, v in l0_results.items() if 0.5 <= v["rate"] < 0.9]
    broken = [k for k, v in l0_results.items() if v["rate"] < 0.5]

    print("\n" + "="*60)
    print("MODELFORECAST SUMMARY")
    print("="*60)
    print(f"\nProduction Ready (≥90%): {len(production_ready)}")
    for k in production_ready:
        m = l0_results[k]
        print(f"  ✅ {m['model'].split('/')[-1]:40} {m['rate']*100:5.1f}% ({m['avg_latency_ms']/1000:.1f}s)")

    print(f"\nUnreliable (50-89%): {len(unreliable)}")
    for k in unreliable:
        m = l0_results[k]
        print(f"  ⚠️  {m['model'].split('/')[-1]:40} {m['rate']*100:5.1f}%")

    print(f"\nBroken (<50%): {len(broken)}")
    for k in broken:
        m = l0_results[k]
        print(f"  ❌ {m['model'].split('/')[-1]:40} {m['rate']*100:5.1f}%")

    return f"""
## Quick Stats (auto-generated)

| Category | Count |
|----------|-------|
| Production Ready (≥90%) | {len(production_ready)} |
| Unreliable (50-89%) | {len(unreliable)} |
| Broken (<50%) | {len(broken)} |
| **Total Tested** | **{len(l0_results)}** |
"""


def main():
    CHARTS_DIR.mkdir(exist_ok=True)

    print("Loading results...")
    results = load_all_results()
    print(f"Loaded {len(results)} result files")

    print("\nGenerating charts...")

    # Generate all charts
    generate_reliability_latency_scatter(
        results,
        CHARTS_DIR / "reliability_vs_latency.png"
    )

    generate_success_bar_chart(
        results,
        CHARTS_DIR / "success_rates_with_ci.png"
    )

    generate_multi_level_comparison(
        results,
        CHARTS_DIR / "multi_level_comparison.png"
    )

    # Always generate ASCII summary
    summary = generate_ascii_summary()

    print("\n" + "="*60)
    print("Charts saved to:", CHARTS_DIR)
    print("="*60)

    return summary


if __name__ == "__main__":
    main()
