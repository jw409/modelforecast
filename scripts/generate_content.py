#!/usr/bin/env python3
"""
Generate content from benchmark analysis results.

Usage:
    uv run python scripts/generate_content.py --all
    uv run python scripts/generate_content.py --readme
    uv run python scripts/generate_content.py --headline
    uv run python scripts/generate_content.py --article agency-gap
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any


def load_analysis_data() -> Dict[str, Any]:
    """Load all analysis output files."""
    analysis_dir = Path("analysis")

    data = {}

    # Load metrics
    metrics_file = analysis_dir / "metrics.json"
    if metrics_file.exists():
        with open(metrics_file) as f:
            data["metrics"] = json.load(f)

    # Load rankings
    rankings_file = analysis_dir / "rankings.json"
    if rankings_file.exists():
        with open(rankings_file) as f:
            data["rankings"] = json.load(f)

    # Load insights
    insights_file = analysis_dir / "insights.json"
    if insights_file.exists():
        with open(insights_file) as f:
            data["insights"] = json.load(f)

    return data


def generate_readme_section(data: Dict[str, Any]) -> str:
    """Generate README update section."""
    today = datetime.now().strftime("%Y-%m-%d")

    metrics = data.get("metrics", {})
    insights = data.get("insights", {})

    # Get counts
    total_tested = metrics.get("total_models", 0)
    production_ready = len([m for m in metrics.get("model_scores", {}).values()
                           if m.get("overall_score", 0) >= 0.90])
    marginal = len([m for m in metrics.get("model_scores", {}).values()
                    if 0.50 <= m.get("overall_score", 0) < 0.90])
    broken = len([m for m in metrics.get("model_scores", {}).values()
                  if m.get("overall_score", 0) < 0.50])

    # Get agency passing count
    agency_passing = len([m for m in metrics.get("model_scores", {}).values()
                         if m.get("agency_score", 0) >= 0.90])

    section = f"""## Today's Forecast ({today})

**{production_ready} of {total_tested} free models are production-ready. Only {agency_passing} handles multi-turn.**

| Category | Count |
|----------|------:|
| Production Ready (≥90%) | {production_ready} |
| Marginal (50-89%) | {marginal} |
| Broken (<50%) | {broken} |

### Top 5 Free Models

"""

    # Add top 5 models
    rankings = data.get("rankings", {})
    top_models = rankings.get("overall_top_10", [])[:5]

    if top_models:
        section += "| Rank | Model | T | R | A | Overall |\n"
        section += "|------|-------|---|---|---|--------:|\n"

        for i, model_data in enumerate(top_models, 1):
            model = model_data.get("model", "unknown")
            scores = metrics.get("model_scores", {}).get(model, {})
            t_score = int(scores.get("tool_calling_score", 0) * 100)
            r_score = int(scores.get("reasoning_score", 0) * 100)
            a_score = int(scores.get("agency_score", 0) * 100)
            overall = int(scores.get("overall_score", 0) * 100)

            section += f"| {i} | {model} | {t_score}% | {r_score}% | {a_score}% | {overall}% |\n"

    section += "\n**T** = Tool Calling, **R** = Reasoning, **A** = Agency (multi-turn)\n"

    return section


def generate_headline(data: Dict[str, Any]) -> str:
    """Generate HEADLINE.md content."""
    today = datetime.now().strftime("%Y-%m-%d")

    metrics = data.get("metrics", {})
    rankings = data.get("rankings", {})
    insights = data.get("insights", {})

    total_tested = metrics.get("total_models", 0)
    production_ready = len([m for m in metrics.get("model_scores", {}).values()
                           if m.get("overall_score", 0) >= 0.90])
    agency_passing = len([m for m in metrics.get("model_scores", {}).values()
                         if m.get("agency_score", 0) >= 0.90])

    # Get top model
    top_models = rankings.get("overall_top_10", [])
    top_model = top_models[0].get("model", "unknown") if top_models else "unknown"
    top_score = int(top_models[0].get("overall_score", 0) * 100) if top_models else 0

    # Get agency gap insights
    agency_failures = insights.get("agency_gap", {}).get("models_failing_agency", [])

    headline = f"""# Model Forecast: {today}

## Executive Summary

**{production_ready} of {total_tested} free models tested are production-ready for tool calling.**

**Critical Finding**: Only {agency_passing} models handle multi-turn conversations (Agency dimension). {len(agency_failures)} models pass basic tool calling but fail when multiple turns are required.

## The Three Dimensions

- **T** (Tool Calling): Can the model invoke functions correctly?
- **R** (Reasoning): Does it choose the right tools for the task?
- **A** (Agency): Can it handle multi-turn conversations?

## Top Performer

**{top_model}** leads with {top_score}% overall score.

"""

    # Add comparison table
    headline += "## Production-Ready Models (≥90% Overall)\n\n"
    headline += "| Model | T | R | A | Overall |\n"
    headline += "|-------|---|---|---|--------:|\n"

    production_models = [m for m in top_models if m.get("overall_score", 0) >= 0.90]
    for model_data in production_models:
        model = model_data.get("model", "unknown")
        scores = metrics.get("model_scores", {}).get(model, {})
        t_score = int(scores.get("tool_calling_score", 0) * 100)
        r_score = int(scores.get("reasoning_score", 0) * 100)
        a_score = int(scores.get("agency_score", 0) * 100)
        overall = int(scores.get("overall_score", 0) * 100)

        headline += f"| {model} | {t_score}% | {r_score}% | {a_score}% | {overall}% |\n"

    # Add key findings
    headline += "\n## Key Findings\n\n"

    if agency_failures:
        headline += f"### The Agency Gap\n\n"
        headline += f"{len(agency_failures)} models pass T0-T2 tests but fail A1 (multi-turn). "
        headline += "This is the #1 reason models fail in production.\n\n"
        headline += "Models with Agency Gap:\n"
        for model in agency_failures[:5]:
            headline += f"- {model}\n"
        if len(agency_failures) > 5:
            headline += f"- ...and {len(agency_failures) - 5} more\n"

    # Add hidden champions
    hidden_champions = rankings.get("hidden_champions", [])
    if hidden_champions:
        headline += f"\n### Hidden Champions\n\n"
        headline += "Lesser-known models outperforming famous ones:\n\n"
        for champ in hidden_champions[:3]:
            model = champ.get("model", "unknown")
            outperforms = champ.get("outperforms", [])
            if outperforms:
                headline += f"- **{model}** beats {', '.join(outperforms[:2])}\n"

    # Add graveyard
    graveyard = insights.get("graveyard", {}).get("broken_models", [])
    if graveyard:
        headline += f"\n### Tool Calling Graveyard\n\n"
        headline += f"{len(graveyard)} models claim tool calling support but score <50%:\n\n"
        for model in graveyard[:5]:
            headline += f"- {model}\n"
        if len(graveyard) > 5:
            headline += f"- ...and {len(graveyard) - 5} more\n"

    headline += "\n---\n\n"
    headline += f"Last updated: {today}\n"
    headline += "Full methodology: See README.md\n"

    return headline


def generate_article_agency_gap(data: Dict[str, Any]) -> str:
    """Generate agency gap article."""
    today = datetime.now().strftime("%Y-%m-%d")

    insights = data.get("insights", {})
    metrics = data.get("metrics", {})

    agency_gap = insights.get("agency_gap", {})
    failing_models = agency_gap.get("models_failing_agency", [])

    article = f"""# The Agency Gap: Why {len(failing_models)} Models Fail Multi-Turn

*{today}*

## The Problem

{len(failing_models)} free models pass basic tool calling tests (T0-T2) but completely fail when you need multi-turn conversations (A1). This is the **#1 reason models fail in production**.

## What is Agency?

The **A** dimension (Agency) tests whether a model can:

1. Call a tool
2. Process the result
3. Call another tool based on that result
4. Continue until the task is complete

This is fundamental for real applications. A chatbot that can't handle "search for X, then summarize the results" is useless.

## Models with the Agency Gap

These models passed T0-T2 but failed A1:

"""

    for model in failing_models:
        scores = metrics.get("model_scores", {}).get(model, {})
        t_score = int(scores.get("tool_calling_score", 0) * 100)
        a_score = int(scores.get("agency_score", 0) * 100)

        article += f"- **{model}**: {t_score}% tool calling, {a_score}% agency\n"

    article += f"""

## Why This Matters

In production, you need models that can:

- Search your database, then format the results
- Fetch user data, then update it based on business logic
- Check permissions, then execute the requested action

**Single-turn tool calling is a parlor trick. Multi-turn agency is what ships.**

## Recommendations

1. **Always test A1** before deploying a model
2. **Don't trust claims** - many models advertise tool calling but fail multi-turn
3. **Use this benchmark** to find models that actually work

## Models That Pass Both

For comparison, these models handle both T and A:

"""

    # Find models that pass both
    passing_both = []
    for model, scores in metrics.get("model_scores", {}).items():
        if scores.get("tool_calling_score", 0) >= 0.90 and scores.get("agency_score", 0) >= 0.90:
            passing_both.append({
                "model": model,
                "t_score": int(scores.get("tool_calling_score", 0) * 100),
                "a_score": int(scores.get("agency_score", 0) * 100)
            })

    passing_both.sort(key=lambda x: x["a_score"], reverse=True)

    for item in passing_both[:5]:
        article += f"- **{item['model']}**: {item['t_score']}% T, {item['a_score']}% A\n"

    article += "\n---\n\n"
    article += "Full benchmark: https://github.com/jw409/modelforecast\n"

    return article


def generate_article_hidden_champion(data: Dict[str, Any]) -> str:
    """Generate hidden champion article."""
    today = datetime.now().strftime("%Y-%m-%d")

    rankings = data.get("rankings", {})
    metrics = data.get("metrics", {})

    hidden_champions = rankings.get("hidden_champions", [])

    if not hidden_champions:
        return "# No hidden champions found\n\nAll top performers are well-known models.\n"

    # Pick the top hidden champion
    champion = hidden_champions[0]
    model = champion.get("model", "unknown")
    outperforms = champion.get("outperforms", [])

    scores = metrics.get("model_scores", {}).get(model, {})
    t_score = int(scores.get("tool_calling_score", 0) * 100)
    r_score = int(scores.get("reasoning_score", 0) * 100)
    a_score = int(scores.get("agency_score", 0) * 100)
    overall = int(scores.get("overall_score", 0) * 100)

    famous_model = outperforms[0] if outperforms else "popular models"

    article = f"""# {model}: The Free Model That Beats {famous_model}

*{today}*

## The Surprise

**{model}** scored {overall}% overall, outperforming {len(outperforms)} better-known models including:

"""

    for beaten in outperforms[:5]:
        beaten_scores = metrics.get("model_scores", {}).get(beaten, {})
        beaten_overall = int(beaten_scores.get("overall_score", 0) * 100)
        article += f"- **{beaten}**: {beaten_overall}% overall\n"

    article += f"""

## The Numbers

| Dimension | Score |
|-----------|------:|
| Tool Calling (T) | {t_score}% |
| Reasoning (R) | {r_score}% |
| Agency (A) | {a_score}% |
| **Overall** | **{overall}%** |

## Why It Works

"""

    # Analyze strengths
    if t_score >= 90:
        article += f"- **Rock-solid tool calling**: {t_score}% pass rate on T0-T2\n"

    if r_score >= 90:
        article += f"- **Smart tool selection**: {r_score}% on reasoning tests\n"

    if a_score >= 90:
        article += f"- **Multi-turn capable**: {a_score}% on agency tests (rare!)\n"

    article += f"""

## Who Should Use This

This model is ideal for:

- Production applications requiring reliable tool calling
- Multi-turn conversational agents
- Budget-conscious projects (it's free!)

## Caveats

This is a **tool calling benchmark**. We test T/R/A dimensions only. We don't test:

- Creative writing quality
- Code generation accuracy
- Factual knowledge
- Response speed

For those use cases, test yourself.

## How We Found It

We tested {metrics.get('total_models', 0)} free models on OpenRouter using a standardized benchmark. Full methodology in the repo.

---

Full benchmark: https://github.com/jw409/modelforecast
"""

    return article


def generate_article_graveyard(data: Dict[str, Any]) -> str:
    """Generate graveyard article."""
    today = datetime.now().strftime("%Y-%m-%d")

    insights = data.get("insights", {})
    metrics = data.get("metrics", {})

    graveyard = insights.get("graveyard", {})
    broken_models = graveyard.get("broken_models", [])

    article = f"""# Tool Calling Graveyard: {len(broken_models)} Models That Claim But Can't Deliver

*{today}*

## The Problem

{len(broken_models)} models on OpenRouter advertise tool calling support but score **below 50%** on basic tests.

This is a waste of your time and API credits.

## The Graveyard

These models claim tool calling but fail:

"""

    for model in broken_models:
        scores = metrics.get("model_scores", {}).get(model, {})
        overall = int(scores.get("overall_score", 0) * 100)
        t_score = int(scores.get("tool_calling_score", 0) * 100)

        article += f"- **{model}**: {overall}% overall, {t_score}% tool calling\n"

    article += f"""

## Common Failure Modes

From analyzing these models, we see:

1. **Malformed JSON**: Model returns invalid tool call syntax
2. **Missing parameters**: Calls functions without required arguments
3. **Hallucinated tools**: Invokes functions that don't exist
4. **Ignoring results**: Calls tool but doesn't process the response
5. **Infinite loops**: Repeats same tool call forever

## Warning Signs

Before deploying a model, watch for:

- **Documentation claims without proof**: "Supports function calling" with no examples
- **Low download counts**: Unpopular models are unpopular for a reason
- **No tool calling examples**: If the model card doesn't show tool use, it probably doesn't work

## What to Use Instead

Models that **actually work** (≥90% overall):

"""

    # Find working models
    working = []
    for model, scores in metrics.get("model_scores", {}).items():
        if scores.get("overall_score", 0) >= 0.90:
            working.append({
                "model": model,
                "overall": int(scores.get("overall_score", 0) * 100),
                "t_score": int(scores.get("tool_calling_score", 0) * 100)
            })

    working.sort(key=lambda x: x["overall"], reverse=True)

    for item in working[:5]:
        article += f"- **{item['model']}**: {item['overall']}% overall, {item['t_score']}% tool calling\n"

    article += f"""

## Testing Methodology

We run 5 standardized tests:

- **T0**: Single tool call (basic)
- **T1**: Tool call with complex parameters
- **T2**: Choosing between multiple tools
- **R1**: Reasoning about which tool to use
- **A1**: Multi-turn conversation (agency)

Models must pass ≥3 to avoid the graveyard.

## The Lesson

**Trust the benchmark, not the marketing.**

Many models claim tool calling support. Few deliver. This benchmark separates hype from reality.

---

Full benchmark: https://github.com/jw409/modelforecast
"""

    return article


def main():
    parser = argparse.ArgumentParser(description="Generate content from benchmark results")
    parser.add_argument("--all", action="store_true", help="Generate all content")
    parser.add_argument("--readme", action="store_true", help="Generate README section")
    parser.add_argument("--headline", action="store_true", help="Generate HEADLINE.md")
    parser.add_argument("--article", choices=["agency-gap", "hidden-champion", "graveyard"],
                       help="Generate specific article")

    args = parser.parse_args()

    # Load analysis data
    print("Loading analysis data...")
    data = load_analysis_data()

    if not data:
        print("ERROR: No analysis data found. Run analyze_results.py first.")
        return 1

    # Generate content
    if args.all or args.readme:
        print("\n" + "="*80)
        print("README SECTION (copy to README.md)")
        print("="*80)
        print(generate_readme_section(data))

    if args.all or args.headline:
        headline_content = generate_headline(data)
        headline_path = Path("HEADLINE.md")
        headline_path.write_text(headline_content)
        print(f"\n✓ Updated {headline_path}")

    if args.article or args.all:
        articles_dir = Path("articles")
        articles_dir.mkdir(exist_ok=True)

        today = datetime.now().strftime("%Y-%m-%d")

        articles_to_generate = []
        if args.all:
            articles_to_generate = ["agency-gap", "hidden-champion", "graveyard"]
        elif args.article:
            articles_to_generate = [args.article]

        for article_type in articles_to_generate:
            if article_type == "agency-gap":
                content = generate_article_agency_gap(data)
                filename = f"{today}-agency-gap.md"
            elif article_type == "hidden-champion":
                content = generate_article_hidden_champion(data)
                filename = f"{today}-hidden-champion.md"
            elif article_type == "graveyard":
                content = generate_article_graveyard(data)
                filename = f"{today}-graveyard.md"

            article_path = articles_dir / filename
            article_path.write_text(content)
            print(f"✓ Generated {article_path}")

    if not any([args.all, args.readme, args.headline, args.article]):
        parser.print_help()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
