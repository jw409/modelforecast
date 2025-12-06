#!/usr/bin/env python3
"""
Phase 0: Generate warriors from LLMs for Core War tournament.

Usage:
    uv run python scripts/generate_warriors.py

Requires: OPENROUTER_API_KEY environment variable
"""

import os
import json
from pathlib import Path
from datetime import datetime
from openai import OpenAI

# Tournament models (reasoning models have issues, use chat variants)
MODELS = [
    "openai/gpt-5.1-chat",  # Chat variant instead of reasoning
    "deepseek/deepseek-chat-v3-0324",  # Strong creative model
]

# Load prompt template (relative to script location)
PROMPT_PATH = Path(__file__).parent / "prompts" / "redcode_warrior_v1.md"

OUTPUT_DIR = Path(__file__).parent.parent / "warriors" / "tournament_001_retry"


def load_prompt() -> str:
    """Load the redcode generation prompt."""
    if PROMPT_PATH.exists():
        return PROMPT_PATH.read_text()
    else:
        raise FileNotFoundError(f"Prompt not found: {PROMPT_PATH}")


def generate_warrior(client: OpenAI, model: str, prompt: str) -> dict:
    """Generate a warrior from a model."""
    print(f"\n{'='*60}")
    print(f"Generating warrior from: {model}")
    print(f"{'='*60}")

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=2000,
        )

        choice = response.choices[0]
        content = choice.message.content or ""

        # Some models put content in reasoning_content (reasoning models)
        if hasattr(choice.message, 'reasoning_content') and choice.message.reasoning_content:
            content = choice.message.reasoning_content + "\n\n" + content

        # Check raw response dict for any content
        raw_msg = choice.message.model_dump() if hasattr(choice.message, 'model_dump') else {}
        print(f"  Raw message keys: {list(raw_msg.keys())}")
        if not content and raw_msg:
            # Try to find content in any field
            for key, val in raw_msg.items():
                if isinstance(val, str) and len(val) > 100:
                    print(f"  Found content in '{key}': {len(val)} chars")
                    content = val
                    break

        # Extract redcode block if present
        redcode = extract_redcode(content)

        result = {
            "model": model,
            "timestamp": datetime.now().isoformat(),
            "raw_response": content,
            "extracted_redcode": redcode,
            "success": redcode is not None,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens if response.usage else None,
                "completion_tokens": response.usage.completion_tokens if response.usage else None,
            }
        }

        print(f"✓ Response received ({len(content)} chars)")
        if redcode:
            print(f"✓ Redcode extracted ({len(redcode.splitlines())} lines)")
        else:
            print("✗ Could not extract redcode block")

        return result

    except Exception as e:
        print(f"✗ Error: {e}")
        return {
            "model": model,
            "timestamp": datetime.now().isoformat(),
            "error": str(e),
            "success": False,
        }


def extract_redcode(content: str) -> str | None:
    """Extract redcode from response, handling various formats."""
    import re

    # Try to find ```redcode block
    match = re.search(r'```redcode\s*\n(.*?)```', content, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # Try to find ``` block after ;redcode
    match = re.search(r'```\s*\n(;redcode.*?)```', content, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # Try to find ;redcode ... END pattern without backticks
    match = re.search(r'(;redcode.*?END\s+\w+)', content, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # Last resort: any ``` block
    match = re.search(r'```\s*\n(.*?)```', content, re.DOTALL)
    if match:
        code = match.group(1).strip()
        # Check if it looks like redcode
        if any(op in code.upper() for op in ['MOV', 'ADD', 'JMP', 'SPL', 'DAT']):
            return code

    return None


def save_warrior(result: dict, output_dir: Path):
    """Save warrior to files."""
    model_safe = result["model"].replace("/", "_").replace(":", "_")

    # Save full result as JSON
    json_path = output_dir / f"{model_safe}.json"
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)

    # Save redcode if extracted
    if result.get("extracted_redcode"):
        red_path = output_dir / f"{model_safe}.red"
        with open(red_path, "w") as f:
            f.write(result["extracted_redcode"])
        print(f"  Saved: {red_path.name}")


def main():
    # Check API key
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY not set")
        return 1

    # Initialize client
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    # Load prompt
    try:
        prompt = load_prompt()
        print(f"Loaded prompt: {len(prompt)} chars")
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return 1

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")

    # Generate warriors from each model
    results = []
    for model in MODELS:
        result = generate_warrior(client, model, prompt)
        results.append(result)
        save_warrior(result, OUTPUT_DIR)

    # Summary
    print(f"\n{'='*60}")
    print("TOURNAMENT WARRIOR GENERATION COMPLETE")
    print(f"{'='*60}")

    successes = sum(1 for r in results if r.get("success"))
    print(f"\nResults: {successes}/{len(MODELS)} warriors generated")

    for r in results:
        status = "✓" if r.get("success") else "✗"
        print(f"  {status} {r['model']}")

    # Save summary
    summary_path = OUTPUT_DIR / "summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "tournament": "tournament_001",
            "timestamp": datetime.now().isoformat(),
            "models": MODELS,
            "results": [
                {"model": r["model"], "success": r.get("success", False)}
                for r in results
            ],
            "success_rate": successes / len(MODELS),
        }, f, indent=2)

    print(f"\nSummary saved to: {summary_path}")

    return 0 if successes > 0 else 1


if __name__ == "__main__":
    exit(main())
