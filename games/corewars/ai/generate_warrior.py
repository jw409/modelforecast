#!/usr/bin/env python3
"""
AI Warrior Generator for CoreWars GPU

Generates CoreWars warriors using LLMs via OpenRouter API.
Integrates with the tournament system to evaluate AI-generated warriors.
"""

import os
import sys
import json
import requests
from pathlib import Path
from typing import Optional, Dict, Any

# Add talent-os to path for OpenRouter integration
REPO_ROOT = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "talent-os"))

try:
    from lib.agent_runtime.openrouter import OpenRouterClient
except ImportError:
    print("Warning: Could not import OpenRouterClient. Will use direct API calls.")
    OpenRouterClient = None


class WarriorGenerator:
    """Generate CoreWars warriors using LLMs"""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY not found in environment")

        self.prompt_template = self._load_prompt_template()
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"

    def _load_prompt_template(self) -> str:
        """Load the warrior generation prompt"""
        prompt_file = Path(__file__).parent / "warrior_prompt.md"
        if not prompt_file.exists():
            raise FileNotFoundError(f"Prompt template not found: {prompt_file}")

        with open(prompt_file, 'r') as f:
            return f.read()

    def generate_warrior(
        self,
        model: str = "google/gemini-2.5-flash-lite-preview-09-2025",
        strategy_hint: str = "",
        temperature: float = 0.8,
        max_tokens: int = 2000
    ) -> str:
        """
        Generate a warrior using specified LLM

        Args:
            model: OpenRouter model ID
            strategy_hint: Additional guidance (e.g., "focus on bombing strategy")
            temperature: Creativity level (0.0-2.0)
            max_tokens: Maximum response length

        Returns:
            Redcode warrior source code
        """
        # Build full prompt
        full_prompt = self.prompt_template
        if strategy_hint:
            full_prompt += f"\n\n## Additional Guidance\n\n{strategy_hint}"

        # Make API call
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "https://github.com/jw409/modelforecast",
            "X-Title": "CoreWars GPU Warrior Generator"
        }

        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": full_prompt
                }
            ],
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        print(f"Generating warrior with {model}...")
        print(f"Strategy hint: {strategy_hint or 'None'}")

        response = requests.post(
            self.base_url,
            headers=headers,
            json=payload
        )

        if response.status_code != 200:
            raise Exception(f"API error: {response.status_code} - {response.text}")

        result = response.json()
        warrior_code = result["choices"][0]["message"]["content"]

        return self._extract_redcode(warrior_code)

    def _extract_redcode(self, response: str) -> str:
        """Extract Redcode from LLM response (may be wrapped in markdown)"""
        # Look for code blocks
        if "```" in response:
            # Extract code from markdown block
            parts = response.split("```")
            for i, part in enumerate(parts):
                if i % 2 == 1:  # Inside code block
                    # Remove language identifier if present
                    lines = part.strip().split('\n')
                    if lines[0].strip().lower() in ['redcode', 'assembly', 'asm', '']:
                        return '\n'.join(lines[1:] if lines[0].strip() else lines)
                    return part.strip()

        # No code blocks, return as-is (filter out obvious non-code)
        lines = response.split('\n')
        code_lines = []
        for line in lines:
            stripped = line.strip()
            # Skip obvious markdown/prose
            if stripped.startswith('#') and not stripped.startswith('#'):
                continue
            if stripped.startswith('**'):
                continue
            code_lines.append(line)

        return '\n'.join(code_lines)

    def save_warrior(self, code: str, output_dir: Path, warrior_name: Optional[str] = None) -> Path:
        """
        Save warrior to file

        Args:
            code: Redcode source
            output_dir: Directory to save warrior
            warrior_name: Optional custom name (otherwise extract from code)

        Returns:
            Path to saved warrior file
        """
        # Extract name from code if not provided
        if not warrior_name:
            for line in code.split('\n'):
                if line.strip().startswith(';name'):
                    warrior_name = line.split('name', 1)[1].strip()
                    break
            if not warrior_name:
                warrior_name = "unnamed"

        # Sanitize filename
        safe_name = "".join(c for c in warrior_name if c.isalnum() or c in (' ', '-', '_'))
        safe_name = safe_name.replace(' ', '_').lower()

        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{safe_name}.red"

        with open(output_file, 'w') as f:
            f.write(code)

        print(f"Saved warrior to: {output_file}")
        return output_file


def generate_warrior_batch(
    num_warriors: int = 5,
    model: str = "google/gemini-2.5-flash-lite-preview-09-2025",
    output_dir: Optional[Path] = None,
    strategies: Optional[list[str]] = None
) -> list[Path]:
    """
    Generate multiple warriors for tournament

    Args:
        num_warriors: Number to generate
        model: OpenRouter model ID
        output_dir: Where to save warriors
        strategies: Optional list of strategy hints

    Returns:
        List of paths to generated warrior files
    """
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "warriors" / model.split('/')[-1]

    generator = WarriorGenerator()

    # Default strategies if none provided
    if strategies is None:
        strategies = [
            "Create an aggressive bomber that clears memory quickly",
            "Create a defensive replicator that spreads copies across memory",
            "Create a scanner that hunts for enemies before attacking",
            "Create a fast imp-based strategy with process management",
            "Create a hybrid strategy combining bombing with replication"
        ]

    # Generate warriors
    warrior_files = []
    for i in range(num_warriors):
        strategy = strategies[i % len(strategies)]
        print(f"\n=== Generating warrior {i+1}/{num_warriors} ===")

        try:
            code = generator.generate_warrior(
                model=model,
                strategy_hint=strategy,
                temperature=0.8 + (i * 0.1)  # Vary temperature for diversity
            )

            # Save to file
            output_path = generator.save_warrior(
                code,
                output_dir,
                warrior_name=f"{model.split('/')[-1]}_warrior_{i+1}"
            )
            warrior_files.append(output_path)

            print(f"✓ Generated warrior {i+1}")

        except Exception as e:
            print(f"✗ Failed to generate warrior {i+1}: {e}")

    return warrior_files


def main():
    """CLI interface"""
    import argparse

    parser = argparse.ArgumentParser(description="Generate CoreWars warriors using AI")
    parser.add_argument("--model", "-m", default="google/gemini-2.5-flash-lite-preview-09-2025",
                       help="OpenRouter model ID")
    parser.add_argument("--count", "-n", type=int, default=5,
                       help="Number of warriors to generate")
    parser.add_argument("--output", "-o", type=Path,
                       help="Output directory (default: warriors/<model_name>)")
    parser.add_argument("--strategy", "-s", type=str,
                       help="Strategy hint for generation")
    parser.add_argument("--batch", action="store_true",
                       help="Generate batch with diverse strategies")

    args = parser.parse_args()

    if args.batch:
        warrior_files = generate_warrior_batch(
            num_warriors=args.count,
            model=args.model,
            output_dir=args.output
        )
        print(f"\n=== Generated {len(warrior_files)} warriors ===")
        for wf in warrior_files:
            print(f"  - {wf}")

    else:
        generator = WarriorGenerator()
        code = generator.generate_warrior(
            model=args.model,
            strategy_hint=args.strategy or ""
        )

        if args.output:
            output_dir = args.output
        else:
            output_dir = Path(__file__).parent.parent / "warriors" / args.model.split('/')[-1]

        warrior_file = generator.save_warrior(code, output_dir)
        print(f"\n✓ Warrior saved to: {warrior_file}")


if __name__ == "__main__":
    main()
