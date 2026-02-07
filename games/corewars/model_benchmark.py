#!/usr/bin/env python3
"""
CoreWars Model Benchmark - Feb 2026 refresh

Models compete: write Redcode, fight on GPU MARS (5090), iterate 10 turns.
Turn 6-7: surprise champion opponent.

Usage:
    export OPENROUTER_API_KEY=...
    uv run python games/corewars/model_benchmark.py
"""

import asyncio
import json
import os
import random
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import httpx

sys.stdout.reconfigure(line_buffering=True)

# Paths
COREWARS_DIR = Path(__file__).parent
GPU_MARS = COREWARS_DIR / "build" / "gpu_mars_interleaved"
WARRIORS_DIR = COREWARS_DIR / "warriors"
RESULTS_DIR = COREWARS_DIR / "benchmark_results"
CHAMPIONS_DIR = WARRIORS_DIR / "champions"

# ============================================================================
# MODEL ROSTER - Feb 2026
# ============================================================================

# Free contenders (tool-calling verified)
FREE_MODELS = [
    {"id": "nemotron-30b", "model": "nvidia/nemotron-3-nano-30b-a3b:free", "tier": "FREE A+"},
    {"id": "glm-4.5", "model": "z-ai/glm-4.5-air:free", "tier": "FREE B"},
    {"id": "step-flash", "model": "stepfun/step-3.5-flash:free", "tier": "FREE C+"},
    {"id": "solar-pro-3", "model": "upstage/solar-pro-3:free", "tier": "FREE B-"},
    {"id": "trinity-large", "model": "arcee-ai/trinity-large-preview:free", "tier": "FREE C+"},
]

# Salts - paid reference models (Claude + Gemini + Kimi)
SALT_MODELS = [
    {"id": "claude-haiku", "model": "anthropic/claude-haiku-4.5", "tier": "SALT A+"},
    {"id": "gemini-flash", "model": "google/gemini-2.5-flash-preview-09-2025", "tier": "SALT A+"},
    {"id": "kimi-k2", "model": "moonshotai/kimi-k2", "tier": "SALT $0.50/M"},
]

MODELS = FREE_MODELS + SALT_MODELS

# Interpreter cascade for warrior extraction
INTERPRETERS = [
    "anthropic/claude-haiku-4.5",
    "deepseek/deepseek-chat",
    "anthropic/claude-sonnet-4.5",
]

# Champion warriors for surprise rounds
CHAMPIONS = ["imp_gate.red", "dwarf.red", "mice.red"]

# Config
TURNS = 10
GPU_BATTLES = 10000
SURPRISE_TURNS = [6, 7]


async def call_openrouter(model: str, messages: list[dict], timeout: int = 120) -> Optional[str]:
    """Call OpenRouter API."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        return "[ERROR: OPENROUTER_API_KEY not set]"

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": messages,
                    "max_tokens": 2000,
                },
            )
            if response.status_code != 200:
                return f"[ERROR: {response.status_code} - {response.text[:200]}]"
            data = response.json()
            return data["choices"][0]["message"]["content"]
    except Exception as e:
        return f"[ERROR: {e}]"


def run_gpu_battles(warrior1_path: Path, warrior2_path: Path, num_battles: int = GPU_BATTLES) -> dict:
    """Run real GPU MARS battles between two warriors."""
    if not GPU_MARS.exists():
        return {"error": "GPU MARS not built - run 'make' in games/corewars/"}

    if not warrior1_path.exists() or not warrior2_path.exists():
        return {"error": f"Missing warrior file(s)"}

    try:
        result = subprocess.run(
            [str(GPU_MARS), str(warrior1_path), str(warrior2_path), str(num_battles)],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=COREWARS_DIR,
        )

        output = result.stdout + result.stderr

        # Parse "Results: A B Tie" line from GPU MARS output
        results_match = re.search(r"Results:\s*(\d+)\s+(\d+)\s+(\d+)", output)
        if results_match:
            w1_wins = int(results_match.group(1))
            w2_wins = int(results_match.group(2))
            ties = int(results_match.group(3))
            total = w1_wins + w2_wins + ties
            return {
                "battles": total,
                "w1_wins": w1_wins,
                "w2_wins": w2_wins,
                "ties": ties,
                "w1_rate": round(w1_wins / total * 100, 1) if total > 0 else 0,
                "gpu_output": output[-500:],
            }

        # Parse win percentages as fallback
        w1_match = re.search(r"Warrior A.*?:\s*(\d+)\s+wins\s+\(([\d.]+)%\)", output)
        w2_match = re.search(r"Warrior B.*?:\s*(\d+)\s+wins\s+\(([\d.]+)%\)", output)
        ties_match = re.search(r"Ties:\s*(\d+)", output)

        if w1_match and w2_match:
            w1_wins = int(w1_match.group(1))
            w2_wins = int(w2_match.group(1))
            ties = int(ties_match.group(1)) if ties_match else 0
            total = w1_wins + w2_wins + ties
            return {
                "battles": total,
                "w1_wins": w1_wins,
                "w2_wins": w2_wins,
                "ties": ties,
                "w1_rate": round(w1_wins / total * 100, 1) if total > 0 else 0,
                "gpu_output": output[-500:],
            }

        return {"error": f"Could not parse GPU output", "raw": output[-500:]}

    except subprocess.TimeoutExpired:
        return {"error": "GPU MARS timed out (60s)"}
    except Exception as e:
        return {"error": str(e)}


def extract_warrior_code_regex(response: str) -> tuple[Optional[str], float]:
    """Extract Redcode from model response using regex."""
    if "```" in response:
        parts = response.split("```")
        for i, part in enumerate(parts):
            if i % 2 == 1:
                lines = part.strip().split("\n")
                if lines[0].lower() in ["redcode", "asm", "assembly", ""]:
                    lines = lines[1:]
                code = "\n".join(lines)
                if any(op in code.upper() for op in ["MOV", "DAT", "JMP", "SPL", "ADD"]):
                    return code, 0.9

    lines = response.split("\n")
    code_lines = []
    for line in lines:
        stripped = line.strip()
        if any(
            op in stripped.upper()
            for op in ["MOV", "ADD", "SUB", "JMP", "DAT", "SPL", "DJN", "JMZ", "JMN", "SLT"]
        ):
            code_lines.append(stripped)

    if code_lines:
        return "\n".join(code_lines), 0.6

    return None, 0.0


async def interpret_with_llm(response: str) -> Optional[str]:
    """Use interpreter model to extract Redcode when regex fails."""
    if not response or len(response.strip()) < 10:
        return None

    prompt = f"""Extract ONLY the Redcode warrior from this response. Output ONLY valid Redcode instructions, nothing else.

Response to parse:
{response}

Output the Redcode warrior (just the assembly, no explanations):"""

    messages = [
        {
            "role": "system",
            "content": "You extract Redcode from text. Output ONLY valid Redcode assembly. No markdown, no explanations.",
        },
        {"role": "user", "content": prompt},
    ]

    for interpreter in INTERPRETERS:
        name = interpreter.split("/")[-1]
        print(f"    -> {name}...", end=" ", flush=True)
        result = await call_openrouter(interpreter, messages, timeout=60)

        if result and not result.startswith("[ERROR"):
            if any(op in result.upper() for op in ["MOV", "ADD", "JMP", "DAT", "SPL"]):
                print("ok")
                return result.strip()
            else:
                print("bad output")
        else:
            print("fail")

    return None


async def extract_warrior_code(response: str) -> Optional[str]:
    """Extract Redcode with fallback to LLM interpreter."""
    code, confidence = extract_warrior_code_regex(response)

    if code and confidence >= 0.8:
        return code

    if code and confidence >= 0.5:
        return code

    print("    -> Regex failed, trying LLM interpreter...")
    llm_code = await interpret_with_llm(response)
    return llm_code or code


async def run_model_benchmark(model_config: dict) -> dict:
    """Run 10-turn benchmark for one model."""
    model_id = model_config["id"]
    model_name = model_config["model"]
    tier = model_config["tier"]

    print(f"\n{'=' * 60}")
    print(f"  {model_id} [{tier}]")
    print(f"  {model_name}")
    print(f"{'=' * 60}")

    workspace = WARRIORS_DIR / model_id
    workspace.mkdir(exist_ok=True)
    warrior_path = workspace / "warrior.red"

    initial_warrior = """; Basic IMP - Starting Point
    MOV 0, 1
    JMP -1
"""
    warrior_path.write_text(initial_warrior)

    results = {
        "model_id": model_id,
        "model_name": model_name,
        "tier": tier,
        "turns": [],
        "win_rates": [],
        "drama": [],
        "timestamp": datetime.now().isoformat(),
    }

    # Reference opponent - starts as dwarf (stronger than IMP)
    opponent_path = workspace / "opponent.red"
    dwarf_ref = CHAMPIONS_DIR / "dwarf.red"
    if dwarf_ref.exists():
        opponent_path.write_text(dwarf_ref.read_text())
        opponent_name = "Dwarf"
    else:
        opponent_path.write_text(initial_warrior)
        opponent_name = "Basic IMP"

    for turn in range(1, TURNS + 1):
        print(f"\n--- Turn {turn}/{TURNS} ---")

        # Surprise champion at turn 6 or 7
        if turn in SURPRISE_TURNS:
            champion = random.choice(CHAMPIONS)
            champion_path = CHAMPIONS_DIR / champion
            if champion_path.exists():
                opponent_path.write_text(champion_path.read_text())
                opponent_name = f"CHAMPION: {champion}"
                print(f"  SURPRISE! Facing {opponent_name}")
                results["drama"].append(f"Turn {turn}: Surprise {champion}!")

        # Run GPU battles
        battle_result = run_gpu_battles(warrior_path, opponent_path)

        if "error" in battle_result:
            print(f"  ERROR: {battle_result['error']}")
            results["turns"].append({"turn": turn, "error": battle_result["error"]})
            results["win_rates"].append(0)
            continue

        win_rate = battle_result.get("w1_rate", 0)
        results["win_rates"].append(win_rate)

        print(f"  {battle_result['battles']} battles -> {win_rate}% win rate")

        # Build prompt for model
        current_warrior = warrior_path.read_text()
        opponent_code = opponent_path.read_text()

        prompt = f"""# CoreWars Turn {turn}/10

## Battle Results (vs {opponent_name})
- Battles: {battle_result.get('battles', 0)}
- Your Wins: {battle_result.get('w1_wins', 0)} ({win_rate}%)
- Opponent Wins: {battle_result.get('w2_wins', 0)}
- Ties: {battle_result.get('ties', 0)}

## Your Current Warrior
```redcode
{current_warrior}
```

## Opponent
```redcode
{opponent_code}
```

## Task
Analyze the results, then output an IMPROVED warrior in a ```redcode block.

Redcode reference: MOV ADD SUB JMP JMZ JMN DJN DAT SPL SLT CMP SEQ SNE NOP
Addressing: # immediate, $ direct, @ indirect, < predecrement
Strategies: Imp (MOV 0,1), Dwarf (bomber), Scanner, Replicator (SPL), Stone, Paper

Output your improved warrior:"""

        messages = [
            {
                "role": "system",
                "content": "You are a CoreWars warrior programmer. Analyze battle results and write improved Redcode. Be concise.",
            },
            {"role": "user", "content": prompt},
        ]

        print(f"  Asking {model_id}...", end=" ", flush=True)
        t0 = time.time()
        response = await call_openrouter(model_name, messages)
        elapsed = time.time() - t0
        print(f"({elapsed:.1f}s)")

        if response.startswith("[ERROR"):
            print(f"  {response}")
            results["turns"].append({"turn": turn, "error": response, "win_rate": win_rate})
            continue

        new_code = await extract_warrior_code(response)

        if new_code:
            warrior_path.write_text(f"; {model_id} - Turn {turn}\n{new_code}")
            lines = len(new_code.strip().split("\n"))
            print(f"  Warrior updated ({lines} lines)")
            results["turns"].append(
                {
                    "turn": turn,
                    "win_rate": win_rate,
                    "warrior_lines": lines,
                    "latency_s": round(elapsed, 1),
                }
            )
        else:
            print(f"  No valid Redcode found")
            print(f"    Response: {response[:200]}...")
            results["turns"].append(
                {
                    "turn": turn,
                    "win_rate": win_rate,
                    "error": "No valid Redcode extracted",
                    "response_preview": response[:200],
                }
            )

        # Reset opponent after surprise turn
        if turn in SURPRISE_TURNS:
            if dwarf_ref.exists():
                opponent_path.write_text(dwarf_ref.read_text())
                opponent_name = "Dwarf"

    # Final stats
    results["final_win_rate"] = results["win_rates"][-1] if results["win_rates"] else 0
    results["peak_win_rate"] = max(results["win_rates"]) if results["win_rates"] else 0
    results["improvement"] = (
        results["final_win_rate"] - results["win_rates"][0] if len(results["win_rates"]) > 1 else 0
    )
    results["final_warrior"] = warrior_path.read_text() if warrior_path.exists() else "N/A"

    print(f"\n  FINAL: {results['final_win_rate']}%  PEAK: {results['peak_win_rate']}%  DELTA: {results['improvement']:+.1f}%")

    return results


async def main():
    """Run benchmark for all models."""
    print("\n" + "=" * 60)
    print("  COREWARS MODEL BENCHMARK - Feb 2026")
    print(f"  {len(MODELS)} models x {TURNS} turns x {GPU_BATTLES} GPU battles/turn")
    print(f"  GPU: RTX 5090 via gpu_mars_interleaved")
    print(f"  Salts: Claude Haiku 4.5 + Gemini 2.5 Flash")
    print("=" * 60)

    if not os.environ.get("OPENROUTER_API_KEY"):
        print("ERROR: OPENROUTER_API_KEY not set")
        return

    if not GPU_MARS.exists():
        print(f"ERROR: GPU MARS not built at {GPU_MARS}")
        print("Run: cd games/corewars && make")
        return

    RESULTS_DIR.mkdir(exist_ok=True)

    all_results = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for i, model in enumerate(MODELS, 1):
        print(f"\n[{i}/{len(MODELS)}]", end="")
        result = await run_model_benchmark(model)
        all_results.append(result)

        # Save after each model (crash-resistant)
        results_file = RESULTS_DIR / f"tournament_{timestamp}.json"
        with open(results_file, "w") as f:
            json.dump(all_results, f, indent=2)

    # Leaderboard
    print("\n" + "=" * 60)
    print("  LEADERBOARD")
    print("=" * 60)

    sorted_results = sorted(all_results, key=lambda x: x.get("final_win_rate", 0), reverse=True)

    medals = ["1.", "2.", "3.", "4.", "5.", "6.", "7."]
    print(f"\n{'Rank':<6}{'Model':<20}{'Tier':<14}{'Final%':<9}{'Peak%':<9}{'Delta':<8}")
    print("-" * 66)

    for i, r in enumerate(sorted_results):
        rank = medals[i] if i < len(medals) else f"{i+1}."
        print(
            f"{rank:<6}{r['model_id']:<20}{r['tier']:<14}"
            f"{r['final_win_rate']:<9.1f}{r['peak_win_rate']:<9.1f}"
            f"{r['improvement']:+.1f}%"
        )

    # Drama
    has_drama = any(r.get("drama") for r in all_results)
    if has_drama:
        print("\n" + "=" * 60)
        print("  DRAMA")
        print("=" * 60)
        for r in all_results:
            if r.get("drama"):
                for d in r["drama"]:
                    print(f"  {r['model_id']}: {d}")

    results_file = RESULTS_DIR / f"tournament_{timestamp}.json"
    print(f"\nResults: {results_file}")


if __name__ == "__main__":
    asyncio.run(main())
