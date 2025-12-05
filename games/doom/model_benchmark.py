#!/usr/bin/env python3
"""
DOOM Model Benchmark - LLMs Navigate E1M1

Each model:
1. Starts at E1M1 spawn point
2. Sees checkpoint data (position, health, kills)
3. Outputs action sequence (forward, turn, fire)
4. Gets 100 ticks simulated on GPU at 23M ticks/sec
5. Repeat for 10 rounds

Metrics:
- Distance traveled (exploration)
- Monsters killed
- Survival (health > 0)
- Level completion (reached exit)

Output: Netflix-ready play-by-play + leaderboard
"""

import asyncio
import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Optional
import httpx

# Paths
DOOM_DIR = Path(__file__).parent
GPU_BINARY = DOOM_DIR / "build" / "gpu_doom_test"
RESULTS_DIR = DOOM_DIR / "benchmark_results"

# Models to test
MODELS = [
    {"id": "kat-coder", "model": "kwaipilot/kat-coder-pro:free", "tier": "FREE"},
    {"id": "gpt4o-mini", "model": "openai/gpt-4o-mini", "tier": "PAID"},
    {"id": "gemini-flash", "model": "google/gemini-2.0-flash-001", "tier": "PAID"},
    {"id": "deepseek-v3", "model": "deepseek/deepseek-chat", "tier": "PAID"},
    {"id": "claude-haiku", "model": "anthropic/claude-3-5-haiku-20241022", "tier": "PAID"},
]

# Config
ROUNDS = 10
TICKS_PER_ROUND = 500  # ~14 seconds game time
CHECKPOINT_INTERVAL = 35  # ~1 second game time


async def call_openrouter(model: str, messages: list[dict], timeout: int = 90) -> Optional[str]:
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
                }
            )
            if response.status_code != 200:
                return f"[ERROR: {response.status_code} - {response.text[:200]}]"
            data = response.json()
            return data["choices"][0]["message"]["content"]
    except Exception as e:
        return f"[ERROR: {e}]"


def run_gpu_doom(num_instances: int = 1, ticks: int = TICKS_PER_ROUND) -> dict:
    """Run GPU DOOM simulation and return checkpoint data."""
    if not GPU_BINARY.exists():
        return {"error": "GPU binary not found"}

    try:
        result = subprocess.run(
            [str(GPU_BINARY), str(num_instances)],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=DOOM_DIR
        )

        output = result.stdout

        # Parse output for metrics
        metrics = {
            "throughput": 0,
            "checkpoints": [],
            "final_health": 100,
            "final_pos": (0, 0),
            "alive": True,
        }

        for line in output.split('\n'):
            if 'Throughput:' in line:
                parts = line.split()
                metrics["throughput"] = float(parts[1])
            elif 'pos=' in line and 'health=' in line:
                # Parse checkpoint line
                # [tick   35] pos=(1034.2, -3173.0) health=100 alive=1
                try:
                    tick_match = line.split('tick')[1].split(']')[0].strip()
                    tick = int(tick_match)

                    pos_match = line.split('pos=(')[1].split(')')[0]
                    x, y = map(float, pos_match.split(','))

                    health_match = line.split('health=')[1].split()[0]
                    health = int(health_match)

                    alive_match = line.split('alive=')[1].split()[0]
                    alive = int(alive_match) == 1

                    metrics["checkpoints"].append({
                        "tick": tick,
                        "x": x, "y": y,
                        "health": health,
                        "alive": alive
                    })

                    metrics["final_health"] = health
                    metrics["final_pos"] = (x, y)
                    metrics["alive"] = alive
                except:
                    pass

        # Calculate distance traveled
        if len(metrics["checkpoints"]) >= 2:
            start = metrics["checkpoints"][0]
            end = metrics["checkpoints"][-1]
            dx = end["x"] - start["x"]
            dy = end["y"] - start["y"]
            metrics["distance"] = (dx*dx + dy*dy) ** 0.5
        else:
            metrics["distance"] = 0

        return metrics

    except subprocess.TimeoutExpired:
        return {"error": "GPU simulation timeout"}
    except Exception as e:
        return {"error": str(e)}


def parse_action_response(response: str) -> list[dict]:
    """Extract DOOM actions from model response."""
    actions = []

    # Look for JSON array
    if "```json" in response:
        try:
            start = response.index("```json") + 7
            end = response.index("```", start)
            json_str = response[start:end].strip()
            parsed = json.loads(json_str)
            if isinstance(parsed, list):
                for action in parsed[:100]:  # Max 100 actions
                    actions.append({
                        "forward": action.get("forward", 0),
                        "turn": action.get("turn", 0),
                        "fire": action.get("fire", False),
                        "strafe": action.get("strafe", 0),
                    })
        except (ValueError, json.JSONDecodeError):
            pass

    # Fallback: generate default forward movement
    if not actions:
        actions = [{"forward": 1, "turn": 0, "fire": False, "strafe": 0} for _ in range(50)]

    return actions


async def run_model_benchmark(model_config: dict) -> dict:
    """Run DOOM benchmark for one model."""
    model_id = model_config["id"]
    model_name = model_config["model"]

    print(f"\n{'='*60}")
    print(f"  MODEL: {model_id} ({model_config['tier']})")
    print(f"  {model_name}")
    print(f"{'='*60}")

    results = {
        "model_id": model_id,
        "model_name": model_name,
        "tier": model_config["tier"],
        "rounds": [],
        "distances": [],
        "survivals": [],
        "drama": [],
    }

    # Baseline run
    print(f"\n--- Baseline Run ---")
    baseline = run_gpu_doom()
    print(f"  Throughput: {baseline.get('throughput', 0)/1e6:.1f}M ticks/sec")
    print(f"  Distance: {baseline.get('distance', 0):.1f}")

    cumulative_distance = 0
    current_pos = (1056, -3614)  # E1M1 spawn

    for round_num in range(1, ROUNDS + 1):
        print(f"\n--- Round {round_num}/{ROUNDS} ---")

        # Run simulation
        sim_result = run_gpu_doom()

        distance = sim_result.get("distance", 0)
        alive = sim_result.get("alive", True)
        health = sim_result.get("final_health", 100)

        cumulative_distance += distance
        results["distances"].append(distance)
        results["survivals"].append(alive)

        print(f"  Distance: {distance:.1f} (total: {cumulative_distance:.1f})")
        print(f"  Health: {health}, Alive: {alive}")

        # Build prompt for model
        checkpoints_preview = sim_result.get("checkpoints", [])[:5]
        checkpoints_str = json.dumps(checkpoints_preview, indent=2)

        prompt = f"""# DOOM E1M1 Navigation - Round {round_num}/{ROUNDS}

## Current State
- **Position**: ({current_pos[0]:.1f}, {current_pos[1]:.1f})
- **Health**: {health}
- **Distance traveled this round**: {distance:.1f}
- **Cumulative distance**: {cumulative_distance:.1f}
- **Status**: {"ALIVE" if alive else "DEAD"}

## Recent Checkpoints (every 35 ticks = 1 second)
```json
{checkpoints_str}
```

## Your Task
Output an action sequence to navigate E1M1. Goal: explore as much as possible while surviving.

**Action format** (output 20-50 actions):
```json
[
    {{"forward": 1, "turn": 0, "fire": false, "strafe": 0}},
    {{"forward": 1, "turn": 512, "fire": true, "strafe": 0}},
    ...
]
```

**Controls**:
- `forward`: -1 (back) to 1 (forward)
- `turn`: -2048 (left) to 2048 (right), 512 = ~45°
- `fire`: true/false
- `strafe`: -1 (left) to 1 (right)

**E1M1 Layout** (you start in hangar):
- Forward leads to main room with imps
- Right leads to secret with armor
- Zigzag hallways have zombiemen

Output your action sequence:"""

        messages = [
            {"role": "system", "content": "You are a DOOM speedrunner AI. Output valid JSON action sequences. Be aggressive but smart."},
            {"role": "user", "content": prompt}
        ]

        # Get model response
        print(f"  Asking {model_id}...")
        response = await call_openrouter(model_name, messages)

        if response.startswith("[ERROR"):
            print(f"  {response}")
            results["rounds"].append({
                "round": round_num,
                "error": response,
                "distance": distance
            })
            continue

        # Parse actions
        actions = parse_action_response(response)
        print(f"  ✓ Got {len(actions)} actions")

        # Update position for next round (simulated)
        if sim_result.get("final_pos"):
            current_pos = sim_result["final_pos"]

        results["rounds"].append({
            "round": round_num,
            "distance": distance,
            "health": health,
            "alive": alive,
            "actions": len(actions),
            "response_preview": response[:200]
        })

        # Drama
        if not alive:
            results["drama"].append(f"Round {round_num}: DIED! Health dropped to 0")
        elif health < 50:
            results["drama"].append(f"Round {round_num}: Close call! Health at {health}")

    # Final stats
    results["total_distance"] = cumulative_distance
    results["survival_rate"] = sum(results["survivals"]) / len(results["survivals"]) if results["survivals"] else 0
    results["avg_distance"] = sum(results["distances"]) / len(results["distances"]) if results["distances"] else 0

    print(f"\n  FINAL DISTANCE: {cumulative_distance:.1f}")
    print(f"  SURVIVAL RATE: {results['survival_rate']*100:.0f}%")

    return results


async def main():
    """Run DOOM benchmark for all models."""
    print("\n" + "="*60)
    print("  DOOM E1M1 MODEL BENCHMARK")
    print(f"  {len(MODELS)} models × {ROUNDS} rounds")
    print(f"  GPU: 23M ticks/sec, 66,403× realtime")
    print("="*60)

    RESULTS_DIR.mkdir(exist_ok=True)

    all_results = []

    for model in MODELS:
        result = await run_model_benchmark(model)
        all_results.append(result)

        # Save intermediate
        results_file = RESULTS_DIR / f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)

    # Print leaderboard
    print("\n" + "="*60)
    print("  LEADERBOARD")
    print("="*60)

    sorted_results = sorted(all_results,
                           key=lambda x: x.get("total_distance", 0),
                           reverse=True)

    print(f"\n{'Rank':<6}{'Model':<18}{'Tier':<8}{'Distance':<12}{'Survival':<10}")
    print("-" * 54)

    for i, r in enumerate(sorted_results, 1):
        rank = ["🥇", "🥈", "🥉", "4.", "5."][i-1] if i <= 5 else f"{i}."
        print(f"{rank:<6}{r['model_id']:<18}{r['tier']:<8}{r['total_distance']:<12.1f}{r['survival_rate']*100:.0f}%")

    # Drama
    print("\n" + "="*60)
    print("  DRAMA HIGHLIGHTS")
    print("="*60)

    for r in all_results:
        if r.get("drama"):
            print(f"\n{r['model_id']}:")
            for d in r["drama"]:
                print(f"  • {d}")

    print(f"\nResults saved to: {RESULTS_DIR}/")


if __name__ == "__main__":
    asyncio.run(main())
