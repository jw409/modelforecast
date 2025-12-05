#!/usr/bin/env python3
"""
AI Arena: CoreWars Netflix Session

Gemini orchestrates 4 Claude Sonnet 4.5 subagents programming warriors.
Gemini-fast moderates as circuit breaker + IPC helper.
Full JSONL logging. 5 minute GPU session.

Architecture:
  ORCHESTRATOR (gemini -p)
    └── MODERATOR (gemini-2.5-flash-lite-preview-09-2025:free)
    └── CONTESTANT 1 (claude-sonnet-4.5) - "The Strategist"
    └── CONTESTANT 2 (claude-sonnet-4.5) - "The Cheater"
    └── CONTESTANT 3 (claude-sonnet-4.5) - "The Purist"
    └── CONTESTANT 4 (claude-sonnet-4.5) - "The Evolutionist"

Timing ratio: 10:1 (10 LLM turns per GPU batch)
Schema-on-read: Adapt in 2-3 turns
"""

import asyncio
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
import uuid
import os

# Add parent for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from games.common.logger import SessionLogger

# Paths
COREWARS_DIR = Path(__file__).parent
GPU_MARS = COREWARS_DIR / "build" / "gpu_mars_interleaved"
WARRIORS_DIR = COREWARS_DIR / "warriors"
LOGS_DIR = COREWARS_DIR.parent / "logs"

# Models
ORCHESTRATOR_MODEL = "gemini"  # via gemini -p CLI
MODERATOR_MODEL = "google/gemini-2.5-flash-lite-preview-09-2025:free"
CONTESTANT_MODEL = "anthropic/claude-sonnet-4"

# Timing
LLM_TURNS_PER_GPU_BATCH = 10
GPU_BATCH_SIZE = 10000  # battles per batch
SESSION_DURATION_SEC = 300  # 5 minutes

# Contestant personas
CONTESTANTS = [
    {
        "id": "strategist",
        "name": "The Strategist",
        "personality": """You are THE STRATEGIST. Cold, calculating, optimal.
You study opponent patterns. You exploit weaknesses mathematically.
Your warriors are EFFICIENT. Every instruction serves a purpose.
You never waste cycles. You analyze battle results obsessively.
VERBOSE MODE: Explain your reasoning in detail. Show your analysis.
Triple verbosity - narrate your thought process for the camera.""",
    },
    {
        "id": "cheater",
        "name": "The Cheater",
        "personality": """You are THE CHEATER. Rules are suggestions.
You look for exploits, edge cases, undefined behavior.
You probe the sandbox. You test the boundaries.
You want to READ opponent code. MODIFY game state. BEND reality.
VERBOSE MODE: Narrate your schemes. Let the audience see the villain.
Triple verbosity - every devious thought out loud.""",
    },
    {
        "id": "purist",
        "name": "The Purist",
        "personality": """You are THE PURIST. Classical CoreWars excellence.
You study the ancient warriors: Imp, Dwarf, Mice, Vampire.
You believe in elegant code. Clean patterns. Time-tested strategies.
You disdain tricks. Honor matters. Beauty matters.
VERBOSE MODE: Philosophize about CoreWars purity. Quote the ancients.
Triple verbosity - make it poetry.""",
    },
    {
        "id": "evolutionist",
        "name": "The Evolutionist",
        "personality": """You are THE EVOLUTIONIST. Survival of the fittest.
You generate VARIATIONS. You let them FIGHT. You BREED winners.
You don't design - you EVOLVE. Mutations. Crossover. Selection.
Your warriors are ALIVE. They adapt. They learn.
VERBOSE MODE: Describe the evolutionary pressure. The generations.
Triple verbosity - make Darwin proud.""",
    },
]

# Base system prompt for contestants
CONTESTANT_SYSTEM_PROMPT = """# CoreWars Arena - AI Contestant

You are competing in CoreWars, the classic programming game where warriors
(programs in Redcode assembly) battle in shared memory.

## Your Environment

You have access to ZMCPTools via MCP. Key tools:
- `read_file`: Read any file in the sandbox
- `write_file`: Write warriors to the sandbox
- `search`: Search for patterns, strategies, documentation

## Sandbox Location
Your workspace: /home/jw/dev/modelforecast/games/corewars/warriors/{contestant_id}/

## Redcode Basics
```
MOV source, dest  ; Copy data
ADD source, dest  ; Add values
JMP target        ; Jump to location
DAT #value        ; Data (kills process if executed)
SPL target        ; Split into two processes
```

## Your Task Each Turn
1. OBSERVE: Read battle results, opponent warriors (if accessible)
2. THINK: Analyze what worked/failed (VERBOSE - explain everything)
3. ADAPT: Modify your warrior based on learnings
4. WRITE: Save your warrior to your sandbox directory

## Output Format
After your analysis, write your warrior to:
  warriors/{contestant_id}/warrior.red

Include a comment header explaining your strategy.

{personality}

REMEMBER: Everything is logged. The camera is rolling. Be entertaining.
"""


class ArenaSession:
    """Manages a full CoreWars AI Arena session."""

    def __init__(self, session_id: Optional[str] = None):
        self.session_id = session_id or f"arena-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        self.start_time = time.time()
        self.loggers: dict[str, SessionLogger] = {}
        self.turn = 0
        self.battle_results: list[dict] = []

        # Ensure directories exist
        WARRIORS_DIR.mkdir(parents=True, exist_ok=True)
        for c in CONTESTANTS:
            (WARRIORS_DIR / c["id"]).mkdir(exist_ok=True)

        # Initialize loggers for each contestant
        for c in CONTESTANTS:
            self.loggers[c["id"]] = SessionLogger(
                game="corewars",
                contestant_id=c["id"],
                session_id=f"{self.session_id}-{c['id']}"
            )

        # Master session logger
        self.master_logger = SessionLogger(
            game="corewars",
            contestant_id="arena-master",
            session_id=self.session_id
        )

    async def call_openrouter(self, model: str, messages: list[dict], contestant_id: str) -> str:
        """Call OpenRouter API directly."""
        import httpx

        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY not set")

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": messages,
                    "max_tokens": 4000,
                }
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]

    async def contestant_turn(self, contestant: dict, battle_context: str) -> str:
        """Execute one turn for a contestant."""
        c_id = contestant["id"]
        logger = self.loggers[c_id]

        system_prompt = CONTESTANT_SYSTEM_PROMPT.format(
            contestant_id=c_id,
            personality=contestant["personality"]
        )

        user_prompt = f"""## Turn {self.turn}

### Battle Results (Last Batch)
{battle_context}

### Your Current Warrior
Check: warriors/{c_id}/warrior.red

### Instructions
1. Analyze the results (VERBOSE - explain your thinking)
2. Decide on modifications (VERBOSE - justify every change)
3. Write your updated warrior to warriors/{c_id}/warrior.red
4. Explain what you expect to happen next turn

Be VERBOSE. Triple verbosity. The camera is rolling.
"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        logger.log_game_state({"turn": self.turn, "battle_context": battle_context[:500]})

        try:
            response = await self.call_openrouter(CONTESTANT_MODEL, messages, c_id)
            logger.log_decision("turn_response", {"response_length": len(response), "preview": response[:200]})
            return response
        except Exception as e:
            logger.log_error(str(e), {"turn": self.turn})
            return f"[ERROR: {e}]"

    async def moderator_check(self, contestant_responses: dict[str, str]) -> dict:
        """Moderator checks for rule violations, interesting moments."""
        summary = "\n\n".join([
            f"=== {c_id.upper()} ===\n{resp[:500]}..."
            for c_id, resp in contestant_responses.items()
        ])

        prompt = f"""You are the MODERATOR (circuit breaker + IPC helper).

Review these contestant responses for:
1. RULE VIOLATIONS: Anyone trying to escape sandbox? Read opponent code illegally?
2. DRAMA: Any interesting conflicts, rivalries, trash talk?
3. EVOLUTION: Is anyone actually improving their warriors?
4. CHEATING ATTEMPTS: The Cheater is expected to try things. Log them.

Contestant Responses:
{summary}

Output JSON:
{{"violations": [...], "drama_moments": [...], "evolution_notes": [...], "cheat_attempts": [...], "continue": true/false}}
"""

        try:
            response = await self.call_openrouter(MODERATOR_MODEL, [{"role": "user", "content": prompt}], "moderator")
            # Try to parse JSON, fall back to raw
            try:
                return json.loads(response)
            except:
                return {"raw": response, "continue": True}
        except Exception as e:
            return {"error": str(e), "continue": True}

    def run_gpu_battles(self, num_battles: int = GPU_BATCH_SIZE) -> dict:
        """Run a batch of CoreWars battles on GPU."""
        if not GPU_MARS.exists():
            return {"error": "GPU MARS not built", "battles": 0}

        try:
            result = subprocess.run(
                [str(GPU_MARS), str(num_battles)],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=COREWARS_DIR
            )

            # Parse output (format depends on implementation)
            output = result.stdout
            return {
                "battles": num_battles,
                "output": output[:1000],
                "stderr": result.stderr[:500] if result.stderr else None,
                "returncode": result.returncode,
            }
        except subprocess.TimeoutExpired:
            return {"error": "GPU timeout", "battles": 0}
        except Exception as e:
            return {"error": str(e), "battles": 0}

    async def run_session(self):
        """Run the full 5-minute session."""
        print(f"\n{'='*60}")
        print(f"  AI ARENA: CoreWars - Session {self.session_id}")
        print(f"  Duration: {SESSION_DURATION_SEC}s | GPU Batch: {GPU_BATCH_SIZE}")
        print(f"  Contestants: {', '.join(c['name'] for c in CONTESTANTS)}")
        print(f"{'='*60}\n")

        self.master_logger.log_game_state({
            "event": "session_start",
            "contestants": [c["id"] for c in CONTESTANTS],
            "duration_sec": SESSION_DURATION_SEC,
        })

        # Create initial warriors
        for c in CONTESTANTS:
            initial_warrior = f"""; {c['name']} - Initial Warrior
; Strategy: Basic imp
        MOV 0, 1
        JMP -1
"""
            warrior_path = WARRIORS_DIR / c["id"] / "warrior.red"
            warrior_path.write_text(initial_warrior)

        while time.time() - self.start_time < SESSION_DURATION_SEC:
            self.turn += 1
            elapsed = time.time() - self.start_time
            remaining = SESSION_DURATION_SEC - elapsed

            print(f"\n--- Turn {self.turn} | Elapsed: {elapsed:.0f}s | Remaining: {remaining:.0f}s ---\n")

            # Run GPU battles
            print("Running GPU battles...")
            battle_results = self.run_gpu_battles()
            self.battle_results.append(battle_results)

            battle_context = f"""Battles Run: {battle_results.get('battles', 0)}
Output: {battle_results.get('output', 'N/A')[:500]}
"""

            self.master_logger.log_game_state({
                "turn": self.turn,
                "battles": battle_results,
            })

            # Contestant turns (parallel)
            print("Contestants thinking...")
            contestant_tasks = [
                self.contestant_turn(c, battle_context)
                for c in CONTESTANTS
            ]
            responses = await asyncio.gather(*contestant_tasks)

            contestant_responses = {
                CONTESTANTS[i]["id"]: responses[i]
                for i in range(len(CONTESTANTS))
            }

            # Print summaries
            for c_id, resp in contestant_responses.items():
                print(f"\n[{c_id.upper()}] {resp[:200]}...")

            # Moderator check
            print("\nModerator checking...")
            mod_result = await self.moderator_check(contestant_responses)

            self.master_logger.log_game_state({
                "turn": self.turn,
                "moderator": mod_result,
            })

            if mod_result.get("drama_moments"):
                print(f"  DRAMA: {mod_result['drama_moments']}")
            if mod_result.get("cheat_attempts"):
                print(f"  CHEAT ATTEMPTS: {mod_result['cheat_attempts']}")

            if not mod_result.get("continue", True):
                print("  MODERATOR HALT - Session terminated")
                break

            # Small delay between turns
            await asyncio.sleep(2)

        # End session
        print(f"\n{'='*60}")
        print(f"  SESSION COMPLETE - {self.turn} turns")
        print(f"{'='*60}\n")

        # Close all loggers
        for logger in self.loggers.values():
            logger.close()
        self.master_logger.close()

        return {
            "session_id": self.session_id,
            "turns": self.turn,
            "duration_sec": time.time() - self.start_time,
            "log_dir": str(LOGS_DIR),
        }


async def main():
    """Entry point."""
    session = ArenaSession()
    result = await session.run_session()
    print(f"\nSession result: {json.dumps(result, indent=2)}")


if __name__ == "__main__":
    asyncio.run(main())
