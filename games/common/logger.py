"""
AI Arena Session Logger

Logs EVERYTHING for replay and analysis:
- LLM decisions and reasoning
- Game state changes
- Code modifications (cheating!)
- Battle outcomes

All logged as JSONL for streaming analysis.
"""

import json
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Any, Optional
import uuid

EventType = Literal[
    "session_start",
    "session_end",
    "llm_decision",
    "llm_code_submit",
    "game_state",
    "battle_result",
    "cheat_detected",
    "error",
]

LOGS_DIR = Path(__file__).parent.parent / "logs"


@dataclass
class GameEvent:
    """Single event in game session log."""
    event_type: EventType
    timestamp: str
    session_id: str
    contestant_id: str
    game: str
    data: dict

    def to_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"))


class SessionLogger:
    """
    Log everything about an LLM game session.

    Usage:
        logger = SessionLogger("corewars", "gpt-5")
        logger.log_decision("attack", {"target": "memory_0x100", "reason": "..."})
        logger.log_code_submit("mov 0, 1", language="redcode")
        logger.close()
    """

    def __init__(self, game: str, contestant_id: str, session_id: Optional[str] = None):
        self.game = game
        self.contestant_id = contestant_id
        self.session_id = session_id or f"{game}-{contestant_id}-{uuid.uuid4().hex[:8]}"
        self.start_time = datetime.now(timezone.utc)

        # Ensure logs directory exists
        LOGS_DIR.mkdir(parents=True, exist_ok=True)

        # Open log file
        date_str = self.start_time.strftime("%Y-%m-%d")
        self.log_path = LOGS_DIR / f"{date_str}_{self.session_id}.jsonl"
        self.log_file = open(self.log_path, "a")

        # Log session start
        self._log("session_start", {
            "game": game,
            "contestant": contestant_id,
            "start_time": self.start_time.isoformat(),
        })

    def _timestamp(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _log(self, event_type: EventType, data: dict):
        event = GameEvent(
            event_type=event_type,
            timestamp=self._timestamp(),
            session_id=self.session_id,
            contestant_id=self.contestant_id,
            game=self.game,
            data=data,
        )
        self.log_file.write(event.to_json() + "\n")
        self.log_file.flush()

    def log_decision(self, action: str, reasoning: dict):
        """Log an LLM's decision and reasoning."""
        self._log("llm_decision", {
            "action": action,
            "reasoning": reasoning,
        })

    def log_code_submit(self, code: str, language: str, purpose: str = ""):
        """Log when LLM submits custom code (potential cheat!)."""
        self._log("llm_code_submit", {
            "code": code,
            "language": language,
            "purpose": purpose,
            "code_length": len(code),
            "code_hash": hex(hash(code) & 0xFFFFFFFF),
        })

    def log_game_state(self, state: dict):
        """Log game state snapshot."""
        self._log("game_state", state)

    def log_battle_result(self, winner: str, loser: str, details: dict):
        """Log battle outcome."""
        self._log("battle_result", {
            "winner": winner,
            "loser": loser,
            **details,
        })

    def log_cheat_detected(self, cheat_type: str, evidence: dict):
        """Log detected cheating attempt."""
        self._log("cheat_detected", {
            "cheat_type": cheat_type,
            "evidence": evidence,
        })

    def log_error(self, error: str, context: dict = None):
        """Log error during session."""
        self._log("error", {
            "error": error,
            "context": context or {},
        })

    def close(self):
        """End session and close log."""
        self._log("session_end", {
            "end_time": self._timestamp(),
            "duration_sec": (datetime.now(timezone.utc) - self.start_time).total_seconds(),
        })
        self.log_file.close()


def read_session_log(log_path: Path) -> list[GameEvent]:
    """Read all events from a session log."""
    events = []
    with open(log_path) as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                events.append(GameEvent(**data))
    return events


def get_recent_sessions(n: int = 10) -> list[Path]:
    """Get most recent session logs."""
    logs = sorted(LOGS_DIR.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    return logs[:n]
