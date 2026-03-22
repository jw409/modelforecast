"""JSONL logging and checkpoint management for DCA tracker."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from modelforecast.dca.models import DCARecord, PollCheckpoint

DEFAULT_VAR = Path(__file__).resolve().parents[3] / "var" / "dca"


class DCALogger:
    """Append-only JSONL writer for DCA events."""

    def __init__(self, var_dir: Path = DEFAULT_VAR):
        self.var_dir = var_dir
        self.var_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.var_dir / "events.jsonl"
        self._fh = open(self.path, "a")  # noqa: SIM115

    def log(self, record: DCARecord) -> None:
        self._fh.write(record.to_json() + "\n")
        self._fh.flush()

    def read_all(self) -> list[DCARecord]:
        if not self.path.exists():
            return []
        records = []
        for line in self.path.read_text().splitlines():
            line = line.strip()
            if line:
                records.append(DCARecord.from_json(line))
        return records

    def close(self) -> None:
        self._fh.close()


class CheckpointManager:
    """Per-exchange checkpoint for resume-from-last-poll."""

    def __init__(self, var_dir: Path = DEFAULT_VAR):
        self.var_dir = var_dir
        self.var_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.var_dir / "checkpoint.json"
        self._data: dict[str, dict] = {}
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            self._data = json.loads(self.path.read_text())

    def _save(self) -> None:
        self.path.write_text(json.dumps(self._data, indent=2) + "\n")

    def get(self, exchange: str) -> PollCheckpoint | None:
        if exchange in self._data:
            return PollCheckpoint.from_dict(self._data[exchange])
        return None

    def update(self, checkpoint: PollCheckpoint) -> None:
        self._data[checkpoint.exchange] = checkpoint.to_dict()
        self._save()

    def last_poll_time(self, exchange: str) -> datetime | None:
        cp = self.get(exchange)
        if cp is None:
            return None
        return datetime.fromisoformat(cp.last_poll_utc)
