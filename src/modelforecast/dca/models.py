"""Data models for DCA tracking."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone


@dataclass
class DCARecord:
    """Single DCA buy/sell record from a CEX."""

    exchange: str  # coinbase, kraken
    timestamp: str  # ISO 8601 UTC
    symbol: str  # ETH-USD, BTC-USD
    side: str  # buy, sell
    amount_fiat: float  # USD spent/received
    amount_crypto: float  # crypto bought/sold
    price: float  # execution price
    fee_fiat: float  # fee in USD
    fee_crypto: float  # fee in crypto (if applicable)
    tx_id: str  # exchange transaction ID
    status: str  # filled, pending, cancelled
    raw: dict = field(default_factory=dict)  # full API response

    def to_json(self) -> str:
        d = asdict(self)
        return json.dumps(d, separators=(",", ":"))

    @classmethod
    def from_json(cls, line: str) -> DCARecord:
        d = json.loads(line)
        return cls(**d)


@dataclass
class PollCheckpoint:
    """Tracks last successful poll per exchange."""

    exchange: str
    last_poll_utc: str  # ISO 8601
    last_tx_id: str | None = None
    cursor: str | None = None  # pagination cursor

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> PollCheckpoint:
        return cls(**d)

    @classmethod
    def fresh(cls, exchange: str) -> PollCheckpoint:
        return cls(
            exchange=exchange,
            last_poll_utc=datetime.now(timezone.utc).isoformat(),
        )
