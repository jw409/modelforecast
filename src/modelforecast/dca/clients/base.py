"""Abstract CEX client protocol."""

from __future__ import annotations

from datetime import datetime
from typing import Protocol

from modelforecast.dca.models import DCARecord


class CEXClient(Protocol):
    """Protocol for CEX API clients."""

    exchange: str

    def poll_buys(self, since: datetime) -> list[DCARecord]:
        """Fetch buy orders since the given timestamp."""
        ...

    def get_deposit_history(self, since: datetime) -> list[dict]:
        """Fetch deposit/withdrawal history."""
        ...
