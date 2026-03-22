"""Kraken REST API client."""

from __future__ import annotations

import base64
import hashlib
import hmac
import os
import time
import urllib.parse
from datetime import datetime, timezone

import httpx

from modelforecast.dca.models import DCARecord

BASE_URL = "https://api.kraken.com"


class KrakenClient:
    """Kraken REST API — trade history for DCA tracking."""

    exchange = "kraken"

    def __init__(
        self,
        api_key: str | None = None,
        api_secret: str | None = None,
    ):
        self.api_key = api_key or os.environ["KRAKEN_API_KEY"]
        self.api_secret = api_secret or os.environ["KRAKEN_API_SECRET"]
        self._client = httpx.Client(base_url=BASE_URL, timeout=30)

    def _sign(self, path: str, data: dict) -> dict[str, str]:
        nonce = str(int(time.time() * 1000))
        data["nonce"] = nonce
        post_data = urllib.parse.urlencode(data)
        encoded = (nonce + post_data).encode()
        message = path.encode() + hashlib.sha256(encoded).digest()
        signature = hmac.new(
            base64.b64decode(self.api_secret),
            message,
            hashlib.sha512,
        )
        return {
            "API-Key": self.api_key,
            "API-Sign": base64.b64encode(signature.digest()).decode(),
            "Content-Type": "application/x-www-form-urlencoded",
        }

    def _post(self, path: str, data: dict | None = None) -> dict:
        data = data or {}
        headers = self._sign(path, data)
        resp = self._client.post(path, headers=headers, data=data)
        resp.raise_for_status()
        result = resp.json()
        if result.get("error"):
            raise RuntimeError(f"Kraken API error: {result['error']}")
        return result.get("result", {})

    def poll_buys(self, since: datetime) -> list[DCARecord]:
        """Fetch trade history since timestamp."""
        start_ts = int(since.timestamp())
        result = self._post("/0/private/TradesHistory", {"start": str(start_ts)})
        trades = result.get("trades", {})
        records = []
        for tx_id, trade in trades.items():
            record = self._parse_trade(tx_id, trade)
            if record.side == "buy":
                records.append(record)
        return records

    def get_deposit_history(self, since: datetime) -> list[dict]:
        """Fetch deposit history."""
        start_ts = int(since.timestamp())
        result = self._post("/0/private/DepositStatus", {"start": str(start_ts)})
        # DepositStatus returns a list
        if isinstance(result, list):
            return result
        return []

    def _parse_trade(self, tx_id: str, trade: dict) -> DCARecord:
        pair = trade.get("pair", "UNKNOWN")
        price = float(trade.get("price", 0))
        vol = float(trade.get("vol", 0))
        cost = float(trade.get("cost", 0))
        fee = float(trade.get("fee", 0))
        ts = float(trade.get("time", 0))

        return DCARecord(
            exchange="kraken",
            timestamp=datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
            symbol=pair,
            side=trade.get("type", "buy"),
            amount_fiat=cost,
            amount_crypto=vol,
            price=price,
            fee_fiat=fee,
            fee_crypto=0.0,
            tx_id=tx_id,
            status="filled",
            raw=trade,
        )

    def close(self) -> None:
        self._client.close()
