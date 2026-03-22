"""Coinbase Advanced Trade API client."""

from __future__ import annotations

import hashlib
import hmac
import os
import time
from datetime import datetime, timezone

import httpx

from modelforecast.dca.models import DCARecord

BASE_URL = "https://api.coinbase.com"


class CoinbaseClient:
    """Coinbase Advanced Trade API — recurring buy history."""

    exchange = "coinbase"

    def __init__(
        self,
        api_key: str | None = None,
        api_secret: str | None = None,
    ):
        self.api_key = api_key or os.environ["COINBASE_API_KEY"]
        self.api_secret = api_secret or os.environ["COINBASE_API_SECRET"]
        self._client = httpx.Client(base_url=BASE_URL, timeout=30)

    def _sign(self, method: str, path: str, body: str = "") -> dict[str, str]:
        timestamp = str(int(time.time()))
        message = f"{timestamp}{method.upper()}{path}{body}"
        signature = hmac.new(
            self.api_secret.encode(),
            message.encode(),
            hashlib.sha256,
        ).hexdigest()
        return {
            "CB-ACCESS-KEY": self.api_key,
            "CB-ACCESS-SIGN": signature,
            "CB-ACCESS-TIMESTAMP": timestamp,
            "Content-Type": "application/json",
        }

    def _get(self, path: str, params: dict | None = None) -> dict:
        headers = self._sign("GET", path)
        resp = self._client.get(path, headers=headers, params=params)
        resp.raise_for_status()
        return resp.json()

    def poll_buys(self, since: datetime) -> list[DCARecord]:
        """Fetch filled buy orders since timestamp."""
        params = {
            "order_status": "FILLED",
            "start_date": since.isoformat(),
            "order_side": "BUY",
            "sort_by": "CREATED_TIME",
        }
        data = self._get("/api/v3/brokerage/orders/historical", params)
        records = []
        for order in data.get("orders", []):
            records.append(self._parse_order(order))
        return records

    def get_deposit_history(self, since: datetime) -> list[dict]:
        """Fetch account transactions (deposits/withdrawals)."""
        # Coinbase Advanced Trade: list accounts, then transactions per account
        accounts = self._get("/api/v3/brokerage/accounts")
        txns = []
        for acct in accounts.get("accounts", []):
            acct_id = acct.get("uuid", "")
            if not acct_id:
                continue
            try:
                resp = self._get(f"/v2/accounts/{acct_id}/transactions")
                for tx in resp.get("data", []):
                    created = tx.get("created_at", "")
                    if created and created >= since.isoformat():
                        txns.append(tx)
            except httpx.HTTPStatusError:
                continue  # skip accounts we can't read
        return txns

    def _parse_order(self, order: dict) -> DCARecord:
        product = order.get("product_id", "UNKNOWN")
        config = order.get("order_configuration", {})
        # Market orders use different config shapes
        filled_value = float(order.get("filled_value", 0))
        filled_size = float(order.get("filled_size", 0))
        total_fees = float(order.get("total_fees", 0))
        avg_price = filled_value / filled_size if filled_size else 0

        return DCARecord(
            exchange="coinbase",
            timestamp=order.get("created_time", datetime.now(timezone.utc).isoformat()),
            symbol=product,
            side=order.get("side", "BUY").lower(),
            amount_fiat=filled_value,
            amount_crypto=filled_size,
            price=avg_price,
            fee_fiat=total_fees,
            fee_crypto=0.0,
            tx_id=order.get("order_id", ""),
            status=order.get("status", "FILLED").lower(),
            raw=order,
        )

    def close(self) -> None:
        self._client.close()
