"""Tests for CEX API clients with mocked responses."""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from modelforecast.dca.clients.coinbase import CoinbaseClient
from modelforecast.dca.clients.kraken import KrakenClient


MOCK_COINBASE_ORDERS = {
    "orders": [
        {
            "order_id": "cb-order-1",
            "product_id": "ETH-USD",
            "side": "BUY",
            "status": "FILLED",
            "created_time": "2026-03-20T10:00:00Z",
            "filled_value": "100.00",
            "filled_size": "0.05",
            "total_fees": "1.50",
            "order_configuration": {},
        },
        {
            "order_id": "cb-order-2",
            "product_id": "BTC-USD",
            "side": "BUY",
            "status": "FILLED",
            "created_time": "2026-03-21T10:00:00Z",
            "filled_value": "200.00",
            "filled_size": "0.0023",
            "total_fees": "3.00",
            "order_configuration": {},
        },
    ]
}

MOCK_KRAKEN_TRADES = {
    "result": {
        "trades": {
            "TX-AAA": {
                "pair": "XETHZUSD",
                "type": "buy",
                "price": "2000.00",
                "vol": "0.05",
                "cost": "100.00",
                "fee": "0.26",
                "time": 1742558400.0,  # 2025-03-21T12:00:00Z
            },
            "TX-BBB": {
                "pair": "XETHZUSD",
                "type": "sell",  # should be filtered out
                "price": "2100.00",
                "vol": "0.02",
                "cost": "42.00",
                "fee": "0.11",
                "time": 1742558500.0,
            },
        }
    }
}


class TestCoinbaseClient:
    @pytest.fixture
    def client(self, monkeypatch):
        monkeypatch.setenv("COINBASE_API_KEY", "test-key")
        monkeypatch.setenv("COINBASE_API_SECRET", "test-secret")
        return CoinbaseClient()

    def test_parse_order(self, client):
        order = MOCK_COINBASE_ORDERS["orders"][0]
        record = client._parse_order(order)
        assert record.exchange == "coinbase"
        assert record.symbol == "ETH-USD"
        assert record.amount_fiat == 100.0
        assert record.amount_crypto == 0.05
        assert record.price == 2000.0
        assert record.fee_fiat == 1.50
        assert record.tx_id == "cb-order-1"

    def test_sign_produces_headers(self, client):
        headers = client._sign("GET", "/api/v3/brokerage/orders/historical")
        assert "CB-ACCESS-KEY" in headers
        assert "CB-ACCESS-SIGN" in headers
        assert "CB-ACCESS-TIMESTAMP" in headers
        assert headers["CB-ACCESS-KEY"] == "test-key"

    @patch("modelforecast.dca.clients.coinbase.CoinbaseClient._get")
    def test_poll_buys(self, mock_get, client):
        mock_get.return_value = MOCK_COINBASE_ORDERS
        since = datetime(2026, 3, 1, tzinfo=timezone.utc)
        records = client.poll_buys(since)
        assert len(records) == 2
        assert records[0].symbol == "ETH-USD"
        assert records[1].symbol == "BTC-USD"


class TestKrakenClient:
    @pytest.fixture
    def client(self, monkeypatch):
        monkeypatch.setenv("KRAKEN_API_KEY", "test-key")
        monkeypatch.setenv("KRAKEN_API_SECRET", "dGVzdC1zZWNyZXQ=")  # base64("test-secret")
        return KrakenClient()

    def test_parse_trade(self, client):
        trade = MOCK_KRAKEN_TRADES["result"]["trades"]["TX-AAA"]
        record = client._parse_trade("TX-AAA", trade)
        assert record.exchange == "kraken"
        assert record.symbol == "XETHZUSD"
        assert record.amount_fiat == 100.0
        assert record.amount_crypto == 0.05
        assert record.fee_fiat == 0.26
        assert record.side == "buy"

    @patch("modelforecast.dca.clients.kraken.KrakenClient._post")
    def test_poll_buys_filters_sells(self, mock_post, client):
        mock_post.return_value = MOCK_KRAKEN_TRADES["result"]
        since = datetime(2025, 1, 1, tzinfo=timezone.utc)
        records = client.poll_buys(since)
        # Only buys, not sells
        assert len(records) == 1
        assert records[0].side == "buy"
        assert records[0].tx_id == "TX-AAA"
