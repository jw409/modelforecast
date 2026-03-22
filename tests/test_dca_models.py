"""Tests for DCA models and persistence."""

import json
import tempfile
from pathlib import Path

from modelforecast.dca.models import DCARecord, PollCheckpoint
from modelforecast.dca.persistence import CheckpointManager, DCALogger


def _sample_record(**overrides) -> DCARecord:
    defaults = dict(
        exchange="coinbase",
        timestamp="2026-03-21T12:00:00+00:00",
        symbol="ETH-USD",
        side="buy",
        amount_fiat=100.0,
        amount_crypto=0.05,
        price=2000.0,
        fee_fiat=1.50,
        fee_crypto=0.0,
        tx_id="abc-123",
        status="filled",
        raw={"order_id": "abc-123"},
    )
    defaults.update(overrides)
    return DCARecord(**defaults)


class TestDCARecord:
    def test_roundtrip_json(self):
        r = _sample_record()
        line = r.to_json()
        r2 = DCARecord.from_json(line)
        assert r2.exchange == "coinbase"
        assert r2.amount_fiat == 100.0
        assert r2.tx_id == "abc-123"
        assert r2.raw == {"order_id": "abc-123"}

    def test_json_compact(self):
        r = _sample_record()
        line = r.to_json()
        # No spaces in compact JSON
        assert " " not in line or '"' in line
        parsed = json.loads(line)
        assert parsed["symbol"] == "ETH-USD"


class TestPollCheckpoint:
    def test_roundtrip(self):
        cp = PollCheckpoint(
            exchange="kraken",
            last_poll_utc="2026-03-21T12:00:00+00:00",
            last_tx_id="xyz-789",
            cursor="next-page",
        )
        d = cp.to_dict()
        cp2 = PollCheckpoint.from_dict(d)
        assert cp2.exchange == "kraken"
        assert cp2.cursor == "next-page"

    def test_fresh(self):
        cp = PollCheckpoint.fresh("coinbase")
        assert cp.exchange == "coinbase"
        assert cp.last_tx_id is None
        assert "2026" in cp.last_poll_utc


class TestDCALogger:
    def test_log_and_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            logger = DCALogger(var_dir=Path(tmp))
            r1 = _sample_record(tx_id="tx-1")
            r2 = _sample_record(tx_id="tx-2", exchange="kraken")
            logger.log(r1)
            logger.log(r2)
            logger.close()

            logger2 = DCALogger(var_dir=Path(tmp))
            records = logger2.read_all()
            assert len(records) == 2
            assert records[0].tx_id == "tx-1"
            assert records[1].exchange == "kraken"

    def test_empty_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            # Don't create events.jsonl
            logger = DCALogger(var_dir=Path(tmp))
            logger.close()
            logger2 = DCALogger(var_dir=Path(tmp))
            assert logger2.read_all() == []


class TestCheckpointManager:
    def test_get_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr = CheckpointManager(var_dir=Path(tmp))
            assert mgr.get("nonexistent") is None

    def test_update_and_get(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr = CheckpointManager(var_dir=Path(tmp))
            cp = PollCheckpoint(
                exchange="coinbase",
                last_poll_utc="2026-03-21T12:00:00+00:00",
                last_tx_id="abc",
            )
            mgr.update(cp)

            cp2 = mgr.get("coinbase")
            assert cp2 is not None
            assert cp2.last_tx_id == "abc"

    def test_persistence_across_instances(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr1 = CheckpointManager(var_dir=Path(tmp))
            mgr1.update(PollCheckpoint(
                exchange="kraken",
                last_poll_utc="2026-03-21T00:00:00+00:00",
            ))

            mgr2 = CheckpointManager(var_dir=Path(tmp))
            cp = mgr2.get("kraken")
            assert cp is not None
            assert cp.exchange == "kraken"

    def test_last_poll_time(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr = CheckpointManager(var_dir=Path(tmp))
            assert mgr.last_poll_time("coinbase") is None

            mgr.update(PollCheckpoint(
                exchange="coinbase",
                last_poll_utc="2026-03-21T12:00:00+00:00",
            ))
            dt = mgr.last_poll_time("coinbase")
            assert dt is not None
            assert dt.year == 2026
