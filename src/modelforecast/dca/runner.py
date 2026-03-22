"""DCA poll runner — orchestrates exchange polling with checkpoints."""

from __future__ import annotations

import os
import time
from datetime import datetime, timedelta, timezone

from modelforecast.dca.clients.base import CEXClient
from modelforecast.dca.clients.coinbase import CoinbaseClient
from modelforecast.dca.clients.kraken import KrakenClient
from modelforecast.dca.models import PollCheckpoint
from modelforecast.dca.persistence import CheckpointManager, DCALogger

# Default lookback for first poll (no checkpoint)
DEFAULT_LOOKBACK_DAYS = 90


def poll_exchange(
    client: CEXClient,
    checkpoint_mgr: CheckpointManager,
    logger: DCALogger,
) -> int:
    """Poll a single exchange. Returns number of new records."""
    since = checkpoint_mgr.last_poll_time(client.exchange)
    if since is None:
        since = datetime.now(timezone.utc) - timedelta(days=DEFAULT_LOOKBACK_DAYS)

    records = client.poll_buys(since)
    for record in records:
        logger.log(record)

    if records:
        latest_ts = max(r.timestamp for r in records)
        latest_tx = records[-1].tx_id
    else:
        latest_ts = datetime.now(timezone.utc).isoformat()
        latest_tx = None

    checkpoint_mgr.update(PollCheckpoint(
        exchange=client.exchange,
        last_poll_utc=latest_ts,
        last_tx_id=latest_tx,
    ))

    return len(records)


def build_clients() -> list[CEXClient]:
    """Build exchange clients from available env vars."""
    clients: list[CEXClient] = []
    if os.environ.get("COINBASE_API_KEY"):
        clients.append(CoinbaseClient())
    if os.environ.get("KRAKEN_API_KEY"):
        clients.append(KrakenClient())
    return clients


def poll_all() -> dict[str, int]:
    """Poll all configured exchanges. Returns {exchange: new_record_count}."""
    logger = DCALogger()
    checkpoint_mgr = CheckpointManager()
    clients = build_clients()

    if not clients:
        raise RuntimeError(
            "No exchange API keys configured. "
            "Set COINBASE_API_KEY/COINBASE_API_SECRET and/or KRAKEN_API_KEY/KRAKEN_API_SECRET"
        )

    results = {}
    for client in clients:
        try:
            count = poll_exchange(client, checkpoint_mgr, logger)
            results[client.exchange] = count
        except Exception as e:
            results[client.exchange] = -1
            print(f"[{client.exchange}] poll failed: {e}")
        finally:
            client.close()
        # Rate limit between exchanges
        time.sleep(1)

    logger.close()
    return results
