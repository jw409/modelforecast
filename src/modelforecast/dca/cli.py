"""CLI for DCA tracker."""

from __future__ import annotations

import sys
from collections import defaultdict
from datetime import datetime

from rich.console import Console
from rich.table import Table

from modelforecast.dca.persistence import DCALogger
from modelforecast.dca.runner import poll_all

console = Console()


def cmd_poll() -> None:
    """Poll all configured exchanges."""
    console.print("[bold]Polling exchanges...[/bold]")
    results = poll_all()
    for exchange, count in results.items():
        if count < 0:
            console.print(f"  {exchange}: [red]FAILED[/red]")
        else:
            console.print(f"  {exchange}: [green]{count} new records[/green]")


def cmd_history() -> None:
    """Show DCA history as a table."""
    logger = DCALogger()
    records = logger.read_all()
    if not records:
        console.print("[dim]No DCA records found.[/dim]")
        return

    table = Table(title="DCA History")
    table.add_column("Date", style="dim")
    table.add_column("Exchange")
    table.add_column("Symbol")
    table.add_column("Side")
    table.add_column("Fiat", justify="right")
    table.add_column("Crypto", justify="right")
    table.add_column("Price", justify="right")
    table.add_column("Fee", justify="right")

    for r in sorted(records, key=lambda x: x.timestamp):
        table.add_row(
            r.timestamp[:10],
            r.exchange,
            r.symbol,
            r.side,
            f"${r.amount_fiat:.2f}",
            f"{r.amount_crypto:.6f}",
            f"${r.price:.2f}",
            f"${r.fee_fiat:.2f}",
        )

    console.print(table)


def cmd_summary() -> None:
    """Aggregate DCA stats per symbol."""
    logger = DCALogger()
    records = logger.read_all()
    if not records:
        console.print("[dim]No DCA records found.[/dim]")
        return

    # Aggregate by symbol
    by_symbol: dict[str, dict] = defaultdict(lambda: {
        "total_fiat": 0.0,
        "total_crypto": 0.0,
        "total_fees": 0.0,
        "count": 0,
        "first": None,
        "last": None,
    })

    for r in records:
        if r.side != "buy":
            continue
        s = by_symbol[r.symbol]
        s["total_fiat"] += r.amount_fiat
        s["total_crypto"] += r.amount_crypto
        s["total_fees"] += r.fee_fiat
        s["count"] += 1
        if s["first"] is None or r.timestamp < s["first"]:
            s["first"] = r.timestamp
        if s["last"] is None or r.timestamp > s["last"]:
            s["last"] = r.timestamp

    table = Table(title="DCA Summary")
    table.add_column("Symbol")
    table.add_column("Buys", justify="right")
    table.add_column("Total Invested", justify="right")
    table.add_column("Total Crypto", justify="right")
    table.add_column("Avg Price", justify="right")
    table.add_column("Total Fees", justify="right")
    table.add_column("Period")

    for symbol, s in sorted(by_symbol.items()):
        avg = s["total_fiat"] / s["total_crypto"] if s["total_crypto"] else 0
        period = f"{s['first'][:10]} → {s['last'][:10]}" if s["first"] else "—"
        table.add_row(
            symbol,
            str(s["count"]),
            f"${s['total_fiat']:.2f}",
            f"{s['total_crypto']:.6f}",
            f"${avg:.2f}",
            f"${s['total_fees']:.2f}",
            period,
        )

    console.print(table)


def main() -> None:
    if len(sys.argv) < 2:
        console.print("Usage: python -m modelforecast.dca <poll|history|summary>")
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd == "poll":
        cmd_poll()
    elif cmd == "history":
        cmd_history()
    elif cmd == "summary":
        cmd_summary()
    else:
        console.print(f"Unknown command: {cmd}")
        sys.exit(1)
