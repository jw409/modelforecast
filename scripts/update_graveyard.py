#!/usr/bin/env python3
"""Update GRAVEYARD.md with models no longer available on OpenRouter.

Usage:
    uv run python scripts/update_graveyard.py --known vendor/model:free vendor/other:free
    uv run python scripts/update_graveyard.py --known-file scripts/previous_models.txt
"""
import argparse
import re
import sys
from datetime import date
from pathlib import Path

# Allow running from project root without installing
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

GRAVEYARD_PATH = Path(__file__).parent.parent / "GRAVEYARD.md"
GRAVEYARD_TABLE_HEADER = "## Graveyard"


def load_graveyard_model_ids(graveyard_text: str) -> set[str]:
    """Extract model IDs already in the graveyard table."""
    # Model IDs appear as: | `vendor/model:free` |
    return set(re.findall(r"\| `([^`]+)` \|", graveyard_text))


def append_grave(graveyard_path: Path, model_id: str, today: str) -> None:
    """Append one row to the Graveyard table in GRAVEYARD.md."""
    row = f"| `{model_id}` | {today} | {today} | Removed from OpenRouter free tier |\n"
    text = graveyard_path.read_text()
    # Insert before the final newline, after the last table row
    graveyard_path.write_text(text.rstrip("\n") + "\n" + row)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Detect and record models removed from OpenRouter free tier"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--known",
        nargs="+",
        metavar="MODEL_ID",
        help="Model IDs to check",
    )
    group.add_argument(
        "--known-file",
        type=Path,
        metavar="FILE",
        help="File with one model ID per line",
    )
    args = parser.parse_args()

    # Load known model list
    if args.known:
        known_models = args.known
    else:
        if not args.known_file.exists():
            print(f"ERROR: File not found: {args.known_file}")
            return 1
        known_models = [
            line.strip()
            for line in args.known_file.read_text().splitlines()
            if line.strip() and not line.startswith("#")
        ]

    # Fetch live roster
    from modelforecast.models import get_available_models
    try:
        live_models = get_available_models()
    except Exception as e:
        print(f"ERROR fetching live models: {e}")
        return 1

    # Load current graveyard
    if not GRAVEYARD_PATH.exists():
        print(f"ERROR: GRAVEYARD.md not found at {GRAVEYARD_PATH}")
        return 1
    graveyard_text = GRAVEYARD_PATH.read_text()
    already_buried = load_graveyard_model_ids(graveyard_text)

    today = date.today().isoformat()
    new_graves = 0
    skipped = 0

    for model_id in known_models:
        if model_id in live_models:
            continue  # Still alive
        if model_id in already_buried:
            skipped += 1
            continue  # Already in graveyard
        append_grave(GRAVEYARD_PATH, model_id, today)
        print(f"  BURIED: {model_id}")
        new_graves += 1

    print(
        f"\n{len(known_models)} models checked — "
        f"{new_graves} new graves added, {skipped} already recorded"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
