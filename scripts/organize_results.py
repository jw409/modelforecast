#!/usr/bin/env python3
"""Organize results by provider."""

import subprocess
from pathlib import Path

results_dir = Path('results')

for f in results_dir.glob('*.json'):
    # Extract provider from filename (first part before _)
    provider = f.stem.split('_')[0]

    # Skip summary files
    if provider in ('summary', 'grok', 'phase3'):
        continue

    # Create provider directory
    provider_dir = results_dir / provider
    provider_dir.mkdir(exist_ok=True)

    # Git mv the file
    new_path = provider_dir / f.name
    subprocess.run(['git', 'mv', str(f), str(new_path)], check=True)
    print(f'{f.name} -> {provider}/')
