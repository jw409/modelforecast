# AI Arena

GPU-accelerated benchmarks for LLM tool-calling in game environments.

## Structure

```
arena/
├── angband/
│   ├── apwborg/          # APWBorg source (1.9MB, borg1-9.c + borg.txt)
│   │   ├── borg.txt      # 1400+ configurable parameters
│   │   ├── borg1.c       # Core data structures
│   │   ├── borg6.c       # Main decision engine (576KB)
│   │   ├── borg9.c       # Control loop, borg_think()
│   │   └── ...
│   └── harness/          # Python arena harness
│       ├── borg_config.py    # configure_borg() MCP tool
│       ├── test_borg_pty.py  # PTY-based automation
│       └── test_borg.exp     # Expect script
├── corewars/             # (TODO) CoreWars arena
└── README.md
```

## Phase 0: APWBorg Baseline

1. Build Angband 4.2.5 with borg enabled
2. Run default borg, collect metrics (depth, deaths, time)
3. Test `configure_borg()` tool interface

## Hardware Targets

- **Local**: RTX 5090 (when available)
- **Cloud**: Colab TPU v5e/v6e
- **API**: OpenRouter (baseline)

## Key Files

| File | Lines | Purpose |
|------|-------|---------|
| borg.txt | 1400+ | Configuration parameters |
| borg6.c | 12K | Decision engine, pathfinding |
| borg9.c | 4K | Main loop, `borg_think()` |
| borg_config.py | 291 | MCP tool interface |

## Tool Interface

```python
from arena.angband.harness.borg_config import configure_borg

result = configure_borg({
    "borg_worships_speed": True,
    "borg_plays_risky": True,
    "borg_no_deeper": 50
})
# Returns: ConfigResult with diff, backup_path
```
