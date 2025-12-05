# DOOM Arena - Browser Playback

**Watch AI play DOOM, or take over yourself.**

## Git Submodules

This directory uses git submodules for external dependencies:

```bash
# Initialize submodules (run from repo root)
git submodule update --init --recursive
```

- **wasm/**: [cloudflare/doom-wasm](https://github.com/cloudflare/doom-wasm) - Chocolate Doom WASM port
- **../wads/**: [Akbar30Bill/DOOM_wads](https://github.com/Akbar30Bill/DOOM_wads) - Game data files

## Architecture

```
GPU (fast)                    GitHub Pages (playable)
    │                              │
    ▼                              ▼
Run 52,000× speed         Load .sav, play in browser
    │                              │
    ▼                              ▼
Find "interesting" moments   Chocolate Doom WASM
    │                              │
    ▼                              ▼
Export top 10 checkpoints    Human takes over
    │                              │
    └──────── git push ────────────┘
```

## The "Interesting" Detector

Not all moments are worth publishing. We rank by:

1. **Near-death escapes** - Health drops to <20, then recovers
2. **Multi-kills** - 3+ enemies in 5 seconds
3. **Boss encounters** - Cyberdemon, Spider Mastermind
4. **Speed records** - Fastest clear of a section
5. **Creative solutions** - Unusual paths, infighting triggers
6. **Dramatic deaths** - Spectacular failures

## Directory Structure

```
browser/
├── wasm/                    # Cloudflare doom-wasm (Chocolate Doom)
├── checkpoints/             # Top 10 .sav files
│   ├── 001-near-death.sav
│   ├── 002-cyberdemon.sav
│   └── ...
├── pages/                   # GitHub Pages site
│   ├── index.html           # Checkpoint selector
│   ├── play.html            # DOOM player with cheats
│   └── assets/
│       ├── doom.wasm
│       ├── doom.js
│       └── doom1.wad        # Shareware WAD (legal)
└── scripts/
    ├── rank_checkpoints.py  # Find interesting moments
    └── publish.py           # Build GitHub Pages
```

## Usage

```bash
# 0. Initialize submodules (first time only)
git submodule update --init --recursive

# 1. Build DOOM WASM (requires emscripten)
./scripts/build_wasm.sh

# 2. Run GPU simulation, export checkpoints
cd ../gpu
make
./build/gpu_doom_test 100 10000 35 --export-checkpoints

# 3. Rank and select top 10
python scripts/rank_checkpoints.py ../gpu/checkpoints/ --top 10 --json > ranked.json

# 4. Build GitHub Pages
python scripts/publish.py \
  --ranked ranked.json \
  --checkpoints ../gpu/checkpoints/ \
  --output dist/

# 5. Push to GitHub
git add dist/
git commit -m "Add AI DOOM highlights"
git push origin main
```

## Cheat Menu (in browser)

When playing a checkpoint:
- **IDKFA** - All keys, weapons, full ammo
- **IDDQD** - God mode
- **IDCLIP** - No-clip (walk through walls)
- **Unlimited Ammo** - Never reload
- **BFG Mode** - Start with BFG 9000

## Legal

- DOOM shareware WAD (doom1.wad) is freely distributable
- Chocolate Doom is GPL
- Our GPU engine is MIT
- Cloudflare doom-wasm is GPL

For full game (doom.wad, doom2.wad), users must provide their own.
