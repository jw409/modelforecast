# DOOM Arena: GPU Recording Export & Browser Playback

## Status: IMPLEMENTED (Dec 5, 2025)

### Completed
1. **Interest Detection** - GPU tracks near-death, multi-kill, death events
2. **JSON Export** - `--export-scenarios` flag exports to browser/dist/scenarios/
3. **Demo Conversion** - Python script converts JSON to .lmp DOOM format
4. **Browser Loading** - UI dynamically loads scenarios from index.json

### Quick Test
```bash
cd /home/jw/dev/modelforecast/games/doom/gpu
./build/doom_sim --export-scenarios ../browser/dist 10000 500
cd ../browser/dist && python3 -m http.server 8080
# Open http://localhost:8080
```

---

## Original Context

We have a working GPU DOOM simulator that runs 200,000 parallel instances at 500x realtime. Now we need to:

1. **Export interesting moments as recordings** from the GPU simulator
2. **Play them back in the browser** with "take control" functionality

## Current State

### GPU Side (Working)
- `/home/jw/dev/modelforecast/games/doom/gpu/doom_sim` - Runs 200k instances
- Already has checkpointing: `WriteCheckpoint_GPU()` saves state every 35 ticks
- Outputs: player position, health, kills, monsters killed
- Test: `./build/doom_sim 100000 1000` runs 100k instances for 1000 ticks

### Browser Side (Partially Working)
- `/home/jw/dev/modelforecast/games/doom/browser/dist/index.html` - Uses js-dos CDN
- Has UI for scenario selection and "Take Control" button
- Server: `cd browser/dist && python3 -m http.server 8080`

## Task 1: GPU Recording Export

Modify `doom_sim.cu` to detect and export "interesting moments":

### Interesting Moment Criteria
```cpp
// In doom_main.cu, add detection logic:
bool is_interesting(int instance_id) {
    // Near-death escape: health dropped below 20, now above 50
    // Multi-kill: 3+ kills in last 5 seconds (175 ticks)
    // Death: player died (for "can you survive?" challenges)
    // Secret found: discovered secret area
    // Speed record: fastest clear of section
}
```

### Recording Format (JSON)
```json
{
    "scenario_id": "near-death-e1m1-42",
    "type": "near_death",
    "description": "Survived imp ambush with 3 HP",
    "level": "E1M1",
    "tick": 847,
    "interest_score": 0.92,
    "checkpoint": {
        "player": {"x": 1056, "y": -3616, "z": 0, "angle": 90, "health": 45, "armor": 0},
        "monsters": [{"type": "IMP", "x": 1200, "y": -3500, "health": 60}, ...]
    },
    "input_history": [
        {"tick": 0, "forward": 50, "side": 0, "angle": 0, "buttons": 0},
        {"tick": 1, "forward": 50, "side": 0, "angle": 0, "buttons": 1},
        ...
    ]
}
```

### Export Code Location
Add to `doom_main.cu` after the simulation loop:
```cpp
void ExportInterestingMoments(const char* output_dir, int num_instances) {
    // Scan all instances for interesting moments
    // Write JSON files to output_dir/scenarios/
}
```

### Files to Modify
- `gpu/doom_main.cu` - Add interest detection + JSON export
- `gpu/doom_types.cuh` - Add InterestType enum if needed
- `gpu/Makefile` - Ensure build works

## Task 2: Browser Playback

Wire the exported recordings to the browser player.

### Approach A: Demo Lump (Preferred)
DOOM has native demo recording format (.lmp files). We can:
1. Convert our TicCmd history to .lmp format
2. Load it in js-dos with `-playdemo`
3. User watches, presses key to "take control"

### Approach B: JavaScript Input Injection
1. Load scenario JSON in browser
2. Feed TicCmds to js-dos programmatically
3. On "Take Control", switch to real keyboard input

### Files to Create/Modify
- `browser/dist/scenarios/` - Directory for exported scenarios
- `browser/dist/index.html` - Wire scenario loading
- `browser/scripts/convert_to_demo.py` - Convert JSON to .lmp format

## Key Files Reference

```
games/doom/
├── gpu/
│   ├── doom_main.cu          # Main GPU simulator (MODIFY)
│   ├── doom_types.cuh        # Type definitions
│   ├── doom_data.cuh         # Canonical DOOM tables
│   └── build/doom_sim        # Built binary
├── browser/
│   ├── dist/
│   │   ├── index.html        # Browser player (MODIFY)
│   │   └── scenarios/        # Exported recordings (CREATE)
│   └── scripts/
│       └── convert_to_demo.py # JSON to .lmp (CREATE)
└── wads/
    └── doom1.wad             # Shareware WAD
```

## Commands

```bash
# Build GPU simulator
cd /home/jw/dev/modelforecast/games/doom/gpu
make doom_main

# Run with recording export
./build/doom_sim 10000 1000 --export-scenarios ../browser/dist/scenarios/

# Start browser server
cd /home/jw/dev/modelforecast/games/doom/browser/dist
python3 -m http.server 8080

# Open http://localhost:8080
```

## Success Criteria

1. Running `doom_sim --export-scenarios` produces JSON files in `scenarios/`
2. Browser loads scenario list from `scenarios/index.json`
3. Clicking a scenario plays back the recording
4. "Take Control" button switches to live input
5. Works on GitHub Pages (static hosting, no server)

## Demo Recording Format (.lmp)

DOOM demo format for reference:
```
Byte 0: Version (e.g., 109 for DOOM 1.9)
Byte 1: Skill (0-4)
Byte 2: Episode (1-4)
Byte 3: Map (1-9)
Bytes 4+: TicCmds (4 bytes each: forwardmove, sidemove, angleturn, buttons)
End: 0x80 byte terminates
```

## Notes

- The GPU simulator uses struct-of-arrays layout for coalesced memory access
- TicCmd format: 8 bytes (forwardmove, sidemove, angleturn, consistency, chatchar, buttons)
- js-dos can load .lmp demos with command line args
- For "take control", we may need to modify js-dos or use keyboard event injection
