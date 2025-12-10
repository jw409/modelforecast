# DOOM Verification Mode - CPU Reference Build

## Overview

This is a headless, instrumented build of DOOM 1.10 for CPU vs GPU verification testing. It outputs game state as JSONL to stdout after each tick.

## Build Instructions

```bash
cd /home/jw/dev/modelforecast/games/doom/source/linuxdoom-1.10
make doom_reference
```

Binary output: `verify/doom_reference`

## Modified Files

### 1. **p_tick.c**
- Added `LogGameState()` function that outputs JSONL after `P_Ticker()`
- JSONL format: `{"tick":N,"player":N,"x":fixed_t,"y":fixed_t,"z":fixed_t,"angle":uint32,"health":int,"armor":int,"kills":int,"alive":bool}`
- Guarded by `#ifdef VERIFICATION_MODE`

### 2. **i_video.c**
- Wrapped all X11/graphics code in `#ifndef HEADLESS`
- Added headless stubs for all `I_*` video functions
- Fixed `errnos.h` → `errno.h` typo
- Stubs: `I_InitGraphics`, `I_ShutdownGraphics`, `I_StartFrame`, `I_FinishUpdate`, etc.

### 3. **i_sound.c**
- Wrapped soundcard includes in `#ifndef HEADLESS`
- Added headless stubs for all sound functions
- Stubs include: `I_InitSound`, `I_UpdateSound`, `I_UpdateSoundParams`, `I_StartSound`, etc.

### 4. **m_misc.c**
- Wrapped `defaults[]` array in `#ifndef HEADLESS` to avoid GCC const initializer errors
- Added minimal headless defaults array (3 entries vs 41)
- Avoids string→int pointer cast issues in modern GCC

### 5. **Makefile**
- Added `doom_reference` target
- Separate `VERIFY_CFLAGS` with `-DVERIFICATION_MODE -DHEADLESS`
- Uses `verify/` directory for object files (not `linux/`)
- No X11 libraries needed (`VERIFY_LIBS=-lm` only)

## Compilation Defines

- **VERIFICATION_MODE**: Enables state logging in p_tick.c
- **HEADLESS**: Disables graphics/sound, stubs I/O functions
- **NORMALUNIX**: Standard DOOM Unix build
- **LINUX**: Linux-specific code paths

## State Output Format

Each game tick outputs one JSON line per active player:

```json
{"tick":100,"player":0,"x":1234567,"y":7654321,"z":0,"angle":2147483648,"health":100,"armor":0,"kills":5,"alive":true}
```

Fields:
- `tick`: Game tic counter (leveltime)
- `player`: Player index (0-3)
- `x`, `y`, `z`: Fixed-point position (divide by 65536 for integer coords)
- `angle`: Angle as uint32 (0-4294967295, where 0=East, rotating CCW)
- `health`: Player health
- `armor`: Armor points
- `kills`: Monster kill count
- `alive`: Boolean (player state == PST_LIVE)

## Next Steps (TicCmd Input)

**NOT YET IMPLEMENTED:**
- Reading TicCmd from stdin (binary format)
- Would need to modify g_game.c or d_net.c to inject commands
- Binary format matches `ticcmd_t` struct (8 bytes):
  ```c
  typedef struct {
      char forwardmove;   // 1 byte
      char sidemove;      // 1 byte
      short angleturn;    // 2 bytes
      short consistancy;  // 2 bytes (unused for verification)
      byte chatchar;      // 1 byte (unused)
      byte buttons;       // 1 byte
  } ticcmd_t;
  ```

## Usage Example (Future)

```bash
# Generate TicCmd sequence
./generate_ticcmds.py > input.bin

# Run doom_reference
./verify/doom_reference -iwad doom1.wad -skill 1 -episode 1 -map 1 \
  < input.bin > state_cpu.jsonl 2>stderr.log
```

## Verification Notes

- State output happens AFTER `P_Ticker()` completes
- Output is to stdout, buffered and flushed per tick
- All rendering/sound/input is stubbed out
- Should be deterministic for same TicCmd sequence
- Suitable for comparing against GPU DOOM implementation
