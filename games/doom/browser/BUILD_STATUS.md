# DOOM Browser WASM Build Status

**Date**: 2025-12-06
**Status**: Working (with minor config issues)

## What Works
- DOOM renders correctly in browser at localhost:9999
- Game starts automatically on page load
- No "PLAYER 4 LEFT THE GAME" phantom player message (fixed in d_net.c)
- Menu navigation works
- 3D rendering works
- Mouse look works (horizontal turning)
- Ctrl fires, Space opens doors
- Left mouse click fires

## Current Issues
- WASD config not loading reliably - config file loading is inconsistent
- Number keys 1-7 for weapon switching may not work

## Build Process

### Prerequisites
```bash
# Emscripten SDK
source /home/jw/emsdk/emsdk_env.sh
```

### Build Commands
```bash
cd /home/jw/dev/modelforecast/games/doom/browser/wasm

# Configure (if needed)
emconfigure ./configure

# Build
make -j4

# Copy to dist
cp src/websockets-doom.js ../dist/
cp src/websockets-doom.wasm ../dist/
```

### Run Dev Server
```bash
cd /home/jw/dev/modelforecast/games/doom/browser/dist
python3 -m http.server 9999
# Open http://localhost:9999/
```

## Key Files

### Source Modifications
- `wasm/src/doom/d_net.c:73-81` - Added netgame check to prevent phantom player quit messages
- `wasm/configure.ac` - Emscripten build flags (SAFE_HEAP=0, ASYNCIFY, etc.)

### Dist Files
- `dist/index.html` - Simplified HTML with autoload
- `dist/websockets-doom.js` - Emscripten JS glue
- `dist/websockets-doom.wasm` - DOOM WASM binary (~7.6MB)
- `dist/doom1.wad` - Shareware WAD
- `dist/default.cfg` - Key bindings config

## Config Values (default.cfg)

Key bindings use internal DOOM key codes:
- `key_up=119` ('w'), `key_down=115` ('s'), `key_strafeleft=97` ('a'), `key_straferight=100` ('d')
- `key_fire=157` (KEY_RCTRL), `key_use=32` (space)
- `key_weapon1-8=49-56` ('1'-'8')
- `key_right=174` (KEY_RIGHTARROW), `key_left=172` (KEY_LEFTARROW)
- `mouseb_fire=0` (left click)
- `novert=1` (disable mouse forward/backward movement)

## Emscripten Flags (configure.ac)
```
EMFLAGS="-s INVOKE_RUN=1 -s USE_SDL=2 -s USE_SDL_MIXER=2 -s LEGACY_GL_EMULATION=0
-s USE_SDL_NET=2 -s ASSERTIONS=0 -s WASM=1 -s ALLOW_MEMORY_GROWTH=1
-s FORCE_FILESYSTEM=1 -s EXPORTED_RUNTIME_METHODS=[['FS','ccall','callMain']]
-s EXPORTED_FUNCTIONS=[['_main','_malloc','_free']] -s SAFE_HEAP=0
-s EXIT_RUNTIME=0 -s STACK_OVERFLOW_CHECK=0 -s PROXY_POSIX_SOCKETS=0
-s USE_PTHREADS=0 -s PROXY_TO_PTHREAD=0 -s INITIAL_MEMORY=64MB
-s ERROR_ON_UNDEFINED_SYMBOLS=0 -s ASYNCIFY -O3"
```

## Next Steps
1. Debug why config file isn't being loaded consistently
2. Test on GitHub Pages deployment
3. Consider switching to vanilla Chocolate Doom (non-websockets fork) for cleaner single-player
