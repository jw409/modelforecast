#!/bin/bash
#
# Build DOOM WASM from cloudflare/doom-wasm submodule
#
# Prerequisites:
#   sudo apt-get install emscripten automake
#   # or on Mac: brew install emscripten automake sdl2 sdl2_mixer sdl2_net
#

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WASM_DIR="$SCRIPT_DIR/../wasm"
BUILD_DIR="$WASM_DIR/build"
WAD_DIR="$SCRIPT_DIR/../../wads"

echo "=== DOOM WASM Build Script ==="

# Check emscripten
if ! command -v emcc &> /dev/null; then
    echo "ERROR: emscripten not found"
    echo ""
    echo "Install with:"
    echo "  Ubuntu/Debian: sudo apt-get install emscripten"
    echo "  Mac: brew install emscripten"
    echo "  Or: git clone https://github.com/emscripten-core/emsdk.git && cd emsdk && ./emsdk install latest && ./emsdk activate latest"
    exit 1
fi

echo "emcc version: $(emcc --version | head -1)"

# Check submodule
if [ ! -f "$WASM_DIR/configure.ac" ]; then
    echo "ERROR: doom-wasm submodule not found"
    echo "Run: git submodule update --init --recursive"
    exit 1
fi

cd "$WASM_DIR"

# Clean if requested
if [ "$1" == "--clean" ]; then
    echo "Cleaning..."
    ./scripts/clean.sh 2>/dev/null || true
    rm -rf build
fi

# Build
echo "Building DOOM WASM..."
./scripts/build.sh

# Create build directory
mkdir -p "$BUILD_DIR"

# Copy artifacts
echo "Copying artifacts..."
if [ -f "$WASM_DIR/src/doom.wasm" ]; then
    cp "$WASM_DIR/src/doom.wasm" "$BUILD_DIR/"
    cp "$WASM_DIR/src/doom.js" "$BUILD_DIR/"
    echo "Build successful!"
else
    # Check alternative locations
    for f in $(find "$WASM_DIR" -name "*.wasm" 2>/dev/null); do
        echo "Found: $f"
        cp "$f" "$BUILD_DIR/"
    done
fi

# Copy WAD for convenience
if [ -f "$WAD_DIR/doom1.wad" ]; then
    cp "$WAD_DIR/doom1.wad" "$BUILD_DIR/"
    echo "Copied doom1.wad to build/"
fi

echo ""
echo "=== Build Complete ==="
echo "Artifacts in: $BUILD_DIR"
ls -la "$BUILD_DIR"
