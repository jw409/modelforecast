#!/bin/bash
# srcml-transpile.sh - Transform C to CUDA using srcML+XSLT pipeline
#
# Usage: srcml-transpile.sh input.c output.cu
#
# Bare-bones scaffold - add XSLT transforms as needed

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XSLT_DIR="$SCRIPT_DIR/xslt"

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 input.c output.cu"
    exit 1
fi

INPUT="$1"
OUTPUT="$2"

if [[ ! -f "$INPUT" ]]; then
    echo "Error: Input file not found: $INPUT"
    exit 1
fi

# Build XSLT pipeline dynamically from available transforms
XSLT_CHAIN=""
for xsl in "$XSLT_DIR"/*.xsl; do
    if [[ -f "$xsl" ]]; then
        XSLT_CHAIN="$XSLT_CHAIN | xsltproc $xsl -"
    fi
done

if [[ -z "$XSLT_CHAIN" ]]; then
    # No transforms yet - just convert to CUDA syntax
    srcml "$INPUT" | srcml --src-encoding UTF-8 > "$OUTPUT"
else
    eval "srcml \"$INPUT\" $XSLT_CHAIN | srcml --src-encoding UTF-8 > \"$OUTPUT\""
fi

echo "✓ $(basename "$INPUT") → $(basename "$OUTPUT")"
