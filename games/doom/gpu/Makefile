# GPU DOOM Makefile
# Builds GPU kernels and verification binaries for CPU vs GPU verification

# =============================================================================
# Compiler Configuration
# =============================================================================

NVCC = nvcc
NVCC_FLAGS = -O3 -arch=sm_120 -std=c++17 -lineinfo

GCC = gcc
GCC_FLAGS = -O2 -Wall -std=c99

# =============================================================================
# Directories
# =============================================================================

BUILD_DIR = build
SOURCE_DIR = ../source/linuxdoom-1.10

# =============================================================================
# Dependencies
# =============================================================================

# Common GPU headers
GPU_HEADERS = doom_types.cuh doom_data.cuh doom_combat.cuh doom_monsters.cuh \
              doom_levels.cuh doom_savegame.cuh doom_interest.cuh doom_export.cuh

# Interest tracking source files (host-only implementations)
GPU_SOURCES = doom_export.cu

# GPU kernels
GPU_KERNELS = p_enemy_gpu.cuh generated/cuda/p_enemy_gpu.cuh \
              generated/cuda/p_inter_gpu.cuh generated/cuda/p_map_gpu.cuh \
              generated/cuda/p_mobj_gpu.cuh generated/cuda/p_sight_gpu.cuh \
              generated/cuda/p_user_gpu.cuh

# =============================================================================
# Main Targets
# =============================================================================

.PHONY: all clean verify test benchmark

all: doom_main doom_verify

# Main GPU DOOM kernel (multi-instance simulation)
doom_main: $(BUILD_DIR)/doom_sim

# GPU verification binary (single-instance for CPU comparison)
doom_verify: $(BUILD_DIR)/doom_verify

# CPU reference binary (original DOOM, headless mode)
doom_reference: $(BUILD_DIR)/doom_reference

# Run verification harness
verify: doom_verify doom_reference
	@echo "=== Running CPU vs GPU Verification ==="
	python3 verify/cpu_gpu_verifier.py --ticks 100 --verbose

# Quick self-test
test: doom_verify
	@echo "=== GPU Self-Test ==="
	$(BUILD_DIR)/doom_verify --self-test

# Performance benchmark
benchmark: doom_main
	@echo "=== Performance Benchmark ==="
	@echo "10K instances, 1K ticks:"
	$(BUILD_DIR)/doom_sim 10000 1000
	@echo ""
	@echo "100K instances, 100 ticks:"
	$(BUILD_DIR)/doom_sim 100000 100

# =============================================================================
# Build Directory
# =============================================================================

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# =============================================================================
# GPU Binaries
# =============================================================================

# doom_sim: Multi-instance GPU DOOM
$(BUILD_DIR)/doom_sim: doom_main.cu $(GPU_SOURCES) $(GPU_HEADERS) $(GPU_KERNELS) | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) doom_main.cu $(GPU_SOURCES) -o $@

# doom_verify: Single-instance verification binary
$(BUILD_DIR)/doom_verify: doom_verify.cu $(GPU_HEADERS) | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) $< -o $@

# test_phase2: Monster AI test binary
$(BUILD_DIR)/test_phase2: test_phase2.cu $(GPU_HEADERS) $(GPU_KERNELS) | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) $< -o $@

# =============================================================================
# CPU Reference Binary (Original DOOM)
# =============================================================================

# Build original DOOM in headless verification mode
# This creates a binary that:
# - Reads TicCmd from stdin (binary format)
# - Outputs game state as JSONL to stdout
# - Runs without X11/sound (headless)
$(BUILD_DIR)/doom_reference: | $(BUILD_DIR)
	@echo "=== Building CPU Reference (Original DOOM) ==="
	@echo "NOTE: This requires modifications to linuxdoom-1.10 for headless mode"
	@echo "See plan document Phase 2 for details"
	@echo ""
	@echo "Required modifications to $(SOURCE_DIR):"
	@echo "  1. Add -DVERIFICATION_MODE to enable state logging"
	@echo "  2. Add -DHEADLESS to stub out video/sound"
	@echo "  3. Add i_verify.c with LogGameState() function"
	@echo "  4. Modify p_tick.c to call LogGameState(gametic)"
	@echo ""
	@echo "Once modified, run: make -C $(SOURCE_DIR) doom_reference"
	@echo ""
	@false  # Fail with message - requires manual source modifications

# Alternative: Build original DOOM as-is (for reference)
doom_reference_original:
	@echo "=== Building Original DOOM (requires X11) ==="
	$(MAKE) -C $(SOURCE_DIR)

# =============================================================================
# Cleanup
# =============================================================================

clean:
	rm -rf $(BUILD_DIR)
	@echo "Cleaned build directory"

clean_all: clean
	$(MAKE) -C $(SOURCE_DIR) clean
	@echo "Cleaned all build artifacts"

# =============================================================================
# Development Targets
# =============================================================================

# Run doom_sim with specific parameters
run: $(BUILD_DIR)/doom_sim
	$(BUILD_DIR)/doom_sim 10000 1000

# Run Phase 2 monster tests
test_phase2: $(BUILD_DIR)/test_phase2
	$(BUILD_DIR)/test_phase2

# Check if doom_data.cuh exists (required for all builds)
check_data:
	@if [ ! -f doom_data.cuh ]; then \
		echo "ERROR: doom_data.cuh not found"; \
		echo "Run: python tools/extract_doom_data.py"; \
		exit 1; \
	fi

# Extract data from original DOOM source
extract_data:
	@echo "=== Extracting Data Tables from Original DOOM ==="
	python3 tools/extract_doom_data.py --source $(SOURCE_DIR) --output doom_data.cuh
	@echo "Generated doom_data.cuh"

# =============================================================================
# Help Target
# =============================================================================

help:
	@echo "GPU DOOM Makefile"
	@echo ""
	@echo "Main Targets:"
	@echo "  all              - Build doom_sim and doom_verify (default)"
	@echo "  doom_main        - Build multi-instance GPU DOOM (doom_sim)"
	@echo "  doom_verify      - Build single-instance verification binary"
	@echo "  doom_reference   - Build CPU reference (requires manual source mods)"
	@echo ""
	@echo "Testing:"
	@echo "  test             - Run GPU self-test"
	@echo "  verify           - Run CPU vs GPU verification harness"
	@echo "  benchmark        - Run performance benchmarks"
	@echo ""
	@echo "Development:"
	@echo "  run              - Run doom_sim with default parameters"
	@echo "  test_phase2      - Run monster AI tests"
	@echo "  extract_data     - Extract data tables from original DOOM"
	@echo "  check_data       - Verify doom_data.cuh exists"
	@echo ""
	@echo "Cleanup:"
	@echo "  clean            - Remove build directory"
	@echo "  clean_all        - Clean GPU and original DOOM builds"
	@echo ""
	@echo "Usage Examples:"
	@echo "  make doom_verify           # Build verification binary"
	@echo "  make test                  # Run self-test"
	@echo "  make verify                # Run full CPU/GPU verification"
	@echo "  make benchmark             # Performance testing"

.PHONY: help run test_phase2 check_data extract_data doom_reference_original
