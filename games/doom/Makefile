# DOOM Project Top-Level Makefile
# Delegates to gpu/ for GPU DOOM builds and verification

# =============================================================================
# Configuration
# =============================================================================

GPU_DIR = gpu
BUILD_DIR = build

# =============================================================================
# Main Targets - Delegate to GPU Makefile
# =============================================================================

.PHONY: all clean test benchmark verify help

# Build all GPU binaries
all:
	$(MAKE) -C $(GPU_DIR) all

# GPU verification binary (single-instance for CPU comparison)
doom_verify:
	$(MAKE) -C $(GPU_DIR) doom_verify

# Main GPU DOOM kernel (multi-instance simulation)
doom_main:
	$(MAKE) -C $(GPU_DIR) doom_main

# CPU reference binary (original DOOM, headless mode)
doom_reference:
	$(MAKE) -C $(GPU_DIR) doom_reference

# =============================================================================
# Testing and Verification
# =============================================================================

# Run GPU self-test
test:
	$(MAKE) -C $(GPU_DIR) test

# Run CPU vs GPU verification harness
verify:
	$(MAKE) -C $(GPU_DIR) verify

# Run performance benchmarks
benchmark:
	$(MAKE) -C $(GPU_DIR) benchmark

# =============================================================================
# Development Targets
# =============================================================================

# Run doom_sim with default parameters
run:
	$(MAKE) -C $(GPU_DIR) run

# Run Phase 2 monster tests
test_phase2:
	$(MAKE) -C $(GPU_DIR) test_phase2

# Extract data tables from original DOOM source
extract_data:
	$(MAKE) -C $(GPU_DIR) extract_data

# Check if doom_data.cuh exists
check_data:
	$(MAKE) -C $(GPU_DIR) check_data

# =============================================================================
# Cleanup
# =============================================================================

clean:
	$(MAKE) -C $(GPU_DIR) clean

clean_all:
	$(MAKE) -C $(GPU_DIR) clean_all
	rm -rf $(BUILD_DIR)

# =============================================================================
# Help
# =============================================================================

help:
	@echo "DOOM Project Makefile (Top-Level)"
	@echo ""
	@echo "This Makefile delegates to gpu/Makefile for all GPU DOOM builds."
	@echo ""
	@echo "Main Targets:"
	@echo "  all              - Build all GPU binaries (doom_sim + doom_verify)"
	@echo "  doom_verify      - Build single-instance verification binary"
	@echo "  doom_main        - Build multi-instance GPU DOOM (doom_sim)"
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
	@echo "  clean_all        - Clean all build artifacts (GPU + original DOOM)"
	@echo ""
	@echo "For detailed GPU-specific help:"
	@echo "  make -C gpu help"

.PHONY: all clean clean_all test benchmark verify help doom_verify doom_main \
        doom_reference run test_phase2 extract_data check_data
