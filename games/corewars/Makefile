# GPU MARS Makefile - Auto-detects NVIDIA GPU compute capability
NVCC = nvcc

# Auto-detect GPU compute capability (e.g., sm_90 for RTX 4090, sm_120 for RTX 5090)
# Falls back to sm_90 if detection fails
GPU_ARCH := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.' | sed 's/^/sm_/')
ifeq ($(GPU_ARCH),sm_)
    GPU_ARCH = sm_90
endif

# Get VRAM in MB for scaling recommendations
VRAM_MB := $(shell nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
ifeq ($(VRAM_MB),)
    VRAM_MB = 8000
endif

# Calculate recommended battles (64KB per battle, use 60% of VRAM)
# Formula: (VRAM_MB * 1024 * 0.6) / 64 = VRAM_MB * 9.6
RECOMMENDED_BATTLES := $(shell echo "$(VRAM_MB) * 96 / 10" | bc 2>/dev/null || echo "100000")

NVCC_FLAGS = -O3 -arch=$(GPU_ARCH) -std=c++17

SRC_DIR = src
BUILD_DIR = build

all: info $(BUILD_DIR)/gpu_mars $(BUILD_DIR)/gpu_mars_opt $(BUILD_DIR)/gpu_mars_interleaved $(BUILD_DIR)/swarm_term

info:
	@echo "╔══════════════════════════════════════════════════════╗"
	@echo "║ GPU MARS Build Configuration                         ║"
	@echo "╠══════════════════════════════════════════════════════╣"
	@echo "║ GPU Architecture: $(GPU_ARCH)"
	@echo "║ VRAM Available:   $(VRAM_MB) MB"
	@echo "║ Recommended Battles: $(RECOMMENDED_BATTLES)"
	@echo "║ NVCC Flags: $(NVCC_FLAGS)"
	@echo "╚══════════════════════════════════════════════════════╝"

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/mars_kernel.o: $(SRC_DIR)/mars_kernel.cu $(SRC_DIR)/mars.h | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

$(BUILD_DIR)/gpu_mars: $(SRC_DIR)/mars_main.cu $(BUILD_DIR)/mars_kernel.o | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) $^ -o $@

$(BUILD_DIR)/gpu_mars_opt: $(SRC_DIR)/mars_opt.cu $(SRC_DIR)/mars.h | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) -Wno-deprecated-gpu-targets $< -o $@

$(BUILD_DIR)/gpu_mars_interleaved: $(SRC_DIR)/mars_interleaved.cu $(SRC_DIR)/mars.h | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) -Wno-deprecated-gpu-targets $< -o $@

$(BUILD_DIR)/swarm_term: $(SRC_DIR)/swarm_terminal.cu $(SRC_DIR)/mars.h | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) -Wno-deprecated-gpu-targets $< -o $@

clean:
	rm -rf $(BUILD_DIR)

# Run with auto-scaled battle count
run: $(BUILD_DIR)/gpu_mars_opt
	@echo "Running with $(RECOMMENDED_BATTLES) battles (auto-scaled for $(VRAM_MB) MB VRAM)"
	$(BUILD_DIR)/gpu_mars_opt $(RECOMMENDED_BATTLES)

# Run with specific battle count
run-%: $(BUILD_DIR)/gpu_mars_opt
	$(BUILD_DIR)/gpu_mars_opt $*

term: $(BUILD_DIR)/swarm_term
	$(BUILD_DIR)/swarm_term 256 30 10000

term_fast: $(BUILD_DIR)/swarm_term
	$(BUILD_DIR)/swarm_term 256 10 50000

# Show detected GPU info
gpu-info:
	@echo "GPU: $$(nvidia-smi --query-gpu=name --format=csv,noheader)"
	@echo "Compute Capability: $$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader)"
	@echo "VRAM: $(VRAM_MB) MB"
	@echo "Detected Architecture: $(GPU_ARCH)"
	@echo "Recommended Battles: $(RECOMMENDED_BATTLES)"

.PHONY: all clean run term term_fast info gpu-info
