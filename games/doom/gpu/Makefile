NVCC = nvcc
NVCC_FLAGS = -O3 -arch=sm_120 -std=c++17

BUILD_DIR = build

all: $(BUILD_DIR)/doom_sim

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/doom_sim: doom_main.cu doom_types.cuh doom_savegame.cuh doom_monsters.cuh doom_combat.cuh | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) $< -o $@

$(BUILD_DIR)/test_phase2: test_phase2.cu doom_types.cuh doom_monsters.cuh doom_combat.cuh | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) $< -o $@

clean:
	rm -rf $(BUILD_DIR)

run: $(BUILD_DIR)/doom_sim
	$(BUILD_DIR)/doom_sim 10000 1000

test_phase2: $(BUILD_DIR)/test_phase2
	$(BUILD_DIR)/test_phase2

benchmark: $(BUILD_DIR)/doom_sim
	@echo "=== 10K instances, 1K ticks ==="
	$(BUILD_DIR)/doom_sim 10000 1000
	@echo ""
	@echo "=== 100K instances, 100 ticks ==="
	$(BUILD_DIR)/doom_sim 100000 100

.PHONY: all clean run benchmark test_phase2
