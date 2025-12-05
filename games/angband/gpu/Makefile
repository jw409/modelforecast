NVCC = nvcc
NVCC_FLAGS = -O3 -arch=sm_120 -std=c++17

BUILD_DIR = build

all: $(BUILD_DIR)/borg_sim $(BUILD_DIR)/borg_sim_v2

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/borg_sim: borg_kernel.cu borg_state.h ../../common/interleaved.h | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) $< -o $@

$(BUILD_DIR)/borg_sim_v2: borg_kernel_v2.cu borg_state.h ../../common/interleaved.h angband_combat.cuh angband_monsters.cuh angband_dungeon.cuh | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) -Wno-deprecated-gpu-targets $< -o $@

clean:
	rm -rf $(BUILD_DIR)

run: $(BUILD_DIR)/borg_sim
	$(BUILD_DIR)/borg_sim 10000 1000

run_v2: $(BUILD_DIR)/borg_sim_v2
	$(BUILD_DIR)/borg_sim_v2 10000 1000

verify_v2: $(BUILD_DIR)/borg_sim_v2
	$(BUILD_DIR)/borg_sim_v2 --verify 12345

benchmark: $(BUILD_DIR)/borg_sim
	@echo "=== 1K instances, 10K turns ==="
	$(BUILD_DIR)/borg_sim 1000 10000
	@echo ""
	@echo "=== 10K instances, 1K turns ==="
	$(BUILD_DIR)/borg_sim 10000 1000
	@echo ""
	@echo "=== 100K instances, 100 turns ==="
	$(BUILD_DIR)/borg_sim 100000 100

benchmark_v2: $(BUILD_DIR)/borg_sim_v2
	@echo "=== V2 (Real Rules): 10K instances, 1K turns ==="
	$(BUILD_DIR)/borg_sim_v2 10000 1000

.PHONY: all clean run run_v2 verify_v2 benchmark benchmark_v2
