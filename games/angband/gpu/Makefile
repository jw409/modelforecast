NVCC = nvcc
NVCC_FLAGS = -O3 -arch=sm_120 -std=c++17

BUILD_DIR = build

all: $(BUILD_DIR)/borg_sim

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/borg_sim: borg_kernel.cu borg_state.h ../../common/interleaved.h | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) $< -o $@

clean:
	rm -rf $(BUILD_DIR)

run: $(BUILD_DIR)/borg_sim
	$(BUILD_DIR)/borg_sim 10000 1000

benchmark: $(BUILD_DIR)/borg_sim
	@echo "=== 1K instances, 10K turns ==="
	$(BUILD_DIR)/borg_sim 1000 10000
	@echo ""
	@echo "=== 10K instances, 1K turns ==="
	$(BUILD_DIR)/borg_sim 10000 1000
	@echo ""
	@echo "=== 100K instances, 100 turns ==="
	$(BUILD_DIR)/borg_sim 100000 100

.PHONY: all clean run benchmark
