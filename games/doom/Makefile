NVCC = nvcc
NVCC_FLAGS = -O3 -arch=native -std=c++17 -lineinfo

SRC_DIR = gpu
BUILD_DIR = build

all: $(BUILD_DIR)/gpu_doom_test

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/gpu_doom_test: $(SRC_DIR)/doom_main.cu $(SRC_DIR)/doom_types.cuh | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) $< -o $@

clean:
	rm -rf $(BUILD_DIR)

test: $(BUILD_DIR)/gpu_doom_test
	$(BUILD_DIR)/gpu_doom_test 100

bench: $(BUILD_DIR)/gpu_doom_test
	$(BUILD_DIR)/gpu_doom_test 1000 500

.PHONY: all clean test bench
