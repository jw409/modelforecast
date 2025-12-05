NVCC = nvcc
NVCC_FLAGS = -O3 -arch=sm_90 

SRC_DIR = src
BUILD_DIR = build

all: $(BUILD_DIR)/gpu_mars $(BUILD_DIR)/gpu_mars_soa $(BUILD_DIR)/gpu_mars_packed $(BUILD_DIR)/gpu_mars_interleaved $(BUILD_DIR)/gpu_mars_aos_optimized

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/mars_kernel.o: $(SRC_DIR)/mars_kernel.cu $(SRC_DIR)/mars.h | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

$(BUILD_DIR)/gpu_mars: $(SRC_DIR)/mars_main.cu $(BUILD_DIR)/mars_kernel.o | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) $^ -o $@

$(BUILD_DIR)/gpu_mars_soa: $(SRC_DIR)/mars_soa.cu $(SRC_DIR)/mars.h | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) $< -o $@

$(BUILD_DIR)/gpu_mars_packed: $(SRC_DIR)/mars_packed.cu $(SRC_DIR)/mars.h | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) $< -o $@

$(BUILD_DIR)/gpu_mars_interleaved: $(SRC_DIR)/mars_interleaved.cu $(SRC_DIR)/mars.h | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) $< -o $@

$(BUILD_DIR)/gpu_mars_aos_optimized: $(SRC_DIR)/mars_aos_optimized.cu $(SRC_DIR)/mars.h | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) $< -o $@

clean:
	rm -rf $(BUILD_DIR)

run: $(BUILD_DIR)/gpu_mars_aos_optimized
	$(BUILD_DIR)/gpu_mars_aos_optimized
