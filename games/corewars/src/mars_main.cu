#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "mars.h"
#include <vector>
#include <iostream>
#include <chrono>

// Forward declare kernel
__global__ void mars_battle_kernel(
    BattleState* battles,
    const Instruction* warriors_a,
    const Instruction* warriors_b,
    const uint16_t* warrior_lengths_a,
    const uint16_t* warrior_lengths_b,
    const uint32_t* start_positions_a,
    const uint32_t* start_positions_b,
    uint32_t num_battles
);

void checkCuda(cudaError_t result, const char* func, const char* file, int line) {
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n",
            file, line, static_cast<unsigned int>(result), cudaGetErrorName(result), func);
        exit(EXIT_FAILURE);
    }
}
#define CUDA_CHECK(val) checkCuda((val), #val, __FILE__, __LINE__)

int main(int argc, char** argv) {
    uint32_t num_battles = 10000; // Default
    if (argc > 1) num_battles = atoi(argv[1]);

    printf("Initializing GPU MARS for %d battles...\n", num_battles);

    // Define Warriors
    // Imp: MOV 0, 1
    std::vector<Instruction> imp;
    imp.push_back({MOV, I, DIRECT, DIRECT, 0, 1});

    // Dwarf (Simplified)
    // ADD #4, 3
    // MOV 2, @2
    // JMP -2
    // DAT #0, #0
    std::vector<Instruction> dwarf;
    dwarf.push_back({ADD, AB, IMMEDIATE, DIRECT, 4, 3});
    dwarf.push_back({MOV, I, DIRECT, INDIRECT_B, 2, 2});
    dwarf.push_back({JMP, I, DIRECT, DIRECT, -2, 0});
    dwarf.push_back({DAT, F, IMMEDIATE, IMMEDIATE, 0, 0});

    // Allocate Device Memory for Warriors
    Instruction *d_imp, *d_dwarf;
    CUDA_CHECK(cudaMalloc(&d_imp, imp.size() * sizeof(Instruction)));
    CUDA_CHECK(cudaMalloc(&d_dwarf, dwarf.size() * sizeof(Instruction)));
    CUDA_CHECK(cudaMemcpy(d_imp, imp.data(), imp.size() * sizeof(Instruction), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_dwarf, dwarf.data(), dwarf.size() * sizeof(Instruction), cudaMemcpyHostToDevice));

    // Prepare Battle Configurations
    std::vector<uint16_t> h_len_a(num_battles, (uint16_t)imp.size());
    std::vector<uint16_t> h_len_b(num_battles, (uint16_t)dwarf.size());
    std::vector<uint32_t> h_pos_a(num_battles);
    std::vector<uint32_t> h_pos_b(num_battles);

    // Randomize start positions
    for (uint32_t i = 0; i < num_battles; i++) {
        h_pos_a[i] = rand() % (CORE_SIZE - 100);
        h_pos_b[i] = (h_pos_a[i] + 4000) % CORE_SIZE; // Start far apart
    }

    uint16_t *d_len_a, *d_len_b;
    uint32_t *d_pos_a, *d_pos_b;
    CUDA_CHECK(cudaMalloc(&d_len_a, num_battles * sizeof(uint16_t)));
    CUDA_CHECK(cudaMalloc(&d_len_b, num_battles * sizeof(uint16_t)));
    CUDA_CHECK(cudaMalloc(&d_pos_a, num_battles * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_pos_b, num_battles * sizeof(uint32_t)));

    CUDA_CHECK(cudaMemcpy(d_len_a, h_len_a.data(), num_battles * sizeof(uint16_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_len_b, h_len_b.data(), num_battles * sizeof(uint16_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pos_a, h_pos_a.data(), num_battles * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pos_b, h_pos_b.data(), num_battles * sizeof(uint32_t), cudaMemcpyHostToDevice));

    // Allocate Battle State (Massive!)
    // 10,000 battles * sizeof(BattleState)
    // sizeof(BattleState) ~= 64KB + 32KB = 96KB
    // 10,000 * 96KB = 960 MB. Easy for 5090.
    size_t battle_size = sizeof(BattleState);
    printf("Size of one BattleState: %zu bytes\n", battle_size);
    printf("Total VRAM required: %.2f MB\n", (double)(num_battles * battle_size) / (1024*1024));

    BattleState* d_battles;
    CUDA_CHECK(cudaMalloc(&d_battles, num_battles * battle_size));

    // Launch Kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (num_battles + threadsPerBlock - 1) / threadsPerBlock;

    printf("Launching kernel: %d blocks, %d threads\n", blocksPerGrid, threadsPerBlock);

    // Synchronize before timing
    cudaDeviceSynchronize();
    auto start = std::chrono::high_resolution_clock::now();

    mars_battle_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_battles,
        d_imp, d_dwarf,
        d_len_a, d_len_b,
        d_pos_a, d_pos_b,
        num_battles
    );

    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    CUDA_CHECK(cudaGetLastError());

    std::chrono::duration<double> diff = end - start;
    printf("Kernel execution time: %.4f s\n", diff.count());
    printf("Throughput: %.2f battles/sec\n", num_battles / diff.count());

    // Retrieve Results
    // We only need the winners, but we'll copy a few full states to debug if needed
    // For now, just copy back winners by allocating a small array or just copy the whole thing if memory allows
    // Copying 1GB back to host is slow, so let's just copy a few
    
    // Wait, we declared d_battles. Let's copy the first one to check winner
    BattleState* h_battles_sample = (BattleState*)malloc(sizeof(BattleState));
    CUDA_CHECK(cudaMemcpy(h_battles_sample, d_battles, sizeof(BattleState), cudaMemcpyDeviceToHost));

    printf("Battle 0 Winner: %d (Cycles: %d)\n", h_battles_sample->winner, h_battles_sample->cycles);

    // Clean up
    cudaFree(d_imp);
    cudaFree(d_dwarf);
    cudaFree(d_len_a);
    cudaFree(d_len_b);
    cudaFree(d_pos_a);
    cudaFree(d_pos_b);
    cudaFree(d_battles);
    free(h_battles_sample);

    return 0;
}
