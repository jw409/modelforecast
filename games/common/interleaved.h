/*
 * INTERLEAVED MEMORY LAYOUT
 *
 * Optimized for GPU coalesced access.
 * Based on CoreWars GPU MARS implementation achieving 27,845 battles/sec.
 *
 * Pattern: array[row * num_instances + instance_id]
 *
 * Why interleaved?
 * - Adjacent threads access adjacent memory addresses
 * - 128-byte cache line utilization maximized
 * - Field access remains intuitive (struct-like)
 */

#ifndef INTERLEAVED_H
#define INTERLEAVED_H

#include <cuda_runtime.h>
#include <stdint.h>

// ============================================================================
// INDEXING HELPERS
// ============================================================================

// 2D index for interleaved arrays
// row = logical index (e.g., position in queue, cell in grid)
// col = instance ID
// width = num_instances (stride)
__device__ __host__ inline size_t idx2d(int row, int col, int width) {
    return (size_t)row * width + col;
}

// 3D index for multi-dimensional interleaved arrays
// E.g., dungeon grid: grid[y][x][instance]
__device__ __host__ inline size_t idx3d(int z, int y, int x, int height, int width) {
    return ((size_t)z * height + y) * width + x;
}

// ============================================================================
// INTERLEAVED ARRAY OPERATIONS
// ============================================================================

// Get value from interleaved array
#define IGET(arr, row, id, n) ((arr)[idx2d(row, id, n)])

// Set value in interleaved array
#define ISET(arr, row, id, n, val) ((arr)[idx2d(row, id, n)] = (val))

// Increment value in interleaved array
#define IINC(arr, row, id, n) ((arr)[idx2d(row, id, n)]++)

// Decrement value in interleaved array
#define IDEC(arr, row, id, n) ((arr)[idx2d(row, id, n)]--)

// ============================================================================
// PROCESS QUEUE (Interleaved)
// Used by both CoreWars (warriors) and Angband (borg actions)
// ============================================================================

// Queue state stored separately for register promotion
// The actual queue data (pcs/actions) is interleaved in global memory

// Push to interleaved queue
__device__ inline void queue_push_interleaved(
    uint16_t* pcs,           // Queue data array [max_size * num_instances]
    uint16_t* tail,          // Tail pointer (in registers ideally)
    uint16_t* count,         // Count (in registers ideally)
    int instance_id,
    int num_instances,
    int max_size,
    uint16_t value
) {
    if (*count < max_size) {
        size_t idx = idx2d(*tail, instance_id, num_instances);
        pcs[idx] = value;
        *tail = (*tail + 1) % max_size;
        (*count)++;
    }
}

// Pop from interleaved queue
__device__ inline uint16_t queue_pop_interleaved(
    uint16_t* pcs,           // Queue data array
    uint16_t* head,          // Head pointer (in registers)
    uint16_t* count,         // Count (in registers)
    int instance_id,
    int num_instances,
    int max_size
) {
    if (*count > 0) {
        size_t idx = idx2d(*head, instance_id, num_instances);
        uint16_t value = pcs[idx];
        *head = (*head + 1) % max_size;
        (*count)--;
        return value;
    }
    return 0xFFFF; // Invalid
}

// ============================================================================
// GRID/DUNGEON (Interleaved)
// For games with 2D maps like Angband
// ============================================================================

// Get cell from interleaved 2D grid
// Grid layout: grid[y * width + x][instance]
__device__ inline uint8_t grid_get_interleaved(
    uint8_t* grid,           // Grid data [grid_size * num_instances]
    int x, int y,
    int grid_width,
    int instance_id,
    int num_instances
) {
    int cell_idx = y * grid_width + x;
    return IGET(grid, cell_idx, instance_id, num_instances);
}

// Set cell in interleaved 2D grid
__device__ inline void grid_set_interleaved(
    uint8_t* grid,
    int x, int y,
    int grid_width,
    int instance_id,
    int num_instances,
    uint8_t value
) {
    int cell_idx = y * grid_width + x;
    ISET(grid, cell_idx, instance_id, num_instances, value);
}

// ============================================================================
// MEMORY ALLOCATION HELPERS (Host)
// ============================================================================

// Calculate memory needed for interleaved array
inline size_t interleaved_size(size_t elements_per_instance,
                                size_t num_instances,
                                size_t element_size) {
    return elements_per_instance * num_instances * element_size;
}

// Allocate interleaved array on device
#define CUDA_ALLOC_INTERLEAVED(ptr, type, elems_per, num_inst) \
    cudaMalloc(&(ptr), interleaved_size(elems_per, num_inst, sizeof(type)))

// ============================================================================
// DEBUG HELPERS
// ============================================================================

// Print interleaved array slice (host, after cudaMemcpy)
inline void print_interleaved_slice(
    void* host_arr,
    size_t elem_size,
    int row,
    int start_instance,
    int count,
    int num_instances
) {
    printf("Row %d, instances %d-%d: ", row, start_instance, start_instance + count - 1);
    for (int i = start_instance; i < start_instance + count; i++) {
        size_t idx = idx2d(row, i, num_instances);
        if (elem_size == 1) printf("%02x ", ((uint8_t*)host_arr)[idx]);
        else if (elem_size == 2) printf("%04x ", ((uint16_t*)host_arr)[idx]);
        else if (elem_size == 4) printf("%08x ", ((uint32_t*)host_arr)[idx]);
    }
    printf("\n");
}

#endif // INTERLEAVED_H
