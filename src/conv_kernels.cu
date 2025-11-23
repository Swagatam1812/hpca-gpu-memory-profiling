
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include "../include/conv_kernels.cuh"

// =============================================================================
// BASELINE IMPLEMENTATION (PROVIDED - DO NOT MODIFY)
// =============================================================================
// This is an intentionally inefficient implementation for comparison purposes.

static __device__ __forceinline__ size_t idx3(int batch_index, int row, int col,
                                               int height, int width) {
    return static_cast<size_t>(batch_index) * height * width +
           static_cast<size_t>(row) * width +
           col;
}

__global__ void kernel_conv2d_baseline(const float* __restrict__ input_images,
                                       const float* __restrict__ kernel,
                                       float* __restrict__ output_images,
                                       int batch_size, int height, int width,
                                       int kernel_size) {
    int batch_index = blockIdx.z;

    int row = blockIdx.y * blockDim.y + threadIdx.x;  // Note: threadIdx.x for row
    int col = blockIdx.x * blockDim.x + threadIdx.y;  // Note: threadIdx.y for col

    if (batch_index >= batch_size || row >= height || col >= width) {
        return;
    }

    int radius = (kernel_size - 1) / 2;
    float accumulated_value = 0.0f;

    for (int kernel_row = 0; kernel_row < kernel_size; ++kernel_row) {
        int input_row = row + kernel_row - radius;
        if (input_row < 0 || input_row >= height) continue;

        for (int kernel_col = 0; kernel_col < kernel_size; ++kernel_col) {
            int input_col = col + kernel_col - radius;
            if (input_col < 0 || input_col >= width) continue;

            float input_pixel = input_images[idx3(batch_index, input_row, input_col,
                                                   height, width)];
            float kernel_weight = kernel[kernel_row * kernel_size + kernel_col];
            accumulated_value += input_pixel * kernel_weight;
        }
    }

    output_images[idx3(batch_index, row, col, height, width)] = accumulated_value;
}

void conv2d_baseline(const float* input, const float* kernel, float* output,
                     int batch_size, int height, int width, int kernel_size,
                     cudaStream_t stream) {
    dim3 threads_per_block(16, 16, 1);
    dim3 blocks_per_grid(
        (width + threads_per_block.x - 1) / threads_per_block.x,
        (height + threads_per_block.y - 1) / threads_per_block.y,
        batch_size
    );

    kernel_conv2d_baseline<<<blocks_per_grid, threads_per_block, 0, stream>>>(
        input, kernel, output, batch_size, height, width, kernel_size
    );
}

// =============================================================================
// VARIANT 1: GLOBAL-MEMORY ACCESS PATTERN (BANDWIDTH EFFICIENCY)
// =============================================================================
// TODO: Restructure the thread/data mapping so that the hardware can merge
//       per-thread loads into fewer, fuller memory transactions.
//
// GOAL:
// - Increase effective global-memory bandwidth by maximizing bytes used per
//   transaction and minimizing transactions per warp.
//
// WHAT TO MEASURE (before & after):
// - L1TEX/L2: average bytes used per sector (aim ~32/32 for 32B sectors).
// - Global load efficiency / requested vs. delivered bytes.
// - DRAM read throughput (GB/s) and “transactions per request” counters.
// - Kernel time and MPix/s.
//
// HINTS (discovery-oriented):
// - Inspect how a warp’s threads (lane 0..31) walk the image in memory.
//   Are neighboring lanes reading neighboring addresses, or are they striding?
// - Revisit your mapping from (threadIdx.x, threadIdx.y) → (row, col).
//   Which dimension in memory is contiguous, and do lanes advance along it?
// - Consider block shapes where each warp spans one logical row of the tile
//   rather than splitting a warp across multiple rows.
// - The order of inner loops matters: move the loop that advances along the
//   contiguous memory dimension into the per-lane direction.
// - When alignment permits, loading wider types (e.g., 16-byte aligned chunks)
//   reduces the number of memory transactions. Handle tails safely.
//
// =============================================================================

__global__ void kernel_conv2d_variant_1(const float* __restrict__ input_images,
                                       const float* __restrict__ kernel,
                                       float* __restrict__ output_images,
                                       int batch_size, int height, int width,
                                       int kernel_size) {
    int batch_index = blockIdx.z;
    // Correct mapping for coalesced memory access:
    int row = blockIdx.y * blockDim.y + threadIdx.y;  
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // contiguous allocation

    if (batch_index >= batch_size || row >= height || col >= width) {
        return;
    }

    int radius = (kernel_size - 1) / 2;
    float accumulated_value = 0.0f;

    for (int kernel_row = 0; kernel_row < kernel_size; ++kernel_row) {
        int input_row = row + kernel_row - radius;
        if (input_row < 0 || input_row >= height) continue;

        for (int kernel_col = 0; kernel_col < kernel_size; ++kernel_col) {
            int input_col = col + kernel_col - radius;
            if (input_col < 0 || input_col >= width) continue;

            float input_pixel = input_images[idx3(batch_index, input_row, input_col,
                                                   height, width)];
            float kernel_weight = kernel[kernel_row * kernel_size + kernel_col];
            accumulated_value += input_pixel * kernel_weight;
        }
    }

    output_images[idx3(batch_index, row, col, height, width)] = accumulated_value;
}

void conv2d_variant1(const float* input, const float* kernel, float* output,
                     int batch_size, int height, int width, int kernel_size,
                     cudaStream_t stream) {
    // TODO: Configure and launch your kernel

    dim3 threads_per_block(16, 16, 1);
    dim3 blocks_per_grid(
        (width + threads_per_block.x - 1) / threads_per_block.x,
        (height + threads_per_block.y - 1) / threads_per_block.y,
        batch_size
    );

    kernel_conv2d_variant_1<<<blocks_per_grid, threads_per_block, 0, stream>>>(
        input, kernel, output, batch_size, height, width, kernel_size
    );
}

// =============================================================================
// VARIANT 2: ON-CHIP MEMORY (SHARED + CONSTANT)
// =============================================================================
//
// In this task, you will explore the different levels of the GPU memory hierarchy
// and analyze how they impact performance.
//
// Begin by profiling the naive convolution implementation using NVIDIA Nsight Compute.
// Record key metrics such as memory bandwidth utilization, cache hit rates, IPC, and
// other relevant performance indicators.
//
// Next, study the various GPU memory types — both on-chip and off-chip — and discuss
// their access latencies and bandwidths. Explain which of these memories are being
// used by the naive convolution kernel and how.
//
// Then, implement Variant 2 by modifying the kernel to make use of different on-chip
// memory spaces. Specifically, explore the use of shared memory and constant memory
// to improve data reuse and reduce global memory traffic.
//
// After your optimization, re-profile the kernel and report changes in cache
// utilization, bandwidth utilization, and overall performance.
//
// Finally, observe and explain an interesting phenomenon: certain optimizations may
// increase memory bandwidth utilization while decreasing cache hit rates, yet still
// lead to better performance. Provide a detailed reasoning for why this happens,
// relating it to reduced cache dependence, more efficient data reuse, and improved
// throughput across the GPU memory hierarchy.
//
// =============================================================================

// ================================================================
// VARIANT 2 — Shared Memory + Constant Memory Kernel
// (Matches TA style, simple cooperative load, same idx3 usage)
// ================================================================

// Variant 2: shared-memory tiling + constant memory for kernel
// Paste into src/conv_kernels.cu alongside idx3 and other kernels.

#define MAX_KERNEL_SIZE 31
#define MAX_KERNEL_ELEMENTS (MAX_KERNEL_SIZE * MAX_KERNEL_SIZE)
#define TILE_WIDTH 16
#define TILE_HEIGHT 16

// Constant memory for kernel weights (read-only, cached)
__constant__ float const_kernel[MAX_KERNEL_ELEMENTS];

__global__ void kernel_conv2d_variant2(const float* __restrict__ input_images,
                                       float* __restrict__ output_images,
                                       int batch_size, int height, int width,
                                       int kernel_size) {
    // dynamic shared memory array passed from kernel launch
    extern __shared__ float scratchpad_input[]; // flattened 2D tile (row-major)

    const int radius = (kernel_size - 1) / 2;
    const int input_tile_width = TILE_WIDTH + 2 * radius;
    const int input_tile_height = TILE_HEIGHT + 2 * radius;
    const int sstride = input_tile_width; // shared stride

    const int tx = threadIdx.x; // 0 .. TILE_WIDTH-1
    const int ty = threadIdx.y; // 0 .. TILE_HEIGHT-1

    const int batch_index = blockIdx.z;

    // Coordinates of the output pixel this thread will compute
    const int output_row = blockIdx.y * TILE_HEIGHT + ty;
    const int output_col = blockIdx.x * TILE_WIDTH + tx;

    // Top-left of the input tile in global coordinates (includes halo)
    const int input_tile_start_row = blockIdx.y * TILE_HEIGHT - radius;
    const int input_tile_start_col = blockIdx.x * TILE_WIDTH - radius;

    // Cooperative load of entire input tile (includes halo). We'll linearize tile indices.
    const int block_threads = TILE_WIDTH * TILE_HEIGHT;          // e.g., 256
    const int thread_id = ty * TILE_WIDTH + tx;                 // linear thread id in block
    const int tile_pixels = input_tile_width * input_tile_height;

    // Loop: distribute tile_pixels across threads in the block.
    for (int i = thread_id; i < tile_pixels; i += block_threads) {
        const int tile_row = i / input_tile_width;               // 0..input_tile_height-1
        const int tile_col = i % input_tile_width;               // 0..input_tile_width-1

        const int global_row = input_tile_start_row + tile_row;
        const int global_col = input_tile_start_col + tile_col;

        float pixel_value = 0.0f;
        if (batch_index < batch_size &&
            global_row >= 0 && global_row < height &&
            global_col >= 0 && global_col < width) {
            pixel_value = input_images[idx3(batch_index, global_row, global_col, height, width)];
        }

        // store into shared memory
        scratchpad_input[tile_row * sstride + tile_col] = pixel_value;
    }

    // All threads must wait for shared tile to be populated
    __syncthreads();

    // Bounds: threads that compute outside image should not write, but must have participated in loads above.
    if (batch_index >= batch_size || output_row >= height || output_col >= width) {
        return;
    }

    // Compute convolution using data from shared memory and kernel from constant memory
    float acc = 0.0f;

    // center position inside shared tile for (ty,tx) is (ty + radius, tx + radius)
    const int center_row = ty + radius;
    const int center_col = tx + radius;

    // iterate kernel
    for (int kr = 0; kr < kernel_size; ++kr) {
        const int srow = center_row + (kr - radius); // alternative indexing below keeps it simple
        // We'll compute scratch indices as (center_row + kr - radius) but since we loaded halo that maps to 0..input_tile_height-1
        for (int kc = 0; kc < kernel_size; ++kc) {
            const int scol = center_col + (kc - radius);
            // access from shared memory (fast)
            float inpix = scratchpad_input[srow * sstride + scol];
            float kw = const_kernel[kr * kernel_size + kc];
            acc += inpix * kw;
        }
    }

    // Write output (coalesced across threads in a warp because threadIdx.x maps to column)
    output_images[idx3(batch_index, output_row, output_col, height, width)] = acc;
}


// Host wrapper: copy kernel into constant memory, compute shared memory size, and launch
void conv2d_variant2(const float* input, const float* kernel, float* output,
                     int batch_size, int height, int width, int kernel_size,
                     cudaStream_t stream) {
    if (kernel_size > MAX_KERNEL_SIZE) {
        fprintf(stderr, "Error: kernel_size %d exceeds MAX_KERNEL_SIZE %d\n",
                kernel_size, MAX_KERNEL_SIZE);
        return;
    }

    // copy kernel weights to constant memory
    size_t kernel_bytes = kernel_size * kernel_size * sizeof(float);
    cudaError_t err = cudaMemcpyToSymbol(const_kernel, kernel, kernel_bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpyToSymbol failed: %s\n", cudaGetErrorString(err));
        return;
    }

    // compute shared memory size and check device limit
    const int radius = (kernel_size - 1) / 2;
    const int input_tile_width = TILE_WIDTH + 2 * radius;
    const int input_tile_height = TILE_HEIGHT + 2 * radius;
    size_t shared_mem_bytes = size_t(input_tile_width) * size_t(input_tile_height) * sizeof(float);

    int dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, dev);
    if (shared_mem_bytes > props.sharedMemPerBlock) {
        fprintf(stderr, "Error: Required shared memory (%zu bytes) exceeds device limit (%zu bytes)\n",
                shared_mem_bytes, (size_t)props.sharedMemPerBlock);
        return;
    }

    dim3 threads_per_block(TILE_WIDTH, TILE_HEIGHT, 1);
    dim3 blocks_per_grid(
        (width  + TILE_WIDTH  - 1) / TILE_WIDTH,
        (height + TILE_HEIGHT - 1) / TILE_HEIGHT,
        batch_size
    );

    kernel_conv2d_variant2<<<blocks_per_grid, threads_per_block, shared_mem_bytes, stream>>>(
        input, output, batch_size, height, width, kernel_size
    );

    // check launch
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
    }
}


// =============================================================================
// VARIANT 3: REGISTER-LEVEL OPTIMIZATION AND DATA LOCALITY
// =============================================================================
//
// In this task, you will investigate the role of the GPU’s register file and
// how exploiting data locality at the thread level can further improve
// performance beyond what shared and global memory optimizations achieve.
//
// Begin by profiling your previous variant and examine metrics related to
// register utilization, instruction-level parallelism (ILP), and arithmetic
// efficiency. Observe how many registers are used per thread and whether
// memory operations still dominate execution time.
//
// Next, study how the GPU register file serves as the fastest storage resource
// available to each thread. Think about ways to reuse data already loaded into
// registers to reduce redundant memory accesses and improve computational
// intensity. Consider whether each thread could perform more useful work by
// computing multiple nearby output elements rather than just one.
//
// Modify the kernel to take advantage of this thread-level reuse and the
// available registers. After your optimization, re-profile and report changes
// in achieved FLOP/s, register utilization, and memory bandwidth usage.
//
// Finally, discuss in your report how locality within the register file and
// the reuse of data across computations can reduce memory pressure and improve
// throughput. Relate your findings to the GPU’s execution model and to the
// balance between register usage, occupancy, and ILP.
//
// =============================================================================

#define A 2
#define B 2

__global__ void conv2d_variant3_kernel(const float* __restrict__ input,
                   float* __restrict__ output,
                   int n, int h, int w, int k) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int batch_index = blockIdx.z;

    int radius = (k - 1) / 2;
    extern __shared__ float sm[];

    int tid = ty * TILE_WIDTH + tx;
    int tw = TILE_WIDTH * B + 2 * radius;
    int th = TILE_HEIGHT * A + 2 * radius;
    int total = tw * th;
    int tpb = TILE_WIDTH * TILE_HEIGHT;

    for (int i = tid; i < total; i += tpb) {
        int rr = i / tw;
        int cc = i % tw;
        int gr = blockIdx.y * (TILE_HEIGHT * A) - radius + rr;
        int gc = blockIdx.x * (TILE_WIDTH  * B) - radius + cc;

        float v = 0.0f;
        if (gr >= 0 && gr < h && gc >= 0 && gc < w && batch_index < n) {
            v = input[idx3(batch_index, gr, gc, h, w)];
        }
        sm[rr * tw + cc] = v;
    }

    __syncthreads();

    float q[A][B];
    #pragma unroll
    for (int i = 0; i < A; i++) {
        #pragma unroll
        for (int j = 0; j < B; j++) {
            q[i][j] = 0.0f;
        }
    }

    for (int kr = 0; kr < k; kr++) {
        for (int kc = 0; kc < k; kc++) {
            float wv = const_kernel[kr * k + kc];
            #pragma unroll
            for (int i = 0; i < A; i++) {
                #pragma unroll
                for (int j = 0; j < B; j++) {
                    int sr = ty * A + i + kr;
                    int sc = tx * B + j + kc;
                    q[i][j] += sm[sr * tw + sc] * wv;
                }
            }
        }
    }

    #pragma unroll
    for (int i = 0; i < A; i++) {
        #pragma unroll
        for (int j = 0; j < B; j++) {
            int rr = blockIdx.y * (TILE_HEIGHT * A) + ty * A + i;
            int cc = blockIdx.x * (TILE_WIDTH  * B) + tx * B + j;
            if (batch_index < n && rr < h && cc < w) {
                output[idx3(batch_index, rr, cc, h, w)] = q[i][j];
            }
        }
    }
}

// void conv2d_variant3(const float* input, const float* kernel, float* output,
//                      int batch_size, int height, int width, int kernel_size,
//                      cudaStream_t stream) 
void conv2d_variant3(const float* input, const float* kernel, float* output,
        int batch_size, int height, int width, int kernel_size, cudaStream_t stream) {
    if (kernel_size > MAX_KERNEL_SIZE) return;

    cudaMemcpyToSymbol(const_kernel, kernel, kernel_size * kernel_size * sizeof(float));

    int radius = (kernel_size - 1) / 2;

    size_t sm_bytes =
        (TILE_WIDTH * B + 2 * radius) *
        (TILE_HEIGHT * A + 2 * radius) *
        sizeof(float);

    cudaDeviceProp p;
    cudaGetDeviceProperties(&p, 0);
    if (sm_bytes > p.sharedMemPerBlock) return;

    dim3 th(TILE_WIDTH, TILE_HEIGHT);
    dim3 bl(
        (width + TILE_WIDTH  * B - 1) / (TILE_WIDTH  * B),
        (height + TILE_HEIGHT * A - 1) / (TILE_HEIGHT * A),
        batch_size
    );

    conv2d_variant3_kernel<<<bl, th, sm_bytes, stream>>>(input, output, batch_size, height, width, kernel_size);
}

// =============================================================================
// BONUS: MULTI-STREAM CONCURRENT EXECUTION
// =============================================================================
//
// In this task, use Nsight Systems to understand the end-to-end timeline and
// then improve throughput by overlapping independent work with CUDA streams.
//
// GOAL:
// - Reduce idle gaps on the copy and compute engines by overlapping operations
//   (e.g., host to device transfers with kernel execution) across a large batch.
//
// WHAT TO EXAMINE IN NSIGHT SYSTEMS (BEFORE):
// - Are H2D/D2H copies serialized with kernel launches?
// - Do copy engines (C/E) or SMs sit idle between batches?
// - Where are the longest gaps on the timeline (host prep, copies, kernels)?
//
// WHAT TO MEASURE (before & after):
// - End-to-end time per full batch; GPU utilization (%), copy engine utilization.
// - Degree of overlap visible on the NSYS timeline (copies concurrent with kernels).
// - Any change in kernel performance (avoid starving compute with too many small chunks).
//
// =============================================================================


void conv2d_variant4(const float* input, const float* kernel, float* output,
                     int batch_size, int height, int width, int kernel_size,
                     cudaStream_t stream) {
    // TODO: Configure and launch

    // Your code here
}

// =============================================================================
// BONUS ROUND
// =============================================================================


void conv2d_bonus(const float* input, const float* kernel1, const float* kernel2,
                  float* output, int batch_size, int height, int width,
                  int kernel_size, cudaStream_t stream) {
    // TODO: Implement multi-stream version
    // You can reuse any of your previous kernels

    // Your code here
}