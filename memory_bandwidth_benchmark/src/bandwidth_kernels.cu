/**
 * Memory Bandwidth Benchmark Suite for Roofline Model Calibration
 * 
 * This suite provides kernels that isolate different memory access patterns
 * to help calibrate penalty factors for runtime prediction models.
 * 
 * Kernels:
 *   1. Coalesced copy (baseline - should hit ~90% peak BW)
 *   2. Strided copy (various strides to measure coalescing penalty)
 *   3. Random gather (indirect access pattern)
 *   4. Reduction (read-heavy workload)
 *   5. Stencil 1D (cache reuse potential)
 *   6. Scatter (write coalescing test)
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <chrono>
#include <random>
#include <algorithm>
#include <vector>
#include <string>
#include <iomanip>
#include <iostream>

// Error checking macro
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// ============================================================================
// Kernel 1: Coalesced Copy (Baseline)
// Expected: ~90% of peak bandwidth
// Memory pattern: Sequential reads, sequential writes
// ============================================================================
__global__ void coalesced_copy(float* __restrict__ dst,
                                const float* __restrict__ src,
                                size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    
    // Process 4 elements at a time for better bandwidth utilization
    size_t n4 = n / 4;
    for (size_t i = idx; i < n4; i += stride) {
        float4 val = reinterpret_cast<const float4*>(src)[i];
        reinterpret_cast<float4*>(dst)[i] = val;
    }
    
    // Handle remainder
    size_t base = n4 * 4;
    for (size_t i = base + idx; i < n; i += stride) {
        dst[i] = src[i];
    }
}

// Non-vectorized version for comparison
__global__ void coalesced_copy_naive(float* __restrict__ dst,
                                      const float* __restrict__ src,
                                      size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    
    for (size_t i = idx; i < n; i += stride) {
        dst[i] = src[i];
    }
}

// ============================================================================
// Kernel 2: Strided Copy
// Expected: Performance degrades with stride, worst at stride=32 (warp size)
// Memory pattern: Non-coalesced reads/writes
// ============================================================================
__global__ void strided_copy(float* __restrict__ dst,
                              const float* __restrict__ src,
                              size_t n,
                              int stride_factor) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t grid_stride = blockDim.x * gridDim.x;
    
    // Each thread accesses elements stride_factor apart
    // This breaks coalescing when stride_factor > 1
    size_t effective_n = n / stride_factor;
    
    for (size_t i = tid; i < effective_n; i += grid_stride) {
        size_t src_idx = i * stride_factor;
        size_t dst_idx = i * stride_factor;
        if (src_idx < n && dst_idx < n) {
            dst[dst_idx] = src[src_idx];
        }
    }
}

// Warp-aware strided access (isolates intra-warp non-coalescing)
__global__ void warp_strided_copy(float* __restrict__ dst,
                                   const float* __restrict__ src,
                                   size_t n,
                                   int stride_factor) {
    size_t warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    size_t lane_id = threadIdx.x % 32;
    size_t num_warps = (gridDim.x * blockDim.x) / 32;
    
    // Each warp processes a contiguous chunk, but lanes access with stride
    size_t elements_per_warp = 32;  // Each warp handles 32 elements
    
    for (size_t warp_offset = warp_id; warp_offset < n / elements_per_warp; warp_offset += num_warps) {
        size_t base = warp_offset * elements_per_warp;
        size_t idx = base + (lane_id * stride_factor) % elements_per_warp;
        if (idx < n) {
            dst[idx] = src[idx];
        }
    }
}

// ============================================================================
// Kernel 3: Random Gather
// Expected: Very poor bandwidth (10-50x penalty)
// Memory pattern: Random reads, sequential writes
// ============================================================================
__global__ void random_gather(float* __restrict__ dst,
                               const float* __restrict__ src,
                               const int* __restrict__ indices,
                               size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    
    for (size_t i = idx; i < n; i += stride) {
        dst[i] = src[indices[i]];
    }
}

// Random scatter (random writes, sequential reads)
__global__ void random_scatter(float* __restrict__ dst,
                                const float* __restrict__ src,
                                const int* __restrict__ indices,
                                size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    
    for (size_t i = idx; i < n; i += stride) {
        dst[indices[i]] = src[i];
    }
}

// ============================================================================
// Kernel 4: Reduction
// Expected: Good bandwidth on reads, minimal writes
// Memory pattern: Many reads, few writes (high read/write ratio)
// ============================================================================
__global__ void reduction_kernel(float* __restrict__ output,
                                  const float* __restrict__ input,
                                  size_t n) {
    extern __shared__ float sdata[];
    
    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    size_t grid_stride = blockDim.x * 2 * gridDim.x;
    
    float sum = 0.0f;
    
    // Grid-stride loop for large arrays
    for (size_t i = idx; i < n; i += grid_stride) {
        sum += input[i];
        if (i + blockDim.x < n) {
            sum += input[i + blockDim.x];
        }
    }
    
    sdata[tid] = sum;
    __syncthreads();
    
    // Reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Warp-level reduction (no sync needed)
    if (tid < 32) {
        volatile float* smem = sdata;
        if (blockDim.x >= 64) smem[tid] += smem[tid + 32];
        if (blockDim.x >= 32) smem[tid] += smem[tid + 16];
        if (blockDim.x >= 16) smem[tid] += smem[tid + 8];
        if (blockDim.x >= 8) smem[tid] += smem[tid + 4];
        if (blockDim.x >= 4) smem[tid] += smem[tid + 2];
        if (blockDim.x >= 2) smem[tid] += smem[tid + 1];
    }
    
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

// ============================================================================
// Kernel 5: 1D Stencil
// Expected: Better than random, cache reuse helps
// Memory pattern: Overlapping reads (data reuse), sequential writes
// ============================================================================
__global__ void stencil_1d(float* __restrict__ dst,
                            const float* __restrict__ src,
                            size_t n,
                            int radius) {
    extern __shared__ float smem[];
    
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int block_start = blockIdx.x * blockDim.x;
    
    // Load data into shared memory with halo
    int smem_idx = tid + radius;
    if (gid < n) {
        smem[smem_idx] = src[gid];
    }
    
    // Load left halo
    if (tid < radius) {
        int halo_idx = block_start - radius + tid;
        smem[tid] = (halo_idx >= 0) ? src[halo_idx] : 0.0f;
    }
    
    // Load right halo
    if (tid >= blockDim.x - radius) {
        int halo_idx = block_start + blockDim.x + (tid - (blockDim.x - radius));
        int smem_halo_idx = blockDim.x + radius + (tid - (blockDim.x - radius));
        smem[smem_halo_idx] = (halo_idx < n) ? src[halo_idx] : 0.0f;
    }
    
    __syncthreads();
    
    // Compute stencil
    if (gid < n) {
        float result = 0.0f;
        for (int i = -radius; i <= radius; i++) {
            result += smem[smem_idx + i];
        }
        dst[gid] = result / (2 * radius + 1);
    }
}

// Naive stencil without shared memory (for comparison)
__global__ void stencil_1d_naive(float* __restrict__ dst,
                                  const float* __restrict__ src,
                                  size_t n,
                                  int radius) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (gid < n) {
        float result = 0.0f;
        for (int i = -radius; i <= radius; i++) {
            int idx = gid + i;
            if (idx >= 0 && idx < n) {
                result += src[idx];
            }
        }
        dst[gid] = result / (2 * radius + 1);
    }
}

// ============================================================================
// Kernel 6: STREAM Triad (standard benchmark)
// a[i] = b[i] + scalar * c[i]
// Expected: Reference bandwidth benchmark
// ============================================================================
__global__ void stream_triad(float* __restrict__ a,
                              const float* __restrict__ b,
                              const float* __restrict__ c,
                              float scalar,
                              size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    
    // Vectorized version
    size_t n4 = n / 4;
    for (size_t i = idx; i < n4; i += stride) {
        float4 bv = reinterpret_cast<const float4*>(b)[i];
        float4 cv = reinterpret_cast<const float4*>(c)[i];
        float4 av;
        av.x = bv.x + scalar * cv.x;
        av.y = bv.y + scalar * cv.y;
        av.z = bv.z + scalar * cv.z;
        av.w = bv.w + scalar * cv.w;
        reinterpret_cast<float4*>(a)[i] = av;
    }
}

// ============================================================================
// Benchmark Infrastructure
// ============================================================================

struct BenchmarkResult {
    std::string name;
    double bandwidth_gbps;
    double time_ms;
    size_t bytes_transferred;
    double efficiency;  // vs peak bandwidth
};

class BandwidthBenchmark {
public:
    BandwidthBenchmark(size_t array_size_mb = 256) {
        // Get device properties
        CUDA_CHECK(cudaGetDevice(&device_));
        CUDA_CHECK(cudaGetDeviceProperties(&props_, device_));
        
        // Calculate peak bandwidth (in GB/s)
        // memory_clock_rate is in kHz, memory_bus_width in bits
        // Factor of 2 for DDR
        peak_bandwidth_gbps_ = 2.0 * props_.memoryClockRate * 1e3 * 
                               (props_.memoryBusWidth / 8) / 1e9;
        
        // Array size
        array_size_ = array_size_mb * 1024 * 1024 / sizeof(float);
        array_bytes_ = array_size_ * sizeof(float);
        
        printf("=======================================================\n");
        printf("GPU: %s\n", props_.name);
        printf("Compute Capability: %d.%d\n", props_.major, props_.minor);
        printf("Peak Memory Bandwidth: %.2f GB/s\n", peak_bandwidth_gbps_);
        printf("L2 Cache Size: %d KB\n", props_.l2CacheSize / 1024);
        printf("Array Size: %zu MB (%.2f billion elements)\n", 
               array_bytes_ / (1024 * 1024), array_size_ / 1e9);
        printf("=======================================================\n\n");
        
        // Allocate memory
        CUDA_CHECK(cudaMalloc(&d_src_, array_bytes_));
        CUDA_CHECK(cudaMalloc(&d_dst_, array_bytes_));
        CUDA_CHECK(cudaMalloc(&d_extra_, array_bytes_));
        CUDA_CHECK(cudaMalloc(&d_indices_, array_size_ * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_reduction_out_, 1024 * sizeof(float)));
        
        // Initialize source data
        std::vector<float> h_src(array_size_);
        for (size_t i = 0; i < array_size_; i++) {
            h_src[i] = static_cast<float>(i % 1000) / 1000.0f;
        }
        CUDA_CHECK(cudaMemcpy(d_src_, h_src.data(), array_bytes_, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_extra_, h_src.data(), array_bytes_, cudaMemcpyHostToDevice));
        
        // Initialize random indices for gather/scatter
        std::vector<int> h_indices(array_size_);
        std::mt19937 rng(42);
        for (size_t i = 0; i < array_size_; i++) {
            h_indices[i] = rng() % array_size_;
        }
        CUDA_CHECK(cudaMemcpy(d_indices_, h_indices.data(), 
                              array_size_ * sizeof(int), cudaMemcpyHostToDevice));
        
        // Create events for timing
        CUDA_CHECK(cudaEventCreate(&start_));
        CUDA_CHECK(cudaEventCreate(&stop_));
    }
    
    ~BandwidthBenchmark() {
        cudaFree(d_src_);
        cudaFree(d_dst_);
        cudaFree(d_extra_);
        cudaFree(d_indices_);
        cudaFree(d_reduction_out_);
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }
    
    // Run a kernel multiple times and measure bandwidth
    template<typename KernelFunc>
    BenchmarkResult benchmark(const std::string& name, 
                              KernelFunc kernel,
                              size_t bytes_per_iteration,
                              int warmup = 5,
                              int iterations = 20) {
        // Warmup
        for (int i = 0; i < warmup; i++) {
            kernel();
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Timed runs
        CUDA_CHECK(cudaEventRecord(start_));
        for (int i = 0; i < iterations; i++) {
            kernel();
        }
        CUDA_CHECK(cudaEventRecord(stop_));
        CUDA_CHECK(cudaEventSynchronize(stop_));
        
        float time_ms;
        CUDA_CHECK(cudaEventElapsedTime(&time_ms, start_, stop_));
        time_ms /= iterations;
        
        double bandwidth = (bytes_per_iteration / 1e9) / (time_ms / 1e3);
        double efficiency = bandwidth / peak_bandwidth_gbps_ * 100.0;
        
        return {name, bandwidth, time_ms, bytes_per_iteration, efficiency};
    }
    
    void run_all_benchmarks() {
        std::vector<BenchmarkResult> results;
        
        int block_size = 256;
        int num_blocks = (array_size_ + block_size - 1) / block_size;
        num_blocks = std::min(num_blocks, props_.multiProcessorCount * 32);
        
        printf("Launch config: %d blocks x %d threads\n\n", num_blocks, block_size);
        
        // 1. Coalesced copy (vectorized)
        results.push_back(benchmark(
            "Coalesced Copy (float4)",
            [&]() { coalesced_copy<<<num_blocks, block_size>>>(d_dst_, d_src_, array_size_); },
            2 * array_bytes_  // read + write
        ));
        
        // 1b. Coalesced copy (naive)
        results.push_back(benchmark(
            "Coalesced Copy (naive)",
            [&]() { coalesced_copy_naive<<<num_blocks, block_size>>>(d_dst_, d_src_, array_size_); },
            2 * array_bytes_
        ));
        
        // 2. Strided copies
        for (int stride : {2, 4, 8, 16, 32}) {
            results.push_back(benchmark(
                "Strided Copy (stride=" + std::to_string(stride) + ")",
                [&, stride]() { 
                    strided_copy<<<num_blocks, block_size>>>(d_dst_, d_src_, array_size_, stride); 
                },
                2 * array_bytes_ / stride  // fewer elements accessed
            ));
        }
        
        // 3. Random gather
        results.push_back(benchmark(
            "Random Gather",
            [&]() { random_gather<<<num_blocks, block_size>>>(d_dst_, d_src_, d_indices_, array_size_); },
            2 * array_bytes_ + array_size_ * sizeof(int)  // src read (random) + dst write + indices read
        ));
        
        // 3b. Random scatter
        results.push_back(benchmark(
            "Random Scatter",
            [&]() { random_scatter<<<num_blocks, block_size>>>(d_dst_, d_src_, d_indices_, array_size_); },
            2 * array_bytes_ + array_size_ * sizeof(int)
        ));
        
        // 4. Reduction
        int reduction_blocks = std::min(1024, num_blocks);
        results.push_back(benchmark(
            "Reduction",
            [&]() { 
                reduction_kernel<<<reduction_blocks, block_size, block_size * sizeof(float)>>>(
                    d_reduction_out_, d_src_, array_size_); 
            },
            array_bytes_  // read only (output is tiny)
        ));
        
        // 5. Stencil 1D
        int stencil_radius = 4;
        int smem_size = (block_size + 2 * stencil_radius) * sizeof(float);
        results.push_back(benchmark(
            "Stencil 1D (radius=4, shared mem)",
            [&]() { 
                stencil_1d<<<num_blocks, block_size, smem_size>>>(
                    d_dst_, d_src_, array_size_, stencil_radius); 
            },
            2 * array_bytes_  // effective: read + write (data reuse in shared mem)
        ));
        
        results.push_back(benchmark(
            "Stencil 1D (radius=4, naive)",
            [&]() { 
                stencil_1d_naive<<<num_blocks, block_size>>>(
                    d_dst_, d_src_, array_size_, stencil_radius); 
            },
            2 * array_bytes_
        ));
        
        // 6. STREAM Triad
        results.push_back(benchmark(
            "STREAM Triad",
            [&]() { 
                stream_triad<<<num_blocks, block_size>>>(
                    d_dst_, d_src_, d_extra_, 2.0f, array_size_); 
            },
            3 * array_bytes_  // 2 reads + 1 write
        ));
        
        // Print results
        print_results(results);
    }
    
    void print_results(const std::vector<BenchmarkResult>& results) {
        printf("\n");
        printf("================================================================================\n");
        printf("%-35s %12s %12s %12s\n", "Kernel", "BW (GB/s)", "Time (ms)", "Efficiency");
        printf("================================================================================\n");
        
        for (const auto& r : results) {
            printf("%-35s %12.2f %12.4f %11.1f%%\n", 
                   r.name.c_str(), r.bandwidth_gbps, r.time_ms, r.efficiency);
        }
        
        printf("================================================================================\n");
        printf("Peak Theoretical Bandwidth: %.2f GB/s\n", peak_bandwidth_gbps_);
        printf("================================================================================\n");
        
        // Print penalty factors relative to best coalesced copy
        printf("\n");
        printf("================================================================================\n");
        printf("Penalty Factors (for Roofline calibration)\n");
        printf("================================================================================\n");
        
        double baseline = results[0].bandwidth_gbps;  // Coalesced copy as baseline
        
        for (const auto& r : results) {
            double penalty = baseline / r.bandwidth_gbps;
            printf("%-35s penalty = %.2fx (factor = %.3f)\n", 
                   r.name.c_str(), penalty, 1.0 / penalty);
        }
        printf("================================================================================\n");
    }
    
private:
    int device_;
    cudaDeviceProp props_;
    double peak_bandwidth_gbps_;
    
    size_t array_size_;
    size_t array_bytes_;
    
    float* d_src_;
    float* d_dst_;
    float* d_extra_;
    int* d_indices_;
    float* d_reduction_out_;
    
    cudaEvent_t start_, stop_;
};

// ============================================================================
// Main
// ============================================================================
int main(int argc, char** argv) {
    size_t array_size_mb = 256;  // Default 256 MB
    
    if (argc > 1) {
        array_size_mb = std::atoi(argv[1]);
    }
    
    printf("Memory Bandwidth Benchmark Suite\n");
    printf("Usage: %s [array_size_mb]\n\n", argv[0]);
    
    BandwidthBenchmark benchmark(array_size_mb);
    benchmark.run_all_benchmarks();
    
    return 0;
}
