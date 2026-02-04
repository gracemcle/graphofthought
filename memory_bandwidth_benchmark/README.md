# Memory Bandwidth Benchmark Suite

A CUDA benchmark suite designed to calibrate penalty factors for memory-bound kernel performance prediction using the Roofline model.

## Quick Start

```bash
# Check your GPU's compute capability
nvidia-smi --query-gpu=compute_cap --format=csv

# Build (adjust SM_ARCH for your GPU)
make SM_ARCH=sm_86    # RTX 3090, 3080, etc.
# make SM_ARCH=sm_80  # A100
# make SM_ARCH=sm_89  # RTX 4090, 4080, etc.
# make SM_ARCH=sm_90  # H100

# Run
make run
```

## Kernels Included

| Kernel | Description | Expected Behavior |
|--------|-------------|-------------------|
| **Coalesced Copy (float4)** | Vectorized sequential copy | ~85-95% peak BW (baseline) |
| **Coalesced Copy (naive)** | Non-vectorized sequential copy | ~70-85% peak BW |
| **Strided Copy** | Access every Nth element | Degrades with stride; stride=32 is worst |
| **Random Gather** | Read from random indices | 5-15% of peak (cache-dependent) |
| **Random Scatter** | Write to random indices | Similar to gather |
| **Reduction** | Sum all elements | Read-heavy; depends on occupancy |
| **Stencil 1D** | Average of neighbors | Tests cache reuse |
| **STREAM Triad** | a = b + s*c | Standard BW benchmark |

## Interpreting Results for Roofline Calibration

### Output Format

```
================================================================================
Kernel                              BW (GB/s)    Time (ms)   Efficiency
================================================================================
Coalesced Copy (float4)                 820.45       0.6234       91.2%
Strided Copy (stride=32)                 82.31       6.2134        9.1%
Random Gather                            45.23      11.3456        5.0%
...

================================================================================
Penalty Factors (for Roofline calibration)
================================================================================
Coalesced Copy (float4)             penalty = 1.00x (factor = 1.000)
Strided Copy (stride=32)            penalty = 9.97x (factor = 0.100)
Random Gather                       penalty = 18.14x (factor = 0.055)
```

### Using Penalty Factors in Your Graph-of-Thought Model

The key insight is that each memory access pattern introduces a multiplicative penalty:

```
predicted_bandwidth = peak_bandwidth × Π(penalty_factors)
```

#### Mapping Graph Nodes to Penalties

```python
# Example penalty lookup table (calibrate on YOUR GPU)
PENALTY_FACTORS = {
    # Coalescing
    'coalesced': 1.0,
    'stride_2': 0.85,
    'stride_4': 0.70,
    'stride_8': 0.50,
    'stride_16': 0.25,
    'stride_32': 0.10,  # Worst case: each thread in warp hits different cache line
    
    # Access pattern
    'sequential': 1.0,
    'random': 0.05,  # Highly variable, depends on working set vs cache
    
    # Vectorization
    'vectorized_float4': 1.0,
    'scalar': 0.85,
    
    # Working set size (relative to L2 cache)
    'fits_l2': 1.0,
    'exceeds_l2_2x': 0.95,
    'exceeds_l2_10x': 0.90,
    'exceeds_l2_100x': 0.85,
}

def predict_bandwidth(peak_bw, access_pattern, coalescing, vectorized, working_set_ratio):
    factor = 1.0
    factor *= PENALTY_FACTORS.get(access_pattern, 0.5)
    factor *= PENALTY_FACTORS.get(coalescing, 0.5)
    factor *= PENALTY_FACTORS.get('vectorized_float4' if vectorized else 'scalar', 0.85)
    # ... etc
    return peak_bw * factor
```

### What Each Benchmark Tells You

1. **Coalesced Copy**: Your baseline. If this doesn't hit ~90% peak, check:
   - ECC memory (reduces effective BW by ~10%)
   - Power/thermal throttling
   - Array size (too small → kernel launch overhead dominates)

2. **Strided Copy**: Quantifies the coalescing penalty curve. 
   - Stride 2-4: Partial cache line utilization
   - Stride 32: Each thread hits a different cache line → 32x memory traffic

3. **Random Gather/Scatter**: Your worst-case bound.
   - If your code has ANY indirect indexing, expect something between coalesced and random
   - Results are heavily cache-dependent: run with both L2-fitting and L2-exceeding sizes

4. **Stencil**: Tests whether your model correctly captures data reuse
   - Shared memory version should show benefit
   - Compare naive vs optimized to quantify shared memory value

5. **Reduction**: Tests high read/write ratio patterns
   - Useful for kernels that aggregate data

## Running Different Configurations

```bash
# Large arrays (exceeds L2 cache)
./bin/bandwidth_benchmark 512

# Small arrays (fits in L2 on most GPUs)
./bin/bandwidth_benchmark 32

# Tiny arrays (fits in L1)
./bin/bandwidth_benchmark 1
```

Run multiple sizes to understand cache effects:

```bash
for size in 1 4 16 32 64 128 256 512 1024; do
    echo "=== Array size: ${size} MB ==="
    ./bin/bandwidth_benchmark $size
done
```

## Profiling with Nsight Compute

For detailed analysis:

```bash
# Full profiling (slow but comprehensive)
ncu --set full -o profile ./bin/bandwidth_benchmark 64

# Quick memory metrics
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed \
./bin/bandwidth_benchmark 64
```

## Common GPU Peak Bandwidths (for reference)

| GPU | Peak BW (GB/s) | Notes |
|-----|----------------|-------|
| RTX 3090 | 936 | GDDR6X |
| RTX 4090 | 1008 | GDDR6X |
| A100 40GB | 1555 | HBM2 |
| A100 80GB | 2039 | HBM2e |
| H100 SXM | 3350 | HBM3 |
| H100 PCIe | 2000 | HBM2e |

Note: Actual achievable bandwidth is typically 80-90% of peak due to memory controller overhead.

## Extending the Suite

To add a new kernel pattern:

1. Add the kernel in `bandwidth_kernels.cu`
2. Add a benchmark call in `run_all_benchmarks()`
3. Calculate the correct `bytes_per_iteration` for accurate BW calculation

## Troubleshooting

**Low baseline bandwidth:**
- Check `nvidia-smi dmon` for power/thermal throttling
- Ensure ECC is accounted for (ECC reduces effective BW)
- Try larger array sizes

**Inconsistent results:**
- Increase warmup iterations
- Check for background GPU processes
- Use `nvidia-smi -pm 1` for persistence mode

**Kernel launch errors:**
- Verify SM_ARCH matches your GPU
- Check `nvidia-smi --query-gpu=compute_cap --format=csv`

## License

MIT - Use freely for research and commercial purposes.
