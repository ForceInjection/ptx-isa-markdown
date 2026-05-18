# Performance Optimization Samples

## 7.1 transpose

- **Path**: `cpp/6_Performance/transpose/transpose.cu`
- **URL**: <https://github.com/NVIDIA/cuda-samples/blob/master/cpp/6_Performance/transpose/transpose.cu>
- **Pattern**: 7 progressive transpose kernels: naive → coalesced → no bank conflicts → diagonal reorder → fine-grain → coarse-grain. The definitive textbook on shared memory optimization patterns.
- **Arch**: All
- **Lines**: ~642

```cuda
#define TILE_DIM 32
#define BLOCK_ROWS 16
__global__ void transpose_coalesced(float *odata, const float *idata, int w, int h) {
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];  // +1 = shared memory padding to avoid bank conflicts
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    #pragma unroll
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
        tile[threadIdx.y + j][threadIdx.x] = idata[(y + j) * w + x];
    __syncthreads();
    // Swizzle thread indices for coalesced write
    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;
    #pragma unroll
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
        odata[(y + j) * w + x] = tile[threadIdx.x][threadIdx.y + j];
}
```

## 7.2 launchConfigTuning (Python)

- **Path**: `python/2_CoreConcepts/launchConfigTuning/launchConfigTuning.py`
- **URL**: <https://github.com/NVIDIA/cuda-samples/blob/master/python/2_CoreConcepts/launchConfigTuning/launchConfigTuning.py>
- **Pattern**: Systematic block-size benchmark for memory-bound and compute-bound kernels, identifying optimal `blockDim` across 32–1024
- **Arch**: All
- **Lines**: ~388

```python
for block_size in [32, 64, 128, 256, 512, 1024]:
    config = LaunchConfig(grid=((N + block_size - 1) // block_size,), block=(block_size,))
    start = device.create_event(); stop = device.create_event()
    stream.record(start)
    launch(stream, config, kernel, buf, N)
    stream.record(stop); stop.sync()
    elapsed = stop - start
```

## 7.3 warpAggregatedAtomicsCG

- **Path**: `cpp/3_CUDA_Features/warpAggregatedAtomicsCG/warpAggregatedAtomicsCG.cu`
- **URL**: <https://github.com/NVIDIA/cuda-samples/blob/master/cpp/3_CUDA_Features/warpAggregatedAtomicsCG/warpAggregatedAtomicsCG.cu>
- **Pattern**: Warp-aggregated atomic operations using cooperative groups — only one thread per warp performs the atomic, then shuffles the result. Reduces atomic contention by up to 32×.
- **Arch**: All
- **Lines**: ~313

```cuda
__global__ void warp_aggregated_atomic_inc(unsigned int *ctr) {
    cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cg::this_thread_block());
    int value;
    if (tile32.thread_rank() == 0) value = atomicAdd(ctr, tile32.size());  // leader only
    value = tile32.shfl(value, 0) + tile32.thread_rank();  // broadcast + offset
}
```

## 7.4 kernelNsysProfile (Python)

- **Path**: `python/1_GettingStarted/kernelNsysProfile/kernelNsysProfile.py`
- **URL**: <https://github.com/NVIDIA/cuda-samples/blob/master/python/1_GettingStarted/kernelNsysProfile/kernelNsysProfile.py>
- **Pattern**: NVTX-annotated profiling of cuda.core kernels for Nsight Systems timeline analysis
- **Arch**: All
- **Lines**: ~327

```python
import nvtx
with nvtx.annotate("vector_add_kernel", color="blue"):
    launch(stream, config, vec_add_kernel, a_buf, b_buf, c_buf, n)
```
