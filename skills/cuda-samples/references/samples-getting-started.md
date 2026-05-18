# Getting Started & Memory Management Samples

## 1.1 vectorAdd

- **Path**: `cpp/0_Introduction/vectorAdd/vectorAdd.cu`
- **URL**: <https://github.com/NVIDIA/cuda-samples/blob/master/cpp/0_Introduction/vectorAdd/vectorAdd.cu>
- **Pattern**: First CUDA kernel — `cudaMalloc` → `cudaMemcpy` → kernel launch → `cudaMemcpy` → `cudaFree` lifecycle
- **Arch**: All
- **Lines**: ~140

```cuda
__global__ void vectorAdd(const float *A, const float *B, float *C, int numElements) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements) C[i] = A[i] + B[i] + 0.0f;
}

int main(void) {
    cudaMalloc(&d_A, size); cudaMalloc(&d_B, size); cudaMalloc(&d_C, size);
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
}
```

## 1.2 vectorAdd (Python)

- **Path**: `python/1_GettingStarted/vectorAdd/vectorAdd.py`
- **URL**: <https://github.com/NVIDIA/cuda-samples/blob/master/python/1_GettingStarted/vectorAdd/vectorAdd.py>
- **Pattern**: Modern `cuda.core` Python workflow — kernel as string → `Program` compile → `LaunchConfig` → CuPy buffers
- **Arch**: All
- **Lines**: ~196

```python
kernel = """
extern "C" __global__ void vector_add(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = a[idx] + b[idx];
}
"""
program = Program(kernel, code_type="c++")
module = program.compile("cubin", options=ProgramOptions(std="c++17", arch=arch))
kernel_obj = module.get_kernel("vector_add")
launch(stream, config, kernel_obj, a_buf, b_buf, c_buf, n)
```

## 1.3 asyncAPI

- **Path**: `cpp/0_Introduction/asyncAPI/asyncAPI.cu`
- **URL**: <https://github.com/NVIDIA/cuda-samples/blob/master/cpp/0_Introduction/asyncAPI/asyncAPI.cu>
- **Pattern**: CUDA event-based GPU timing with `cudaEventCreate` / `cudaEventRecord` / `cudaEventSynchronize` / `cudaEventElapsedTime`
- **Arch**: All
- **Lines**: ~144

```cuda
cudaEvent_t start, stop; cudaEventCreate(&start); cudaEventCreate(&stop);
cudaEventRecord(start, 0);
kernel<<<grid, block>>>(d_data, N);
cudaEventRecord(stop, 0);
cudaEventSynchronize(stop);
float ms; cudaEventElapsedTime(&ms, start, stop);
```

## 1.4 streamOrderedAllocation

- **Path**: `cpp/2_Concepts_and_Techniques/streamOrderedAllocation/streamOrderedAllocation.cu`
- **URL**: <https://github.com/NVIDIA/cuda-samples/blob/master/cpp/2_Concepts_and_Techniques/streamOrderedAllocation/streamOrderedAllocation.cu>
- **Pattern**: `cudaMallocAsync` / `cudaFreeAsync` with `cudaMemPool_t` — stream-ordered allocation that eliminates synchronous `cudaMalloc` overhead
- **Arch**: All (CUDA 11.2+)
- **Lines**: ~235

```cuda
int old_pool_threshold;
cudaMemPool_t mem_pool;
cudaDeviceGetMemPool(&mem_pool, device);
cudaMemPoolGetAttribute(mem_pool, cudaMemPoolAttrReleaseThreshold, &old_pool_threshold);
// Allocate and free within stream — no host-side synchronization
cudaMallocAsync(&ptr, size, stream);
kernel<<<grid, block, 0, stream>>>(ptr);
cudaFreeAsync(ptr, stream);
```

## 1.5 UnifiedMemoryPerf

- **Path**: `cpp/6_Performance/UnifiedMemoryPerf/UnifiedMemoryPerf.cu`
- **URL**: <https://github.com/NVIDIA/cuda-samples/blob/master/cpp/6_Performance/UnifiedMemoryPerf/UnifiedMemoryPerf.cu>
- **Pattern**: Compares Unified Memory (with/without prefetch hints), zero-copy, pageable, and pinned memory performance for matrix multiplication
- **Arch**: All
- **Lines**: ~350

```cuda
cudaMallocManaged(&um_data, bytes);
cudaMemPrefetchAsync(um_data, bytes, target_device, stream);  // hint for migration
kernel<<<grid, block, 0, stream>>>(um_data);
cudaStreamSynchronize(stream);
```

## 1.6 memoryResources (Python)

- **Path**: `python/2_CoreConcepts/memoryResources/memoryResources.py`
- **URL**: <https://github.com/NVIDIA/cuda-samples/blob/master/python/2_CoreConcepts/memoryResources/memoryResources.py>
- **Pattern**: Side-by-side comparison of `DeviceMemoryResource`, `PinnedMemoryResource`, `ManagedMemoryResource` — the definitive reference for cuda.core memory model
- **Arch**: All
- **Lines**: ~248

```python
# Device memory (standard GPU allocation)
dev_mr = DeviceMemoryResource(device, options=DeviceMemoryResourceOptions())
buf = dev_mr.allocate(size)
# Pinned memory (for async H2D/D2H)
pin_mr = PinnedMemoryResource(options=PinnedMemoryResourceOptions())
buf = pin_mr.allocate(size)
# Managed memory (automatic migration)
mng_mr = ManagedMemoryResource(options=ManagedMemoryResourceOptions())
buf = mng_mr.allocate(size)
```
