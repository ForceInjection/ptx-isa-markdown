# Reduction, Scan & Sort Samples

## 3.1 reduction (C++)

- **Path**: `cpp/2_Concepts_and_Techniques/reduction/reduction_kernel.cu`
- **URL**: <https://github.com/NVIDIA/cuda-samples/blob/master/cpp/2_Concepts_and_Techniques/reduction/reduction_kernel.cu>
- **Pattern**: 10 progressive reduction kernels — naive interleaved → sequential addressing → warp shuffle (`__shfl_down_sync`) → `__reduce_add_sync` (SM 8.0) → cooperative groups reduce. The definitive textbook on CUDA parallel reduction.
- **Arch**: All (SM 8.0+ for `__reduce_add_sync`)
- **Lines**: ~991

```cuda
// reduce6: warp shuffle — one of the most practical variants
template <class T> __global__ void reduce6(T *g_idata, T *g_odata, unsigned int n) {
    T *sdata = SharedMemory<T>();
    unsigned int tid = threadIdx.x, i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    T mySum = (i < n ? g_idata[i] : 0) + (i + blockDim.x < n ? g_idata[i + blockDim.x] : 0);
    sdata[tid] = mySum; __syncthreads();
    // Tree reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) sdata[tid] = mySum = mySum + sdata[tid + s];
        __syncthreads();
    }
    // Warp-level reduction with shuffle
    if (tid < 32) {
        unsigned int mask = __activemask();
        for (unsigned int offset = 16; offset > 0; offset >>= 1)
            mySum += __shfl_down_sync(mask, mySum, offset);
    }
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}
```

## 3.2 reduction (Python)

- **Path**: `python/2_CoreConcepts/reduction/reduction.py`
- **URL**: <https://github.com/NVIDIA/cuda-samples/blob/master/python/2_CoreConcepts/reduction/reduction.py>
- **Pattern**: Two-phase parallel reduction: block-level shared memory tree reduction → CPU partial sum combination. Template-typed for float/double/int.
- **Arch**: All
- **Lines**: ~485

```python
kernel_src = """
extern "C" __global__ void reduction_kernel(const T* input, T* output, int n) {
    extern __shared__ char smem[];
    T* sdata = (T*)smem;
    unsigned int tid = threadIdx.x, i = blockIdx.x * (blockDim.x * 2) + tid;
    T sum = (i + blockDim.x < n && i < n) ? input[i] + input[i + blockDim.x]
               : (i < n ? input[i] : 0);
    sdata[tid] = sum; __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) output[blockIdx.x] = sdata[0];
}
"""
```

## 3.3 parallelReduction (Python)

- **Path**: `python/2_CoreConcepts/parallelReduction/parallelReduction.py`
- **URL**: <https://github.com/NVIDIA/cuda-samples/blob/master/python/2_CoreConcepts/parallelReduction/parallelReduction.py>
- **Pattern**: Contrasts hand-written tree reduction with `cuda.compute.reduce_into()` (wraps CUB DeviceReduce) — education vs production usage side by side
- **Arch**: All
- **Lines**: ~375

```python
# Production: single-line reduction via cuda.compute (CUB backend)
from cuda.compute import reduce_into
reduce_into(result_buf, input_buf, op=OpKind.PLUS, stream=stream)
```

## 3.4 reductionMultiBlockCG

- **Path**: `cpp/2_Concepts_and_Techniques/reductionMultiBlockCG/reductionMultiBlockCG.cu`
- **URL**: <https://github.com/NVIDIA/cuda-samples/blob/master/cpp/2_Concepts_and_Techniques/reductionMultiBlockCG/reductionMultiBlockCG.cu>
- **Pattern**: Single-kernel reduction using Cooperative Groups `grid_group::sync()` — eliminates the second kernel pass
- **Arch**: SM 6.0+ (cooperative launch)
- **Lines**: ~144

```cuda
__global__ void reduce_grid(cg::grid_group grid, double *inputVec, double *outputVec) {
    cg::thread_block cta = cg::this_thread_block();
    double beta = /* block-level reduction */;
    // After all blocks finish, block 0 combines partial sums
    grid.sync();
    if (cta.thread_rank() == 0) {
        for (int i = 0; i < grid.num_blocks(); i++) beta += g_odata[i];
        *outputVec = beta;
    }
}
// Launch: cudaLaunchCooperativeKernel((void*)reduce_grid, grid, block, args, smem, stream);
```

## 3.5 shfl_scan

- **Path**: `cpp/2_Concepts_and_Techniques/shfl_scan/shfl_scan.cu`
- **URL**: <https://github.com/NVIDIA/cuda-samples/blob/master/cpp/2_Concepts_and_Techniques/shfl_scan/shfl_scan.cu>
- **Pattern**: Warp-level inclusive/exclusive prefix sum using `__shfl_up_sync` — avoids shared memory entirely for warp-sized scans
- **Arch**: SM 3.0+
- **Lines**: ~419

```cuda
__device__ int warp_scan_inclusive(int val, unsigned int mask) {
    #pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        int n = __shfl_up_sync(mask, val, offset, 32);
        if ((threadIdx.x & 31) >= offset) val += n;
    }
    return val;
}
```

## 3.6 radixSortThrust

- **Path**: `cpp/2_Concepts_and_Techniques/radixSortThrust/radixSortThrust.cu`
- **URL**: <https://github.com/NVIDIA/cuda-samples/blob/master/cpp/2_Concepts_and_Techniques/radixSortThrust/radixSortThrust.cu>
- **Pattern**: Production-quality key-value radix sort wrapping Thrust library
- **Arch**: All
- **Lines**: ~217

```cuda
void RadixSort(KeyT *keys_d, KeyT *keysSorted_d, ValueT *vals_d, ValueT *valsSorted_d, int N) {
    thrust::device_ptr<KeyT> d_keys(keys_d);
    thrust::device_ptr<ValueT> d_vals(vals_d);
    if (keysSorted_d != keys_d) thrust::copy(d_keys, d_keys + N, thrust::device_ptr<KeyT>(keysSorted_d));
    thrust::sort_by_key(thrust::device_ptr<KeyT>(keysSorted_d), thrust::device_ptr<KeyT>(keysSorted_d + N), d_vals);
}
```
