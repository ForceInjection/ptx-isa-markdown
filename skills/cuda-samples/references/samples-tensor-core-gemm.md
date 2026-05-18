# Tensor Core GEMM Samples

## 4.1 cudaTensorCoreGemm

- **Path**: `cpp/3_CUDA_Features/cudaTensorCoreGemm/cudaTensorCoreGemm.cu`
- **URL**: <https://github.com/NVIDIA/cuda-samples/blob/master/cpp/3_CUDA_Features/cudaTensorCoreGemm/cudaTensorCoreGemm.cu>
- **Pattern**: FP16 GEMM using `nvcuda::wmma::mma_sync` with 128×128 CTA tiling, shared memory skew-padding to avoid bank conflicts, int4 vectorized global loads, double-buffer fallback for >48 KB smem devices
- **Arch**: SM 7.0+ (Volta)
- **Lines**: ~617

```cuda
// Core WMMA GEMM pattern — 8 warps per CTA, each computes 4×2 sub-tiles
wmma::fragment<wmma::accumulator, 16, 16, 16, float> c[WARP_COL_TILES][WARP_ROW_TILES];
for (int k_step = 0; k_step < K / (CHUNK_K * WMMA_K); k_step++) {
    // Load A and B tiles into shared memory (with skew for bank conflict avoidance)
    *A_tile_smem = A_global[idx_A];  // int4 vectorized load
    *B_tile_smem = B_global[idx_B];
    __syncthreads();
    // Each warp performs WMMA mma_sync on its sub-tile
    for (int i = 0; i < WARP_COL_TILES; i++)
        for (int j = 0; j < WARP_ROW_TILES; j++)
            wmma::mma_sync(c[i][j], a[i], b[j], c[i][j]);
    __syncthreads();
}
```

## 4.2 bf16TensorCoreGemm

- **Path**: `cpp/3_CUDA_Features/bf16TensorCoreGemm/bf16TensorCoreGemm.cu`
- **URL**: <https://github.com/NVIDIA/cuda-samples/blob/master/cpp/3_CUDA_Features/bf16TensorCoreGemm/bf16TensorCoreGemm.cu>
- **Pattern**: BF16 GEMM combining WMMA API with `__pipeline_memcpy_async` (async global→shared copy via cuda::pipeline) — the standard production pattern for Ampere+ AI workloads
- **Arch**: SM 8.0+ (Ampere)
- **Lines**: ~600

```cuda
// Async copy pipeline for overlapping global→shared loads with compute
cuda::pipeline<cuda::thread_scope_thread> pipe;
pipe.producer_acquire();
cuda::memcpy_async(smem_block, global_block, pipe, threadIdx.x);
pipe.producer_commit();
// ... compute using WMMA ...
pipe.consumer_wait();
// ... reuse smem for next tile ...
```

## 4.3 tf32TensorCoreGemm

- **Path**: `cpp/3_CUDA_Features/tf32TensorCoreGemm/tf32TensorCoreGemm.cu`
- **URL**: <https://github.com/NVIDIA/cuda-samples/blob/master/cpp/3_CUDA_Features/tf32TensorCoreGemm/tf32TensorCoreGemm.cu>
- **Pattern**: TF32 (tensor-float-32) GEMM — 19 bits of mantissa, 8× faster than FP32, the default math mode for cuDNN/PyTorch on Ampere+. Combines `nvcuda::wmma::precision::tf32` with `cuda::pipeline` for async shared memory copies.
- **Arch**: SM 8.0+
- **Lines**: ~853

```cuda
// TF32 WMMA GEMM with async copy pipeline — 128×128 CTA tile, 8 warps
wmma::fragment<wmma::matrix_a, 16, 16, 16, wmma::precision::tf32, wmma::row_major> a[WARP_COL_TILES];
wmma::fragment<wmma::matrix_b, 16, 16, 16, wmma::precision::tf32, wmma::col_major> b[WARP_ROW_TILES];
for (int k_step = 0; k_step < CHUNK_K; k_step++) {
    wmma::load_matrix_sync(a[i], tile_ptr, K * CHUNK_K + SKEW_FLOAT);
    // Convert float inputs to tf32 precision before mma
    for (int t = 0; t < a[i].num_elements; t++)
        a[i].x[t] = wmma::__float_to_tf32(a[i].x[t]);
    wmma::load_matrix_sync(b[j], tile_ptr, K * CHUNK_K + SKEW_FLOAT);
    wmma::mma_sync(c[i][j], a[i], b[j], c[i][j]);
}
```

## 4.4 dmmaTensorCoreGemm

- **Path**: `cpp/3_CUDA_Features/dmmaTensorCoreGemm/dmmaTensorCoreGemm.cu`
- **URL**: <https://github.com/NVIDIA/cuda-samples/blob/master/cpp/3_CUDA_Features/dmmaTensorCoreGemm/dmmaTensorCoreGemm.cu>
- **Pattern**: Double-precision (FP64) GEMM using WMMA API (`nvcuda::wmma` with `double` type). Uses 64×64 CTA tiles (smaller than FP16 due to double element size), shared memory skew-padding (`SKEW_DOUBLE`), and async copy pipeline for Ampere+.
- **Arch**: SM 8.0+ (Ampere)
- **Lines**: ~1042

```cuda
// Double-precision WMMA GEMM — 64×64 CTA tile, 8 warps, 8×8 sub-tiles
wmma::fragment<wmma::matrix_a, 8, 8, 4, double, wmma::row_major> a[WARP_COL_TILES];
wmma::fragment<wmma::matrix_b, 8, 8, 4, double, wmma::col_major> b[WARP_ROW_TILES];
wmma::fragment<wmma::accumulator, 8, 8, 4, double> c[WARP_COL_TILES][WARP_ROW_TILES];
for (int k_step = 0; k_step < CHUNK_K; k_step++) {
    wmma::load_matrix_sync(a[i], tile_ptr, K * CHUNK_K + SKEW_DOUBLE);
    wmma::load_matrix_sync(b[j], tile_ptr, K * CHUNK_K + SKEW_DOUBLE);
    wmma::mma_sync(c[i][j], a[i], b[j], c[i][j]);
}
```

## 4.5 globalToShmemAsyncCopy

- **Path**: `cpp/3_CUDA_Features/globalToShmemAsyncCopy/globalToShmemAsyncCopy.cu`
- **URL**: <https://github.com/NVIDIA/cuda-samples/blob/master/cpp/3_CUDA_Features/globalToShmemAsyncCopy/globalToShmemAsyncCopy.cu>
- **Pattern**: Matrix multiplication demonstrating `cuda::pipeline` for multi-stage async copy from global to shared memory (SM 8.0+). Uses rotating shared memory buffers with up to 4 pipeline stages for overlap, eliminating explicit `__syncthreads` between load and compute phases.
- **Arch**: SM 8.0+ (Ampere)
- **Lines**: ~1028

```cuda
// Multi-stage async copy pipeline — 4 stages, rotating shared memory buffers
constexpr size_t maxPipelineStages = 4;
__shared__ alignas(alignof(float4)) float As[maxPipelineStages][BLOCK_SIZE][BLOCK_SIZE];
__shared__ alignas(alignof(float4)) float Bs[maxPipelineStages][BLOCK_SIZE][BLOCK_SIZE];
cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();
for (int a = aBegin; a <= aEnd; a += aStep) {
    pipe.producer_acquire();
    // Async copy from global to rotating shared memory buffer
    cuda::memcpy_async(As[j][t4x], &A[a], aStep_bytes, pipe);
    pipe.producer_commit();
    pipe.consumer_wait();
    // Compute using the loaded tile — overlaps with next producer_acquire
}
```

## 4.6 tmaTensorMap (Python)

- **Path**: `python/2_CoreConcepts/tmaTensorMap/tmaTensorMap.py`
- **URL**: <https://github.com/NVIDIA/cuda-samples/blob/master/python/2_CoreConcepts/tmaTensorMap/tmaTensorMap.py>
- **Pattern**: Tensor Memory Accelerator (TMA) — Hopper hardware feature for async bulk tensor copy. Creates `tensor_map` descriptors, passes them as `__grid_constant__` to kernels, and reuses descriptors via `replace_address`.
- **Arch**: SM 9.0+ (Hopper: H100/H200/B200)
- **Lines**: ~281

```python
tensor_map = smv.as_tensor_map(box_dim=(height, width))
kernel = """
template<typename T> __global__ void tma_load_kernel(__grid_constant__ CUtensorMap tma_desc, T *dst) {
    cp.async.bulk.tensor.2d.shared::cluster.global.tile(dst, &tma_desc, 0, 0);
    // ...
}
"""
tensor_map.replace_address(smv.ptr)
```
