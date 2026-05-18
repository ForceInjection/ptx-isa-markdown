# CUDA Libraries Samples

## 6.1 simpleCUBLAS

- **Path**: `cpp/4_CUDA_Libraries/simpleCUBLAS/simpleCUBLAS.cpp`
- **URL**: <https://github.com/NVIDIA/cuda-samples/blob/master/cpp/4_CUDA_Libraries/simpleCUBLAS/simpleCUBLAS.cpp>
- **Pattern**: Complete cuBLAS workflow: `cublasCreate` → `cublasSetVector` → `cublasSgemm` → `cublasGetVector` → `cublasDestroy`
- **Arch**: All
- **Lines**: ~253

```cuda
cublasHandle_t handle; cublasCreate(&handle);
cublasSetVector(n2, sizeof(float), h_A, 1, d_A, 1);  // upload data
float alpha = 1.0f, beta = 0.0f;
cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N,
            &alpha, d_A, N, d_B, N, &beta, d_C, N);
cublasGetVector(n2, sizeof(float), d_C, 1, h_C, 1);  // download result
cublasDestroy(handle);
```

## 6.2 batchCUBLAS

- **Path**: `cpp/4_CUDA_Libraries/batchCUBLAS/batchCUBLAS.cpp`
- **URL**: <https://github.com/NVIDIA/cuda-samples/blob/master/cpp/4_CUDA_Libraries/batchCUBLAS/batchCUBLAS.cpp>
- **Pattern**: `cublasSgemmBatched` and `cublasSgemmStridedBatched` — batched small-GEMM patterns essential for multi-head attention and ensemble models
- **Arch**: All
- **Lines**: ~350

```cuda
// Pointer array variant
float *d_Aarray[batchCount], *d_Barray[batchCount], *d_Carray[batchCount];
cublasSgemmBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K,
                   &alpha, d_Aarray, M, d_Barray, K, &beta, d_Carray, M, batchCount);
// Strided variant (single contiguous buffer)
cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K,
                          &alpha, d_A, M, strideA, d_B, K, strideB, &beta, d_C, M, strideC, batchCount);
```

## 6.3 conjugateGradient

- **Path**: `cpp/4_CUDA_Libraries/conjugateGradient/main.cpp`
- **URL**: <https://github.com/NVIDIA/cuda-samples/blob/master/cpp/4_CUDA_Libraries/conjugateGradient/main.cpp>
- **Pattern**: Complete Conjugate Gradient solver combining CUBLAS (BLAS ops: dot, axpy, gemv) and CUSPARSE (sparse matrix-vector multiply) — the reference pattern for numerical solvers
- **Arch**: All
- **Lines**: ~400

```cuda
// CG iteration combining cuBLAS + cuSPARSE
cublasDdot(handle, N, d_r, 1, d_r, 1, &r1);    // residual norm
cusparseSpMV(handle_sp, CUSPARSE_OPERATION_NON_TRANSPOSE,
             &alpha, matA, vecX, &beta, vecY, CUDA_R_64F, ...);  // Ap = A * p
cublasDaxpy(handle, N, &alpha, d_p, 1, d_x, 1);  // x = x + alpha * p
cublasDaxpy(handle, N, &neg_beta, d_p, 1, d_r, 1); // r = r - alpha * Ap
```

## 6.4 simpleCUFFT

- **Path**: `cpp/4_CUDA_Libraries/simpleCUFFT/simpleCUFFT.cu`
- **URL**: <https://github.com/NVIDIA/cuda-samples/blob/master/cpp/4_CUDA_Libraries/simpleCUFFT/simpleCUFFT.cu>
- **Pattern**: Frequency-domain convolution via cuFFT: forward FFT → complex multiply → inverse FFT. Both simple and advanced plan APIs.
- **Arch**: All
- **Lines**: ~170

```cuda
cufftHandle plan; cufftPlan1d(&plan, N, CUFFT_C2C, 1);
cufftExecC2C(plan, d_signal, d_signal, CUFFT_FORWARD);  // FFT
complex_pointwise_multiply<<<grid, block>>>(d_signal, d_filter, N);
cufftExecC2C(plan, d_signal, d_signal, CUFFT_INVERSE);   // IFFT
```

## 6.5 histEqualizationNPP

- **Path**: `cpp/4_CUDA_Libraries/histEqualizationNPP/histEqualizationNPP.cpp`
- **URL**: <https://github.com/NVIDIA/cuda-samples/blob/master/cpp/4_CUDA_Libraries/histEqualizationNPP/histEqualizationNPP.cpp>
- **Pattern**: GPU-accelerated image histogram equalization using NPP (NVIDIA Performance Primitives) — demonstrates the complete NPP workflow: compute histogram → build cumulative LUT → apply LUT transform
- **Arch**: All
- **Lines**: ~300

```cpp
// NPP histogram equalization workflow
nppiHistogramEvenGetBufferSize_8u_C1R_Ctx(oSizeROI, levelCount, &nDeviceBufferSize, nppStreamCtx);
cudaMalloc(&pDeviceBuffer, nDeviceBufferSize);
nppiEvenLevelsHost_32s(levelsHost, levelCount, 0, binCount);
// Compute histogram on GPU
nppiHistogramEven_8u_C1R_Ctx(oDeviceSrc.data(), oDeviceSrc.pitch(), oSizeROI,
                              histDevice, levelCount, 0, binCount,
                              pDeviceBuffer, nppStreamCtx);
cudaMemcpy(histHost, histDevice, binCount * sizeof(Npp32s), cudaMemcpyDeviceToHost);
// Build cumulative LUT on host, upload to device, apply
cudaMemcpy(lutDevice, lutHost, sizeof(Npp32s) * levelCount, cudaMemcpyHostToDevice);
nppiLUT_Linear_8u_C1R_Ctx(oDeviceSrc.data(), oDeviceSrc.pitch(),
                           oDeviceDst.data(), oDeviceDst.pitch(),
                           oSizeROI, lutDevice, lvlsDevice, nppStreamCtx);
```
