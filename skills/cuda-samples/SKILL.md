---
name: cuda-samples
description: "Curated index of NVIDIA official CUDA Samples with code patterns, GitHub permalinks, and key snippets. Use when looking for working CUDA code examples, GPU kernel patterns (reduction, scan, GEMM, transpose, histogram), Tensor Core WMMA usage, CUDA Graphs API, cuBLAS/cuFFT/cuSPARSE library integration, stream/event async patterns, multi-GPU P2P/IPC, or framework interop (PyTorch/TensorFlow custom kernels). Covers 50+ curated samples organized by topic. Triggers on CUDA sample, code example, kernel pattern, GEMM example, reduction example, CUDA graph example, cuBLAS example, Tensor Core WMMA, how to write CUDA kernel, CUDA code reference."
---

# CUDA Samples Skill

Curated index of ~50 official [NVIDIA CUDA Samples](https://github.com/NVIDIA/cuda-samples) organized by common CUDA programming tasks. Each entry links to the canonical source, describes the demonstrated pattern, lists architecture requirements, and provides a key code snippet.

**Usage**: Search this file by pattern name (e.g., "reduction", "GEMM", "graph capture") or API name (e.g., "cublasSgemm", "cudaMallocAsync"). When a match is found, follow the GitHub permalink for the full implementation.

**Source**: NVIDIA/cuda-samples (CUDA Toolkit 13.2), 175 `.cu` files, ~57K lines of CUDA C++.

---

## Quick Reference Table

| Task                                 | Category        | C++ Sample                    | Python Sample                    |
| ------------------------------------ | --------------- | ----------------------------- | -------------------------------- |
| First CUDA kernel                    | Getting Started | `vectorAdd`                   | `vectorAdd`                      |
| Device properties query              | Utilities       | `deviceQuery`                 | `deviceQuery`                    |
| GPU timing with events               | Getting Started | `asyncAPI`                    | —                                |
| Multi-stream overlap                 | Streams & Async | `simpleStreams`               | `streamingCopyComputeOverlap`    |
| Stream-ordered allocation            | Memory          | `streamOrderedAllocation`     | —                                |
| Unified Memory perf                  | Memory          | `UnifiedMemoryPerf`           | `blurImageUnifiedMemory`         |
| cuda.core memory model               | Memory          | —                             | `memoryResources`                |
| Thread/block indexing patterns       | Getting Started | —                             | `blockwiseSum`                   |
| IPC between processes                | Multi-GPU       | `simpleIPC`                   | `ipcMemoryPool`                  |
| P2P between GPUs                     | Multi-GPU       | `simpleP2P`                   | `simpleP2P`                      |
| Parallel reduction                   | Kernel Patterns | `reduction`                   | `reduction`, `parallelReduction` |
| Multi-block CG reduction             | Kernel Patterns | `reductionMultiBlockCG`       | `reductionMultiBlockCG`          |
| Warp shuffle scan                    | Kernel Patterns | `shfl_scan`                   | —                                |
| Prefix sum (library)                 | Kernel Patterns | —                             | `prefixSum`                      |
| Sorting networks                     | Kernel Patterns | `sortingNetworks`             | —                                |
| Radix sort (Thrust)                  | Kernel Patterns | `radixSortThrust`             | —                                |
| Histogram with atomics               | Kernel Patterns | `histogram`                   | `parallelHistogram`              |
| FP16 Tensor Core GEMM                | Tensor Core     | `cudaTensorCoreGemm`          | —                                |
| BF16 Tensor Core GEMM                | Tensor Core     | `bf16TensorCoreGemm`          | —                                |
| TF32 Tensor Core GEMM                | Tensor Core     | `tf32TensorCoreGemm`          | —                                |
| Double-precision TC GEMM             | Tensor Core     | `dmmaTensorCoreGemm`          | —                                |
| Async copy global→shared (pipeline)  | Tensor Core     | `globalToShmemAsyncCopy`      | —                                |
| TMA (Hopper+)                        | Tensor Core     | —                             | `tmaTensorMap`                   |
| CUDA Graphs basics                   | Graphs          | `simpleCudaGraphs`            | `cudaGraphs`                     |
| Graph with updates                   | Graphs          | `jacobiCudaGraphs`            | —                                |
| Graph memory nodes                   | Graphs          | `graphMemoryNodes`            | —                                |
| Conditional graph nodes              | Graphs          | `graphConditionalNodes`       | —                                |
| cuBLAS GEMM                          | Libraries       | `simpleCUBLAS`                | —                                |
| Batched cuBLAS GEMM                  | Libraries       | `batchCUBLAS`                 | —                                |
| cuBLAS with CUDA Graphs              | Libraries       | `conjugateGradientCudaGraphs` | —                                |
| Conjugate Gradient (cuBLAS+cuSPARSE) | Libraries       | `conjugateGradient`           | —                                |
| cuFFT convolution                    | Libraries       | `simpleCUFFT`                 | `fftSignalAnalysis`              |
| Matrix transpose                     | Performance     | `transpose`                   | —                                |
| Block size tuning                    | Performance     | —                             | `launchConfigTuning`             |
| Warp-aggregated atomics              | Performance     | `warpAggregatedAtomicsCG`     | —                                |
| PyTorch custom kernel                | Framework       | —                             | `customPyTorchKernel`            |
| TensorFlow custom kernel             | Framework       | —                             | `customTensorFlowKernel`         |
| Multi-GPU gradient average           | Distributed     | —                             | `multiGPUGradientAverage`        |
| cuSOLVER linear solver               | Libraries       | `cuSolverDn_LinearSolver`     | —                                |
| JIT LTO linking                      | Advanced        | —                             | `jitLtoLinking`                  |
| NVTX profiling markers               | Profiling       | —                             | `kernelNsysProfile`              |
| Green context (SM partitioning)      | Advanced        | —                             | `greenContext`                   |
| Image processing (NPP)               | Libraries       | `histEqualizationNPP`         | —                                |

The table above is a fast lookup index. Detailed code snippets and patterns for the most important samples are in the [references/](references/) directory, organized by topic (some entries are index-only without detailed snippets):

| Category                 | File                                                                        | Contents                                                                                                             |
| ------------------------ | --------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| Getting Started & Memory | [samples-getting-started.md](references/samples-getting-started.md)         | vectorAdd (C++/Python), asyncAPI, streamOrderedAllocation, UnifiedMemoryPerf, memoryResources                        |
| Streams & Async          | [samples-streams-async.md](references/samples-streams-async.md)             | simpleStreams, streamingCopyComputeOverlap, simpleIPC, simpleP2P                                                     |
| Reduction, Scan & Sort   | [samples-reduction-scan-sort.md](references/samples-reduction-scan-sort.md) | reduction (C++/Python), reductionMultiBlockCG, shfl_scan, radixSortThrust, parallelReduction                         |
| Tensor Core GEMM         | [samples-tensor-core-gemm.md](references/samples-tensor-core-gemm.md)       | cudaTensorCoreGemm, bf16TensorCoreGemm, tf32TensorCoreGemm, dmmaTensorCoreGemm, globalToShmemAsyncCopy, tmaTensorMap |
| CUDA Graphs              | [samples-cuda-graphs.md](references/samples-cuda-graphs.md)                 | simpleCudaGraphs, cudaGraphs (Python), jacobiCudaGraphs, graphMemoryNodes                                            |
| CUDA Libraries           | [samples-cuda-libraries.md](references/samples-cuda-libraries.md)           | simpleCUBLAS, batchCUBLAS, conjugateGradient, simpleCUFFT, histEqualizationNPP                                       |
| Performance Optimization | [samples-performance.md](references/samples-performance.md)                 | transpose, launchConfigTuning, warpAggregatedAtomicsCG, kernelNsysProfile                                            |
| Framework Interop        | [samples-framework-interop.md](references/samples-framework-interop.md)     | customPyTorchKernel, customTensorFlowKernel                                                                          |

---

## Optimization Strategy → Sample Mapping

Map optimization strategies to relevant samples:

| Bottleneck Type   | Optimization Strategy                                   | Reference Sample(s)                                                      |
| ----------------- | ------------------------------------------------------- | ------------------------------------------------------------------------ |
| DRAM_MEMORY_BOUND | Block tiling, vectorized load, prefetching              | `transpose` (merge→coalesced), `cudaTensorCoreGemm` (int4 loads)         |
| L1_PRESSURE_BOUND | Shared memory padding, data transpose, fragment caching | `transpose` (no bank conflict variant), `reduction` (shared memory tree) |
| LATENCY_BOUND     | Double buffering, ILP, loop unrolling                   | `bf16TensorCoreGemm` (pipeline), `reduction` (multi-element per thread)  |
| COMPUTE_BOUND     | FMA, FP16/TF32, Tensor Cores                            | `cudaTensorCoreGemm`, `tf32TensorCoreGemm`, `bf16TensorCoreGemm`         |
| OCCUPANCY_BOUND   | Adjust block size, `__launch_bounds__`, reduce smem     | `launchConfigTuning`, `SHARED_MEMORY_LIMIT_64K` in tensor core samples   |
| MIXED_BOUND       | Narrower NCU section sets; target top metric first      | `transpose` (coarse-grain)                                               |

---

## Architecture Compatibility

| Feature                          | Min SM | Samples                                                                 |
| -------------------------------- | ------ | ----------------------------------------------------------------------- |
| Basic CUDA                       | All    | `vectorAdd`, `asyncAPI`, `simpleStreams`, `simpleCUBLAS`, `simpleCUFFT` |
| Warp shuffle (`__shfl_*`)        | 3.0    | `shfl_scan`, `reduction` (variants 5-6)                                 |
| Cooperative groups (block-level) | 5.2    | `reduction` (CG variants), `warpAggregatedAtomicsCG`                    |
| Cooperative launch (grid sync)   | 6.0    | `reductionMultiBlockCG`                                                 |
| WMMA (Tensor Cores)              | 7.0    | `cudaTensorCoreGemm`, `dmmaTensorCoreGemm`                              |
| Async copy (`memcpy_async`)      | 8.0    | `bf16TensorCoreGemm`, `globalToShmemAsyncCopy`                          |
| TF32 Tensor Cores                | 8.0    | `tf32TensorCoreGemm`                                                    |
| `__reduce_add_sync`              | 8.0    | `reduction` (variant 7)                                                 |
| TMA (Tensor Memory Accelerator)  | 9.0    | `tmaTensorMap` (Python)                                                 |
| Green context (SM partitioning)  | 9.0    | `greenContext` (Python)                                                 |

---

## Search Tips

```bash
# Find samples by API usage (search references/ directory)
grep -r "cublasSgemm" skills/cuda-samples/references/
grep -r "cudaStreamBeginCapture\|cudaGraph" skills/cuda-samples/references/
grep -r "wmma::mma_sync\|__pipeline_memcpy_async" skills/cuda-samples/references/

# Find samples by pattern
grep -r "reduction\|scan\|histogram" skills/cuda-samples/references/
grep -r "PyTorch\|TensorFlow\|framework" skills/cuda-samples/references/

# Find samples by architecture requirement
grep -r "SM 8.0\|SM 9.0\|Hopper\|Ampere" skills/cuda-samples/references/

# Find samples by name in the quick reference table
grep -i "vectorAdd\|simpleCUBLAS\|transpose" skills/cuda-samples/SKILL.md

# Clone full repo for compilation and detailed study
git clone --depth 1 https://github.com/NVIDIA/cuda-samples.git
```
