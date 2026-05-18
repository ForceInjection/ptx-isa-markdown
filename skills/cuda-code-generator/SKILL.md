---
name: cuda-code-generator
description: Generates optimized CUDA kernel code based on performance analysis reports or algorithm type. Reads NCU analysis reports (e.g. *_analysis.md) and optionally existing kernel code, then produces high-quality compilable .cu files with applied optimizations. Use when the user provides an NCU analysis report or requests CUDA kernel generation, optimization, or implementation of techniques like Shared Memory Tiling, vectorized loads, bank conflict elimination, or double buffering. Does not handle compilation, execution, or profiling.
---

# CUDA Code Generator

Reads performance analysis reports and generates optimized CUDA kernel code (`.cu` files). **Does not handle compilation, execution, or profiling.**

## Execution Workflow

### Progress Tracking

```text
Task Progress:
- [ ] Step 1: Requirement Analysis
- [ ] Step 2: Read Input Files
- [ ] Step 3: Determine Optimization Strategy
- [ ] Step 4: Generate Code and Write to File
```

---

### Step 1: Requirement Analysis

Confirm the following information (can be obtained from the report or user input):

| Item                                                       | Source             |
| ---------------------------------------------------------- | ------------------ |
| Algorithm Type (MatMul / Reduction / Convolution / Custom) | Report Kernel Name |
| Data Scale (M, N, K, or other dimensions)                  | Provided by User   |
| Precision (FP32 / FP16 / INT8)                             | Report or User     |
| GPU Architecture (sm_XX)                                   | Report Device CC   |

---

### Step 2: Read Input Files

1. **Read NCU Analysis Report** (`*_analysis.md`, preferred)
   - Extract bottleneck type (DRAM_MEMORY_BOUND / L1_PRESSURE_BOUND / LATENCY_BOUND / COMPUTE_BOUND / OCCUPANCY_BOUND / MIXED_BOUND)
   - Extract optimization priority list (P0~Pn)
   - Extract key metrics (SM Busy, DRAM Throughput, Warp Cycles, etc.)

2. **Read Existing Kernel Code** (If user provides path)
   - Identify current thread block configuration, memory access pattern, and computation logic
   - Keep the host-side API interface unchanged (parameter list, function name)

If **no analysis report** is available, select a default strategy based on the algorithm type, see [cuda-optimization-strategies.md](references/cuda-optimization-strategies.md).

---

### Step 3: Determine Optimization Strategy

Find the corresponding strategy combination in [cuda-optimization-strategies.md](references/cuda-optimization-strategies.md) based on the **bottleneck type** from the report, then apply them in order of priority from P0 → Pn.

Quick reference for strategies corresponding to each bottleneck type (see reference documentation for detailed explanations and code templates):

| Bottleneck Type   | Characteristic Condition       | Priority Strategy                                     |
| ----------------- | ------------------------------ | ----------------------------------------------------- |
| DRAM_MEMORY_BOUND | DRAM > 70%, SM < 30%           | Block Tiling → Vectorized Load → Prefetching          |
| L1_PRESSURE_BOUND | L1 > 80%, DRAM < 30%           | Shared Memory Tiling → Padding → Data Transpose       |
| LATENCY_BOUND     | SM Busy < 50%, Occupancy > 60% | Double Buffering → ILP → Loop Unrolling               |
| COMPUTE_BOUND     | SM > 60%, SM Busy > 80%        | FMA → FP16/TF32 → Tensor Core                         |
| OCCUPANCY_BOUND   | Occupancy < 30%, SM Busy > 70% | Adjust Block Size → `__launch_bounds__` → Reduce smem |
| MIXED_BOUND       | No single dominant bottleneck  | Profile with narrower section sets; address top metric first |

---

### Step 4: Generate Code and Write to File

#### Output File Path

1. Execute `date +%Y%m%d_%H%M%S` to get the timestamp (e.g., `20260316_153045`)
2. Append the timestamp suffix based on the original `.cu` file path, **always create a new file, do not overwrite the original file**:
   - `kernel/MatrixMultiplication/solution.cu` → `kernel/MatrixMultiplication/solution_opt_20260316_153045.cu`
   - If the user does not provide the original path, default to naming it `solution_opt_<timestamp>.cu` in the same directory as the kernel.

#### File Header Comment Template

```cuda
/*
 * Optimized CUDA Kernel - <Algorithm Name>
 *
 * Generation Time: <YYYY-MM-DD HH:MM:SS>
 *
 * Optimization Measures (from NCU Analysis Report):
 *   [P0] Shared Memory Tiling (TILE_SIZE=16)
 *   [P1] Shared Memory Padding (+1 to eliminate Bank Conflict)
 *   [P2] Vectorized Load (float4)
 *
 * Compilation Command:
 *   nvcc -O3 -arch=sm_89 -o kernel solution_opt_<timestamp>.cu
 *
 * Target Device: Ada Lovelace (sm_89)
 * Precision: FP32
 */
```

#### Code Quality Requirements

- **Boundary Check**: Must handle tail Tile out-of-bounds when K is not an integer multiple of TILE_SIZE
- **Architecture Compatibility**: `cp.async` / `__pipeline` are only supported on sm_80+ (Ampere) and above
- **Function Signature Unchanged**: Host calling interface must remain consistent with the original kernel

---

## References

- Detailed explanations and code templates for optimization strategies → [cuda-optimization-strategies.md](references/cuda-optimization-strategies.md)
- Concrete working code examples for each optimization pattern → [`../cuda-samples/references/`](../cuda-samples/references/) (curated NVIDIA official samples with code snippets, organized by topic; use [`../cuda-samples/SKILL.md`](../cuda-samples/SKILL.md) for the quick reference table and optimization mapping)
- **MANDATORY REQUIREMENT: When dealing with complex APIs (such as cuBLASLt, Tensor Core, half precision \_\_half, etc.), you MUST first use `grep` or semantic search to consult: (1) the corresponding official API documentation under `../cuda-knowledge/references/` to ensure function signatures and data layout parameters strictly comply with NVIDIA official specifications; and (2) the relevant code patterns in `../cuda-samples/references/` to find working examples of the API in context.**
