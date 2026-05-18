---
name: cuda-optimizer
description: Orchestrates a full profiling-driven CUDA kernel optimization loop (write → validate → profile → analyze → optimize) until performance converges or no further gains are possible. Capabilities include generating reference implementations, writing initial kernels via cuda-code-generator, running correctness validation and benchmarks via kernel-benchmarker, profiling and analyzing NCU reports via ncu-rep-analyzer, and applying targeted optimizations via cuda-code-generator. Use when the user wants to optimize a .cu file or CUDA kernel, improve GPU kernel performance, run a CUDA optimization workflow.
---

# CUDA Kernel Optimizer

Drives the complete cycle of **Generation → Validation & Benchmarking → Evaluation → NCU Analysis → Re-optimization** until the kernel has no further optimization room or performance converges.

> ⚠️ **CORE CONSTRAINTS (MUST READ)**
>
> This skill acts as the **Orchestrator**, responsible for driving the entire loop until exit conditions are met.
> **After every sub-skill (kernel-benchmarker / ncu-rep-analyzer / cuda-code-generator) call returns, you MUST immediately return to the main workflow and execute the next step. You MUST NEVER stop after a sub-skill returns.**
> The output of a sub-skill is the input for the next step, not the final destination.

---

## Execution Workflow

### Progress Tracking

Output the following checklist in the conversation before each loop starts and update it in real-time:

```text
Optimization Loop #N:
- [ ] Step 1: Correctness Validation + Performance Benchmarking (kernel-benchmarker)
- [ ] Step 2: Evaluate Exit Conditions
- [ ] Step 3: NCU Profiling + Bottleneck Analysis (ncu-rep-analyzer)  ← Execute ONLY if Step 2 decides to "Continue"
- [ ] Step 4: Implement Optimizations (cuda-code-generator)  ← Execute ONLY if Step 2 decides to "Continue"
```

---

### Preparation Phase: Generate Reference

**If the user has provided a reference file**: Use it directly and skip to "First Version Kernel".

**If the user has not provided a reference file**:

1. Create `<algo_name>_ref.py` in the `kernel/<AlgoName>/` directory based on the algorithm type.
2. Reference format (using vector addition as an example):

```python
"""Reference for: solve(const float* A, const float* B, float* C, int N)"""
import torch


def reference(*, A, B, C, N, **kwargs):
    C[:N] = A[:N] + B[:N]
```

Rules:

- The docstring at the top of the file MUST write out the complete signature of `solve(...)`.
- The function name is fixed as `reference`, and all parameters are keyword-only (after `*`).
- Implement the algorithm logic using PyTorch tensor operations, writing results to the output tensor.
- Accept `**kwargs` to ignore extra parameters.

---

### First Version Kernel

**If the user has provided a `.cu` file** (e.g., `solution.cu`): Use it directly and immediately enter optimization Loop #1.

**If the user has not provided a `.cu` file**:

1. Read the reference file and extract the algorithm semantics (input/output tensors, dimension parameters).
2. Call the **cuda-code-generator skill** to generate the first version of the kernel (naive implementation, no complex optimizations) and save it as:
   - `kernel/<AlgoName>/solution.cu`
3. The Kernel function signature MUST be `extern "C" void solve(...)` and be consistent with the reference docstring.
4. **After cuda-code-generator returns**, immediately enter optimization Loop #1.

---

### Optimization Loop (Repeat until exit conditions are met)

> **Orchestration Note**: Step 1 ~ Step 4 constitute a complete loop body. After each step completes, **you MUST immediately execute the next step**. You must not stop at the current step just because a sub-skill output a report. The entire loop is fully driven by this skill (cuda-optimizer), and sub-skills only provide information inputs.

---

#### Step 1: Correctness Validation + Performance Benchmarking

Call the **kernel-benchmarker skill** to execute the following on the current kernel file:

- Correctness validation (compare against reference)
- Performance benchmarking (record Average / Median / Min / Max / Bandwidth, **including Speedup vs reference**)

**After kernel-benchmarker returns**, immediately process the result and decide the next step:

| Result            | Action                                                                              |
| ----------------- | ----------------------------------------------------------------------------------- |
| Validation Failed | Call **cuda-code-generator** to fix the bug → After fixing, **restart from Step 1** |
| Validation Passed | **Immediately enter Step 2**                                                        |

---

#### Step 2: Evaluate Exit Conditions

Based on the benchmark results from Step 1, evaluate the following exit conditions:

| Condition               | Description                                                                    |
| ----------------------- | ------------------------------------------------------------------------------ |
| ① Performance Converged | Performance improvement is < 2% after **2 consecutive rounds** of optimization |

**If exit conditions are met**: Exit the loop and output the [Final Report](#final-report).

**If exit conditions are not met**: Explicitly state "Exit conditions not met, continuing to Step 3" in the conversation, and **immediately enter Step 3**.

---

#### Step 3: NCU Profiling + Bottleneck Analysis

Call the **ncu-rep-analyzer skill** to execute the following on the current kernel file:

- NCU Profiling (generate `.ncu-rep` file)
- Read and analyze the report, outputting:
  - Bottleneck type (DRAM_MEMORY_BOUND / L1_PRESSURE_BOUND / LATENCY_BOUND / COMPUTE_BOUND / OCCUPANCY_BOUND / MIXED_BOUND)
  - Optimization priority list (P0 ~ Pn)

**After ncu-rep-analyzer returns**, immediately process the result:

| Result                                                     | Action                                                                                         |
| ---------------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| NCU Profiling Failed                                       | **Immediately stop the loop**, output the failure reason, wait for user to fix the environment |
| NCU Success + Analysis Done (No P0/P1 suggestions)         | Exit the loop and output the [Final Report](#final-report) (no room for optimization)          |
| NCU Success + Analysis Done (Has optimization suggestions) | Extract optimization suggestions, **immediately enter Step 4**                                 |

> ⚠️ When NCU fails: Do not silently skip, and do not reuse `.ncu-rep` files from other algorithms or rounds.
> ⚠️ **STRICT CONSTRAINT**: You MUST use the **`.ncu-rep` corresponding to the current kernel**. Reusing NCU reports from other algorithms or previous rounds is strictly prohibited.

---

#### Step 4: Implement Optimizations

Call the **cuda-code-generator skill**, based on the optimization suggestions output from Step 3 (prioritizing P0), to generate a new version of the kernel:

- The new file must be named with a timestamp, **do not overwrite the original file** (follow the naming rules of the cuda-code-generator skill).
- Explain the optimization items implemented in this round in the file header comments.

**After cuda-code-generator returns**, use the path of the newly generated `.cu` file as the current kernel for the next round, and **immediately return to Step 1** to start a new loop round (Loop #N+1).

---

### Final Report

Once exit conditions are met, output the complete optimization report:

```markdown
## CUDA Kernel Optimization Report

### Algorithm

<Algorithm Name>

### Reference File

`kernel/<AlgoName>/<algo_name>_ref.py`

### Optimization History

| Round | Kernel File           | Average (ms) | Speedup vs Ref | Main Optimization Items   |
| ----- | --------------------- | ------------ | -------------- | ------------------------- |
| First | solution.cu           | X.XX         | 0.XXx          | Naive implementation      |
| #1    | solution*opt*<ts1>.cu | X.XX         | X.XXx          | P0: Shared Memory Tiling  |
| #2    | solution*opt*<ts2>.cu | X.XX         | X.XXx          | P1: Bank Conflict Padding |
| ...   | ...                   | ...          | ...            | ...                       |

### Conclusion

<Exit Reason: No optimization room / Performance converged>

### Best Kernel

`kernel/<AlgoName>/solution_opt_<timestamp>.cu`

- Average: X.XX ms
- Speedup: X.XXx vs reference
```

---

## Parameter Inference Rules

| Parameter                    | Inference Method                                                                   |
| ---------------------------- | ---------------------------------------------------------------------------------- |
| Algorithm Name               | Infer from user description or reference filename                                  |
| Dimension Parameters Default | MatMul: M=K=N=4096; Vector Add: N=1,000,000; Conv: N=1,000,000                     |
| Reference Path               | Provided by user; otherwise create `kernel/<AlgoName>/<algo_name>_ref.py`          |
| Kernel Output Path           | `kernel/<AlgoName>/solution.cu` (First version); subsequent versions add timestamp |

---

## Skill Dependencies and Calling Specifications

This skill orchestrates the following three sub-skills. **After calling a sub-skill, you MUST immediately process its return result and continue the main workflow; you MUST NOT stop.**

| Sub-Skill             | Purpose                                                   | Action After Call Completes                                    |
| --------------------- | --------------------------------------------------------- | -------------------------------------------------------------- |
| `cuda-code-generator` | Generate / Fix / Optimize kernel code                     | Record new file path, proceed to next step                     |
| `kernel-benchmarker`  | Correctness validation + Performance benchmarking         | Record benchmark data, enter Step 2                            |
| `ncu-rep-analyzer`    | NCU Profiling + Interpret report + Output opt suggestions | Extract optimization suggestions, **immediately enter Step 4** |

> **Common Interruption Trap**: `ncu-rep-analyzer` outputs a formatted analysis report. This report is an input intended for Step 4 (cuda-code-generator), **it is NOT the end of the optimization loop**. Upon receiving the analysis report, you MUST immediately execute Step 4 to implement optimizations, without waiting for user instructions.
