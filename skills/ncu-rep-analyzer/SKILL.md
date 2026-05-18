---
name: ncu-rep-analyzer
description: Profiles a CUDA kernel with NCU and analyzes the resulting .ncu-rep report to diagnose performance bottlenecks and generate optimization recommendations. Use when the user provides a .cu file or a .ncu-rep file and asks for performance analysis, NCU profiling, or bottleneck diagnosis. If given a .cu file, runs NCU via benchmark.py to produce a .ncu-rep, then imports it with `ncu --import` to extract metrics (SM throughput, DRAM/L1 bandwidth, occupancy), classifies the bottleneck (DRAM-bound, compute-bound, latency-bound, etc.), and saves a structured *_analysis.md report alongside the .ncu-rep file.
---

# NCU Profiling and Performance Analysis

Executes NCU profiling on a CUDA kernel, analyzes the `.ncu-rep` report, diagnoses performance bottlenecks, and provides optimization recommendations.

All commands MUST be executed in the project **root directory**.

---

## Workflow

### Progress Tracking

Copy the following checklist and update it in real-time:

```text
Task Progress:
- [ ] Step 1: Locate File
- [ ] Step 2: NCU Profiling (Generate .ncu-rep)
- [ ] Step 3: Read Report Summary
- [ ] Step 4: Automatically Diagnose Bottleneck
- [ ] Step 5: Generate Analysis Report
```

---

### Step 1: Locate File

**If the user provides a `.ncu-rep` file**: Proceed directly to Step 3, skipping Step 2.

**If the user provides a `.cu` file** (or profiling needs to be re-run): Confirm the following information before executing Step 2.

| Information                    | Inference Method                                                                                                                                 |
| ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| `<cu_file>`                    | The `.cu` file path provided by the user.                                                                                                        |
| Dimension Params (`--M`, etc.) | Infer parameter names from `extern "C" void solve(...)` signature; if unspecified, use reasonable defaults (MatMul: M=K=N=4096, Add: N=1000000). |
| `{kernel_stem}`                | The `.cu` filename without the extension.                                                                                                        |
| `{kernel_dir}`                 | The directory where the `.cu` file is located.                                                                                                   |

If the user specifies neither `.ncu-rep` nor `.cu`, search the current directory and common subdirectories:

```bash
find . -name "*.ncu-rep" 2>/dev/null
```

---

### Step 2: NCU Profiling

Two-tier approach — use the self-contained profiling executable for reliability.

**Step 2a: Build the profiling executable**：

```bash
python3 skills/kernel-benchmarker/scripts/ncu_profile.py <cu_file> \
    [--PARAM=VALUE ...] --build-only
```

This generates `<cu_stem>_bench` — a standalone executable that allocates its own GPU memory and launches the kernel. No subprocess, no Python dependency at profile time.

**Step 2b: Run NCU**：

Default (works everywhere, even on restricted containers without PMU access):

```bash
ncu --kernel-name solve \
    --launch-skip 10 \
    --launch-count 1 \
    --set launch \
    -o {kernel_dir}/{kernel_stem} -f \
    {kernel_dir}/{kernel_stem}_bench [--PARAM=VALUE ...] --warmup=10 --repeat=22
```

For detailed performance metrics (requires host PMU access: `perf_event_paranoid=0`):

```bash
ncu --kernel-name solve \
    --launch-skip 10 \
    --launch-count 1 \
    --set full \
    -o {kernel_dir}/{kernel_stem} -f \
    {kernel_dir}/{kernel_stem}_bench [--PARAM=VALUE ...] --warmup=10 --repeat=22
```

> **Legacy mode**: You may still profile via `benchmark.py` (`ncu ... -f python3 skills/kernel-benchmarker/scripts/benchmark.py ...`), but NCU may disconnect when `benchmark.py` spawns `nvcc` subprocess. The `ncu_profile.py` / `_bench` approach is preferred.

**Naming**: Use `{kernel_stem}` directly as the output filename in `{kernel_dir}`. Do not append a timestamp.

> `--launch-skip N` skips warmup iterations. `--launch-count 1` captures only the first profiling iteration. The bench executable must run ≥ `launch-skip + launch-count` total iterations.

#### Handling NCU Execution Failures

If `ncu` fails, you **MUST**:

1. Output the explicit reason for the failure.
2. Mark in the final report: `NCU Profiling Status: ❌ FAILED: <Reason>`
3. Never skip silently, and never substitute with a `.ncu-rep` file from another algorithm.

Common failures and fixes:

| Error                   | Cause                                                  | Fix                                                                                                                   |
| ----------------------- | ------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------- |
| `ERR_NVGPUCTRPERM`      | Container/profiling permission denied                  | Use `--set launch` instead of `--set full`, OR on the host: `echo 0 \| sudo tee /proc/sys/kernel/perf_event_paranoid` |
| `command not found`     | ncu not in PATH (usually at `/usr/local/cuda/bin/ncu`) | `export PATH=/usr/local/cuda-12.8/bin:$PATH`                                                                          |
| `==PROF== Disconnected` | Target process exited or spawned child processes       | Use `ncu_profile.py --build-only` to create a standalone bench executable                                             |

---

### Step 3: Read Report Summary

```bash
ncu --import <file.ncu-rep> --print-summary per-kernel
```

---

### Step 4: Automatically Diagnose Bottleneck

**If `--set launch` was used** (restricted environment without PMU access): The report contains static device properties and kernel launch configuration only. Use these to provide a preliminary analysis:

- Extract `device__attribute_clock_rate`, `device__attribute_fb_bus_width`, `device__attribute_compute_capability` from the report
- Compare grid/block dimensions against theoretical occupancy limits
- Note in the report: `NCU Mode: launch (static analysis only — use --set full on unrestricted hosts for dynamic metrics)`

**If `--set full` or `--set basic` was used** (unrestricted host): Determine the primary bottleneck based on dynamic metrics:

```text
roofline  = sm__throughput %
dram      = gpu__dram_throughput %
l1tex     = l1tex__throughput %
sm_busy   = sm__cycles_active %
occupancy = sm__warps_active %

IF sm_throughput < 30:
    IF dram > 70:       → DRAM_MEMORY_BOUND
    ELIF l1tex > 80 AND dram < 30: → L1_PRESSURE_BOUND
    ELSE:               → LATENCY_BOUND
ELIF sm_throughput > 60:
    IF sm_busy > 80:    → COMPUTE_BOUND
    ELSE:               → OCCUPANCY_BOUND
ELSE:                   → MIXED_BOUND
```

---

### Step 5: Generate Analysis Report

**MANDATORY REQUIREMENT**: Before generating optimization recommendations, you **MUST** consult the local knowledge base:

- If shared memory conflicts or non-contiguous memory accesses are found, you must consult `../cuda-knowledge/references/performance-traps.md` to extract avoidance paradigms (e.g., Padding rules, float4 vectorized loading).
- If you need to check detailed explanations of NCU metrics, refer to `../cuda-knowledge/references/ncu-guide.md`.

Output the analysis results following the template below, and automatically save the analysis report in the **same kernel directory** as the `.ncu-rep`:

```text
project_root/
├── kernel/
│   └── <AlgoName>/
│       ├── solution.cu
│       ├── <kernel_stem>.ncu-rep         # NCU Report
│       └── <kernel_stem>_analysis.md    # AI Analysis Report
```

---

## Output Template

```markdown
# NCU Performance Analysis Report

## Report Information

- **File**: {file.ncu-rep}
- **Kernel**: {kernel_name}
- **Analysis Time**: {timestamp}

## Executive Summary

| Item                | Value             |
| ------------------- | ----------------- |
| **Main Bottleneck** | {bottleneck_type} |
| **Confidence**      | {confidence}      |
| **Opt. Potential**  | {potential}x      |

## Key Metrics

### Performance Metrics

| Metric        | Value            | Healthy Threshold | Status   |
| ------------- | ---------------- | ----------------- | -------- |
| sm_throughput | {sm_throughput}% | > 60%             | {status} |
| SM Busy       | {sm_busy}%       | > 70%             | {status} |
| Occupancy     | {occupancy}%     | > 50%             | {status} |

### Memory Metrics

| Metric            | Value    | Healthy Threshold | Status   |
| ----------------- | -------- | ----------------- | -------- |
| DRAM Throughput   | {dram}%  | < 50%             | {status} |
| L1/TEX Throughput | {l1tex}% | < 80%             | {status} |
| L2 Throughput     | {l2}%    | < 80%             | {status} |

## Diagnostic Details

**Bottleneck Type**: {bottleneck_type}

**Basis for Judgment**:

- {reason_1}
- {reason_2}

## Optimization Recommendations

### High Priority

{high_priority_suggestions}

### Verification Checklist

- [ ] Implement optimization suggestions
- [ ] Re-run NCU collection (Step 2)
- [ ] Compare metrics before and after optimization
```

---

## Bottleneck Diagnosis & Optimization Strategies

### DRAM_MEMORY_BOUND

```text
IF dram_throughput > 70% AND sm_throughput < 30%:
    Diagnosis: DRAM_MEMORY_BOUND (Confidence: HIGH)

    Optimization Strategies:
    1. Block Tiling (Shared Memory Caching)
    2. Vectorized Load (float4)
    3. Prefetching (Data Prefetching)
```

### L1_PRESSURE_BOUND

```text
IF l1tex_throughput > 80% AND dram_throughput < 30%:
    Diagnosis: L1_PRESSURE_BOUND (Confidence: HIGH)

    Optimization Strategies:
    1. Shared Memory Padding
    2. Data Transpose
    3. Fragment Caching
```

### LATENCY_BOUND

```text
IF sm_busy < 50% AND occupancy > 60%:
    Diagnosis: LATENCY_BOUND (Confidence: HIGH)

    Optimization Strategies:
    1. Double Buffering
    2. Instruction-level Parallelism
    3. Loop Unrolling
```

### COMPUTE_BOUND

```text
IF sm_throughput > 60% AND sm_busy > 80%:
    Diagnosis: COMPUTE_BOUND (Confidence: HIGH)

    Optimization Strategies:
    1. Use FMA instructions
    2. Reduce precision (FP32 -> FP16/TF32)
    3. Tensor Cores
```

### OCCUPANCY_BOUND

```text
IF occupancy < 30% AND sm_busy > 70%:
    Diagnosis: OCCUPANCY_BOUND (Confidence: HIGH)

    Optimization Strategies:
    1. Reduce register usage
    2. Adjust block size
    3. Use __launch_bounds__
```

---

## Common Misconceptions

1. **High Throughput ≠ High Efficiency** — High Compute + Memory Throughput but low `sm_throughput` means the GPU is "busily waiting".
2. **Low DRAM Throughput can be good** — It indicates data is reused in the cache, which is a sign of successful optimization.
3. **Higher Occupancy is not always better** — The goal is the minimum sufficient occupancy to hide latency.
