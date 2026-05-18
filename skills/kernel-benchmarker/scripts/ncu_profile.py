#!/usr/bin/env python3
"""Build a self-contained profiling executable for a CUDA kernel.

Generates a host wrapper, compiles it with the user's .cu file, and
produces a standalone executable suitable for direct NCU profiling.

No subprocess — the executable allocates its own GPU memory, fills input
buffers, and launches the kernel.  NCU profiles it cleanly.

Usage:
    # Build
    python3 ncu_profile.py solution.cu --N=1000000 --build-only

    # Profile with NCU
    ncu --kernel-name solve --launch-skip 10 --launch-count 1 --set full \\
        -o report -f ./solution_bench --N=1000000 --warmup=10 --repeat=22
"""

import argparse
import os
import re
import subprocess
import sys

WRAPPER_TEMPLATE = """\
// Auto-generated NCU profiling bench — self-contained, no subprocess.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

// User's kernel source (solve function defined here)
#include "{cu_file_abs}"

// Simple fill kernel for test data
__global__ void fill_data(float *buf, int n, unsigned int seed) {{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) buf[i] = ((float)(i * 2654435761U + seed)) / 4294967296.0f;
}}

int main(int argc, char **argv) {{
    int warmup = 10, repeat = 22, N = {default_n};
    // Parse args
    for (int i = 1; i < argc; i++) {{
        if (strncmp(argv[i], "--N=", 4) == 0) N = atoi(argv[i] + 4);
        if (strncmp(argv[i], "--warmup=", 9) == 0) warmup = atoi(argv[i] + 9);
        if (strncmp(argv[i], "--repeat=", 9) == 0) repeat = atoi(argv[i] + 9);
    }}

    size_t bytes = (size_t)N * sizeof(float);
    int threads = 256, blocks = (N + threads - 1) / threads;

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    fill_data<<<blocks, threads>>>(d_A, N, 1);
    fill_data<<<blocks, threads>>>(d_B, N, 2);
    cudaMemset(d_C, 0, bytes);

    cudaDeviceSynchronize();

    if (N <= 100) {{
        // Verify: solve should compute C = A + B
        solve<<<blocks, threads>>>(d_A, d_B, d_C, N);
        cudaDeviceSynchronize();
        float *h_C = (float*)malloc(bytes);
        cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);
        int ok = 1;
        for (int i = 0; i < N && i < 8; i++) {{
            float expected = ((float)(i * 2654435761U + 1) + (float)(i * 2654435761U + 2)) / 4294967296.0f;
            printf("  C[%d] = %.6f (expected %.6f)\\n", i, h_C[i], expected);
            if (fabsf(h_C[i] - expected) > 0.01f) ok = 0;
        }}
        printf("  verify: %s\\n", ok ? "PASS" : "FAIL");
        free(h_C);
        cudaMemset(d_C, 0, bytes);
    }}

    // Warmup
    for (int i = 0; i < warmup; i++)
        solve<<<blocks, threads>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();

    // Timed
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    double total = 0.0;
    for (int i = 0; i < repeat; i++) {{
        cudaEventRecord(start, 0);
        solve<<<blocks, threads>>>(d_A, d_B, d_C, N);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        float ms; cudaEventElapsedTime(&ms, start, stop);
        total += ms;
        if (i < 2 || i >= repeat - 2) printf("  iter %d: %.4f ms\\n", i, ms);
    }}
    printf("  avg: %.4f ms\\n", total / repeat);

    cudaEventDestroy(start); cudaEventDestroy(stop);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    printf("done\\n");
    return 0;
}}
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cu_file")
    parser.add_argument("--arch", type=str, default="")
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=22)

    args, unknown = parser.parse_known_args()
    dims = {}
    for u in unknown:
        if u.startswith("--") and "=" in u:
            k, v = u[2:].split("=", 1)
            try: dims[k] = int(v)
            except ValueError: pass

    default_n = dims.get("N", 1000000)

    # Resolve absolute path for #include
    cu_abs = os.path.abspath(args.cu_file)

    # Detect arch
    arch = args.arch
    if not arch:
        try:
            import torch
            if torch.cuda.is_available():
                mj, mn = torch.cuda.get_device_capability(0)
                arch = f"sm_{mj}{mn}"
        except ImportError:
            arch = "sm_80"

    # Write wrapper and compile
    exe_path = os.path.splitext(args.cu_file)[0] + "_bench"
    wrapper_src = WRAPPER_TEMPLATE.format(cu_file_abs=cu_abs, default_n=default_n)
    wrapper_path = exe_path + "_wrapper.cu"
    with open(wrapper_path, "w") as f:
        f.write(wrapper_src)

    cmd = ["nvcc", f"-arch={arch}", "-O3", "-lineinfo", "-o", exe_path, wrapper_path]
    print(f"[compile] {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True)
    os.remove(wrapper_path)

    if result.returncode != 0:
        print(f"Compilation failed:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)
    print(f"[compile] -> {exe_path}", flush=True)

    if args.build_only:
        # Default: --set launch works on restricted containers without PMU access.
        # For unrestricted hosts, replace with --set full for detailed metrics.
        print(f"\nReady for NCU:\n"
              f"  ncu --kernel-name solve --launch-skip {args.warmup} "
              f"--launch-count 1 --set launch -o report.ncu-rep -f "
              f"{exe_path} --N={default_n} --warmup={args.warmup} --repeat={args.repeat}\n"
              f"\n"
              f"  # For detailed metrics (requires host PMU access: "
              f"perf_event_paranoid=0):\n"
              f"  # ncu ... --set full ...")
        return

    exe_args = [exe_path, f"--N={default_n}",
                f"--warmup={args.warmup}", f"--repeat={args.repeat}"]
    subprocess.run(exe_args)


if __name__ == "__main__":
    main()
