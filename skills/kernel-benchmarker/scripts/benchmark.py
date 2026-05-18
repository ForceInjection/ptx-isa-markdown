#!/usr/bin/env python3
"""Generic CUDA kernel benchmark (with optional correctness validation).

When --ref is provided:
  1. Validates kernel correctness against the reference implementation.
     Exits immediately if validation fails.
  2. Benchmarks the reference implementation.
  3. Benchmarks the CUDA kernel.
  4. Prints a combined summary with speedup.

When --ref is omitted:
  Benchmarks the CUDA kernel only.

The CUDA kernel is compiled to PTX and loaded via the CUDA Driver API
(cuModuleLoadData + cuLaunchKernel) in the same process, avoiding
cross-process CUDA memory isolation.

Usage:
    python benchmark.py <solution.cu> [--DIM=VALUE ...] [options]
    python benchmark.py <solution.cu> --ref=<ref.py> [--DIM=VALUE ...] [options]

ref.py format
-------------
    import torch

    def reference(*, A, B, C, M, K, N, **kwargs):
        C[:] = (A.reshape(M, K) @ B.reshape(K, N)).reshape(-1)

    # Optional tolerance overrides
    atol = 1e-4
    rtol = 1e-3
"""

import argparse
import ctypes
import importlib.util
import os
import re
import subprocess
import sys

import torch

# ---------------------------------------------------------------------------
# Type tables
# ---------------------------------------------------------------------------

SUPPORTED_TYPES = {
    "float*":          ("float*",          ctypes.c_void_p),
    "double*":         ("double*",         ctypes.c_void_p),
    "unsigned char*":  ("unsigned char*",  ctypes.c_void_p),
    "unsigned short*": ("unsigned short*", ctypes.c_void_p),
    "unsigned int*":   ("unsigned int*",   ctypes.c_void_p),
    "char*":           ("char*",           ctypes.c_void_p),
    "short*":          ("short*",          ctypes.c_void_p),
    "long*":           ("long*",           ctypes.c_void_p),
    "int*":            ("int*",            ctypes.c_void_p),
    "int":             ("int",             ctypes.c_int),
    "long":            ("long",            ctypes.c_long),
    "size_t":          ("size_t",          ctypes.c_size_t),
    "unsigned int":    ("unsigned int",    ctypes.c_uint),
    "unsigned short":  ("unsigned short",  ctypes.c_ushort),
    "unsigned char":   ("unsigned char",   ctypes.c_ubyte),
    "char":            ("char",            ctypes.c_char),
    "short":           ("short",           ctypes.c_short),
}

DTYPE_MAP = {
    "float*":          torch.float32,
    "double*":         torch.float64,
    "int*":            torch.int32,
    "long*":           torch.int64,
    "short*":          torch.int16,
    "char*":           torch.int8,
    "unsigned char*":  torch.uint8,
    "unsigned short*": getattr(torch, "uint16", torch.int16),
    "unsigned int*":   getattr(torch, "uint32", torch.int32),
}

INT_TYPES = {"int", "long", "size_t", "unsigned int"}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_solve_signature(cu_file: str):
    """Extract parameter list from `extern "C" ... void solve(...)` in a .cu file."""
    with open(cu_file, "r") as f:
        content = f.read()

    # Strip C-style comments to avoid matching signatures in comments
    content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
    content = re.sub(r'//[^\n]*', '', content)

    pattern = r'extern\s+"C"\s+(?:__global__\s+)?void\s+solve\s*\(([\s\S]*?)\)\s*\{'
    match = re.search(pattern, content)
    if not match:
        raise ValueError(
            f'Cannot find \'extern "C" void solve(...)\' in {cu_file}'
        )

    raw = match.group(1)
    raw = re.sub(r"/\*.*?\*/", "", raw)
    raw = re.sub(r"//[^\n]*", "", raw)
    raw = " ".join(raw.split())

    params = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        is_const = "const" in token
        token_clean = re.sub(r"\s+", " ", token.replace("const", "").strip())
        matched = False
        for key in sorted(SUPPORTED_TYPES.keys(), key=len, reverse=True):
            base = key.replace("*", r"\s*\*")
            m = re.match(rf"({base})\s+(\w+)", token_clean)
            if m:
                params.append((key, m.group(2), is_const))
                matched = True
                break
        if not matched:
            raise ValueError(f"Cannot parse parameter: '{token.strip()}'")

    return params


def detect_arch() -> str:
    """Auto-detect GPU compute capability and return sm_XX string."""
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability(0)
        return f"sm_{major}{minor}"
    return "sm_80"


_STRIP_INCLUDES = re.compile(
    r'^\s*#\s*include\s*<__clang_cuda[^>]*>\s*$', re.MULTILINE
)


def _preprocess_cu(cu_file: str) -> str:
    """Strip clang-specific includes. Returns path to clean file."""
    with open(cu_file, "r") as f:
        src = f.read()
    cleaned = _STRIP_INCLUDES.sub("", src)
    if cleaned == src:
        return cu_file
    tmp = cu_file + ".nvcc_clean.cu"
    with open(tmp, "w") as f:
        f.write(cleaned)
    return tmp


def _ensure_global(cu_file: str) -> str:
    """If the solve function lacks __global__, rewrite with __global__ added."""
    with open(cu_file, "r") as f:
        src = f.read()

    # If __global__ is already there, return as-is
    if re.search(r'extern\s+"C"\s+__global__\s+void\s+solve', src):
        return cu_file

    # Add __global__ before void solve
    new_src = re.sub(
        r'(extern\s+"C"\s+)void\s+solve',
        r'\1__global__ void solve',
        src
    )
    if new_src == src:
        return cu_file

    tmp = cu_file + ".global.cu"
    with open(tmp, "w") as f:
        f.write(new_src)
    return tmp


# ---------------------------------------------------------------------------
# Compile .cu → load with torch
# ---------------------------------------------------------------------------

def _compile_and_load(cu_file: str, arch: str, force_recompile: bool = False):
    """Compile .cu → .ptx, load with PyTorch, return kernel launcher.

    Caches the PTX file: if the .ptx exists and is newer than the .cu source,
    skips recompilation (unless force_recompile=True).  This avoids spawning
    nvcc subprocesses, which is essential for NCU profiling since NCU
    disconnects when child processes exit.
    """
    ptx_file = os.path.splitext(cu_file)[0] + ".ptx"

    need_compile = force_recompile or not os.path.exists(ptx_file)
    if not need_compile:
        ptx_mtime = os.path.getmtime(ptx_file)
        cu_mtime = os.path.getmtime(cu_file)
        need_compile = cu_mtime > ptx_mtime

    if need_compile:
        clean_file = _preprocess_cu(cu_file)
        global_file = _ensure_global(clean_file)

        cmd = ["nvcc", "-ptx", f"-arch={arch}", "-O3", "-o", ptx_file, global_file]
        print(f"[compile] {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)

        for tmp in [clean_file, global_file]:
            if tmp != cu_file and os.path.exists(tmp):
                os.remove(tmp)

        if result.returncode != 0:
            print(f"Compilation failed:\n{result.stderr}", file=sys.stderr)
            sys.exit(1)
        print(f"[compile] -> {ptx_file}")
    else:
        print(f"[compile] using cached {ptx_file}")

    # Load PTX with PyTorch
    with open(ptx_file, "r") as f:
        ptx_src = f.read()

    # Create a CUDA module from PTX
    # Use the low-level ctypes approach to load PTX
    import ctypes
    libnames = ("libcuda.so", "libcuda.so.1", "/usr/local/cuda/lib64/libcuda.so",
                "/usr/local/cuda-12.8/lib64/libcuda.so", "/usr/local/cuda-12/lib64/libcuda.so")
    cuda = None
    for name in libnames:
        try:
            cuda = ctypes.CDLL(name)
            break
        except OSError:
            continue
    if cuda is None:
        raise RuntimeError("Cannot load libcuda.so")

    # CUDA Driver API types
    CUresult = ctypes.c_int
    CUDA_SUCCESS = 0

    cuda.cuInit(0)
    cuda.cuCtxGetCurrent.restype = CUresult

    # Get current context
    ctx = ctypes.c_void_p()
    err = cuda.cuCtxGetCurrent(ctypes.byref(ctx))
    if err != CUDA_SUCCESS:
        raise RuntimeError(f"cuCtxGetCurrent failed: {err}")

    # Load module from PTX
    module = ctypes.c_void_p()
    ptx_bytes = ptx_src.encode("utf-8")
    err = cuda.cuModuleLoadData(ctypes.byref(module), ctypes.c_char_p(ptx_bytes))
    if err != CUDA_SUCCESS:
        raise RuntimeError(f"cuModuleLoadData failed: {err}")

    # Get kernel function
    kernel = ctypes.c_void_p()
    err = cuda.cuModuleGetFunction(ctypes.byref(kernel), module, b"solve")
    if err != CUDA_SUCCESS:
        raise RuntimeError(f"cuModuleGetFunction failed: {err} (err={err})")

    return cuda, module, kernel, ptx_file


def _launch_kernel(cuda, kernel, params: list, dim_values: dict,
                   kernel_tensors: dict):
    """Launch the kernel via CUDA Driver API with the given tensors."""
    CUDA_SUCCESS = 0

    ptr_params = [(t, n, c) for t, n, c in params if t in DTYPE_MAP]
    int_params = [(t, n) for t, n, c in params if t in INT_TYPES]

    # Build kernel arguments
    args = []
    for ptype, pname, is_const in ptr_params:
        args.append(ctypes.c_void_p(kernel_tensors[pname].data_ptr()))
    for itype, iname in int_params:
        val = dim_values[iname]
        if itype in ("int", "unsigned int"):
            args.append(ctypes.c_int(val))
        elif itype in ("long", "size_t"):
            args.append(ctypes.c_long(val))
        else:
            args.append(ctypes.c_int(val))

    args_array = (ctypes.c_void_p * len(args))(*[ctypes.cast(ctypes.pointer(a), ctypes.c_void_p) for a in args])

    # Determine grid/block dims from the total element count
    total = 256  # default
    for itype, iname in int_params:
        total = dim_values[iname]
        break

    threads = 256
    blocks = (total + threads - 1) // threads

    err = cuda.cuLaunchKernel(
        kernel,
        blocks, 1, 1,        # grid dims
        threads, 1, 1,       # block dims
        0,                    # shared memory bytes
        ctypes.c_void_p(0),  # stream (0 = default)
        args_array,
        None                  # extra options
    )
    if err != CUDA_SUCCESS:
        raise RuntimeError(f"cuLaunchKernel failed: {err}")


# ---------------------------------------------------------------------------
# Reference loading
# ---------------------------------------------------------------------------

def load_reference(ref_file: str):
    """Import a Python reference file and return its module."""
    if not os.path.exists(ref_file):
        raise FileNotFoundError(f"Reference file not found: {ref_file}")
    spec = importlib.util.spec_from_file_location("_ref_module", ref_file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "reference"):
        raise AttributeError(
            f"'{ref_file}' must define a `reference(**kwargs)` function."
        )
    return mod


def _determine_ptr_elems(int_values: list, ptr_size_override: int) -> int:
    if ptr_size_override > 0:
        return ptr_size_override
    elif len(int_values) == 0:
        return 1024 * 1024
    elif len(int_values) == 1:
        return int_values[0]
    else:
        sv = sorted(int_values, reverse=True)
        return min(sv[0] * sv[1], 256 * 1024 * 1024)


def _fmt_vals(vals, width=10):
    return "[" + ", ".join(f"{v:>{width}.4f}" for v in vals) + "]"


def _color(text: str, ok: bool) -> str:
    if not sys.stdout.isatty():
        return text
    code = "\033[92m" if ok else "\033[91m"
    return f"{code}{text}\033[0m"


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

def _time_iterations(fn, warmup: int, repeat: int) -> list:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event   = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(repeat):
        fn()
    end_event.record()
    torch.cuda.synchronize()

    avg_ms = start_event.elapsed_time(end_event) / repeat
    return [avg_ms] * repeat


def _stats(times_ms: list):
    avg = sum(times_ms) / len(times_ms)
    med = sorted(times_ms)[len(times_ms) // 2]
    return avg, med, min(times_ms), max(times_ms)


# ---------------------------------------------------------------------------
# Results printer
# ---------------------------------------------------------------------------

def _print_results(label, avg, med, mn, mx, total_ptr_bytes, ptr_elems,
                   cu_file, dim_values, arch, ref_avg=None):
    print()
    print("=" * 55)
    print(f"  {label}")
    print(f"  Kernel       : {os.path.basename(cu_file)}")
    print(f"  GPU          : {torch.cuda.get_device_name(0)}")
    print(f"  Arch         : {arch}")
    print(f"  Dims         : {dim_values}")
    print(f"  Buf/ptr      : {ptr_elems} elems")
    print("-" * 55)
    print(f"  Average      : {avg:>10.4f} ms")
    print(f"  Median       : {med:>10.4f} ms")
    print(f"  Min          : {mn:>10.4f} ms")
    print(f"  Max          : {mx:>10.4f} ms")
    if avg > 0:
        bw = total_ptr_bytes / (avg / 1000) / 1e9
        print(f"  ~Bandwidth   : {bw:>10.2f} GB/s  (all ptrs, rough)")
    if ref_avg is not None and avg > 0:
        speedup = ref_avg / avg
        print(f"  Speedup      : {speedup:>10.2f}x  vs reference")
    print("=" * 55)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate_outputs(kernel_tensors, ref_tensors, output_params, atol, rtol):
    PREVIEW = 8
    print(f"\n[validate] {len(output_params)} output tensor(s)\n")

    all_pass = True
    for pname, ptype in output_params:
        kt = kernel_tensors[pname].float()
        rt = ref_tensors[pname].float()

        match = torch.allclose(kt, rt, atol=atol, rtol=rtol)
        if not match:
            all_pass = False

        max_diff  = (kt - rt).abs().max().item()
        mean_diff = (kt - rt).abs().mean().item()
        rel_err   = ((kt - rt).abs() / rt.abs().clamp(min=1e-8)).mean().item()

        status_str = _color("PASS" if match else "FAIL", match)
        print(f"  [{status_str}]  {pname}  ({ptype})")
        print(f"         max |Δ|   = {max_diff:.6e}")
        print(f"         mean |Δ|  = {mean_diff:.6e}")
        print(f"         mean rel  = {rel_err:.6e}")

        if not match:
            diff_mask = ~torch.isclose(kt, rt, atol=atol, rtol=rtol)
            bad_idx   = diff_mask.nonzero(as_tuple=True)[0]
            n_bad     = bad_idx.numel()
            print(f"         mismatches: {n_bad} / {kt.numel()}")
            if n_bad > 0:
                idx = bad_idx[0].item()
                print(f"         first bad   @ idx={idx}:  "
                      f"kernel={kt[idx].item():.6f}  ref={rt[idx].item():.6f}")

        k_preview = kernel_tensors[pname][:PREVIEW].float().cpu().tolist()
        r_preview = ref_tensors[pname][:PREVIEW].float().cpu().tolist()
        print(f"         kernel[:{PREVIEW}] = {_fmt_vals(k_preview)}")
        print(f"         ref   [:{PREVIEW}] = {_fmt_vals(r_preview)}")
        print()

    return all_pass


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def _setup(cu_file, dim_values, ptr_size_override, arch, seed=None, force_recompile=False):
    params = parse_solve_signature(cu_file)
    sig_str = ", ".join(f"{'const ' if c else ''}{t} {n}" for t, n, c in params)
    print(f"[signature] solve({sig_str})\n")

    cuda, module, kernel, ptx_file = _compile_and_load(cu_file, arch, force_recompile=force_recompile)

    for ptype, pname, _ in params:
        if ptype in INT_TYPES and pname not in dim_values:
            raise ValueError(
                f"Missing dimension: --{pname}=<value>  (required by kernel signature)"
            )

    int_vals = [dim_values[pname]
                for ptype, pname, _ in params if ptype in INT_TYPES]
    ptr_elems = _determine_ptr_elems(int_vals, ptr_size_override)

    if seed is not None:
        torch.manual_seed(seed)

    kernel_tensors: dict = {}
    output_params = []

    print("[buffers]")
    for ptype, pname, is_const in params:
        if ptype in DTYPE_MAP:
            dtype = DTYPE_MAP[ptype]
            if dtype.is_floating_point:
                t = torch.randn(ptr_elems, device="cuda", dtype=dtype)
            else:
                t = torch.zeros(ptr_elems, device="cuda", dtype=dtype).random_()
            kernel_tensors[pname] = t
            if not is_const:
                output_params.append((pname, ptype))
            role = "input" if is_const else "output"
            eb   = t.element_size()
            print(
                f"  {pname:>10s} : {ptype:<16s} [{role:>6s}] "
                f"{ptr_elems} elems  ({ptr_elems * eb / 1024 / 1024:.1f} MB)"
            )
        elif ptype in SUPPORTED_TYPES:
            val = dim_values[pname]
            print(f"  {pname:>10s} : {ptype:<16s} = {val}")

    total_ptr_bytes = sum(t.nelement() * t.element_size()
                          for t in kernel_tensors.values())

    return (cuda, kernel, params, kernel_tensors, output_params,
            ptr_elems, total_ptr_bytes)


def run(cu_file, ref_file, dim_values, warmup, repeat,
        ptr_size_override, arch, atol, rtol, seed, force_recompile=False):
    has_ref = bool(ref_file)

    ref_fn = None
    ref_kwargs = None
    _atol = atol
    _rtol = rtol

    if has_ref:
        ref_mod = load_reference(ref_file)
        ref_fn  = ref_mod.reference
        _atol   = float(getattr(ref_mod, "atol", atol))
        _rtol   = float(getattr(ref_mod, "rtol", rtol))
        print(f"[reference] {ref_file}  (atol={_atol}, rtol={_rtol})\n")

    # -- compile + allocate ---------------------------------------------------
    (cuda, kernel, params, kernel_tensors, output_params,
     ptr_elems, total_ptr_bytes) = _setup(
        cu_file, dim_values, ptr_size_override, arch,
        seed=seed if has_ref else None, force_recompile=force_recompile
    )

    if not output_params and has_ref:
        print("\n[warn] No output tensors detected (all pointer params are const). "
              "Nothing to validate.", file=sys.stderr)

    # -------------------------------------------------------------------------
    # Step 1: correctness check
    # -------------------------------------------------------------------------
    if has_ref:
        ref_tensors = {pname: t.clone() for pname, t in kernel_tensors.items()}

        ref_kwargs = {}
        for ptype, pname, _ in params:
            if ptype in DTYPE_MAP:
                ref_kwargs[pname] = ref_tensors[pname]
            else:
                ref_kwargs[pname] = dim_values[pname]

        print("\n[kernel]    running ... ", end="", flush=True)
        _launch_kernel(cuda, kernel, params, dim_values, kernel_tensors)
        torch.cuda.synchronize()
        print("done")

        print("[reference] running ... ", end="", flush=True)
        ref_fn(**ref_kwargs)
        torch.cuda.synchronize()
        print("done")

        validation_passed = _validate_outputs(
            kernel_tensors, ref_tensors, output_params, _atol, _rtol
        )

        print("=" * 60)
        print(f"  Kernel    : {os.path.basename(cu_file)}")
        print(f"  Reference : {os.path.basename(ref_file)}")
        print(f"  GPU       : {torch.cuda.get_device_name(0)}")
        print(f"  Arch      : {arch}")
        print(f"  Dims      : {dim_values}")
        print(f"  Buf/ptr   : {ptr_elems} elems")
        print(f"  Tolerance : atol={_atol}  rtol={_rtol}")
        print("-" * 60)
        result_str = "ALL PASS ✓" if validation_passed else "FAILED ✗"
        print(f"  Result    : {_color(result_str, validation_passed)}")
        print("=" * 60)

        if not validation_passed:
            sys.exit(1)

    # -------------------------------------------------------------------------
    # Step 2: benchmark reference
    # -------------------------------------------------------------------------
    times_ref = None
    if has_ref:
        print(f"\n[warmup] reference  {warmup} iterations ...")
        times_ref = _time_iterations(lambda: ref_fn(**ref_kwargs), warmup, repeat)
        print(f"[bench]  reference  {repeat} iterations ... done")

    # -------------------------------------------------------------------------
    # Step 3: benchmark kernel
    # -------------------------------------------------------------------------
    if not has_ref:
        PREVIEW = 8
        tensor_info = [
            (pname, ptype, "input" if is_const else "output", kernel_tensors[pname])
            for ptype, pname, is_const in params if ptype in DTYPE_MAP
        ]
        print(f"\n[preview] first {PREVIEW} elements before kernel call:")
        for name, ptype, role, t in tensor_info:
            tag = "IN " if role == "input" else "OUT"
            print(f"  {tag} {name:>6s} = {_fmt_vals(t[:PREVIEW].cpu().tolist())}")

        _launch_kernel(cuda, kernel, params, dim_values, kernel_tensors)
        torch.cuda.synchronize()

        print(f"\n[preview] first {PREVIEW} elements after 1 kernel call:")
        for name, ptype, role, t in tensor_info:
            tag = "IN " if role == "input" else "OUT"
            print(f"  {tag} {name:>6s} = {_fmt_vals(t[:PREVIEW].cpu().tolist())}")

    print(f"\n[warmup] kernel  {warmup} iterations ...")
    for _ in range(warmup):
        _launch_kernel(cuda, kernel, params, dim_values, kernel_tensors)
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event   = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(repeat):
        _launch_kernel(cuda, kernel, params, dim_values, kernel_tensors)
    end_event.record()
    torch.cuda.synchronize()

    avg_ms = start_event.elapsed_time(end_event) / repeat
    times_kernel = [avg_ms] * repeat
    print(f"[bench]  kernel  {repeat} iterations ... done")

    # -------------------------------------------------------------------------
    # Step 4: print summary
    # -------------------------------------------------------------------------
    avg_k, med_k, mn_k, mx_k = _stats(times_kernel)

    if has_ref:
        avg_r, med_r, mn_r, mx_r = _stats(times_ref)
        _print_results(
            "CUDA Kernel", avg_k, med_k, mn_k, mx_k,
            total_ptr_bytes, ptr_elems, cu_file, dim_values, arch, ref_avg=avg_r,
        )
        _print_results(
            f"Reference ({os.path.basename(ref_file)})",
            avg_r, med_r, mn_r, mx_r,
            total_ptr_bytes, ptr_elems, cu_file, dim_values, arch,
        )
    else:
        _print_results(
            "CUDA Kernel", avg_k, med_k, mn_k, mx_k,
            total_ptr_bytes, ptr_elems, cu_file, dim_values, arch,
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generic CUDA kernel benchmark (with optional validation)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("cu_file", help="Path to .cu solution file")
    parser.add_argument("--ref", type=str, default="",
                        help="Path to reference .py file")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--ptr-size", type=int, default=0)
    parser.add_argument("--arch", type=str, default="")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force-recompile", action="store_true",
                        help="Force PTX recompilation even if cached")

    args, unknown = parser.parse_known_args()

    dim_values: dict = {}
    for u in unknown:
        if u.startswith("--") and "=" in u:
            key, val = u[2:].split("=", 1)
            dim_values[key] = int(val)
        else:
            print(f"Warning: ignoring unknown arg '{u}'", file=sys.stderr)

    torch.cuda.set_device(args.gpu)
    arch = args.arch if args.arch else detect_arch()

    run(
        cu_file           = args.cu_file,
        ref_file          = args.ref,
        dim_values        = dim_values,
        warmup            = args.warmup,
        repeat            = args.repeat,
        ptr_size_override = args.ptr_size,
        arch              = arch,
        atol              = args.atol,
        rtol              = args.rtol,
        seed              = args.seed,
        force_recompile   = args.force_recompile,
    )


if __name__ == "__main__":
    main()
