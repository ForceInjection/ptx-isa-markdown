# NVIDIA CUDA Documentation + Claude Code Skill

NVIDIA's PTX ISA 9.1, CUDA Runtime API 13.1, CUDA Driver API 13.1, CUDA Math API 13.x, cuBLAS 13.2, and NCCL documentation converted to searchable markdown, with a Claude Code skill for GPU development.

## What's Here

1. **PTX ISA 9.1 Documentation** (405 markdown files, 2.3MB)
   - Complete instruction set reference
   - All tables, code blocks, and mathematical notation preserved
   - Organized by chapter with section numbers
   - Images linked to NVIDIA's CDN

2. **CUDA Runtime API 13.1 Documentation** (104 markdown files, 1.2MB)
   - Complete function and data structure reference
   - 37 API modules (device, memory, streams, events, graphs, etc.)
   - 66 data structures (cudaDeviceProp, cudaMemcpy3DParms, etc.)
   - All parameters, return values, and detailed descriptions
   - Navigation, duplicate content, redundant URLs, and boilerplate removed

3. **CUDA Driver API 13.1 Documentation** (129 markdown files, 1.2MB)
   - Complete low-level driver API reference
   - 49 API modules (context, module loading, virtual memory, graphs, etc.)
   - 79 data structures (CUdevice, CUcontext, launch parameters, etc.)
   - All parameters, return values, and detailed descriptions
   - Navigation, duplicate TOC, cross-references, and redundant URLs removed

4. **CUDA Math API 13.x Documentation** (41 markdown files, 0.5MB)
   - Complete device math intrinsics reference
   - 14 API modules (single, double, half, FP8, SIMD, cast, etc.)
   - 26 data structures (FP8 types, half types, etc.)
   - Fast intrinsics, narrow-precision types, and type casting functions

5. **cuBLAS 13.2 Documentation** (319 markdown files, 2.9MB)
   - Complete GEMM, batched ops, cuBLASLt, cuBLASXt, and legacy API reference
   - Organized by chapter: cuBLAS API, cuBLASLt, cuBLASXt, cuBLASDx, legacy, Fortran bindings
   - Mixed-precision GEMM (FP8, BF16, FP16), fused epilogues (bias, ReLU, GELU)
   - All function signatures, parameters, return values, and descriptions

6. **NCCL Documentation** (34 markdown files, 0.5MB)
   - Complete collective operations, communicator, and P2P API reference
   - 12 API pages (collectives, communicators, P2P, groups, device ops, types, etc.)
   - 11 usage guides (multi-node setup, CUDA graph, MPI integration, etc.)
   - Environment variables, troubleshooting, and migration from NCCL 1

7. **CUDA Development Skill** for Claude Code
   - PTX instruction lookup and examples
   - CUDA Runtime & Driver API function reference
   - CUDA Math intrinsics (fast device math, FP8, half precision)
   - cuBLAS / cuBLASLt GEMM and batched operations
   - NCCL collective operations and multi-GPU communication
   - Profiling workflows (nsys, ncu)
   - Debugging patterns (compute-sanitizer, cuda-gdb)
   - TensorCore operation reference (WMMA, WGMMA, TMA)

## Why

NVIDIA's official documentation is:

- **PTX ISA**: A single 5MB HTML page requiring Ctrl+F through megabytes
- **CUDA Runtime API**: 75+ separate HTML pages requiring multiple clicks
- **CUDA Driver API**: 130+ separate HTML pages requiring multiple clicks
- **CUDA Math API**: 40+ separate HTML pages requiring multiple clicks
- **cuBLAS**: 300+ separate HTML pages across multiple API tiers
- **NCCL**: Sphinx multi-page site spread across usage, API, and env sections

This conversion enables:

- `grep -r "register fragment" ptx-docs/` instead of Ctrl+F
- `grep -r "cudaErrorInvalidValue" cuda-runtime-docs/` instead of clicking through modules
- `grep -r "cuCtxCreate" cuda-driver-docs/` for low-level API lookup
- `grep -r "cublasGemmEx" cublas-docs/` for cuBLAS GEMM signatures
- `grep -r "ncclAllReduce" nccl-docs/` for NCCL collective operations
- `grep -r "__expf\|__fmaf_rn" cuda-math-docs/` for fast device intrinsics
- Direct file access for AI tools (Claude, Copilot)
- Offline reference with proper organization

**Example 1**: Find how to disable TMA swizzling:

```bash
$ grep -r "swizzle_mode.*no swizzling" cuda_skill/references/ptx-docs/
9.7.9.28-data-movement-and-conversion-instructionstensormapreplace.md:
  0 | `.u8` | No interleave | No swizzling | 16B | Zero fill
```

Answer: use `tensormap.replace` with `.swizzle_mode = 0`.

**Example 2**: Look up what cudaErrorInvalidValue means:

```bash
grep -A 5 "cudaErrorInvalidValue" cuda_skill/references/cuda-runtime-docs/
```

Answer: Instantly find error code documentation with description and related errors.

**Example 3**: Understand context management in Driver API:

```bash
grep -A 20 "cuCtxCreate" cuda_skill/references/cuda-driver-docs/modules/group__cuda__ctx.md
```

Answer: Full cuCtxCreate parameters, return values, and context behavior.

**Example 4**: Look up cuBLAS mixed-precision GEMM:

```bash
grep -A 30 "cublasGemmEx" cuda_skill/references/cublas-docs/2-using-the-cublas-api/
```

Answer: Full cublasGemmEx signature with compute type and algorithm options.

**Example 5**: Find NCCL collective operation signatures:

```bash
grep -A 20 "^## ncclAllReduce" cuda_skill/references/nccl-docs/api/colls.md
```

Answer: Full ncclAllReduce signature, parameters, and usage notes.

## Structure

```text
cuda_skill/                              # Portable Claude Code skill (~8.7MB)
├── SKILL.md                             # Main skill definition
└── references/
    ├── ptx-docs/                        # 405 markdown files (2.3MB)
    │   ├── 9-instruction-set/           # 186 instruction files
    │   │   ├── 9.7.15.5-*.md           # WGMMA register layouts
    │   │   └── 9.7.16-*.md             # TensorCore Gen5 (Blackwell)
    │   ├── 5-state-spaces-types-and-variables/
    │   ├── 8-memory-consistency-model/
    │   └── INDEX.md
    ├── cuda-runtime-docs/               # 104 markdown files (1.2MB)
    │   ├── modules/                     # 37 API modules
    │   │   ├── group__cudart__device.md
    │   │   ├── group__cudart__memory.md
    │   │   ├── group__cudart__stream.md
    │   │   └── ...
    │   ├── data-structures/             # 66 structs/unions
    │   │   ├── structcudadeviceprop.md
    │   │   ├── structcudamemcpy3dparms.md
    │   │   └── ...
    │   └── INDEX.md
    ├── cuda-driver-docs/                # 129 markdown files (1.2MB)
    │   ├── modules/                     # 49 API modules
    │   │   ├── group__cuda__ctx.md
    │   │   ├── group__cuda__mem.md
    │   │   ├── group__cuda__stream.md
    │   │   ├── group__cuda__module.md
    │   │   ├── group__cuda__va.md
    │   │   └── ...
    │   ├── data-structures/             # 79 structs
    │   │   ├── structcudevprop__v1.md
    │   │   ├── structcuda__memcpy3d__v2.md
    │   │   └── ...
    │   └── INDEX.md
    ├── cuda-math-docs/                  # 41 markdown files (0.5MB)
    │   ├── modules/                     # 14 API modules
    │   │   ├── group__cuda__math__single.md
    │   │   ├── group__cuda__math__intrinsic__half.md
    │   │   ├── group__cuda__math__intrinsic__cast.md
    │   │   └── ...
    │   ├── data-structures/             # 26 FP8/half types
    │   └── INDEX.md
    ├── cublas-docs/                     # 319 markdown files (2.9MB)
    │   ├── 1-introduction/
    │   ├── 2-using-the-cublas-api/      # cublasSgemm, cublasGemmEx, etc.
    │   ├── 3-using-the-cublaslt-api/    # cublasLtMatmul, epilogues
    │   ├── 4-using-the-cublasxt-api/
    │   ├── 5-using-the-cublasdx-api/
    │   ├── 6-using-the-cublas-legacy-api/
    │   ├── 7-cublas-fortran-bindings/
    │   └── 8-interaction-with-other-libraries-and-tools/
    ├── nccl-docs/                       # 34 markdown files (0.5MB)
    │   ├── api/                         # 12 API pages (collectives, comms, P2P, etc.)
    │   ├── usage/                       # 11 usage guides
    │   ├── troubleshooting/             # RAS subsystem guide
    │   ├── env.md                       # All NCCL_* environment variables
    │   ├── api.md                       # API overview
    │   └── INDEX.md
    ├── ptx-isa.md                       # PTX search guide and examples
    ├── cuda-runtime.md                  # Runtime API search guide
    ├── cuda-driver.md                   # Driver API search guide
    ├── cuda-math.md                     # Math API search guide
    ├── cublas.md                        # cuBLAS search guide
    ├── nccl.md                          # NCCL search guide
    ├── nsys-guide.md                    # Nsight Systems patterns
    ├── ncu-guide.md                     # Nsight Compute metrics
    ├── nvtx-patterns.md                 # NVTX instrumentation patterns
    ├── performance-traps.md             # Bank conflicts, coalescing, scale traps
    └── debugging-tools.md               # compute-sanitizer, cuda-gdb

scrape_cuda_docs.py                      # Unified scraper (uv script, all doc types)
```

## Using the Skill

Install:

```bash
cp -r cuda_skill ~/.claude/skills/cuda
```

The skill activates automatically for CUDA work. Ask Claude:

- "What's the register fragment layout for WGMMA m64n16k16?"
- "How do I disable TMA swizzling?"
- "What does cudaErrorInvalidValue mean?"
- "What fields are in cudaDeviceProp?"
- "How do I create a CUDA context with the Driver API?"
- "What's the difference between cuMemAlloc and cudaMalloc?"
- "Profile this kernel with nsys"
- "What's the signature of cublasGemmEx with FP8?"
- "How do I set up ncclAllReduce across multiple GPUs?"
- "What does __fmaf_rn do vs regular fma?"

Claude searches the local documentation and provides answers with section references.

## Search Examples

### PTX ISA

Find WGMMA register fragments:

```bash
grep -r "register fragment" cuda_skill/references/ptx-docs/9-instruction-set/ | grep -i wgmma
```

Find all swizzling modes:

```bash
find cuda_skill/references/ptx-docs -name "*swizzl*"
```

Search for any instruction:

```bash
grep -r "mbarrier.init" cuda_skill/references/ptx-docs/
```

### CUDA Runtime API

Look up error code:

```bash
grep -A 10 "cudaErrorInvalidValue" cuda_skill/references/cuda-runtime-docs/
```

Find device properties:

```bash
cat cuda_skill/references/cuda-runtime-docs/data-structures/structcudadeviceprop.md
```

### CUDA Driver API

Look up context creation:

```bash
grep -A 20 "cuCtxCreate" cuda_skill/references/cuda-driver-docs/modules/group__cuda__ctx.md
```

Find virtual memory management:

```bash
ls cuda_skill/references/cuda-driver-docs/modules/*va*.md
cat cuda_skill/references/cuda-driver-docs/modules/group__cuda__va.md
```

Understand module loading:

```bash
grep -A 15 "cuModuleLoad" cuda_skill/references/cuda-driver-docs/modules/group__cuda__module.md
```

Search for stream functions:

```bash
grep -r "cudaStreamSynchronize" cuda_skill/references/cuda-runtime-docs/
```

### CUDA Math API

Find fast single-precision intrinsics:

```bash
grep "^__device__ float __" cuda_skill/references/cuda-math-docs/modules/group__cuda__math__intrinsic__single.md | head -20
```

Look up FP8 type conversion:

```bash
cat cuda_skill/references/cuda-math-docs/modules/group__cuda__math__intrinsic__cast.md
```

Find half-precision math:

```bash
grep -A 8 "__hfma\b" cuda_skill/references/cuda-math-docs/modules/group__cuda__math__intrinsic__half.md
```

### cuBLAS

Look up GEMM signatures:

```bash
grep -r "cublasHgemm\|cublasSgemm\|cublasDgemm" cuda_skill/references/cublas-docs/
```

Find cublasGemmEx with mixed precision:

```bash
grep -A 30 "cublasGemmEx\b" cuda_skill/references/cublas-docs/2-using-the-cublas-api/
```

Look up cuBLASLt fused epilogue:

```bash
grep -A 20 "cublasLtMatmul\b" cuda_skill/references/cublas-docs/3-using-the-cublaslt-api/
```

### NCCL

Look up collective operation signatures:

```bash
grep -A 20 "^## ncclAllReduce" cuda_skill/references/nccl-docs/api/colls.md
```

Find environment variables for tuning:

```bash
grep -E "^## NCCL_(ALGO|PROTO|BUFFSIZE)" cuda_skill/references/nccl-docs/env.md
```

Understand communicator initialization:

```bash
cat cuda_skill/references/nccl-docs/usage/communicators.md
```

Debug NCCL issues:

```bash
grep -A 5 "NCCL_DEBUG" cuda_skill/references/nccl-docs/env.md
```

## Regenerating

Run the scraper to update from NVIDIA's latest docs:

```bash
# Update PTX ISA docs
uv run scrape_cuda_docs.py ptx

# Update CUDA Runtime API docs
uv run scrape_cuda_docs.py runtime

# Update CUDA Driver API docs
uv run scrape_cuda_docs.py driver

# Update CUDA Math API docs
uv run scrape_cuda_docs.py math

# Update cuBLAS docs
uv run scrape_cuda_docs.py cublas

# Update NCCL docs
uv run scrape_cuda_docs.py nccl

# Re-run cleanup without re-downloading (uses cached raw files, runtime/driver only)
uv run scrape_cuda_docs.py driver --skip-download

# Force re-download even if cache exists
uv run scrape_cuda_docs.py driver --force

# Custom output directory
uv run scrape_cuda_docs.py ptx --output-dir /path/to/output
```

All scrapers are unified in a single `uv` script with inline PEP 723 dependencies (no separate install needed).

**PTX scraper**:

- Parses single-page HTML documentation
- Splits by section into individual markdown files
- Preserves tables using markdown syntax
- Converts image references to absolute URLs
- Maintains section hierarchy

**Runtime API / Driver API scraper**:

- Crawls 75+/130+ module and data structure pages
- Caches raw files to `*-raw/` directories for fast iteration
- Automatically runs cleanup to produce final docs
- Use `--skip-download` to re-run cleanup without re-downloading
- Navigation, duplicate TOC, redundant URLs, boilerplate removed

**cuBLAS scraper**:

- Sphinx single-page format: scrapes entire doc tree from `index.html`
- Organized by chapter directories matching the official docs structure
- 319 files covering all API tiers (cuBLAS, cuBLASLt, cuBLASXt, cuBLASDx, legacy, Fortran)

**NCCL scraper**:

- Sphinx multi-page format: discovers pages from index and toctree links
- Downloads and organizes into `api/`, `usage/`, and `troubleshooting/` directories
- Also preserves flat top-level pages (env.md, api.md, examples.md, etc.)

**CUDA Math API scraper**:

- Doxygen multi-page format: crawls `modules.html` and `annotated.html`
- 14 API modules + 26 data structures, cleanup removes duplicate TOC/boilerplate

**API cleanup process**:

- Removes duplicate function TOC (detailed docs remain)
- Removes verbose "See also:" cross-references (grep provides same discoverability)
- Removes anchor links, `[inherited]` tags, zero-width spaces
- Removes footer (Privacy Policy, Copyright, NVIDIA logo)
- Removes redundant URLs from markdown links (type/function names preserved)
- Removes generic boilerplate notes (async errors, initialization errors, callback restrictions)
- Cleans up excessive whitespace

## Quality

**PTX ISA**: Tables verified against HTML source. Mathematical notation preserved. 1049 images accessible via NVIDIA CDN links.

**CUDA Runtime API**: All 104 pages successfully converted. Function signatures, parameters, return values, and descriptions fully preserved. Navigation, duplicate content, redundant URLs, and formatting noise removed.

**CUDA Driver API**: All 129 pages successfully converted. Function signatures, parameters, return values, and descriptions fully preserved. Duplicate function TOC, verbose "See also" sections, redundant URLs, and formatting noise removed.

**CUDA Math API**: All 41 pages successfully converted. Device intrinsic signatures, FP8/half type constructors, and SIMD operations fully preserved.

**cuBLAS**: All 319 pages successfully converted. Full coverage of cuBLAS, cuBLASLt, cuBLASXt, cuBLASDx, legacy API, and Fortran bindings.

**NCCL**: 34 pages successfully converted. All collective operations, communicator APIs, P2P, environment variables, and troubleshooting guides preserved.

Verification:

- All function parameters documented
- All return values documented
- Data structure fields accessible
- Real-world queries tested and working

Known limitations:

- Cross-file anchor links don't resolve (use grep instead)
- Images not downloaded locally (fetch from NVIDIA CDN as needed)

## Technical Details

**PTX ISA**:

- Version: 9.1
- Files: 405 markdown files
- Size: 2.3 MB
- Source: https://docs.nvidia.com/cuda/parallel-thread-execution/

**CUDA Runtime API**:

- Version: 13.1
- Files: 104 markdown files (37 modules + 66 data structures)
- Size: 1.2 MB
- Source: https://docs.nvidia.com/cuda/cuda-runtime-api/

**CUDA Driver API**:

- Version: 13.1
- Files: 129 markdown files (49 modules + 79 data structures)
- Size: 1.2 MB
- Source: https://docs.nvidia.com/cuda/cuda-driver-api/

**CUDA Math API**:

- Version: 13.x
- Files: 41 markdown files (14 modules + 26 data structures)
- Size: 0.5 MB
- Source: https://docs.nvidia.com/cuda/cuda-math-api/

**cuBLAS**:

- Version: 13.2
- Files: 319 markdown files (organized by chapter)
- Size: 2.9 MB
- Source: https://docs.nvidia.com/cuda/cublas/index.html

**NCCL**:

- Files: 34 markdown files (api/, usage/, troubleshooting/, flat pages)
- Size: 0.5 MB
- Source: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/

**License**: Documentation © NVIDIA Corporation

The skill uses Claude Code's progressive disclosure: `SKILL.md` is always loaded (~13KB), reference files load on-demand, and documentation is searched rather than loaded into context.

**Total skill size**: ~8.7MB (2.3MB PTX + 1.2MB Runtime + 1.2MB Driver + 0.5MB Math + 2.9MB cuBLAS + 0.5MB NCCL + guides)

## Use Cases

- **Low-level CUDA optimization** — Inline PTX, instruction selection
- **Compiler output understanding** — `cuobjdump -ptx` analysis
- **TensorCore programming** — WMMA/WGMMA/TMA operations
- **Runtime API reference** — Error codes, function parameters, struct fields
- **Driver API reference** — Context management, module loading, virtual memory
- **Multi-context applications** — Explicit context control with Driver API
- **PTX/CUBIN module loading** — Dynamic kernel loading at runtime
- **Advanced memory features** — Virtual memory, multicast, tensor maps
- **Context and stream debugging** — Understanding CUDA Runtime behavior
- **Memory management** — Choosing between malloc variants
- **GEMM optimization** — cuBLAS/cuBLASLt mixed-precision, FP8, batched ops
- **Multi-GPU communication** — NCCL collectives, P2P, multi-node setup
- **Device math intrinsics** — Fast single/half-precision math, FP8 types
- **GPU architecture research**
- **Training AI models** on CUDA/PTX/Runtime API

---

Unofficial conversion for convenience. Refer to NVIDIA's official documentation for authoritative reference:

- [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/)
- [CUDA Runtime API](https://docs.nvidia.com/cuda/cuda-runtime-api/)
- [CUDA Driver API](https://docs.nvidia.com/cuda/cuda-driver-api/)
- [CUDA Math API](https://docs.nvidia.com/cuda/cuda-math-api/)
- [cuBLAS](https://docs.nvidia.com/cuda/cublas/index.html)
- [NCCL](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/)
