# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A multi-skill monorepo for CUDA kernel development assistance, combining scraped NVIDIA documentation (offline RAG knowledge base) with Agent skills that form an automated profile→analyze→optimize loop for GPU kernels. Designed for AI IDE integration (Claude Code, Trae, Qoder).

## Commands

### Documentation Scraper

The scraper is a single `uv` script with inline PEP 723 dependencies — no virtualenv or `pip install` needed:

```bash
uv run nvidia_doc_sync/scrape_cuda_docs.py ptx
uv run nvidia_doc_sync/scrape_cuda_docs.py runtime
uv run nvidia_doc_sync/scrape_cuda_docs.py driver
uv run nvidia_doc_sync/scrape_cuda_docs.py math
uv run nvidia_doc_sync/scrape_cuda_docs.py cublas
uv run nvidia_doc_sync/scrape_cuda_docs.py nccl

# Re-run cleanup only (no re-download):
uv run nvidia_doc_sync/scrape_cuda_docs.py driver --skip-download

# Force re-download ignoring cache:
uv run nvidia_doc_sync/scrape_cuda_docs.py driver --force
```

### Kernel Benchmarking

```bash
# Validate + benchmark kernel against a Python reference
python3 skills/kernel-benchmarker/scripts/benchmark.py <solution.cu> \
    --ref=<ref.py> --M=4096 --N=4096 --K=4096 --repeat=20

# Benchmark only (no validation)
python3 skills/kernel-benchmarker/scripts/benchmark.py <solution.cu> --N=1000000 --repeat=20
```

Reference files must define `def reference(*, <tensors>, <dims>, **kwargs):` with optional module-level `atol`/`rtol`.

### NCU Profiling

```bash
ncu --target-processes all --profile-from-start on \
    --launch-skip 20 --launch-count 1 --set full \
    -o <output_stem> -f \
    python3 skills/kernel-benchmarker/scripts/benchmark.py <solution.cu> \
    --PARAM=VALUE --repeat=22
```

### Searching the Knowledge Base

```bash
# cuBLAS
grep -r "cublasLtMatmul" skills/cuda-knowledge/references/cublas-docs/3-using-the-cublaslt-api/
grep -r "CUBLASLT_EPILOGUE_" skills/cuda-knowledge/references/cublas-docs/

# NCCL
grep -r "ncclAllReduce" skills/cuda-knowledge/references/nccl-docs/api/

# CUDA Math API
grep -r "__nv_fp8_e4m3\|__nv_fp8_e5m2" skills/cuda-knowledge/references/cuda-math-docs/
```

### Finding Code Patterns (cuda-samples)

```bash
# Find samples by pattern or API (search reference files for code)
grep -r "cublasSgemm\|cudaStreamBeginCapture\|__shfl_down_sync" skills/cuda-samples/references/

# Find samples by name or architecture in the quick reference index
grep -r "reduction\|GEMM\|CUDA Graph" skills/cuda-samples/SKILL.md

# Find samples requiring specific GPU architecture
grep -r "SM 8.0\|SM 9.0\|Hopper\|Ampere" skills/cuda-samples/SKILL.md
```

### No lint, test, or build commands are configured.

## Architecture

### Skills Pipeline (skills/)

Six skills form a complete optimization loop, orchestrated by `cuda-optimizer`:

```
cuda-knowledge (API reference docs)  +  cuda-samples (code pattern index)
         ↓                                    ↓
cuda-optimizer (orchestrator)
    ├── kernel-benchmarker   → compile, validate, benchmark
    ├── ncu-rep-analyzer     → NCU profile, diagnose bottleneck, suggest fixes
    └── cuda-code-generator  → generate/rewrite .cu files with optimizations
```

`cuda-knowledge` (~1040 markdown files) provides API reference; `cuda-samples` (SKILL.md ~13 KB, 8 reference files 48 KB; ~50 curated entries) provides concrete working code patterns from official NVIDIA samples with GitHub permalinks and key snippets. Both follow progressive disclosure: lightweight SKILL.md stays in context, detailed references load on demand. All three action skills ground their work in both to reduce hallucination. The optimizer drives this loop: **benchmark → evaluate exit conditions → NCU profile → implement optimizations → repeat** until performance converges (<2% improvement over 2 consecutive rounds).

### Kernel Interface Convention

All `.cu` kernels follow a strict interface:

```cuda
extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    // ...
}
```

- Function name is always `solve`, with `extern "C"` linkage.
- Pointer parameters: `const` prefix means input; no `const` means output.
- Supported types: `float*`, `double*`, `int*`, `unsigned char*`, `unsigned short*`, plus scalar int types.
- `benchmark.py` auto-parses this signature to infer dimension parameter names and allocate tensors.

### Scraper Design (nvidia_doc_sync/scrape_cuda_docs.py)

A single 951-line script with three scraper classes for different NVIDIA doc formats:

| Format | Class | Examples | Strategy |
|--------|-------|----------|----------|
| Sphinx single-page | `SphinxScraper` | PTX ISA, cuBLAS | Split monolithic HTML by heading hierarchy into per-section .md files |
| Sphinx multi-page | `SphinxMultiPageScraper` | NCCL | Crawl toctree links, convert each page |
| Doxygen multi-page | `APIScraper` | Runtime, Driver, Math API | Two-phase: download raw → clean (strip TOC, footer, boilerplate, URLs; 76-83% size reduction). `--skip-download` re-runs only the clean phase. |

### Directory Structure

```
skills/
  cuda-knowledge/references/     # ~1040 .md files (PTX, cuBLAS, Runtime, Driver, Math, NCCL)
    *-docs/                      # Scraped documentation organized by chapter/module
    *.md                         # Search guides with grep patterns per API
  cuda-samples/
    SKILL.md                     # Quick reference table, optimization mapping, arch compatibility
    references/                  # 8 topic files with detailed code snippets & GitHub permalinks
  cuda-optimizer/SKILL.md        # Orchestrator — drives the full optimization loop
  cuda-code-generator/SKILL.md   # Generates .cu files, must consult cuda-knowledge + cuda-samples
  ncu-rep-analyzer/SKILL.md      # NCU profiling + bottleneck classification + optimization suggestions
  kernel-benchmarker/SKILL.md    # Compile, validate, benchmark via benchmark.py
nvidia_doc_sync/                 # Documentation scraper and its README
```

### Key Design Decisions

- **Skills are independent but chainable** — each skill can be invoked standalone or as part of the optimizer loop.
- **Optimizer never stops mid-loop** — after each sub-skill returns, the orchestrator immediately proceeds to the next step. The output of one sub-skill is the input for the next.
- **New kernel versions get timestamped filenames** — `solution_opt_20260316_153045.cu`, never overwrite the original.
- **Knowledge-grounding is mandatory** — code-generator and ncu-rep-analyzer must consult both `cuda-knowledge/references/` (API signatures) and `cuda-samples/SKILL.md` (working code patterns) before generating code or recommendations involving complex APIs (cuBLASLt, Tensor Core, FP8 types).
