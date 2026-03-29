# scrape_cuda_docs.py Usage Guide

A unified NVIDIA CUDA documentation scraper that converts CUDA HTML docs into searchable markdown files. Supports PTX ISA, CUDA Runtime API, CUDA Driver API, CUDA Math API, cuBLAS, and NCCL.

## Prerequisites

[`uv`](https://github.com/astral-sh/uv) must be installed. All Python dependencies (`beautifulsoup4`, `html2text`, `requests`) are resolved automatically via PEP 723 inline metadata — no separate `pip install` or virtual environment setup needed.

---

## Quick Start

```bash
# Scrape PTX ISA documentation
uv run scrape_cuda_docs.py ptx

# Scrape CUDA Runtime API documentation
uv run scrape_cuda_docs.py runtime

# Scrape CUDA Driver API documentation
uv run scrape_cuda_docs.py driver

# Scrape CUDA Math API documentation
uv run scrape_cuda_docs.py math

# Scrape cuBLAS documentation
uv run scrape_cuda_docs.py cublas

# Scrape NCCL documentation
uv run scrape_cuda_docs.py nccl
```

---

## Command Reference

```bash
usage: scrape_cuda_docs.py [-h] [--output-dir OUTPUT_DIR] [--skip-download] [--force]
                           {ptx,runtime,driver,math,cublas,nccl}
```

### Positional Arguments

| Argument  | Description                                                                                       |
| --------- | ------------------------------------------------------------------------------------------------- |
| `ptx`     | Scrape the PTX ISA documentation (single monolithic page, split into ~405 files)                  |
| `runtime` | Scrape the CUDA Runtime API documentation (multi-page: modules + data structures)                 |
| `driver`  | Scrape the CUDA Driver API documentation (multi-page: modules + data structures)                  |
| `math`    | Scrape the CUDA Math API documentation (multi-page: modules + data structures, ~54 files)         |
| `cublas`  | Scrape the cuBLAS documentation (single monolithic Sphinx page, split into ~319 files by chapter) |
| `nccl`    | Scrape the NCCL documentation (Sphinx multi-page site, ~33 files organized by topic)              |

### Options

| Flag                | Default                            | Description                                                             |
| ------------------- | ---------------------------------- | ----------------------------------------------------------------------- |
| `--output-dir PATH` | `cuda_skill/references/<api>-docs` | Override the output directory                                           |
| `--skip-download`   | `false`                            | Skip network fetching; re-run only the cleanup pass on cached raw files |
| `--force`           | `false`                            | Force re-download all pages even if already cached                      |
| `-h, --help`        | —                                  | Show help message and exit                                              |

---

## Output Structure

### PTX ISA (`ptx`)

```text
cuda_skill/references/ptx-docs/
├── 1-introduction/
├── 2-programming-model/
├── ...
└── 9-instruction-set/      # ~186 files
```

Sections are split by heading hierarchy and organized into numbered chapter directories.

### cuBLAS (`cublas`)

```text
cuda_skill/references/cublas-docs/
├── 1-introduction/
├── 2-using-the-cublas-api/
├── 3-using-the-cublaslt-api/
├── 4-using-the-cublasxt-api/
└── ...                     # ~319 files total
```

Same split-by-heading approach as PTX ISA — the cuBLAS docs are also a single monolithic Sphinx HTML page.

### Runtime / Driver API (`runtime` / `driver`)

```text
cuda_skill/references/cuda-{runtime,driver}-docs/
├── modules/                # One file per API module group
├── data-structures/        # One file per struct/union
└── INDEX.md                # Generated index with links to all files
```

A sibling `cuda-{runtime,driver}-docs-raw/` cache directory stores the original downloaded files before cleanup.

### CUDA Math API (`math`)

```text
cuda_skill/references/cuda-math-docs/
├── modules/                # 14 files — single/double, half, bfloat16, FP8/FP6/FP4, SIMD, cast, integer
├── data-structures/        # 26 files — __half, __nv_bfloat16, __nv_fp8_*, __nv_fp6_*, __nv_fp4_*
└── INDEX.md
```

A sibling `cuda-math-docs-raw/` cache directory stores the original downloaded files before cleanup.

### NCCL (`nccl`)

```text
cuda_skill/references/nccl-docs/
├── usage/                  # 13 files — communicators, collectives, streams, P2P, CUDA graphs, etc.
├── api/                    # 12 files — colls, comms, p2p, types, ops, flags, device API
├── troubleshooting/        # 1 file  — RAS subsystem
├── overview.md             # NCCL concepts and topology
├── setup.md                # Installation
├── env.md                  # Full environment variable reference
├── examples.md             # Code examples
├── mpi.md                  # NCCL + MPI integration
├── troubleshooting.md      # Hang and error diagnosis
├── nccl1.md                # NCCL 1→2 migration guide
└── INDEX.md
```

Notes:

- No raw cache directory (no cleanup phase needed — Sphinx multi-page pages are already clean)
- Pages are fetched directly to the output directory (same as cuBLAS/PTX)
- `--skip-download` flag has no effect for this format

---

## Two-Phase Pipeline (Runtime / Driver)

The API scrapers use a download-then-clean pipeline:

**Phase 1 — Download** fetches each page and saves raw markdown to `*-raw/modules/` and `*-raw/data-structures/`.

**Phase 2 — Cleanup** reads from the cache and writes cleaned output, performing:

- Remove duplicate function TOC
- Strip footer (NVIDIA logo, copyright, privacy policy)
- Remove anchor-only links, zero-width spaces, `[inherited]` tags
- Remove `See also:` sections and boilerplate notes
- Strip all URLs from hyperlinks (keep link text only)
- Collapse excessive blank lines

This separation allows iterating on cleanup logic without re-downloading:

```bash
# Re-run only cleanup after editing clean_markdown_file()
uv run scrape_cuda_docs.py driver --skip-download
```

Typical size reduction after cleanup: **~76–83%**.

---

## Common Workflows

### Full fresh scrape of all documentation sets

```bash
uv run scrape_cuda_docs.py ptx
uv run scrape_cuda_docs.py runtime
uv run scrape_cuda_docs.py driver
uv run scrape_cuda_docs.py math
uv run scrape_cuda_docs.py cublas
uv run scrape_cuda_docs.py nccl
```

### Re-download everything from scratch (overwrite cache)

```bash
uv run scrape_cuda_docs.py driver --force
```

### Scrape to a custom output directory

```bash
uv run scrape_cuda_docs.py ptx --output-dir /path/to/my/output
```

### Iterate on cleanup logic without re-downloading

```bash
# Edit clean_markdown_file() in the script, then:
uv run scrape_cuda_docs.py driver --skip-download
```

---

## Adding a New Documentation Set

This section describes the complete end-to-end workflow for integrating a new NVIDIA documentation set into the skill library. All steps are derived from how cuBLAS and CUDA Math API were added.

### Step 1 — Identify the doc format

NVIDIA docs fall into three distinct HTML formats. Probe the target URL before writing any code:

```bash
# Download the index/landing page and check its size and structure
curl -s "https://docs.nvidia.com/cuda/<new-api>/index.html" | wc -c
# > 1MB single file  →  Sphinx single-page (like PTX ISA, cuBLAS)
# 50KB–200KB index   →  Sphinx multi-page (like NCCL) — headerlinks, toctree
# < 50KB index page  →  Doxygen multi-page (like Runtime, Driver, Math API)

# Check for Sphinx vs Doxygen markers in the HTML
curl -s "https://docs.nvidia.com/<path>/index.html" | grep -c "headerlink"   # Sphinx: >10
curl -s "https://docs.nvidia.com/<path>/modules.html" 2>/dev/null | wc -c   # Doxygen: found
```

| Format                 | Examples                  | HTML marker                                             | Scraper class            |
| ---------------------- | ------------------------- | ------------------------------------------------------- | ------------------------ |
| **Sphinx single-page** | PTX ISA, cuBLAS           | `div.body`, internal `<a class="headerlink">`, >1MB     | `SphinxScraper`          |
| **Sphinx multi-page**  | NCCL                      | `<a class="headerlink">`, toctree links, 50–200KB index | `SphinxMultiPageScraper` |
| **Doxygen multi-page** | Runtime, Driver, Math API | `modules.html` or `index.html` links to group pages     | `APIScraper`             |

### Step 2 — Implement the scraper

**For Sphinx single-page docs**, add an entry to `SphinxScraper.KNOWN_DOCS` in `scrape_cuda_docs.py`:

```python
class SphinxScraper(DocumentationScraper):
    KNOWN_DOCS: dict[str, tuple[str, str]] = {
        "cublas": (
            "cuBLAS",
            "https://docs.nvidia.com/cuda/cublas/index.html",
        ),
        "new-api": (                                     # ← new entry
            "New API Display Name",
            "https://docs.nvidia.com/cuda/new-api/index.html",
        ),
    }
```

**For Sphinx multi-page docs**, add an entry to `SphinxMultiPageScraper.KNOWN_DOCS`:

```python
class SphinxMultiPageScraper(DocumentationScraper):
    KNOWN_DOCS: dict[str, tuple[str, str, str]] = {
        "nccl": (
            "NCCL",
            "https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/",
            "index.html",
        ),
        "new-api": (                                     # ← new entry
            "New API Display Name",
            "https://docs.nvidia.com/<path>/",
            "index.html",                                # discovery page with toctree links
        ),
    }
```

The scraper discovers all linked `.html` pages from the toctree/index, then fetches and converts each one to markdown preserving the relative path structure (e.g. `usage/communicators.html` → `usage/communicators.md`).

**For Doxygen multi-page docs**, add an entry to `APIScraper._CONFIG`:

```python
_CONFIG: dict[str, tuple[str, str, str, str, str]] = {
    # (base_url, modules_discovery_path, modules_href_pattern,
    #  structs_discovery_path, structs_href_pattern)
    "math": (
        "https://docs.nvidia.com/cuda/cuda-math-api/",
        "index.html",
        r"cuda_math_api/group__CUDA__MATH__.*\.html",
        "cuda_math_api/structs.html",
        r"struct.*\.html",
    ),
    "nvml": (                                           # ← new entry
        "https://docs.nvidia.com/deploy/nvml-api/",
        "modules.html",
        r"group__nvml.*\.html",
        "annotated.html",
        r"struct.*\.html",
    ),
}
```

> **Discovery path tip:** If module hrefs use a subdirectory prefix (e.g. `cuda_math_api/group__...`), the discovery URL and href pattern must include that prefix. The scraper uses `Path(href).name` for filenames and resolves URLs against the discovery page URL — not the base URL.

### Step 3 — Register the CLI command

In `main()`, add the new name to two places:

```python
# 1. Add to choices
parser.add_argument(
    "api_type",
    choices=["ptx", "runtime", "driver", "math", "cublas", "nccl"],  # ← add here
)

# 2. Add default output directory
default_dirs = {
    ...
    "nccl": "cuda_skill/references/nccl-docs",  # ← add here
}

# 3. For Sphinx docs, add dispatch in the scraper selection block
elif args.api_type in SphinxScraper.KNOWN_DOCS:
    scraper = SphinxScraper.from_doc_type(args.api_type, args.output_dir)
```

### Step 4 — Download the documentation

```bash
uv run scrape_cuda_docs.py <new-api>
```

Verify the output:

```bash
# Check file count and directory structure
find cuda_skill/references/<new-api>-docs -name "*.md" | wc -l
ls cuda_skill/references/<new-api>-docs/

# Spot-check a file for correct title and content
head -10 cuda_skill/references/<new-api>-docs/modules/<first-file>.md
```

If titles look wrong (e.g. a function name like `log()` instead of a chapter title), or files are missing, re-run with `--force` after fixing the discovery patterns.

### Step 5 — Create a search guide

Create `cuda_skill/references/<new-api>.md` following the same structure as `cublas.md` or `cuda-math.md`:

```text
# <New API> Reference

**Related guides:** ...

## Table of Contents
## Local Documentation       ← file count, size, quality markers
## When to Use               ← numbered list of use cases
## Quick Search Examples     ← grep commands for the most common queries
## Documentation Structure   ← annotated directory tree
## Search Tips               ← naming patterns for this API
## Common Workflows          ← step-by-step grep workflows
## Troubleshooting           ← common errors and fixes
## Version Information
```

### Step 6 — Update SKILL.md

Two things must be updated in `cuda_skill/SKILL.md`:

**1. Add an entry to the "Local API Documentation" section:**

```markdown
**<New API>** — `references/<new-api>-docs/` (N files, X.XMB)

- Search guide: `references/<new-api>.md`
- Use for: ...
```

**2. Add trigger keywords to the `description` frontmatter** so the skill activates on relevant queries:

```yaml
description: "... Triggers on ..., <new-api>, <key-function-names>, ..."
```

Without updating the trigger keywords, Claude Code will not activate this skill when a user asks about the new library.

---

## Source URLs

| API              | URL                                                                 |
| ---------------- | ------------------------------------------------------------------- |
| PTX ISA          | `https://docs.nvidia.com/cuda/parallel-thread-execution/index.html` |
| CUDA Runtime API | `https://docs.nvidia.com/cuda/cuda-runtime-api/`                    |
| CUDA Driver API  | `https://docs.nvidia.com/cuda/cuda-driver-api/`                     |
| CUDA Math API    | `https://docs.nvidia.com/cuda/cuda-math-api/`                       |
| cuBLAS           | `https://docs.nvidia.com/cuda/cublas/index.html`                    |
| NCCL             | `https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/`        |
