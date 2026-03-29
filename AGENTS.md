# AGENTS.md

This file provides guidance to Qoder (qoder.com) when working with code in this repository.

## Project Overview

This repository converts NVIDIA CUDA documentation (PTX ISA 9.1, CUDA Runtime API 13.1, CUDA Driver API 13.1) from HTML into searchable markdown files. It also packages the output as a Claude Code skill for GPU development assistance.

The repo has two functional parts:

1. **Scraper** (`scrape_cuda_docs.py`) — A Python script that fetches, converts, and cleans NVIDIA's HTML documentation into organized markdown
2. **Skill** (`cuda_skill/`) — The output artifact: a portable skill directory with `SKILL.md` and `references/` containing ~640 markdown files across three documentation sets

## Commands

### Running the scraper

The scraper is a single `uv` script with inline dependency metadata. No separate install step is needed — `uv` resolves dependencies automatically.

```bash
# Scrape PTX ISA documentation
uv run scrape_cuda_docs.py ptx

# Scrape CUDA Runtime API documentation
uv run scrape_cuda_docs.py runtime

# Scrape CUDA Driver API documentation
uv run scrape_cuda_docs.py driver

# Re-run cleanup without re-downloading (uses cached raw files)
uv run scrape_cuda_docs.py driver --skip-download

# Force re-download even if cache exists
uv run scrape_cuda_docs.py driver --force

# Custom output directory
uv run scrape_cuda_docs.py ptx --output-dir /path/to/output
```

### Linting / Testing

There are no lint, test, or build commands configured for this repository.

## Architecture

### Scraper (`scrape_cuda_docs.py`)

A single-file scraper using class inheritance to handle two distinct documentation formats:

- **`DocumentationScraper`** — Base class with shared HTTP fetching, HTML-to-markdown conversion (`html2text`), navigation removal, and filename sanitization
- **`PTXScraper(DocumentationScraper)`** — Handles the PTX ISA, which is a single monolithic HTML page. Splits it by heading hierarchy into ~405 individual markdown files organized by chapter directories
- **`APIScraper(DocumentationScraper)`** — Handles Runtime and Driver APIs, which are multi-page HTML docs. Discovers pages by crawling `modules.html` and `annotated.html`, downloads each page to a cache directory (`*-raw/`), then runs a cleanup pass that strips duplicate TOCs, footers, boilerplate, and redundant URLs (producing 76-83% size reduction). The two-phase approach (download + clean) allows iterating on cleanup logic without re-downloading via `--skip-download`

Dependencies: `beautifulsoup4`, `html2text`, `requests` (declared inline via PEP 723 script metadata).

### Skill output (`cuda_skill/`)

```
cuda_skill/
├── SKILL.md                    # Skill definition (~13KB, always loaded by Claude Code)
└── references/
    ├── ptx-docs/               # 405 files organized by chapter (e.g., 9-instruction-set/)
    ├── cuda-runtime-docs/      # modules/ (41 files) + data-structures/ (66 files)
    ├── cuda-driver-docs/       # modules/ (50 files) + data-structures/ (80 files)
    ├── ptx-isa.md              # Search guide with grep examples
    ├── cuda-runtime.md         # Search guide with grep examples
    ├── cuda-driver.md          # Search guide with grep examples
    ├── nsys-guide.md           # Nsight Systems profiling patterns
    ├── ncu-guide.md            # Nsight Compute metrics interpretation
    ├── debugging-tools.md      # compute-sanitizer, cuda-gdb, cuobjdump
    ├── nvtx-patterns.md        # NVTX instrumentation patterns
    └── performance-traps.md    # Bank conflicts, coalescing, scale traps
```

The `*-docs/` directories contain raw reference content. The top-level `.md` guides in `references/` are hand-written search guides and workflow references. Each `*-docs/` directory has an `INDEX.md` listing all files.

### Key design decisions

- **Scraper is a single file** — All three API types (PTX, Runtime, Driver) are handled by one script with a subcommand interface, not separate scripts (the README references separate scripts for backward compatibility but the code has been unified)
- **Cache-then-clean pipeline** — API docs are first downloaded raw to `*-raw/` cache dirs, then cleaned to final output. This separates network-dependent work from text processing
- **Inline uv dependencies** — No `requirements.txt` or `pyproject.toml`; dependencies are declared in the script's PEP 723 header so `uv run` resolves everything
