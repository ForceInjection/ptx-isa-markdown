# AGENTS.md

This file provides guidance to Qoder (qoder.com) when working with code in this repository.

## Project Overview

This repository provides an automated, knowledge-augmented CUDA kernel optimization pipeline for Qoder (qoder.com). It combines deep local knowledge (via NVIDIA documentation) with an AI Agent-driven optimization loop.

The repo has two functional parts:

1. **Scraper** (`nvidia_doc_sync/scrape_cuda_docs.py`) — A Python script that fetches, converts, and cleans NVIDIA's HTML documentation into organized markdown.
2. **Skills Monorepo** (`skills/`) — A collection of specialized Agent skills that work together to automatically profile, analyze, and optimize CUDA code, grounded in the scraped documentation.

## Commands

### Running the scraper

The scraper is a single `uv` script with inline dependency metadata. No separate install step is needed — `uv` resolves dependencies automatically.

```bash
# Scrape PTX ISA documentation
uv run nvidia_doc_sync/scrape_cuda_docs.py ptx

# Scrape CUDA Runtime API documentation
uv run nvidia_doc_sync/scrape_cuda_docs.py runtime

# Scrape CUDA Driver API documentation
uv run nvidia_doc_sync/scrape_cuda_docs.py driver

# Re-run cleanup without re-downloading (uses cached raw files)
uv run nvidia_doc_sync/scrape_cuda_docs.py driver --skip-download

# Force re-download even if cache exists
uv run nvidia_doc_sync/scrape_cuda_docs.py driver --force

# Custom output directory
uv run nvidia_doc_sync/scrape_cuda_docs.py ptx --output-dir /path/to/output
```

### Linting / Testing

There are no lint, test, or build commands configured for this repository.

## Architecture

### Scraper (`nvidia_doc_sync/scrape_cuda_docs.py`)

A single-file scraper using class inheritance to handle two distinct documentation formats:

- **`DocumentationScraper`** — Base class with shared HTTP fetching, HTML-to-markdown conversion (`html2text`), navigation removal, and filename sanitization
- **`PTXScraper(DocumentationScraper)`** — Handles the PTX ISA, which is a single monolithic HTML page. Splits it by heading hierarchy into ~405 individual markdown files organized by chapter directories
- **`APIScraper(DocumentationScraper)`** — Handles Runtime and Driver APIs, which are multi-page HTML docs. Discovers pages by crawling `modules.html` and `annotated.html`, downloads each page to a cache directory (`*-raw/`), then runs a cleanup pass that strips duplicate TOCs, footers, boilerplate, and redundant URLs (producing 76-83% size reduction). The two-phase approach (download + clean) allows iterating on cleanup logic without re-downloading via `--skip-download`

Dependencies: `beautifulsoup4`, `html2text`, `requests` (declared inline via PEP 723 script metadata).

### Skills Monorepo (`skills/`)

The skills are organized as a multi-skill monorepo to separate concerns and allow the Agent to orchestrate complex optimization workflows:

```text
skills/
├── cuda-knowledge/             # Knowledge base skill (formerly cuda_skill)
│   ├── SKILL.md                # Defines how to search the documentation
│   └── references/             # 640+ markdown files (PTX, cuBLAS, Math API, etc.)
├── cuda-optimizer/             # The main orchestrator skill
│   └── SKILL.md                # Drives the profile-analyze-optimize loop
├── cuda-code-generator/        # Code generation and modification skill
│   ├── SKILL.md                # Instructed to search cuda-knowledge for API accuracy
│   └── references/
│       └── cuda-optimization-strategies.md
├── ncu-rep-analyzer/           # NCU profiling and bottleneck analysis skill
│   └── SKILL.md                # Instructed to use performance-traps.md for diagnosis
└── kernel-benchmarker/         # Compilation, validation, and benchmarking skill
    ├── SKILL.md                # Instructed to use debugging-tools.md on failure
    └── scripts/benchmark.py
```

The action skills (`cuda-code-generator`, `ncu-rep-analyzer`, `kernel-benchmarker`) are strictly instructed to ground their operations in the documentation provided by `cuda-knowledge`, reducing hallucination and improving the depth of optimization.

### Key design decisions

- **Scraper is a single file** — All three API types (PTX, Runtime, Driver) are handled by one script with a subcommand interface, not separate scripts (the README references separate scripts for backward compatibility but the code has been unified)
- **Cache-then-clean pipeline** — API docs are first downloaded raw to `*-raw/` cache dirs, then cleaned to final output. This separates network-dependent work from text processing
- **Inline uv dependencies** — No `requirements.txt` or `pyproject.toml`; dependencies are declared in the script's PEP 723 header so `uv run` resolves everything
