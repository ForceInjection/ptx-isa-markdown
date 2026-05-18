#!/usr/bin/env python3
"""Verify documented file counts match actual filesystem.

File sizes are documented as `du -sh` (disk usage), which includes filesystem
block overhead. This script compares counts exactly and sizes within tolerance.

Usage: python3 scripts/check_counts.py
"""

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REFS_DIR = ROOT / "skills" / "cuda-knowledge" / "references"

# (relative_file, regex_pattern, count_group, size_group, unit_group)
SEARCH_GUIDE_TOC = [
    ("skills/cuda-knowledge/references/cuda-runtime.md",
     r"Local Documentation.*?(\d+)\s+markdown files,\s+([\d.]+)\s*(MB|KB)"),
    ("skills/cuda-knowledge/references/cuda-driver.md",
     r"Local Documentation.*?(\d+)\s+markdown files,\s+([\d.]+)\s*(MB|KB)"),
    ("skills/cuda-knowledge/references/cuda-math.md",
     r"Local Documentation.*?(\d+)\s+markdown files,\s+([\d.]+)\s*(KB)"),
    ("skills/cuda-knowledge/references/nccl.md",
     r"Local Documentation.*?(\d+)\s+markdown files,\s+([\d.]+)\s*(KB)"),
    ("skills/cuda-knowledge/references/ptx-isa.md",
     r"Local Documentation.*?(\d+)\s+markdown files,\s+([\d.]+)\s*(MB|KB)"),
    ("skills/cuda-knowledge/references/cublas.md",
     r"Local Documentation.*?(\d+)\s+markdown files,\s+([\d.]+)\s*(MB|KB)"),
]

GUIDE_DIR = {
    "cuda-runtime.md": "cuda-runtime-docs", "cuda-driver.md": "cuda-driver-docs",
    "cuda-math.md": "cuda-math-docs", "nccl.md": "nccl-docs",
    "ptx-isa.md": "ptx-docs", "cublas.md": "cublas-docs",
}

SKILL_MD_CHECKS = [
    ("PTX ISA", "ptx-docs", 405, "2.3MB"), ("CUDA Runtime API", "cuda-runtime-docs", 104, "1.2MB"),
    ("CUDA Driver API", "cuda-driver-docs", 129, "1.2MB"), ("cuBLAS", "cublas-docs", 319, "2.9MB"),
    ("CUDA Math API", "cuda-math-docs", 41, "528K"), ("NCCL", "nccl-docs", 34, "516K"),
]


def count_md_files(d: Path) -> int:
    return len(list(d.rglob("*.md")))


def du_size_str(d: Path) -> str:
    """Return 'du -sh' style size string, e.g. '2.3M' or '528K'."""
    result = subprocess.run(["du", "-sh", str(d)], capture_output=True, text=True)
    return result.stdout.split()[0]  # e.g. "2.3M" or "528K"


def parse_size(s: str) -> float:
    """Parse a size string like '2.3M', '2.3MB', '528K', '528KB' to kilobytes."""
    s = s.strip().upper()
    if s.endswith("MB"):
        return float(s[:-2]) * 1024
    elif s.endswith("KB"):
        return float(s[:-2])
    elif s.endswith("M"):
        return float(s[:-1]) * 1024
    elif s.endswith("K"):
        return float(s[:-1])
    else:
        return float(s)


def sizes_match(doc: str, actual: str, tolerance_kb: float = 100) -> bool:
    """Check if documented and actual sizes match within tolerance."""
    return abs(parse_size(doc) - parse_size(actual)) <= tolerance_kb


def main() -> int:
    errors = 0

    # Check search guide TOC entries
    print("=== Search Guide TOC Counts ===")
    for path, pattern in SEARCH_GUIDE_TOC:
        text = (ROOT / path).read_text()
        m = re.search(pattern, text)
        if not m:
            print(f"  ✗ {path}: pattern not found")
            errors += 1
            continue

        doc_files = int(m.group(1))
        guide_name = Path(path).name
        dir_name = GUIDE_DIR[guide_name]
        actual_files = count_md_files(REFS_DIR / dir_name)

        if doc_files != actual_files:
            print(f"  ✗ {path}: {doc_files} documented, {actual_files} actual files")
            errors += 1
        else:
            print(f"  ✓ {path}: {doc_files} files")

    # Check SKILL.md entries
    print("\n=== cuda-knowledge/SKILL.md Counts ===")
    for api_name, dir_name, expected_files, expected_size in SKILL_MD_CHECKS:
        actual_files = count_md_files(REFS_DIR / dir_name)
        actual_size = du_size_str(REFS_DIR / dir_name)

        if actual_files != expected_files:
            print(f"  ✗ {api_name}: {expected_files} documented, {actual_files} actual files")
            errors += 1
        elif not sizes_match(expected_size, actual_size):
            print(f"  ✗ {api_name}: {expected_size} documented, {actual_size} actual size")
            errors += 1
        else:
            print(f"  ✓ {api_name}: {expected_files} files, {expected_size}")

    # Summary
    print(f"\n{'='*60}")
    if errors == 0:
        print("All counts match ✓")
    else:
        print(f"{errors} mismatch(es) found ✗")
    return 1 if errors > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
