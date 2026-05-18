#!/usr/bin/env python3
"""Validate cross-skill interface consistency.

Checks:
  1. SKILL.md frontmatter name matches parent directory
  2. Sub-skill names in cuda-optimizer match actual skill names
  3. Bottleneck type names consistent across optimizer, analyzer, generator
  4. All ../ relative paths in SKILL.md files resolve
  5. Every skill the optimizer references exists

Usage: python3 scripts/check_skills.py
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SKILLS_DIR = ROOT / "skills"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_frontmatter(path: Path) -> dict:
    """Extract name and description from YAML frontmatter."""
    text = path.read_text()
    m = re.search(r"^---\s*\n(.*?)\n---", text, re.DOTALL)
    if not m:
        return {}
    fm = {}
    for line in m.group(1).split("\n"):
        kv = re.match(r"(\w+):\s*(.*)", line)
        if kv:
            fm[kv.group(1)] = kv.group(2).strip().strip('"')
    return fm


def find_skill_dirs() -> list[Path]:
    return sorted(d for d in SKILLS_DIR.iterdir() if d.is_dir() and (d / "SKILL.md").exists())


def extract_bottleneck_types(text: str) -> set[str]:
    """Extract bottleneck type names like DRAM_MEMORY_BOUND from text."""
    return set(re.findall(r'\b(?:DRAM_MEMORY_BOUND|L1_PRESSURE_BOUND|LATENCY_BOUND'
                          r'|COMPUTE_BOUND|OCCUPANCY_BOUND|MIXED_BOUND)\b', text))


def extract_subskill_names(text: str) -> set[str]:
    """Extract skill names referenced by cuda-optimizer as sub-skills to call."""
    # Match skill names that look like 'cuda-code-generator', 'kernel-benchmarker', etc.
    # in the dependency table or calling specifications
    names = set()
    # Find in table cells: | `kernel-benchmarker` | or `cuda-code-generator`
    for m in re.finditer(r'`(cuda-code-generator|kernel-benchmarker|ncu-rep-analyzer|cuda-samples|cuda-knowledge|cuda-optimizer)`', text):
        names.add(m.group(1))
    return names


def extract_relative_paths(text: str, source_file: Path) -> list[str]:
    """Extract '../xxx' relative paths from markdown links."""
    # Match `../dir/file.md` or `../dir/` in markdown links/backticks
    paths = re.findall(r'`(\.\.[\w\-/]+\.?\w*)`', text)
    # Also catch paths mentioned in prose like `../cuda-knowledge/references/`
    paths += re.findall(r'(\.\./[\w\-]+/[\w\-/]+)', text)
    return [p.rstrip('/') for p in paths]


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def check_frontmatter_names(skill_dirs: list[Path]) -> list[str]:
    errors = []
    for d in skill_dirs:
        fm = parse_frontmatter(d / "SKILL.md")
        name = fm.get("name", "")
        if name != d.name:
            errors.append(f"{d.name}/SKILL.md: name '{name}' != dir '{d.name}'")
    return errors


def check_subskill_names(skill_dirs: list[Path]) -> list[str]:
    errors = []
    all_names = {d.name for d in skill_dirs}

    optim_path = SKILLS_DIR / "cuda-optimizer" / "SKILL.md"
    if not optim_path.exists():
        return ["cuda-optimizer/SKILL.md not found"]

    text = optim_path.read_text()
    # Extract the sub-skills the optimizer explicitly calls
    referenced = extract_subskill_names(text)

    # The optimizer orchestrates three action skills (knowledge bases are
    # referenced by the action skills themselves, not by the optimizer)
    expected = {"cuda-code-generator", "kernel-benchmarker", "ncu-rep-analyzer"}

    missing = expected - referenced
    unknown = referenced - all_names

    if missing:
        errors.append(f"cuda-optimizer missing references to: {missing}")
    if unknown:
        errors.append(f"cuda-optimizer references unknown skills: {unknown}")

    # Also check that referenced skills actually exist
    for ref in referenced:
        if ref != "cuda-optimizer" and ref not in all_names:
            errors.append(f"cuda-optimizer references '{ref}' which has no SKILL.md")

    return errors


def check_bottleneck_types() -> list[str]:
    errors = []
    files = {
        "optimizer": SKILLS_DIR / "cuda-optimizer" / "SKILL.md",
        "analyzer": SKILLS_DIR / "ncu-rep-analyzer" / "SKILL.md",
        "generator": SKILLS_DIR / "cuda-code-generator" / "SKILL.md",
    }

    types = {}
    for key, path in files.items():
        if not path.exists():
            errors.append(f"{key}: {path} not found")
            return errors
        types[key] = extract_bottleneck_types(path.read_text())

    # The analyzer defines the canonical set; others should be subsets
    canonical = types["analyzer"]
    for key in ("optimizer", "generator"):
        missing_in = canonical - types[key]
        extra_in = types[key] - canonical
        if missing_in:
            errors.append(
                f"bottleneck types in analyzer but missing from {key}: {missing_in}"
            )
        if extra_in:
            errors.append(
                f"bottleneck types in {key} but not in analyzer: {extra_in}"
            )

    return errors


def check_relative_paths() -> list[str]:
    errors = []
    for skill_dir in find_skill_dirs():
        skill_md = skill_dir / "SKILL.md"
        text = skill_md.read_text()
        paths = extract_relative_paths(text, skill_md)
        for p in paths:
            resolved = (skill_dir / p).resolve()
            # Try adding .md extension if the bare path doesn't exist
            if not resolved.exists() and not resolved.suffix:
                resolved_md = Path(str(resolved) + ".md")
                if resolved_md.exists():
                    continue
            if not resolved.exists():
                errors.append(
                    f"{skill_dir.name}/SKILL.md: '../{p}' → not found"
                )
    return errors


def check_cross_reference_consistency() -> list[str]:
    """Verify cuda-code-generator's mandatory requirement references are consistent."""
    errors = []
    gen_path = SKILLS_DIR / "cuda-code-generator" / "SKILL.md"
    if not gen_path.exists():
        return errors

    text = gen_path.read_text()

    # Check references to cuda-knowledge and cuda-samples
    if "../cuda-knowledge/references/" not in text:
        errors.append("cuda-code-generator missing reference to ../cuda-knowledge/references/")
    if "../cuda-samples/" not in text:
        errors.append("cuda-code-generator missing reference to ../cuda-samples/")

    # Check ncu-rep-analyzer references
    an_path = SKILLS_DIR / "ncu-rep-analyzer" / "SKILL.md"
    if an_path.exists():
        an_text = an_path.read_text()
        if "../cuda-knowledge/references/performance-traps.md" not in an_text:
            errors.append("ncu-rep-analyzer missing reference to performance-traps.md")
        if "../cuda-knowledge/references/ncu-guide.md" not in an_text:
            errors.append("ncu-rep-analyzer missing reference to ncu-guide.md")

    # Check kernel-benchmarker references
    kb_path = SKILLS_DIR / "kernel-benchmarker" / "SKILL.md"
    if kb_path.exists():
        kb_text = kb_path.read_text()
        if "benchmark.py" not in kb_text:
            errors.append("kernel-benchmarker missing reference to benchmark.py")

    return errors


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    skill_dirs = find_skill_dirs()
    if len(skill_dirs) < 6:
        print(f"WARNING: only {len(skill_dirs)} skills found (expected >= 6)")

    checks = [
        ("Frontmatter names", check_frontmatter_names(skill_dirs)),
        ("Sub-skill references", check_subskill_names(skill_dirs)),
        ("Bottleneck type consistency", check_bottleneck_types()),
        ("Cross-skill paths", check_relative_paths()),
        ("Cross-reference consistency", check_cross_reference_consistency()),
    ]

    total_errors = 0
    for label, errors in checks:
        print(f"\n=== {label} ===")
        if errors:
            for e in errors:
                print(f"  ✗ {e}")
            total_errors += len(errors)
        else:
            print("  ✓ All OK")

    print(f"\n{'='*60}")
    if total_errors == 0:
        print("All skill interfaces consistent ✓")
    else:
        print(f"{total_errors} issue(s) found ✗")
    return 1 if total_errors > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
