# /// script
# requires-python = ">=3.9"
# ///
"""Verify all file paths referenced in cuda-samples skill exist in the submodule.

Also checks that code snippets in reference files approximately match the
actual source files (whitespace-insensitive prefix/suffix match).

Usage:
    uv run scripts/check_links.py              # check existence only
    uv run scripts/check_links.py --snippets   # also verify code snippets
"""

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SKILL_DIR = ROOT / "skills" / "cuda-samples"
SUBMODULE = ROOT / "cuda-samples"

# Extract repo-relative paths from reference file entries, e.g.:
# - **Path**: `cpp/0_Introduction/vectorAdd/vectorAdd.cu`
PATH_RE = re.compile(r"-\s+\*\*Path\*\*:\s*`(\S+)`")


def extract_paths(filepath: Path) -> list[str]:
    return PATH_RE.findall(filepath.read_text())


def check_existence(submodule: Path, sample_path: str) -> tuple[bool, str]:
    full = submodule / sample_path
    if full.exists():
        return True, "OK"
    return False, "MISSING"


def check_snippet(submodule: Path, sample_path: str, ref_file: Path, entry_heading: str) -> tuple[bool, str]:
    """Verify the code snippet for an entry approximately matches the source."""
    full = submodule / sample_path
    if not full.exists():
        return False, "FILE MISSING"

    source = full.read_text()

    # Find the entry in the reference file
    ref_text = ref_file.read_text()
    idx = ref_text.find(entry_heading)
    if idx == -1:
        return False, "heading not found in ref"

    # Find the code block after the heading
    block_idx = ref_text.find("```", idx)
    if block_idx == -1 or ref_text[block_idx + 3 : block_idx + 6] == "":
        return True, "no snippet"

    lang_end = ref_text.find("\n", block_idx)
    snippet_start = lang_end + 1
    snippet_end = ref_text.find("```", snippet_start)
    if snippet_end == -1:
        return True, "unclosed block"

    snippet = ref_text[snippet_start:snippet_end].strip()

    # Split snippet into significant lines (skip comments and blank lines)
    sig_lines = [
        l for l in snippet.splitlines()
        if l.strip() and not l.strip().startswith("//") and not l.strip().startswith("#")
    ]

    if not sig_lines:
        return True, "ok (comment-only snippet)"

    # Check first and last significant lines exist in source (whitespace-insensitive)
    first = "".join(sig_lines[0].split())
    last = "".join(sig_lines[-1].split())
    source_compact = "".join(source.split())

    ok = first in source_compact and last in source_compact
    if ok:
        return True, "OK"
    elif first in source_compact:
        return False, f"last line not found: {sig_lines[-1][:60]}..."
    elif last in source_compact:
        return False, f"first line not found: {sig_lines[0][:60]}..."
    else:
        return False, f"neither first nor last line found in source"


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify cuda-samples skill against submodule")
    parser.add_argument("--snippets", action="store_true", help="Also verify code snippets match source")
    args = parser.parse_args()

    if not SUBMODULE.exists():
        print("ERROR: cuda-samples submodule not found. Run:")
        print("  git submodule update --init")
        return 1

    all_paths: dict[str, list[str]] = {}
    for md_file in sorted(SKILL_DIR.rglob("*.md")):
        paths = extract_paths(md_file)
        if paths:
            all_paths[str(md_file.relative_to(ROOT))] = paths

    total = 0
    missing = 0
    snippet_errors = 0

    for rel_file, paths in sorted(all_paths.items()):
        print(f"\n{rel_file}")
        ref_path = ROOT / rel_file
        for sample_path in paths:
            total += 1
            exists, status = check_existence(SUBMODULE, sample_path)
            if not exists:
                missing += 1
            mark = "\033[92m✓\033[0m" if exists else "\033[91m✗\033[0m"
            msg = f"  {mark} {status:>10}  {sample_path}"

            if args.snippets and exists:
                # Find the heading that precedes this path entry
                ref_text = ref_path.read_text()
                path_pos = ref_text.find(f"`{sample_path}`")
                heading_match = re.search(
                    r"## \d+\.\d+ .*", ref_text[max(0, path_pos - 500) : path_pos][::-1]
                )
                if heading_match:
                    heading = heading_match.group(0)[::-1]
                    _, snippet_status = check_snippet(SUBMODULE, sample_path, ref_path, heading)
                    if not snippet_status.startswith("OK") and snippet_status != "no snippet":
                        snippet_errors += 1
                        msg += f"  [{snippet_status}]"

            print(msg)

    print(f"\n{'='*60}")
    print(f"Total: {total} paths, {total - missing} found, {missing} missing")
    if args.snippets:
        print(f"Snippet mismatches: {snippet_errors}")
    return 1 if missing > 0 or snippet_errors > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
