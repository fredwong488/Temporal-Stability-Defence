#!/usr/bin/env python3
"""
tools/filter_sort_frames.py
----------------------------
Interactive filter and sort tool for frame-results JSONL files.

Reads a JSONL file produced by eval_pipeline.runner (one JSON object per line),
lets the user interactively choose the run directory and file, then choose keys
to exclude and sort keys with priority and direction, then writes a new JSONL
file (same filename) into the specified output directory.

Usage:
    pixi run python tools/filter_sort_frames.py <results_dir> <output_dir>
"""

from __future__ import annotations

import functools
import json
import sys
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def flatten_scalar_keys(obj: Any, prefix: str = "") -> list[str]:
    """Return dot-notation paths to all scalar (non-list) leaf values."""
    keys: list[str] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            path = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                sub = flatten_scalar_keys(v, path)
                keys.extend(sub) if sub else None
            elif not isinstance(v, list):
                keys.append(path)
    return keys


def flatten_all_keys(obj: Any, prefix: str = "") -> list[str]:
    """Return dot-notation paths to all keys, including intermediate dict nodes."""
    keys: list[str] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            path = f"{prefix}.{k}" if prefix else k
            keys.append(path)
            if isinstance(v, dict):
                keys.extend(flatten_all_keys(v, path))
    return keys


def delete_nested_key(obj: dict, dot_path: str) -> dict:
    """Return a copy of obj with the key at dot_path removed."""
    parts = dot_path.split(".", 1)
    if len(parts) == 1:
        return {k: v for k, v in obj.items() if k != dot_path}
    top, rest = parts
    if top not in obj or not isinstance(obj[top], dict):
        return obj
    result = dict(obj)
    result[top] = delete_nested_key(result[top], rest)
    return result


def get_value(obj: Any, dot_path: str) -> Any:
    cur = obj
    for part in dot_path.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def make_comparator(key_path: str, asc: bool):
    """Return a cmp-style comparator that always sorts None last."""
    def compare(a: dict, b: dict) -> int:
        va = get_value(a, key_path)
        vb = get_value(b, key_path)
        if va is None and vb is None:
            return 0
        if va is None:
            return 1    # None goes after non-None
        if vb is None:
            return -1
        try:
            cmp = (va > vb) - (va < vb)
        except TypeError:
            cmp = (str(va) > str(vb)) - (str(va) < str(vb))
        return cmp if asc else -cmp
    return compare


# ---------------------------------------------------------------------------
# Interactive prompts
# ---------------------------------------------------------------------------

def _parse_selection(raw: str, n: int) -> list[int] | None:
    """Parse comma-separated 1-based indices. Returns None on invalid input."""
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    try:
        indices = [int(p) - 1 for p in parts]
    except ValueError:
        return None
    if any(idx < 0 or idx >= n for idx in indices):
        return None
    return indices


def prompt_multi_select(title: str, options: list[str]) -> list[int]:
    """Prompt the user to select zero or more items by number."""
    print(f"\n{title}")
    for i, opt in enumerate(options, 1):
        print(f"  {i:2d}. {opt}")
    while True:
        raw = input("  Selection (comma-separated numbers, or Enter for none): ").strip()
        if not raw:
            return []
        parsed = _parse_selection(raw, len(options))
        if parsed is not None:
            return parsed
        print(f"  Invalid — enter numbers 1–{len(options)}, comma-separated.")


def prompt_single_select(title: str, options: list[str]) -> int | None:
    """Prompt the user to select exactly one item, or nothing (Enter to stop)."""
    print(f"\n{title}")
    for i, opt in enumerate(options, 1):
        print(f"  {i:2d}. {opt}")
    while True:
        raw = input("  Selection (number, or Enter to stop): ").strip()
        if not raw:
            return None
        try:
            idx = int(raw) - 1
            if 0 <= idx < len(options):
                return idx
        except ValueError:
            pass
        print(f"  Invalid — enter a number 1–{len(options)}, or press Enter to stop.")


def prompt_direction(key: str) -> bool:
    """Return True for ascending, False for descending."""
    while True:
        raw = input(f"  Direction for '{key}' [asc/desc, default=asc]: ").strip().lower()
        if raw in ("", "asc"):
            return True
        if raw == "desc":
            return False
        print("  Enter 'asc', 'desc', or press Enter for asc.")


# ---------------------------------------------------------------------------
# File selection
# ---------------------------------------------------------------------------

def select_input_file(results_dir: Path) -> Path:
    """Interactively choose a run subdirectory then a JSONL file within it."""
    if not results_dir.exists():
        print(f"Error: results directory '{results_dir}' not found.")
        sys.exit(1)

    # Collect run subdirectories that contain at least one _frames.jsonl
    run_dirs = sorted(
        [d for d in results_dir.iterdir() if d.is_dir() and any(d.glob("*_frames.jsonl"))],
        reverse=True,
    )
    if not run_dirs:
        print(f"Error: no run directories with *_frames.jsonl found under '{results_dir}'.")
        sys.exit(1)

    run_idx = prompt_single_select(
        "Select a run directory:",
        [d.name for d in run_dirs],
    )
    if run_idx is None:
        print("No run selected. Exiting.")
        sys.exit(0)
    run_dir = run_dirs[run_idx]

    jsonl_files = sorted(run_dir.glob("*_frames.jsonl"))
    if not jsonl_files:
        print(f"Error: no *_frames.jsonl files found in '{run_dir}'.")
        sys.exit(1)

    file_idx = prompt_single_select(
        "Select a JSONL file:",
        [f.name for f in jsonl_files],
    )
    if file_idx is None:
        print("No file selected. Exiting.")
        sys.exit(0)

    return jsonl_files[file_idx]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if len(sys.argv) < 3:
        print(f"Usage: python {sys.argv[0]} <results_dir> <output_dir>")
        sys.exit(1)

    results_dir = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])

    # ---- File selection ----
    print("\n" + "=" * 60)
    print("SELECT INPUT FILE")
    print("=" * 60)

    input_path = select_input_file(results_dir)
    output_path = output_dir / input_path.name
    print(f"  → Input:  {input_path}")
    print(f"  → Output: {output_path}")

    # Load entries
    entries: list[dict] = []
    with open(input_path) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Warning: skipping malformed line {lineno}: {e}")

    if not entries:
        print("No entries found in input file.")
        sys.exit(1)

    print(f"\nLoaded {len(entries)} frame entries from {input_path}")

    first = entries[0]
    top_level_keys = list(first.keys())

    # Warn if entries have inconsistent keys (defensive check)
    key_sets = [frozenset(e.keys()) for e in entries]
    if len(set(key_sets)) > 1:
        print("Warning: not all entries share the same top-level keys. "
              "Using first entry as the key reference.")

    # ---- Step 1: Exclude keys ----
    print("\n" + "=" * 60)
    print("STEP 1: Exclude Keys")
    print("=" * 60)
    print("Choose keys to REMOVE from every entry in the output.")
    print("Selecting an intermediate key (e.g. 'metrics') removes the entire subtree.")

    all_keys = flatten_all_keys(first)
    excl_indices = prompt_multi_select("Available keys:", all_keys)
    exclude_keys: set[str] = {all_keys[i] for i in excl_indices}

    if exclude_keys:
        print(f"  → Excluding: {', '.join(sorted(exclude_keys))}")
    else:
        print("  → No keys excluded.")

    # ---- Step 2: Sort keys ----
    print("\n" + "=" * 60)
    print("STEP 2: Sort Keys")
    print("=" * 60)
    print("Choose scalar sort keys in priority order (highest priority first).")
    print("List and dict fields are omitted — only scalar (leaf) values can be sorted on.")

    sortable_keys = flatten_scalar_keys(first)
    # Remove keys that were excluded or live under an excluded subtree
    if exclude_keys:
        sortable_keys = [
            k for k in sortable_keys
            if not any(k == ek or k.startswith(ek + ".") for ek in exclude_keys)
        ]

    sort_spec: list[tuple[str, bool]] = []  # (dot_path, ascending)
    used: set[str] = set()

    while sortable_keys:
        remaining = [k for k in sortable_keys if k not in used]
        if not remaining:
            break
        idx = prompt_single_select(
            f"Sort key #{len(sort_spec) + 1} (Enter to finish):", remaining
        )
        if idx is None:
            break
        key = remaining[idx]
        asc = prompt_direction(key)
        sort_spec.append((key, asc))
        used.add(key)
        print(f"  → Added: {key} ({'asc' if asc else 'desc'})")

    # ---- Apply transformations ----

    result = list(entries)
    for key_path in exclude_keys:
        result = [delete_nested_key(entry, key_path) for entry in result]

    if sort_spec:
        # Stable multi-key sort: apply in reverse priority order
        for key, asc in reversed(sort_spec):
            result.sort(key=functools.cmp_to_key(make_comparator(key, asc)))

    # ---- Write output ----
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for entry in result:
            f.write(json.dumps(entry) + "\n")

    print(f"\nWrote {len(result)} entries to {output_path}")
    if sort_spec:
        spec_str = ", ".join(f"{k} ({'asc' if a else 'desc'})" for k, a in sort_spec)
        print(f"Sorted by: {spec_str}")
    if exclude_keys:
        print(f"Excluded keys: {', '.join(sorted(exclude_keys))}")


if __name__ == "__main__":
    main()
