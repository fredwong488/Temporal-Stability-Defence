"""
tools/stratified_error_rates.py
--------------------------------
Stratified false-positive rate (FPR) and false-negative rate (FNR) for
radial-jitter (or any) defense evaluation, broken down by NuScenes scene type.

Two stratification axes are reported:

  1. Geographic location   — ``log.location`` (e.g. boston-seaport,
                             singapore-onenorth).  Clean categorical split.
  2. Scene description     — keyword-derived stratum from ``scene.description``
                             (intersection / parking / construction / other).

For each stratum:
  FPR = FP / (FP + TN)   on clean frames   (defense false-alarms)
  FNR = FN / (FN + TP)   on attacked frames (defense misses)

Frames with no defense_result are skipped (consistent with
eval_pipeline/metrics/common.py::compute_defense_metrics).

Run:
    pixi run python tools/stratified_error_rates.py                  # interactive picker
    pixi run python tools/stratified_error_rates.py results/dir ...  # explicit run dir(s)
    pixi run python tools/stratified_error_rates.py results/dir ... \\
        --version v1.0-mini --dataroot data/datasets/nuscenes-v1.0-mini
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

RESULTS_ROOT = Path(__file__).parent.parent / "results"

DEFAULT_NUSCENES_VERSION = "v1.0-mini"
DEFAULT_NUSCENES_DATAROOT = str(Path(__file__).parent.parent / "data/datasets/nuscenes-v1.0-mini")

# Keyword lookup for scene description → coarse stratum.
# First matching key wins; descriptions are checked lowercase.
DESCRIPTION_KEYWORDS: list[tuple[str, list[str]]] = [
    ("intersection", ["intersection", "crossroad", "cross road", "crossing", "junction"]),
    ("highway",      ["highway", "freeway", "motorway", "merge", "on-ramp", "off-ramp"]),
    ("parking",      ["parking lot", "parking", "parked"]),
    ("construction", ["construction", "roadwork", "road work", "work zone"]),
]


# ---------------------------------------------------------------------------
# NuScenes scene-metadata helpers
# ---------------------------------------------------------------------------

def build_location_strata(nusc) -> dict[str, str]:
    """Return {scene_token: location} for every scene in the NuScenes object."""
    return {
        scene["token"]: nusc.get("log", scene["log_token"])["location"]
        for scene in nusc.scene
    }


def classify_description(desc: str) -> str:
    """Map a free-text scene description to a coarse stratum label."""
    lower = desc.lower()
    for stratum, keywords in DESCRIPTION_KEYWORDS:
        if any(k in lower for k in keywords):
            return stratum
    return "other"


def build_description_strata(nusc) -> dict[str, str]:
    """Return {scene_token: description_stratum} for every scene."""
    return {
        scene["token"]: classify_description(scene["description"])
        for scene in nusc.scene
    }


def load_nuscenes(version: str, dataroot: str):
    try:
        from nuscenes.nuscenes import NuScenes
    except ImportError:
        sys.exit(
            "nuscenes-devkit is required. "
            "Add 'nuscenes-devkit' to pixi.toml and run 'pixi install'."
        )
    return NuScenes(version=version, dataroot=dataroot, verbose=False)


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def _empty_bucket() -> dict:
    return {"fp": 0, "tn": 0, "tp": 0, "fn": 0}


def compute_stratified_rates(
    jsonl_path: Path,
    scene_strata: dict[str, str],
) -> dict[str, dict]:
    """Single pass over a *_frames.jsonl file; returns per-stratum counts.

    Skips frames with no defense_result (same rule as compute_defense_metrics).
    Frames whose sequence_id is not in scene_strata are bucketed as 'unknown'.
    """
    buckets: dict[str, dict] = defaultdict(_empty_bucket)

    with jsonl_path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)

            defense = d.get("defense_result") or {}
            if defense.get("is_attack_detected") is None:
                # No defense result — skip
                continue

            is_attacked = bool(d.get("is_attacked"))
            detected = bool(defense["is_attack_detected"])
            stratum = scene_strata.get(d.get("sequence_id", ""), "unknown")

            b = buckets[stratum]
            if is_attacked:
                if detected:
                    b["tp"] += 1
                else:
                    b["fn"] += 1
            else:
                if detected:
                    b["fp"] += 1
                else:
                    b["tn"] += 1

    # Derive rates
    result: dict[str, dict] = {}
    for stratum, b in sorted(buckets.items()):
        n_clean = b["fp"] + b["tn"]
        n_attacked = b["tp"] + b["fn"]
        fpr = b["fp"] / n_clean if n_clean > 0 else float("nan")
        fnr = b["fn"] / n_attacked if n_attacked > 0 else float("nan")
        result[stratum] = {
            "n_clean": n_clean,
            "fp": b["fp"],
            "tn": b["tn"],
            "fpr": fpr,
            "n_attacked": n_attacked,
            "tp": b["tp"],
            "fn": b["fn"],
            "fnr": fnr,
        }
    return result


def accumulate(
    totals: dict[str, dict],
    counts: dict[str, dict],
) -> None:
    """Merge per-stratum raw counts (fp/tn/tp/fn) into a running total dict."""
    for stratum, b in counts.items():
        if stratum not in totals:
            totals[stratum] = {"fp": 0, "tn": 0, "tp": 0, "fn": 0}
        t = totals[stratum]
        for k in ("fp", "tn", "tp", "fn"):
            t[k] += b[k]


def finalise(totals: dict[str, dict]) -> dict[str, dict]:
    """Derive rates from accumulated raw counts."""
    result: dict[str, dict] = {}
    for stratum, b in sorted(totals.items()):
        n_clean = b["fp"] + b["tn"]
        n_attacked = b["tp"] + b["fn"]
        fpr = b["fp"] / n_clean if n_clean > 0 else float("nan")
        fnr = b["fn"] / n_attacked if n_attacked > 0 else float("nan")
        result[stratum] = {
            "n_clean": n_clean,
            "fp": b["fp"],
            "tn": b["tn"],
            "fpr": fpr,
            "n_attacked": n_attacked,
            "tp": b["tp"],
            "fn": b["fn"],
            "fnr": fnr,
        }
    return result


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def _fmt(v: float) -> str:
    return f"{v:.4f}" if v == v else "  n/a "  # nan check


def print_table(title: str, rates: dict[str, dict]) -> None:
    col_w = max(len(s) for s in rates) + 2
    header = (
        f"  {'stratum':<{col_w}}  "
        f"{'n_clean':>8}  {'fp':>6}  {'fpr':>7}    "
        f"{'n_atk':>7}  {'fn':>6}  {'fnr':>7}"
    )
    sep = "  " + "-" * (len(header) - 2)
    print(f"\n{title}")
    print(sep)
    print(header)
    print(sep)
    for stratum, r in rates.items():
        print(
            f"  {stratum:<{col_w}}"
            f"  {r['n_clean']:>8}  {r['fp']:>6}  {_fmt(r['fpr']):>7}"
            f"    {r['n_attacked']:>7}  {r['fn']:>6}  {_fmt(r['fnr']):>7}"
        )
    print(sep)


# ---------------------------------------------------------------------------
# CLI helpers (mirrors clustering_quality.py)
# ---------------------------------------------------------------------------

def list_run_dirs(results_root: Path) -> list[Path]:
    if not results_root.exists():
        return []
    return sorted(
        d for d in results_root.iterdir()
        if d.is_dir() and list(d.glob("*_frames.jsonl"))
    )


def pick_run_dir(results_root: Path) -> Path:
    dirs = list_run_dirs(results_root)
    if not dirs:
        sys.exit(f"No run directories with *_frames.jsonl found under '{results_root}'.")
    print(f"\nAvailable runs in '{results_root}':")
    for i, d in enumerate(dirs, 1):
        n = len(list(d.glob("*_frames.jsonl")))
        print(f"  [{i}] {d.name}  ({n} experiment{'s' if n != 1 else ''})")
    while True:
        raw = input(f"\nChoose a run [1-{len(dirs)}]: ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(dirs):
            return dirs[int(raw) - 1]
        print(f"  Please enter a number between 1 and {len(dirs)}.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stratified FPR / FNR by NuScenes scene type."
    )
    parser.add_argument(
        "run_dirs",
        nargs="*",
        metavar="DIR",
        help="Run directories containing *_frames.jsonl files. "
             "If omitted, an interactive picker is shown.",
    )
    parser.add_argument(
        "--version",
        default=DEFAULT_NUSCENES_VERSION,
        help=f"NuScenes version string (default: {DEFAULT_NUSCENES_VERSION})",
    )
    parser.add_argument(
        "--dataroot",
        default=DEFAULT_NUSCENES_DATAROOT,
        help=f"NuScenes dataroot path (default: {DEFAULT_NUSCENES_DATAROOT})",
    )
    args = parser.parse_args()

    if args.run_dirs:
        search_dirs = [Path(a) for a in args.run_dirs]
    else:
        run_dir = pick_run_dir(RESULTS_ROOT)
        print(f"\nSelected run: {run_dir}\n")
        search_dirs = [run_dir]

    files: list[Path] = []
    for d in search_dirs:
        files.extend(sorted(d.glob("*_frames.jsonl")))

    if not files:
        sys.exit("No *_frames.jsonl files found in the specified directories.")

    print(f"Loading NuScenes ({args.version}) from {args.dataroot} …")
    nusc = load_nuscenes(args.version, args.dataroot)

    location_strata = build_location_strata(nusc)
    description_strata = build_description_strata(nusc)

    loc_totals: dict[str, dict] = {}
    desc_totals: dict[str, dict] = {}

    for path in files:
        label = f"{path.parent.name}/{path.stem}"
        print(f"\n── {label} ──")

        loc_rates = compute_stratified_rates(path, location_strata)
        desc_rates = compute_stratified_rates(path, description_strata)

        print_table("By location", loc_rates)
        print_table("By scene description", desc_rates)

        # Accumulate raw counts for aggregate (not the derived rates)
        raw_loc = {s: {k: r[k] for k in ("fp", "tn", "tp", "fn")} for s, r in loc_rates.items()}
        raw_desc = {s: {k: r[k] for k in ("fp", "tn", "tp", "fn")} for s, r in desc_rates.items()}
        accumulate(loc_totals, raw_loc)
        accumulate(desc_totals, raw_desc)

    if len(files) > 1:
        print("\n\n══ AGGREGATE ══")
        print_table("By location (aggregate)", finalise(loc_totals))
        print_table("By scene description (aggregate)", finalise(desc_totals))


if __name__ == "__main__":
    main()
