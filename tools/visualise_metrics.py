"""
tools/visualise_metrics.py
--------------------------
Interactive CLI to visualise ORA sweep metric files produced by either
scripts/run_ora_sweep.py or scripts/generate_metrics.py.

Supported metric files
----------------------
  ora_ap_sweep.csv          → AP vs budget line plots (per class / difficulty)
  ora_pr_curves.json        → Precision–Recall curves, overlaid across budgets
  ora_recall_iou_curves.json → Recall vs IoU threshold, overlaid across budgets

Usage
-----
    python tools/visualise_metrics.py
    python tools/visualise_metrics.py --results-dir /path/to/results
    python tools/visualise_metrics.py --run 2026-04-09-18-15-47   # skip dir picker
    python tools/visualise_metrics.py --no-interactive            # save PNGs instead of showing

Notes on format differences between run_ora_sweep.py and generate_metrics.py
-----------------------------------------------------------------------------
* AP CSV   : generate_metrics includes ALL classes found in attacked_map (including
             all-zero ones like DontCare/Misc). run_ora_sweep only includes
             args.classes (default: Car, Pedestrian, Cyclist).
* PR JSON  : generate_metrics saves the full pr_curves dict (all classes).
             run_ora_sweep filters to args.classes before saving.
* Recall-IoU JSON: same filtering difference as PR. Additionally, run_ora_sweep
             always writes confidence_threshold per entry; generate_metrics only
             writes it when present in the config dict.
This script handles both variants transparently.
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import re
import sys
from typing import Any

# ---------------------------------------------------------------------------
# Optional matplotlib import – fail gracefully with a helpful message
# ---------------------------------------------------------------------------
try:
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import numpy as np
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


DEFAULT_RESULTS_DIR = "results"

# Classes to show by default when a file contains many (e.g. all-zero extras)
PREFERRED_CLASSES = {"Car", "Pedestrian", "Cyclist"}

# Colour cycle for budgets
BUDGET_CMAP = "plasma"


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def list_run_dirs(results_dir: pathlib.Path) -> list[pathlib.Path]:
    metric_globs = ["ora_ap_sweep.csv", "ora_pr_curves.json", "ora_recall_iou_curves.json"]
    dirs = sorted(
        d for d in results_dir.iterdir()
        if d.is_dir() and any(list(d.glob(g)) for g in metric_globs)
    )
    return dirs


def load_run_metadata(run_dir: pathlib.Path) -> dict:
    """Read num_frames, attack_type, detector_type from the lowest-budget JSON file."""
    budget_files = sorted(run_dir.glob("ora_budget_*.json"),
                          key=lambda p: int(re.search(r"(\d+)", p.stem).group(1)))
    if not budget_files:
        return {}
    try:
        with open(budget_files[0]) as f:
            data = json.load(f)
        cfg = data.get("config", {})
        return {
            "num_frames":    data.get("num_frames"),
            "attack_type":   cfg.get("attack_type"),
            "detector_type": cfg.get("detector_type"),
        }
    except Exception:
        return {}


def pick_run_dir(results_dir: pathlib.Path, run_name: str | None) -> pathlib.Path:
    dirs = list_run_dirs(results_dir)
    if not dirs:
        sys.exit(f"No metric directories found under '{results_dir}'.")

    if run_name:
        matches = [d for d in dirs if d.name == run_name]
        if not matches:
            sys.exit(f"Run '{run_name}' not found. Available: {[d.name for d in dirs]}")
        return matches[0]

    print(f"\nAvailable run directories in '{results_dir}':")
    for i, d in enumerate(dirs, 1):
        tags = []
        if (d / "ora_ap_sweep.csv").exists():
            tags.append("ap")
        if (d / "ora_pr_curves.json").exists():
            tags.append("pr")
        if (d / "ora_recall_iou_curves.json").exists():
            tags.append("recall_iou")
        meta = load_run_metadata(d)
        meta_parts = []
        if meta.get("detector_type"):
            meta_parts.append(meta["detector_type"])
        if meta.get("attack_type"):
            meta_parts.append(meta["attack_type"])
        if meta.get("num_frames") is not None:
            meta_parts.append(f"{meta['num_frames']} frames")
        meta_str = f"  [{', '.join(meta_parts)}]" if meta_parts else ""
        print(f"  [{i}] {d.name}  ({', '.join(tags)}){meta_str}")

    while True:
        raw = input(f"\nChoose a directory [1-{len(dirs)}]: ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(dirs):
            return dirs[int(raw) - 1]
        print(f"  Please enter a number between 1 and {len(dirs)}.")


def pick_metric(run_dir: pathlib.Path) -> tuple[str, pathlib.Path]:
    options: list[tuple[str, pathlib.Path]] = []
    for fname, label in [
        ("ora_ap_sweep.csv",           "AP sweep  (AP vs budget, per class/difficulty)"),
        ("ora_pr_curves.json",         "PR curves (Precision–Recall, overlaid across budgets)"),
        ("ora_recall_iou_curves.json", "Recall-IoU curves (Recall vs IoU threshold)"),
    ]:
        p = run_dir / fname
        if p.exists():
            options.append((label, p))

    if not options:
        sys.exit("No recognised metric files found in the selected run directory.")

    print("\nAvailable metric files:")
    for i, (label, path) in enumerate(options, 1):
        print(f"  [{i}] {label}")
        print(f"       {path.name}")

    while True:
        raw = input(f"\nChoose a metric [1-{len(options)}]: ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            label, path = options[int(raw) - 1]
            return label, path
        print(f"  Please enter a number between 1 and {len(options)}.")


def budget_colours(budgets: list[int]) -> dict[int, Any]:
    cmap = cm.get_cmap(BUDGET_CMAP, max(len(budgets), 2))
    return {b: cmap(i / max(len(budgets) - 1, 1)) for i, b in enumerate(budgets)}


def save_or_show(fig: "plt.Figure", out_path: pathlib.Path | None) -> None:
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {out_path}")
    else:
        plt.tight_layout()
        plt.show()


# ---------------------------------------------------------------------------
# AP sweep (CSV)
# ---------------------------------------------------------------------------

def load_ap_csv(path: pathlib.Path) -> tuple[list[int], dict[str, dict[str, list[float]]]]:
    """Returns (budgets, {class_name: {difficulty: [ap_values]}})."""
    budgets: list[int] = []
    # {cls: {diff: [ap, ...]}}
    data: dict[str, dict[str, list[float]]] = {}

    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            budgets.append(int(row["budget"]))
            for key, val in row.items():
                if key == "budget":
                    continue
                # key format: <class>_ap_<difficulty>
                m = re.match(r"^(.+)_ap_(easy|moderate|hard)$", key, re.IGNORECASE)
                if not m:
                    continue
                cls_raw, diff_raw = m.group(1).title(), m.group(2).title()
                data.setdefault(cls_raw, {}).setdefault(diff_raw, []).append(float(val))

    return budgets, data


def print_ap_table(
    budgets: list[int],
    data: dict[str, dict[str, list[float]]],
    display_classes: list[str],
    difficulties: list[str],
) -> None:
    """Print AP values as a plain-text table: rows = budgets, columns = class × difficulty."""
    col_width = 10
    bud_width = 8

    # Header row: class names spanning difficulty columns
    cls_header = " " * bud_width
    for cls in display_classes:
        span = col_width * len(difficulties)
        cls_header += cls.center(span)
    print(cls_header)

    # Sub-header: difficulty names
    diff_header = "budget".ljust(bud_width)
    for _ in display_classes:
        for diff in difficulties:
            diff_header += diff[:col_width].center(col_width)
    print(diff_header)

    # Separator
    total_width = bud_width + col_width * len(display_classes) * len(difficulties)
    print("-" * total_width)

    # Data rows
    for i, budget in enumerate(budgets):
        row = str(budget).ljust(bud_width)
        for cls in display_classes:
            for diff in difficulties:
                val = data.get(cls, {}).get(diff, [None] * (i + 1))[i]
                cell = f"{val:.4f}" if val is not None else "  n/a  "
                row += cell.center(col_width)
        print(row)


def visualise_ap(path: pathlib.Path, interactive: bool, run_dir: pathlib.Path, fmt: str = "plot") -> None:
    budgets, data = load_ap_csv(path)

    # Filter to classes that have at least one non-zero AP value
    active_classes = sorted(
        cls for cls, diffs in data.items()
        if any(any(v > 0 for v in vals) for vals in diffs.values())
    )
    if not active_classes:
        active_classes = sorted(data.keys())

    # Prefer the canonical three if present
    display_classes = [c for c in active_classes if c in PREFERRED_CLASSES] or active_classes

    difficulties = sorted({d for cls in data for d in data[cls]},
                          key=lambda x: ["Easy", "Moderate", "Hard"].index(x)
                          if x in ["Easy", "Moderate", "Hard"] else 99)

    if fmt == "table":
        print_ap_table(budgets, data, display_classes, difficulties)
        return

    n_cls = len(display_classes)
    fig, axes = plt.subplots(1, n_cls, figsize=(5 * n_cls, 4), sharey=True, squeeze=False)
    axes = axes[0]

    line_styles = ["-", "--", ":"]
    colours = {"Easy": "#2196F3", "Moderate": "#FF9800", "Hard": "#F44336"}

    for ax, cls in zip(axes, display_classes):
        for i, diff in enumerate(difficulties):
            vals = data.get(cls, {}).get(diff, [])
            if not vals:
                continue
            ax.plot(
                budgets, vals,
                marker="o", markersize=4,
                linestyle=line_styles[i % len(line_styles)],
                color=colours.get(diff, f"C{i}"),
                label=diff,
            )
        ax.set_title(cls)
        ax.set_xlabel("ORA Budget (points removed)")
        ax.set_ylabel("Average Precision")
        ax.set_ylim(0, 1.05)
        ax.legend(title="Difficulty")
        ax.grid(True, alpha=0.3)

    fig.suptitle("AP vs ORA Budget", fontsize=13, fontweight="bold")

    out = (run_dir / "plot_ap_sweep.png") if not interactive else None
    save_or_show(fig, out)


# ---------------------------------------------------------------------------
# PR curves (JSON)
# ---------------------------------------------------------------------------

def load_pr_json(path: pathlib.Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def available_classes_pr(entries: list[dict]) -> list[str]:
    classes: set[str] = set()
    for entry in entries:
        classes.update(entry.get("curves", {}).keys())
    return sorted(classes)


def available_difficulties_pr(entries: list[dict], cls: str) -> list[str]:
    diffs: set[str] = set()
    for entry in entries:
        diffs.update(entry.get("curves", {}).get(cls, {}).keys())
    return sorted(diffs, key=lambda x: ["Easy", "Moderate", "Hard"].index(x)
                  if x in ["Easy", "Moderate", "Hard"] else 99)


def pick_from_list(label: str, options: list[str]) -> str:
    if len(options) == 1:
        print(f"  (Auto-selected {label}: {options[0]})")
        return options[0]
    print(f"\n{label}:")
    for i, o in enumerate(options, 1):
        print(f"  [{i}] {o}")
    while True:
        raw = input(f"Choose [1-{len(options)}]: ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return options[int(raw) - 1]
        print(f"  Please enter a number between 1 and {len(options)}.")


def pick_multiple(label: str, options: list[str], defaults: list[str]) -> list[str]:
    preferred = [o for o in options if o in defaults] or options
    print(f"\n{label} (space-separated numbers, or Enter for [{', '.join(preferred)}]):")
    for i, o in enumerate(options, 1):
        print(f"  [{i}] {o}")
    raw = input("Choose: ").strip()
    if not raw:
        return preferred
    chosen = []
    for tok in raw.split():
        if tok.isdigit() and 1 <= int(tok) <= len(options):
            chosen.append(options[int(tok) - 1])
    return chosen or preferred


def visualise_pr(path: pathlib.Path, interactive: bool, run_dir: pathlib.Path) -> None:
    entries = load_pr_json(path)
    budgets = [e["budget"] for e in entries]
    colours = budget_colours(budgets)

    all_classes = available_classes_pr(entries)
    # Filter to non-empty classes
    nonempty = [
        cls for cls in all_classes
        if any(
            entry.get("curves", {}).get(cls, {}).get(diff, {}).get("recall")
            for entry in entries
            for diff in entry.get("curves", {}).get(cls, {})
        )
    ]
    display_classes = pick_multiple("Select classes", nonempty, list(PREFERRED_CLASSES & set(nonempty)))
    all_diffs = available_difficulties_pr(entries, display_classes[0] if display_classes else all_classes[0])
    sel_diff = pick_from_list("Select difficulty", all_diffs)

    n_cls = len(display_classes)
    fig, axes = plt.subplots(1, n_cls, figsize=(5 * n_cls, 4), squeeze=False)
    axes = axes[0]

    for ax, cls in zip(axes, display_classes):
        for entry in entries:
            budget = entry["budget"]
            curve = entry.get("curves", {}).get(cls, {}).get(sel_diff, {})
            recall = curve.get("recall", [])
            precision = curve.get("precision", [])
            ap = curve.get("ap", None)
            if not recall:
                continue
            lbl = f"budget={budget}" + (f"  AP={ap:.3f}" if ap is not None else "")
            ax.plot(recall, precision, color=colours[budget], label=lbl, linewidth=1.5)

        ax.set_title(f"{cls} — {sel_diff}")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_xlim(0, 1.02)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=7, title="ORA budget")
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"PR Curves ({sel_diff})", fontsize=13, fontweight="bold")

    out = (run_dir / f"plot_pr_{sel_diff.lower()}.png") if not interactive else None
    save_or_show(fig, out)


# ---------------------------------------------------------------------------
# Recall-IoU curves (JSON)
# ---------------------------------------------------------------------------

def load_riou_json(path: pathlib.Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def available_classes_riou(entries: list[dict]) -> list[str]:
    classes: set[str] = set()
    for entry in entries:
        classes.update(entry.get("curves", {}).keys())
    return sorted(classes)


def visualise_recall_iou(path: pathlib.Path, interactive: bool, run_dir: pathlib.Path) -> None:
    entries = load_riou_json(path)
    budgets = [e["budget"] for e in entries]
    colours = budget_colours(budgets)

    all_classes = available_classes_riou(entries)
    nonempty = [
        cls for cls in all_classes
        if any(
            any(v > 0 for v in entry.get("curves", {}).get(cls, {}).get("recall", []))
            for entry in entries
        )
    ]
    display_classes = pick_multiple("Select classes", nonempty, list(PREFERRED_CLASSES & set(nonempty)))

    # confidence threshold (may differ across entries or be absent)
    conf_thresholds = {e.get("confidence_threshold") for e in entries} - {None}
    conf_label = (
        f"(confidence threshold={next(iter(conf_thresholds)):.2f})"
        if len(conf_thresholds) == 1 else ""
    )

    n_cls = len(display_classes)
    fig, axes = plt.subplots(1, n_cls, figsize=(5 * n_cls, 4), squeeze=False)
    axes = axes[0]

    for ax, cls in zip(axes, display_classes):
        for entry in entries:
            budget = entry["budget"]
            curve = entry.get("curves", {}).get(cls, {})
            iou_thresholds = curve.get("iou_thresholds", [])
            recall = curve.get("recall", [])
            if not recall or not iou_thresholds:
                continue
            ax.plot(iou_thresholds, recall, color=colours[budget],
                    label=f"budget={budget}", linewidth=1.5, marker="o", markersize=3)

        ax.set_title(cls)
        ax.set_xlabel("IoU Threshold")
        ax.set_ylabel("Recall")
        ax.set_xlim(0, 1.0)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=7, title="ORA budget")
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"Recall vs IoU Threshold  {conf_label}",
        fontsize=13, fontweight="bold",
    )

    out = (run_dir / "plot_recall_iou.png") if not interactive else None
    save_or_show(fig, out)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualise ORA sweep metrics (AP sweep, PR curves, Recall-IoU curves)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR,
                        help="Base results directory")
    parser.add_argument("--run", default=None, metavar="NAME",
                        help="Run directory name to skip the interactive picker")
    parser.add_argument("--no-interactive", action="store_true",
                        help="Save PNG files instead of showing interactive windows")
    args = parser.parse_args()

    if not HAS_MPL:
        sys.exit(
            "matplotlib and numpy are required.\n"
            "Install with:  pip install matplotlib numpy"
        )

    results_dir = pathlib.Path(args.results_dir)
    if not results_dir.exists():
        sys.exit(f"Results directory not found: {results_dir.resolve()}")

    run_dir = pick_run_dir(results_dir, args.run)
    print(f"\nSelected run: {run_dir}")

    label, metric_path = pick_metric(run_dir)
    print(f"\nVisualising: {metric_path.name}")

    interactive = not args.no_interactive

    if metric_path.suffix == ".csv":
        fmt = pick_from_list("Output format", ["plot", "table"])
        visualise_ap(metric_path, interactive, run_dir, fmt=fmt)
    elif "pr_curves" in metric_path.name:
        visualise_pr(metric_path, interactive, run_dir)
    elif "recall_iou" in metric_path.name:
        visualise_recall_iou(metric_path, interactive, run_dir)
    else:
        sys.exit(f"Unrecognised metric file: {metric_path.name}")


if __name__ == "__main__":
    main()
