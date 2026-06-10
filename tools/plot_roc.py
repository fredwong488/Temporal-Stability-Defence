"""
tools/plot_roc.py
-----------------
Plot ROC curves from *_roc_*.json files produced by the sweep pipeline.

Each JSON file contains a grid of (point_threshold × centroid_threshold) evaluations,
each with tp/fp/tn/fn/tpr/fpr.  The ROC curve is the Pareto-optimal front in
(fpr, tpr) space — for each achievable FPR the highest TPR across all threshold
combinations.

Run interactively (pick run + files):
    pixi run python tools/plot_roc.py

Or pass files directly:
    pixi run python tools/plot_roc.py results/dir1/single_run_roc_jitter.json [...]
    pixi run python tools/plot_roc.py results/*/single_run_roc_*.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
from sklearn.metrics import auc as sklearn_auc

RESULTS_DIR = Path(__file__).parent.parent / "results"


# ---------------------------------------------------------------------------
# Interactive pickers
# ---------------------------------------------------------------------------

def _list_roc_dirs(results_dir: Path) -> list[Path]:
    return sorted(
        d for d in results_dir.iterdir()
        if d.is_dir() and any(d.glob("*_roc_*.json"))
    )


def _load_run_notes(run_dir: Path) -> str | None:
    p = run_dir / "run_metadata.json"
    if not p.exists():
        return None
    try:
        with open(p) as f:
            return json.load(f).get("notes")
    except Exception:
        return None


def _pick_run_dir(results_dir: Path) -> Path:
    dirs = _list_roc_dirs(results_dir)
    if not dirs:
        sys.exit(f"No result directories with ROC JSON files found under '{results_dir}'.")

    print(f"\nAvailable run directories in '{results_dir}':")
    for i, d in enumerate(dirs, 1):
        roc_files = sorted(d.glob("*_roc_*.json"))
        notes = _load_run_notes(d)
        notes_str = f"  — {notes}" if notes else ""
        print(f"  [{i}] {d.name}  ({len(roc_files)} ROC file(s)){notes_str}")
    print()

    n = len(dirs)
    prompt = f"Choose run [1-{n}]: "
    while True:
        raw = input(prompt).strip()
        if raw.isdigit() and 1 <= int(raw) <= n:
            return dirs[int(raw) - 1]
        print(f"  Please enter a number between 1 and {n}.")


def _pick_roc_files(run_dir: Path) -> list[Path]:
    roc_files = sorted(run_dir.glob("*_roc_*.json"))
    if not roc_files:
        sys.exit(f"No *_roc_*.json files found in {run_dir}.")

    print(f"\nROC files in '{run_dir.name}':")
    print(f"  [1] all")
    for i, p in enumerate(roc_files, 2):
        print(f"  [{i}] {p.name}")
    print()

    n = len(roc_files)
    prompt = f"Choose [1] for all, or space-separated [2-{n + 1}]: "
    while True:
        raw = input(prompt).strip()
        parts = raw.split()
        try:
            choices = [int(p) for p in parts]
        except ValueError:
            print(f"  Please enter numbers between 1 and {n + 1}.")
            continue
        if parts and all(1 <= c <= n + 1 for c in choices):
            if 1 in choices:
                return roc_files
            return [roc_files[c - 2] for c in sorted(set(choices))]
        print(f"  Please enter numbers between 1 and {n + 1}.")


# ---------------------------------------------------------------------------
# ROC maths
# ---------------------------------------------------------------------------

def load_roc_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def roc_envelope(points: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Return (fpr, tpr) arrays for the Pareto-optimal ROC front.

    For each unique FPR value in the grid, keep only the highest TPR.
    Then sort by FPR and compute a running maximum so the curve is monotone.
    """
    fprs = np.array([p["fpr"] for p in points])
    tprs = np.array([p["tpr"] for p in points])

    order = np.argsort(fprs)
    fprs, tprs = fprs[order], tprs[order]

    tprs_env = np.maximum.accumulate(tprs)

    keep = np.concatenate([[True], np.diff(tprs_env) > 0])
    fpr_env = np.concatenate([[0.0], fprs[keep], [1.0]])
    tpr_env = np.concatenate([[0.0], tprs_env[keep], [1.0]])

    return fpr_env, tpr_env


def suggest_label(path: Path, flag_condition: str) -> str:
    stem = path.stem  # e.g. "single_run_roc_jitter"
    for prefix in ("single_run_roc_", "roc_"):
        if stem.startswith(prefix):
            stem = stem[len(prefix):]
            break
    name = stem.replace("_", " ").title()
    return f"{name} defence ({flag_condition.upper()})"


def prompt(question: str, default: str) -> str:
    answer = input(f"{question} [{default}]: ").strip()
    return answer if answer else default


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if len(sys.argv) >= 2:
        paths = [Path(a) for a in sys.argv[1:]]
        missing = [p for p in paths if not p.exists()]
        if missing:
            for p in missing:
                print(f"ERROR: file not found: {p}")
            sys.exit(1)
        # Try to read notes from the parent dir of the first file
        run_notes = _load_run_notes(paths[0].parent)
    else:
        run_dir = _pick_run_dir(RESULTS_DIR)
        paths = _pick_roc_files(run_dir)
        run_notes = _load_run_notes(run_dir)

    # Pre-load all files so we can show AUCs before asking for the title
    loaded = []
    for path in paths:
        data = load_roc_json(path)
        points = data["points"]
        flag_condition = data.get("flag_condition", "?")
        fpr, tpr = roc_envelope(points)
        roc_auc = sklearn_auc(fpr, tpr)
        suggested = suggest_label(path, flag_condition)
        curve_label = prompt(f"Label for {path.name}", suggested)
        loaded.append((path, data, points, fpr, tpr, roc_auc, curve_label))

    auc_str = "  ".join(f"{lbl}: AUC={auc:.3f}" for _, _, _, _, _, auc, lbl in loaded)
    default_title = run_notes if run_notes else ("ROC Curve" if len(loaded) == 1 else "ROC Curves")
    title_base = prompt("Plot title", default_title)
    title = f"{title_base}  ({auc_str})" if len(loaded) == 1 else f"{title_base}\n{auc_str}"

    out_path = paths[0].parent / "roc_curves.png"

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot([0, 1], [0, 1], color="lightgrey", linestyle="--", linewidth=1)

    prop_cycler = plt.rcParams["axes.prop_cycle"]()
    curve_handles = []

    for path, data, points, fpr, tpr, roc_auc, curve_label in loaded:
        color = next(prop_cycler)["color"]

        ax.plot(fpr, tpr, linewidth=2, color=color)

        all_fpr = [p["fpr"] for p in points]
        all_tpr = [p["tpr"] for p in points]
        ax.scatter(all_fpr, all_tpr, s=12, alpha=0.25, color=color, linewidths=0)

        auc_label = curve_label if len(loaded) == 1 else f"{curve_label}  (AUC={roc_auc:.3f})"
        curve_handles.append(mlines.Line2D(
            [], [], color=color, linewidth=2,
            marker="o", markersize=5, alpha=0.7,
            label=auc_label,
        ))

    key_line = mlines.Line2D([], [], color="grey", linewidth=2,
                             label="Pareto ROC envelope")
    key_dots = mlines.Line2D([], [], color="grey", linewidth=0,
                             marker="o", markersize=5, alpha=0.5,
                             label="Threshold grid points")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(handles=curve_handles + [key_line, key_dots], loc="lower right", fontsize=8)
    ax.set_aspect("equal")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
