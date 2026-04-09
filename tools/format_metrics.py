"""
scripts/generate_metrics.py
---------------------------
Regenerate aggregated metric files from per-budget raw JSON data saved by runner.py.

Lists available run directories under results/ and prompts the user to choose one.
Reads ora_budget_*.json files and produces whichever of the following the data supports:
  - ora_ap_sweep.csv          (when attack_effectiveness is present)
  - ora_pr_curves.json        (when pr_curves is present)
  - ora_recall_iou_curves.json (when recall_iou_curves is present)

Usage
-----
    python scripts/generate_metrics.py
    python scripts/generate_metrics.py --results-dir /path/to/results
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import re
import sys


DEFAULT_RESULTS_DIR = "results"


def list_run_dirs(results_dir: pathlib.Path) -> list[pathlib.Path]:
    """Return sorted subdirectories of results_dir that contain budget JSON files."""
    if not results_dir.exists():
        return []
    dirs = sorted(
        d for d in results_dir.iterdir()
        if d.is_dir() and list(d.glob("ora_budget_*.json"))
    )
    return dirs


def pick_run_dir(results_dir: pathlib.Path) -> pathlib.Path:
    """List available run directories and return the user's choice."""
    dirs = list_run_dirs(results_dir)
    if not dirs:
        print(f"No run directories with ora_budget_*.json found in '{results_dir}'.")
        sys.exit(1)

    print(f"\nAvailable run directories in '{results_dir}':")
    for i, d in enumerate(dirs, 1):
        json_count = len(list(d.glob("ora_budget_*.json")))
        print(f"  [{i}] {d.name}  ({json_count} budget file{'s' if json_count != 1 else ''})")

    while True:
        raw = input(f"\nChoose a directory [1-{len(dirs)}]: ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(dirs):
            return dirs[int(raw) - 1]
        print(f"  Please enter a number between 1 and {len(dirs)}.")


def load_budget_files(run_dir: pathlib.Path) -> list[tuple[int, dict]]:
    """Load all ora_budget_*.json files, sorted by budget number."""
    files = run_dir.glob("ora_budget_*.json")

    def budget_num(p: pathlib.Path) -> int:
        m = re.search(r"ora_budget_(\d+)\.json", p.name)
        return int(m.group(1)) if m else -1

    sorted_files = sorted(files, key=budget_num)
    results = []
    for path in sorted_files:
        budget = budget_num(path)
        with open(path) as f:
            results.append((budget, json.load(f)))
    return results


def detect_classes_and_difficulties(attacked_map: dict) -> tuple[list[str], list[str]]:
    """Infer classes and difficulties from the attacked_map dict."""
    classes = [k for k in attacked_map if k != "mAP"]
    difficulties: list[str] = []
    for cls in classes:
        difficulties = [k for k in attacked_map[cls] if k != "all"]
        break  # all classes have the same difficulties
    return classes, difficulties


def generate_ap_csv(
    run_dir: pathlib.Path,
    budget_data: list[tuple[int, dict]],
) -> pathlib.Path | None:
    """Write ora_ap_sweep.csv from attack_effectiveness data. Returns path or None."""
    rows: list[dict] = []
    classes: list[str] = []
    difficulties: list[str] = []

    for budget, summary in budget_data:
        attacked_map = summary.get("attack_effectiveness", {}).get("attacked_map", {})
        if not attacked_map:
            continue
        if not classes:
            classes, difficulties = detect_classes_and_difficulties(attacked_map)

        row: dict = {"budget": budget}
        for cls in classes:
            cls_ap = attacked_map.get(cls, {})
            for diff in difficulties:
                row[f"{cls.lower()}_ap_{diff.lower()}"] = cls_ap.get(diff, float("nan"))
        rows.append(row)

    if not rows:
        return None

    fieldnames = ["budget"] + [
        f"{cls.lower()}_ap_{diff.lower()}"
        for cls in classes
        for diff in difficulties
    ]
    out_path = run_dir / "ora_ap_sweep.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Print table to stdout (same format as run_ora_sweep.py)
    print(f"\nAP results: {out_path}")
    print(",".join(fieldnames))
    for row in rows:
        vals = [str(row["budget"])] + [
            f"{row[f'{cls.lower()}_ap_{diff.lower()}']:.4f}"
            for cls in classes
            for diff in difficulties
        ]
        print(",".join(vals))

    return out_path


def generate_pr_json(
    run_dir: pathlib.Path,
    budget_data: list[tuple[int, dict]],
) -> pathlib.Path | None:
    """Write ora_pr_curves.json. Returns path or None."""
    entries: list[dict] = []
    for budget, summary in budget_data:
        pr_curves = summary.get("pr_curves")
        if pr_curves is None:
            continue
        entries.append({"budget": budget, "curves": pr_curves})

    if not entries:
        return None

    out_path = run_dir / "ora_pr_curves.json"
    with open(out_path, "w") as f:
        json.dump(entries, f, indent=2)
    print(f"\nPR curves:  {out_path}")
    return out_path


def generate_recall_iou_json(
    run_dir: pathlib.Path,
    budget_data: list[tuple[int, dict]],
) -> pathlib.Path | None:
    """Write ora_recall_iou_curves.json. Returns path or None."""
    entries: list[dict] = []
    for budget, summary in budget_data:
        riou_curves = summary.get("recall_iou_curves")
        if riou_curves is None:
            continue
        confidence_threshold = (
            summary.get("config", {}).get("recall_iou_confidence_threshold", None)
        )
        entry: dict = {"budget": budget, "curves": riou_curves}
        if confidence_threshold is not None:
            entry["confidence_threshold"] = confidence_threshold
        entries.append(entry)

    if not entries:
        return None

    out_path = run_dir / "ora_recall_iou_curves.json"
    with open(out_path, "w") as f:
        json.dump(entries, f, indent=2)
    print(f"\nRecall-IoU: {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regenerate aggregated metric files from per-budget raw JSON data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--results-dir", type=str, default=DEFAULT_RESULTS_DIR,
        help="Base results directory containing timestamped run subdirectories",
    )
    args = parser.parse_args()

    results_dir = pathlib.Path(args.results_dir)
    run_dir = pick_run_dir(results_dir)

    print(f"\nLoading budget files from: {run_dir}")
    budget_data = load_budget_files(run_dir)
    if not budget_data:
        print("No ora_budget_*.json files found.")
        sys.exit(1)

    budgets = [b for b, _ in budget_data]
    print(f"Found {len(budget_data)} budget file(s): {budgets}")

    written: list[pathlib.Path] = []

    ap_path = generate_ap_csv(run_dir, budget_data)
    if ap_path:
        written.append(ap_path)

    pr_path = generate_pr_json(run_dir, budget_data)
    if pr_path:
        written.append(pr_path)

    riou_path = generate_recall_iou_json(run_dir, budget_data)
    if riou_path:
        written.append(riou_path)

    if not written:
        print("\nNo metric data found in any budget file (no attack_effectiveness, pr_curves, or recall_iou_curves keys).")
        sys.exit(1)

    print(f"\nDone. {len(written)} file(s) written to {run_dir}/")


if __name__ == "__main__":
    main()
