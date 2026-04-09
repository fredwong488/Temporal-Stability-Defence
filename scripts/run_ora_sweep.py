"""
scripts/run_ora_sweep.py
------------------------
Run ORA attack sweep on a subset of KITTI velodyne data using PointPillars
and export results to CSV/JSON.

Three metric types are available (controlled via --metric-types):
  ap         — Average Precision per class/difficulty (flat floats → CSV)
  pr         — Precision–Recall curves per class/difficulty (→ JSON)
  recall_iou — Recall vs IoU threshold curves per class (→ JSON)

Usage
-----
    python scripts/run_ora_sweep.py                                         # defaults
    python scripts/run_ora_sweep.py --num-frames 20
    python scripts/run_ora_sweep.py --frames 000125 000070
    python scripts/run_ora_sweep.py --budgets 0 40 200
    python scripts/run_ora_sweep.py --classes Car Pedestrian Cyclist
    python scripts/run_ora_sweep.py --difficulties Easy Moderate
    python scripts/run_ora_sweep.py --metric-types ap pr recall_iou
    python scripts/run_ora_sweep.py --results-dir /path/to/outputs
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import logging
import pathlib
import sys

# Ensure the project root (parent of scripts/) is on sys.path so that
# eval_pipeline and other project modules can be imported.
_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

KITTI_ROOT = "/vol/bitbucket/cyw122/FYP/experiment_pipeline/data/datasets/KITTI"

DEFAULT_DETECTOR = "pointrcnn"
DEFAULT_BUDGETS = [0, 10, 20, 40, 60, 100, 150, 200]
DEFAULT_CLASSES = ["Car", "Pedestrian", "Cyclist"]
DEFAULT_DIFFICULTIES = ["Easy", "Moderate", "Hard"]
DEFAULT_METRIC_TYPES = ["ap"]
DEFAULT_NUM_FRAMES = 50
DEFAULT_RESULTS_DIR = "results"

VALID_METRIC_TYPES = {"ap", "pr", "recall_iou"}
VALID_DIFFICULTIES = {"Easy", "Moderate", "Hard"}


def get_frame_ids(num_frames: int) -> list[str]:
    """Return the first `num_frames` frame IDs from the velodyne training split."""
    velodyne_dir = pathlib.Path(KITTI_ROOT) / "data_object_velodyne" / "training" / "velodyne"
    if not velodyne_dir.exists():
        raise FileNotFoundError(
            f"Velodyne directory not found: {velodyne_dir.resolve()}\n"
            f"Set KITTI_ROOT at the top of the script (currently: '{KITTI_ROOT}') "
            f"or pass explicit frame IDs with --frames."
        )
    bins = sorted(velodyne_dir.glob("*.bin"))
    if not bins:
        raise FileNotFoundError(f"No .bin files found in {velodyne_dir.resolve()}")
    return [p.stem for p in bins[:num_frames]]


def run_budget(
    frame_ids: list[str],
    budget: int,
    classes: list[str],
    difficulties: list[str],
    metric_types: list[str],
    confidence_threshold: float,
    output_dir: str,
    detector_type: str = DEFAULT_DETECTOR,
) -> dict:
    """Run one experiment for the given budget and return the summary dict.

    Budget=0 still runs ORA (removing 0 points) so attacked predictions equal clean
    predictions — this keeps all metric paths consistent across budgets.
    """
    from eval_pipeline.config import ExperimentConfig
    from eval_pipeline.runner import run_experiment

    config = ExperimentConfig(
        kitti_root=KITTI_ROOT,
        frame_ids=frame_ids,
        attack_type="ora",
        attack_params={"budget": budget, "target_types": classes},
        detector_type=detector_type,
        output_dir=output_dir,
        experiment_name=f"ora_budget_{budget}",
        metric_types=metric_types,
        difficulties=difficulties,
        recall_iou_confidence_threshold=confidence_threshold,
    )
    return run_experiment(config)


# ---------------------------------------------------------------------------
# Per-metric extractors
# ---------------------------------------------------------------------------

def extract_ap_row(summary: dict, budget: int, classes: list[str], difficulties: list[str]) -> dict:
    """Extract AP floats for each class/difficulty into a flat dict for the CSV."""
    attacked_map = summary.get("attack_effectiveness", {}).get("attacked_map", {})
    row: dict = {"budget": budget}
    for cls in classes:
        cls_ap = attacked_map.get(cls, {})
        for diff in difficulties:
            row[f"{cls.lower()}_ap_{diff.lower()}"] = cls_ap.get(diff, float("nan"))
    return row


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="ORA budget sweep — exports AP to CSV, PR/recall-IoU curves to JSON",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--num-frames", type=int, default=DEFAULT_NUM_FRAMES,
                       help="Number of frames to sample from the training split")
    group.add_argument("--frames", nargs="+", metavar="ID",
                       help="Explicit frame IDs, e.g. 000125 000070")
    parser.add_argument("--budgets", type=int, nargs="+", default=DEFAULT_BUDGETS,
                        metavar="N", help="Attack budgets to sweep")
    parser.add_argument("--classes", nargs="+", default=DEFAULT_CLASSES,
                        metavar="CLASS",
                        help="Object classes to evaluate. E.g. Car Pedestrian Cyclist")
    parser.add_argument("--difficulties", nargs="+", default=DEFAULT_DIFFICULTIES,
                        choices=sorted(VALID_DIFFICULTIES), metavar="DIFF",
                        help="Difficulty levels for AP/PR metrics (Easy Moderate Hard)")
    parser.add_argument("--metric-types", nargs="+", default=DEFAULT_METRIC_TYPES,
                        choices=sorted(VALID_METRIC_TYPES), metavar="METRIC",
                        help=(
                            "Metric types to compute: "
                            "ap (Average Precision → CSV), "
                            "pr (Precision-Recall curves → JSON), "
                            "recall_iou (Recall vs IoU → JSON)"
                        ))
    parser.add_argument("--results-dir", type=str, default=DEFAULT_RESULTS_DIR,
                        help="Base directory for outputs; each run is saved under a "
                             "timestamped subdirectory (e.g. results/2026-04-08-14-30-00/)")
    parser.add_argument("--detector", type=str, default=DEFAULT_DETECTOR,
                        help="Detector type to use (e.g. pointrcnn, pointpillars)")
    parser.add_argument("--confidence-threshold", type=float, default=0.3,
                        help="Confidence threshold used for recall_iou metric")
    args = parser.parse_args()

    metric_types: list[str] = args.metric_types

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    run_dir = pathlib.Path(args.results_dir) / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    # Resolve frame IDs
    frame_ids = args.frames if args.frames else get_frame_ids(args.num_frames)

    logging.info("Run dir     : %s", run_dir)
    logging.info("Detector    : %s", args.detector)
    logging.info("Frames      : %d  (%s … %s)", len(frame_ids), frame_ids[0], frame_ids[-1])
    logging.info("Budgets     : %s", args.budgets)
    logging.info("Classes     : %s", args.classes)
    logging.info("Difficulties: %s", args.difficulties)
    logging.info("Metrics     : %s", sorted(metric_types))

    ap_rows: list[dict] = []
    pr_all: list[dict] = []
    recall_iou_all: list[dict] = []

    for budget in args.budgets:
        logging.info("--- Budget %d ---", budget)
        summary = run_budget(
            frame_ids, budget, args.classes, args.difficulties,
            metric_types, args.confidence_threshold, str(run_dir),
            detector_type=args.detector,
        )

        if "ap" in metric_types:
            row = extract_ap_row(summary, budget, args.classes, args.difficulties)
            ap_rows.append(row)
            for cls in args.classes:
                values = "  ".join(
                    f"{d}={row[f'{cls.lower()}_ap_{d.lower()}']:.2f}"
                    for d in args.difficulties
                )
                logging.info("  %s AP  %s", cls, values)

        if "pr" in metric_types:
            pr_entry = {"budget": budget, "curves": summary.get("pr_curves", {})}
            # Filter to requested classes only
            pr_entry["curves"] = {
                cls: summary["pr_curves"].get(cls, {})
                for cls in args.classes
            }
            pr_all.append(pr_entry)

        if "recall_iou" in metric_types:
            recall_iou_all.append({
                "budget": budget,
                "confidence_threshold": args.confidence_threshold,
                "curves": {
                    cls: summary.get("recall_iou_curves", {}).get(cls, {})
                    for cls in args.classes
                },
            })

    # AP → CSV
    if "ap" in metric_types and ap_rows:
        fieldnames = ["budget"] + [
            f"{cls.lower()}_ap_{d.lower()}"
            for cls in args.classes
            for d in args.difficulties
        ]
        out_path = run_dir / "ora_ap_sweep.csv"
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(ap_rows)
        logging.info("AP CSV written to %s", out_path)
        print(f"\nAP results: {out_path}")
        print(",".join(fieldnames))
        for row in ap_rows:
            vals = [str(row["budget"])] + [
                f"{row[f'{cls.lower()}_ap_{d.lower()}']:.4f}"
                for cls in args.classes
                for d in args.difficulties
            ]
            print(",".join(vals))

    # PR curves → JSON
    if "pr" in metric_types and pr_all:
        pr_path = run_dir / "ora_pr_curves.json"
        with open(pr_path, "w") as f:
            json.dump(pr_all, f, indent=2)
        logging.info("PR curves written to %s", pr_path)
        print(f"\nPR curves:  {pr_path}")

    # Recall-IoU curves → JSON
    if "recall_iou" in metric_types and recall_iou_all:
        riou_path = run_dir / "ora_recall_iou_curves.json"
        with open(riou_path, "w") as f:
            json.dump(recall_iou_all, f, indent=2)
        logging.info("Recall-IoU curves written to %s", riou_path)
        print(f"\nRecall-IoU: {riou_path}")


if __name__ == "__main__":
    main()