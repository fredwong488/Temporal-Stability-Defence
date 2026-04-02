"""
run_ora_sweep.py
----------------
Run ORA attack sweep (budgets 0, 40, 200) on a subset of KITTI velodyne data
using PointPillars and export Car class AP (Easy/Moderate/Hard) to CSV.

Usage
-----
    python run_ora_sweep.py                          # use default 50 frames
    python run_ora_sweep.py --num-frames 20          # use first 20 frames
    python run_ora_sweep.py --frames 000125 000070   # specific frame IDs
    python run_ora_sweep.py --output results/sweep.csv
"""

from __future__ import annotations

import argparse
import csv
import logging
import pathlib

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

# Paths — adjust if your layout differs
KITTI_ROOT = "/vol/bitbucket/cyw122/FYP/experiment_pipeline/data/datasets/KITTI"
CONFIG_PATH = "OpenPCDet/tools/cfgs/kitti_models/pointpillar.yaml"
CHECKPOINT_PATH = "models/openpcdet/pointpillar_7728.pth"
BUDGETS = [0, 40, 200]
DEFAULT_NUM_FRAMES = 50
OUTPUT_DEFAULT = "results/ora_car_ap_sweep.csv"


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


def run_budget(frame_ids: list[str], budget: int, output_dir: str) -> dict:
    """Run one experiment for the given budget and return the summary dict.

    Budget=0 still uses the ORA attack (removing 0 points) so the attacked frame
    equals the clean frame — this keeps `attack_effectiveness()` populated for all
    budgets and lets us extract `attacked_map` consistently.
    """
    from eval_pipeline.config import ExperimentConfig
    from eval_pipeline.runner import run_experiment

    config = ExperimentConfig(
        kitti_root=KITTI_ROOT,
        frame_ids=frame_ids,
        attack_type="ora",
        attack_params={"budget": budget, "target_types": ["Car"]},
        detector_type="pointpillars",
        detector_params={
            "config_path": CONFIG_PATH,
            "checkpoint_path": CHECKPOINT_PATH,
        },
        output_dir=output_dir,
        experiment_name=f"ora_budget_{budget}",
    )
    return run_experiment(config)


def extract_car_ap(summary: dict, budget: int) -> dict:
    """Pull Car Easy/Moderate/Hard AP from the summary dict."""
    ae = summary.get("attack_effectiveness", {})
    # All budgets (including 0) use attacked_map — for budget=0 this equals clean_map
    car_ap = ae.get("attacked_map", {}).get("Car", {})
    return {
        "budget": budget,
        "car_ap_easy": car_ap.get("Easy", float("nan")),
        "car_ap_moderate": car_ap.get("Moderate", float("nan")),
        "car_ap_hard": car_ap.get("Hard", float("nan")),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="ORA budget sweep → Car AP CSV")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--num-frames", type=int, default=DEFAULT_NUM_FRAMES,
                       help="Number of frames to sample from the training split")
    group.add_argument("--frames", nargs="+", metavar="ID",
                       help="Explicit frame IDs, e.g. 000125 000070")
    parser.add_argument("--output", type=str, default=OUTPUT_DEFAULT,
                        help="Path for the output CSV file")
    parser.add_argument("--budgets", type=int, nargs="+", default=BUDGETS,
                        help="Attack budgets to sweep (default: 0 40 200)")
    parser.add_argument("--results-dir", type=str, default="results",
                        help="Directory for per-experiment JSON outputs")
    args = parser.parse_args()

    # Resolve frame IDs
    if args.frames:
        frame_ids = args.frames
    else:
        frame_ids = get_frame_ids(args.num_frames)

    logging.info("Running sweep on %d frames: %s … %s",
                 len(frame_ids), frame_ids[0], frame_ids[-1])
    logging.info("Budgets: %s", args.budgets)

    rows: list[dict] = []
    for budget in args.budgets:
        logging.info("--- Budget %d ---", budget)
        summary = run_budget(frame_ids, budget, args.results_dir)

        # Save per-budget JSON for inspection
        json_path = pathlib.Path(args.results_dir) / f"ora_budget_{budget}.json"
        logging.info("Full results saved to %s", json_path)

        row = extract_car_ap(summary, budget)
        rows.append(row)
        logging.info("  Car AP  Easy=%.2f  Moderate=%.2f  Hard=%.2f",
                     row["car_ap_easy"], row["car_ap_moderate"], row["car_ap_hard"])

    # Write CSV
    out_path = pathlib.Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["budget", "car_ap_easy", "car_ap_moderate", "car_ap_hard"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    logging.info("CSV written to %s", out_path)
    print(f"\nResults saved to: {out_path}")
    print("\nbudget,car_ap_easy,car_ap_moderate,car_ap_hard")
    for row in rows:
        print(f"{row['budget']},{row['car_ap_easy']:.4f},{row['car_ap_moderate']:.4f},{row['car_ap_hard']:.4f}")


if __name__ == "__main__":
    main()
