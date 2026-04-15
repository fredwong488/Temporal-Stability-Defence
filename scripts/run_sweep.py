"""
scripts/run_sweep.py
--------------------
Generalised parameter sweep runner.

Iterates one attack or defense parameter over a list of values and collects
AP / PR / recall-IoU metrics.  At least one of --attack, --defense, --detector
must be supplied.

Usage
-----
    # ORA budget sweep with PointRCNN (attack + detector)
    python scripts/run_sweep.py --attack ora --detector pointrcnn \\
        --sweep-param budget --sweep-values 0 10 40 100 200

    # ORA sweep + void-region defense, no explicit detector
    python scripts/run_sweep.py --attack ora --defense void_region \\
        --sweep-param budget --sweep-values 0 40 200

    # Sweep a defense parameter (no attack)
    python scripts/run_sweep.py --defense void_region --detector pointrcnn \\
        --sweep-target defense --sweep-param threshold --sweep-values 0.1 0.3 0.5

    # Attack-only sweep (no detector, no defense)
    python scripts/run_sweep.py --attack ora \\
        --sweep-param budget --sweep-values 0 40 200
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import logging
import pathlib
import sys

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

KITTI_ROOT = "/vol/bitbucket/cyw122/FYP/experiment_pipeline/data/datasets/KITTI"

DEFAULT_SWEEP_PARAM = "budget"
DEFAULT_SWEEP_VALUES = [0, 10, 20, 40, 60, 100, 150, 200]
DEFAULT_CLASSES = ["Car", "Pedestrian", "Cyclist"]
DEFAULT_DIFFICULTIES = ["Easy", "Moderate", "Hard"]
DEFAULT_METRIC_TYPES = ["ap"]
DEFAULT_RESULTS_DIR = "results"
DEFAULT_SPLIT = "val"

VALID_METRIC_TYPES = {"ap", "pr", "recall_iou"}
VALID_DIFFICULTIES = {"Easy", "Moderate", "Hard"}
VALID_SPLITS = {"train", "val", "test"}
VALID_SWEEP_TARGETS = {"attack", "defense"}

IMAGESETS_DIR = _PROJECT_ROOT / "OpenPCDet" / "data" / "kitti" / "ImageSets"


def get_split_frame_ids(split: str, num_frames: int | None = None) -> list[str]:
    """Return frame IDs from an OpenPCDet ImageSets split file."""
    split_file = IMAGESETS_DIR / f"{split}.txt"
    if not split_file.exists():
        raise FileNotFoundError(
            f"Split file not found: {split_file}\n"
            f"Expected OpenPCDet ImageSets at {IMAGESETS_DIR}"
        )
    frame_ids = [line.strip() for line in split_file.read_text().splitlines() if line.strip()]
    if num_frames is not None:
        frame_ids = frame_ids[:num_frames]
    return frame_ids


def run_single(
    frame_ids: list[str],
    attack_type: str | None,
    attack_params: dict,
    defense_type: str | None,
    defense_params: dict,
    detector_type: str | None,
    detector_params: dict,
    classes: list[str],
    difficulties: list[str],
    metric_types: list[str],
    confidence_threshold: float,
    output_dir: str,
    experiment_name: str,
) -> dict:
    """Run one experiment and return the summary dict."""
    from eval_pipeline.config import ExperimentConfig
    from eval_pipeline.runner import run_experiment

    config = ExperimentConfig(
        kitti_root=KITTI_ROOT,
        frame_ids=frame_ids,
        attack_type=attack_type,
        attack_params=attack_params,
        defense_type=defense_type,
        defense_params=defense_params,
        detector_type=detector_type,
        detector_params=detector_params,
        output_dir=output_dir,
        experiment_name=experiment_name,
        metric_types=metric_types,
        difficulties=difficulties,
        recall_iou_confidence_threshold=confidence_threshold,
    )
    return run_experiment(config)


# ---------------------------------------------------------------------------
# Per-metric extractors
# ---------------------------------------------------------------------------

def extract_ap_row(
    summary: dict,
    sweep_param: str,
    sweep_val: float | int,
    classes: list[str],
    difficulties: list[str],
) -> dict:
    """Extract AP floats for each class/difficulty into a flat dict for the CSV."""
    attacked_map = summary.get("attack_effectiveness", {}).get("attacked_map", {})
    row: dict = {sweep_param: sweep_val}
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
        description="Generalised parameter sweep — exports AP to CSV, PR/recall-IoU curves to JSON",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Components (all optional, but at least one required)
    parser.add_argument("--attack", type=str, default=None,
                        help="Attack type to apply (e.g. ora)")
    parser.add_argument("--defense", type=str, default=None,
                        help="Defense type to apply (e.g. void_region)")
    parser.add_argument("--detector", type=str, default=None,
                        help="Detector type to run (e.g. pointrcnn, pointpillars)")

    # Sweep configuration
    parser.add_argument("--sweep-target", type=str, default="attack",
                        choices=sorted(VALID_SWEEP_TARGETS),
                        help="Which component's params to sweep over")
    parser.add_argument("--sweep-param", type=str, default=DEFAULT_SWEEP_PARAM,
                        help="Name of the parameter to sweep (e.g. budget, threshold, seed)")
    parser.add_argument("--sweep-values", type=float, nargs="+",
                        default=None,
                        metavar="V",
                        help=f"Values to sweep. Defaults to {DEFAULT_SWEEP_VALUES} "
                             "when sweep-param is 'budget'")

    # Frames
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--num-frames", type=int, default=None,
                       help="Use the first N frames from the split (default: all)")
    group.add_argument("--frames", nargs="+", metavar="ID",
                       help="Explicit frame IDs, e.g. 000125 000070")
    parser.add_argument("--split", type=str, default=DEFAULT_SPLIT,
                        choices=sorted(VALID_SPLITS),
                        help="KITTI split to use (reads from OpenPCDet ImageSets/)")

    # Evaluation
    parser.add_argument("--classes", nargs="+", default=DEFAULT_CLASSES,
                        metavar="CLASS",
                        help="Object classes to evaluate")
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
    parser.add_argument("--confidence-threshold", type=float, default=0.3,
                        help="Confidence threshold used for recall_iou metric and detector scoring")

    parser.add_argument("--results-dir", type=str, default=DEFAULT_RESULTS_DIR,
                        help="Base directory for outputs; each run is saved under a "
                             "timestamped subdirectory")
    args = parser.parse_args()

    # Validate: at least one component must be specified
    if not any([args.attack, args.defense, args.detector]):
        parser.error("At least one of --attack, --defense, --detector must be supplied.")

    # Validate: sweep target requires the matching component
    if args.sweep_target == "attack" and args.attack is None:
        parser.error("--sweep-target attack requires --attack to be specified.")
    if args.sweep_target == "defense" and args.defense is None:
        parser.error("--sweep-target defense requires --defense to be specified.")

    # Resolve sweep values
    sweep_values: list[float | int]
    if args.sweep_values is not None:
        sweep_values = args.sweep_values
    elif args.sweep_param == DEFAULT_SWEEP_PARAM:
        sweep_values = DEFAULT_SWEEP_VALUES
    else:
        parser.error(
            f"--sweep-values is required when --sweep-param is not '{DEFAULT_SWEEP_PARAM}'."
        )

    # Cast to int when all values are whole numbers (e.g. budget sweep)
    if all(v == int(v) for v in sweep_values):
        sweep_values = [int(v) for v in sweep_values]

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    run_dir = pathlib.Path(args.results_dir) / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    # Resolve frame IDs
    if args.frames:
        frame_ids = args.frames
    else:
        frame_ids = get_split_frame_ids(args.split, args.num_frames)

    detector_params: dict = {}
    if args.detector is not None:
        detector_params["score_threshold"] = args.confidence_threshold

    logging.info("Run dir      : %s", run_dir)
    logging.info("Attack       : %s", args.attack or "(none)")
    logging.info("Defense      : %s", args.defense or "(none)")
    logging.info("Detector     : %s", args.detector or "(none)")
    logging.info("Sweep target : %s", args.sweep_target)
    logging.info("Sweep param  : %s", args.sweep_param)
    logging.info("Sweep values : %s", sweep_values)
    logging.info("Split        : %s", args.split if not args.frames else "custom")
    logging.info("Frames       : %d  (%s … %s)", len(frame_ids), frame_ids[0], frame_ids[-1])
    logging.info("Classes      : %s", args.classes)
    logging.info("Difficulties : %s", args.difficulties)
    logging.info("Metrics      : %s", sorted(args.metric_types))

    ap_rows: list[dict] = []
    pr_all: list[dict] = []
    recall_iou_all: list[dict] = []

    base_attack_params: dict = {"target_types": args.classes} if args.attack else {}
    base_defense_params: dict = {}

    for val in sweep_values:
        logging.info("--- %s=%s ---", args.sweep_param, val)

        if args.sweep_target == "attack":
            attack_params = base_attack_params | {args.sweep_param: val}
            defense_params = base_defense_params
        else:
            attack_params = base_attack_params
            defense_params = base_defense_params | {args.sweep_param: val}

        experiment_name = f"{args.attack or args.defense}_{args.sweep_param}_{val}"

        summary = run_single(
            frame_ids=frame_ids,
            attack_type=args.attack,
            attack_params=attack_params,
            defense_type=args.defense,
            defense_params=defense_params,
            detector_type=args.detector,
            detector_params=detector_params,
            classes=args.classes,
            difficulties=args.difficulties,
            metric_types=args.metric_types,
            confidence_threshold=args.confidence_threshold,
            output_dir=str(run_dir),
            experiment_name=experiment_name,
        )

        if "ap" in args.metric_types:
            row = extract_ap_row(summary, args.sweep_param, val, args.classes, args.difficulties)
            ap_rows.append(row)
            for cls in args.classes:
                values = "  ".join(
                    f"{d}={row[f'{cls.lower()}_ap_{d.lower()}']:.2f}"
                    for d in args.difficulties
                )
                logging.info("  %s AP  %s", cls, values)

        if "pr" in args.metric_types:
            pr_all.append({
                args.sweep_param: val,
                "curves": {
                    cls: summary.get("pr_curves", {}).get(cls, {})
                    for cls in args.classes
                },
            })

        if "recall_iou" in args.metric_types:
            recall_iou_all.append({
                args.sweep_param: val,
                "confidence_threshold": args.confidence_threshold,
                "curves": {
                    cls: summary.get("recall_iou_curves", {}).get(cls, {})
                    for cls in args.classes
                },
            })

    sweep_tag = f"{args.sweep_target}_{args.sweep_param}"

    # AP → CSV
    if "ap" in args.metric_types and ap_rows:
        fieldnames = [args.sweep_param] + [
            f"{cls.lower()}_ap_{d.lower()}"
            for cls in args.classes
            for d in args.difficulties
        ]
        out_path = run_dir / f"sweep_{sweep_tag}_ap.csv"
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(ap_rows)
        logging.info("AP CSV written to %s", out_path)
        print(f"\nAP results: {out_path}")
        print(",".join(fieldnames))
        for row in ap_rows:
            vals = [str(row[args.sweep_param])] + [
                f"{row[f'{cls.lower()}_ap_{d.lower()}']:.4f}"
                for cls in args.classes
                for d in args.difficulties
            ]
            print(",".join(vals))

    # PR curves → JSON
    if "pr" in args.metric_types and pr_all:
        pr_path = run_dir / f"sweep_{sweep_tag}_pr_curves.json"
        with open(pr_path, "w") as f:
            json.dump(pr_all, f, indent=2)
        logging.info("PR curves written to %s", pr_path)
        print(f"\nPR curves:  {pr_path}")

    # Recall-IoU curves → JSON
    if "recall_iou" in args.metric_types and recall_iou_all:
        riou_path = run_dir / f"sweep_{sweep_tag}_recall_iou_curves.json"
        with open(riou_path, "w") as f:
            json.dump(recall_iou_all, f, indent=2)
        logging.info("Recall-IoU curves written to %s", riou_path)
        print(f"\nRecall-IoU: {riou_path}")


if __name__ == "__main__":
    main()
