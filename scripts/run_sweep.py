"""
scripts/run_sweep.py
--------------------
Generalised parameter sweep runner.

Iterates one attack or defense parameter over a list of values and collects
AP / PR / recall-IoU / defense metrics.  At least one of --attack, --defense,
--detector must be supplied.

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
from datetime import datetime
import json
import logging
import pathlib
import sys

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

_DATASETS_BASE = "/vol/bitbucket/cyw122/FYP/experiment_pipeline/data/datasets"
KITTI_ROOT = f"{_DATASETS_BASE}/KITTI"
DEFAULT_NUSCENES_ROOT = f"{_DATASETS_BASE}/nuscenes-v1.0-mini"
DEFAULT_NUSCENES_VERSION = "v1.0-mini"
DEFAULT_NUSCENES_SPLIT = "mini_val"

DEFAULT_SWEEP_PARAM = "budget"
DEFAULT_SWEEP_VALUES = [0, 10, 20, 40, 60, 100, 150, 200]
KITTI_DEFAULT_CLASSES = ["Car", "Pedestrian", "Cyclist"]
NUSCENES_DEFAULT_CLASSES = ["car", "pedestrian", "bicycle"]
DEFAULT_DIFFICULTIES = ["Easy", "Moderate", "Hard"]
DEFAULT_METRIC_TYPES = ["ap"]
DEFAULT_RESULTS_DIR = "results"
DEFAULT_SPLIT = "val"

VALID_METRIC_TYPES = {"ap", "pr", "recall_iou", "detection_rate"}
VALID_DIFFICULTIES = {"Easy", "Moderate", "Hard"}
VALID_SPLITS = {"train", "val", "test"}
VALID_SWEEP_TARGETS = {"attack", "defense"}

IMAGESETS_DIR = _PROJECT_ROOT / "OpenPCDet" / "data" / "kitti" / "ImageSets"


def _parse_kv_params(pairs: list[str]) -> dict:
    """Parse KEY=VALUE strings into a dict, auto-casting values to int/float/str."""
    out: dict = {}
    for item in pairs:
        if "=" not in item:
            raise argparse.ArgumentTypeError(
                f"Invalid parameter '{item}': expected KEY=VALUE format"
            )
        key, _, raw = item.partition("=")
        try:
            out[key] = int(raw)
        except ValueError:
            try:
                out[key] = float(raw)
            except ValueError:
                out[key] = raw
    return out


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
    attack_fraction: float = 1.0,
    attack_fraction_seed: int = 0,
    save_frame_results: bool = False,
    desc: str | None = None,
    dataset_type: str = "kitti",
    dataset_params: dict | None = None,
    precomputed_cache_path: str | None = None,
    use_cached_attacks: bool = False,
    use_predicted_labels: bool = False,
    pred_label_score_threshold: float = 0.5,
    min_unattacked_frames: int = 6,     # defaulted to 6 to suit jitter defense
    min_attacked_frames: int = 6,       # defaulted to 6 to suit jitter defense
) -> dict:
    """Run one experiment and return the summary dict."""
    from eval_pipeline.config import ExperimentConfig
    from eval_pipeline.runner import run_experiment

    config = ExperimentConfig(
        dataset_type=dataset_type,
        dataset_params=dataset_params or {},
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
        attack_fraction=attack_fraction,
        attack_fraction_seed=attack_fraction_seed,
        save_frame_results=save_frame_results,
        precomputed_cache_path=precomputed_cache_path,
        use_cached_attacks=use_cached_attacks,
        use_predicted_labels=use_predicted_labels,
        pred_label_score_threshold=pred_label_score_threshold,
        min_unattacked_frames=min_unattacked_frames,
        min_attacked_frames=min_attacked_frames,
    )
    return run_experiment(config, desc=desc)


# ---------------------------------------------------------------------------
# Per-metric extractors
# ---------------------------------------------------------------------------

def extract_detection_rate_row(
    summary: dict,
    sweep_param: str,
    sweep_val: float | int,
) -> dict:
    """Extract defense F1, precision and recall into a flat dict for the CSV."""
    de = summary.get("defense_effectiveness", {})
    return {
        sweep_param: sweep_val,
        "detection_f1": de.get("f1", float("nan")),
        "detection_precision": de.get("precision", float("nan")),
        "detection_recall": de.get("recall", float("nan")),
    }


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

    # Dataset
    parser.add_argument("--dataset", type=str, default="kitti",
                        choices=["kitti", "nuscenes"],
                        help="Dataset backend to use")
    parser.add_argument("--nuscenes-root", type=str, default=DEFAULT_NUSCENES_ROOT,
                        help="NuScenes dataset root directory")
    parser.add_argument("--nuscenes-version", type=str, default=DEFAULT_NUSCENES_VERSION,
                        help="NuScenes version string (must match the metadata folder name)")
    parser.add_argument("--nuscenes-split", type=str, default=DEFAULT_NUSCENES_SPLIT,
                        help="NuScenes split (e.g. mini_val, mini_train, val, train)")
    parser.add_argument("--nuscenes-scene-names", nargs="+", default=None, metavar="SCENE",
                        help="Restrict NuScenes run to these scene names (e.g. scene-0061 scene-0103)")

    # Components (all optional, but at least one required)
    parser.add_argument("--attack", type=str, default=None,
                        help="Attack type to apply (e.g. ora)")
    parser.add_argument("--attack-fraction", type=float, default=1.0,
                        metavar="F",
                        help="Fraction of frames to attack, chosen randomly (0.0–1.0)")
    parser.add_argument("--attack-fraction-seed", type=int, default=0,
                        help="RNG seed for attack sampling")
    parser.add_argument("--min-unattacked-frames", type=int, default=6,
                        metavar="N",
                        help="Minimum frames left unattacked at the start of each attacked scene "
                             "(NuScenes / scene-granularity datasets only). "
                             "Actual prefix is randomised in [N, scene_length - min-attacked-frames].")
    parser.add_argument("--min-attacked-frames", type=int, default=6,
                        metavar="N",
                        help="Minimum frames that must be attacked in a chosen scene. "
                             "Scenes too short to satisfy both minima revert to unattacked.")
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

    # Frames (KITTI-only)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--kitti-num-frames", type=int, default=None,
                       help="Use the first N frames from the KITTI split (default: all)")
    group.add_argument("--kitti-frames", nargs="+", metavar="ID",
                       help="Explicit KITTI frame IDs, e.g. 000125 000070")
    parser.add_argument("--kitti-split", type=str, default=DEFAULT_SPLIT,
                        choices=sorted(VALID_SPLITS),
                        help="KITTI split to use (reads from OpenPCDet ImageSets/)")

    # Evaluation
    parser.add_argument("--classes", nargs="+", default=None,
                        metavar="CLASS",
                        help="Object classes to evaluate "
                             "(default: Car Pedestrian Cyclist for KITTI; "
                             "car pedestrian bicycle for NuScenes)")
    parser.add_argument("--difficulties", nargs="+", default=DEFAULT_DIFFICULTIES,
                        choices=sorted(VALID_DIFFICULTIES), metavar="DIFF",
                        help="Difficulty levels for AP/PR metrics (Easy Moderate Hard)")
    parser.add_argument("--metric-types", nargs="+", default=DEFAULT_METRIC_TYPES,
                        choices=sorted(VALID_METRIC_TYPES), metavar="METRIC",
                        help=(
                            "Metric types to compute: "
                            "ap (Average Precision → CSV), "
                            "pr (Precision-Recall curves → JSON), "
                            "recall_iou (Recall vs IoU → JSON), "
                            "detection_rate (defense F1/precision/recall → CSV)"
                        ))
    parser.add_argument("--confidence-threshold", type=float, default=0.3,
                        help="Confidence threshold used for recall_iou metric and detector scoring")

    parser.add_argument("--results-dir", type=str, default=DEFAULT_RESULTS_DIR,
                        help="Base directory for outputs; each run is saved under a "
                             "timestamped subdirectory")
    parser.add_argument("--save-frames", action="store_true", default=False,
                        help="Save per-frame JSONL alongside each experiment's results JSON")
    parser.add_argument(
        "--precomputed-cache-dir", type=str, default=None, metavar="DIR",
        help=(
            "Directory for precomputed prediction caches.  For each sweep value "
            "a file named '<sweep-param>_<value>.pkl' is written (if absent) or "
            "read (if present), allowing detector inference to be skipped on "
            "subsequent runs with the same configuration."
        ),
    )
    parser.add_argument(
        "--use-cached-attacks", action="store_true", default=False,
        help=(
            "When a precomputed cache is loaded, use cached attacked predictions "
            "and attack_metadata directly instead of re-applying the attack live. "
            "Guarantees consistency between predictions and metadata but the "
            "defense sees clean lidar rather than the original attacked cloud."
        ),
    )
    parser.add_argument(
        "--use-predicted-labels", action="store_true", default=False,
        help=(
            "Use clean detector predictions as attack labels instead of ground-truth "
            "annotations. Use for datasets where not every frame is labeled "
            "(e.g. NuScenes at 10 Hz) so the attack fires on every frame."
        ),
    )
    parser.add_argument(
        "--pred-label-score-threshold", type=float, default=0.5,
        help="Minimum detection score for a prediction to be used as an attack label "
             "when --use-predicted-labels is set.",
    )
    parser.add_argument(
        "--attack-noise-preset", type=str, default="worst_case",
        choices=["none", "worst_case", "vlp16", "vlp32c", "os1_32", "helios", "horizon", "l515", "xt32"],
        help="Sato 2024 spoofing noise preset for attack reinjection "
             "(default vlp32c; 'none' disables δ_inner/δ_inter/δ_rand).",
    )
    parser.add_argument(
        "--attack-params", nargs="*", default=[], metavar="KEY=VALUE",
        help="Extra fixed attack parameters as KEY=VALUE pairs (e.g. --attack-params eps=0.1 n_iter=10). "
             "Values are auto-cast to int, float, or string. These are merged into base attack params "
             "and are NOT the swept parameter.",
    )
    parser.add_argument(
        "--defense-params", nargs="*", default=[], metavar="KEY=VALUE",
        help="Extra fixed defense parameters as KEY=VALUE pairs (e.g. --defense-params threshold=0.3). "
             "Values are auto-cast to int, float, or string. These are merged into base defense params "
             "and are NOT the swept parameter.",
    )
    args = parser.parse_args()

    if args.classes is None:
        args.classes = (
            NUSCENES_DEFAULT_CLASSES if args.dataset == "nuscenes" else KITTI_DEFAULT_CLASSES
        )

    start_time = datetime.now()

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

    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    run_dir = pathlib.Path(args.results_dir) / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    detector_params: dict = {}
    if args.detector is not None:
        detector_params["score_threshold"] = args.confidence_threshold

    # Build dataset_params for the chosen dataset backend
    dataset_type = args.dataset
    dataset_params: dict = {}
    if dataset_type == "nuscenes":
        dataset_params["root"] = args.nuscenes_root
        dataset_params["version"] = args.nuscenes_version
        dataset_params["split"] = args.nuscenes_split
        if args.nuscenes_scene_names is not None:
            dataset_params["scene_names"] = args.nuscenes_scene_names
    elif dataset_type == "kitti":
        frame_ids = (
            args.kitti_frames
            if args.kitti_frames
            else get_split_frame_ids(args.kitti_split, args.kitti_num_frames)
        )
        dataset_params["root"] = KITTI_ROOT
        dataset_params["frame_ids"] = frame_ids

    extra_attack_params = _parse_kv_params(args.attack_params or [])
    extra_defense_params = _parse_kv_params(args.defense_params or [])

    logging.info("Start time      : %s", start_time)
    logging.info("Run dir      : %s", run_dir)
    logging.info("Dataset      : %s", dataset_type)
    if dataset_type == "nuscenes":
        logging.info("NuScenes     : %s  version=%s  split=%s",
                     args.nuscenes_root, args.nuscenes_version, args.nuscenes_split)
        if args.nuscenes_scene_names:
            logging.info("Scenes       : %s", args.nuscenes_scene_names)
    logging.info("Attack       : %s", args.attack or "(none)")
    if args.attack and args.attack_fraction < 1.0:
        logging.info("Atk fraction : %.2f (seed=%d)", args.attack_fraction, args.attack_fraction_seed)
        if args.min_unattacked_frames != 6 or args.min_attacked_frames != 6:    # If not equal cmd arg defaults
            logging.info(
                "Scene prefix : min_unattacked=%d  min_attacked=%d",
                args.min_unattacked_frames, args.min_attacked_frames,
            )
    logging.info("Defense      : %s", args.defense or "(none)")
    logging.info("Detector     : %s", args.detector or "(none)")
    logging.info("Sweep target : %s", args.sweep_target)
    logging.info("Sweep param  : %s", args.sweep_param)
    logging.info("Sweep values : %s", sweep_values)
    if extra_attack_params:
        logging.info("Atk extras   : %s", extra_attack_params)
    if extra_defense_params:
        logging.info("Def extras   : %s", extra_defense_params)
    if dataset_type == "kitti":
        logging.info("Split        : %s", args.kitti_split if not args.kitti_frames else "custom")
        logging.info("Frames       : %d  (%s … %s)", len(frame_ids), frame_ids[0], frame_ids[-1])
    logging.info("Classes      : %s", args.classes)
    logging.info("Difficulties : %s", args.difficulties)
    logging.info("Metrics      : %s", sorted(args.metric_types))

    ap_rows: list[dict] = []
    pr_all: list[dict] = []
    recall_iou_all: list[dict] = []
    detection_rate_rows: list[dict] = []

    base_attack_params: dict = {"target_types": args.classes} if args.attack else {}
    base_attack_params.update(extra_attack_params)
    if args.attack == "ora" and args.attack_noise_preset != "none":
        from eval_pipeline.utils.spoofing_noise import SpoofingNoiseModel
        base_attack_params["noise_model"] = SpoofingNoiseModel.from_preset(
            args.attack_noise_preset, seed=args.attack_fraction_seed
        )
    base_defense_params: dict = {}
    base_defense_params.update(extra_defense_params)

    # Build the iteration list: when sweeping attack, prepend a no-attack baseline
    # represented by the sentinel string "no_attack" in the sweep_param column.
    sweep_iter: list[float | int | str] = list(sweep_values)
    if args.sweep_target == "attack":
        sweep_iter = ["no_attack"] + sweep_iter

    for val in sweep_iter:
        is_baseline = val == "no_attack"
        logging.info("--- %s=%s ---", args.sweep_param, val)

        if args.sweep_target == "attack":
            if is_baseline:
                attack_params = {}
                active_attack_type: str | None = None
            else:
                attack_params = base_attack_params | {args.sweep_param: val}
                active_attack_type = args.attack
            defense_params = base_defense_params
        else:
            active_attack_type = args.attack
            attack_params = base_attack_params
            defense_params = base_defense_params | {args.sweep_param: val}

        if is_baseline:
            experiment_name = f"baseline_no_attack"
        else:
            experiment_name = f"{args.attack or args.defense}_{args.sweep_param}_{val}"

        # Compute cache path.  When sweeping defense params the attack/detector
        # outputs are identical for every value, so all iterations share one file.
        val_cache_path: str | None = None
        if args.precomputed_cache_dir is not None:
            if args.sweep_target == "defense":
                val_cache_path = str(
                    pathlib.Path(args.precomputed_cache_dir) / "defense_sweep_shared.pkl"
                )
            elif is_baseline:
                val_cache_path = str(
                    pathlib.Path(args.precomputed_cache_dir) / "baseline_no_attack.pkl"
                )
            else:
                val_str = str(int(val)) if val == int(val) else str(val)
                val_cache_path = str(
                    pathlib.Path(args.precomputed_cache_dir) / f"{args.sweep_param}_{val_str}.pkl"
                )

        summary = run_single(
            attack_type=active_attack_type,
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
            attack_fraction=args.attack_fraction,
            attack_fraction_seed=args.attack_fraction_seed,
            save_frame_results=args.save_frames,
            desc="no_attack (baseline)" if is_baseline else f"{args.sweep_param}={val}",
            dataset_type=dataset_type,
            dataset_params=dataset_params,
            precomputed_cache_path=val_cache_path,
            use_cached_attacks=False if is_baseline else args.use_cached_attacks,
            use_predicted_labels=args.use_predicted_labels,
            pred_label_score_threshold=args.pred_label_score_threshold,
            min_unattacked_frames=args.min_unattacked_frames,
            min_attacked_frames=args.min_attacked_frames,
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

        if "detection_rate" in args.metric_types:
            row = extract_detection_rate_row(summary, args.sweep_param, val)
            detection_rate_rows.append(row)
            logging.info(
                "  Detection  F1=%.3f  precision=%.3f  recall=%.3f",
                row["detection_f1"], row["detection_precision"], row["detection_recall"],
            )

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

    # Detection rate → CSV
    if "detection_rate" in args.metric_types and detection_rate_rows:
        fieldnames = [args.sweep_param, "detection_f1", "detection_precision", "detection_recall"]
        out_path = run_dir / f"sweep_{sweep_tag}_detection_rate.csv"
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(detection_rate_rows)
        logging.info("Detection rate CSV written to %s", out_path)
        print(f"\nDetection rate: {out_path}")
        print(",".join(fieldnames))
        for row in detection_rate_rows:
            print(",".join([
                str(row[args.sweep_param]),
                f"{row['detection_f1']:.4f}",
                f"{row['detection_precision']:.4f}",
                f"{row['detection_recall']:.4f}",
            ]))

    end_time = datetime.now()
    logging.info(f"Start time: {start_time}")
    logging.info(f"End time:   {end_time}")
    logging.info(f"Total time: {end_time - start_time}")


if __name__ == "__main__":
    main()
