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
from datetime import datetime
import json
import logging
import pathlib
import subprocess
import sys

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from eval_pipeline.config import ExperimentConfig
from eval_pipeline.runner import run_experiment
from scripts.sweep_metrics import (
    extract_ap_row,
    extract_detection_rate_row,
    extract_defense_effectiveness_row,
    extract_clustering_quality_row,
    extract_pacts_effectiveness_row,
    extract_llm_attack_type_accuracy_row,
    extract_llm_cost_metrics_row,
    extract_timing_metrics_row,
    log_summary_metrics,
    write_ap_csv,
    write_detection_rate_csv,
    write_defense_effectiveness_csv,
    write_clustering_quality_csv,
    write_pacts_effectiveness_csv,
    write_llm_attack_type_accuracy_csv,
    write_llm_cost_metrics_csv,
    write_timing_metrics_csv,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

_DATASETS_BASE = "/vol/bitbucket/cyw122/FYP/experiment_pipeline/data/datasets"
KITTI_ROOT = f"{_DATASETS_BASE}/KITTI"
DEFAULT_NUSCENES_ROOT = f"{_DATASETS_BASE}/nuscenes-v1.0-mini"
DEFAULT_NUSCENES_VERSION = "v1.0-mini"
DEFAULT_NUSCENES_SPLIT = "mini_val"

KITTI_DEFAULT_CLASSES = ["Car", "Pedestrian", "Cyclist"]
NUSCENES_DEFAULT_CLASSES = ["car", "pedestrian", "bicycle"]
DEFAULT_DIFFICULTIES = ["Easy", "Moderate", "Hard"]
DEFAULT_METRIC_TYPES = ["ap"]
DEFAULT_RESULTS_DIR = "results"
DEFAULT_SPLIT = "val"

VALID_METRIC_TYPES = {"ap", "pr", "recall_iou", "detection_rate", "defense_effectiveness", "clustering_quality", "pacts_effectiveness", "roc_jitter", "llm_attack_type_accuracy", "llm_cost_metrics", "timing_metrics"}
VALID_DIFFICULTIES = {"Easy", "Moderate", "Hard"}
VALID_SPLITS = {"train", "val", "test"}
VALID_SWEEP_TARGETS = {"attack", "defense"}

IMAGESETS_DIR = _PROJECT_ROOT / "OpenPCDet" / "data" / "kitti" / "ImageSets"


def _parse_kv_params(pairs: list[str]) -> dict:
    """Parse KEY=VALUE strings into a dict, auto-casting values to bool/int/float/str."""
    out: dict = {}
    for item in pairs:
        if "=" not in item:
            raise argparse.ArgumentTypeError(
                f"Invalid parameter '{item}': expected KEY=VALUE format"
            )
        key, _, raw = item.partition("=")
        if raw.lower() == "true":
            out[key] = True
        elif raw.lower() == "false":
            out[key] = False
        else:
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
    verbose_frame_results: bool = False,
    desc: str | None = None,
    dataset_type: str = "kitti",
    dataset_params: dict | None = None,
    precomputed_cache_path: str | None = None,
    read_only_cache: bool = True,
    use_cached_attacks: bool = False,
    use_predicted_labels: bool = False,
    pred_label_score_threshold: float = 0.5,
    min_unattacked_frames: int = 6,     # defaulted to 6 to suit jitter defense
    min_attacked_frames: int = 6,       # defaulted to 6 to suit jitter defense
    checkpoint_path: str | None = None,
) -> dict:
    """Run one experiment and return the summary dict."""

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
        verbose_frame_results=verbose_frame_results,
        precomputed_cache_path=precomputed_cache_path,
        read_only_cache=read_only_cache,
        use_cached_attacks=use_cached_attacks,
        use_predicted_labels=use_predicted_labels,
        pred_label_score_threshold=pred_label_score_threshold,
        min_unattacked_frames=min_unattacked_frames,
        min_attacked_frames=min_attacked_frames,
        checkpoint_path=checkpoint_path,
    )
    return run_experiment(config, desc=desc)


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

def write_sweep_metadata(
    *,
    run_dir: pathlib.Path,
    timestamp: str,
    notes: str | None,
    cmd_args: list[str],
    sweep_target: str | None,
    sweep_param: str | None,
    sweep_values: list,
    base_attack_params: dict,
    base_defense_params: dict,
    attack_noise_preset: str,
    dataset_type: str,
    dataset_params: dict,
    attack_type: str | None,
    defense_type: str | None,
    detector_type: str | None,
    detector_params: dict,
    metric_types: list,
    difficulties: list,
    confidence_threshold: float,
    attack_fraction: float,
    attack_fraction_seed: int,
    use_cached_attacks: bool,
    use_predicted_labels: bool,
    pred_label_score_threshold: float,
    min_unattacked_frames: int,
    min_attacked_frames: int,
    save_frame_results: bool,
    verbose_frame_results: bool,
    precomputed_cache_dir: str | None,
) -> None:
    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_PROJECT_ROOT, text=True
        ).strip()
    except subprocess.CalledProcessError:
        git_hash = "unknown"

    is_sweep = any([sweep_target, sweep_param, sweep_values])

    if is_sweep:
        first_val = sweep_values[0]
        rep_attack_params = base_attack_params | ({sweep_param: first_val} if sweep_target == "attack" else {})
        rep_defense_params = base_defense_params | ({sweep_param: first_val} if sweep_target == "defense" else {})
    else:
        rep_attack_params = base_attack_params
        rep_defense_params = base_defense_params

    rep_config = ExperimentConfig(
        dataset_type=dataset_type,
        dataset_params=dataset_params,
        attack_type=attack_type,
        attack_params=rep_attack_params,
        defense_type=defense_type,
        defense_params=rep_defense_params,
        detector_type=detector_type,
        detector_params=detector_params,
        metric_types=metric_types,
        difficulties=difficulties,
        recall_iou_confidence_threshold=confidence_threshold,
        attack_fraction=attack_fraction,
        attack_fraction_seed=attack_fraction_seed,
        use_cached_attacks=use_cached_attacks,
        use_predicted_labels=use_predicted_labels,
        pred_label_score_threshold=pred_label_score_threshold,
        min_unattacked_frames=min_unattacked_frames,
        min_attacked_frames=min_attacked_frames,
        save_frame_results=save_frame_results,
        verbose_frame_results=verbose_frame_results,
    )
    config_dict = rep_config.to_dict()

    for k in ("output_dir", "experiment_name", "precomputed_cache_path", "cache_clean_preds", "iou_thresholds"):
        config_dict.pop(k, None)

    if is_sweep:
        # Remove the single swept value; the full sweep block replaces it below
        swept_params = config_dict.get("attack_params" if sweep_target == "attack" else "defense_params", {})
        swept_params.pop(sweep_param, None)

    # noise_model is not JSON-serialisable; record the preset string instead
    attack_params_dict = config_dict.get("attack_params", {})
    if "noise_model" in attack_params_dict:
        attack_params_dict.pop("noise_model")
        attack_params_dict["noise_preset"] = attack_noise_preset

    metadata = {
        "notes": notes,
        "git_commit": git_hash,
        "timestamp": timestamp,
        "cmd_args": cmd_args,
        "precomputed_cache_dir": precomputed_cache_dir,
        "sweep": (
            {"target": sweep_target, "param": sweep_param, "values": sweep_values}
            if is_sweep else None
        ),
        **config_dict,
    }
    metadata_path = run_dir / "run_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logging.info("Metadata written to %s", metadata_path)


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
    parser.add_argument("--nuscenes-keyframes-only", action="store_true", default=False,
                        help="Yield only annotated keyframes (2 Hz) instead of all sweeps (~20 Hz)")
    parser.add_argument("--nuscenes-num-scenes", type=int, default=None, metavar="N",
                        help="Use the first N scenes from the NuScenes split (default: all)")

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
    parser.add_argument("--sweep-target", type=str, default=None,
                        choices=sorted(VALID_SWEEP_TARGETS),
                        help="Which component's params to sweep over (omit for single-run mode)")
    parser.add_argument("--sweep-param", type=str, default=None,
                        help="Name of the parameter to sweep (e.g. budget, threshold, seed; omit for single-run mode)")
    parser.add_argument("--sweep-values", type=float, nargs="+",
                        default=None,
                        metavar="V",
                        help="Values to sweep (required when --sweep-param is set)")

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
                            "ap (Average Precision → CSV, KITTI only), "
                            "pr (Precision-Recall curves → JSON, KITTI only), "
                            "recall_iou (Recall vs IoU → JSON, KITTI only), "
                            "detection_rate (recall drop clean→attacked vs GT → CSV, all datasets), "
                            "defense_effectiveness (defense F1/precision/recall → CSV), "
                            "pacts_effectiveness (PACTS cluster-level F1/precision/recall → CSV), "
                            "roc_jitter (radial-jitter 2-D ROC surface → per-experiment JSON), "
                            "llm_cost_metrics (LLM token stats → CSV), "
                            "timing_metrics (per-phase wall-clock stats for any defense → CSV)"
                        ))
    parser.add_argument("--confidence-threshold", type=float, default=0.3,
                        help="Confidence threshold used for recall_iou metric and detector scoring")

    parser.add_argument("--results-dir", type=str, default=DEFAULT_RESULTS_DIR,
                        help="Base directory for outputs; each run is saved under a "
                             "timestamped subdirectory")
    parser.add_argument(
        "--run-name", type=str, default=None, metavar="NAME",
        help=(
            "Run name (timestamp) to write (or resume) results into.  "
            "The run directory is <results-dir>/<run-name>.  "
            "If the directory already exists the run is resumed: completed "
            "experiments (whose <name>.json is present) are skipped and the "
            "partial experiment is continued from its last scene checkpoint.  "
            "When omitted, a fresh timestamped name is generated."
        ),
    )
    parser.add_argument("--save-frames", action="store_true", default=False,
                        help="Save per-frame JSONL alongside each experiment's results JSON")
    parser.add_argument("--verbose-frames", action="store_true", default=False,
                        help="Include full prediction lists in per-frame JSONL (default: counts only)")
    parser.add_argument("--no-checkpoint", action="store_true", default=False,
                        help="Disable scene-level checkpointing (required for async defenses)")
    parser.add_argument(
        "--checkpoint-dir", type=str, default="/vol/bitbucket/cyw122/FYP/experiment_pipeline/checkpoints", metavar="DIR",
        help=(
            "Base directory for scene-level checkpoint files. A subdirectory named after "
            "the run name is created beneath it, e.g. <checkpoint-dir>/<run-name>/<experiment>. "
            "Defaults to <run-dir> (checkpoints co-located with results). "
            "Set to a fast/large scratch volume to keep results separate from bulky pickle files."
        ),
    )
    parser.add_argument(
        "--precomputed-cache-dir", type=str, default=None, metavar="DIR",
        help=(
            "Directory for precomputed prediction caches.  For each sweep value "
            "a file named '<sweep-param>_<value>.pkl' is written (if absent) or "
            "read (if present), allowing detector inference to be skipped on "
            "subsequent runs with the same configuration."
        ),
    )
    parser.add_argument("--writeable-cache", action="store_true", default=False,
                        help=(
                            "Write to an existing or create a new precomputed cache (default to False, safe for parallel runs). "
                            "When true, opens precomputed cache writable so that cache misses are run live "
                            "and written back, resuming a crashed or partial cache generation."
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
        "--notes", type=str, default=None,
        help="Free-text notes about this sweep run, stored in sweep_metadata.json.",
    )
    parser.add_argument(
        "--attack-noise-preset", type=str, default="worst_case",
        choices=["none", "worst_case", "worst_case_high_error", "vlp16", "vlp32c", "os1_32", "helios", "horizon", "l515", "xt32"],
        help="Sato 2024 spoofing noise preset for attack reinjection "
             "('none' disables δ_inner/δ_inter/δ_rand).",
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

    # Single-run mode when none of the sweep args are supplied
    is_sweep = any([
        args.sweep_param is not None,
        args.sweep_values is not None,
        args.sweep_target is not None,
    ])

    sweep_values: list[float | int] = []
    if is_sweep:
        # Apply defaults for unset sweep args
        if args.sweep_target is None:
            args.sweep_target = "attack"
        if args.sweep_param is None:
            parser.error("--sweep-param is required when sweep mode is active.")

        # Validate: sweep target requires the matching component
        if args.sweep_target == "attack" and args.attack is None:
            parser.error("--sweep-target attack requires --attack to be specified.")
        if args.sweep_target == "defense" and args.defense is None:
            parser.error("--sweep-target defense requires --defense to be specified.")

        # Resolve sweep values
        if args.sweep_values is not None:
            sweep_values = args.sweep_values
        else:
            parser.error("--sweep-values is required when sweep mode is active.")

        # Cast to int when all values are whole numbers (e.g. budget sweep)
        if all(v == int(v) for v in sweep_values):
            sweep_values = [int(v) for v in sweep_values]

    run_name = args.run_name if args.run_name is not None else datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    run_dir = pathlib.Path(args.results_dir) / run_name
    resuming = run_dir.exists()
    run_dir.mkdir(parents=True, exist_ok=True)
    if resuming:
        logging.info("Resuming existing run dir: %s", run_dir)

    checkpoint_run_dir = (pathlib.Path(args.checkpoint_dir) / run_name) if args.checkpoint_dir else run_dir
    if args.checkpoint_dir:
        checkpoint_run_dir.mkdir(parents=True, exist_ok=True)
        logging.info("Checkpoint dir: %s", checkpoint_run_dir)

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
        elif args.nuscenes_num_scenes is not None:
            from nuscenes.utils.splits import create_splits_scenes
            _all_scenes = create_splits_scenes()[args.nuscenes_split]
            dataset_params["scene_names"] = _all_scenes[: args.nuscenes_num_scenes]
        if args.nuscenes_keyframes_only:
            dataset_params["keyframes_only"] = True
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
        elif args.nuscenes_num_scenes is not None:
            logging.info("Num scenes   : %d (first from %s)", args.nuscenes_num_scenes, args.nuscenes_split)
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
    if is_sweep:
        logging.info("Sweep target : %s", args.sweep_target)
        logging.info("Sweep param  : %s", args.sweep_param)
        logging.info("Sweep values : %s", sweep_values)
    else:
        logging.info("Mode         : single run")
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
    defense_effectiveness_rows: list[dict] = []
    clustering_quality_rows: list[dict] = []
    pacts_effectiveness_rows: list[dict] = []
    llm_attack_type_accuracy_rows: list[dict] = []
    llm_cost_metrics_rows: list[dict] = []
    timing_metrics_rows: list[dict] = []

    base_attack_params: dict = {"target_types": args.classes} if args.attack and args.attack == "ora" else {}
    base_attack_params.update(extra_attack_params)
    if args.attack and args.attack_noise_preset != "none":
        from eval_pipeline.utils.spoofing_noise import SpoofingNoiseModel
        base_attack_params["noise_model"] = SpoofingNoiseModel.from_preset(
            args.attack_noise_preset, seed=args.attack_fraction_seed
        )
    base_defense_params: dict = {}
    base_defense_params.update(extra_defense_params)

    if not resuming:
        write_sweep_metadata(
            run_dir=run_dir,
            timestamp=run_name,
            notes=args.notes,
            cmd_args=sys.argv,
            sweep_target=args.sweep_target,
            sweep_param=args.sweep_param,
            sweep_values=sweep_values,
            base_attack_params=base_attack_params,
            base_defense_params=base_defense_params,
            attack_noise_preset=args.attack_noise_preset,
            dataset_type=dataset_type,
            dataset_params=dataset_params,
            attack_type=args.attack,
            defense_type=args.defense,
            detector_type=args.detector,
            detector_params=detector_params,
            metric_types=args.metric_types,
            difficulties=args.difficulties,
            confidence_threshold=args.confidence_threshold,
            attack_fraction=args.attack_fraction,
            attack_fraction_seed=args.attack_fraction_seed,
            use_cached_attacks=args.use_cached_attacks,
            use_predicted_labels=args.use_predicted_labels,
            pred_label_score_threshold=args.pred_label_score_threshold,
            min_unattacked_frames=args.min_unattacked_frames,
            min_attacked_frames=args.min_attacked_frames,
            save_frame_results=args.save_frames,
            verbose_frame_results=args.verbose_frames,
            precomputed_cache_dir=args.precomputed_cache_dir,
        )
    else:
        logging.info("Skipping metadata write (resuming existing run dir)")

    if not is_sweep:
        # -----------------------------------------------------------------------
        # Single evaluation run
        # -----------------------------------------------------------------------
        _experiment_name = "single_run"
        _result_json = run_dir / f"{_experiment_name}.json"
        if resuming and _result_json.exists():
            logging.info("Skipping %s (already complete)", _experiment_name)
            with open(_result_json) as _f:
                summary = json.load(_f)
        else:
            summary = run_single(
                attack_type=args.attack,
                attack_params=base_attack_params,
                defense_type=args.defense,
                defense_params=base_defense_params,
                detector_type=args.detector,
                detector_params=detector_params,
                classes=args.classes,
                difficulties=args.difficulties,
                metric_types=args.metric_types,
                confidence_threshold=args.confidence_threshold,
                output_dir=str(run_dir),
                experiment_name=_experiment_name,
                attack_fraction=args.attack_fraction,
                attack_fraction_seed=args.attack_fraction_seed,
                save_frame_results=args.save_frames,
                verbose_frame_results=args.verbose_frames,
                dataset_type=dataset_type,
                dataset_params=dataset_params,
                precomputed_cache_path=(
                    str(pathlib.Path(args.precomputed_cache_dir) / _experiment_name)
                    if args.precomputed_cache_dir else None
                ),
                read_only_cache=not args.writeable_cache,
                use_cached_attacks=args.use_cached_attacks,
                use_predicted_labels=args.use_predicted_labels,
                pred_label_score_threshold=args.pred_label_score_threshold,
                min_unattacked_frames=args.min_unattacked_frames,
                min_attacked_frames=args.min_attacked_frames,
                desc="single run",
                checkpoint_path=None if args.no_checkpoint else str(checkpoint_run_dir / _experiment_name),
            )

        log_summary_metrics(summary, args.metric_types, args.classes, args.difficulties)

        end_time = datetime.now()
        logging.info(f"Start time: {start_time}")
        logging.info(f"End time:   {end_time}")
        logging.info(f"Total time: {end_time - start_time}")
        return

    # -----------------------------------------------------------------------
    # Sweep mode
    # -----------------------------------------------------------------------

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
                    pathlib.Path(args.precomputed_cache_dir) / "defense_sweep_shared"
                )
            elif is_baseline:
                val_cache_path = str(
                    pathlib.Path(args.precomputed_cache_dir) / "baseline_no_attack"
                )
            else:
                val_str = str(int(val)) if val == int(val) else str(val)
                val_cache_path = str(
                    pathlib.Path(args.precomputed_cache_dir) / f"{args.sweep_param}_{val_str}"
                )

        # Two-level resume: skip completed experiments whose results JSON is present.
        _result_json = run_dir / f"{experiment_name}.json"
        if resuming and _result_json.exists():
            logging.info("Skipping %s (already complete)", experiment_name)
            with open(_result_json) as _f:
                summary = json.load(_f)
        else:
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
                verbose_frame_results=args.verbose_frames,
                desc="no_attack (baseline)" if is_baseline else f"{args.sweep_param}={val}",
                dataset_type=dataset_type,
                dataset_params=dataset_params,
                precomputed_cache_path=val_cache_path,
                read_only_cache=not args.writeable_cache,
                use_cached_attacks=False if is_baseline else args.use_cached_attacks,
                use_predicted_labels=args.use_predicted_labels,
                pred_label_score_threshold=args.pred_label_score_threshold,
                min_unattacked_frames=args.min_unattacked_frames,
                min_attacked_frames=args.min_attacked_frames,
                checkpoint_path=None if args.no_checkpoint else str(checkpoint_run_dir / experiment_name),
            )

        if "ap" in args.metric_types:
            ap_rows.append(extract_ap_row(summary, args.sweep_param, val, args.classes, args.difficulties))

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
            detection_rate_rows.append(
                extract_detection_rate_row(summary, args.sweep_param, val, args.classes)
            )

        if "defense_effectiveness" in args.metric_types:
            defense_effectiveness_rows.append(extract_defense_effectiveness_row(summary, args.sweep_param, val))

        if "clustering_quality" in args.metric_types:
            clustering_quality_rows.append(extract_clustering_quality_row(summary, args.sweep_param, val))

        if "pacts_effectiveness" in args.metric_types:
            pacts_effectiveness_rows.append(extract_pacts_effectiveness_row(summary, args.sweep_param, val))

        if "llm_attack_type_accuracy" in args.metric_types:
            llm_attack_type_accuracy_rows.append(extract_llm_attack_type_accuracy_row(summary, args.sweep_param, val))

        if "llm_cost_metrics" in args.metric_types:
            llm_cost_metrics_rows.append(extract_llm_cost_metrics_row(summary, args.sweep_param, val))

        if "timing_metrics" in args.metric_types:
            timing_metrics_rows.append(extract_timing_metrics_row(summary, args.sweep_param, val))

        log_summary_metrics(summary, args.metric_types, args.classes, args.difficulties)

    sweep_tag = f"{args.sweep_target}_{args.sweep_param}"

    if "ap" in args.metric_types and ap_rows:
        write_ap_csv(run_dir, sweep_tag, args.sweep_param, ap_rows, args.classes, args.difficulties)

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

    if "detection_rate" in args.metric_types and detection_rate_rows:
        write_detection_rate_csv(run_dir, sweep_tag, args.sweep_param, detection_rate_rows, args.classes)

    if "defense_effectiveness" in args.metric_types and defense_effectiveness_rows:
        write_defense_effectiveness_csv(run_dir, sweep_tag, args.sweep_param, defense_effectiveness_rows)

    if "clustering_quality" in args.metric_types and clustering_quality_rows:
        write_clustering_quality_csv(run_dir, sweep_tag, args.sweep_param, clustering_quality_rows)

    if "pacts_effectiveness" in args.metric_types and pacts_effectiveness_rows:
        write_pacts_effectiveness_csv(run_dir, sweep_tag, args.sweep_param, pacts_effectiveness_rows)

    if "llm_attack_type_accuracy" in args.metric_types and llm_attack_type_accuracy_rows:
        write_llm_attack_type_accuracy_csv(run_dir, sweep_tag, args.sweep_param, llm_attack_type_accuracy_rows)

    if "llm_cost_metrics" in args.metric_types and llm_cost_metrics_rows:
        write_llm_cost_metrics_csv(run_dir, sweep_tag, args.sweep_param, llm_cost_metrics_rows)

    if "timing_metrics" in args.metric_types and timing_metrics_rows:
        write_timing_metrics_csv(run_dir, sweep_tag, args.sweep_param, timing_metrics_rows)

    end_time = datetime.now()
    logging.info(f"Start time: {start_time}")
    logging.info(f"End time:   {end_time}")
    logging.info(f"Total time: {end_time - start_time}")


if __name__ == "__main__":
    main()
