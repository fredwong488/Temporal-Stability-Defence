"""
scripts/optuna_search.py
------------------------
Multi-objective Optuna (MOTPE) search for radial-jitter defense clustering params.

Optimises a Pareto front of (spoofed_f1, pred_f1) — the clustering-quality
metric defined in eval_pipeline/metrics/common.py.  The first trial builds a
shared precomputed cache (detector + ORA pass over the dataset); all subsequent
trials replay from that cache so only the defense re-runs per trial.

Edit _suggest_params_clustering() or _suggest_params_defense to change which params are searched and over what ranges.
Fixed (non-searched) defense params can be passed with --defense-params KEY=VALUE.

Usage:
    pixi run python scripts/optuna_search.py \\
        --defense radial_jitter \\
        --attack ora \\
        --detector pointpillars_nuscenes \\
        --dataset nuscenes \\
        --n-trials 100 \\
        --use-predicted-labels \\
        --use-cached-attacks

Outputs under results/optuna/<study_name>/:
    search_metadata.json  — config snapshot
    <study_name>.db       — SQLite study (resume with --study-name)
    trials.csv            — all trials
    pareto.csv            — Pareto-optimal configs
    trial_NNNN.json       — per-trial summary JSON (from run_experiment)
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import pathlib
import subprocess
import sys
from datetime import datetime

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import optuna

from eval_pipeline.config import ExperimentConfig
from eval_pipeline.runner import run_experiment

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
optuna.logging.set_verbosity(optuna.logging.WARNING)

_DATASETS_BASE = "/vol/bitbucket/cyw122/FYP/experiment_pipeline/data/datasets"
DEFAULT_NUSCENES_ROOT = f"{_DATASETS_BASE}/nuscenes-v1.0-mini"
DEFAULT_NUSCENES_VERSION = "v1.0-mini"
DEFAULT_NUSCENES_SPLIT = "mini_val"
DEFAULT_RESULTS_DIR = "results/optuna"

# ---------------------------------------------------------------------------
# Search space
#
# cluster_on_bev is a conditional categorical: when True, dbscan_eps is
# suggested under the name "dbscan_eps_bev"; when False,
# under "dbscan_eps_3d".  Separate names keep the NSGA-II
# crossover histories clean so BEV-optimal eps values never contaminate
# 3D-optimal ones.  For HDBSCAN, cluster_on_bev has no dependent parameter
# and is suggested as a plain categorical.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_kv_params(pairs: list[str]) -> dict:
    out: dict = {}
    for item in pairs:
        if "=" not in item:
            raise ValueError(f"Invalid parameter '{item}': expected KEY=VALUE format")
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


def _suggest_params_clustering(trial: optuna.Trial, clusterer: str) -> dict:
    """Search space for clustering_quality objective (spoofed_f1 + pred_f1) for Phase 1."""
    cluster_on_bev = trial.suggest_categorical("cluster_on_bev", [True, False])
    params: dict = {
        "cluster_on_bev":        cluster_on_bev,
        "dbscan_min_samples":    trial.suggest_int  ("dbscan_min_samples",    3,   20),
        "temporal_window":       trial.suggest_int  ("temporal_window",        6,   14),
        "motion_tolerance":      trial.suggest_float("motion_tolerance",      0.3,  3.0),
        "min_frames_associated": trial.suggest_int  ("min_frames_associated",  2,    6),
        "centroid_method":       trial.suggest_categorical(
                                     "centroid_method", ["linear_velocity", "first_diff"]),
    }
    if clusterer == "dbscan":
        param_name = "dbscan_eps_bev" if cluster_on_bev else "dbscan_eps_3d"
        params["dbscan_eps"] = trial.suggest_float(param_name, 0.2, 2.0)
    else:  # hdbscan
        params["hdbscan_min_cluster_size"] = trial.suggest_int(
            "hdbscan_min_cluster_size", 5, 30,
        )
    if trial.number == 0:
        logging.info("Search space (trial 0): %s", trial.distributions)
    return params


def _suggest_params_defense(trial: optuna.Trial, base_defense_params: dict) -> dict:
    """Search space for defense_effectiveness (Phase 2): thresholds and flag_condition only.

    Clustering params are assumed fixed via --defense-params from a Phase 1 clustering run.
    """
    use_point    = base_defense_params.get("use_point",    True)
    use_centroid = base_defense_params.get("use_centroid", True)
    force_or     = (not use_point) or (not use_centroid)

    params: dict = {
        "flag_condition": "or" if force_or else trial.suggest_categorical("flag_condition", ["and", "or"]),
    }
    if use_centroid:
        params["centroid_threshold"] = trial.suggest_float("centroid_threshold", 0.1, 1.0)
    if use_point:
        params["point_threshold"] = trial.suggest_float("point_threshold", 0.03, 0.25)
    if trial.number == 0:
        logging.info("Search space (trial 0): %s", trial.distributions)
    return params


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------

def build_objective(
    base_defense_params: dict,
    clusterer: str,
    attack_type: str | None,
    attack_params: dict,
    attack_fraction: float,
    attack_fraction_seed: int,
    defense_type: str,
    detector_type: str | None,
    detector_params: dict,
    dataset_type: str,
    dataset_params: dict,
    run_dir: pathlib.Path,
    shared_cache_path: str,
    use_cached_attacks: bool,
    use_predicted_labels: bool,
    pred_label_score_threshold: float,
    min_unattacked_frames: int,
    min_attacked_frames: int,
    objective_mode: str,
):
    def objective(trial: optuna.Trial):
        if objective_mode == "defense_effectiveness":
            trial_params = _suggest_params_defense(trial, base_defense_params)
        else:
            trial_params = _suggest_params_clustering(trial, clusterer)
        defense_params = {**base_defense_params, **trial_params}

        config = ExperimentConfig(
            dataset_type=dataset_type,
            dataset_params=dataset_params,
            attack_type=attack_type,
            attack_params=attack_params,
            attack_fraction=attack_fraction,
            attack_fraction_seed=attack_fraction_seed,
            defense_type=defense_type,
            defense_params=defense_params,
            detector_type=detector_type,
            detector_params=detector_params,
            output_dir=str(run_dir),
            experiment_name=f"trial_{trial.number:04d}",
            metric_types=["clustering_quality"],
            save_frame_results=False,
            precomputed_cache_path=shared_cache_path,
            use_cached_attacks=use_cached_attacks,
            use_predicted_labels=use_predicted_labels,
            pred_label_score_threshold=pred_label_score_threshold,
            min_unattacked_frames=min_unattacked_frames,
            min_attacked_frames=min_attacked_frames,
        )
        summary = run_experiment(config, desc=f"trial {trial.number}")

        if objective_mode == "defense_effectiveness":
            de = summary.get("defense_effectiveness") or {}
            f1 = float(de.get("f1") or 0.0)
            logging.info(
                "Trial %d  defense_f1=%.4f  params=%s",
                trial.number, f1,
                {k: f"{v:.3f}" if isinstance(v, float) else v for k, v in trial_params.items()},
            )
            return f1
        else:
            cq = summary.get("clustering_quality") or {}
            spoofed_f1 = float(cq.get("spoofed_f1") or 0.0)
            pred_f1 = float(cq.get("pred_f1") or 0.0)
            logging.info(
                "Trial %d  spoofed_f1=%.4f  pred_f1=%.4f  params=%s",
                trial.number, spoofed_f1, pred_f1,
                {k: f"{v:.3f}" if isinstance(v, float) else v for k, v in trial_params.items()},
            )
            return spoofed_f1, pred_f1

    return objective


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _save_pareto(pareto_trials: list[optuna.trial.FrozenTrial], path: pathlib.Path) -> None:
    if not pareto_trials:
        return
    rows = []
    for t in pareto_trials:
        row: dict = {"number": t.number, "spoofed_f1": t.values[0], "pred_f1": t.values[1]}
        row.update(t.params)
        rows.append(row)

    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _print_pareto(pareto_trials: list[optuna.trial.FrozenTrial]) -> None:
    if not pareto_trials:
        logging.warning("No Pareto-front trials found.")
        return

    param_keys = sorted({k for t in pareto_trials for k in t.params})
    header = f"{'#':>6}  {'spoofed_f1':>10}  {'pred_f1':>10}"
    for k in param_keys:
        header += f"  {k}"
    print(f"\n=== Pareto front ({len(pareto_trials)} configs) ===")
    print(header)
    for t in pareto_trials:
        line = f"{t.number:>6}  {t.values[0]:>10.4f}  {t.values[1]:>10.4f}"
        for k in param_keys:
            v = t.params.get(k, "?")
            line += f"  {v:.4f}" if isinstance(v, float) else f"  {v}"
        print(line)


def _save_best(best_trial: optuna.trial.FrozenTrial, path: pathlib.Path) -> None:
    row: dict = {"number": best_trial.number, "defense_f1": best_trial.value}
    row.update(best_trial.params)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()), extrasaction="ignore")
        writer.writeheader()
        writer.writerow(row)


def _print_best(best_trial: optuna.trial.FrozenTrial) -> None:
    param_keys = sorted(best_trial.params)
    print(f"\n=== Best trial (#{best_trial.number})  defense_f1={best_trial.value:.4f} ===")
    for k in param_keys:
        v = best_trial.params[k]
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-objective Optuna search for radial-jitter clustering params.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Dataset
    parser.add_argument("--dataset", default="nuscenes", choices=["kitti", "nuscenes"])
    parser.add_argument("--nuscenes-root", default=DEFAULT_NUSCENES_ROOT)
    parser.add_argument("--nuscenes-version", default=DEFAULT_NUSCENES_VERSION)
    parser.add_argument("--nuscenes-split", default=DEFAULT_NUSCENES_SPLIT)
    parser.add_argument("--nuscenes-scene-names", nargs="+", default=None, metavar="SCENE")

    # Components
    parser.add_argument("--defense", required=True,
                        help="Defense type to tune (e.g. radial_jitter)")
    parser.add_argument("--attack", default="ora")
    parser.add_argument("--detector", default=None)
    parser.add_argument("--attack-noise-preset", default="worst_case",
                        choices=["none", "worst_case", "worst_case_high_error",
                                 "vlp16", "vlp32c", "os1_32", "helios", "horizon", "l515", "xt32"])
    parser.add_argument("--attack-fraction", type=float, default=1.0)
    parser.add_argument("--attack-fraction-seed", type=int, default=0)
    parser.add_argument("--min-unattacked-frames", type=int, default=6)
    parser.add_argument("--min-attacked-frames", type=int, default=6)
    parser.add_argument("--use-cached-attacks", action="store_true", default=False)
    parser.add_argument("--use-predicted-labels", action="store_true", default=False)
    parser.add_argument("--pred-label-score-threshold", type=float, default=0.5)
    parser.add_argument("--confidence-threshold", type=float, default=0.3)

    # Fixed (non-searched) defense params
    parser.add_argument("--defense-params", nargs="*", default=[], metavar="KEY=VALUE",
                        help="Fixed defense params not in the search space (e.g. use_point=False)")

    # Optuna
    parser.add_argument("--objective", default="clustering_quality",
                        choices=["clustering_quality", "defense_effectiveness"],
                        help="Metric to optimise: clustering_quality (multi-obj Pareto) or "
                             "defense_effectiveness (single-obj F1)")
    parser.add_argument("--n-trials", type=int, default=100)
    parser.add_argument("--study-name", default=None,
                        help="Optuna study name; pass an existing name with --results-dir to resume")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--population-size", type=int, default=50,
                        help="NSGA-II population size (clustering_quality only)")
    parser.add_argument("--tpe-startup-trials", type=int, default=10,
                        help="Random startup trials before TPE begins (defense_effectiveness only). "
                             "Default 10; reduce for small search spaces (e.g. 5 for Phase 2).")

    # Output
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    parser.add_argument(
        "--precomputed-cache-dir", default=None, metavar="DIR",
        help="Directory for the shared precomputed cache. Uses defense_sweep_shared.pkl "
             "inside that directory. Defaults to run_dir/shared_cache.pkl if not set.",
    )
    parser.add_argument("--notes", default=None)

    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    study_name = args.study_name or f"{args.defense}_{timestamp}"
    run_dir = pathlib.Path(args.results_dir) / study_name
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.precomputed_cache_dir is not None:
        shared_cache_path = str(
            pathlib.Path(args.precomputed_cache_dir) / "defense_sweep_shared.pkl"
        )
    else:
        shared_cache_path = str(run_dir / "shared_cache.pkl")

    # Dataset params
    dataset_type = args.dataset
    dataset_params: dict = {}
    if dataset_type == "nuscenes":
        dataset_params.update({
            "root": args.nuscenes_root,
            "version": args.nuscenes_version,
            "split": args.nuscenes_split,
        })
        if args.nuscenes_scene_names:
            dataset_params["scene_names"] = args.nuscenes_scene_names
        classes = ["car", "pedestrian", "bicycle"]
    else:
        classes = ["Car", "Pedestrian", "Cyclist"]

    # Attack params
    attack_params: dict = {"target_types": classes}
    if args.attack == "ora" and args.attack_noise_preset != "none":
        from eval_pipeline.utils.spoofing_noise import SpoofingNoiseModel
        attack_params["noise_model"] = SpoofingNoiseModel.from_preset(
            args.attack_noise_preset, seed=args.attack_fraction_seed
        )

    base_defense_params = _parse_kv_params(args.defense_params)
    clusterer = base_defense_params.get("clusterer")
    if clusterer not in ("dbscan", "hdbscan"):
        raise ValueError(
            "clusterer must be provided via --defense-params (e.g. clusterer=dbscan or "
            "clusterer=hdbscan). It is required to select the correct search space: "
            "DBSCAN searches dbscan_eps; HDBSCAN searches hdbscan_min_cluster_size instead."
        )

    logging.info("Clusterer: %s", clusterer)
    detector_params: dict = {}
    if args.detector:
        detector_params["score_threshold"] = args.confidence_threshold

    # Optuna study
    storage = f"sqlite:///{run_dir / (study_name + '.db')}"
    if args.objective == "defense_effectiveness":
        sampler = optuna.samplers.TPESampler(seed=args.seed, n_startup_trials=args.tpe_startup_trials)
        study = optuna.create_study(
            direction="maximize",
            study_name=study_name,
            storage=storage,
            load_if_exists=True,
            sampler=sampler,
        )
    else:
        sampler = optuna.samplers.NSGAIISampler(
            seed=args.seed,
            population_size=args.population_size,
        )
        study = optuna.create_study(
            directions=["maximize", "maximize"],
            study_name=study_name,
            storage=storage,
            load_if_exists=True,
            sampler=sampler,
        )

    # Resume guard: if an existing study is being continued, check for config drift.
    metadata_path = run_dir / "search_metadata.json"
    if args.study_name is not None and metadata_path.exists():
        with open(metadata_path) as f:
            saved = json.load(f)
        _COMPARE_KEYS = ("defense", "attack", "dataset", "clusterer",
                         "attack_noise_preset", "use_predicted_labels")
        current = {
            "defense": args.defense,
            "attack": args.attack,
            "dataset": dataset_type,
            "clusterer": clusterer,
            "attack_noise_preset": args.attack_noise_preset,
            "use_predicted_labels": args.use_predicted_labels,
        }
        diffs = {k: (saved.get(k), current[k]) for k in _COMPARE_KEYS if saved.get(k) != current[k]}
        if diffs:
            print("\nWARNING: Current arguments differ from the saved study metadata:")
            for k, (old, new) in diffs.items():
                print(f"  {k}: saved={old!r}  current={new!r}")
            answer = input("\nContinue anyway? [y/N] ").strip().lower()
            if answer != "y":
                print("Aborted.")
                return

    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_PROJECT_ROOT, text=True
        ).strip()
    except subprocess.CalledProcessError:
        git_hash = "unknown"

    # Metadata snapshot (written / overwritten each run so it reflects latest n_trials target)
    metadata = {
        "study_name": study_name,
        "timestamp": timestamp,
        "git_commit": git_hash,
        "cmd_args": sys.argv,
        "notes": args.notes,
        "defense": args.defense,
        "attack": args.attack,
        "dataset": dataset_type,
        "dataset_params": dataset_params,
        "n_trials": args.n_trials,
        "seed": args.seed,
        "population_size": args.population_size,
        "clusterer": clusterer,
        "base_defense_params": {k: str(v) for k, v in base_defense_params.items()},
        "attack_noise_preset": args.attack_noise_preset,
        "use_cached_attacks": args.use_cached_attacks,
        "use_predicted_labels": args.use_predicted_labels,
    }
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logging.info("Study: %s", study_name)
    logging.info("Storage: %s", storage)
    logging.info("Cache: %s", shared_cache_path)

    # Work out how many trials remain to reach the target.
    completed = sum(
        1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
    )
    remaining = max(0, args.n_trials - completed)
    if remaining == 0:
        logging.info(
            "Study already has %d completed trials (target: %d). Nothing to do.",
            completed, args.n_trials,
        )
    else:
        logging.info(
            "Completed: %d / %d — running %d more.", completed, args.n_trials, remaining,
        )

    objective = build_objective(
        base_defense_params=base_defense_params,
        clusterer=clusterer,
        attack_type=args.attack,
        attack_params=attack_params,
        attack_fraction=args.attack_fraction,
        attack_fraction_seed=args.attack_fraction_seed,
        defense_type=args.defense,
        detector_type=args.detector,
        detector_params=detector_params,
        dataset_type=dataset_type,
        dataset_params=dataset_params,
        run_dir=run_dir,
        shared_cache_path=shared_cache_path,
        use_cached_attacks=args.use_cached_attacks,
        use_predicted_labels=args.use_predicted_labels,
        pred_label_score_threshold=args.pred_label_score_threshold,
        min_unattacked_frames=args.min_unattacked_frames,
        min_attacked_frames=args.min_attacked_frames,
        objective_mode=args.objective,
    )

    study.optimize(objective, n_trials=remaining, show_progress_bar=True)

    # Save all trials
    trials_path = run_dir / "trials.csv"
    study.trials_dataframe().to_csv(trials_path, index=False)
    logging.info("All trials → %s", trials_path)

    # Save best result(s)
    if args.objective == "defense_effectiveness":
        best_path = run_dir / "best.csv"
        _save_best(study.best_trial, best_path)
        logging.info("Best trial → %s", best_path)
        _print_best(study.best_trial)
    else:
        pareto_path = run_dir / "pareto.csv"
        _save_pareto(study.best_trials, pareto_path)
        logging.info("Pareto front (%d configs) → %s", len(study.best_trials), pareto_path)
        _print_pareto(study.best_trials)

    print(f"\nResults: {run_dir}")
    print(f"Resume:  pixi run python scripts/optuna_search.py --study-name {study_name} "
          f"--results-dir {args.results_dir} --n-trials <total_target>")


if __name__ == "__main__":
    main()
