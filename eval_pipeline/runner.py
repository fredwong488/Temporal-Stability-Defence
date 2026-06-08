"""
runner.py
---------
CLI entry point and programmatic experiment runner.

Usage (CLI)
-----------
    python -m eval_pipeline.runner --config experiment.yaml
    python -m eval_pipeline.runner --kitti-root data/datasets/KITTI \\
        --attack ora --defense void_region --frames 000125 000070

Usage (programmatic)
--------------------
    from eval_pipeline.runner import run_experiment
    from eval_pipeline.config import ExperimentConfig

    config = ExperimentConfig(
        attack_type="ora",
        attack_params={"budget": 200, "seed": 42},
        defense_type="void_region",
    )
    summary = run_experiment(config)
    print(summary["defense_effectiveness"])
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import pathlib

from .config import ExperimentConfig
from .pipeline import EvalPipeline
from .types import FrameResult, Prediction

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-frame serialisation helpers
# ---------------------------------------------------------------------------

def _serialise_prediction(p: Prediction) -> dict:
    return {
        "type": p.type,
        "score": p.score,
        "x": p.x, "y": p.y, "z": p.z,
        "height": p.height, "width": p.width, "length": p.length,
        "rotation_y": p.rotation_y,
        "corners_velo": p.corners_velo.tolist(),
    }


def _serialise_frame_result(fr: FrameResult) -> dict:
    result: dict = {
        "frame_id": fr.frame_id,
        "sequence_id": fr.sequence_id,
        "frame_index_in_scene": fr.frame_index_in_scene,
        "scene_length": fr.scene_length,
        "attack_start_index": fr.attack_start_index,
        "attack_start_frame_id": fr.attack_start_frame_id,
        "is_attacked": fr.is_attacked,
        "attack_metadata": fr.attack_metadata,
        "clean_predictions": [_serialise_prediction(p) for p in fr.clean_predictions],
        "attacked_predictions": (
            [_serialise_prediction(p) for p in fr.attacked_predictions]
            if fr.attacked_predictions is not None else None
        ),
    }
    if fr.defense_result is not None:
        result["defense_result"] = {
            "is_attack_detected": fr.defense_result.is_attack_detected,
            "confidence": fr.defense_result.confidence,
            "metadata": fr.defense_result.metadata,
        }
    else:
        result["defense_result"] = None
    return result


# ---------------------------------------------------------------------------
# Component registries — extend these to add new attacks / detectors / defenses
# ---------------------------------------------------------------------------

def _attack_registry() -> dict[str, type]:
    from .attacks.ora import ORAAttack
    from .attacks.ghost_object.ghost_object import GhostObjectAttack
    return {
        "ora": ORAAttack,
        "ghost": GhostObjectAttack
    }


def _detector_registry() -> dict[str, type]:
    from .detectors.pointpillars import PointPillarsDetector
    from .detectors.pointpillars_nuscenes import PointPillarsNuScenesDetector
    from .detectors.pointrcnn import PointRCNNDetector
    from .detectors.precomputed import PrecomputedDetector
    return {
        "pointpillars": PointPillarsDetector,
        "pointpillars_nuscenes": PointPillarsNuScenesDetector,
        "pointrcnn": PointRCNNDetector,
        "precomputed": PrecomputedDetector,
    }


def _defense_registry() -> dict[str, type]:
    from .defenses.void_region import VoidRegionDefense
    from .defenses.tc2 import TC2Defense
    from .defenses.fsd import FSDDefense
    from .defenses.carlo import CARLODefense
    import functools
    from .defenses.radial_jitter import RadialJitterDefense
    from .defenses.wasserstein_anisotropy import WassersteinAnisotropyDefense
    from .defenses.llm import LLMDefense
    from .defenses.bouhamidi import BouhamidiDefense
    return {
        "void_region": VoidRegionDefense,
        "tc2": TC2Defense,
        "fsd": FSDDefense,
        "carlo": CARLODefense,
        "radial_jitter": RadialJitterDefense,
        "radial_jitter_bev": functools.partial(RadialJitterDefense, cluster_on_bev=True),   # For backwards compatibility
        "radial_jitter_patchwork": functools.partial(RadialJitterDefense, ground_method="patchwork"),
        "wasserstein": WassersteinAnisotropyDefense,
        "llm": LLMDefense,
        "bouhamidi": BouhamidiDefense,
    }


def _dataset_registry() -> dict[str, type]:
    from .datasets.kitti import KittiObjectDataset
    from .datasets.nuscenes import NuScenesDataset
    return {"kitti": KittiObjectDataset, "nuscenes": NuScenesDataset}


# ---------------------------------------------------------------------------
# Pipeline builder
# ---------------------------------------------------------------------------

def build_pipeline(config: ExperimentConfig, desc: str = "Frames") -> EvalPipeline:
    """Instantiate all pipeline components from a config."""
    dataset_cls = _dataset_registry().get(config.dataset_type)
    if dataset_cls is None:
        raise ValueError(
            f"Unknown dataset_type '{config.dataset_type}'. "
            f"Available: {list(_dataset_registry())}"
        )
    dataset_params = dict(config.dataset_params)
    dataset = dataset_cls(**dataset_params)

    attack = None
    if config.attack_type:
        cls = _attack_registry().get(config.attack_type)
        if cls is None:
            raise ValueError(
                f"Unknown attack_type '{config.attack_type}'. "
                f"Available: {list(_attack_registry())}"
            )
        attack = cls(**config.attack_params)

    detector = None
    if config.detector_type:
        cls = _detector_registry().get(config.detector_type)
        if cls is None:
            raise ValueError(
                f"Unknown detector_type '{config.detector_type}'. "
                f"Available: {list(_detector_registry())}"
            )
        detector = cls(**config.detector_params)

    defense = None
    if config.defense_type:
        cls = _defense_registry().get(config.defense_type)
        if cls is None:
            raise ValueError(
                f"Unknown defense_type '{config.defense_type}'. "
                f"Available: {list(_defense_registry())}"
            )
        defense = cls(**config.defense_params)

    return EvalPipeline(
        dataset=dataset,
        attack=attack,
        detector=detector,
        defense=defense,
        cache_clean_preds=config.cache_clean_preds,
        attack_fraction=config.attack_fraction,
        attack_fraction_seed=config.attack_fraction_seed,
        desc=desc,
        precomputed_cache_path=config.precomputed_cache_path,
        read_only_cache=config.read_only_cache,
        use_cached_attacks=config.use_cached_attacks,
        use_predicted_labels=config.use_predicted_labels,
        pred_label_score_threshold=config.pred_label_score_threshold,
        min_unattacked_frames=config.min_unattacked_frames,
        min_attacked_frames=config.min_attacked_frames,
        checkpoint_path=config.checkpoint_path,
    )


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

def run_experiment(config: ExperimentConfig, desc: str | None = None) -> dict:  # noqa: C901
    """Run a full experiment and return a JSON-serialisable results dict.

    Which metrics are computed is controlled by config.metric_types:
      "ap"         — Attack effectiveness (clean vs attacked mAP per class/difficulty)
      "pr"         — Precision-Recall curves per class and difficulty
      "recall_iou" — Recall vs IoU threshold curves per class
    """
    if desc is None:
        desc = f"Budget {config.attack_params.get('budget', '?')}" if config.attack_type else "Frames"
    pipeline = build_pipeline(config, desc=desc)
    eval_results = pipeline.run()
    # Keep a reference so cleanup_checkpoint() can be called after saving results.

    summary: dict = {
        "experiment_name": config.experiment_name,
        "config": config.to_dict(),
        "num_frames": len(eval_results.frame_results),
    }

    if config.detector_type:
        metric_types = set(config.metric_types)
        frame_results = eval_results.frame_results

        # KITTI-specific metrics are not applicable to NuScenes (different class set,
        # no Easy/Moderate/Hard difficulty, no bbox_2d pixel-height filter).
        _kitti_metrics_available = config.dataset_type == "kitti"

        if "ap" in metric_types:
            if _kitti_metrics_available:
                summary["attack_effectiveness"] = eval_results.attack_effectiveness(
                    iou_thresholds=config.iou_thresholds
                )
            else:
                logger.info(
                    "AP metrics skipped for dataset_type='%s' (KITTI-specific). "
                    "Only defense_effectiveness is computed for NuScenes.",
                    config.dataset_type,
                )

        if "pr" in metric_types and _kitti_metrics_available:
            from itertools import product
            from tqdm import tqdm
            from .metrics import compute_pr_curve, _DEFAULT_IOU_THRESHOLDS  # noqa: PLC2701
            classes = sorted({lbl.type for fr in frame_results for lbl in fr.labels})
            pr_curves: dict = {}
            combos = list(product(classes, config.difficulties))
            for cls, diff in tqdm(combos, desc="PR curves", unit="curve"):
                pr_curves.setdefault(cls, {})[diff] = compute_pr_curve(
                    frame_results, cls,
                    _DEFAULT_IOU_THRESHOLDS.get(cls, 0.5),
                    use_clean=False, difficulty=diff,
                )
            summary["pr_curves"] = pr_curves

        if "recall_iou" in metric_types and _kitti_metrics_available:
            from .metrics import compute_recall_vs_iou
            classes = sorted({lbl.type for fr in frame_results for lbl in fr.labels})
            summary["recall_iou_curves"] = {
                cls: compute_recall_vs_iou(
                    frame_results, cls,
                    config.recall_iou_confidence_threshold,
                    use_clean=False,
                )
                for cls in classes
            }

        if "detection_rate" in metric_types:
            from .metrics import _NUSCENES_IOU_THRESHOLDS
            iou_thr = (
                _NUSCENES_IOU_THRESHOLDS
                if config.dataset_type == "nuscenes"
                else config.iou_thresholds
            )
            summary["detection_rate"] = eval_results.detection_rate(
                iou_thresholds=iou_thr
            )
            if summary["detection_rate"]:
                overall = summary["detection_rate"].get("overall", {})
                logger.info(
                    "Detection rate  clean=%.3f  attacked=%.3f  abs_drop=%.3f  rel_drop=%.1f%%",
                    overall.get("detection_rate_clean", float("nan")),
                    overall.get("detection_rate_attacked", float("nan")),
                    overall.get("absolute_drop", float("nan")),
                    overall.get("relative_drop", float("nan")) * 100,
                )
            else:
                logger.info(
                    "Detection-rate metric: no qualifying attacked frames with GT labels "
                    "(check that the dataset has annotations and the attack is applied)."
                )

    if config.defense_type:
        summary["defense_effectiveness"] = eval_results.defense_effectiveness()
        summary["clustering_quality"] = eval_results.clustering_quality()
        if config.defense_type == "radial_jitter":
            summary["pacts_effectiveness"] = eval_results.pacts_effectiveness()
        if config.defense_type == "llm":
            summary["llm_attack_type_accuracy"] = eval_results.llm_attack_type_accuracy()
            summary["llm_cost_metrics"] = eval_results.llm_cost_metrics()

    # Optionally save results to disk
    if config.output_dir:
        out_dir = pathlib.Path(config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{config.experiment_name}.json"
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        logger.info("Results saved to %s", out_path)

        if config.save_frame_results:
            frames_path = out_dir / f"{config.experiment_name}_frames.jsonl"
            with open(frames_path, "w") as f:
                for fr in eval_results.frame_results:
                    f.write(json.dumps(_serialise_frame_result(fr)) + "\n")
            logger.info("Per-frame results saved to %s", frames_path)

        # Results JSON is the authoritative completion marker — delete checkpoint
        # sidecars now that the experiment has finished successfully.
        pipeline.cleanup_checkpoint()

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        description="Adversarial LiDAR perception evaluation pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config file")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Dataset type (e.g. kitti, nuscenes)")
    parser.add_argument("--dataset-root", type=str, default=None,
                        help="Root path for the dataset (sets dataset_params['root'])")

    parser.add_argument("--attack", type=str, default=None,
                        help="Attack type (e.g. ora)")
    parser.add_argument("--detector", type=str, default=None,
                        help="Detector type (e.g. pointpillars)")
    parser.add_argument("--defense", type=str, default=None,
                        help="Defense type (e.g. void_region)")
    parser.add_argument("--frames", type=str, nargs="*", default=None,
                        help="Frame IDs to process (e.g. 000125 000070)")
    parser.add_argument("--budget", type=int, default=None,
                        help="ORA attack point budget (shortcut for --attack ora)")
    parser.add_argument("--experiment-name", type=str, default="default")
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--save-frames", action="store_true", default=False,
                        help="Save per-frame JSONL alongside results JSON for visualisation")
    parser.add_argument("--use-predicted-labels", action="store_true", default=False,
                        help="Use clean detector predictions as attack labels instead of "
                             "ground-truth annotations. Use for datasets where not every "
                             "frame is labeled (e.g. NuScenes at 10 Hz).")
    parser.add_argument("--pred-label-score-threshold", type=float, default=None,
                        help="Minimum detection score for a prediction to be used as an "
                             "attack label when --use-predicted-labels is set (default 0.5).")
    args = parser.parse_args()

    # Base config from YAML or defaults
    if args.config:
        config = ExperimentConfig.from_yaml(args.config)
    else:
        config = ExperimentConfig()

    # CLI overrides
    overrides: dict = {}
    if args.dataset:
        overrides["dataset_type"] = args.dataset
    if args.dataset_root:
        overrides["dataset_params"] = {**config.dataset_params, "root": args.dataset_root}
    if args.attack:
        overrides["attack_type"] = args.attack
    if args.detector:
        overrides["detector_type"] = args.detector
    if args.defense:
        overrides["defense_type"] = args.defense
    if args.frames:
        overrides["dataset_params"] = {**config.dataset_params, **overrides.get("dataset_params", {}), "frame_ids": args.frames}
    if args.experiment_name != "default":
        overrides["experiment_name"] = args.experiment_name
    if args.output_dir != "results":
        overrides["output_dir"] = args.output_dir
    if args.budget is not None:
        overrides["attack_params"] = {**config.attack_params, "budget": args.budget}
    if args.save_frames:
        overrides["save_frame_results"] = True
    if args.use_predicted_labels:
        overrides["use_predicted_labels"] = True
    if args.pred_label_score_threshold is not None:
        overrides["pred_label_score_threshold"] = args.pred_label_score_threshold

    if overrides:
        config = dataclasses.replace(config, **overrides)

    summary = run_experiment(config)
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
