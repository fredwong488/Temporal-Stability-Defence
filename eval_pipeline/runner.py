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
        kitti_root="data/datasets/KITTI",
        frame_ids=["000125"],
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

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Component registries — extend these to add new attacks / detectors / defenses
# ---------------------------------------------------------------------------

def _attack_registry() -> dict[str, type]:
    from .attacks.ora import ORAAttack
    return {"ora": ORAAttack}


def _detector_registry() -> dict[str, type]:
    from .detectors.pointpillars import PointPillarsDetector
    from .detectors.pointrcnn import PointRCNNDetector
    return {"pointpillars": PointPillarsDetector, "pointrcnn": PointRCNNDetector}


def _defense_registry() -> dict[str, type]:
    from .defenses.void_region import VoidRegionDefense
    return {"void_region": VoidRegionDefense}


# ---------------------------------------------------------------------------
# Pipeline builder
# ---------------------------------------------------------------------------

def build_pipeline(config: ExperimentConfig, desc: str = "Frames") -> EvalPipeline:
    """Instantiate all pipeline components from a config."""
    from .datasets.kitti import KittiObjectDataset

    dataset = KittiObjectDataset(
        root=config.kitti_root,
        frame_ids=config.frame_ids,
    )

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
        desc=desc,
    )


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

def run_experiment(config: ExperimentConfig) -> dict:
    """Run a full experiment and return a JSON-serialisable results dict.

    Which metrics are computed is controlled by config.metric_types:
      "ap"         — Attack effectiveness (clean vs attacked mAP per class/difficulty)
      "pr"         — Precision-Recall curves per class and difficulty
      "recall_iou" — Recall vs IoU threshold curves per class
    """
    desc = f"Budget {config.attack_params.get('budget', '?')}" if config.attack_type else "Frames"
    pipeline = build_pipeline(config, desc=desc)
    eval_results = pipeline.run()

    summary: dict = {
        "experiment_name": config.experiment_name,
        "config": config.to_dict(),
        "num_frames": len(eval_results.frame_results),
    }

    if config.detector_type:
        metric_types = set(config.metric_types)
        frame_results = eval_results.frame_results

        if "ap" in metric_types:
            summary["attack_effectiveness"] = eval_results.attack_effectiveness(
                iou_thresholds=config.iou_thresholds
            )

        if "pr" in metric_types:
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

        if "recall_iou" in metric_types:
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

    if config.defense_type:
        summary["defense_effectiveness"] = eval_results.defense_effectiveness()

    # Optionally save results to disk
    if config.output_dir:
        out_dir = pathlib.Path(config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{config.experiment_name}.json"
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        logger.info("Results saved to %s", out_path)

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
    parser.add_argument("--kitti-root", type=str, default=None)
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
    args = parser.parse_args()

    # Base config from YAML or defaults
    if args.config:
        config = ExperimentConfig.from_yaml(args.config)
    else:
        config = ExperimentConfig()

    # CLI overrides
    overrides: dict = {}
    if args.kitti_root:
        overrides["kitti_root"] = args.kitti_root
    if args.attack:
        overrides["attack_type"] = args.attack
    if args.detector:
        overrides["detector_type"] = args.detector
    if args.defense:
        overrides["defense_type"] = args.defense
    if args.frames:
        overrides["frame_ids"] = args.frames
    if args.experiment_name != "default":
        overrides["experiment_name"] = args.experiment_name
    if args.output_dir != "results":
        overrides["output_dir"] = args.output_dir
    if args.budget is not None:
        overrides["attack_params"] = {**config.attack_params, "budget": args.budget}

    if overrides:
        config = dataclasses.replace(config, **overrides)

    summary = run_experiment(config)
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
