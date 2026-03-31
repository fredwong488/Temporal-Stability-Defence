"""
eval_pipeline
-------------
Adversarial LiDAR perception evaluation pipeline.

Quick start
-----------
    from eval_pipeline import EvalPipeline, KittiObjectDataset
    from eval_pipeline.attacks import ORAAttack
    from eval_pipeline.defenses import VoidRegionDefense

    dataset = KittiObjectDataset("data/datasets/KITTI", frame_ids=["000125"])
    pipeline = EvalPipeline(
        dataset,
        attack=ORAAttack(budget=200, seed=42),
        defense=VoidRegionDefense(),
    )
    results = pipeline.run()
    print(results.defense_effectiveness())
"""

from .base import BaseAttack, BaseDefense, BaseDetector
from .datasets import KittiObjectDataset
from .pipeline import EvalPipeline
from .runner import run_experiment
from .types import (
    Calibration,
    DetectionResult,
    EvalResults,
    Frame,
    FrameResult,
    ObjectLabel,
    Prediction,
)

__all__ = [
    # Types
    "Frame",
    "ObjectLabel",
    "Calibration",
    "Prediction",
    "DetectionResult",
    "FrameResult",
    "EvalResults",
    # Bases
    "BaseAttack",
    "BaseDetector",
    "BaseDefense",
    # Pipeline
    "EvalPipeline",
    "KittiObjectDataset",
    "run_experiment",
]
