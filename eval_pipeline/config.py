"""
config.py
---------
ExperimentConfig — dataclass-based configuration for pipeline experiments.
"""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass
class ExperimentConfig:
    """Configuration for a single evaluation experiment.

    Designed to be serialisable to/from YAML or plain dicts for
    reproducible experiment tracking.

    Fields
    ------
    Dataset:
        kitti_root  : Path to the root KITTI directory.
        frame_ids   : List of frame ID strings to process (e.g. ["000125", "000070"]).
                      None = all available frames in the dataset.

    Attack:
        attack_type   : Attack to apply, e.g. "ora". None = no attack.
        attack_params : Keyword arguments forwarded to the attack constructor.
                        ORA example: {"budget": 200, "target_types": ["Car"], "seed": 42}

    Detector:
        detector_type   : Detector to run, e.g. "pointpillars". None = no detection.
        detector_params : Keyword arguments forwarded to the detector constructor.
                          PointPillars example: {"config_path": "...", "checkpoint_path": "..."}

    Defense:
        defense_type   : Defense to run, e.g. "void_region". None = no defense.
        defense_params : Keyword arguments forwarded to the defense constructor.
                         VoidRegion example: {"roi_min": [4.5, -5.0], "roi_max": [30.0, 5.0]}

    Evaluation:
        iou_thresholds                : Per-class 3D IoU matching thresholds.
                                        Defaults: Car=0.7, Pedestrian=0.5, Cyclist=0.5.
        cache_clean_preds             : Cache clean detector predictions by frame ID to
                                        avoid re-running the detector across experiments.
        metric_types                  : List of metrics to compute. Options:
                                          "ap"         — Average Precision per class/difficulty
                                          "pr"         — Precision-Recall curves per class/difficulty
                                          "recall_iou" — Recall vs IoU threshold curves per class
                                        Default: ["ap"]
        difficulties                  : KITTI difficulty levels to evaluate for AP and PR curves.
                                        Options: "Easy", "Moderate", "Hard". Default: all three.
        recall_iou_confidence_threshold: Confidence score threshold applied when computing
                                        recall-vs-IoU curves. Default: 0.3.

    Output:
        output_dir      : Directory where per-experiment JSON results are written.
        experiment_name : Filename stem for the saved JSON (e.g. "ora_budget_200").

    Example YAML
    ------------
    kitti_root: data/datasets/KITTI
    frame_ids: ["000125", "000070", "002612"]
    attack_type: ora
    attack_params:
      budget: 200
      target_types: ["Car"]
      seed: 42
    defense_type: void_region
    defense_params:
      roi_min: [4.5, -5.0]
      roi_max: [30.0, 5.0]
    metric_types: ["ap", "pr", "recall_iou"]
    difficulties: ["Easy", "Moderate", "Hard"]
    recall_iou_confidence_threshold: 0.3
    output_dir: results
    experiment_name: ora_200pt_void_region
    """

    # Dataset
    kitti_root: str = "data/datasets/KITTI"
    frame_ids: list[str] | None = None          # None = all available frames

    # Attack
    attack_type: str | None = None              # "ora" | None
    attack_params: dict = dataclasses.field(default_factory=dict)
    attack_fraction: float = 1.0                # fraction of frames to attack (0.0–1.0)
    attack_fraction_seed: int = 0               # RNG seed for frame sampling

    # Detector
    detector_type: str | None = None            # "pointpillars" | None
    detector_params: dict = dataclasses.field(default_factory=dict)

    # Defense
    defense_type: str | None = None             # "void_region" | None
    defense_params: dict = dataclasses.field(default_factory=dict)

    # Evaluation
    iou_thresholds: dict = dataclasses.field(
        default_factory=lambda: {"Car": 0.7, "Pedestrian": 0.5, "Cyclist": 0.5}
    )
    cache_clean_preds: bool = True
    metric_types: list = dataclasses.field(default_factory=lambda: ["ap"])
    difficulties: list = dataclasses.field(default_factory=lambda: ["Easy", "Moderate", "Hard"])
    recall_iou_confidence_threshold: float = 0.3

    # Output
    output_dir: str = "results"
    experiment_name: str = "default"
    save_frame_results: bool = False   # write per-frame JSONL alongside results JSON

    # ---------------------------------------------------------------------------
    # Constructors
    # ---------------------------------------------------------------------------

    @classmethod
    def from_dict(cls, d: dict) -> ExperimentConfig:
        """Build from a plain dict, ignoring unknown keys."""
        valid = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in valid})

    @classmethod
    def from_yaml(cls, path: str) -> ExperimentConfig:
        """Load from a YAML file (requires PyYAML)."""
        try:
            import yaml
        except ImportError as e:
            raise ImportError("PyYAML is required for from_yaml(). pip install pyyaml") from e

        with open(path) as f:
            return cls.from_dict(yaml.safe_load(f))

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)
