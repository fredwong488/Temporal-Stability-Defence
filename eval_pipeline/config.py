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
    output_dir: results
    experiment_name: ora_200pt_void_region
    """

    # Dataset
    kitti_root: str = "data/datasets/KITTI"
    frame_ids: list[str] | None = None          # None = all available frames

    # Attack
    attack_type: str | None = None              # "ora" | None
    attack_params: dict = dataclasses.field(default_factory=dict)

    # Detector
    detector_type: str | None = None            # "pointpillars" | None
    detector_params: dict = dataclasses.field(default_factory=dict)

    # Defense
    defense_type: str | None = None             # "void_region" | None
    defense_params: dict = dataclasses.field(default_factory=dict)

    # Evaluation
    iou_threshold: float = 0.5
    cache_clean_preds: bool = True

    # Output
    output_dir: str = "results"
    experiment_name: str = "default"

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
