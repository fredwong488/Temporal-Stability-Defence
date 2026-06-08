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
        dataset_type  : Dataset backend: "kitti" (default) or "nuscenes".
        dataset_params: Keyword arguments forwarded to the dataset constructor.
                        NuScenes example: {"root": "data/datasets/nuscenes", "version": "v1.0-mini"}

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
        use_predicted_labels          : When True, the attack receives clean detector
                                        predictions as labels rather than frame.labels.
                                        Use for datasets where not every frame is annotated
                                        (e.g. NuScenes at 10 Hz) so the attack fires on
                                        every frame. Requires a detector.
        pred_label_score_threshold    : Minimum detection score for a prediction to be
                                        used as an attack label when use_predicted_labels
                                        is True. Default: 0.5.
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
    dataset_params:
        root: data/datasets/KITTI
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
    dataset_type: str = "kitti"                 # "kitti" | "nuscenes"
    dataset_params: dict = dataclasses.field(default_factory=dict)
    # kitti_root: str = "data/datasets/KITTI"     # backward compat; prefer dataset_params["root"]
    # frame_ids: list[str] | None = None          # backward compat; prefer dataset_params["frame_ids"]

    # Attack
    attack_type: str | None = None              # "ora" | None
    attack_params: dict = dataclasses.field(default_factory=dict)
    attack_fraction: float = 1.0                # fraction of scenes/frames to attack (0.0–1.0)
    attack_fraction_seed: int = 0               # RNG seed for attack sampling
    # Scene-aware attack prefix (only used when dataset granularity == "scene", e.g. NuScenes).
    # A random number of frames in [min_unattacked_frames, scene_length - min_attacked_frames]
    # are left unattacked at the start of each attacked scene.  Scenes that cannot satisfy
    # both minima revert to fully unattacked.
    min_unattacked_frames: int = 0
    min_attacked_frames: int = 1

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

    # Precomputed cache
    # Path to a shelve-backed cache of FrameCacheEntry objects.
    # - Absent path → pipeline runs live and generates a new cache there.
    # - Present path + read_only_cache=True (default) → read-only replay; safe
    #   for concurrent Optuna trials.
    # - Present path + read_only_cache=False → writable; cache hits are replayed
    #   and misses are run live and written back (resume a crashed generation).
    precomputed_cache_path: str | None = None
    read_only_cache: bool = True
    # If True, cached attacked_predictions and attack_metadata are used directly
    # and the attack is not re-applied.  If False (default), the attack is re-run
    # live for each flagged frame and the detector is re-run on the new lidar.
    use_cached_attacks: bool = False

    # When True, the attack receives clean detector predictions as labels rather
    # than frame.labels.  Use for datasets where not every frame is annotated
    # (e.g. NuScenes at 10 Hz) so the attack fires on every frame.
    use_predicted_labels: bool = False
    pred_label_score_threshold: float = 0.5

    # Resumeable checkpointing (scene-granularity + synchronous defenses only).
    # Path stem for checkpoint sidecars (<stem>.ckpt.pkl, <stem>.frames.pkl).
    # Set by run_sweep; None disables checkpointing.
    checkpoint_path: str | None = None

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
