"""
types.py
--------
Core dataclasses that flow through the evaluation pipeline.
"""

from __future__ import annotations

import dataclasses
from collections import deque
from typing import TYPE_CHECKING

import numpy as np


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class Calibration:
    """KITTI-style calibration matrices."""
    R0_rect: np.ndarray          # (3, 3) rectification matrix
    Tr_velo_to_cam: np.ndarray   # (3, 4) velodyne-to-camera transform
    P2: np.ndarray | None = None # (3, 4) projection matrix (optional)
    image_shape: tuple[int, int] | None = None  # (H, W) of the reference camera image


# ---------------------------------------------------------------------------
# ObjectLabel
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class ObjectLabel:
    """Ground-truth annotation for a single object.

    Coordinates (x, y, z) and corners_velo are in the velodyne/sensor frame.
    Dimensions (height, width, length) are in metres.

    truncated, occluded, alpha, bbox_2d are KITTI-specific fields and may be
    None when the label originates from a non-KITTI source (e.g. nuScenes
    annotations or detector predictions).  Code that requires these fields
    (e.g. KITTI difficulty filtering) must guard against None.
    """
    type: str                                         # "Car", "Pedestrian", "Cyclist", ...
    truncated: float | None
    occluded: int | None
    alpha: float | None
    bbox_2d: tuple[float, float, float, float] | None  # (x1, y1, x2, y2) image pixels
    height: float
    width: float
    length: float
    x: float                                          # sensor-frame 3D location
    y: float
    z: float
    rotation_y: float
    corners_velo: np.ndarray                          # (8, 3) bbox corners in sensor frame


# ---------------------------------------------------------------------------
# Frame
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class Frame:
    """Immutable unit of data flowing through the pipeline.

    lidar is (N, 4): columns are x, y, z, intensity in velodyne frame.
    image is (H, W, 3) or None for lidar-only pipelines.
    predictions holds the detector output for this frame (populated by the
    pipeline before the defense stage so defenses can inspect predicted labels).
    """
    frame_id: str
    sequence_id: str
    timestamp: float                                  # 0.0 when not available (e.g. KITTI Object)
    lidar: np.ndarray                                 # (N, 4) — x, y, z, intensity
    image: np.ndarray | None
    labels: list[ObjectLabel]
    kitti_calib: Calibration | None = None            # None for datasets without KITTI-style calib
    nuscenes_ego_pose: np.ndarray | None = None       # (4, 4) sensor-to-global; None for KITTI Object
    nuscenes_sensor_to_ego: np.ndarray | None = None  # (4, 4) sensor-to-ego (car) frame; None for KITTI Object
    is_attacked: bool = False
    attacked_modalities: frozenset[str] = dataclasses.field(default_factory=frozenset)
    attack_metadata: dict = dataclasses.field(default_factory=dict)
    predictions: list[Prediction] = dataclasses.field(default_factory=list)

    def with_predictions(self, predictions: list[Prediction]) -> Frame:
        """Return a new Frame with predictions attached."""
        return dataclasses.replace(self, predictions=predictions)

    def with_lidar(
        self,
        new_lidar: np.ndarray,
        *,
        is_attacked: bool = True,
        attacked_modalities: frozenset[str] | None = None,
        attack_metadata: dict | None = None,
    ) -> Frame:
        """Return a new Frame with the lidar replaced and attack fields updated."""
        return dataclasses.replace(
            self,
            lidar=new_lidar,
            is_attacked=is_attacked,
            attacked_modalities=attacked_modalities if attacked_modalities is not None
                                else frozenset({"lidar"}),
            attack_metadata=attack_metadata if attack_metadata is not None
                            else self.attack_metadata,
        )


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class Prediction:
    """One predicted 3D bounding box from a detector."""
    type: str
    score: float
    # 3D box centre and dimensions in velodyne frame
    x: float
    y: float
    z: float
    height: float
    width: float
    length: float
    rotation_y: float
    corners_velo: np.ndarray                          # (8, 3)


# ---------------------------------------------------------------------------
# DetectionResult
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class DetectionResult:
    """Output of a defense module for a single frame."""
    is_attack_detected: bool
    confidence: float                                 # 0.0–1.0
    metadata: dict = dataclasses.field(default_factory=dict)


# ---------------------------------------------------------------------------
# FrameResult
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class FrameResult:
    """All outputs collected for a single frame."""
    frame_id: str
    labels: list[ObjectLabel]
    is_attacked: bool
    clean_predictions: list[Prediction]
    attacked_predictions: list[Prediction] | None     # None if no attack or no detector
    defense_result: DetectionResult | None            # None if no defense
    # Scene-position fields — always populated; attack_start_* are None for unattacked scenes
    sequence_id: str = ""
    frame_index_in_scene: int = 0
    scene_length: int = 0
    attack_start_index: int | None = None             # 0-indexed within scene; None if unattacked
    attack_start_frame_id: str | None = None          # frame_id of the first attacked frame
    attack_metadata: dict = dataclasses.field(default_factory=dict)


# ---------------------------------------------------------------------------
# EvalResults
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class EvalResults:
    """Aggregated results across all frames."""
    frame_results: list[FrameResult]
    attack_types: frozenset[str] = dataclasses.field(default_factory=frozenset)

    def attack_effectiveness(self, iou_thresholds: dict[str, float] | None = None) -> dict:
        """Compare clean vs attacked mAP per class and difficulty.

        Returns dict with keys: clean_map, attacked_map, map_degradation.
        Each is a nested dict: result[class_name][difficulty] = AP float.
        Difficulty keys are "Easy", "Moderate", "Hard", and "all".
        Requires a detector to have been configured (else returns empty dict).
        """
        from .metrics import compute_map

        if not any(r.attacked_predictions is not None for r in self.frame_results):
            return {}

        clean = compute_map(self.frame_results, iou_thresholds=iou_thresholds, use_clean=True, desc="AP (clean)")
        attacked = compute_map(self.frame_results, iou_thresholds=iou_thresholds, use_clean=False, desc="AP (attacked)")

        degradation = {
            cls: {
                diff: clean[cls][diff] - attacked[cls].get(diff, 0.0)
                for diff in clean[cls]
            }
            for cls in clean
        }
        return {
            "clean_map": clean,
            "attacked_map": attacked,
            "map_degradation": degradation,
        }

    def detection_rate(self, iou_thresholds: dict[str, float] | None = None) -> dict:
        """Frame-level detection-rate drop: recall on clean vs attacked lidar.

        Compares how many GT objects are detected before and after the attack,
        micro-averaged across all attacked frames that carry GT labels.  Works
        for any dataset (KITTI or NuScenes); NuScenes inter-sweeps with no
        annotations are automatically excluded.

        Parameters
        ----------
        iou_thresholds
            Per-class 3D-IoU matching threshold.  Pass
            ``_NUSCENES_IOU_THRESHOLDS`` for NuScenes or the KITTI
            ``iou_thresholds`` dict from ``ExperimentConfig``.  Falls back to
            0.5 for unrecognised classes.

        Returns
        -------
        Dict with one entry per class plus ``"overall"``, each containing
        ``detection_rate_clean``, ``detection_rate_attacked``, ``absolute_drop``,
        ``relative_drop``, ``total_gt``, ``total_clean_tp``,
        ``total_attacked_tp``, ``n_frames``.
        Empty dict when no qualifying attacked frames exist.
        """
        from .metrics import compute_detection_rate

        has_data = any(
            r.is_attacked and r.attacked_predictions is not None and r.labels
            for r in self.frame_results
        )
        if not has_data:
            return {}

        return compute_detection_rate(self.frame_results, iou_thresholds=iou_thresholds)

    def defense_effectiveness(self) -> dict:
        """Compute TPR, FPR, precision, recall, F1 for the defense.

        Requires a defense to have been configured (else returns empty dict).
        """
        from .metrics import compute_defense_metrics

        if not any(r.defense_result is not None for r in self.frame_results):
            return {}

        return compute_defense_metrics(self.frame_results)

    def clustering_quality(self) -> dict:
        """Compute spatial cluster-matching F1 scores for radial-jitter defenses.

        Matches cluster centroids against ORA phantom targets and vehicle
        predictions via Hungarian assignment.  Returns spoofed_f1, vehicle_f1,
        and the component precision/recall values.  Empty dict if no attacked
        frames carry cluster_details.
        """
        from .metrics import compute_clustering_quality_metrics

        has_data = any(
            r.is_attacked
            and r.defense_result is not None
            and r.defense_result.metadata.get("cluster_details")
            for r in self.frame_results
        )
        if not has_data:
            return {}

        return compute_clustering_quality_metrics(self.frame_results)

    def pacts_effectiveness(self) -> dict:
        """Cluster-level F1 for radial_jitter: correct flags vs reinjected centroids.

        TP = flagged cluster matched to a reinjected phantom within gate,
        FP = flagged cluster pointing at nothing reinjected,
        FN = reinjected phantom missed entirely.
        Returns precision, recall, f1, tp, fp, fn.
        Empty dict if no qualifying attacked frames exist.
        """
        from .metrics import compute_pacts_effectiveness

        has_data = any(
            r.defense_result is not None
            and r.defense_result.metadata.get("cluster_details") is not None
            for r in self.frame_results
        )
        if not has_data:
            return {}

        return compute_pacts_effectiveness(self.frame_results)

    def llm_attack_type_accuracy(self) -> dict:
        """Check whether LLMDefense correctly identified the attack category.

        Guards on LLM defense metadata being present (``"verdict"`` key) and
        ``attack_types`` being non-empty (i.e. the active attack declares its
        expected categories).  Returns an empty dict when either condition is
        unmet.

        Returns dict with n_tp_detected, n_correct_type, n_incorrect_type,
        type_accuracy.
        """
        from .metrics import compute_llm_attack_type_accuracy

        if not self.attack_types:
            return {}
        has_llm = any(
            r.defense_result is not None
            and "verdict" in r.defense_result.metadata
            for r in self.frame_results
        )
        if not has_llm:
            return {}

        return compute_llm_attack_type_accuracy(self.frame_results, self.attack_types)


# ---------------------------------------------------------------------------
# FrameCacheEntry
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class FrameCacheEntry:
    """Persisted detector outputs for a single frame.

    Stored by EvalPipeline when ``precomputed_cache_path`` points to a
    non-existent file; loaded on subsequent runs to skip detector inference.

    clean_predictions
        Detector output on the unmodified frame lidar.
    attacked_predictions
        Detector output on the attacked frame lidar, or None when the frame
        was not selected by the attack-fraction sampler.
    is_attacked
        Whether the attack-fraction sampler chose to attack this frame.
        Stored so replay can reconstruct the same attack/no-attack decisions
        without re-rolling the RNG.
    attack_metadata
        Copy of ``Frame.attack_metadata`` after the attack was applied.
        Forwarded to the reconstructed attacked frame during replay so that
        defenses receive consistent metadata.
    attacked_lidar
        The perturbed lidar point cloud (N, 4), or None when the frame was
        not attacked.  Stored so that ``use_cached_attacks=True`` replay can
        present the exact same lidar to the defense without re-running the attack.
    """
    clean_predictions: list[Prediction]
    attacked_predictions: list[Prediction] | None
    is_attacked: bool
    attack_metadata: dict = dataclasses.field(default_factory=dict)
    attacked_lidar: np.ndarray | None = None


# ---------------------------------------------------------------------------
# FrameHistory
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class FrameHistory:
    """Pair of rolling history deques passed to BaseDefense.detect.

    clean    — frames as yielded by the dataset (pre-attack stage).
    dirty — frames as the vehicle received them (post-attack stage).

    Both are oldest-first and sized to temporal_window - 1.
    Defenses should only read these; the pipeline owns them.
    """
    clean: deque[Frame]
    dirty: deque[Frame]
