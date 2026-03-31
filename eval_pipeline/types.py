"""
types.py
--------
Core dataclasses that flow through the evaluation pipeline.
"""

from __future__ import annotations

import dataclasses
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


# ---------------------------------------------------------------------------
# ObjectLabel
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class ObjectLabel:
    """Ground-truth annotation for a single KITTI object.

    Coordinates (x, y, z) and corners_velo are in the velodyne frame.
    Dimensions (height, width, length) are in metres.
    """
    type: str                                         # "Car", "Pedestrian", "Cyclist", ...
    truncated: float
    occluded: int
    alpha: float
    bbox_2d: tuple[float, float, float, float]        # (x1, y1, x2, y2) image pixels
    height: float
    width: float
    length: float
    x: float                                          # camera-frame 3D location
    y: float
    z: float
    rotation_y: float
    corners_velo: np.ndarray                          # (8, 3) bbox corners in velodyne frame


# ---------------------------------------------------------------------------
# Frame
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class Frame:
    """Immutable unit of data flowing through the pipeline.

    lidar is (N, 4): columns are x, y, z, intensity in velodyne frame.
    image is (H, W, 3) or None for lidar-only pipelines.
    """
    frame_id: str
    sequence_id: str
    timestamp: float                                  # 0.0 when not available (e.g. KITTI Object)
    lidar: np.ndarray                                 # (N, 4) — x, y, z, intensity
    image: np.ndarray | None
    labels: list[ObjectLabel]
    calib: Calibration
    is_attacked: bool = False
    attacked_modalities: frozenset[str] = dataclasses.field(default_factory=frozenset)
    attack_metadata: dict = dataclasses.field(default_factory=dict)

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


# ---------------------------------------------------------------------------
# EvalResults
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class EvalResults:
    """Aggregated results across all frames."""
    frame_results: list[FrameResult]

    def attack_effectiveness(self, iou_threshold: float = 0.5) -> dict:
        """Compare clean vs attacked mAP.

        Returns dict with keys: clean_map, attacked_map, map_degradation (per-class and overall).
        Requires a detector to have been configured (else returns empty dicts).
        """
        from .metrics import compute_map

        if not any(r.attacked_predictions is not None for r in self.frame_results):
            return {}

        clean = compute_map(self.frame_results, iou_threshold=iou_threshold, use_clean=True)
        attacked = compute_map(self.frame_results, iou_threshold=iou_threshold, use_clean=False)

        degradation = {
            cls: clean.get(cls, 0.0) - attacked.get(cls, 0.0)
            for cls in clean
        }
        return {
            "clean_map": clean,
            "attacked_map": attacked,
            "map_degradation": degradation,
        }

    def defense_effectiveness(self) -> dict:
        """Compute TPR, FPR, precision, recall, F1 for the defense.

        Requires a defense to have been configured (else returns empty dict).
        """
        from .metrics import compute_defense_metrics

        if not any(r.defense_result is not None for r in self.frame_results):
            return {}

        return compute_defense_metrics(self.frame_results)
