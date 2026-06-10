"""
metrics/__init__.py
-------------------
Re-exports all public symbols from submodules for backward compatibility.

Import structure:
  common.py       — dataset-agnostic: 3D IoU, frame matching, defense metrics
  kitti_metrics.py — KITTI-specific: R40 AP, PR curves, recall-IoU, difficulties
"""

from .common import (
    _corners_to_bev_polygon,
    compute_iou_3d,
    _match_frame,
    compute_defense_metrics,
    compute_defense_metrics_filtered,
    compute_attack_success,
    compute_attack_success_rate,
    compute_clustering_quality_metrics,
    compute_pacts_effectiveness,
    compute_llm_attack_type_accuracy,
    compute_llm_cost_metrics,
    compute_timing_metrics,
    _NUSCENES_IOU_THRESHOLDS,
    compute_detection_rate,
    compute_roc_jitter,
)

from .kitti_metrics import (
    N_SAMPLE_PTS,
    _DEFAULT_IOU_THRESHOLDS,
    DIFFICULTIES,
    _DIFFICULTY_CRITERIA,
    _label_passes_difficulty,
    _get_thresholds,
    _compute_ap_class,
    compute_map,
    compute_pr_curve,
    compute_recall_vs_iou,
)

__all__ = [
    # common
    "compute_iou_3d",
    "compute_defense_metrics",
    "compute_defense_metrics_filtered",
    "compute_attack_success",
    "compute_attack_success_rate",
    "compute_clustering_quality_metrics",
    "compute_pacts_effectiveness",
    "compute_llm_attack_type_accuracy",
    "compute_llm_cost_metrics",
    "compute_timing_metrics",
    "_NUSCENES_IOU_THRESHOLDS",
    "compute_detection_rate",
    "compute_roc_jitter",
    # kitti
    "N_SAMPLE_PTS",
    "DIFFICULTIES",
    "_DEFAULT_IOU_THRESHOLDS",
    "compute_map",
    "compute_pr_curve",
    "compute_recall_vs_iou",
]
