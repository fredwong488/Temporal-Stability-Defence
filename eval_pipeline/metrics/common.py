"""
metrics/common.py
-----------------
Dataset-agnostic metrics and helpers shared across KITTI, NuScenes, etc.

Includes:
  - 3D IoU via oriented BEV polygon intersection (compute_iou_3d)
  - Greedy per-frame detection matching (_match_frame)
  - Defense binary classification metrics (compute_defense_metrics)
"""

from __future__ import annotations

import numpy as np
from shapely.geometry import Polygon

from ..types import FrameResult, ObjectLabel, Prediction


# ---------------------------------------------------------------------------
# 3D IoU — oriented BEV polygon × height overlap
# ---------------------------------------------------------------------------

def _corners_to_bev_polygon(corners: np.ndarray) -> Polygon:
    """Convert (8, 3) corners_velo to a Shapely Polygon in the BEV (x, y) plane."""
    pts = corners[:4, :2]  # (4, 2) — bottom face projected to xy
    centroid = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - centroid[1], pts[:, 0] - centroid[0])
    return Polygon(pts[np.argsort(angles)])


def compute_iou_3d(corners_a: np.ndarray, corners_b: np.ndarray) -> float:
    """Compute 3D IoU matching the KITTI devkit box3DOverlap().

    Method: inter_vol = BEV_polygon_intersection_area × height_overlap
            IoU = inter_vol / (vol_a + vol_b − inter_vol)
    """
    poly_a = _corners_to_bev_polygon(corners_a)
    poly_b = _corners_to_bev_polygon(corners_b)

    inter_area = poly_a.intersection(poly_b).area
    if inter_area == 0.0:
        return 0.0

    z_top = min(corners_a[:, 2].max(), corners_b[:, 2].max())
    z_bot = max(corners_a[:, 2].min(), corners_b[:, 2].min())
    h_overlap = max(0.0, z_top - z_bot)
    if h_overlap == 0.0:
        return 0.0

    inter_vol = inter_area * h_overlap
    vol_a = poly_a.area * (corners_a[:, 2].max() - corners_a[:, 2].min())
    vol_b = poly_b.area * (corners_b[:, 2].max() - corners_b[:, 2].min())
    union_vol = vol_a + vol_b - inter_vol

    return float(inter_vol / union_vol) if union_vol > 0.0 else 0.0


# ---------------------------------------------------------------------------
# Frame-level matching
# ---------------------------------------------------------------------------

def _match_frame(
    predictions: list[Prediction],
    labels: list[ObjectLabel],
    class_name: str,
    iou_threshold: float,
    score_threshold: float | None = None,
    difficulty: str | None = None,
    _label_passes_difficulty_fn=None,
) -> tuple[list[tuple[float, bool]], int]:
    """Greedy highest-IoU matching for one frame and class.

    Parameters
    ----------
    difficulty
        Optional difficulty level string (e.g. "Easy"). Requires
        _label_passes_difficulty_fn to be provided; ignored if None.
    _label_passes_difficulty_fn
        Callable(label, difficulty) -> bool used to filter GT by difficulty.
        Only used when difficulty is not None.
    score_threshold
        Predictions with score < threshold are skipped.

    Returns
    -------
    detections : list of (score, is_tp)
    n_gt       : number of valid GT boxes of this class in this frame
    """
    all_gt = [lbl for lbl in labels if lbl.type == class_name]

    if difficulty is not None and _label_passes_difficulty_fn is not None:
        valid_gt   = [g for g in all_gt if     _label_passes_difficulty_fn(g, difficulty)]
        ignored_gt = [g for g in all_gt if not _label_passes_difficulty_fn(g, difficulty)]
    else:
        valid_gt, ignored_gt = all_gt, []

    preds = [p for p in predictions if p.type == class_name]
    if score_threshold is not None:
        preds = [p for p in preds if p.score >= score_threshold]

    n_gt = len(valid_gt)
    if not valid_gt and not ignored_gt:
        return [(p.score, False) for p in preds], 0
    if not preds:
        return [], n_gt

    preds = sorted(preds, key=lambda p: p.score, reverse=True)
    matched_valid: set[int] = set()
    result: list[tuple[float, bool]] = []

    for pred in preds:
        best_iou, best_j = 0.0, -1
        for j, g in enumerate(valid_gt):
            if j in matched_valid:
                continue
            iou = compute_iou_3d(pred.corners_velo, g.corners_velo)
            if iou > best_iou:
                best_iou, best_j = iou, j

        if best_iou >= iou_threshold and best_j >= 0:
            result.append((pred.score, True))
            matched_valid.add(best_j)
        else:
            ignored = any(
                compute_iou_3d(pred.corners_velo, g.corners_velo) >= iou_threshold
                for g in ignored_gt
            )
            if not ignored:
                result.append((pred.score, False))

    return result, n_gt


# ---------------------------------------------------------------------------
# Defense metrics
# ---------------------------------------------------------------------------

def compute_defense_metrics(frame_results: list[FrameResult]) -> dict:
    """Compute binary classification metrics for the defense.

    Ground truth  : frame_result.is_attacked
    Prediction    : frame_result.defense_result.is_attack_detected

    Returns dict with tp, fp, tn, fn, tpr, fpr, accuracy, precision, recall, f1.
    Frames with no defense_result are skipped.
    """
    tp = fp = tn = fn = 0

    for fr in frame_results:
        if fr.defense_result is None:
            continue
        actual = fr.is_attacked
        predicted = fr.defense_result.is_attack_detected

        if actual and predicted:
            tp += 1
        elif not actual and predicted:
            fp += 1
        elif not actual and not predicted:
            tn += 1
        else:
            fn += 1

    total = tp + fp + tn + fn
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tpr
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)
    accuracy = (tp + tn) / total if total > 0 else 0.0

    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "tpr": tpr,
        "fpr": fpr,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
