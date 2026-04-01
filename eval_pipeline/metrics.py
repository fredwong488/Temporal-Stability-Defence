"""
metrics.py
----------
Evaluation metrics matching the KITTI devkit (evaluate_object.cpp).

3D IoU      : oriented BEV polygon intersection × height overlap (via Shapely),
              matching devkit box3DOverlap() / toPolygon().
AP          : R40 — data-driven 41-point threshold selection, computed globally
              across all frames per class and difficulty, matching devkit
              getThresholds() / eval_class().
Difficulty  : Easy / Moderate / Hard per KITTI devkit criteria.
PR curve    : compute_pr_curve() — (recall, precision) pairs at each R40 sample.
Recall–IoU  : compute_recall_vs_iou() — recall swept over IoU thresholds at a
              fixed confidence threshold.
"""

from __future__ import annotations

import numpy as np
from shapely.geometry import Polygon

from .types import FrameResult, ObjectLabel, Prediction

# ---------------------------------------------------------------------------
# Constants matching KITTI devkit
# ---------------------------------------------------------------------------

N_SAMPLE_PTS = 41  # 41 sample points → 40 recall intervals (R40)

_DEFAULT_IOU_THRESHOLDS: dict[str, float] = {
    "Car": 0.7,
    "Pedestrian": 0.5,
    "Cyclist": 0.5,
}

DIFFICULTIES: tuple[str, ...] = ("Easy", "Moderate", "Hard")

# Devkit evaluate_object.cpp: MIN_HEIGHT, MAX_OCCLUSION, MAX_TRUNCATION arrays
_DIFFICULTY_CRITERIA: dict[str, dict] = {
    "Easy":     {"min_height": 40, "max_occlusion": 0, "max_truncation": 0.15},
    "Moderate": {"min_height": 25, "max_occlusion": 1, "max_truncation": 0.30},
    "Hard":     {"min_height": 25, "max_occlusion": 2, "max_truncation": 0.50},
}


# ---------------------------------------------------------------------------
# Difficulty helpers
# ---------------------------------------------------------------------------

def _label_passes_difficulty(label: ObjectLabel, difficulty: str) -> bool:
    """Return True if *label* meets the devkit criteria for *difficulty*."""
    c = _DIFFICULTY_CRITERIA[difficulty]
    height = label.bbox_2d[3] - label.bbox_2d[1]   # y2 − y1 in image pixels
    return (
        height >= c["min_height"]
        and label.occluded <= c["max_occlusion"]
        and label.truncated <= c["max_truncation"]
    )


# ---------------------------------------------------------------------------
# 3D IoU — oriented BEV polygon × height overlap
# ---------------------------------------------------------------------------

def _corners_to_bev_polygon(corners: np.ndarray) -> Polygon:
    """Convert (8, 3) corners_velo to a Shapely Polygon in the BEV (x, y) plane.

    Uses the bottom-face corners (indices 0–3) ordered CCW by angle from centroid,
    matching the devkit's toPolygon() which constructs an oriented footprint.
    """
    pts = corners[:4, :2]  # (4, 2) — bottom face projected to xy
    centroid = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - centroid[1], pts[:, 0] - centroid[0])
    return Polygon(pts[np.argsort(angles)])


def compute_iou_3d(corners_a: np.ndarray, corners_b: np.ndarray) -> float:
    """Compute 3D IoU matching the KITTI devkit box3DOverlap().

    Method (identical to devkit):
        inter_vol = BEV_polygon_intersection_area × height_overlap
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
) -> tuple[list[tuple[float, bool]], int]:
    """Greedy highest-IoU matching for one frame and class.

    Difficulty handling (mirrors devkit computeStatistics / ignoreGT logic):
      - GT boxes that pass the difficulty filter are *valid* and count toward n_gt.
      - GT boxes that fail the filter are *ignored*: a prediction matched to one
        above iou_threshold is excluded from the TP/FP list (neither TP nor FP).

    Parameters
    ----------
    difficulty
        One of "Easy", "Moderate", "Hard", or None (no difficulty filter).
    score_threshold
        If given, predictions with score < threshold are skipped.

    Returns
    -------
    detections : list of (score, is_tp) — one entry per non-ignored prediction.
    n_gt       : number of valid GT boxes of this class in this frame.
    """
    all_gt = [lbl for lbl in labels if lbl.type == class_name]

    if difficulty is not None:
        valid_gt   = [g for g in all_gt if     _label_passes_difficulty(g, difficulty)]
        ignored_gt = [g for g in all_gt if not _label_passes_difficulty(g, difficulty)]
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
            result.append((pred.score, True))    # TP
            matched_valid.add(best_j)
        else:
            # Prediction matched an ignored GT → exclude it from TP/FP
            ignored = any(
                compute_iou_3d(pred.corners_velo, g.corners_velo) >= iou_threshold
                for g in ignored_gt
            )
            if not ignored:
                result.append((pred.score, False))  # FP

    return result, n_gt


# ---------------------------------------------------------------------------
# R40 threshold selection
# ---------------------------------------------------------------------------

def _get_thresholds(tp_scores: list[float], n_gt: int) -> list[float]:
    """Port of devkit getThresholds().

    Selects up to N_SAMPLE_PTS−1 confidence score thresholds that best
    approximate linearly spaced recall levels at 1/(N_SAMPLE_PTS−1) intervals.
    """
    if not tp_scores or n_gt == 0:
        return []

    v = sorted(tp_scores, reverse=True)
    thresholds: list[float] = []
    current_recall = 0.0

    for i, score in enumerate(v):
        l_recall = (i + 1) / n_gt
        r_recall = (i + 2) / n_gt if i < len(v) - 1 else l_recall

        if (r_recall - current_recall) < (current_recall - l_recall) and i < len(v) - 1:
            continue

        thresholds.append(score)
        current_recall += 1.0 / (N_SAMPLE_PTS - 1.0)

    return thresholds


# ---------------------------------------------------------------------------
# Per-class, per-difficulty AP — global across all frames, R40
# ---------------------------------------------------------------------------

def _compute_ap_class(
    frame_results: list[FrameResult],
    class_name: str,
    iou_threshold: float,
    use_clean: bool,
    difficulty: str | None = None,
) -> float:
    """AP for one class/difficulty, computed globally across all frames (devkit eval_class()).

    Two-pass algorithm:
      Pass 1 — collect TP confidence scores + total GT count (no score filter).
      Pass 2 — at each R40 threshold, accumulate TP/FP across all frames,
               compute precision, apply monotone envelope, average.
    """
    all_tp_scores: list[float] = []
    n_gt_total = 0

    for fr in frame_results:
        preds = fr.clean_predictions if use_clean else (fr.attacked_predictions or [])
        detections, n_gt = _match_frame(
            preds, fr.labels, class_name, iou_threshold, difficulty=difficulty
        )
        n_gt_total += n_gt
        all_tp_scores.extend(score for score, is_tp in detections if is_tp)

    if n_gt_total == 0:
        return 0.0

    thresholds = _get_thresholds(all_tp_scores, n_gt_total)
    if not thresholds:
        return 0.0

    tp_arr = np.zeros(len(thresholds))
    fp_arr = np.zeros(len(thresholds))

    for fr in frame_results:
        preds = fr.clean_predictions if use_clean else (fr.attacked_predictions or [])
        for t_idx, thresh in enumerate(thresholds):
            detections, _ = _match_frame(
                preds, fr.labels, class_name, iou_threshold,
                score_threshold=thresh, difficulty=difficulty,
            )
            tp_arr[t_idx] += sum(1 for _, is_tp in detections if is_tp)
            fp_arr[t_idx] += sum(1 for _, is_tp in detections if not is_tp)

    precision = tp_arr / (tp_arr + fp_arr + 1e-9)

    # Monotone envelope: precision[i] = max(precision[i:])  (devkit line 699)
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])

    padded = np.zeros(N_SAMPLE_PTS)
    padded[: len(precision)] = precision
    return float(padded.mean())


# ---------------------------------------------------------------------------
# mAP — per class, per difficulty
# ---------------------------------------------------------------------------

def compute_map(
    frame_results: list[FrameResult],
    iou_thresholds: dict[str, float] | None = None,
    use_clean: bool = True,
    difficulties: tuple[str, ...] = DIFFICULTIES,
) -> dict[str, dict[str, float]]:
    """Compute AP (R40) per class and difficulty level.

    Parameters
    ----------
    iou_thresholds
        Per-class IoU thresholds. Defaults to KITTI standard
        (Car=0.7, Pedestrian=0.5, Cyclist=0.5). Unknown classes fall back to 0.5.
    use_clean
        If True, use clean_predictions; otherwise use attacked_predictions.
    difficulties
        Difficulty levels to evaluate. Defaults to all three KITTI levels.
        Each is evaluated independently. An "all" key (no difficulty filter)
        is always included.

    Returns
    -------
    Nested dict: result[class_name][difficulty] = AP, plus result["mAP"].

    Example
    -------
    {
        "Car":        {"Easy": 0.85, "Moderate": 0.72, "Hard": 0.61, "all": 0.73},
        "Pedestrian": {"Easy": 0.60, "Moderate": 0.48, "Hard": 0.35, "all": 0.48},
        "mAP":        {"Easy": 0.73, "Moderate": 0.60, "Hard": 0.48, "all": 0.61},
    }
    """
    if iou_thresholds is None:
        iou_thresholds = _DEFAULT_IOU_THRESHOLDS

    all_classes: set[str] = set()
    for fr in frame_results:
        for lbl in fr.labels:
            all_classes.add(lbl.type)

    result: dict[str, dict[str, float]] = {}
    for cls in sorted(all_classes):
        threshold = iou_thresholds.get(cls, 0.5)
        cls_ap: dict[str, float] = {}
        for diff in difficulties:
            cls_ap[diff] = _compute_ap_class(
                frame_results, cls, threshold, use_clean, difficulty=diff
            )
        cls_ap["all"] = _compute_ap_class(
            frame_results, cls, threshold, use_clean, difficulty=None
        )
        result[cls] = cls_ap

    if result:
        all_keys = list(difficulties) + ["all"]
        result["mAP"] = {
            key: float(np.mean([result[cls][key] for cls in result]))
            for key in all_keys
        }

    return result


# ---------------------------------------------------------------------------
# Precision–Recall curves
# ---------------------------------------------------------------------------

def compute_pr_curve(
    frame_results: list[FrameResult],
    class_name: str,
    iou_threshold: float,
    use_clean: bool = True,
    difficulty: str | None = None,
) -> dict:
    """Return the precision–recall curve for one class/difficulty.

    Uses the same R40 score thresholds as AP computation. Precision has the
    monotone envelope applied (matching the AP calculation), so the curve is
    non-increasing in precision as recall increases.

    Parameters
    ----------
    difficulty
        One of "Easy", "Moderate", "Hard", or None (no filter).

    Returns
    -------
    dict with keys:
        "recall"    : list[float] — recall at each R40 sample point
        "precision" : list[float] — precision at each point (monotone envelope)
        "ap"        : float — scalar AP (mean of padded precision array)
        "n_gt"      : int — total valid GT count used for normalisation
    """
    all_tp_scores: list[float] = []
    n_gt_total = 0

    for fr in frame_results:
        preds = fr.clean_predictions if use_clean else (fr.attacked_predictions or [])
        detections, n_gt = _match_frame(
            preds, fr.labels, class_name, iou_threshold, difficulty=difficulty
        )
        n_gt_total += n_gt
        all_tp_scores.extend(score for score, is_tp in detections if is_tp)

    if n_gt_total == 0:
        return {"recall": [], "precision": [], "ap": 0.0, "n_gt": 0}

    thresholds = _get_thresholds(all_tp_scores, n_gt_total)
    if not thresholds:
        return {"recall": [], "precision": [], "ap": 0.0, "n_gt": n_gt_total}

    tp_arr = np.zeros(len(thresholds))
    fp_arr = np.zeros(len(thresholds))

    for fr in frame_results:
        preds = fr.clean_predictions if use_clean else (fr.attacked_predictions or [])
        for t_idx, thresh in enumerate(thresholds):
            detections, _ = _match_frame(
                preds, fr.labels, class_name, iou_threshold,
                score_threshold=thresh, difficulty=difficulty,
            )
            tp_arr[t_idx] += sum(1 for _, is_tp in detections if is_tp)
            fp_arr[t_idx] += sum(1 for _, is_tp in detections if not is_tp)

    precision = tp_arr / (tp_arr + fp_arr + 1e-9)
    recall = tp_arr / (n_gt_total + 1e-9)

    # Monotone envelope on precision
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])

    padded_p = np.zeros(N_SAMPLE_PTS)
    padded_p[: len(precision)] = precision
    ap = float(padded_p.mean())

    return {
        "recall": recall.tolist(),
        "precision": precision.tolist(),
        "ap": ap,
        "n_gt": n_gt_total,
    }


# ---------------------------------------------------------------------------
# Recall vs IoU threshold curve
# ---------------------------------------------------------------------------

def compute_recall_vs_iou(
    frame_results: list[FrameResult],
    class_name: str,
    confidence_threshold: float,
    use_clean: bool = True,
    iou_values: list[float] | None = None,
) -> dict:
    """Recall as a function of IoU threshold at a fixed confidence threshold.

    Useful for visualising attack impact: under adversarial attack, the curve
    shifts down — the detector needs a lower IoU threshold to maintain recall.

    Parameters
    ----------
    confidence_threshold
        Only predictions with score >= this value are evaluated.
    iou_values
        IoU thresholds to sweep. Defaults to 0.10 … 0.90 in steps of 0.05.

    Returns
    -------
    dict with keys:
        "iou_thresholds" : list[float]
        "recall"         : list[float] — recall at each IoU threshold
        "n_gt"           : int — total GT count
    """
    if iou_values is None:
        iou_values = [round(v, 2) for v in np.arange(0.10, 0.91, 0.05).tolist()]

    n_gt_total = sum(
        sum(1 for lbl in fr.labels if lbl.type == class_name)
        for fr in frame_results
    )

    recalls: list[float] = []
    for iou_thresh in iou_values:
        tp_total = 0
        for fr in frame_results:
            preds = fr.clean_predictions if use_clean else (fr.attacked_predictions or [])
            detections, _ = _match_frame(
                preds, fr.labels, class_name, iou_thresh,
                score_threshold=confidence_threshold,
            )
            tp_total += sum(1 for _, is_tp in detections if is_tp)
        recalls.append(float(tp_total / n_gt_total) if n_gt_total > 0 else 0.0)

    return {
        "iou_thresholds": iou_values,
        "recall": recalls,
        "n_gt": n_gt_total,
    }


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
