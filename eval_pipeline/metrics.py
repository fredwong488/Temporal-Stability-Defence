"""
metrics.py
----------
Evaluation metrics for attack effectiveness and defense effectiveness.
"""

from __future__ import annotations

import numpy as np

from .types import FrameResult, ObjectLabel, Prediction


# ---------------------------------------------------------------------------
# 3D IoU (AABB approximation)
# ---------------------------------------------------------------------------

def compute_iou_3d(corners_a: np.ndarray, corners_b: np.ndarray) -> float:
    """Compute 3D IoU between two boxes using their (8, 3) corner arrays.

    Uses axis-aligned bounding box approximation — fast but approximate for
    rotated boxes.  Sufficient for mAP computation with standard KITTI eval.
    """
    min_a, max_a = corners_a.min(axis=0), corners_a.max(axis=0)
    min_b, max_b = corners_b.min(axis=0), corners_b.max(axis=0)

    inter_min = np.maximum(min_a, min_b)
    inter_max = np.minimum(max_a, max_b)
    inter_dims = np.maximum(inter_max - inter_min, 0.0)
    inter_vol = float(inter_dims[0] * inter_dims[1] * inter_dims[2])

    if inter_vol == 0.0:
        return 0.0

    vol_a = float(np.prod(max_a - min_a))
    vol_b = float(np.prod(max_b - min_b))
    union_vol = vol_a + vol_b - inter_vol
    return inter_vol / union_vol if union_vol > 0 else 0.0


# ---------------------------------------------------------------------------
# Per-frame AP
# ---------------------------------------------------------------------------

def compute_ap(
    predictions: list[Prediction],
    labels: list[ObjectLabel],
    iou_threshold: float = 0.5,
    class_name: str = "Car",
) -> float:
    """Compute Average Precision for a single frame and class.

    Uses the 11-point interpolation method (PASCAL VOC / KITTI standard).
    Returns 0.0 if there are no ground-truth objects of the given class.
    """
    gt = [l for l in labels if l.type == class_name]
    preds = sorted(
        [p for p in predictions if p.type == class_name],
        key=lambda p: p.score, reverse=True,
    )

    if not gt:
        return 0.0
    if not preds:
        return 0.0

    n_gt = len(gt)
    matched_gt: set[int] = set()
    tp = np.zeros(len(preds))
    fp = np.zeros(len(preds))

    for i, pred in enumerate(preds):
        best_iou = 0.0
        best_j = -1
        for j, g in enumerate(gt):
            if j in matched_gt:
                continue
            iou = compute_iou_3d(pred.corners_velo, g.corners_velo)
            if iou > best_iou:
                best_iou = iou
                best_j = j
        if best_iou >= iou_threshold and best_j >= 0:
            tp[i] = 1
            matched_gt.add(best_j)
        else:
            fp[i] = 1

    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)
    recall = cum_tp / n_gt
    precision = cum_tp / (cum_tp + cum_fp + 1e-9)

    # 11-point interpolation
    ap = 0.0
    for t in np.linspace(0.0, 1.0, 11):
        p = precision[recall >= t].max() if (recall >= t).any() else 0.0
        ap += p / 11.0

    return float(ap)


# ---------------------------------------------------------------------------
# mAP across frames
# ---------------------------------------------------------------------------

def compute_map(
    frame_results: list[FrameResult],
    iou_threshold: float = 0.5,
    use_clean: bool = True,
) -> dict[str, float]:
    """Compute mean Average Precision across all frames, per class.

    Parameters
    ----------
    use_clean
        If True, use clean_predictions; otherwise use attacked_predictions.

    Returns
    -------
    dict with per-class AP and overall "mAP" key.
    """
    # Collect all class names appearing in labels
    all_classes: set[str] = set()
    for fr in frame_results:
        for lbl in fr.labels:
            all_classes.add(lbl.type)

    class_aps: dict[str, list[float]] = {c: [] for c in all_classes}

    for fr in frame_results:
        preds = fr.clean_predictions if use_clean else (fr.attacked_predictions or [])
        for cls in all_classes:
            ap = compute_ap(preds, fr.labels, iou_threshold=iou_threshold, class_name=cls)
            class_aps[cls].append(ap)

    result: dict[str, float] = {}
    for cls, aps in class_aps.items():
        result[cls] = float(np.mean(aps)) if aps else 0.0

    if result:
        result["mAP"] = float(np.mean(list(result.values())))
    return result


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
