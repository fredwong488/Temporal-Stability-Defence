"""
metrics/common.py
-----------------
Dataset-agnostic metrics and helpers shared across KITTI, NuScenes, etc.

Includes:
  - 3D IoU via oriented BEV polygon intersection (compute_iou_3d)
  - Greedy per-frame detection matching (_match_frame)
  - Defense binary classification metrics (compute_defense_metrics)
  - Frame-level detection-rate drop (compute_detection_rate)
"""

from __future__ import annotations

import numpy as np
from shapely.geometry import Point, Polygon

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

SPOOF_DIST_THRESHOLD = 1  # metres — phantom match gate
PREDICTION_MATCH_MARGIN = 0.5   # metres buffer added to BEV polygon for prediction match


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


# ---------------------------------------------------------------------------
# Clustering-quality metrics
# ---------------------------------------------------------------------------

def compute_clustering_quality_metrics(frame_results: list[FrameResult]) -> dict:
    """Spatial cluster-matching F1 for radial-jitter defenses.

    For each attacked frame with cluster_details, one global Hungarian assignment
    matches cluster centroids against:
      - Phantom targets: ORA reinjected_centroid from attack_metadata
      - Prediction targets: all attacked_predictions boxes (BEV containment)

    Micro-averages counts across frames then derives:
      spoofed_f1 = H(spoofed_recall, precision)
      pred_f1    = H(pred_recall, precision)

    where precision is shared (global over-clustering penalty).
    Frames with no defense_result or no cluster_details are skipped.
    """
    from scipy.optimize import linear_sum_assignment

    _INF = 1e9

    def _h(a: float, b: float) -> float:
        return 2 * a * b / (a + b) if (a + b) > 0.0 else 0.0

    mp = mpr = mc = tc = tp_count = tpr = 0

    for fr in frame_results:
        if not fr.is_attacked:
            continue
        if fr.defense_result is None:
            continue

        cluster_details = fr.defense_result.metadata.get("cluster_details")
        if not cluster_details:
            continue

        centroids: list[np.ndarray] = []
        for cd in cluster_details:
            if cd.get("skipped"):
                continue
            c = cd.get("centroid")
            if c and len(c) == 3:
                centroids.append(np.array(c, dtype=float))

        phantom_targets: list[np.ndarray] = []
        for obj in fr.attack_metadata.get("removed_per_obj", []):
            rc = obj.get("reinjected_centroid")
            if rc and len(rc) == 3:
                phantom_targets.append(np.array(rc, dtype=float))

        pred_targets: list[Prediction] = fr.attacked_predictions or []

        n_c = len(centroids)
        n_p = len(phantom_targets)
        n_pr = len(pred_targets)
        n_t = n_p + n_pr

        tc += n_c
        tp_count += n_p
        tpr += n_pr

        if n_c == 0 or n_t == 0:
            continue

        cost = np.full((n_c, n_t), _INF)

        for i, cen in enumerate(centroids):
            for j, pt in enumerate(phantom_targets):
                d = float(np.linalg.norm(cen - pt))
                if d <= SPOOF_DIST_THRESHOLD:
                    cost[i, j] = d

            for j, pred in enumerate(pred_targets):
                poly = _corners_to_bev_polygon(pred.corners_velo).buffer(PREDICTION_MATCH_MARGIN)
                if poly.contains(Point(float(cen[0]), float(cen[1]))):
                    cost[i, n_p + j] = float(np.linalg.norm(cen[:2] - np.array([pred.x, pred.y])))

        row_ind, col_ind = linear_sum_assignment(cost)
        for r, c_idx in zip(row_ind, col_ind):
            if cost[r, c_idx] < _INF:
                mc += 1
                if c_idx < n_p:
                    mp += 1
                else:
                    mpr += 1

    precision = mc / tc if tc > 0 else 0.0
    spoofed_recall = mp / tp_count if tp_count > 0 else 0.0
    pred_recall = mpr / tpr if tpr > 0 else 0.0

    return {
        "spoofed_f1": _h(spoofed_recall, precision),
        "pred_f1": _h(pred_recall, precision),
        "precision": precision,
        "spoofed_recall": spoofed_recall,
        "pred_recall": pred_recall,
        "matched_phantoms": mp,
        "total_phantoms": tp_count,
        "matched_predictions": mpr,
        "total_predictions": tpr,
        "matched_clusters": mc,
        "total_clusters": tc,
    }


# ---------------------------------------------------------------------------
# PACTS effectiveness metric (radial_jitter defense only)
# ---------------------------------------------------------------------------

def compute_pacts_effectiveness(frame_results: list[FrameResult]) -> dict:
    """Cluster-level F1 for radial_jitter: correct flags vs reinjected centroids.

    Ground truth positives are the reinjected phantom centroids from
    ``attack_metadata["removed_per_obj"]``.  Predicted positives are clusters
    where ``cluster_details[i]["flagged"] == True``.

      TP — flagged cluster within gate of any reinjected centroid;
           multiple flags near the same phantom each count as TP
      FP — flagged cluster not within gate of any reinjected centroid,
           including all flagged clusters in non-attacked frames
      FN — reinjected centroid with no flagged cluster within gate

    precision = TP / (TP + FP)    [0 when nothing flagged → F1 = 0]
    recall    = TP / (TP + FN)    [0 when all phantoms missed]
    f1        = harmonic mean

    Design note — asymmetric many-to-one matching
    ---------------------------------------------
    Matching is many-to-one in both directions: N flagged clusters near the
    same phantom count as N TPs, and one flagged cluster near M phantoms counts
    as 1 TP (with all M phantoms treated as covered, so none become FNs).

    This creates a known asymmetry: if DBSCAN over-segments a phantom into
    multiple clusters and the defense correctly flags all of them, recall is
    inflated because each extra TP increases the numerator while only one FN
    would have been charged had the phantom been missed entirely.

    The alternative — capping each phantom's contribution at 1 TP regardless of
    how many flagged clusters cover it — avoids that recall inflation, but
    introduces a symmetric precision deflation: every correctly flagged cluster
    beyond the first would not count toward TP yet every single incorrectly
    flagged cluster still counts toward FP, making precision appear worse than
    the defence actually is.

    The current scheme was chosen because artificially deflating precision is the
    more misleading distortion for Optuna optimisation: it would push the
    optimiser toward higher thresholds (fewer flags) to recover precision, even
    when the defence is flagging the right clusters.

    Frames without a defense result are skipped.  Attacked frames without
    ``removed_per_obj`` metadata are also skipped (reinjected centroids unknown).
    """
    tp = fp = fn = 0

    for fr in frame_results:
        if fr.defense_result is None:
            continue

        cluster_details = fr.defense_result.metadata.get("cluster_details")
        if cluster_details is None:
            continue

        flagged: list[np.ndarray] = []
        for cd in cluster_details:
            if not cd.get("flagged"):
                continue
            c = cd.get("centroid")
            if c and len(c) == 3:
                flagged.append(np.array(c, dtype=float))

        if not fr.is_attacked:
            fp += len(flagged)
            continue

        reinjected_centroids: list[np.ndarray] = []
        for obj in fr.attack_metadata.get("removed_per_obj", []):
            rc = obj.get("reinjected_centroid")
            if rc and len(rc) == 3:
                reinjected_centroids.append(np.array(rc, dtype=float))

        if not reinjected_centroids:
            continue

        n_f = len(flagged)
        n_r = len(reinjected_centroids)

        if n_f == 0:
            fn += n_r
            continue

        phantom_arr = np.array(reinjected_centroids)  # (P, 3)
        flagged_arr = np.array(flagged)                # (F, 3)

        # Each flagged cluster independently checks proximity to any phantom.
        # Multiple flagged clusters matching the same phantom all count as TP —
        # they are each individually correct flags, not errors.
        for fc in flagged:
            if np.linalg.norm(phantom_arr - fc, axis=1).min() <= SPOOF_DIST_THRESHOLD:
                tp += 1
            else:
                fp += 1

        # Each phantom is a FN only if no flagged cluster covers it.
        for rc in reinjected_centroids:
            if np.linalg.norm(flagged_arr - rc, axis=1).min() > SPOOF_DIST_THRESHOLD:
                fn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# ---------------------------------------------------------------------------
# LLM attack-type accuracy metric (LLMDefense only)
# ---------------------------------------------------------------------------

def compute_llm_attack_type_accuracy(
    frame_results: list[FrameResult],
    expected_attack_types: frozenset[str],
) -> dict:
    """Check whether LLMDefense correctly identified the attack category.

    Only applies to frames where:
      - the defense result carries LLM metadata (``"verdict"`` key present), AND
      - the frame was actually attacked, AND
      - an attack was detected (``is_attack_detected=True``).

    For each such frame, the LLM is considered to have given a correct type if
    at least one entry in ``suspected_attacks`` has an ``attack_type`` value that
    is in ``expected_attack_types``.

    Parameters
    ----------
    expected_attack_types
        Frozenset of ``AttackType`` string values the active attack may produce
        (e.g. ``frozenset({"OBJECT_HIDING", "OBJECT_TRANSLATION"})`` for ORA).

    Returns
    -------
    dict with keys:
      n_tp_detected    — frames that were attacked and detected
      n_correct_type   — subset where at least one suspected type is expected
      n_incorrect_type — subset where no suspected type is expected
      type_accuracy    — n_correct_type / n_tp_detected  (0 if none)
    """
    n_tp_detected = 0
    n_correct_type = 0

    for fr in frame_results:
        if fr.defense_result is None:
            continue
        meta = fr.defense_result.metadata
        if "verdict" not in meta:
            continue
        if not fr.is_attacked or not fr.defense_result.is_attack_detected:
            continue

        n_tp_detected += 1
        suspected = meta.get("suspected_attacks", [])
        types_reported = {s["attack_type"] for s in suspected if "attack_type" in s}
        if types_reported & expected_attack_types:
            n_correct_type += 1

    n_incorrect_type = n_tp_detected - n_correct_type
    type_accuracy = n_correct_type / n_tp_detected if n_tp_detected > 0 else 0.0

    return {
        "n_tp_detected": n_tp_detected,
        "n_correct_type": n_correct_type,
        "n_incorrect_type": n_incorrect_type,
        "type_accuracy": type_accuracy,
    }


# ---------------------------------------------------------------------------
# Detection-rate drop (dataset-agnostic, designed for NuScenes)
# ---------------------------------------------------------------------------

# Per-class 3D-IoU thresholds matching the NuScenes detection task convention.
# Vehicle-sized classes use 0.5, small objects use lighter thresholds so that
# compute_iou_3d (BEV polygon × height overlap) is not excessively harsh.
_NUSCENES_IOU_THRESHOLDS: dict[str, float] = {
    "car": 0.5,
    "truck": 0.5,
    "bus": 0.5,
    "trailer": 0.5,
    "construction_vehicle": 0.5,
    "motorcycle": 0.3,
    "bicycle": 0.3,
    "pedestrian": 0.25,
    "traffic_cone": 0.25,
    "barrier": 0.25,
}


def _detection_rate_one_class(
    frame_results: list[FrameResult],
    class_name: str,
    iou_threshold: float,
    score_threshold: float | None,
) -> dict:
    """Return detection-rate stats for a single class over all qualifying frames.

    Qualifying frames: is_attacked=True, attacked_predictions is not None,
    labels non-empty.

    Returns dict with:
      detection_rate_clean, detection_rate_attacked, absolute_drop,
      relative_drop, total_gt, total_clean_tp, total_attacked_tp, n_frames.
    """
    total_gt = 0
    total_clean_tp = 0
    total_attacked_tp = 0
    n_frames = 0

    for fr in frame_results:
        if not fr.is_attacked:
            continue
        if fr.attacked_predictions is None:
            continue
        if not fr.labels:
            continue
        if not any(lbl.type == class_name for lbl in fr.labels):
            continue

        n_frames += 1

        _, n_gt_clean = _match_frame(
            fr.clean_predictions, fr.labels, class_name, iou_threshold,
            score_threshold=score_threshold,
        )
        clean_tps = sum(
            1 for _, is_tp in _match_frame(
                fr.clean_predictions, fr.labels, class_name, iou_threshold,
                score_threshold=score_threshold,
            )[0]
            if is_tp
        )

        _, n_gt_att = _match_frame(
            fr.attacked_predictions, fr.labels, class_name, iou_threshold,
            score_threshold=score_threshold,
        )
        attacked_tps = sum(
            1 for _, is_tp in _match_frame(
                fr.attacked_predictions, fr.labels, class_name, iou_threshold,
                score_threshold=score_threshold,
            )[0]
            if is_tp
        )

        total_gt += n_gt_clean        # n_gt is the same regardless of predictions
        total_clean_tp += clean_tps
        total_attacked_tp += attacked_tps

    dr_clean = total_clean_tp / total_gt if total_gt > 0 else 0.0
    dr_attacked = total_attacked_tp / total_gt if total_gt > 0 else 0.0
    abs_drop = dr_clean - dr_attacked
    rel_drop = abs_drop / dr_clean if dr_clean > 0.0 else 0.0

    return {
        "detection_rate_clean": dr_clean,
        "detection_rate_attacked": dr_attacked,
        "absolute_drop": abs_drop,
        "relative_drop": rel_drop,
        "total_gt": total_gt,
        "total_clean_tp": total_clean_tp,
        "total_attacked_tp": total_attacked_tp,
        "n_frames": n_frames,
    }


def compute_detection_rate(
    frame_results: list[FrameResult],
    iou_thresholds: dict[str, float] | None = None,
    classes: list[str] | None = None,
    default_iou: float = 0.5,
    score_threshold: float | None = None,
) -> dict:
    """Frame-level detection-rate drop: clean recall vs attacked recall against GT.

    Only attacked frames that have ground-truth labels are included (NuScenes
    inter-sweeps without annotations are automatically excluded).

    Parameters
    ----------
    iou_thresholds
        Per-class IoU matching threshold.  Falls back to ``default_iou`` for any
        class not in the dict.  Pass ``_NUSCENES_IOU_THRESHOLDS`` for NuScenes or
        the KITTI ``iou_thresholds`` dict from ``ExperimentConfig``.
    classes
        Classes to evaluate.  When None, derived from labels in qualifying frames.
    default_iou
        Fallback IoU threshold for classes not in ``iou_thresholds``.
    score_threshold
        Minimum detection score for a prediction to count.

    Returns
    -------
    Dict with one key per class (plus ``"overall"``), each mapping to::

        {
          "detection_rate_clean": float,    # micro-avg recall on clean lidar
          "detection_rate_attacked": float, # micro-avg recall on attacked lidar
          "absolute_drop": float,           # clean − attacked
          "relative_drop": float,           # drop / clean  (0 when clean=0)
          "total_gt": int,
          "total_clean_tp": int,
          "total_attacked_tp": int,
          "n_frames": int,
        }

    Returns an empty dict when no qualifying frames exist.
    """
    qualifying = [
        fr for fr in frame_results
        if fr.is_attacked and fr.attacked_predictions is not None and fr.labels
    ]
    if not qualifying:
        return {}

    if classes is None:
        classes = sorted({lbl.type for fr in qualifying for lbl in fr.labels})

    thresholds = iou_thresholds or {}
    result: dict = {}

    for cls in classes:
        iou_thr = thresholds.get(cls, default_iou)
        result[cls] = _detection_rate_one_class(
            frame_results, cls, iou_thr, score_threshold
        )

    # Overall micro-average across all classes
    total_gt = sum(result[cls]["total_gt"] for cls in result)
    total_clean_tp = sum(result[cls]["total_clean_tp"] for cls in result)
    total_attacked_tp = sum(result[cls]["total_attacked_tp"] for cls in result)
    dr_clean = total_clean_tp / total_gt if total_gt > 0 else 0.0
    dr_attacked = total_attacked_tp / total_gt if total_gt > 0 else 0.0
    abs_drop = dr_clean - dr_attacked
    result["overall"] = {
        "detection_rate_clean": dr_clean,
        "detection_rate_attacked": dr_attacked,
        "absolute_drop": abs_drop,
        "relative_drop": abs_drop / dr_clean if dr_clean > 0.0 else 0.0,
        "total_gt": total_gt,
        "total_clean_tp": total_clean_tp,
        "total_attacked_tp": total_attacked_tp,
        "n_frames": sum(result[cls]["n_frames"] for cls in classes),
    }

    return result
