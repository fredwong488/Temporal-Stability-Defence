"""
Smoke tests for compute_detection_rate and the EvalResults.detection_rate method.

Runs 7 test cases covering:
  1. Perfect clean recall, full suppression after attack.
  2. Partial suppression (1 of 2 objects suppressed).
  3. Unattacked frame is excluded from stats.
  4. Frame with no labels is excluded.
  5. Multi-frame micro-average aggregation.
  6. Empty result when no qualifying frames exist.
  7. Per-class breakdown with two classes.
"""

import sys
import numpy as np

sys.path.insert(0, "/Users/fred.wong/Developer/Imperial/Year_4_Code/FYP/FYP-experiment-pipeline")

from eval_pipeline.types import FrameResult, ObjectLabel, Prediction, EvalResults
from eval_pipeline.metrics import compute_detection_rate, _NUSCENES_IOU_THRESHOLDS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_corners(cx: float, cy: float, cz: float,
                 l: float = 4.0, w: float = 2.0, h: float = 1.5) -> np.ndarray:
    """Return (8, 3) box corners.  corners[:4] form a valid BEV quadrilateral."""
    return np.array([
        [cx + l/2, cy + w/2, cz + h/2],
        [cx + l/2, cy - w/2, cz + h/2],
        [cx - l/2, cy - w/2, cz + h/2],
        [cx - l/2, cy + w/2, cz + h/2],
        [cx + l/2, cy + w/2, cz - h/2],
        [cx + l/2, cy - w/2, cz - h/2],
        [cx - l/2, cy - w/2, cz - h/2],
        [cx - l/2, cy + w/2, cz - h/2],
    ], dtype=np.float32)


def make_label(cls: str, cx: float, cy: float, cz: float,
               l: float = 4.0, w: float = 2.0, h: float = 1.5) -> ObjectLabel:
    corners = make_corners(cx, cy, cz, l, w, h)
    return ObjectLabel(
        type=cls, truncated=None, occluded=None, alpha=None, bbox_2d=None,
        height=h, width=w, length=l, x=cx, y=cy, z=cz, rotation_y=0.0,
        corners_velo=corners,
    )


def make_pred(cls: str, cx: float, cy: float, cz: float,
              score: float = 0.9,
              l: float = 4.0, w: float = 2.0, h: float = 1.5) -> Prediction:
    corners = make_corners(cx, cy, cz, l, w, h)
    return Prediction(
        type=cls, score=score, x=cx, y=cy, z=cz,
        height=h, width=w, length=l, rotation_y=0.0,
        corners_velo=corners,
    )


def make_frame(
    labels: list[ObjectLabel],
    clean_preds: list[Prediction],
    attacked_preds: list[Prediction] | None,
    is_attacked: bool = True,
) -> FrameResult:
    return FrameResult(
        frame_id="f0",
        labels=labels,
        is_attacked=is_attacked,
        clean_predictions=clean_preds,
        attacked_predictions=attacked_preds,
        defense_result=None,
    )


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {name}" + (f"  ({detail})" if detail else ""))
    if not condition:
        raise AssertionError(f"Test failed: {name}  {detail}")


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

def test_full_suppression():
    """Clean detects both objects; attacked detects none."""
    label_a = make_label("car", 5.0, 0.0, 0.0)
    label_b = make_label("car", 15.0, 0.0, 0.0)
    clean_preds  = [make_pred("car", 5.0, 0.0, 0.0), make_pred("car", 15.0, 0.0, 0.0)]
    attacked_preds = []  # detector suppressed both

    fr = make_frame([label_a, label_b], clean_preds, attacked_preds)
    result = compute_detection_rate([fr])

    ov = result["overall"]
    check("full_suppression: total_gt == 2", ov["total_gt"] == 2,
          f"got {ov['total_gt']}")
    check("full_suppression: clean recall == 1.0",
          abs(ov["detection_rate_clean"] - 1.0) < 1e-6,
          f"got {ov['detection_rate_clean']:.4f}")
    check("full_suppression: attacked recall == 0.0",
          ov["detection_rate_attacked"] == 0.0,
          f"got {ov['detection_rate_attacked']:.4f}")
    check("full_suppression: absolute_drop == 1.0",
          abs(ov["absolute_drop"] - 1.0) < 1e-6,
          f"got {ov['absolute_drop']:.4f}")
    check("full_suppression: relative_drop == 1.0",
          abs(ov["relative_drop"] - 1.0) < 1e-6,
          f"got {ov['relative_drop']:.4f}")
    print("  test_full_suppression passed")


def test_partial_suppression():
    """Clean detects both objects; attacked detects one."""
    label_a = make_label("car",  5.0, 0.0, 0.0)
    label_b = make_label("car", 15.0, 0.0, 0.0)
    clean_preds    = [make_pred("car",  5.0, 0.0, 0.0), make_pred("car", 15.0, 0.0, 0.0)]
    attacked_preds = [make_pred("car", 15.0, 0.0, 0.0)]  # object at 5.0 suppressed

    fr = make_frame([label_a, label_b], clean_preds, attacked_preds)
    result = compute_detection_rate([fr])

    ov = result["overall"]
    check("partial_suppression: total_gt == 2", ov["total_gt"] == 2,
          f"got {ov['total_gt']}")
    check("partial_suppression: clean recall == 1.0",
          abs(ov["detection_rate_clean"] - 1.0) < 1e-6,
          f"got {ov['detection_rate_clean']:.4f}")
    check("partial_suppression: attacked recall == 0.5",
          abs(ov["detection_rate_attacked"] - 0.5) < 1e-6,
          f"got {ov['detection_rate_attacked']:.4f}")
    check("partial_suppression: absolute_drop == 0.5",
          abs(ov["absolute_drop"] - 0.5) < 1e-6,
          f"got {ov['absolute_drop']:.4f}")
    check("partial_suppression: relative_drop == 0.5",
          abs(ov["relative_drop"] - 0.5) < 1e-6,
          f"got {ov['relative_drop']:.4f}")
    print("  test_partial_suppression passed")


def test_unattacked_frame_excluded():
    """An unattacked frame must not contribute to stats."""
    label = make_label("car", 5.0, 0.0, 0.0)
    clean_preds = [make_pred("car", 5.0, 0.0, 0.0)]

    fr_unattacked = make_frame([label], clean_preds, attacked_preds=None, is_attacked=False)
    result = compute_detection_rate([fr_unattacked])

    check("unattacked_excluded: empty result", result == {},
          f"got {result}")
    print("  test_unattacked_frame_excluded passed")


def test_no_labels_excluded():
    """A frame flagged as attacked but with no labels must be excluded."""
    fr = make_frame(labels=[], clean_preds=[], attacked_preds=[], is_attacked=True)
    result = compute_detection_rate([fr])

    check("no_labels_excluded: empty result", result == {},
          f"got {result}")
    print("  test_no_labels_excluded passed")


def test_multi_frame_micro_average():
    """Micro-average across two frames: each has 1 GT object, clean recall both, attacked recalls one."""
    label = make_label("car", 5.0, 0.0, 0.0)
    clean = [make_pred("car", 5.0, 0.0, 0.0)]
    suppressed: list[Prediction] = []

    fr1 = make_frame([label], clean, suppressed)    # attacked suppresses it
    fr2 = make_frame([label], clean, clean)          # attacked still detects it

    result = compute_detection_rate([fr1, fr2])
    ov = result["overall"]

    check("multi_frame: total_gt == 2", ov["total_gt"] == 2,
          f"got {ov['total_gt']}")
    check("multi_frame: clean recall == 1.0",
          abs(ov["detection_rate_clean"] - 1.0) < 1e-6,
          f"got {ov['detection_rate_clean']:.4f}")
    check("multi_frame: attacked recall == 0.5",
          abs(ov["detection_rate_attacked"] - 0.5) < 1e-6,
          f"got {ov['detection_rate_attacked']:.4f}")
    check("multi_frame: n_frames == 2", ov["n_frames"] == 2,
          f"got {ov['n_frames']}")
    print("  test_multi_frame_micro_average passed")


def test_no_qualifying_frames():
    """No attacked frames at all → empty dict."""
    result = compute_detection_rate([])
    check("no_qualifying: empty result", result == {},
          f"got {result}")
    print("  test_no_qualifying_frames passed")


def test_two_classes():
    """Two classes in the same frame: per-class and overall breakdown."""
    car_label  = make_label("car",  5.0, 0.0, 0.0)
    ped_label  = make_label("pedestrian", 10.0, 0.0, 0.0)

    car_clean  = make_pred("car",  5.0, 0.0, 0.0)
    ped_clean  = make_pred("pedestrian", 10.0, 0.0, 0.0)

    # After attack: car suppressed, pedestrian still detected
    attacked_preds = [ped_clean]

    fr = make_frame([car_label, ped_label],
                    [car_clean, ped_clean],
                    attacked_preds)
    result = compute_detection_rate(
        [fr],
        iou_thresholds={"car": 0.5, "pedestrian": 0.25},
    )

    check("two_classes: 'car' key present",   "car"        in result)
    check("two_classes: 'pedestrian' key",     "pedestrian" in result)
    check("two_classes: 'overall' key",        "overall"    in result)

    car_res = result["car"]
    check("two_classes: car clean recall == 1.0",
          abs(car_res["detection_rate_clean"] - 1.0) < 1e-6,
          f"got {car_res['detection_rate_clean']:.4f}")
    check("two_classes: car attacked recall == 0.0",
          car_res["detection_rate_attacked"] == 0.0,
          f"got {car_res['detection_rate_attacked']:.4f}")

    ped_res = result["pedestrian"]
    check("two_classes: ped clean recall == 1.0",
          abs(ped_res["detection_rate_clean"] - 1.0) < 1e-6,
          f"got {ped_res['detection_rate_clean']:.4f}")
    check("two_classes: ped attacked recall == 1.0",
          abs(ped_res["detection_rate_attacked"] - 1.0) < 1e-6,
          f"got {ped_res['detection_rate_attacked']:.4f}")

    ov = result["overall"]
    # 2 GT total: 1 car TP clean + 1 ped TP clean = 2; attacked = 1 ped TP
    check("two_classes: overall total_gt == 2", ov["total_gt"] == 2,
          f"got {ov['total_gt']}")
    check("two_classes: overall clean recall == 1.0",
          abs(ov["detection_rate_clean"] - 1.0) < 1e-6,
          f"got {ov['detection_rate_clean']:.4f}")
    check("two_classes: overall attacked recall == 0.5",
          abs(ov["detection_rate_attacked"] - 0.5) < 1e-6,
          f"got {ov['detection_rate_attacked']:.4f}")
    print("  test_two_classes passed")


def test_eval_results_method():
    """EvalResults.detection_rate() delegates correctly and returns {} with no data."""
    label = make_label("car", 5.0, 0.0, 0.0)
    clean = [make_pred("car", 5.0, 0.0, 0.0)]

    fr_attacked = make_frame([label], clean, attacked_preds=[], is_attacked=True)
    er = EvalResults(frame_results=[fr_attacked])
    result = er.detection_rate(iou_thresholds=_NUSCENES_IOU_THRESHOLDS)

    check("eval_results_method: returns non-empty dict", bool(result),
          f"got {result}")
    check("eval_results_method: 'overall' key present", "overall" in result)
    check("eval_results_method: car key present", "car" in result)

    # No-data path
    fr_no_attack = make_frame([label], clean, attacked_preds=None, is_attacked=False)
    er_empty = EvalResults(frame_results=[fr_no_attack])
    result_empty = er_empty.detection_rate()
    check("eval_results_method: no-data returns {}", result_empty == {},
          f"got {result_empty}")
    print("  test_eval_results_method passed")


def test_nuscenes_iou_thresholds_keys():
    """Sanity-check that _NUSCENES_IOU_THRESHOLDS has the expected 10 classes."""
    expected = {
        "car", "truck", "bus", "trailer", "construction_vehicle",
        "motorcycle", "bicycle", "pedestrian", "traffic_cone", "barrier",
    }
    check("nuscenes_thresholds: exactly 10 keys",
          set(_NUSCENES_IOU_THRESHOLDS.keys()) == expected,
          f"got {set(_NUSCENES_IOU_THRESHOLDS.keys())}")
    check("nuscenes_thresholds: pedestrian == 0.25",
          _NUSCENES_IOU_THRESHOLDS["pedestrian"] == 0.25)
    check("nuscenes_thresholds: car == 0.5",
          _NUSCENES_IOU_THRESHOLDS["car"] == 0.5)
    print("  test_nuscenes_iou_thresholds_keys passed")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

TESTS = [
    test_full_suppression,
    test_partial_suppression,
    test_unattacked_frame_excluded,
    test_no_labels_excluded,
    test_multi_frame_micro_average,
    test_no_qualifying_frames,
    test_two_classes,
    test_eval_results_method,
    test_nuscenes_iou_thresholds_keys,
]

if __name__ == "__main__":
    failures = 0
    for test_fn in TESTS:
        print(f"\n{test_fn.__name__}")
        try:
            test_fn()
        except AssertionError as e:
            print(f"  ERROR: {e}")
            failures += 1

    print(f"\n{'=' * 50}")
    total = len(TESTS)
    passed = total - failures
    print(f"Results: {passed}/{total} passed")
    if failures:
        sys.exit(1)
