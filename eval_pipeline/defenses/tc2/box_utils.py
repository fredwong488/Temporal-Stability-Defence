from __future__ import annotations

import numpy as np


def points_in_box_bev(
    corners_velo: np.ndarray,
    points_xy: np.ndarray,
) -> np.ndarray:
    """Check which 2D points lie inside a bounding box in the BEV plane.

    This is the 2D analogue of TC2's ``points_2D_in_box`` (TC2.py:54-88),
    rewritten to accept ``Prediction.corners_velo`` (8, 3) instead of a
    NuScenes ``Box`` object.

    IMPORTANT — detector compatibility
    ----------------------------------
    The in-plane edge selection below (corners 0, 3, 1) assumes the corner
    ordering produced by ``PointPillarsDetector._box_to_corners`` (and its
    subclass ``PointPillarsNuScenesDetector`` / ``PointRCNNDetector``), where
    corner 0 = (-l, -w, -h), corner 1 is its +length neighbour and corner 3 is
    its +width neighbour. TC2 must therefore be paired with a pointpillars-family
    detector.

    Parameters
    ----------
    corners_velo : (8, 3) box corners in sensor/velodyne frame.
    points_xy    : (2, N) points in the BEV plane (x, y rows).

    Returns
    -------
    mask : (N,) boolean array — True if point is inside the BEV footprint.
    """
    corners = corners_velo.T  # (3, 8)

    # NOTE: corners_velo uses PointPillarsDetector._box_to_corners ordering, NOT
    # the NuScenes Box.corners() ordering that TC2's original indices (0, 4, 1)
    # assumed. In that layout, corner 0 = (-l, -w, -h) and its two in-plane
    # neighbours are corner 1 (length, +l) and corner 3 (width, +w); corner 4 is
    # the *vertical* neighbour (+h), so picking it would collapse the in-plane
    # width edge to (0, 0) and accept an infinite slab along the width axis.
    p1 = corners[:2, 0]
    p_x = corners[:2, 3]   # width edge  (corner 3 - corner 0)
    p_y = corners[:2, 1]   # length edge (corner 1 - corner 0)

    i = p_x - p1
    j = p_y - p1

    v = points_xy - p1.reshape((-1, 1))

    iv = np.dot(i, v)
    jv = np.dot(j, v)

    mask_x = np.logical_and(0 <= iv, iv <= np.dot(i, i))
    mask_y = np.logical_and(0 <= jv, jv <= np.dot(j, j))
    return np.logical_and(mask_x, mask_y)
