"""
defenses/carlo/free_space.py
-----------------------------
Free Space Detection (FSD) — fallback second stage of CARLO.

Reference: Sun et al. 2020, §7.1.1, Equation 5.

For a detected bounding box B, FSD computes:

    f_B = |free cells ∩ B| / |B|

where:
  |B|           = number of 0.25³ m³ voxels whose centres lie inside B.
  free cells    = voxels traversed by at least one laser ray on its way from
                  the sensor origin to a frustum hit point (3D Bresenham mark).

A high f_B means much of B's interior is traversed by rays that hit something
farther away, i.e. the box is largely hollow — consistent with a spoofed vehicle.

This is CARLO's Free Space Detection (Sun et al. 2020). It is distinct from the
Fake Shadow Detection (Cao et al. 2022) implemented in defenses/fsd.py.
"""

from __future__ import annotations

import math

import numpy as np

from ...types import Prediction
from .bresenham3d import traverse_segment
from .geometry import points_in_obb


def compute_fsd(
    pred: Prediction,
    frustum_pts: np.ndarray,
    voxel_size: float = 0.25,
) -> tuple[float, dict]:
    """Compute the FSD free-space ratio f_B for bounding box *pred*.

    # Assumption 5 (carlo_replication_spec.md §13): The voxel grid is anchored
    # at the LiDAR origin. Voxel (ix, iy, iz) spans [ix·vs, (ix+1)·vs) in
    # each axis. The paper specifies the cell size (0.25 m) but not the grid
    # origin; anchoring at the sensor is the natural choice.

    Parameters
    ----------
    pred          : one predicted bounding box (OBB in velodyne frame).
    frustum_pts   : (M, 3) LiDAR points whose rays intersect *pred* (F_B).
                    Obtained from lpd.compute_lpd's ``_frustum_mask``.
    voxel_size    : cubic cell edge length in metres; paper specifies 0.25 m.

    Returns
    -------
    f             : float — the FSD ratio in [0, 1].  0.0 when |B| = 0.
    details       : dict — diagnostic counts; all values are JSON-serialisable.
    """
    vs = float(voxel_size)

    # ---------------------------------------------------------------
    # 1. Enumerate voxel centres inside the OBB B.
    # ---------------------------------------------------------------
    corners = pred.corners_velo  # (8, 3)
    aabb_min = corners.min(axis=0)  # (3,) axis-aligned bounding box of the OBB
    aabb_max = corners.max(axis=0)

    # Integer voxel index range covering the OBB's AABB.
    ix_min = math.floor(aabb_min[0] / vs)
    ix_max = math.floor(aabb_max[0] / vs)
    iy_min = math.floor(aabb_min[1] / vs)
    iy_max = math.floor(aabb_max[1] / vs)
    iz_min = math.floor(aabb_min[2] / vs)
    iz_max = math.floor(aabb_max[2] / vs)

    if ix_max < ix_min or iy_max < iy_min or iz_max < iz_min:
        return 0.0, {"n_box_voxels": 0, "n_free_in_box": 0}

    half = vs / 2.0
    ixs = range(ix_min, ix_max + 1)
    iys = range(iy_min, iy_max + 1)
    izs = range(iz_min, iz_max + 1)

    # Build (K, 3) array of candidate voxel centres (from AABB grid).
    indices = [
        (ix, iy, iz)
        for ix in ixs
        for iy in iys
        for iz in izs
    ]
    if not indices:
        return 0.0, {"n_box_voxels": 0, "n_free_in_box": 0}

    candidates = np.array(
        [[ix * vs + half, iy * vs + half, iz * vs + half] for ix, iy, iz in indices],
        dtype=float,
    )

    # Filter to centres that lie inside the oriented box B.
    in_obb = points_in_obb(candidates, pred)
    box_voxel_indices: set[tuple[int, int, int]] = {
        indices[k] for k in range(len(indices)) if in_obb[k]
    }
    n_box_voxels = len(box_voxel_indices)

    if n_box_voxels == 0:
        return 0.0, {"n_box_voxels": 0, "n_free_in_box": 0}

    # ---------------------------------------------------------------
    # 2. Mark free voxels via 3D Bresenham traversal.
    #    Each frustum ray is traversed from origin (0,0,0) to its hit point.
    # ---------------------------------------------------------------
    free_voxels: set[tuple[int, int, int]] = set()
    for p in frustum_pts:
        free_voxels |= traverse_segment(p, vs)

    # ---------------------------------------------------------------
    # 3. Equation 5: f_B = |free cells ∩ B| / |B|
    # ---------------------------------------------------------------
    n_free_in_box = len(box_voxel_indices & free_voxels)
    f = float(n_free_in_box) / float(n_box_voxels)

    return f, {"n_box_voxels": n_box_voxels, "n_free_in_box": n_free_in_box}
