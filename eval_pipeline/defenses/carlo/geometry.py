"""
defenses/carlo/geometry.py
--------------------------
OBB-ray intersection and point-in-OBB tests for CARLO.

All geometry is in the velodyne frame (sensor at origin).
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from ...types import Prediction


class OBBFrustumResult(NamedTuple):
    """Per-point result of the OBB frustum test.

    For N input points:
      in_frustum : bool (N,)  — True if the ray from origin through p hits B.
      t_near     : float (N,) — ray param at B entry (box front face).
      t_far      : float (N,) — ray param at B exit  (box back face).
      t_p        : float (N,) — |p|, the param at which the laser returned.

    B↑ : in_frustum & (t_p < t_near)    — point in front of the box
    B  : in_frustum & (t_near ≤ t_p ≤ t_far) — point inside the box
    B↓ : in_frustum & (t_p > t_far)     — point behind the box (penetrated)
    """

    in_frustum: np.ndarray
    t_near: np.ndarray
    t_far: np.ndarray
    t_p: np.ndarray


def obb_frustum_classify(
    pred: Prediction, pts_xyz: np.ndarray
) -> OBBFrustumResult:
    """Test all N points against the OBB of *pred* using the slab method.

    The ray for point p goes from the sensor origin (0,0,0) in direction
    p / |p| and is treated as infinite for frustum membership, consistent
    with Algorithm 1 line 5 of the CARLO paper.

    # Assumption 2 (carlo_replication_spec.md §13): A ray belongs to F_B iff
    # the infinite ray from origin in direction p/|p| intersects box B.
    # Assumption 1 (carlo_replication_spec.md §13): All geometry is in the
    # velodyne sensor frame; the sensor origin is (0, 0, 0).
    """
    N = len(pts_xyz)

    # Ray parameter at the returned point = Euclidean distance from sensor.
    t_p = np.linalg.norm(pts_xyz, axis=1)  # (N,)
    valid = t_p > 1e-9

    # Unit ray directions (safe division — zero-vectors get direction 0).
    safe_t = np.where(valid, t_p, 1.0)
    directions = pts_xyz / safe_t[:, np.newaxis]  # (N, 3)

    center = np.array([pred.x, pred.y, pred.z], dtype=float)
    half_extents = np.array(
        [pred.length / 2.0, pred.width / 2.0, pred.height / 2.0], dtype=float
    )
    ry = float(pred.rotation_y)

    cos_ry = np.cos(ry)
    sin_ry = np.sin(ry)
    # R_inv = R(-ry): rotates world vectors into the OBB's local coordinate frame.
    # The box's local x-axis is (cos ry, sin ry, 0) in world, so its inverse is a
    # CCW rotation by -ry around z.
    R_inv = np.array(
        [
            [cos_ry, sin_ry, 0.0],
            [-sin_ry, cos_ry, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )

    # In OBB local frame the ray is: q(t) = o_local + t * d_local, where
    #   o_local = R_inv @ (origin − center)  [world origin in box-local coords]
    #   d_local = R_inv @ d                  [ray direction in box-local coords]
    o_local = R_inv @ (-center)  # (3,)
    d_local = directions @ R_inv.T  # (N, 3)

    t_near = np.full(N, -np.inf)
    t_far = np.full(N, np.inf)

    for i in range(3):
        d_i = d_local[:, i]
        o_i = o_local[i]
        he_i = half_extents[i]

        nonparallel = np.abs(d_i) > 1e-9
        safe_d = np.where(nonparallel, d_i, 1.0)

        t0 = (-he_i - o_i) / safe_d
        t1 = (he_i - o_i) / safe_d

        t_near_i = np.where(nonparallel, np.minimum(t0, t1), -np.inf)

        outside_i = (o_i < -he_i) | (o_i > he_i)
        t_far_i = np.where(
            nonparallel,
            np.maximum(t0, t1),
            np.where(outside_i, -np.inf, np.inf),
        )

        t_near = np.maximum(t_near, t_near_i)
        t_far = np.minimum(t_far, t_far_i)

    # A ray hits B iff t_near ≤ t_far and the box is in front of the sensor.
    in_frustum = (t_near <= t_far) & (t_far > 0.0) & valid

    return OBBFrustumResult(
        in_frustum=in_frustum,
        t_near=t_near,
        t_far=t_far,
        t_p=t_p,
    )


def points_in_obb(pts: np.ndarray, pred: Prediction) -> np.ndarray:
    """Return bool (N,) mask — True for each point inside the OBB of *pred*.

    Transforms points into the OBB's local coordinate frame and tests against
    the axis-aligned half-extents.
    """
    center = np.array([pred.x, pred.y, pred.z], dtype=float)
    half_extents = np.array(
        [pred.length / 2.0, pred.width / 2.0, pred.height / 2.0], dtype=float
    )
    ry = float(pred.rotation_y)
    cos_ry, sin_ry = np.cos(ry), np.sin(ry)
    R_inv = np.array(
        [
            [cos_ry, sin_ry, 0.0],
            [-sin_ry, cos_ry, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    local = (pts - center) @ R_inv.T  # (N, 3) in OBB local frame
    return np.all(np.abs(local) <= half_extents, axis=1)
