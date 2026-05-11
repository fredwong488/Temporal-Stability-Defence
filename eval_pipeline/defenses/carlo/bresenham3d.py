"""
defenses/carlo/bresenham3d.py
-----------------------------
3D voxel traversal (Amanatides–Woo algorithm) used by CARLO's Free Space
Detection stage to mark cells as free along each laser ray.

# Assumption 6 (carlo_replication_spec.md §13): The paper cites Bresenham [16]
# for the cell-traversal step. We implement the Amanatides–Woo (1987)
# ray-voxel traversal, which is the standard 3D extension and produces
# identical free-cell sets in practice.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


def traverse_segment(
    end: "np.ndarray", voxel_size: float
) -> set[tuple[int, int, int]]:
    """Return all voxel indices (ix, iy, iz) the segment (origin → *end*) passes through.

    Each integer index triple corresponds to the world-space voxel
    [ix·vs, (ix+1)·vs) × [iy·vs, (iy+1)·vs) × [iz·vs, (iz+1)·vs).

    The grid is anchored at the LiDAR origin:
    # Assumption 5 (carlo_replication_spec.md §13): The voxel grid is aligned
    # per-frustum to the LiDAR origin, so voxel (ix, iy, iz) spans
    # [ix·vs, (ix+1)·vs) in each axis. Negative indices cover space below or
    # behind the sensor (e.g. ground level at z ≈ -1.73 m gives iz = -7).

    Parameters
    ----------
    end        : (3,) world-frame endpoint — the LiDAR hit point p⃗.
    voxel_size : edge length of each cubic voxel in metres (paper: 0.25 m).
    """
    vs = float(voxel_size)
    x1, y1, z1 = float(end[0]), float(end[1]), float(end[2])

    dx, dy, dz = x1, y1, z1  # displacement from origin (0,0,0) to end

    if dx * dx + dy * dy + dz * dz < 1e-18:
        return set()

    # Starting voxel (always the origin voxel)
    ix, iy, iz = 0, 0, 0  # math.floor(0 / vs) == 0

    # Ending voxel
    ex = math.floor(x1 / vs)
    ey = math.floor(y1 / vs)
    ez = math.floor(z1 / vs)

    step_x = 1 if dx > 0.0 else (-1 if dx < 0.0 else 0)
    step_y = 1 if dy > 0.0 else (-1 if dy < 0.0 else 0)
    step_z = 1 if dz > 0.0 else (-1 if dz < 0.0 else 0)

    def _first_t(d: float, step: int, i: int) -> float:
        """Ray parameter at which the current voxel boundary is first crossed."""
        if d == 0.0:
            return math.inf
        # For i=0 and step=-1 (d<0) this correctly gives 0.0: the ray is already
        # on the left boundary of voxel 0 and exits immediately.
        return ((i + 1) * vs) / d if step > 0 else (i * vs) / d

    t_max_x = _first_t(dx, step_x, ix)
    t_max_y = _first_t(dy, step_y, iy)
    t_max_z = _first_t(dz, step_z, iz)

    t_delta_x = (vs / abs(dx)) if dx != 0.0 else math.inf
    t_delta_y = (vs / abs(dy)) if dy != 0.0 else math.inf
    t_delta_z = (vs / abs(dz)) if dz != 0.0 else math.inf

    visited: set[tuple[int, int, int]] = {(ix, iy, iz)}

    # Manhattan distance in voxel space upper-bounds the step count.
    max_steps = abs(ex - ix) + abs(ey - iy) + abs(ez - iz) + 3

    for _ in range(max_steps):
        if ix == ex and iy == ey and iz == ez:
            break
        if t_max_x < t_max_y:
            if t_max_x < t_max_z:
                ix += step_x
                t_max_x += t_delta_x
            else:
                iz += step_z
                t_max_z += t_delta_z
        else:
            if t_max_y < t_max_z:
                iy += step_y
                t_max_y += t_delta_y
            else:
                iz += step_z
                t_max_z += t_delta_z
        visited.add((ix, iy, iz))

    return visited
