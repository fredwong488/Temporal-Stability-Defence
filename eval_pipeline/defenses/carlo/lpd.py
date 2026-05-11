"""
defenses/carlo/lpd.py
---------------------
Laser Penetration Detection (LPD) — first stage of CARLO.

Reference: Sun et al. 2020, §7.1.2, Equation 6.

For a detected bounding box B, LPD computes:

    g_B = |B↓| / (|B↑| + |B| + |B↓|)

where B↑ / B / B↓ partition the points in the ray-frustum of B:
  B↑ — returned point is closer to the sensor than B's near face (along ray).
  B  — returned point is inside B.
  B↓ — returned point is farther from the sensor than B's far face (penetrated).

A high g_B means many rays passed THROUGH B, consistent with a spoofed
(sparse, low-density) box.
"""

from __future__ import annotations

import numpy as np

from ...types import Prediction
from .geometry import obb_frustum_classify


def compute_lpd(
    pred: Prediction, pts_xyz: np.ndarray
) -> tuple[float, dict]:
    """Compute the LPD penetration ratio g_B for bounding box *pred*.

    # Assumption 4 (carlo_replication_spec.md §13): The LiDAR ray set L is
    # derived from returned points. Each LiDAR point p is treated as the hit
    # point of a ray fired from origin in direction p/|p|. Rays that fired but
    # returned no point (e.g. maximum-range misses) are absent. This avoids
    # requiring Velodyne HDL-64E angular specifications.

    # Assumption 2 (carlo_replication_spec.md §13): A ray belongs to F_B iff
    # its direction (p/|p|) intersects the oriented bounding box B.

    # Assumption 7 (carlo_replication_spec.md §13): B↑/B/B↓ classification
    # uses the signed distance along each ray relative to the box's near/far
    # intersection parameters t_near and t_far from the slab method.
    # B↑: t_p < t_near  (point in front of box)
    # B:  t_near ≤ t_p ≤ t_far  (point inside box)
    # B↓: t_p > t_far  (ray penetrated box, hit behind it)

    Parameters
    ----------
    pred     : one predicted bounding box from the upstream detector.
    pts_xyz  : (N, 3) LiDAR points in velodyne frame.

    Returns
    -------
    g        : float — the LPD ratio in [0, 1].  0.0 when frustum is empty.
    details  : dict — diagnostic counts; all values are JSON-serialisable.
    """
    result = obb_frustum_classify(pred, pts_xyz)
    fm = result.in_frustum
    t_near = result.t_near
    t_far = result.t_far
    t_p = result.t_p

    n_frustum = int(fm.sum())

    if n_frustum == 0:
        return 0.0, {
            "n_frustum": 0,
            "n_up": 0,
            "n_box": 0,
            "n_down": 0,
            # frustum_mask exposed so defense.py can extract frustum_pts for FSD
            "_frustum_mask": fm,
        }

    b_up = fm & (t_p < t_near)
    b_in = fm & (t_p >= t_near) & (t_p <= t_far)
    b_down = fm & (t_p > t_far)

    n_up = int(b_up.sum())
    n_in = int(b_in.sum())
    n_down = int(b_down.sum())
    total = n_up + n_in + n_down

    g = float(n_down / total) if total > 0 else 0.0

    return g, {
        "n_frustum": n_frustum,
        "n_up": n_up,
        "n_box": n_in,
        "n_down": n_down,
        # Internal key — stripped from JSON metadata by defense.py.
        "_frustum_mask": fm,
    }
