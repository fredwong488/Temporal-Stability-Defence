"""
attacks/ora.py
--------------
Object Removal Attack (ORA) for LiDAR point clouds.

Adapted from prune_obj_points() in 01-04_ORA_front-near-2.ipynb.

Algorithm
---------
For each object whose type is in target_types:
  1. Find all lidar points inside the object's axis-aligned bounding box.
  2. From those, select candidates whose azimuth angle is within
     azimuth_constraint_deg of the object's centroid direction.
  3. Randomly sample up to budget candidate points to remove.
  4. Re-inject each removed point behind the object by extending its radial
     vector from the sensor origin by a random distance in reinject_distance_range.
The modified full point cloud is returned as a new Frame.
"""

from __future__ import annotations

import random

import numpy as np

from ..base import BaseAttack
from ..types import Frame, ObjectLabel


class ORAAttack(BaseAttack):
    """Object Removal Attack — removes object points and re-injects them
    behind the bounding box to suppress object detection.

    Parameters
    ----------
    budget
        Maximum number of points to remove (and re-inject) per object.
    target_types
        KITTI class names to attack.  Defaults to {"Car"}.
    azimuth_constraint_deg
        Half-angle of the azimuth cone used to select candidate points.
        Points outside this cone (relative to the object centroid direction)
        are not perturbed.
    reinject_distance_range
        (min, max) extra distance in metres added to the radial distance when
        re-injecting points behind the object.
    seed
        Random seed for reproducibility.  None = non-deterministic.
    """

    def __init__(
        self,
        budget: int = 200,
        target_types: set[str] | None = None,
        azimuth_constraint_deg: float = 10.0,
        reinject_distance_range: tuple[float, float] = (2.0, 3.0),
        seed: int | None = None,
    ) -> None:
        self.budget = budget
        self.target_types = target_types if target_types is not None else {"Car"}
        self.azimuth_constraint_deg = azimuth_constraint_deg
        self.reinject_distance_range = reinject_distance_range
        self._rng = random.Random(seed)
        self._np_rng = np.random.default_rng(seed)

    @property
    def modality(self) -> str:
        return "lidar"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def apply(self, frame: Frame) -> Frame:
        """Return a new attacked Frame with object points pruned and re-injected."""
        pts = frame.lidar.copy()          # (N, 4) — will be modified in place on the copy
        removed_per_obj: list[dict] = []

        for label in frame.labels:
            if label.type not in self.target_types:
                continue

            pts, removed_info = self._attack_object(pts, label)
            removed_per_obj.append(removed_info)

        return frame.with_lidar(
            pts,
            is_attacked=True,
            attacked_modalities=frozenset({"lidar"}),
            attack_metadata={
                "attack": "ORA",
                "budget": self.budget,
                "target_types": list(self.target_types),
                "removed_per_obj": removed_per_obj,
            },
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _attack_object(
        self, pts: np.ndarray, label: ObjectLabel
    ) -> tuple[np.ndarray, dict]:
        """Remove and re-inject points for a single object.

        Returns the updated (N', 4) point array and a dict of removal info.
        """
        xyz = pts[:, :3]
        corners = label.corners_velo
        mins = corners.min(axis=0)   # (3,)
        maxs = corners.max(axis=0)   # (3,)

        # 1. Points inside the AABB
        in_box_mask = np.all((xyz >= mins) & (xyz <= maxs), axis=1)
        in_box_idx = np.where(in_box_mask)[0]

        if len(in_box_idx) == 0:
            return pts, {"n_removed": 0, "label_type": label.type}

        # 2. Azimuth constraint — filter to candidates near the object centroid
        centroid_xy = corners.mean(axis=0)[:2]          # (x, y) of bbox centre
        obj_azimuth = np.arctan2(centroid_xy[1], centroid_xy[0])

        candidate_idx = self._azimuth_candidates(
            xyz[in_box_idx], obj_azimuth, self.azimuth_constraint_deg
        )
        candidate_global = in_box_idx[candidate_idx]

        # 3. Sample up to budget
        n_sample = min(self.budget, len(candidate_global))
        if n_sample == 0:
            return pts, {"n_removed": 0, "label_type": label.type}

        chosen = np.array(
            self._rng.sample(candidate_global.tolist(), n_sample), dtype=int
        )

        # 4. Build re-injected points
        removed_pts = pts[chosen, :3]                   # (n_sample, 3)
        removed_int = pts[chosen, 3]                    # (n_sample,)
        reinjected = self._reinject(removed_pts, removed_int)

        # 5. Delete chosen rows and append reinjected
        keep_mask = np.ones(len(pts), dtype=bool)
        keep_mask[chosen] = False
        pts = np.concatenate([pts[keep_mask], reinjected], axis=0)

        return pts, {
            "label_type": label.type,
            "n_removed": n_sample,
        }

    def _azimuth_candidates(
        self,
        pts_xyz: np.ndarray,
        obj_azimuth: float,
        constraint_deg: float,
    ) -> np.ndarray:
        """Return indices (into pts_xyz) within the azimuth constraint cone.

        Mirrors the candidate selection in prune_obj_points():
        a point is a candidate if its y-distance from the object's radial line
        is within the chord length for constraint_deg at its range.
        """
        x = pts_xyz[:, 0]
        y = pts_xyz[:, 1]
        rad = np.sqrt(x ** 2 + y ** 2)
        # Chord length at range `rad` for angle `constraint_deg`
        theta = np.radians(constraint_deg)
        chord = np.sqrt(2 * rad ** 2 - 2 * rad ** 2 * np.cos(theta))

        # The notebook uses start_y as the object's max-y; approximate with centroid
        start_y = pts_xyz[:, 1].max() if len(pts_xyz) > 0 else 0.0
        within = y >= (start_y - chord)
        return np.where(within)[0]

    def _reinject(self, pts_xyz: np.ndarray, intensity: np.ndarray) -> np.ndarray:
        """Re-inject removed points behind the object along the radial direction."""
        norms = np.linalg.norm(pts_xyz, axis=1, keepdims=True)       # (n, 1)
        # Avoid division by zero for points at origin
        norms = np.where(norms == 0, 1.0, norms)
        unit_vecs = pts_xyz / norms                                   # (n, 3)

        extra = self._np_rng.uniform(
            self.reinject_distance_range[0],
            self.reinject_distance_range[1],
            size=(len(pts_xyz), 1),
        )
        new_xyz = pts_xyz + unit_vecs * extra                         # (n, 3)
        new_pts = np.hstack([new_xyz, intensity.reshape(-1, 1)])      # (n, 4)
        return new_pts.astype(np.float32)
