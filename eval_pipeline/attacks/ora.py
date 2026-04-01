"""
attacks/ora.py
--------------
Object Removal Attack (ORA) for LiDAR point clouds.

Two implementations are provided:

ORAAttack
    Corrected implementation.  Uses the true per-point 2D range for the azimuth
    constraint and compares azimuth angles directly against the object centroid
    direction.

ORAAttackNotebook
    Faithful re-implementation of prune_obj_points() from
    01-04_ORA_front-near-2.ipynb.  Reproduces both quirks of the original:
      - radius uses the point's x mixed with a fixed start_y (bbox max-y)
        rather than the true per-point 2D range
      - start_y is the bbox max-y, not the centroid y, so the cone is
        anchored to the left edge of the object rather than its centre
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

        A point is a candidate if its azimuth angle (from the sensor origin) is
        within constraint_deg of the object centroid's azimuth direction.
        """
        x = pts_xyz[:, 0]
        y = pts_xyz[:, 1]
        pt_azimuth = np.arctan2(y, x)
        diff = pt_azimuth - obj_azimuth
        # Wrap to [-pi, pi] to handle the ±180° boundary
        diff = (diff + np.pi) % (2 * np.pi) - np.pi
        within = np.abs(diff) <= np.radians(constraint_deg)
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


class ORAAttackNotebook(BaseAttack):
    """Faithful re-implementation of prune_obj_points() from
    01-04_ORA_front-near-2.ipynb.

    Reproduces the notebook behaviour exactly, including its two quirks:

    1. The azimuth-filter radius is computed as ``sqrt(pt_x² + start_y²)``
       where ``start_y`` is the bbox max-y — not the true per-point 2D range.
    2. The cone is anchored to ``start_y`` (bbox max-y / left edge of the
       object) rather than the centroid y, so the filter is off-centre.

    Parameters
    ----------
    budget, target_types, reinject_distance_range, seed
        Same as ORAAttack.  azimuth_constraint_deg is fixed at 10° to match
        the notebook.
    """

    def __init__(
        self,
        budget: int = 200,
        target_types: set[str] | None = None,
        reinject_distance_range: tuple[float, float] = (2.0, 3.0),
        seed: int | None = None,
    ) -> None:
        self.budget = budget
        self.target_types = target_types if target_types is not None else {"Car"}
        self.reinject_distance_range = reinject_distance_range
        self._rng = random.Random(seed)
        self._np_rng = np.random.default_rng(seed)

    @property
    def modality(self) -> str:
        return "lidar"

    def apply(self, frame: Frame) -> Frame:
        """Return a new attacked Frame with object points pruned and re-injected.

        Reproduces the notebook's bulk-removal scope: all attacks are planned
        against the original point cloud, then all bbox points are removed at
        once, and all attacked object clouds are appended.  This prevents
        reinjected points from one object being re-attacked when they happen to
        fall inside another object's bounding box.
        """
        pts = frame.lidar.copy()

        # Pass 1 — plan every attack against the ORIGINAL point cloud.
        plans: list[dict] = []
        for label in frame.labels:
            if label.type not in self.target_types:
                continue
            plan = self._plan_attack(pts, label)
            plans.append(plan)

        # Pass 2 — bulk-remove ALL bbox points across all objects.
        all_bbox_idx: set[int] = set()
        for plan in plans:
            all_bbox_idx.update(plan["bbox_idx"].tolist())

        keep_mask = np.ones(len(pts), dtype=bool)
        for idx in all_bbox_idx:
            keep_mask[idx] = False
        pts = pts[keep_mask]

        # Pass 3 — append each object's attacked cloud (non-sampled + reinjected).
        for plan in plans:
            pts = np.concatenate([pts, plan["attacked_pts"]], axis=0)

        return frame.with_lidar(
            pts,
            is_attacked=True,
            attacked_modalities=frozenset({"lidar"}),
            attack_metadata={
                "attack": "ORANotebook",
                "budget": self.budget,
                "target_types": list(self.target_types),
                "removed_per_obj": [
                    {"label_type": p["label_type"], "n_removed": p["n_removed"]}
                    for p in plans
                ],
            },
        )

    def _plan_attack(self, pts: np.ndarray, label: ObjectLabel) -> dict:
        """Plan the attack for one object against the original point cloud.

        Returns a dict with:
          bbox_idx    — indices of ALL points inside the AABB (to bulk-remove)
          attacked_pts — (N', 4) array: non-sampled bbox points + reinjected
          n_removed   — number of points sampled for removal
          label_type  — object class string
        """
        xyz = pts[:, :3]
        corners = label.corners_velo
        mins = corners.min(axis=0)
        maxs = corners.max(axis=0)

        in_box_mask = np.all((xyz >= mins) & (xyz <= maxs), axis=1)
        in_box_idx = np.where(in_box_mask)[0]
        in_box_pts = pts[in_box_idx]          # (M, 4) — all bbox points

        if len(in_box_idx) == 0:
            return {
                "bbox_idx": in_box_idx,
                "attacked_pts": np.empty((0, 4), dtype=np.float32),
                "n_removed": 0,
                "label_type": label.type,
            }

        # Azimuth filter (notebook quirks preserved)
        start_y = in_box_pts[:, 1].max()
        candidate_local = self._azimuth_candidates_notebook(in_box_pts[:, :3], start_y)

        n_sample = min(self.budget, len(candidate_local))

        if n_sample == 0:
            return {
                "bbox_idx": in_box_idx,
                "attacked_pts": in_box_pts.copy(),
                "n_removed": 0,
                "label_type": label.type,
            }

        # Sample indices local to in_box_pts
        chosen_local = np.array(
            self._rng.sample(candidate_local.tolist(), n_sample), dtype=int
        )

        reinjected = self._reinject(in_box_pts[chosen_local, :3], in_box_pts[chosen_local, 3])

        keep_local = np.ones(len(in_box_pts), dtype=bool)
        keep_local[chosen_local] = False
        attacked_pts = np.concatenate([in_box_pts[keep_local], reinjected], axis=0)

        return {
            "bbox_idx": in_box_idx,
            "attacked_pts": attacked_pts,
            "n_removed": n_sample,
            "label_type": label.type,
        }

    def _azimuth_candidates_notebook(
        self, pts_xyz: np.ndarray, start_y: float
    ) -> np.ndarray:
        """Notebook candidate filter (prune_obj_points lines verbatim).

        radius = sqrt(pt_x² + start_y²)  — mixes point's x with fixed start_y.
        chord  = sqrt(2r² - 2r²·cos(10°))
        admit if pt_y >= start_y - chord
        """
        x = pts_xyz[:, 0]
        y = pts_xyz[:, 1]
        rad = np.sqrt(x ** 2 + start_y ** 2)
        chord = np.sqrt(2 * rad ** 2 - 2 * rad ** 2 * np.cos(np.radians(10)))
        within = y >= (start_y - chord)
        return np.where(within)[0]

    def _reinject(self, pts_xyz: np.ndarray, intensity: np.ndarray) -> np.ndarray:
        """Re-inject removed points behind the object along the radial direction."""
        norms = np.linalg.norm(pts_xyz, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        unit_vecs = pts_xyz / norms

        extra = self._np_rng.uniform(
            self.reinject_distance_range[0],
            self.reinject_distance_range[1],
            size=(len(pts_xyz), 1),
        )
        new_xyz = pts_xyz + unit_vecs * extra
        new_pts = np.hstack([new_xyz, intensity.reshape(-1, 1)])
        return new_pts.astype(np.float32)
