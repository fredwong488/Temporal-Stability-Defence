"""
defenses/void_region.py
-----------------------
Void-region attack detector.

Adapted from detection_void_region-Copy1.ipynb.

Algorithm
---------
1. Extract ground-level points from the lidar scan.
2. Build a 2D occupancy grid over a region of interest (ROI).
3. Identify empty (unoccupied) grid cells.
4. Cluster empty cells with DBSCAN.
5. For each cluster centroid, backtrace a frustum from the sensor origin
   through the void region.
6. Count lidar points inside the frustum.
7. Check whether those points overlap any ground-truth bounding box.
8. If a void region's frustum intersects a labelled object, flag an attack.
"""

from __future__ import annotations

import math
import pathlib
import sys
from collections import deque

import numpy as np
from sklearn.cluster import DBSCAN

# Geometry primitives from project root
_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from ..utils.utils_3d import Face, Vector, isInPoly  # noqa: E402

from ..base import BaseDefense
from ..types import DetectionResult, Frame, ObjectLabel


class VoidRegionDefense(BaseDefense):
    """Detect adversarial object removal attacks by finding anomalous void
    regions in the LiDAR ground plane and backtracing their shadow frustums.

    Parameters
    ----------
    roi_min
        (x_min, y_min) of the region of interest in velodyne metres.
    roi_max
        (x_max, y_max) of the region of interest.
    ground_height
        Sensor-relative height of the ground plane (default -1.73 m for KITTI).
    ground_delta
        Half-thickness of the ground-plane slice to extract.
    grid_stride
        Occupancy grid cell size in metres.
    dbscan_eps
        DBSCAN neighbourhood radius.
    dbscan_min_samples
        DBSCAN minimum cluster size.
    min_frustum_points
        Minimum number of lidar points inside a frustum to consider it
        non-empty (frustums with fewer points are ignored).
    """

    def __init__(
        self,
        roi_min: tuple[float, float] = (4.5, -5.0),
        roi_max: tuple[float, float] = (30.0, 5.0),
        ground_height: float = -1.73,
        ground_delta: float = 0.2,
        grid_stride: float = 0.6,
        dbscan_eps: float = 0.8,
        dbscan_min_samples: int = 5,
        min_frustum_points: int = 0,
    ) -> None:
        self.roi_min = roi_min
        self.roi_max = roi_max
        self.ground_height = ground_height
        self.ground_delta = ground_delta
        self.grid_stride = grid_stride
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_samples = dbscan_min_samples
        self.min_frustum_points = min_frustum_points

    @property
    def temporal_window(self) -> int:
        return 1  # stateless — operates on the current frame only

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, frame: Frame, history: deque[Frame]) -> DetectionResult:
        """Determine whether the frame contains an adversarial void region."""
        lidar = frame.lidar  # (N, 4)

        # 1. Ground slice
        ground_pts = self._extract_ground_points(lidar)

        # 2. Occupancy grid → empty cells
        empty_centers = self._find_empty_cells(ground_pts)

        if len(empty_centers) == 0:
            return DetectionResult(
                is_attack_detected=False,
                confidence=0.0,
                metadata={"reason": "no_empty_cells"},
            )

        # 3. DBSCAN clustering of empty cells
        pts_2d = empty_centers[:, :2]
        labels = DBSCAN(
            eps=self.dbscan_eps, min_samples=self.dbscan_min_samples
        ).fit_predict(pts_2d)

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

        if n_clusters == 0:
            return DetectionResult(
                is_attack_detected=False,
                confidence=0.0,
                metadata={"n_clusters": 0, "reason": "no_clusters"},
            )

        # 4. For each cluster — backtrace frustum, check GT labels
        clusters = self._get_clusters(empty_centers, labels, n_clusters)
        matched_obj_indices: list[int] = []
        cluster_details: list[dict] = []

        for cluster in clusters:
            centroid = np.mean(cluster, axis=0)
            frustum_result = self._backtrace_frustum(centroid, lidar[:, :3])
            pts_in = frustum_result["pts"]

            matched = self._check_frustum_vs_labels(pts_in, frame.labels)
            matched_obj_indices.extend(matched)

            cluster_details.append({
                "centroid": centroid.tolist(),
                "cluster_size": len(cluster),
                "frustum_pt_count": frustum_result["count"],
                "matched_objects": matched,
            })

        attack_detected = len(matched_obj_indices) > 0
        confidence = float(len(set(matched_obj_indices))) / max(len(frame.labels), 1)

        return DetectionResult(
            is_attack_detected=attack_detected,
            confidence=min(confidence, 1.0),
            metadata={
                "n_clusters": n_clusters,
                "n_empty_cells": len(empty_centers),
                "matched_obj_indices": list(set(matched_obj_indices)),
                "cluster_details": cluster_details,
            },
        )

    # ------------------------------------------------------------------
    # Ground extraction
    # ------------------------------------------------------------------

    def _extract_ground_points(self, lidar: np.ndarray) -> np.ndarray:
        """Return points within ground_delta of ground_height."""
        z = lidar[:, 2]
        mask = (z >= self.ground_height - self.ground_delta) & \
               (z <= self.ground_height + self.ground_delta)
        return lidar[mask, :3]

    # ------------------------------------------------------------------
    # Occupancy grid
    # ------------------------------------------------------------------

    def _find_empty_cells(self, ground_pts: np.ndarray) -> np.ndarray:
        """Build a 2D occupancy grid over the ROI and return unoccupied cell centres."""
        x_min, y_min = self.roi_min
        x_max, y_max = self.roi_max
        stride = self.grid_stride

        x_centers = np.arange(x_min, x_max, stride)
        y_centers = np.arange(y_min, y_max, stride)

        if len(ground_pts) == 0:
            # All cells empty if no ground points
            centers = [
                [x, y, self.ground_height]
                for x in x_centers
                for y in y_centers
            ]
            return np.array(centers)

        pts_x = ground_pts[:, 0]
        pts_y = ground_pts[:, 1]

        # Bin each ground point into the grid (vectorized)
        half = stride / 2.0
        xi = np.digitize(pts_x, x_centers + half)  # which x-cell
        yi = np.digitize(pts_y, y_centers + half)  # which y-cell

        # Build occupied set
        occupied: set[tuple[int, int]] = set(zip(xi.tolist(), yi.tolist()))

        empty: list[list[float]] = []
        for ix, x in enumerate(x_centers):
            for iy, y in enumerate(y_centers):
                if (ix, iy) not in occupied:
                    # Additionally check points are in ROI
                    if x_min <= x <= x_max and y_min <= y <= y_max:
                        empty.append([x, y, self.ground_height])

        return np.array(empty) if empty else np.empty((0, 3))

    # ------------------------------------------------------------------
    # Cluster extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _get_clusters(
        empty_centers: np.ndarray,
        labels: np.ndarray,
        n_clusters: int,
    ) -> list[np.ndarray]:
        clusters: list[np.ndarray] = []
        for i in range(n_clusters):
            mask = labels == i
            clusters.append(empty_centers[mask])
        return clusters

    # ------------------------------------------------------------------
    # Frustum backtrace
    # ------------------------------------------------------------------

    def _backtrace_frustum(
        self, centroid: np.ndarray, pts_xyz: np.ndarray
    ) -> dict:
        """Construct a frustum from the sensor origin through the void centroid
        and return the lidar points inside it.

        Adapted directly from backtrace_shadow_from_pt() in the notebook.
        The frustum is a small wedge-shaped volume defined by 8 vertices;
        we pre-filter with an AABB and then apply the exact isInPoly test.
        """
        cx, cy, cz = float(centroid[0]), float(centroid[1]), float(centroid[2])

        angle_rad = np.arctan(abs(cy) / abs(cx)) if cx != 0 else math.pi / 2
        angle_new = (math.pi / 2) - angle_rad
        delta_y = 0.1 * math.sin(angle_new)
        delta_x = 0.1 * math.cos(angle_new)

        points_top = [
            [-0.1,  0.0,  0.0],
            [-0.1,  0.0, -0.1],
            [ 0.1,  0.0,  0.0],
            [ 0.1,  0.0, -0.1],
            [cx - delta_x, cy - delta_y, cz + 0.1 + 0.5],
            [cx - delta_x, cy - delta_y, cz + 0.1],
            [cx + delta_x, cy + delta_y, cz + 0.1 + 0.5],
            [cx + delta_x, cy + delta_y, cz + 0.1],
        ]
        points_bot = [
            [-0.1,  0.0,  0.0],
            [-0.1,  0.0, -0.1],
            [ 0.1,  0.0,  0.0],
            [ 0.1,  0.0, -0.1],
            [cx + delta_x, cy + delta_y, cz + 0.1 + 0.5],
            [cx + delta_x, cy + delta_y, cz + 0.1],
            [cx - delta_x, cy - delta_y, cz + 0.1 + 0.5],
            [cx - delta_x, cy - delta_y, cz + 0.1],
        ]

        if math.atan2(cy, cx) <= 0:
            points = points_top
            f1 = Face([Vector(points[0]), Vector(points[2]), Vector(points[3]), Vector(points[1])])
            f2 = Face([Vector(points[5]), Vector(points[7]), Vector(points[6]), Vector(points[4])])
            f3 = Face([Vector(points[0]), Vector(points[4]), Vector(points[6]), Vector(points[2])])
            f4 = Face([Vector(points[7]), Vector(points[5]), Vector(points[1]), Vector(points[3])])
            f5 = Face([Vector(points[6]), Vector(points[7]), Vector(points[3]), Vector(points[2])])
            f6 = Face([Vector(points[0]), Vector(points[1]), Vector(points[5]), Vector(points[4])])
        else:
            points = points_bot
            f1 = Face([Vector(points[1]), Vector(points[3]), Vector(points[2]), Vector(points[0])])
            f2 = Face([Vector(points[4]), Vector(points[6]), Vector(points[7]), Vector(points[5])])
            f3 = Face([Vector(points[2]), Vector(points[6]), Vector(points[4]), Vector(points[0])])
            f4 = Face([Vector(points[3]), Vector(points[1]), Vector(points[5]), Vector(points[7])])
            f5 = Face([Vector(points[2]), Vector(points[3]), Vector(points[7]), Vector(points[6])])
            f6 = Face([Vector(points[4]), Vector(points[5]), Vector(points[1]), Vector(points[0])])

        poly = [f1, f2, f3, f4, f5, f6]

        # AABB pre-filter for speed
        arr = np.array(points)
        aabb_min = arr.min(axis=0)
        aabb_max = arr.max(axis=0)
        pre_mask = np.all((pts_xyz >= aabb_min) & (pts_xyz <= aabb_max), axis=1)
        candidates = pts_xyz[pre_mask]

        pts_in: list[np.ndarray] = []
        for pt in candidates:
            if isInPoly(pt.tolist(), poly):
                pts_in.append(pt)

        return {"count": len(pts_in), "pts": pts_in}

    # ------------------------------------------------------------------
    # Label intersection check
    # ------------------------------------------------------------------

    @staticmethod
    def _check_frustum_vs_labels(
        pts_in_frustum: list[np.ndarray],
        labels: list[ObjectLabel],
    ) -> list[int]:
        """Return indices of labels whose AABB contains at least one frustum point."""
        if not pts_in_frustum:
            return []

        pts = np.array(pts_in_frustum)          # (K, 3)
        matched: list[int] = []

        for i, label in enumerate(labels):
            corners = label.corners_velo         # (8, 3)
            mins = corners.min(axis=0)
            maxs = corners.max(axis=0)
            in_box = np.all((pts >= mins) & (pts <= maxs), axis=1)
            if in_box.any():
                matched.append(i)

        return matched
