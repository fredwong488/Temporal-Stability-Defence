"""
defenses/void_region.py
-----------------------
Void-region attack detector.

Reference
---------
Hau, Z., Demetriou, S., & Lupu, E. C. (2022).
Using 3D Shadows to Detect Object Hiding Attacks on Autonomous Vehicle Perception.
IEEE Security and Privacy Workshops (SPW), pp. 229–235.
https://doi.org/10.1109/SPW54247.2022.9833890

BibTeX::

    @inproceedings{Hau2022using,
      author    = {Hau, Zhongyuan and Demetriou, Soteris and Lupu, Emil C.},
      booktitle = {2022 IEEE Security and Privacy Workshops (SPW)},
      title     = {Using 3D Shadows to Detect Object Hiding Attacks on Autonomous Vehicle Perception},
      year      = {2022},
      pages     = {229--235},
      doi       = {10.1109/SPW54247.2022.9833890},
      publisher = {IEEE Computer Society},
      month     = {May}
    }

Adapted from detection_void_region-Copy1.ipynb.

Algorithm
---------
1. Extract ground-level points from the lidar scan.
2. Build a 2D occupancy grid over a region of interest (ROI).
3. Identify empty (unoccupied) grid cells.
4. Cluster empty cells with DBSCAN.
5. For each cluster, backtrace one frustum per cell centre (not per-cluster
   centroid) and union the resulting point indices across the cluster.
6. Exclude frustum points already inside a detector-predicted bounding box.
7. DBSCAN the remaining (unidentified) points into obstacle clusters.
8. If any obstacle cluster survives, flag an attack.
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
from ..types import DetectionResult, Frame, ObjectLabel, Prediction


class VoidRegionDefense(BaseDefense):
    """Detect adversarial object removal attacks by finding anomalous void
    regions in the LiDAR ground plane and backtracing their shadow frustums.

    Parameters
    ----------
    roi_min
        (x_min, y_min) of the region of interest in velodyne metres.
        Paper specifies a 10 m × 30 m front-near region starting at x=0.
    roi_max
        (x_max, y_max) of the region of interest.
    ground_height
        Sensor-relative height of the ground plane (default -1.73 m for KITTI).
    ground_delta
        Half-thickness of the ground-plane slice to extract.
    grid_stride
        Occupancy grid cell size in metres. Paper specifies 0.3 m.
    dbscan_eps
        DBSCAN neighbourhood radius for shadow (empty-cell) clustering.
    dbscan_min_samples
        DBSCAN minimum cluster size for shadow clustering.
    min_frustum_points
        Minimum lidar points in a single cell's frustum for that cell to
        contribute to the cluster aggregate.  Cells below this threshold
        are skipped.
    obstacle_dbscan_eps
        DBSCAN neighbourhood radius for the second-pass obstacle clustering
        (applied to frustum points not explained by detector predictions).
    obstacle_dbscan_min_samples
        DBSCAN minimum cluster size for the second-pass obstacle clustering.
    """

    def __init__(
        self,
        roi_min: tuple[float, float] = (0.0, -5.0),
        roi_max: tuple[float, float] = (30.0, 5.0),
        ground_height: float = -1.73,
        ground_delta: float = 0.2,
        grid_stride: float = 0.3,
        dbscan_eps: float = 0.8,
        dbscan_min_samples: int = 5,
        min_frustum_points: int = 0,
        obstacle_dbscan_eps: float = 0.5,
        obstacle_dbscan_min_samples: int = 10,
    ) -> None:
        self.roi_min = roi_min
        self.roi_max = roi_max
        self.ground_height = ground_height
        self.ground_delta = ground_delta
        self.grid_stride = grid_stride
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_samples = dbscan_min_samples
        self.min_frustum_points = min_frustum_points
        self.obstacle_dbscan_eps = obstacle_dbscan_eps
        self.obstacle_dbscan_min_samples = obstacle_dbscan_min_samples

    @property
    def temporal_window(self) -> int:
        return 1  # stateless — operates on the current frame only

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, frame: Frame, history: deque[Frame]) -> DetectionResult:
        """Determine whether the frame contains an adversarial void region."""
        lidar = frame.lidar       # (N, 4)
        pts_xyz = lidar[:, :3]    # (N, 3)

        # 1. Ground slice
        ground_pts = self._extract_ground_points(lidar)

        # 2. Occupancy grid → empty cells
        empty_centers = self._find_empty_cells(ground_pts)

        if len(empty_centers) == 0:
            return DetectionResult(
                is_attack_detected=False,
                confidence=0.0,
                metadata={
                    "reason": "no_empty_cells",
                    "empty_cell_positions": [],
                    "empty_cell_cluster_labels": [],
                },
            )

        # 3. DBSCAN clustering of empty cells into shadow clusters
        pts_2d = empty_centers[:, :2]
        labels = DBSCAN(
            eps=self.dbscan_eps, min_samples=self.dbscan_min_samples
        ).fit_predict(pts_2d)

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

        if n_clusters == 0:
            return DetectionResult(
                is_attack_detected=False,
                confidence=0.0,
                metadata={
                    "n_clusters": 0,
                    "reason": "no_clusters",
                    "empty_cell_positions": empty_centers[:, :2].tolist(),
                    "empty_cell_cluster_labels": labels.tolist(),
                },
            )

        # 4. For each shadow cluster, backtrace one frustum per cell centre
        #    Union point indices across all cells; skip cells whose frustum
        #    has fewer than min_frustum_points hits
        clusters = self._get_clusters(empty_centers, labels, n_clusters)
        all_frustum_indices: set[int] = set()
        cluster_details: list[dict] = []

        for cluster in clusters:
            cluster_indices: set[int] = set()
            valid_cells = 0
            for cell_center in cluster:
                result = self._backtrace_frustum(cell_center, pts_xyz)
                if result["count"] < self.min_frustum_points:
                    continue
                valid_cells += 1
                cluster_indices.update(result["indices"])
            all_frustum_indices.update(cluster_indices)
            cluster_details.append({
                "centroid": np.mean(cluster, axis=0).tolist(),
                "cluster_size": len(cluster),
                "valid_cells": valid_cells,
                "frustum_pt_count": len(cluster_indices),
            })

        # Materialise the deduplicated frustum points.
        if all_frustum_indices:
            frustum_pts = pts_xyz[sorted(all_frustum_indices)]
        else:
            frustum_pts = np.empty((0, 3))

        # 5. Exclude points inside any detector-predicted bounding box
        unidentified_pts = self._exclude_predicted_boxes(frustum_pts, frame.predictions)

        # 6. Second DBSCAN on the remaining unidentified points
        n_obstacle_clusters = 0
        obstacle_centroids: list[list[float]] = []
        obstacle_sizes: list[int] = []
        obs_labels_arr: np.ndarray | None = None

        obstacle_aabbs: list[list] = []
        if len(unidentified_pts) >= self.obstacle_dbscan_min_samples:
            obs_labels_arr = DBSCAN(
                eps=self.obstacle_dbscan_eps,
                min_samples=self.obstacle_dbscan_min_samples,
            ).fit_predict(unidentified_pts)
            n_obstacle_clusters = (
                len(set(obs_labels_arr)) - (1 if -1 in obs_labels_arr else 0)
            )
            for i in range(n_obstacle_clusters):
                pts_i = unidentified_pts[obs_labels_arr == i]
                obstacle_centroids.append(pts_i.mean(axis=0).tolist())
                obstacle_sizes.append(len(pts_i))
                obstacle_aabbs.append([
                    pts_i.min(axis=0).tolist(),
                    pts_i.max(axis=0).tolist(),
                ])

        attack_detected = n_obstacle_clusters > 0
        confidence = 1.0 if attack_detected else 0.0

        # GT matching for evaluation metadata only — not used in the decision.
        gt_matches = (
            self._match_clusters_to_gt(
                unidentified_pts, obs_labels_arr, n_obstacle_clusters, frame.labels
            )
            if n_obstacle_clusters > 0 else []
        )

        return DetectionResult(
            is_attack_detected=attack_detected,
            confidence=confidence,
            metadata={
                "n_clusters": n_clusters,
                "n_empty_cells": len(empty_centers),
                "n_obstacle_clusters": n_obstacle_clusters,
                "obstacle_centroids": obstacle_centroids,
                "obstacle_cluster_sizes": obstacle_sizes,
                "obstacle_cluster_aabbs": obstacle_aabbs,
                "obstacle_matches_gt": gt_matches,
                "cluster_details": cluster_details,
                "empty_cell_positions": empty_centers[:, :2].tolist(),
                "empty_cell_cluster_labels": labels.tolist(),
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
            centers = [
                [x, y, self.ground_height]
                for x in x_centers
                for y in y_centers
            ]
            return np.array(centers)

        pts_x = ground_pts[:, 0]
        pts_y = ground_pts[:, 1]

        half = stride / 2.0
        xi = np.digitize(pts_x, x_centers + half)
        yi = np.digitize(pts_y, y_centers + half)

        occupied: set[tuple[int, int]] = set(zip(xi.tolist(), yi.tolist()))

        empty: list[list[float]] = []
        for ix, x in enumerate(x_centers):
            for iy, y in enumerate(y_centers):
                if (ix, iy) not in occupied:
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
        """Construct a frustum from the sensor origin through the given cell
        centre and return the indices of lidar points inside it.

        Adapted directly from backtrace_shadow_from_pt() in the notebook.
        The frustum is a small wedge-shaped volume defined by 8 vertices;
        we pre-filter with an AABB and then apply the exact isInPoly test.

        Returns {"count": int, "indices": list[int]} — indices into pts_xyz.
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
        candidate_indices = np.where(pre_mask)[0]
        candidates = pts_xyz[candidate_indices]

        in_indices: list[int] = []
        for local_i, pt in enumerate(candidates):
            if isInPoly(pt.tolist(), poly):
                in_indices.append(int(candidate_indices[local_i]))

        return {"count": len(in_indices), "indices": in_indices}

    # ------------------------------------------------------------------
    # Prediction-based exclusion filter (paper step 6)
    # ------------------------------------------------------------------

    @staticmethod
    def _exclude_predicted_boxes(
        pts: np.ndarray,
        predictions: list[Prediction],
    ) -> np.ndarray:
        """Return only those points that do not fall inside any predicted
        bounding-box AABB.  Points inside a detected object's box are already
        explained by the detector and are not indicative of a hidden obstacle.
        """
        if len(pts) == 0 or not predictions:
            return pts
        mask = np.ones(len(pts), dtype=bool)
        for pred in predictions:
            corners = pred.corners_velo    # (8, 3)
            mins = corners.min(axis=0)
            maxs = corners.max(axis=0)
            in_box = np.all((pts >= mins) & (pts <= maxs), axis=1)
            mask &= ~in_box
        return pts[mask]

    # ------------------------------------------------------------------
    # GT matching — evaluation metadata only, not used in decision
    # ------------------------------------------------------------------

    @staticmethod
    def _match_clusters_to_gt(
        unidentified_pts: np.ndarray,
        obs_labels_arr: np.ndarray,
        n_obstacle_clusters: int,
        gt_labels: list[ObjectLabel],
    ) -> list[list[int]]:
        """For each obstacle cluster, return the GT label indices whose AABB
        contains at least one cluster point.  Used only for TPR/FPR reporting
        in metadata — the attack decision does not depend on this.
        """
        matches: list[list[int]] = []
        for i in range(n_obstacle_clusters):
            cluster_pts = unidentified_pts[obs_labels_arr == i]
            matched_gt: list[int] = []
            for j, label in enumerate(gt_labels):
                corners = label.corners_velo    # (8, 3)
                mins = corners.min(axis=0)
                maxs = corners.max(axis=0)
                if np.all((cluster_pts >= mins) & (cluster_pts <= maxs), axis=1).any():
                    matched_gt.append(j)
            matches.append(matched_gt)
        return matches
