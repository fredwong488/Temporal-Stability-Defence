"""
defenses/fsd.py
---------------
Fake Shadow Detection (FSD) — adversarial Physical Removal Attack detector.

Reference
---------
Cao, Y., Xiao, C., Anber, F., & Morley Mao, Z. (2022).
You Can't See Me: Physical Removal Attacks on LiDAR-based Autonomous Vehicles
Driving Frameworks.  arXiv:2210.09482v2, §7.2.

Algorithm
---------
1. Identify candidate shadow regions in the ROI by finding empty ground cells
   (Hau et al. 2022 void-region approach — logic duplicated from void_region.py)
   and clustering them with DBSCAN.
2. Obtain the set of obstacle clusters from Autoware's Euclidean Cluster
   Extraction.  This step is a TODO placeholder — see _autoware_euclidean_clusters.
3. Build a 3-D voxel grid over the ROI.
   For each shadow cluster s_j, voxelise the union of pyramidal frustums from
   the sensor origin to each empty-cell footprint (shadow voxels).
   For each detected obstacle cluster o_i, voxelise the 3-D shadow it is
   expected to cast: the set of voxels whose sensor-origin ray passes through
   the cluster's 3-D AABB (explained voxels).  The slab-method ray–AABB test
   captures the tapered-wedge geometry (the shadow thins with distance as the
   occluding ray descends toward the ground).
4. For each shadow cluster, compute:
       residual_volume = |shadow_voxels \ explained_voxels| × voxel_size³
   If any residual exceeds ``volume_threshold``, flag a Physical Removal Attack.
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import DBSCAN

from ..base import BaseDefense
from ..types import DetectionResult, Frame, FrameHistory


class FSDDefense(BaseDefense):
    """Detect Physical Removal Attacks by finding shadow regions that are not
    explained by any detected obstacle's expected LiDAR shadow.

    Parameters
    ----------
    roi_min
        (x_min, y_min) of the region of interest in velodyne metres.
        Defaults match VoidRegionDefense (Assumption 2).
    roi_max
        (x_max, y_max) of the region of interest.
    ground_height
        Sensor-relative height of the ground plane (−1.73 m for KITTI).
    ground_delta
        Half-thickness of the ground-plane slice to extract.
    grid_stride
        Occupancy-grid cell size (m). Matches VoidRegionDefense default.
    dbscan_eps
        DBSCAN neighbourhood radius for shadow (empty-cell) clustering.
    dbscan_min_samples
        DBSCAN minimum cluster size for shadow clustering.
    volume_threshold
        Residual shadow volume (m³) above which a PRA is declared.
        Paper empirical value: ~15 m³ (vehicle targets, §7.2).
    voxel_size
        Edge length of each cubic voxel in the 3-D grid (m).
    z_ceiling
        Upper Z boundary of the voxel grid in sensor coordinates.
        Default 0.0 = sensor plane; all typical KITTI obstacles fit below it.
    """

    def __init__(
        self,
        roi_min: tuple[float, float] = (0.0, -5.0),
        roi_max: tuple[float, float] = (30.0, 5.0),
        ground_height: float = -1.73,
        ground_delta: float = 0.2,
        grid_stride: float = 0.6,
        dbscan_eps: float = 0.8,
        dbscan_min_samples: int = 5,
        volume_threshold: float = 15.0,
        voxel_size: float = 0.3,
        z_ceiling: float = 0.0,
    ) -> None:
        self.roi_min = roi_min
        self.roi_max = roi_max
        self.ground_height = ground_height
        self.ground_delta = ground_delta
        self.grid_stride = grid_stride
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_samples = dbscan_min_samples
        self.volume_threshold = volume_threshold
        self.voxel_size = voxel_size
        self.z_ceiling = z_ceiling

    @property
    def temporal_window(self) -> int:
        return 1  # stateless — operates on the current frame only

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, frame: Frame, history: FrameHistory) -> DetectionResult:
        """Determine whether the frame contains a Physical Removal Attack."""
        lidar = frame.lidar      # (N, 4)
        pts_xyz = lidar[:, :3]   # (N, 3)

        # 1. Shadow region identification (Hau et al. — see void_region.py)
        ground_pts = self._extract_ground_points(lidar)
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

        labels = DBSCAN(
            eps=self.dbscan_eps, min_samples=self.dbscan_min_samples
        ).fit_predict(empty_centers[:, :2])
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

        if n_clusters == 0:
            return DetectionResult(
                is_attack_detected=False,
                confidence=0.0,
                metadata={
                    "n_shadow_clusters": 0,
                    "reason": "no_shadow_clusters",
                    "empty_cell_positions": empty_centers[:, :2].tolist(),
                    "empty_cell_cluster_labels": labels.tolist(),
                },
            )

        shadow_clusters = self._get_clusters(empty_centers, labels, n_clusters)

        # 2. Autoware Euclidean clustering (TODO placeholder)
        detected_clusters = self._autoware_euclidean_clusters(pts_xyz)

        # 3. Voxel grid + explained mask
        voxel_centers, _grid_shape = self._build_voxel_grid()

        # Accumulate explained voxels across all detected clusters (Assumption 7)
        explained_mask = np.zeros(len(voxel_centers), dtype=bool)
        for cluster_pts in detected_clusters:
            explained_mask |= self._voxelize_cluster_frustum(cluster_pts, voxel_centers)

        # 4. Per-shadow-cluster residual volume and scan-level decision
        shadow_cluster_volumes: list[float] = []
        residual_volumes: list[float] = []
        exceeded_threshold: list[bool] = []
        attack_detected = False
        voxel_vol = self.voxel_size ** 3

        for shadow_cluster in shadow_clusters:
            shadow_mask = self._voxelize_shadow_pyramid(shadow_cluster, voxel_centers)
            raw_volume = float(shadow_mask.sum()) * voxel_vol
            residual_mask = shadow_mask & ~explained_mask
            residual_volume = float(residual_mask.sum()) * voxel_vol
            exceeded = residual_volume > self.volume_threshold

            shadow_cluster_volumes.append(raw_volume)
            residual_volumes.append(residual_volume)
            exceeded_threshold.append(exceeded)
            if exceeded:
                attack_detected = True

        return DetectionResult(
            is_attack_detected=attack_detected,
            confidence=1.0 if attack_detected else 0.0,
            metadata={
                "n_shadow_clusters": n_clusters,
                "n_detected_clusters": len(detected_clusters),
                "shadow_cluster_volumes": shadow_cluster_volumes,
                "residual_volumes": residual_volumes,
                "exceeded_threshold": exceeded_threshold,
                "volume_threshold": self.volume_threshold,
                # mirror VoidRegionDefense keys so the visualiser BEV panel works
                "empty_cell_positions": empty_centers[:, :2].tolist(),
                "empty_cell_cluster_labels": labels.tolist(),
            },
        )

    # ------------------------------------------------------------------
    # Autoware clustering placeholder
    # ------------------------------------------------------------------

    def _autoware_euclidean_clusters(
        self, pts_xyz: np.ndarray
    ) -> list[np.ndarray]:
        """TODO(autoware): replace with Autoware Euclidean Cluster Extraction
        (pcl::EuclideanClusterExtraction) once a Python binding is wired in.
        Must return a list of (M_i, 3) xyz arrays — one per detected cluster —
        in the velodyne frame and restricted to the ROI.

        See docs/fsd_replication_spec.md §3.1 and §6 for the paper's clustering
        source specification.
        """
        raise NotImplementedError(
            "Autoware Euclidean clustering is not yet integrated. "
            "FSD requires the paper-specified clustering source — see "
            "docs/fsd_replication_spec.md §3.1 and §6."
        )

    # ------------------------------------------------------------------
    # Voxel grid construction
    # ------------------------------------------------------------------

    def _build_voxel_grid(self) -> tuple[np.ndarray, tuple[int, int, int]]:
        """Return (voxel_centers, grid_shape).

        voxel_centers : (N_vox, 3) array of voxel-centre coordinates.
        grid_shape    : (nx, ny, nz) for reshaping flat boolean masks.
        """
        half = self.voxel_size / 2.0
        xs = np.arange(self.roi_min[0] + half, self.roi_max[0], self.voxel_size)
        ys = np.arange(self.roi_min[1] + half, self.roi_max[1], self.voxel_size)
        zs = np.arange(self.ground_height + half, self.z_ceiling, self.voxel_size)
        xg, yg, zg = np.meshgrid(xs, ys, zs, indexing="ij")
        voxel_centers = np.stack([xg.ravel(), yg.ravel(), zg.ravel()], axis=1)
        return voxel_centers, (len(xs), len(ys), len(zs))

    # ------------------------------------------------------------------
    # Shadow-pyramid voxelisation (adapted from void_region._backtrace_frustum)
    # ------------------------------------------------------------------

    def _voxelize_shadow_pyramid(
        self, cells: np.ndarray, voxel_centers: np.ndarray
    ) -> np.ndarray:
        """Return (N_vox,) bool mask: True for voxels inside the union of
        pyramidal frustums from the sensor origin to each empty-cell footprint.

        Geometry mirrors VoidRegionDefense._backtrace_frustum: apex at sensor
        origin, base = grid_stride × grid_stride at the cell distance, with
        cross-section tapering linearly from zero at the apex.
        """
        mask = np.zeros(len(voxel_centers), dtype=bool)
        for cell_center in cells:
            cell_vec = cell_center[:3].astype(float)
            cell_dist = float(np.linalg.norm(cell_vec))
            if cell_dist < 1e-6:
                continue
            cell_dir = cell_vec / cell_dist

            up = np.array([0.0, 0.0, 1.0])
            if abs(float(cell_dir @ up)) > 0.999:
                up = np.array([1.0, 0.0, 0.0])
            u = np.cross(up, cell_dir)
            u /= np.linalg.norm(u)
            v_ax = np.cross(cell_dir, u)

            t      = voxel_centers @ cell_dir
            du     = voxel_centers @ u
            dv     = voxel_centers @ v_ax
            half_t = np.where(t > 0.0, (self.grid_stride / 2.0) * t / cell_dist, 0.0)

            in_pyr = (
                (t > 0.0)
                & (t < cell_dist)
                & (np.abs(du) < half_t)
                & (np.abs(dv) < half_t)
            )
            mask |= in_pyr
        return mask

    # ------------------------------------------------------------------
    # Object-frustum voxelisation (ray–AABB slab test)
    # ------------------------------------------------------------------

    def _voxelize_cluster_frustum(
        self, cluster_pts: np.ndarray, voxel_centers: np.ndarray
    ) -> np.ndarray:
        """Return (N_vox,) bool mask: True for voxels in the expected shadow
        of a detected obstacle cluster.

        A voxel is in the shadow iff the ray from the sensor origin to the
        voxel centre passes through the cluster's 3-D AABB at some parameter
        t ∈ (0, 1) — i.e., the cluster lies between the sensor and the voxel.

        The slab-method ray–AABB intersection naturally captures the tapered-
        wedge shadow geometry: distant voxels have a narrower eligible Z range
        because the occluding sensor ray descends toward the ground.
        """
        pts = cluster_pts[:, :3]
        aabb_min = pts.min(axis=0)  # (3,)
        aabb_max = pts.max(axis=0)  # (3,)

        N = len(voxel_centers)
        t_near = np.full(N, -np.inf)
        t_far  = np.full(N,  np.inf)

        for i in range(3):
            vi = voxel_centers[:, i]
            nonparallel = np.abs(vi) > 1e-9

            safe_vi = np.where(nonparallel, vi, 1.0)
            t0 = aabb_min[i] / safe_vi
            t1 = aabb_max[i] / safe_vi
            tni = np.where(nonparallel, np.minimum(t0, t1), -np.inf)

            # Parallel ray outside the slab → this axis rules out intersection.
            outside_i = bool(aabb_min[i] > 0.0 or aabb_max[i] < 0.0)
            tfi_parallel = -np.inf if outside_i else np.inf
            tfi = np.where(nonparallel, np.maximum(t0, t1), tfi_parallel)

            t_near = np.maximum(t_near, tni)
            t_far  = np.minimum(t_far,  tfi)

        # Valid intersection at t ∈ (0, 1): cluster between sensor and voxel.
        return (t_near <= t_far) & (t_far > 0.0) & (t_near < 1.0)

    # ------------------------------------------------------------------
    # Ground extraction (duplicated from void_region.py — Assumption 4)
    # ------------------------------------------------------------------

    def _extract_ground_points(self, lidar: np.ndarray) -> np.ndarray:
        z = lidar[:, 2]
        mask = (
            (z >= self.ground_height - self.ground_delta)
            & (z <= self.ground_height + self.ground_delta)
        )
        return lidar[mask, :3]

    # ------------------------------------------------------------------
    # Occupancy grid (duplicated from void_region.py — Assumption 2)
    # ------------------------------------------------------------------

    def _find_empty_cells(self, ground_pts: np.ndarray) -> np.ndarray:
        x_min, y_min = self.roi_min
        x_max, y_max = self.roi_max
        stride = self.grid_stride

        x_centers = np.arange(x_min, x_max, stride)
        y_centers = np.arange(y_min, y_max, stride)

        roi_mask = (
            (ground_pts[:, 0] >= x_min) & (ground_pts[:, 0] < x_max)
            & (ground_pts[:, 1] >= y_min) & (ground_pts[:, 1] < y_max)
        )
        ground_pts = ground_pts[roi_mask]

        if len(ground_pts) == 0:
            centers = [
                [x, y, self.ground_height]
                for x in x_centers
                for y in y_centers
            ]
            return np.array(centers)

        half = stride / 2.0
        xi = np.digitize(ground_pts[:, 0], x_centers + half)
        yi = np.digitize(ground_pts[:, 1], y_centers + half)
        occupied: set[tuple[int, int]] = set(zip(xi.tolist(), yi.tolist()))

        empty: list[list[float]] = []
        for ix, x in enumerate(x_centers):
            for iy, y in enumerate(y_centers):
                if (ix, iy) not in occupied:
                    if x_min <= x <= x_max and y_min <= y <= y_max:
                        empty.append([x, y, self.ground_height])

        return np.array(empty) if empty else np.empty((0, 3))

    # ------------------------------------------------------------------
    # Cluster extraction (duplicated from void_region.py)
    # ------------------------------------------------------------------

    @staticmethod
    def _get_clusters(
        empty_centers: np.ndarray,
        labels: np.ndarray,
        n_clusters: int,
    ) -> list[np.ndarray]:
        return [empty_centers[labels == i] for i in range(n_clusters)]
