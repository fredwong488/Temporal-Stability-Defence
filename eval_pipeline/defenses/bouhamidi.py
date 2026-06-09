"""
defenses/bouhamidi.py
---------------------
Physical-coherence attack detector (Bouhamidi et al., IWSSIP 2025).

Reference
---------
Yacine Bouhamidi, Kai Wang, José-Ernesto Gomez-Balderas.
"An Efficient Method to Detect both Insertion and Removal Attacks in 3D
LiDAR Point Clouds for Autonomous Vehicle Security."
IWSSIP 2025 — 32nd International Conference on Systems, Signals and Image
Processing, Jun 2025, Skopje, Macedonia, pp. 1–5.
https://hal.science/hal-05294984v1

Algorithm (Section III, Table I)
---------------------------------
1.  Level the point cloud to the gravity-aligned ego frame via
    ``frame.nuscenes_sensor_to_ego`` so that the ground lies at z ≈ 0.
2.  Convert points to spherical coordinates (r, polar θ, azimuth φ) and
    restrict to the frontal ROI:
        θ ∈ [θ_min, θ_max]   (polar angle from +z; larger ↔ closer to ground)
        φ ∈ [φ_min, φ_max]   (azimuth; 0 = straight ahead in ego frame)
3.  Segment the ROI into ground / non-ground using Patchwork++.
4.  Bin ground and non-ground points into an M × N (θ, φ) grid.
5.  Coherence check:
        insertion_cells = ground_occ  AND non-ground_occ
        removal_cells   = NOT ground_occ AND NOT non-ground_occ
6.  Post-processing:
    •  Ignore boundary cells (outermost row/column) — paper §III-D.
    •  Insertion filter: within insertion_cells, keep only non-ground points
       that have ≥ ``filter_neighbor_threshold`` neighbours within
       ``filter_radius`` metres (using scipy KDTree).  Re-count surviving
       cells.
    •  Scene decision: flag attack if surviving group count > scene_threshold.

Notes
-----
*  NuScenes only.  The spherical ROI assumes a gravity-aligned frame;
   the raw sensor on NuScenes is tilted, so we must level via
   ``nuscenes_sensor_to_ego`` before the spherical transform.  If that
   transform is missing, a ``ValueError`` is raised.
*  No ego-box removal.  Carving out points near the origin would create
   artificially empty (θ, φ) cells, causing systematic false-positive
   removal detections on every frame.  The paper's spherical ROI bounds
   (θ_max = 1.9 cuts the near-field blind spot) serve this purpose instead.
"""

from __future__ import annotations

import math
import time
from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial import cKDTree

from ..base import BaseDefense
from ..types import DetectionResult, Frame, FrameHistory
from ._multiframe_common import patchwork_ground_segment

if TYPE_CHECKING:
    pass


class BouhamidiDefense(BaseDefense):
    """Physical-coherence detector (Bouhamidi et al., IWSSIP 2025).

    Parameters
    ----------
    theta_min : float
        Minimum polar angle (from +z) in radians.  Points with θ < θ_min
        are high altitude or in the sensor blind-spot.  Default: π/2.
    theta_max : float
        Maximum polar angle in radians.  Default: 1.9 rad.
    theta_step : float
        Grid step in the polar direction.  Default: 0.017 rad (≈ 1°).
    phi_min : float
        Minimum azimuth in radians (negative = left of forward).
        Default: −0.5 rad.
    phi_max : float
        Maximum azimuth in radians.  Default: 0.5 rad.
    phi_step : float
        Grid step in the azimuth direction.  Default: 0.0087 rad (≈ 0.5°).
    scene_threshold : int
        Minimum number of inconsistent (θ, φ) cell groups required to flag
        an attack at the scene level.  Default: 5.
    filter_neighbor_threshold : int
        Minimum number of neighbours required within ``filter_radius`` for a
        detected insertion point to be kept (removal of isolated detections).
        Default: 35.
    filter_radius : float
        Neighbourhood radius in metres for the insertion-point filter.
        Default: 0.25 m.
    ignore_boundary : bool
        If True, zero out the outermost row/column of each inconsistency
        grid before counting groups.  Default: True.
    patchwork_sensor_height : float
        ``sensor_height`` parameter passed to Patchwork++.  In the NuScenes
        levelled ego frame the LIDAR_TOP is mounted ≈ 1.84 m above ground.
        Default: 1.84.
    patchwork_num_iter : int or None
        Number of Patchwork++ iterations.  If ``None`` (default), the
        Patchwork++ library default is used.
    patchwork_uprightness_thr : float or None
        Uprightness threshold for Patchwork++.  If ``None`` (default), the
        Patchwork++ library default is used.
    """

    # --------------------------------------------------------------------- #
    #  Construction                                                           #
    # --------------------------------------------------------------------- #

    def __init__(
        self,
        *,
        theta_min: float = math.pi / 2,
        theta_max: float = 1.9,
        theta_step: float = 0.017,
        phi_min: float = -0.5,
        phi_max: float = 0.5,
        phi_step: float = 0.0087,
        scene_threshold: int = 5,
        filter_neighbor_threshold: int = 35,
        filter_radius: float = 0.25,
        ignore_boundary: bool = True,
        patchwork_sensor_height: float = 1.84,
        patchwork_num_iter: int | None = None,
        patchwork_uprightness_thr: float | None = None,
    ) -> None:
        self.theta_min = theta_min
        self.theta_max = theta_max
        self.theta_step = theta_step
        self.phi_min = phi_min
        self.phi_max = phi_max
        self.phi_step = phi_step
        self.scene_threshold = scene_threshold
        self.filter_neighbor_threshold = filter_neighbor_threshold
        self.filter_radius = filter_radius
        self.ignore_boundary = ignore_boundary
        self.patchwork_sensor_height = patchwork_sensor_height
        self.patchwork_num_iter = patchwork_num_iter
        self.patchwork_uprightness_thr = patchwork_uprightness_thr

    # --------------------------------------------------------------------- #
    #  BaseDefense interface                                                  #
    # --------------------------------------------------------------------- #

    @property
    def temporal_window(self) -> int:
        return 1  # stateless: history is ignored

    # --------------------------------------------------------------------- #
    #  Patchwork++                                                            #
    # --------------------------------------------------------------------- #

    def _ground_segment(self, xyzw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Segment ``xyzw`` (N×4, XYZI) into ground / non-ground indices.

        Returns
        -------
        ground_idx : np.ndarray of int
        nonground_idx : np.ndarray of int
        """
        return patchwork_ground_segment(
            xyzw,
            self.patchwork_sensor_height,
            self.patchwork_num_iter,
            self.patchwork_uprightness_thr,
        )

    # --------------------------------------------------------------------- #
    #  Main detect                                                            #
    # --------------------------------------------------------------------- #

    def detect(self, frame: Frame, history: FrameHistory) -> DetectionResult:
        _t0_total = time.perf_counter()

        # ------------------------------------------------------------------ #
        #  1.  Level to gravity-aligned ego frame                             #
        # ------------------------------------------------------------------ #
        if frame.nuscenes_sensor_to_ego is None:
            raise ValueError(
                "BouhamidiDefense requires NuScenes data (nuscenes_sensor_to_ego is None). "
                "This defense is not supported for KITTI frames."
            )

        xyz_sensor = frame.lidar[:, :3].astype(np.float64)
        intensity = frame.lidar[:, 3:4].astype(np.float64)
        s2e = frame.nuscenes_sensor_to_ego.astype(np.float64)
        # Full ego transform for spherical coordinates and ROI (x-forward, y-left, z-up).
        xyz_ego = (s2e[:3, :3] @ xyz_sensor.T + s2e[:3, 3:4]).T  # (N, 3)
        # Rotation-only levelling for Patchwork++ (ground at z ≈ -sensor_height).
        xyz_leveled = (s2e[:3, :3] @ xyz_sensor.T).T  # (N, 3)

        # ------------------------------------------------------------------ #
        #  2.  Spherical coordinates + ROI mask                               #
        # ------------------------------------------------------------------ #
        _t0 = time.perf_counter()
        r = np.linalg.norm(xyz_ego, axis=1)
        # Guard against origin-point division-by-zero
        r_safe = np.where(r > 0, r, 1e-9)

        theta = np.arccos(np.clip(xyz_ego[:, 2] / r_safe, -1.0, 1.0))  # polar from +z
        phi = np.arctan2(xyz_ego[:, 1], xyz_ego[:, 0])                  # azimuth

        roi_mask = (
            (theta >= self.theta_min) & (theta <= self.theta_max)
            & (phi >= self.phi_min) & (phi <= self.phi_max)
            & (r > 0)
        )

        # Return benign result if ROI is empty
        if not roi_mask.any():
            return DetectionResult(
                is_attack_detected=False,
                confidence=0.0,
                metadata={
                    "insertion_detected": False,
                    "removal_detected": False,
                    "insertion_groups": 0,
                    "removal_groups": 0,
                    "insertion_points_xyz": [],
                    "removal_cell_indices": [],
                },
            )

        xyz_roi = xyz_ego[roi_mask]
        intensity_roi = intensity[roi_mask]
        theta_roi = theta[roi_mask]
        phi_roi = phi[roi_mask]

        # Pass rotation-levelled points to Patchwork++ (ground at z ≈ -sensor_height).
        xyzw_leveled_roi = np.concatenate(
            [xyz_leveled[roi_mask], intensity_roi], axis=1
        ).astype(np.float32)
        _elapsed_spherical_roi_s = time.perf_counter() - _t0

        # ------------------------------------------------------------------ #
        #  3.  Ground / non-ground segmentation (Patchwork++)                 #
        # ------------------------------------------------------------------ #
        _t0 = time.perf_counter()
        ground_idx, nonground_idx = self._ground_segment(xyzw_leveled_roi)
        _elapsed_ground_segment_s = time.perf_counter() - _t0

        # ------------------------------------------------------------------ #
        #  4.  Grid binning                                                   #
        # ------------------------------------------------------------------ #
        _t0 = time.perf_counter()
        # Build edges: paper uses step = +0.017 for θ and +0.0087 for φ
        theta_edges = np.arange(self.theta_min, self.theta_max + self.theta_step, self.theta_step)
        phi_edges = np.arange(self.phi_min, self.phi_max + self.phi_step, self.phi_step)
        M = len(theta_edges) - 1  # number of θ bins
        N = len(phi_edges) - 1    # number of φ bins

        if M <= 0 or N <= 0:
            return DetectionResult(
                is_attack_detected=False,
                confidence=0.0,
                metadata={
                    "insertion_detected": False,
                    "removal_detected": False,
                    "insertion_groups": 0,
                    "removal_groups": 0,
                    "insertion_points_xyz": [],
                    "removal_cell_indices": [],
                },
            )

        # np.digitize returns 1-indexed bins; subtract 1 for 0-indexed, clip to valid range
        theta_bin = np.clip(np.digitize(theta_roi, theta_edges) - 1, 0, M - 1)
        phi_bin = np.clip(np.digitize(phi_roi, phi_edges) - 1, 0, N - 1)

        # Boolean occupancy grids
        ground_occ = np.zeros((M, N), dtype=bool)
        nonground_occ = np.zeros((M, N), dtype=bool)

        if len(ground_idx) > 0:
            gt_bins = theta_bin[ground_idx]
            gp_bins = phi_bin[ground_idx]
            ground_occ[gt_bins, gp_bins] = True

        if len(nonground_idx) > 0:
            nt_bins = theta_bin[nonground_idx]
            np_bins = phi_bin[nonground_idx]
            nonground_occ[nt_bins, np_bins] = True

        # ------------------------------------------------------------------ #
        #  5.  Coherence check                                                #
        # ------------------------------------------------------------------ #
        insertion_cells = ground_occ & nonground_occ         # both filled
        removal_cells = ~ground_occ & ~nonground_occ         # neither filled
        _elapsed_grid_coherence_s = time.perf_counter() - _t0

        # ------------------------------------------------------------------ #
        #  6.  Post-processing                                                #
        # ------------------------------------------------------------------ #
        _t0 = time.perf_counter()
        # 6a. Ignore boundary cells
        if self.ignore_boundary and M > 2 and N > 2:
            insertion_cells[[0, -1], :] = False
            insertion_cells[:, [0, -1]] = False
            removal_cells[[0, -1], :] = False
            removal_cells[:, [0, -1]] = False

        # 6b. Insertion: filter isolated detections via neighbour count
        #     Gather non-ground points in flagged cells, then radius filter.
        insertion_nonground_mask = np.zeros(len(nonground_idx), dtype=bool)
        if nonground_idx.size > 0:
            ng_t = theta_bin[nonground_idx]
            ng_p = phi_bin[nonground_idx]
            in_insertion_cell = insertion_cells[ng_t, ng_p]
            insertion_nonground_mask = in_insertion_cell

        insertion_points_xyz = xyz_roi[nonground_idx[insertion_nonground_mask]]

        # Radius-neighbour filter (paper: keep points with ≥35 neighbours in 25 cm)
        surviving_insertion_pts: np.ndarray = np.empty((0, 3), dtype=np.float32)
        surviving_insertion_cells: set[tuple[int, int]] = set()

        if len(insertion_points_xyz) > 0:
            tree = cKDTree(insertion_points_xyz)
            counts = np.array(
                [len(tree.query_ball_point(p, self.filter_radius)) - 1
                 for p in insertion_points_xyz],
                dtype=int,
            )
            keep = counts >= self.filter_neighbor_threshold
            surviving_insertion_pts = insertion_points_xyz[keep].astype(np.float32)

            # Determine which (θ, φ) cells still have surviving points
            if keep.any():
                kept_ng_idx = nonground_idx[insertion_nonground_mask][keep]
                surviving_insertion_cells = set(
                    zip(theta_bin[kept_ng_idx].tolist(), phi_bin[kept_ng_idx].tolist())
                )

        insertion_groups = len(surviving_insertion_cells)

        # 6c. Removal: count flagged cells (after boundary ignore)
        removal_cell_indices = list(zip(*np.where(removal_cells))) if removal_cells.any() else []
        removal_groups = len(removal_cell_indices)
        _elapsed_postprocess_s = time.perf_counter() - _t0

        # ------------------------------------------------------------------ #
        #  7.  Scene-level decision                                           #
        # ------------------------------------------------------------------ #
        insertion_detected = insertion_groups > self.scene_threshold
        removal_detected = removal_groups > self.scene_threshold
        is_attack = insertion_detected or removal_detected

        # Confidence: max of the two normalised group-count ratios, capped at 1.0
        ins_conf = min(insertion_groups / max(self.scene_threshold, 1), 1.0)
        rem_conf = min(removal_groups / max(self.scene_threshold, 1), 1.0)
        confidence = max(ins_conf, rem_conf) if is_attack else max(ins_conf, rem_conf)

        return DetectionResult(
            is_attack_detected=is_attack,
            confidence=float(confidence),
            metadata={
                "insertion_detected": bool(insertion_detected),
                "removal_detected": bool(removal_detected),
                "insertion_groups": int(insertion_groups),
                "removal_groups": int(removal_groups),
                "insertion_points_xyz": surviving_insertion_pts.tolist(),
                "removal_cell_indices": removal_cell_indices,
                "elapsed_s": {
                    "spherical_roi": _elapsed_spherical_roi_s,
                    "ground_segment": _elapsed_ground_segment_s,
                    "grid_coherence": _elapsed_grid_coherence_s,
                    "postprocess": _elapsed_postprocess_s,
                    "total": time.perf_counter() - _t0_total,
                },
            },
        )
