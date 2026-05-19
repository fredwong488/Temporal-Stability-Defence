"""
defenses/wasserstein_anisotropy.py
-----------------------------------
Wasserstein-Anisotropy Defense — detects LiDAR spoofing by the radial
dominance of frame-to-frame distributional shift in ego-compensated clusters.

The spoofing noise model (Sato et al. 2024) injects a per-frame rigid
translation δ_inter ~ N(0, ~35 cm) along the sensor's radial direction.
After ego-motion compensation and bulk-motion subtraction, a spoofed cluster's
frame-to-frame change is dominated by this radial shift.  A benign cluster's
residual change is roughly isotropic LiDAR ranging noise.

The test statistic is:

    ρ = median_t[ W_r²(t-1, t) / (W_⊥²(t-1, t) + ε) ]

where W_r² and W_⊥² are the squared 1D Wasserstein-2 distances between
consecutive frames' point projections onto the radial axis ê_r and the
summed perpendicular contribution (ê_p1 + ê_p2), respectively.  A large ρ
(radial change dominates lateral/vertical change) indicates spoofing.

Preprocessing, history compensation, and cluster-association are shared with
RadialJitterDefense via _multiframe_common.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Literal

import numpy as np
from sklearn.cluster import DBSCAN

from ..base import BaseDefense
from ..types import DetectionResult, Frame, FrameHistory
from ._multiframe_common import (
    associate_cluster_chain,
    compensate_history,
    dbscan_past_sweeps,
    remove_ego_box,
)

logger = logging.getLogger(__name__)


def _wasserstein_1d_sq(u: np.ndarray, v: np.ndarray, n_quantiles: int) -> float:
    """Squared 1D Wasserstein-2 distance via quantile sampling.

    Works for arrays of unequal length.  Approximation quality improves with
    larger n_quantiles; 64 is sufficient for typical cluster sizes (20–500 pts).
    """
    qs = (np.arange(n_quantiles) + 0.5) / n_quantiles
    qu = np.quantile(u, qs)
    qv = np.quantile(v, qs)
    return float(np.mean((qu - qv) ** 2))


class WassersteinAnisotropyDefense(BaseDefense):
    """Multi-frame defense based on radial vs. perpendicular Wasserstein anisotropy.

    Parameters
    ----------
    temporal_window
        Total frames visible (current + past history).
    min_history_frames
        Minimum past frames required before the defense produces a decision.
    ground_z_max
        Z threshold (metres, velodyne frame) below which points are dropped as
        ground before clustering.
    dbscan_eps
        DBSCAN neighbourhood radius in metres.
    dbscan_min_samples
        DBSCAN minimum neighbourhood size.
    min_points_per_cluster
        Clusters smaller than this (in any frame) are skipped.
    motion_tolerance
        Maximum centroid-to-centroid distance (metres) allowed between
        consecutive frames when chaining clusters backward in time.
    min_frames_associated
        Minimum number of past frames that must yield a matched cluster.
    ratio_threshold
        ρ above which a cluster is flagged as spoofed.  Sweep target.
    ratio_eps
        Regularisation added to the denominator of ρ to avoid division by zero.
    n_quantiles
        Resolution of the quantile-based 1D-W₂ approximation.
    ego_front / ego_rear / ego_side
        Dimensions of the ego-vehicle exclusion box (metres).
    history_source
        ``"dirty"`` (default) — post-attack history.
        ``"clean"`` — pre-attack history for ablation.
    """

    def __init__(
        self,
        temporal_window: int = 6,
        min_history_frames: int = 5,
        ground_z_max: float = -1.65,
        dbscan_eps: float = 0.7,
        dbscan_min_samples: int = 10,
        min_points_per_cluster: int = 10,
        motion_tolerance: float = 1.0,
        min_frames_associated: int = 2,
        ratio_threshold: float = 5.0,
        ratio_eps: float = 1e-4,
        n_quantiles: int = 64,
        ego_front: float = 3.0,
        ego_rear: float = 2.0,
        ego_side: float = 1.4,
        history_source: Literal["clean", "dirty"] = "dirty",
    ) -> None:
        self._temporal_window = temporal_window
        self.min_history_frames = min_history_frames
        self.ground_z_max = ground_z_max
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_samples = dbscan_min_samples
        self.min_points_per_cluster = min_points_per_cluster
        self.motion_tolerance = motion_tolerance
        self.min_frames_associated = min_frames_associated
        self.ratio_threshold = ratio_threshold
        self.ratio_eps = ratio_eps
        self.n_quantiles = n_quantiles
        self.ego_front = ego_front
        self.ego_rear = ego_rear
        self.ego_side = ego_side
        self.history_source = history_source

    @property
    def temporal_window(self) -> int:
        return self._temporal_window

    # ------------------------------------------------------------------
    # BaseDefense contract
    # ------------------------------------------------------------------

    def detect(self, frame: Frame, history: FrameHistory) -> DetectionResult:
        if frame.nuscenes_ego_pose is None:
            raise ValueError(
                "WassersteinAnisotropyDefense requires frame.nuscenes_ego_pose. "
                "Use NuScenesDataset with this defense."
            )

        hist: deque[Frame] = (
            history.clean if self.history_source == "clean" else history.dirty
        )
        if len(hist) < self.min_history_frames:
            return DetectionResult(
                is_attack_detected=False,
                confidence=0.0,
                metadata={
                    "reason": "cold_start",
                    "n_history": len(hist),
                    "min_history_required": self.min_history_frames,
                },
            )

        past_xyz_list = compensate_history(
            frame, hist, self.ground_z_max,
            self.ego_front, self.ego_rear, self.ego_side,
        )

        cur_xyz = frame.lidar[:, :3]
        cur_xyz_filt = remove_ego_box(
            cur_xyz[cur_xyz[:, 2] > self.ground_z_max],
            self.ego_front, self.ego_rear, self.ego_side,
        )

        if len(cur_xyz_filt) == 0:
            return DetectionResult(
                is_attack_detected=False,
                confidence=0.0,
                metadata={"reason": "no_points_above_ground"},
            )

        labels_cur = DBSCAN(
            eps=self.dbscan_eps,
            min_samples=self.dbscan_min_samples,
            n_jobs=1,
        ).fit_predict(cur_xyz_filt)

        unique_labels = set(labels_cur) - {-1}
        if not unique_labels:
            return DetectionResult(
                is_attack_detected=False,
                confidence=0.0,
                metadata={"reason": "no_clusters"},
            )

        past_clustered = dbscan_past_sweeps(
            past_xyz_list, self.dbscan_eps, self.dbscan_min_samples,
        )

        cluster_details: list[dict] = []
        n_tested = 0
        n_flagged = 0

        for lbl in unique_labels:
            pts_cur = cur_xyz_filt[labels_cur == lbl]
            if len(pts_cur) < self.min_points_per_cluster:
                continue

            centroid_cur = pts_cur.mean(axis=0)

            valid_past = associate_cluster_chain(
                centroid_cur, past_clustered,
                self.motion_tolerance, self.min_points_per_cluster,
            )

            if len(valid_past) < self.min_frames_associated:
                continue

            n_tested += 1
            rho, w_r, w_perp = self._compute_anisotropy_ratio(
                pts_cur, valid_past, centroid_cur,
            )
            flagged = rho > self.ratio_threshold
            if flagged:
                n_flagged += 1

            cluster_details.append({
                "centroid": centroid_cur.tolist(),
                "n_points_cur": int(len(pts_cur)),
                "n_frames_associated": len(valid_past),
                "w_r": round(w_r, 4),
                "w_perp": round(w_perp, 4),
                "rho": round(rho, 4),
                "flagged": flagged,
            })

        is_attack = n_flagged > 0
        confidence = n_flagged / n_tested if n_tested > 0 else 0.0

        return DetectionResult(
            is_attack_detected=is_attack,
            confidence=float(confidence),
            metadata={
                "n_clusters_tested": n_tested,
                "n_clusters_flagged": n_flagged,
                "ratio_threshold": self.ratio_threshold,
                "cluster_details": cluster_details,
            },
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_anisotropy_ratio(
        self,
        pts_cur: np.ndarray,
        valid_past: list[np.ndarray],
        centroid_cur: np.ndarray,
    ) -> tuple[float, float, float]:
        """Compute ρ = median(W_r² / (W_⊥² + ε)) across consecutive frame pairs.

        Returns (rho, w_r_rms, w_perp_rms) where w_r_rms and w_perp_rms are
        the RMS Wasserstein distances (sqrt of mean squared) for metadata.

        Motion compensation
        -------------------
        All K+1 frames (K past + current) are re-centred by fitting a
        constant-velocity trajectory to the cluster centroids and subtracting
        the fitted centroid at each time step, then shifting so the current
        frame sits at its original location.  This removes bulk ego-frame and
        cluster velocity; the residual change is the per-frame noise we test.
        """
        all_frames: list[np.ndarray] = list(valid_past) + [pts_cur]
        K = len(all_frames) - 1  # index of current frame

        # --- 3D constant-velocity fit on centroids (closed-form OLS) ----------
        centroids = np.array([f.mean(axis=0) for f in all_frames])  # (K+1, 3)
        t = np.arange(K + 1, dtype=np.float64)
        t_c = t - t.mean()
        denom = float((t_c ** 2).sum())

        b = (t_c[:, None] * (centroids - centroids.mean(axis=0))).sum(axis=0) / denom
        c_fitted = centroids.mean(axis=0) + b[None, :] * t_c[:, None]

        # Shift each frame so its fitted centroid equals the current fitted centroid
        shift = c_fitted[K] - c_fitted  # (K+1, 3)
        aligned: list[np.ndarray] = [
            all_frames[i].astype(np.float64) + shift[i]
            for i in range(K + 1)
        ]

        # --- Radial basis at the current frame's centroid ----------------------
        e_r = centroid_cur / (np.linalg.norm(centroid_cur) + 1e-9)

        # Pick an orthogonal basis in the plane perpendicular to e_r
        ref = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(e_r, ref)) > 0.9:
            ref = np.array([1.0, 0.0, 0.0])
        e_p1 = ref - np.dot(ref, e_r) * e_r
        e_p1 /= np.linalg.norm(e_p1)
        e_p2 = np.cross(e_r, e_p1)

        # --- 1D projections for every frame ------------------------------------
        proj_r  = [f @ e_r  for f in aligned]
        proj_p1 = [f @ e_p1 for f in aligned]
        proj_p2 = [f @ e_p2 for f in aligned]

        # --- W₂² across consecutive pairs (t-1, t) for t ∈ [1, K] -----------
        w2_r_list:  list[float] = []
        w2_p1_list: list[float] = []
        w2_p2_list: list[float] = []

        for t_idx in range(1, K + 1):
            nq = self.n_quantiles
            w2_r_list.append(_wasserstein_1d_sq(proj_r[t_idx - 1],  proj_r[t_idx],  nq))
            w2_p1_list.append(_wasserstein_1d_sq(proj_p1[t_idx - 1], proj_p1[t_idx], nq))
            w2_p2_list.append(_wasserstein_1d_sq(proj_p2[t_idx - 1], proj_p2[t_idx], nq))

        w2_r  = np.array(w2_r_list)
        w2_p1 = np.array(w2_p1_list)
        w2_p2 = np.array(w2_p2_list)

        # Per-pair ratio, then median for robustness against a single outlier frame
        rhos = w2_r / (w2_p1 + w2_p2 + self.ratio_eps)
        rho  = float(np.median(rhos))

        w_r_rms    = float(np.sqrt(w2_r.mean()))
        w_perp_rms = float(np.sqrt((w2_p1 + w2_p2).mean()))

        return rho, w_r_rms, w_perp_rms
