"""
defenses/radial_jitter.py
-------------------------
Radial Jitter Defense — detects LiDAR spoofing by the temporal variance of
point positions along the sensor's radial direction.

The spoofing noise model (Sato et al. 2024) injects error along the radial
direction via three terms that are independently re-sampled every frame:

    δ_rand  — per-point, LiDAR-specific timing randomisation
    δ_inner — per-point, N(0, ~10 cm)
    δ_inter — per-frame scalar, N(0, ~35 cm), shared by ALL injected points

Real LiDAR returns on a rigid cluster are radially stable after ego-motion
compensation; their centroid and individual points stay within a few centimetres
of a smooth motion trajectory.  A spoofed cluster is not stable: every point
receives fresh δ_inner + δ_rand noise each frame, and the entire cluster is
radially offset by fresh δ_inter each frame.

Two complementary statistics capture this:

    σ_centroid  — std of residuals after fitting a linear velocity model
                  r(t) = a + bt to the per-frame cluster-centroid radial
                  distances.  Captures δ_inter (~35 cm).
                  Subtracting the linear fit removes constant-velocity radial
                  motion as well as the mean, so σ_centroid reflects only the
                  frame-to-frame scatter around a straight-line trajectory.
                  For δ_inter i.i.d. N(0, 0.35 m) the residual std is ~0.35 m;
                  for real clusters undergoing smooth motion the residuals
                  reflect only sensor noise + residual acceleration (~2–5 cm).

    σ_point     — std of per-correspondence ICP radial residuals pooled across
                  frames.  For each past-frame cluster, ICP gives explicit
                  matched point pairs (src → tgt).  The residual of each pair
                  projected onto the radial direction at the target point measures
                  δ_inner + δ_rand directly without shape contribution.  δ_inter
                  is absorbed by the ICP translation.

A frame is flagged as attacked when any cluster satisfies:

    σ_centroid > centroid_threshold  (default 0.25 m)
    OR σ_point  > point_threshold    (default 0.08 m)

Past-frame association
----------------------
Each past sweep is independently DBSCAN-clustered (same parameters as the
current frame).  Clusters are chained backward in time: starting from the
current cluster's centroid, the closest cluster in the most recent past frame
is matched if it is within motion_tolerance metres.  The matched cluster's
centroid then becomes the source for the next frame back, and so on.  A broken
link — no cluster within motion_tolerance — terminates the chain.

Using whole DBSCAN clusters (rather than a distance ball around the current
centroid) ensures each past-frame point belongs to exactly one association,
avoiding contamination from neighbouring clusters.  The per-step constant
threshold means the search is always local to the previous frame, regardless
of how many frames back the chain extends.
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


class RadialJitterDefense(BaseDefense):
    """Multi-frame defense exploiting the radial noise signature of LiDAR spoofing.

    Parameters
    ----------
    temporal_window
        Total frames visible (current + past history).  Pipeline retains at
        most ``temporal_window - 1`` past frames.
    min_history_frames
        Minimum past frames required before the defense produces a decision.
    ground_z_max
        Z threshold (metres, velodyne frame) below which points are dropped as
        ground before clustering.  Tuned for NuScenes lidar sensor height of
        1.84 m + 20 cm buffer.
    dbscan_eps
        DBSCAN neighbourhood radius in metres.  Applied to both the current
        frame and every past compensated sweep.
    dbscan_min_samples
        DBSCAN minimum neighbourhood size.
    min_points_per_cluster
        Clusters smaller than this (in any frame) are skipped.
    motion_tolerance
        Maximum centroid-to-centroid distance (metres) allowed between
        consecutive frames when chaining clusters backward in time.  The chain
        starts at the current cluster and matches the most recent past frame,
        then uses that matched cluster as the source for the next frame back,
        and so on.  A broken link terminates the chain.
    min_frames_associated
        Minimum number of past frames that must yield a matched cluster for the
        current cluster to be evaluated.
    icp_max_iter
        Maximum ICP iterations when aligning a past-frame cluster to the
        current one (used for σ_point computation).
    icp_max_correspondence_dist
        Maximum point-to-point correspondence distance in metres for ICP.
    centroid_method
        How to compute σ_centroid.  ``"linear_velocity"`` (default) fits
        r(t) = a + bt via OLS and measures the std of the residuals after
        subtracting that trend — removes constant-velocity radial motion
        before measuring scatter.  ``"first_diff"`` instead takes first
        differences Δr(t) = r(t) − r(t−1) and measures their std — a
        simpler statistic that is sensitive to frame-to-frame jumps but
        does not require a global linear fit.
    centroid_threshold
        σ_centroid (metres) above which a cluster is flagged as spoofed.
        Default 0.25 m — well below the expected residual std of ~0.35 m
        for δ_inter, with headroom above real cluster motion noise.
        Sweep target.
    point_threshold
        σ_point (metres) above which a cluster is flagged as spoofed.
        Default 0.08 m — just below δ_inner's std of ~10 cm.  Sweep target.
    ego_front
        Distance forward (metres) of the ego-vehicle exclusion box.  Points
        with x ≤ ego_front, x ≥ -ego_rear, |y| ≤ ego_side are removed before
        clustering.  Prevents the car body from forming spurious clusters.
    ego_rear
        Distance behind (metres) of the ego-vehicle exclusion box.
    ego_side
        Half-width (metres) of the ego-vehicle exclusion box.
    use_centroid
        Whether to compute and use σ_centroid for flagging.
    use_point
        Whether to compute and use σ_point (ICP-based) for flagging.
    history_source
        ``"dirty"`` (default) — use the post-attack history the vehicle
        received, so phantom objects inserted by the attack are present.
        ``"clean"`` — use pre-attack history for ablation.
    """

    def __init__(
        self,
        temporal_window: int = 6,
        min_history_frames: int = 5,
        ground_z_max: float = -1.65,
        dbscan_eps: float = 0.7,
        dbscan_min_samples: int = 10,
        min_points_per_cluster: int = 10,
        motion_tolerance: float = 1,
        min_frames_associated: int = 2,
        icp_max_iter: int = 30,
        icp_max_correspondence_dist: float = 0.5,
        centroid_threshold: float = 0.3,
        point_threshold: float = 0.08,
        ego_front: float = 3.0,
        ego_rear: float = 2.0,
        ego_side: float = 1.4,
        use_centroid: bool = True,
        use_point: bool = True,
        centroid_method: Literal["linear_velocity", "first_diff"] = "linear_velocity",
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
        self.icp_max_iter = icp_max_iter
        self.icp_max_correspondence_dist = icp_max_correspondence_dist
        self.centroid_threshold = centroid_threshold
        self.point_threshold = point_threshold
        self.ego_front = ego_front
        self.ego_rear = ego_rear
        self.ego_side = ego_side
        self.use_centroid = use_centroid
        self.use_point = use_point
        self.centroid_method = centroid_method
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
                "RadialJitterDefense requires frame.nuscenes_ego_pose to be set. "
                "Use NuScenesDataset (not KittiObjectDataset) with this defense."
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

        # Ego-compensate past sweeps into the current sensor frame (oldest-first)
        past_xyz_list = compensate_history(
            frame, hist, self.ground_z_max,
            self.ego_front, self.ego_rear, self.ego_side,
        )

        # DBSCAN the current frame (above-ground only)
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

        # DBSCAN each past compensated sweep (same parameters).
        past_clustered = dbscan_past_sweeps(
            past_xyz_list, self.dbscan_eps, self.dbscan_min_samples,
        )

        cluster_details: list[dict] = []
        n_tested = 0
        n_flagged = 0

        for lbl in unique_labels:
            pts_cur = cur_xyz_filt[labels_cur == lbl]
            centroid_cur = pts_cur.mean(axis=0)

            if len(pts_cur) < self.min_points_per_cluster:
                cluster_details.append({
                    "centroid": centroid_cur.tolist(),
                    "n_points_cur": int(len(pts_cur)),
                    "n_frames_associated": 0,
                    "sigma_centroid": None,
                    "sigma_point": None,
                    "flagged_centroid": None,
                    "flagged_point": None,
                    "flagged": False,
                    "skipped": "too_few_points",
                })
                continue

            valid_past = associate_cluster_chain(
                centroid_cur, past_clustered,
                self.motion_tolerance, self.min_points_per_cluster,
            )

            if len(valid_past) < self.min_frames_associated:
                cluster_details.append({
                    "centroid": centroid_cur.tolist(),
                    "n_points_cur": int(len(pts_cur)),
                    "n_frames_associated": len(valid_past),
                    "sigma_centroid": None,
                    "sigma_point": None,
                    "flagged_centroid": None,
                    "flagged_point": None,
                    "flagged": False,
                    "skipped": "too_few_frames",
                })
                continue

            n_tested += 1

            # --- σ_centroid: captures δ_inter --------------------------------
            if self.use_centroid:
                all_centroids = np.array(
                    [p.mean(axis=0) for p in valid_past] + [centroid_cur]
                )  # (K+1, 3)
                r_c = np.linalg.norm(all_centroids, axis=1)  # (K+1,)
                if self.centroid_method == "linear_velocity":
                    # Fit r(t) = a + b*t (closed-form OLS) and subtract to
                    # remove constant-velocity radial motion before measuring
                    # scatter.
                    t = np.arange(len(r_c), dtype=np.float64)
                    t_c = t - t.mean()
                    b = (t_c * (r_c - r_c.mean())).sum() / (t_c ** 2).sum()
                    r_detrended = r_c - r_c.mean() - b * t_c
                    sigma_centroid = float(np.std(r_detrended))
                else:  # first_diff
                    sigma_centroid = float(np.std(np.diff(r_c)))
            else:
                sigma_centroid = 0.0

            # --- σ_point: captures δ_inner + δ_rand --------------------------
            sigma_point = self._compute_sigma_point(pts_cur, valid_past) if self.use_point else 0.0

            flagged = (
                (self.use_centroid and sigma_centroid > self.centroid_threshold)
                or (self.use_point and sigma_point > self.point_threshold)
            )
            if flagged:
                n_flagged += 1

            cluster_details.append({
                "centroid": centroid_cur.tolist(),
                "n_points_cur": int(len(pts_cur)),
                "n_frames_associated": len(valid_past),
                "sigma_centroid": round(sigma_centroid, 4) if self.use_centroid else None,
                "sigma_point": round(sigma_point, 4) if self.use_point else None,
                "flagged_centroid": (sigma_centroid > self.centroid_threshold) if self.use_centroid else None,
                "flagged_point": (sigma_point > self.point_threshold) if self.use_point else None,
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
                "centroid_threshold": self.centroid_threshold,
                "point_threshold": self.point_threshold,
                "cluster_details": cluster_details,
            },
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_sigma_point(
        self,
        pts_cur: np.ndarray,
        valid_past: list[np.ndarray],
    ) -> float:
        """Compute σ_point via ICP correspondence radial residuals.

        For each past-frame cluster, run ICP against pts_cur to obtain explicit
        matched point pairs (src_idx → tgt_idx).  For each pair, project the
        residual vector (aligned_src − tgt) onto the radial direction at the
        target point.  Pool these residuals across all frames and return their std.

        This measures δ_inner + δ_rand directly: ICP absorbs the rigid per-frame
        δ_inter translation, and the correspondence residuals reflect only the
        per-point noise that cannot be explained by a rigid alignment.  Crucially,
        this avoids the cluster-depth bias of the previous centroid-deviation
        approach — residuals are ~1–3 cm for real rigid clusters and ~10–33 cm
        for spoofed clusters (depending on LiDAR δ_inner / δ_rand std).
        """
        try:
            import open3d as o3d
        except ImportError:
            logger.warning("open3d not available; σ_point will be 0")
            return 0.0

        tgt_pcd = o3d.geometry.PointCloud()
        tgt_pcd.points = o3d.utility.Vector3dVector(pts_cur.astype(np.float64))

        all_residuals: list[np.ndarray] = []

        for pts_past in valid_past:
            src_pcd = o3d.geometry.PointCloud()
            src_pcd.points = o3d.utility.Vector3dVector(pts_past.astype(np.float64))

            reg = o3d.pipelines.registration.registration_icp(
                src_pcd,
                tgt_pcd,
                max_correspondence_distance=self.icp_max_correspondence_dist,
                init=np.eye(4),
                estimation_method=(
                    o3d.pipelines.registration.TransformationEstimationPointToPoint()
                ),
                criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                    max_iteration=self.icp_max_iter
                ),
            )
            corrs = np.asarray(reg.correspondence_set)  # (M, 2): (src_idx, tgt_idx)
            if len(corrs) == 0:
                continue

            T = np.asarray(reg.transformation)
            aligned_corr = (
                T[:3, :3] @ pts_past[corrs[:, 0]].T + T[:3, 3:4]
            ).T  # (M, 3)
            tgt_corr = pts_cur[corrs[:, 1]]  # (M, 3)

            # Radial unit direction at each target point
            tgt_g = tgt_corr / (
                np.linalg.norm(tgt_corr, axis=1, keepdims=True) + 1e-9
            )  # (M, 3)

            # Signed radial residual: positive = aligned point is farther from sensor
            residuals = ((aligned_corr - tgt_corr) * tgt_g).sum(axis=1)  # (M,)
            all_residuals.append(residuals)

        if not all_residuals:
            return 0.0

        return float(np.std(np.concatenate(all_residuals)))
