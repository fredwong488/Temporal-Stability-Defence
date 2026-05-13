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

    σ_centroid  — std of per-frame cluster-centroid radial distance after
                  removing a linear motion trend.  Captures δ_inter (~35 cm).

    σ_point     — std of per-point radial deviations from the cluster centroid,
                  pooled across frames after ICP-aligning each past-frame cluster
                  to the current one.  Captures δ_inner + δ_rand; δ_inter is
                  absorbed by the ICP translation.

A frame is flagged as attacked when any cluster satisfies:

    σ_centroid > centroid_threshold  (default 0.30 m)
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
    centroid_threshold
        σ_centroid (metres) above which a cluster is flagged as spoofed.
        Default 0.30 m — just below δ_inter's std of ~35 cm.  Sweep target.
    point_threshold
        σ_point (metres) above which a cluster is flagged as spoofed.
        Default 0.08 m — just below δ_inner's std of ~10 cm.  Sweep target.
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
        dbscan_eps: float = 0.25,
        dbscan_min_samples: int = 17,
        min_points_per_cluster: int = 17,
        motion_tolerance: float = 1,
        min_frames_associated: int = 2,
        icp_max_iter: int = 30,
        icp_max_correspondence_dist: float = 1.0,
        centroid_threshold: float = 0.30,
        point_threshold: float = 0.08,
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
        past_xyz_list = self._compensate_history(frame, hist)

        # DBSCAN the current frame (above-ground only)
        cur_xyz = frame.lidar[:, :3]
        cur_xyz_filt = cur_xyz[cur_xyz[:, 2] > self.ground_z_max]

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
        # past_clustered[i] = (xyz_filt, labels) for the i-th past frame, oldest-first.
        past_clustered: list[tuple[np.ndarray, np.ndarray]] = []
        for xyz_past in past_xyz_list:
            if len(xyz_past) < self.dbscan_min_samples:
                past_clustered.append((xyz_past, np.full(len(xyz_past), -1, dtype=int)))
                continue
            lbl_past = DBSCAN(
                eps=self.dbscan_eps,
                min_samples=self.dbscan_min_samples,
                n_jobs=1,
            ).fit_predict(xyz_past)
            past_clustered.append((xyz_past, lbl_past))

        n_past = len(past_clustered)

        cluster_details: list[dict] = []
        n_tested = 0
        n_flagged = 0

        for lbl in unique_labels:
            pts_cur = cur_xyz_filt[labels_cur == lbl]
            if len(pts_cur) < self.min_points_per_cluster:
                continue

            centroid_cur = pts_cur.mean(axis=0)

            # Chain-based backward tracking: start from the current centroid,
            # match the most recent past frame, then use that matched cluster's
            # centroid as the source for the next frame back, and so on.
            # A broken link (no cluster within motion_tolerance) stops the chain.
            valid_past_reversed: list[np.ndarray] = []
            source_centroid = centroid_cur
            for i in reversed(range(n_past)):  # most recent → oldest
                xyz_past, lbl_past = past_clustered[i]

                past_unique = [l for l in set(lbl_past) if l != -1]
                if not past_unique:
                    break

                past_centroids = np.array([
                    xyz_past[lbl_past == l].mean(axis=0) for l in past_unique
                ])
                dists = np.linalg.norm(past_centroids - source_centroid, axis=1)
                dists_gated = np.where(dists < self.motion_tolerance, dists, np.inf)
                best_idx = int(np.argmin(dists_gated))

                if not np.isfinite(dists_gated[best_idx]):
                    break  # chain broken

                best_pts = xyz_past[lbl_past == past_unique[best_idx]]
                if len(best_pts) < self.min_points_per_cluster:
                    break  # cluster too sparse to continue

                valid_past_reversed.append(best_pts)
                source_centroid = best_pts.mean(axis=0)

            # Reverse to restore chronological order (oldest first → current last)
            valid_past = valid_past_reversed[::-1]

            if len(valid_past) < self.min_frames_associated:
                continue

            n_tested += 1

            # --- σ_centroid: captures δ_inter --------------------------------
            # Per-frame centroid radial distances (valid past, oldest first → current last)
            all_centroids = np.array(
                [p.mean(axis=0) for p in valid_past] + [centroid_cur]
            )  # (K+1, 3)
            r_c = np.linalg.norm(all_centroids, axis=1)  # (K+1,)
            t_idx = np.arange(len(r_c), dtype=float)
            trend = np.polyval(np.polyfit(t_idx, r_c, 1), t_idx)
            sigma_centroid = float(np.std(r_c - trend))

            # --- σ_point: captures δ_inner + δ_rand --------------------------
            sigma_point = self._compute_sigma_point(pts_cur, valid_past)

            flagged = (
                sigma_centroid > self.centroid_threshold
                or sigma_point > self.point_threshold
            )
            if flagged:
                n_flagged += 1

            cluster_details.append({
                "centroid": centroid_cur.tolist(),
                "n_points_cur": int(len(pts_cur)),
                "n_frames_associated": len(valid_past),
                "sigma_centroid": round(sigma_centroid, 4),
                "sigma_point": round(sigma_point, 4),
                "flagged_centroid": sigma_centroid > self.centroid_threshold,
                "flagged_point": sigma_point > self.point_threshold,
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

    def _compensate_history(
        self,
        current_frame: Frame,
        hist: deque[Frame],
    ) -> list[np.ndarray]:
        """Return past sweeps transformed into the current sensor frame.

        Each entry is an (N_t, 3) float32 xyz array with ground points removed,
        in the same order (oldest-first) as hist.
        """
        cur_inv = np.linalg.inv(current_frame.nuscenes_ego_pose.astype(np.float64))
        result: list[np.ndarray] = []
        for f in hist:
            if f.nuscenes_ego_pose is None:
                logger.warning(
                    "RadialJitterDefense: past frame %s missing ego pose — skipping",
                    f.frame_id,
                )
                continue
            T = cur_inv @ f.nuscenes_ego_pose.astype(np.float64)
            pts = f.lidar[:, :3].astype(np.float64)
            compensated = (T[:3, :3] @ pts.T + T[:3, 3:4]).T.astype(np.float32)
            result.append(compensated[compensated[:, 2] > self.ground_z_max])
        return result

    def _compute_sigma_point(
        self,
        pts_cur: np.ndarray,
        valid_past: list[np.ndarray],
    ) -> float:
        """Compute σ_point via ICP-aligned per-point radial deviations.

        For each past-frame cluster, run ICP to align it to pts_cur (absorbing
        rigid object motion).  Project all points — current and each aligned
        past batch — onto the radial direction from the sensor origin, then
        return the std of per-point deviations from the cluster centroid along
        that axis.

        δ_inter (a rigid per-frame radial shift) is absorbed by ICP translation,
        leaving residuals due to δ_inner + δ_rand.
        """
        try:
            import open3d as o3d
        except ImportError:
            logger.warning("open3d not available; σ_point will be 0")
            return 0.0

        centroid_cur = pts_cur.mean(axis=0)
        g = centroid_cur / (np.linalg.norm(centroid_cur) + 1e-9)

        dev_cur = ((pts_cur - centroid_cur) * g).sum(axis=1)
        all_devs: list[np.ndarray] = [dev_cur]

        tgt_pcd = o3d.geometry.PointCloud()
        tgt_pcd.points = o3d.utility.Vector3dVector(pts_cur.astype(np.float64))

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
            T = np.asarray(reg.transformation)
            aligned = (T[:3, :3] @ pts_past.T + T[:3, 3:4]).T.astype(np.float32)

            centroid_aligned = aligned.mean(axis=0)
            dev_past = ((aligned - centroid_aligned) * g).sum(axis=1)
            all_devs.append(dev_past)

        return float(np.std(np.concatenate(all_devs)))
