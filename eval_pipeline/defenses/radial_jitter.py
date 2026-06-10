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
import time
from collections import deque
from typing import Callable, Literal

import numpy as np
from sklearn.cluster import DBSCAN, HDBSCAN

from ..base import BaseDefense
from ..types import DetectionResult, Frame, FrameHistory, Prediction
from ._multiframe_common import (
    associate_cluster_chain,
    patchwork_ground_segment,
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
        Z threshold (metres, ego/car frame) below which points are dropped as
        ground before clustering.  Ground is at z ≈ 0 in the ego frame
        regardless of sensor tilt; 0.1 m provides a 10 cm clearance buffer.
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
    clusterer
        Which clustering algorithm to use.  ``"dbscan"`` (default) uses
        sklearn DBSCAN with ``dbscan_eps`` and ``dbscan_min_samples``.
        ``"hdbscan"`` uses sklearn HDBSCAN with ``hdbscan_min_cluster_size``
        as the minimum cluster size and ``dbscan_min_samples`` as the density
        estimation parameter (HDBSCAN ``min_samples``).  Note: ``dbscan_eps``
        is ignored for HDBSCAN.
    hdbscan_min_cluster_size
        Minimum number of points for a group to survive as a cluster under
        HDBSCAN.  Defaults to 10 to match ``min_points_per_cluster`` — the
        floor the rest of the pipeline already enforces.  Ignored when
        ``clusterer="dbscan"``.
    use_centroid
        Whether to compute and use σ_centroid for flagging.
    use_point
        Whether to compute and use σ_point (ICP-based) for flagging.
    flag_condition
        How to combine the two signal thresholds when both are active.
        ``"or"`` (default) flags a cluster if either σ_centroid or σ_point
        exceeds its threshold.  ``"and"`` requires both to exceed their
        respective thresholds simultaneously.
    history_source
        ``"dirty"`` (default) — use the post-attack history the vehicle
        received, so phantom objects inserted by the attack are present.
        ``"clean"`` — use pre-attack history for ablation.
    cluster_on_bev
        If ``True``, cluster on the 2-D bird's-eye-view projection (x, y)
        rather than full 3-D (x, y, z).  Useful when height variation within
        a cluster is large enough to split it under 3-D clustering but the
        object is coherent in the horizontal plane.  Applies to both DBSCAN
        and HDBSCAN.
    ground_method
        How to remove ground points before clustering.  ``"zcut"`` (default)
        drops all points whose z-coordinate in the level ego frame is at or
        below ``ground_z_max`` (fast, no external dependency).  ``"patchwork"``
        uses Patchwork++ (region-wise plane fitting) for more accurate ground
        segmentation on slopes, curbs, and varying terrain.  Requires the
        ``pypatchworkpp`` package.
    patchwork_sensor_height
        Sensor height above the ground plane (metres) passed to Patchwork++.
        In the NuScenes levelled ego frame the LIDAR_TOP is mounted ≈ 1.84 m
        above ground.  Only used when ``ground_method="patchwork"``.
    patchwork_num_iter
        Number of Patchwork++ ground-estimation iterations.  If ``None``
        (default), the Patchwork++ library default is used.  Only used
        when ``ground_method="patchwork"``.
    patchwork_uprightness_thr
        Uprightness threshold for Patchwork++: candidate planes whose normal
        has a dot-product with +z below this value are rejected.  If ``None``
        (default), the Patchwork++ library default is used.  Only used
        when ``ground_method="patchwork"``.
    use_predictions
        If ``True``, skip DBSCAN/HDBSCAN and instead derive clusters from the
        detector predictions attached to each frame (``frame.predictions``).
        Each predicted bounding box becomes one cluster: points are assigned to
        the first box whose oriented extent they fall within (first-match wins
        on overlap).  Boxes with no points inside are dropped.  Ground removal
        and ego-box exclusion still apply before the assignment.  Requires
        ``frame.predictions`` to be populated on both the current frame and all
        history frames; ``dbscan_eps``, ``dbscan_min_samples``, ``clusterer``,
        ``hdbscan_min_cluster_size``, and ``cluster_on_bev`` are ignored when
        this flag is set.
    """

    def __init__(
        self,
        temporal_window: int = 12,
        min_history_frames: int = 5,
        ground_z_max: float = 0.1,
        dbscan_eps: float = 0.7,
        dbscan_min_samples: int = 20,
        min_points_per_cluster: int = 10,
        motion_tolerance: float = 1.92,
        min_frames_associated: int = 2,
        icp_max_iter: int = 30,
        icp_max_correspondence_dist: float = 0.5,
        centroid_threshold: float = 0.3,
        point_threshold: float = 0.08,
        ego_front: float = 2.0,
        ego_rear: float = 2.0,
        ego_side: float = 1.4,
        use_centroid: bool = True,
        use_point: bool = True,
        flag_condition: Literal["or", "and"] = "or",
        centroid_method: Literal["linear_velocity", "first_diff"] = "linear_velocity",
        history_source: Literal["clean", "dirty"] = "dirty",
        cluster_on_bev: bool = False,
        clusterer: Literal["dbscan", "hdbscan"] = "dbscan",
        hdbscan_min_cluster_size: int = 10,
        ground_method: Literal["zcut", "patchwork"] = "zcut",
        patchwork_sensor_height: float = 1.84,
        patchwork_num_iter: int | None = None,
        patchwork_uprightness_thr: float | None = None,
        use_predictions: bool = False,
        debug: bool = False,
        defense_frame_hook: Callable[[Frame, "DetectionResult", list[np.ndarray]], None] | None = None,
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
        self.flag_condition = flag_condition
        self.centroid_method = centroid_method
        self.history_source = history_source
        self.cluster_on_bev = cluster_on_bev
        self.clusterer = clusterer
        self.hdbscan_min_cluster_size = hdbscan_min_cluster_size
        self.ground_method = ground_method
        self.patchwork_sensor_height = patchwork_sensor_height
        self.patchwork_num_iter = patchwork_num_iter
        self.patchwork_uprightness_thr = patchwork_uprightness_thr
        self.use_predictions = use_predictions
        self.debug = debug
        self._defense_frame_hook = defense_frame_hook
        # Maps (frame_id, is_attacked) → (xyz_filt, labels, cluster_pts, centroids)
        # in own sensor frame.  DBSCAN is distance-invariant under rigid transforms,
        # so labels and per-cluster data computed here are reused after
        # ego-compensation in detect().
        self._dbscan_cache: dict[
            tuple[str, bool],
            tuple[np.ndarray, np.ndarray, list[np.ndarray], np.ndarray],
        ] = {}

    @property
    def temporal_window(self) -> int:
        return self._temporal_window

    def reset(self) -> None:
        self._dbscan_cache.clear()

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
        # Control the history length ourselves rather than trusting the caller to
        # size it to exactly temporal_window - 1.  The deque is oldest-first, so
        # keep the most recent temporal_window - 1 frames.  This keeps the defense
        # invariant to an over-sized history (a longer window would otherwise add
        # frames the σ statistics were never tuned for).
        max_past = max(0, self._temporal_window - 1)
        if len(hist) > max_past:
            hist = deque(list(hist)[-max_past:])

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

        # Current frame: filter + DBSCAN (result cached so it's free next call as
        # a past frame).
        _t0_total = time.perf_counter()
        _t0_cluster = time.perf_counter()
        cur_xyz_filt, labels_cur, _, _ = self._get_or_cluster_frame(frame)
        _elapsed_cluster_s = time.perf_counter() - _t0_cluster

        if len(cur_xyz_filt) == 0:
            return DetectionResult(
                is_attack_detected=False,
                confidence=0.0,
                metadata={"reason": "no_points_above_ground"},
            )

        unique_labels = set(labels_cur) - {-1}
        if not unique_labels:
            return DetectionResult(
                is_attack_detected=False,
                confidence=0.0,
                metadata={"reason": "no_clusters"},
            )

        # Past frames: transform cached per-cluster pts and centroids into the
        # current sensor frame.  Centroids transform as points (mean commutes
        # with rigid transforms), so this is O(K) not O(N_points) per frame.
        cur_inv = np.linalg.inv(frame.nuscenes_ego_pose.astype(np.float64))
        active_ids = {frame.frame_id} | {f.frame_id for f in hist}
        self._evict_stale_cache(active_ids)

        past_frame_data: list[tuple[list[np.ndarray], np.ndarray]] = []
        past_xyz_list: list[np.ndarray] = []  # populated only when defense_frame_hook is set
        for f in hist:
            if f.nuscenes_ego_pose is None:
                logger.warning(
                    "detect: past frame %s missing ego pose — skipping", f.frame_id
                )
                past_frame_data.append(([], np.empty((0, 3), dtype=np.float32)))
                if self._defense_frame_hook is not None:
                    past_xyz_list.append(np.empty((0, 3), dtype=np.float32))
                continue
            xyz_filt_own, _, cluster_pts_own, centroids_own = self._get_or_cluster_frame(f)
            R = cur_inv[:3, :3]
            t_cur = cur_inv[:3, 3:4]
            T_past = R @ f.nuscenes_ego_pose.astype(np.float64)[:3, :3]
            t_past = R @ f.nuscenes_ego_pose.astype(np.float64)[:3, 3:4] + t_cur
            if self._defense_frame_hook is not None:
                past_xyz_list.append(
                    (T_past @ xyz_filt_own.astype(np.float64).T + t_past).T.astype(np.float32)
                    if len(xyz_filt_own) > 0 else np.empty((0, 3), dtype=np.float32)
                )
            if len(cluster_pts_own) == 0:
                past_frame_data.append(([], np.empty((0, 3), dtype=np.float32)))
                continue
            comp_centroids = (
                T_past @ centroids_own.astype(np.float64).T + t_past
            ).T.astype(np.float32)
            comp_cluster_pts = [
                (T_past @ pts.astype(np.float64).T + t_past).T.astype(np.float32)
                for pts in cluster_pts_own
            ]
            past_frame_data.append((comp_cluster_pts, comp_centroids))

        cluster_details: list[dict] = []
        n_tested = 0
        n_flagged = 0
        _elapsed_sigma_centroid_s = 0.0
        _elapsed_sigma_point_s = 0.0

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
                centroid_cur, past_frame_data,
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
            _t0_sc = time.perf_counter()
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
            _elapsed_sigma_centroid_s += time.perf_counter() - _t0_sc

            # --- σ_point: captures δ_inner  --------------------------
            _t0_sp = time.perf_counter()
            sigma_point = self._compute_sigma_point(pts_cur, valid_past) if self.use_point else 0.0
            _elapsed_sigma_point_s += time.perf_counter() - _t0_sp

            centroid_flagged = self.use_centroid and sigma_centroid > self.centroid_threshold
            point_flagged = self.use_point and sigma_point > self.point_threshold
            if self.flag_condition == "and":
                flagged = centroid_flagged and point_flagged
            else:
                flagged = centroid_flagged or point_flagged
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

        meta: dict = {
            "n_clusters_tested": n_tested,
            "n_clusters_flagged": n_flagged,
            "centroid_threshold": self.centroid_threshold,
            "point_threshold": self.point_threshold,
            "cluster_details": cluster_details,
            "elapsed_s": {
                "cluster": _elapsed_cluster_s,
                "sigma_centroid": _elapsed_sigma_centroid_s,
                "sigma_point": _elapsed_sigma_point_s,
                "total": time.perf_counter() - _t0_total,
            },
        }
        if self.debug:
            meta["xyz_filt"] = cur_xyz_filt
            meta["labels_cur"] = labels_cur

        result = DetectionResult(
            is_attack_detected=is_attack,
            confidence=float(confidence),
            metadata=meta,
        )
        if self._defense_frame_hook is not None:
            self._defense_frame_hook(frame, result, past_xyz_list)
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _evict_stale_cache(self, active_frame_ids: set[str]) -> None:
        stale = [k for k in self._dbscan_cache if k[0] not in active_frame_ids]
        for k in stale:
            del self._dbscan_cache[k]

    def _get_or_cluster_frame(
        self, frame: Frame
    ) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], np.ndarray]:
        """Return (xyz_filt, labels, cluster_pts, centroids) for frame.

        All arrays are in the frame's own sensor coordinates.  Computed once
        and cached; subsequent calls are a dict lookup.

        cluster_pts : list of (N_k, 3) arrays, one per non-noise cluster
        centroids   : (K, 3) float32 array of per-cluster centroids
        """
        key = (frame.frame_id, bool(frame.is_attacked))
        if key in self._dbscan_cache:
            return self._dbscan_cache[key]

        xyz = frame.lidar[:, :3].astype(np.float64)
        if frame.nuscenes_sensor_to_ego is None:
            raise ValueError(f"nuscenes_sensor_to_ego missing from frame {frame.frame_id}")
        s2e = frame.nuscenes_sensor_to_ego.astype(np.float64)

        if self.ground_method == "patchwork":
            # Apply only the rotation part of s2e to gravity-level the points
            # (ground lies at z ≈ -sensor_height in this frame, matching the
            # sensor_height parameter).  The full translation is not needed for
            # ground segmentation and is not applied here.  Downstream caching
            # and ego-compensation continue to use the original sensor-frame xyz.
            xyz_leveled = (s2e[:3, :3] @ xyz.T).T
            intensity = frame.lidar[:, 3:4].astype(np.float64)
            xyzw_leveled = np.concatenate([xyz_leveled, intensity], axis=1).astype(np.float32)
            _, nonground_idx = patchwork_ground_segment(
                xyzw_leveled,
                self.patchwork_sensor_height,
                self.patchwork_num_iter,
                self.patchwork_uprightness_thr,
            )
            xyz = xyz[nonground_idx]
        else:
            # Compute z in the level ego frame (ground is at z≈0 regardless of
            # sensor tilt).
            z_ego = s2e[2, :3] @ xyz.T + s2e[2, 3]
            xyz = xyz[z_ego > self.ground_z_max]

        xyz_filt = remove_ego_box(
            xyz,
            self.ego_front, self.ego_rear, self.ego_side,
        )
        if self.use_predictions:
            labels, cluster_pts, centroids = self._cluster_from_predictions(
                xyz_filt, frame.predictions
            )
        else:
            if len(xyz_filt) >= self.dbscan_min_samples:
                cluster_input = xyz_filt[:, :2] if self.cluster_on_bev else xyz_filt
                if self.clusterer == "hdbscan":
                    labels = HDBSCAN(
                        min_cluster_size=self.hdbscan_min_cluster_size,
                        min_samples=self.dbscan_min_samples,
                        n_jobs=1,
                        copy=False,
                    ).fit_predict(cluster_input)
                else:
                    labels = DBSCAN(
                        eps=self.dbscan_eps,
                        min_samples=self.dbscan_min_samples,
                        n_jobs=1,
                    ).fit_predict(cluster_input)
            else:
                labels = np.full(len(xyz_filt), -1, dtype=int)

            unique = sorted(l for l in set(labels) if l != -1)
            if unique:
                cluster_pts = [xyz_filt[labels == l] for l in unique]
                centroids = np.array(
                    [p.mean(axis=0) for p in cluster_pts], dtype=np.float32
                )
            else:
                cluster_pts = []
                centroids = np.empty((0, 3), dtype=np.float32)

        self._dbscan_cache[key] = (xyz_filt, labels, cluster_pts, centroids)
        return xyz_filt, labels, cluster_pts, centroids

    def _cluster_from_predictions(
        self,
        xyz_filt: np.ndarray,
        predictions: list[Prediction],
    ) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
        """Assign points to clusters using predicted 3-D bounding boxes.

        Points are tested against each box in order; the first matching box
        wins (no double-assignment).  Boxes with no assigned points are
        silently dropped.  Returns (labels, cluster_pts, centroids) with the
        same contract as the DBSCAN path: labels are 0-based cluster indices,
        -1 for unassigned points.

        Both xyz_filt and the predictions are expected to be in the sensor
        (velodyne) frame, which is the convention used throughout this class.
        """
        labels = np.full(len(xyz_filt), -1, dtype=int)
        cluster_pts: list[np.ndarray] = []
        centroids_list: list[np.ndarray] = []
        cluster_idx = 0

        for pred in predictions:
            dx = xyz_filt[:, 0] - pred.x
            dy = xyz_filt[:, 1] - pred.y
            dz = xyz_filt[:, 2] - pred.z

            cos_r = np.cos(-pred.rotation_y)
            sin_r = np.sin(-pred.rotation_y)
            lx = cos_r * dx - sin_r * dy
            ly = sin_r * dx + cos_r * dy

            inside = (
                (np.abs(lx) <= pred.length / 2)
                & (np.abs(ly) <= pred.width / 2)
                & (np.abs(dz) <= pred.height / 2)
            )
            assign = inside & (labels == -1)
            if not assign.any():
                continue

            labels[assign] = cluster_idx
            pts = xyz_filt[assign]
            cluster_pts.append(pts)
            centroids_list.append(pts.mean(axis=0))
            cluster_idx += 1

        centroids = (
            np.array(centroids_list, dtype=np.float32)
            if centroids_list
            else np.empty((0, 3), dtype=np.float32)
        )
        return labels, cluster_pts, centroids

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
