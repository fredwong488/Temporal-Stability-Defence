"""
defenses/_multiframe_common.py
-------------------------------
Shared preprocessing, history compensation, and cluster-association helpers
for multi-frame LiDAR defenses.  Both RadialJitterDefense and
WassersteinAnisotropyDefense call these functions directly.
"""

from __future__ import annotations

import logging
from collections import deque

import numpy as np
from sklearn.cluster import DBSCAN

from ..types import Frame

logger = logging.getLogger(__name__)


def remove_ego_box(
    xyz: np.ndarray,
    ego_front: float,
    ego_rear: float,
    ego_side: float,
) -> np.ndarray:
    """Remove points inside the ego-vehicle bounding box.

    In the NuScenes LiDAR sensor frame: +y is forward, +x is right.
    """
    in_box = (
        (xyz[:, 1] <= ego_front)
        & (xyz[:, 1] >= -ego_rear)
        & (np.abs(xyz[:, 0]) <= ego_side)
    )
    return xyz[~in_box]


def patchwork_ground_segment(
    xyzw: np.ndarray,
    sensor_height: float | None = None,
    num_iter: int | None = None,
    uprightness_thr: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Segment ``xyzw`` (N×4, XYZI, gravity-levelled) into ground / non-ground.

    Parameters
    ----------
    xyzw
        (N, 4) float32 array in the gravity-aligned (ego) frame so that
        ground lies at z ≈ 0.
    sensor_height
        Patchwork++ ``sensor_height`` parameter.  In the levelled ego frame
        the sensor is at its mounting height above the ground plane.
        If ``None``, the Patchwork++ default is used.
    num_iter
        Number of Patchwork++ ground-estimation iterations.
        If ``None``, the Patchwork++ default is used.
    uprightness_thr
        Uprightness threshold: planes with normal dot-product with +z below
        this value are rejected as non-ground.
        If ``None``, the Patchwork++ default is used.

    Returns
    -------
    ground_idx : np.ndarray of int
        Indices into ``xyzw`` of ground points.
    nonground_idx : np.ndarray of int
        Indices into ``xyzw`` of non-ground points.
    """
    import pypatchworkpp  # lazy import — avoids hard dep at module load time

    params = pypatchworkpp.Parameters()
    if sensor_height is not None:
        params.sensor_height = sensor_height
    if num_iter is not None:
        params.num_iter = num_iter
    if uprightness_thr is not None:
        params.uprightness_thr = uprightness_thr
    params.verbose = False

    ppp = pypatchworkpp.patchworkpp(params)
    ppp.estimateGround(xyzw.astype(np.float32))

    ground_idx = np.asarray(ppp.getGroundIndices(), dtype=int)
    nonground_idx = np.asarray(ppp.getNongroundIndices(), dtype=int)
    return ground_idx, nonground_idx


def compensate_history(
    current_frame: Frame,
    hist: deque[Frame],
    ground_z_max: float,
    ego_front: float,
    ego_rear: float,
    ego_side: float,
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
                "compensate_history: past frame %s missing ego pose — skipping",
                f.frame_id,
            )
            continue
        T = cur_inv @ f.nuscenes_ego_pose.astype(np.float64)
        pts = f.lidar[:, :3].astype(np.float64)
        pts = pts[pts[:, 2] > ground_z_max]
        pts = remove_ego_box(pts, ego_front, ego_rear, ego_side)
        compensated = (T[:3, :3] @ pts.T + T[:3, 3:4]).T.astype(np.float32)
        result.append(compensated)
    return result


def dbscan_past_sweeps(
    past_xyz_list: list[np.ndarray],
    eps: float,
    min_samples: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """DBSCAN each past compensated sweep.

    Returns a list parallel to past_xyz_list where each entry is
    ``(xyz_filt, labels)``.  If an xyz array has fewer points than
    ``min_samples`` a trivial all-noise label array is returned.
    """
    past_clustered: list[tuple[np.ndarray, np.ndarray]] = []
    for xyz_past in past_xyz_list:
        if len(xyz_past) < min_samples:
            past_clustered.append(
                (xyz_past, np.full(len(xyz_past), -1, dtype=int))
            )
            continue
        lbl = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=1).fit_predict(xyz_past)
        past_clustered.append((xyz_past, lbl))
    return past_clustered


def precompute_cluster_data(
    past_clustered: list[tuple[np.ndarray, np.ndarray]],
) -> list[tuple[list[np.ndarray], np.ndarray]]:
    """Convert (xyz, labels) pairs into precomputed (cluster_pts_list, centroids).

    Separates each past frame's points into per-cluster arrays and computes
    their centroids once, so ``associate_cluster_chain`` can skip recomputing
    them for every current cluster it tests.

    Parameters
    ----------
    past_clustered
        Output of ``dbscan_past_sweeps``: list of (xyz, labels) per past frame,
        where xyz is in the current sensor frame.

    Returns
    -------
    list of (cluster_pts_list, centroids) per past frame:
        cluster_pts_list : list of (N_k, 3) arrays, one per non-noise cluster
        centroids        : (K, 3) float32 array of per-cluster centroids
    """
    result: list[tuple[list[np.ndarray], np.ndarray]] = []
    for xyz, labels in past_clustered:
        unique = sorted(l for l in set(labels) if l != -1)
        if not unique:
            result.append(([], np.empty((0, 3), dtype=np.float32)))
            continue
        cluster_pts = [xyz[labels == l] for l in unique]
        centroids = np.array(
            [p.mean(axis=0) for p in cluster_pts], dtype=np.float32
        )
        result.append((cluster_pts, centroids))
    return result


def associate_cluster_chain(
    centroid_cur: np.ndarray,
    past_frame_data: list[tuple[list[np.ndarray], np.ndarray]],
    motion_tolerance: float,
    min_points_per_cluster: int,
) -> list[np.ndarray]:
    """Chain-based backward tracking from the current cluster centroid.

    Starting from ``centroid_cur``, match the most recent past frame, then
    use that cluster's centroid as the source for the next frame back, and so
    on.  A broken link (no cluster within ``motion_tolerance``) terminates the
    chain.

    Parameters
    ----------
    past_frame_data
        Output of ``precompute_cluster_data``: list of (cluster_pts_list,
        centroids) per past frame in the current sensor frame, oldest-first.

    Returns the matched past clusters in **oldest-first** (chronological)
    order as a list of (N_t, 3) xyz arrays.  The current frame is NOT included.
    """
    n_past = len(past_frame_data)
    valid_past_reversed: list[np.ndarray] = []
    source_centroid = centroid_cur

    for i in reversed(range(n_past)):  # most recent → oldest
        cluster_pts, centroids = past_frame_data[i]

        if len(cluster_pts) == 0:
            break

        dists = np.linalg.norm(centroids - source_centroid, axis=1)
        dists_gated = np.where(dists < motion_tolerance, dists, np.inf)
        best_idx = int(np.argmin(dists_gated))

        if not np.isfinite(dists_gated[best_idx]):
            break  # chain broken

        best_pts = cluster_pts[best_idx]
        if len(best_pts) < min_points_per_cluster:
            break  # cluster too sparse to continue

        valid_past_reversed.append(best_pts)
        source_centroid = centroids[best_idx]  # reuse precomputed centroid

    return valid_past_reversed[::-1]  # restore chronological order
