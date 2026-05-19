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
    """Remove points inside the ego-vehicle bounding box."""
    in_box = (
        (xyz[:, 0] <= ego_front)
        & (xyz[:, 0] >= -ego_rear)
        & (np.abs(xyz[:, 1]) <= ego_side)
    )
    return xyz[~in_box]


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
        compensated = (T[:3, :3] @ pts.T + T[:3, 3:4]).T.astype(np.float32)
        result.append(
            remove_ego_box(
                compensated[compensated[:, 2] > ground_z_max],
                ego_front, ego_rear, ego_side,
            )
        )
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


def associate_cluster_chain(
    centroid_cur: np.ndarray,
    past_clustered: list[tuple[np.ndarray, np.ndarray]],
    motion_tolerance: float,
    min_points_per_cluster: int,
) -> list[np.ndarray]:
    """Chain-based backward tracking from the current cluster centroid.

    Starting from ``centroid_cur``, match the most recent past frame, then
    use that cluster's centroid as the source for the next frame back, and so
    on.  A broken link (no cluster within ``motion_tolerance``) terminates the
    chain.

    Returns the matched past clusters in **oldest-first** (chronological)
    order as a list of (N_t, 3) xyz arrays.  The current frame is NOT included.
    """
    n_past = len(past_clustered)
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
        dists_gated = np.where(dists < motion_tolerance, dists, np.inf)
        best_idx = int(np.argmin(dists_gated))

        if not np.isfinite(dists_gated[best_idx]):
            break  # chain broken

        best_pts = xyz_past[lbl_past == past_unique[best_idx]]
        if len(best_pts) < min_points_per_cluster:
            break  # cluster too sparse to continue

        valid_past_reversed.append(best_pts)
        source_centroid = best_pts.mean(axis=0)

    return valid_past_reversed[::-1]  # restore chronological order
