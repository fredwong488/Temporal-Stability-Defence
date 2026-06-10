"""
detectors/_sweep_accumulation.py
--------------------------------
Multi-sweep accumulation for NuScenes-style detectors.

The NuScenes ``cbgs_pp_multihead`` model is trained on point clouds accumulated
from up to 10 LiDAR sweeps (MAX_SWEEPS: 10), each transformed into the current
sensor frame with the 5th channel holding the per-point time-lag (seconds back
from the current frame; the current sweep is 0).

``accumulate_sweeps`` reconstructs that input at inference time from the
pipeline's frame history.  The ego-motion transform mirrors
``defenses/_multiframe_common.compensate_history`` (T = inv(cur_pose) @ past_pose),
but here we keep intensity and append the time channel, and we do NOT apply any
ground / ego-box filtering — the detector needs the full cloud.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable

import numpy as np

from ..types import Frame

logger = logging.getLogger(__name__)


def accumulate_sweeps(
    frame: Frame,
    history: Iterable[Frame] | None = None,
    max_sweeps: int = 10,
    max_time_span: float = 0.5,
) -> np.ndarray:
    """Accumulate recent sweeps into the current sensor frame.

    Parameters
    ----------
    frame
        The current frame (its lidar becomes the time-lag-0 sweep).
    history
        Past frames, oldest-first (as stored in the pipeline history deques).
        Only frames within ``max_time_span`` seconds and up to ``max_sweeps - 1``
        of them (newest-first) are included.
    max_sweeps
        Maximum total sweeps (current + past).
    max_time_span
        Maximum seconds back from the current frame to include a past sweep.

    Returns
    -------
    np.ndarray
        (N, 5) float32 array: x, y, z, intensity, time_lag (current sensor frame).
    """
    # Current sweep at time-lag 0.
    cur = frame.lidar.astype(np.float32)
    sweeps: list[np.ndarray] = [
        np.hstack([cur[:, :4], np.zeros((cur.shape[0], 1), dtype=np.float32)])
    ]

    if history is not None and frame.nuscenes_ego_pose is not None:
        cur_inv = np.linalg.inv(frame.nuscenes_ego_pose.astype(np.float64))
        # Iterate newest-first; history deques are oldest-first.
        for f in reversed(list(history)):
            if len(sweeps) >= max_sweeps:
                break
            dt = frame.timestamp - f.timestamp
            if dt > max_time_span:
                break
            if f.nuscenes_ego_pose is None:
                logger.warning(
                    "accumulate_sweeps: past frame %s missing ego pose — skipping",
                    f.frame_id,
                )
                continue
            T = cur_inv @ f.nuscenes_ego_pose.astype(np.float64)
            pts = f.lidar.astype(np.float64)
            xyz = (T[:3, :3] @ pts[:, :3].T + T[:3, 3:4]).T
            intensity = pts[:, 3:4]
            tcol = np.full((pts.shape[0], 1), dt, dtype=np.float64)
            sweeps.append(np.hstack([xyz, intensity, tcol]).astype(np.float32))

    return np.vstack(sweeps).astype(np.float32)
