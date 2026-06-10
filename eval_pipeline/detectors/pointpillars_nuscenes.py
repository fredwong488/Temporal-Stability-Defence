"""
detectors/pointpillars_nuscenes.py
-----------------------------------
PointPillars multihead detector for NuScenes, backed by OpenPCDet.

The NuScenes model expects 5-channel points (x, y, z, intensity, timestamp).
Since our dataset loader provides 4-channel sweeps (ring index already dropped),
this subclass pads a zero timestamp column before inference.

Output boxes are 9D (x y z dx dy dz heading vx vy); velocity is ignored.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from ..types import Frame
from ._sweep_accumulation import accumulate_sweeps
from .pointpillars import PointPillarsDetector


class PointPillarsNuScenesDetector(PointPillarsDetector):
    """PointPillars multihead detector trained on NuScenes (cbgs_pp_multihead).

    Parameters
    ----------
    config_path
        Path to the OpenPCDet YAML config for the NuScenes multihead model.
    checkpoint_path
        Path to trained model weights (.pth).
    score_threshold
        Minimum detection confidence to include in results.
    device
        PyTorch device string.
    max_sweeps
        Maximum total sweeps (current + past) accumulated per inference.
        The model is trained with 10 (MAX_SWEEPS: 10).
    max_time_span
        Maximum seconds back from the current frame to include a past sweep.
        Bounds accumulation to the ~0.5 s training span regardless of cadence.
    """

    def __init__(
        self,
        config_path: str = "OpenPCDet/tools/cfgs/nuscenes_models/cbgs_pp_multihead.yaml",
        checkpoint_path: str = "models/openpcdet/pp_multihead_nds5823_updated.pth",
        score_threshold: float = 0.3,
        device: str = "cuda:0",
        max_sweeps: int = 10,
        max_time_span: float = 0.5,
    ) -> None:
        super().__init__(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            score_threshold=score_threshold,
            device=device,
        )
        self.max_sweeps = max_sweeps
        self.max_time_span = max_time_span

    @property
    def num_sweeps(self) -> int:
        return self.max_sweeps

    def _prepare_points(
        self, frame: Frame, history: Iterable[Frame] | None = None
    ) -> np.ndarray:
        # Accumulate up to max_sweeps past sweeps into the current sensor frame,
        # producing (N, 5) points: x, y, z, intensity, time_lag.
        return accumulate_sweeps(
            frame, history, self.max_sweeps, self.max_time_span
        )

    def _run_inference(self, lidar: np.ndarray) -> list[dict]:
        # Model expects 5 features (x, y, z, intensity, timestamp).
        # Pad a zero timestamp column onto the 4-channel sweep points.
        if lidar.shape[1] == 4:
            lidar = np.hstack([lidar, np.zeros((lidar.shape[0], 1), dtype=lidar.dtype)])
        return super()._run_inference(lidar)
