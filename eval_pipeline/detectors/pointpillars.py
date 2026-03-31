"""
detectors/pointpillars.py
-------------------------
PointPillars detector stub.

To activate, install either mmdetection3d or OpenPCDet and implement
the _load_model() and _run_inference() methods.
"""

from __future__ import annotations

import numpy as np

from ..base import BaseDetector
from ..types import Frame, Prediction


class PointPillarsDetector(BaseDetector):
    """Wraps a PointPillars model for 3D object detection.

    Parameters
    ----------
    config_path
        Path to the model configuration file (mmdetection3d or OpenPCDet format).
    checkpoint_path
        Path to trained model weights.
    score_threshold
        Minimum detection confidence to include in results.
    device
        PyTorch device string, e.g. "cuda:0" or "cpu".

    Usage (once backend is installed)
    ----------------------------------
    detector = PointPillarsDetector(
        config_path="configs/pointpillars_kitti.py",
        checkpoint_path="checkpoints/pointpillars_kitti.pth",
        score_threshold=0.3,
        device="cuda:0",
    )
    predictions = detector.predict(frame)
    """

    def __init__(
        self,
        config_path: str,
        checkpoint_path: str,
        score_threshold: float = 0.3,
        device: str = "cpu",
    ) -> None:
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.score_threshold = score_threshold
        self.device = device
        self._model = self._load_model()

    def _load_model(self):
        """Load the detector model.

        TODO: Implement using mmdetection3d:
            from mmdet3d.apis import init_model
            return init_model(self.config_path, self.checkpoint_path, device=self.device)

        Or OpenPCDet:
            from pcdet.config import cfg, cfg_from_yaml_file
            from pcdet.models import build_network, load_data_to_gpu
            cfg_from_yaml_file(self.config_path, cfg)
            model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=None)
            model.load_params_from_file(filename=self.checkpoint_path, to_cpu=self.device=="cpu")
            return model
        """
        raise NotImplementedError(
            "PointPillarsDetector requires mmdetection3d or OpenPCDet. "
            "Install one and implement _load_model() and _run_inference()."
        )

    def _run_inference(self, lidar: np.ndarray) -> list[dict]:
        """Run inference on a (N, 4) lidar array.

        TODO: Convert lidar to backend format, run forward pass, return raw results.

        Expected return format — list of dicts with keys:
            type, score, x, y, z, height, width, length, rotation_y
        """
        raise NotImplementedError

    def predict(self, frame: Frame) -> list[Prediction]:
        """Run PointPillars on the frame and return filtered predictions."""
        raw = self._run_inference(frame.lidar)
        predictions: list[Prediction] = []

        for r in raw:
            if r["score"] < self.score_threshold:
                continue
            corners = self._box_to_corners(
                r["x"], r["y"], r["z"],
                r["height"], r["width"], r["length"],
                r["rotation_y"],
            )
            predictions.append(Prediction(
                type=r["type"],
                score=r["score"],
                x=r["x"], y=r["y"], z=r["z"],
                height=r["height"], width=r["width"], length=r["length"],
                rotation_y=r["rotation_y"],
                corners_velo=corners,
            ))

        return predictions

    @staticmethod
    def _box_to_corners(
        x: float, y: float, z: float,
        h: float, w: float, l: float,
        ry: float,
    ) -> np.ndarray:
        """Compute (8, 3) corner array from box parameters in velodyne frame.

        TODO: For oriented (non-axis-aligned) boxes this needs the full
        rotation matrix.  Currently returns an AABB approximation.
        """
        half_l, half_w, half_h = l / 2, w / 2, h / 2
        corners = np.array([
            [x - half_l, y - half_w, z - half_h],
            [x + half_l, y - half_w, z - half_h],
            [x + half_l, y + half_w, z - half_h],
            [x - half_l, y + half_w, z - half_h],
            [x - half_l, y - half_w, z + half_h],
            [x + half_l, y - half_w, z + half_h],
            [x + half_l, y + half_w, z + half_h],
            [x - half_l, y + half_w, z + half_h],
        ], dtype=np.float32)
        return corners
