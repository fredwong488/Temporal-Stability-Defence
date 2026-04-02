"""
detectors/pointpillars.py
-------------------------
PointPillars detector backed by OpenPCDet.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch

from ..base import BaseDetector
from ..types import Frame, Prediction


class PointPillarsDetector(BaseDetector):
    """Wraps OpenPCDet's PointPillars model for 3D object detection.

    Parameters
    ----------
    config_path
        Path to the OpenPCDet YAML config (e.g. cfgs/kitti_models/pointpillar.yaml).
    checkpoint_path
        Path to trained model weights (.pth).
    score_threshold
        Minimum detection confidence to include in results.
    device
        PyTorch device string, e.g. "cuda:0" or "cpu".
    """

    def __init__(
        self,
        config_path: str,
        checkpoint_path: str,
        score_threshold: float = 0.3,
        device: str = "cuda:0",
    ) -> None:
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.score_threshold = score_threshold
        self.device = device
        self._model, self._dataset, self._class_names = self._load_model()

    def _load_model(self):
        from pcdet.config import cfg, cfg_from_yaml_file
        from pcdet.datasets.dataset import DatasetTemplate
        from pcdet.models import build_network

        logger = logging.getLogger(__name__)
        cfg_from_yaml_file(self.config_path, cfg)

        dataset = DatasetTemplate(
            dataset_cfg=cfg.DATA_CONFIG,
            class_names=cfg.CLASS_NAMES,
            training=False,
            root_path=Path(self.config_path).parent,
            logger=logger,
        )

        model = build_network(
            model_cfg=cfg.MODEL,
            num_class=len(cfg.CLASS_NAMES),
            dataset=dataset,
        )
        model.load_params_from_file(
            filename=self.checkpoint_path,
            logger=logger,
            to_cpu=True,
        )
        model.cuda().eval()
        return model, dataset, cfg.CLASS_NAMES

    def _run_inference(self, lidar: np.ndarray) -> list[dict]:
        from pcdet.models import load_data_to_gpu

        data_dict = self._dataset.prepare_data({"points": lidar, "frame_id": 0})
        data_dict = self._dataset.collate_batch([data_dict])
        load_data_to_gpu(data_dict)

        with torch.no_grad():
            pred_dicts, _ = self._model.forward(data_dict)

        boxes = pred_dicts[0]["pred_boxes"].cpu().numpy()   # (N, 7): x y z dx dy dz heading
        scores = pred_dicts[0]["pred_scores"].cpu().numpy()  # (N,)
        labels = pred_dicts[0]["pred_labels"].cpu().numpy()  # (N,) 1-indexed

        results = []
        for box, score, label in zip(boxes, scores, labels):
            results.append({
                "type": self._class_names[label - 1],
                "score": float(score),
                "x": float(box[0]),
                "y": float(box[1]),
                "z": float(box[2]),
                "length": float(box[3]),
                "width": float(box[4]),
                "height": float(box[5]),
                "rotation_y": float(box[6]),
            })
        return results

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
        """Compute (8, 3) corner array from oriented box parameters in velodyne frame."""
        half_l, half_w, half_h = l / 2, w / 2, h / 2
        # 8 corners centred at origin in box frame
        corners = np.array([
            [-half_l, -half_w, -half_h],
            [ half_l, -half_w, -half_h],
            [ half_l,  half_w, -half_h],
            [-half_l,  half_w, -half_h],
            [-half_l, -half_w,  half_h],
            [ half_l, -half_w,  half_h],
            [ half_l,  half_w,  half_h],
            [-half_l,  half_w,  half_h],
        ], dtype=np.float32)
        # Rotate around z-axis by ry, then translate to world position
        cos_ry, sin_ry = np.cos(ry), np.sin(ry)
        R = np.array([
            [cos_ry, -sin_ry, 0.0],
            [sin_ry,  cos_ry, 0.0],
            [0.0,     0.0,    1.0],
        ], dtype=np.float32)
        return (R @ corners.T).T + np.array([x, y, z], dtype=np.float32)
