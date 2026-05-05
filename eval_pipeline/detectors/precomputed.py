"""
detectors/precomputed.py
------------------------
PrecomputedDetector — loads bounding-box predictions from a pickle file and
replays them keyed by frame_id, avoiding the need to run a live 3D detector.

Supported pickle formats
------------------------
1. **Native eval_pipeline format** (produced by scripts/precompute_detections.py):
       dict[frame_id, list[eval_pipeline.types.Prediction]]
   All coordinates are already in the sensor/velodyne frame, so no transform
   is applied.  Pass ``is_global_frame=False`` (or omit it — it is detected
   automatically).

2. **3D-TC2 / NuScenes format** (list-of-lists or flat list of box objects):
   Each object must expose:
       .sample_token    : str
       .translation     : (3,) xyz in *global* NuScenes coordinates
       .size            : (3,) wlh in metres
       .rotation        : [w, x, y, z] quaternion
       .detection_name  : str  (e.g. "car")
       .detection_score : float
   Boxes are transformed from global → sensor frame using each Frame's
   nuscenes_ego_pose when ``is_global_frame=True`` (the default).

If your pickle uses a different schema, subclass and override _load_pickle.
"""

from __future__ import annotations

import logging
import pickle
from collections import defaultdict

import numpy as np

from ..base import BaseDetector
from ..types import Frame, Prediction

logger = logging.getLogger(__name__)


class PrecomputedDetector(BaseDetector):
    """Replay detector predictions from a pre-saved pickle.

    Parameters
    ----------
    pickle_path
        Path to the detection pickle.  See module docstring for the expected
        object schema.
    score_threshold
        Minimum detection score to include.
    is_global_frame
        When True (default for NuScenes pickles), box translations are in the
        global NuScenes coordinate frame and are transformed to the sensor
        frame using ``frame.nuscenes_ego_pose``.  Set to False if the pickle already
        stores boxes in the sensor/velodyne frame.
    """

    def __init__(
        self,
        pickle_path: str,
        score_threshold: float = 0.0,
        is_global_frame: bool = True,
    ) -> None:
        self.pickle_path = pickle_path
        self.score_threshold = score_threshold
        self.is_global_frame = is_global_frame
        self._native_format: bool = False  # set True by _load_pickle for eval_pipeline pickles
        self._index: dict[str, list] = self._load_pickle(pickle_path)
        logger.info(
            "PrecomputedDetector: loaded %d frame entries from %s (native=%s)",
            len(self._index), pickle_path, self._native_format,
        )

    # ------------------------------------------------------------------
    # BaseDetector contract
    # ------------------------------------------------------------------

    def predict(self, frame: Frame) -> list[Prediction]:
        boxes = self._index.get(frame.frame_id, [])
        if not boxes:
            boxes = self._find_by_prefix(frame.frame_id)

        # Native eval_pipeline format: values are already Prediction objects in
        # the sensor frame — return them directly after score filtering.
        if self._native_format:
            return [p for p in boxes if p.score >= self.score_threshold]

        # 3D-TC2 / NuScenes format: transform global box objects → sensor frame.
        ego_pose_inv = (
            np.linalg.inv(frame.nuscenes_ego_pose.astype(np.float64))
            if (self.is_global_frame and frame.nuscenes_ego_pose is not None)
            else None
        )

        predictions: list[Prediction] = []
        for box in boxes:
            score = getattr(box, "detection_score", getattr(box, "score", 1.0))
            if score < self.score_threshold:
                continue
            pred = self._box_to_prediction(box, ego_pose_inv)
            if pred is not None:
                predictions.append(pred)
        return predictions

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_pickle(self, path: str) -> dict[str, list]:
        with open(path, "rb") as f:
            raw = pickle.load(f)

        # Native eval_pipeline format produced by scripts/precompute_detections.py:
        #   dict[frame_id, list[Prediction]]
        if isinstance(raw, dict):
            sample_list = next((v for v in raw.values() if v), [])
            if not sample_list or isinstance(sample_list[0], Prediction):
                self._native_format = True
                return dict(raw)

        # 3D-TC2 / NuScenes format: list[list[Box]] or list[Box]
        index: dict[str, list] = defaultdict(list)
        if isinstance(raw, list) and len(raw) > 0 and isinstance(raw[0], list):
            flat = [box for cat in raw for box in cat]
        else:
            flat = raw

        for box in flat:
            token = getattr(box, "sample_token", None) or getattr(box, "sample_data_token", None)
            if token is None:
                logger.warning("PrecomputedDetector: box without sample_token — skipping")
                continue
            index[token].append(box)
        return dict(index)

    def _find_by_prefix(self, frame_id: str) -> list:
        for key, boxes in self._index.items():
            if key.startswith(frame_id) or frame_id.startswith(key[:len(frame_id)]):
                return boxes
        return []

    def _box_to_prediction(self, box, ego_pose_inv: np.ndarray | None) -> Prediction | None:
        try:
            from pyquaternion import Quaternion
            translation = np.array(box.translation, dtype=np.float64)
            w, l, h = box.size  # NuScenes wlh

            if ego_pose_inv is not None:
                # Global → sensor frame
                centre_h = np.append(translation, 1.0)
                centre_sensor = (ego_pose_inv @ centre_h)[:3]
                R_global = Quaternion(box.rotation).rotation_matrix
                R_sensor = ego_pose_inv[:3, :3] @ R_global
            else:
                centre_sensor = translation
                R_sensor = Quaternion(box.rotation).rotation_matrix

            x, y, z = float(centre_sensor[0]), float(centre_sensor[1]), float(centre_sensor[2])

            # Yaw in sensor frame
            fwd = R_sensor[:, 0]
            rotation_y = float(np.arctan2(fwd[1], fwd[0]))

            # 8 box corners in sensor frame
            half = np.array([l / 2, w / 2, h / 2])
            signs = np.array([
                [ 1,  1,  1], [ 1,  1, -1], [ 1, -1,  1], [ 1, -1, -1],
                [-1,  1,  1], [-1,  1, -1], [-1, -1,  1], [-1, -1, -1],
            ], dtype=np.float64)
            corners_sensor = (R_sensor @ (signs * half).T).T + centre_sensor

            det_name = getattr(box, "detection_name", getattr(box, "label", "unknown"))
            score = float(getattr(box, "detection_score", getattr(box, "score", 1.0)))

            return Prediction(
                type=det_name,
                score=score,
                x=x, y=y, z=z,
                height=float(h),
                width=float(w),
                length=float(l),
                rotation_y=rotation_y,
                corners_velo=corners_sensor.astype(np.float32),
            )
        except Exception as exc:
            logger.debug("PrecomputedDetector: failed to convert box: %s", exc)
            return None
