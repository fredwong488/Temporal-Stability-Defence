"""
datasets/kitti.py
-----------------
KITTI Object Detection split dataset loader.

Wraps the parsing utilities in detection_v2_1.py (project root) and converts
them into the pipeline's Frame / ObjectLabel / Calibration dataclasses.
"""

from __future__ import annotations

import pathlib
import sys
from typing import Iterator

import numpy as np

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path so detection_v2_1 can be imported
# ---------------------------------------------------------------------------
_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from detection_v2_1 import (  # noqa: E402
    KittiObject,
    _cam_corners_to_velo,
    _compute_3d_bbox_corners_cam,
    _parse_calib_file,
    _parse_label_file,
)

from ..types import Calibration, Frame, ObjectLabel


# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------

def _kitti_object_to_label(obj: KittiObject, corners_velo: np.ndarray) -> ObjectLabel:
    return ObjectLabel(
        type=obj.type,
        truncated=obj.truncated,
        occluded=obj.occluded,
        alpha=obj.alpha,
        bbox_2d=(obj.bbox[0], obj.bbox[1], obj.bbox[2], obj.bbox[3]),
        height=obj.height,
        width=obj.width,
        length=obj.length,
        x=obj.x,
        y=obj.y,
        z=obj.z,
        rotation_y=obj.rotation_y,
        corners_velo=corners_velo,
    )


def _load_labels_and_calib(
    label_file: pathlib.Path,
    calib_file: pathlib.Path,
) -> tuple[list[ObjectLabel], Calibration]:
    """Parse a KITTI label file and calibration file into pipeline types."""
    kitti_objects = _parse_label_file(str(label_file))
    R0_rect, Tr_velo_to_cam = _parse_calib_file(str(calib_file))

    calib = Calibration(R0_rect=R0_rect, Tr_velo_to_cam=Tr_velo_to_cam)

    labels: list[ObjectLabel] = []
    for obj in kitti_objects:
        corners_cam = _compute_3d_bbox_corners_cam(obj)
        corners_velo = _cam_corners_to_velo(corners_cam, R0_rect, Tr_velo_to_cam)
        labels.append(_kitti_object_to_label(obj, corners_velo))

    return labels, calib


def _load_velodyne(lidar_file: pathlib.Path) -> np.ndarray:
    """Load a KITTI .bin file as a raw (N, 4) float32 numpy array."""
    return np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 4)


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------

class KittiObjectDataset:
    """KITTI Object Detection split — yields Frame objects in sorted order.

    Parameters
    ----------
    root
        Path to the KITTI root directory, expected layout::

            <root>/
                training_labels/label_2/    000000.txt ...
                data_object_calib/training/calib/  000000.txt ...
                data_object_velodyne/training/velodyne/  000000.bin ...

    frame_ids
        Optional list of zero-padded 6-digit frame IDs (e.g. ["000125"]).
        If None, all .bin files in the velodyne directory are used.
    sequence_id
        Sequence identifier stored on each Frame.  Defaults to "training".
    """

    def __init__(
        self,
        root: str | pathlib.Path,
        frame_ids: list[str] | None = None,
        sequence_id: str = "training",
    ) -> None:
        self.root = pathlib.Path(root)
        self.sequence_id = sequence_id

        self._label_dir = self.root / "training_labels" / "label_2"
        self._calib_dir = self.root / "data_object_calib" / "training" / "calib"
        self._velo_dir  = self.root / "data_object_velodyne" / "training" / "velodyne"

        if frame_ids is None:
            self.frame_ids: list[str] = sorted(
                p.stem for p in self._velo_dir.glob("*.bin")
            )
        else:
            self.frame_ids = list(frame_ids)

    # ------------------------------------------------------------------
    # Sequence protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.frame_ids)

    def __getitem__(self, idx: int) -> Frame:
        return self._load_frame(self.frame_ids[idx])

    def __iter__(self) -> Iterator[Frame]:
        for fid in self.frame_ids:
            yield self._load_frame(fid)

    # ------------------------------------------------------------------
    # Internal loading
    # ------------------------------------------------------------------

    def _load_frame(self, frame_id: str) -> Frame:
        """Load a single KITTI frame into the pipeline Frame dataclass."""
        label_file = self._label_dir / f"{frame_id}.txt"
        calib_file = self._calib_dir / f"{frame_id}.txt"
        lidar_file = self._velo_dir  / f"{frame_id}.bin"

        labels, calib = _load_labels_and_calib(label_file, calib_file)
        lidar = _load_velodyne(lidar_file)

        return Frame(
            frame_id=frame_id,
            sequence_id=self.sequence_id,
            timestamp=0.0,           # KITTI Object split has no timestamps
            lidar=lidar,
            image=None,              # camera images not loaded by default
            labels=labels,
            calib=calib,
        )

    # ------------------------------------------------------------------
    # Utility — reconstruct get_bbox_from_files from missing detection_v2_time
    # ------------------------------------------------------------------

    def get_bbox_corners(self, frame_id: str) -> dict[int, np.ndarray]:
        """Return {object_index: corners_velo (8, 3)} for a frame.

        Reconstructs the functionality of the missing detection_v2_time.get_bbox_from_files().
        """
        label_file = self._label_dir / f"{frame_id}.txt"
        calib_file = self._calib_dir / f"{frame_id}.txt"
        labels, _ = _load_labels_and_calib(label_file, calib_file)
        return {i: lbl.corners_velo for i, lbl in enumerate(labels)}
