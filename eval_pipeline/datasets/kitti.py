"""
datasets/kitti.py
-----------------
KITTI Object Detection split dataset loader.

Wraps the parsing utilities in kitti_utils.py and converts
them into the pipeline's Frame / ObjectLabel / Calibration dataclasses.
"""

from __future__ import annotations

import logging
import pathlib
import sys
from typing import Iterator

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path so kitti_utils can be imported
# ---------------------------------------------------------------------------
_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from ..utils.kitti_utils import (  # noqa: E402
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


def _get_image_shape(img_file: pathlib.Path) -> tuple[int, int] | None:
    """Read (H, W) from a KITTI image file, or None if it doesn't exist."""
    if not img_file.exists():
        return None
    from PIL import Image
    with Image.open(img_file) as img:
        w, h = img.size
    return (h, w)


def _load_labels_and_calib(
    label_file: pathlib.Path,
    calib_file: pathlib.Path,
    img_file: pathlib.Path | None = None,
) -> tuple[list[ObjectLabel], Calibration]:
    """Parse a KITTI label file and calibration file into pipeline types."""
    kitti_objects = _parse_label_file(str(label_file))
    R0_rect, Tr_velo_to_cam, P2 = _parse_calib_file(str(calib_file))

    image_shape = _get_image_shape(img_file) if img_file is not None else None
    calib = Calibration(
        R0_rect=R0_rect, Tr_velo_to_cam=Tr_velo_to_cam,
        P2=P2, image_shape=image_shape,
    )

    labels: list[ObjectLabel] = []
    for obj in kitti_objects:
        corners_cam = _compute_3d_bbox_corners_cam(obj)
        corners_velo = _cam_corners_to_velo(corners_cam, R0_rect, Tr_velo_to_cam)
        labels.append(_kitti_object_to_label(obj, corners_velo))

    return labels, calib


def _load_velodyne(lidar_file: pathlib.Path) -> np.ndarray:
    """Load a KITTI .bin file as a raw (N, 4) float32 numpy array."""
    return np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 4)


def _filter_fov(lidar: np.ndarray, calib: Calibration) -> np.ndarray:
    """Keep only LiDAR points visible in the camera FOV.

    Replicates OpenPCDet's FOV_POINTS_ONLY filtering (kitti_dataset.yaml).
    KITTI only labels objects within the cam2 image, so points outside the
    FOV would produce false positives and dilute point-based sampling.
    """
    P2 = calib.P2
    if P2 is None or calib.image_shape is None:
        return lidar

    R0 = calib.R0_rect          # (3, 3)
    Tr = calib.Tr_velo_to_cam   # (3, 4)

    # velodyne → rectified camera coordinates
    pts = lidar[:, :3]
    N = pts.shape[0]
    pts_h = np.hstack([pts, np.ones((N, 1), dtype=pts.dtype)])  # (N, 4)
    pts_cam = (Tr @ pts_h.T).T            # (N, 3)
    pts_rect = (R0 @ pts_cam.T).T         # (N, 3)

    # rectified camera → image pixel coordinates
    pts_rect_h = np.hstack([pts_rect, np.ones((N, 1), dtype=pts.dtype)])
    pts_img = (P2 @ pts_rect_h.T).T       # (N, 3)
    depth = pts_img[:, 2]
    pts_img[:, 0] /= depth
    pts_img[:, 1] /= depth

    img_h, img_w = calib.image_shape
    mask = (
        (depth > 0)
        & (pts_img[:, 0] >= 0) & (pts_img[:, 0] < img_w)
        & (pts_img[:, 1] >= 0) & (pts_img[:, 1] < img_h)
    )
    return lidar[mask]


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
        self._img_dir   = self.root / "data_object_image_2" / "training" / "image_2"

        for name, path in [("label", self._label_dir), ("calib", self._calib_dir),
                           ("velodyne", self._velo_dir)]:
            if not path.exists():
                raise FileNotFoundError(
                    f"Required {name} directory not found: {path}"
                )

        if not self._img_dir.exists():
            logger.warning(
                "Image directory not found: %s — FOV filtering will be "
                "disabled. Point-based detectors (e.g. PointRCNN) may "
                "produce degraded results.", self._img_dir,
            )

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
        img_file   = self._img_dir   / f"{frame_id}.png"

        labels, calib = _load_labels_and_calib(label_file, calib_file, img_file)
        lidar = _filter_fov(_load_velodyne(lidar_file), calib)

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
