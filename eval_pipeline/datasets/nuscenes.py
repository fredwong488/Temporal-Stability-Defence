"""
datasets/nuscenes.py
--------------------
NuScenes dataset loader for the eval_pipeline.

Yields Frame objects in scene-temporal order at the LiDAR's native 10 Hz cadence
(or 2 Hz keyframes-only mode). Each Frame carries:
  - lidar  : (N, 4) xyz+intensity in the current sensor frame
  - ego_pose : (4, 4) sensor-to-global transform at this sweep's timestamp
  - timestamp: float seconds
  - sequence_id: NuScenes scene token (resets pipeline history at scene boundaries)
  - labels : ObjectLabel list populated for keyframes only; empty for inter-sweeps

Temporal defenses can reconstruct past-sweep stacks from the pipeline's
clean/dirty history deques using each frame's ego_pose.
"""

from __future__ import annotations

import logging
import pathlib
from typing import Iterator

import numpy as np

from ..types import Calibration, Frame, ObjectLabel

logger = logging.getLogger(__name__)

# NuScenes category name → short detection name mapping (mirrors nusc-devkit eval)
_NUSCENES_DETECTION_NAMES: dict[str, str] = {
    "vehicle.car": "car",
    "vehicle.truck": "truck",
    "vehicle.bus.bendy": "bus",
    "vehicle.bus.rigid": "bus",
    "vehicle.motorcycle": "motorcycle",
    "vehicle.bicycle": "bicycle",
    "vehicle.trailer": "trailer",
    "vehicle.construction": "construction_vehicle",
    "human.pedestrian.adult": "pedestrian",
    "human.pedestrian.child": "pedestrian",
    "human.pedestrian.wheelchair": "pedestrian",
    "human.pedestrian.stroller": "pedestrian",
    "human.pedestrian.personal_mobility": "pedestrian",
    "human.pedestrian.police_officer": "pedestrian",
    "human.pedestrian.construction_worker": "pedestrian",
    "movable_object.barrier": "barrier",
    "movable_object.trafficcone": "traffic_cone",
    "movable_object.pushable_pullable": "traffic_cone",
    "movable_object.debris": "traffic_cone",
    "static_object.bicycle_rack": None,  # ignored
}


def _make_transform(translation: list[float], rotation_wxyz: list[float]) -> np.ndarray:
    """Build a 4×4 transform matrix from translation + quaternion (w, x, y, z)."""
    from pyquaternion import Quaternion
    q = Quaternion(rotation_wxyz)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = q.rotation_matrix
    T[:3, 3] = translation
    return T


def _sensor_to_global(nusc, sample_data: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return (sensor-to-global, sensor-to-ego) 4×4 transforms for the given sample_data record."""
    ep = nusc.get("ego_pose", sample_data["ego_pose_token"])
    cs = nusc.get("calibrated_sensor", sample_data["calibrated_sensor_token"])
    car_from_sensor = _make_transform(cs["translation"], cs["rotation"])
    global_from_car = _make_transform(ep["translation"], ep["rotation"])
    return global_from_car @ car_from_sensor, car_from_sensor


def _load_lidar(nusc_dataroot: pathlib.Path, sample_data: dict) -> np.ndarray:
    """Load a NuScenes LiDAR sweep as (N, 4) float32 (x, y, z, intensity)."""
    path = nusc_dataroot / sample_data["filename"]
    pts = np.fromfile(path, dtype=np.float32).reshape(-1, 5)
    return pts[:, :4]  # drop ring index


def _annotation_to_label(nusc, ann_token: str, ego_pose: np.ndarray) -> ObjectLabel | None:
    """Convert a NuScenes annotation to ObjectLabel in the sensor frame."""
    ann = nusc.get("sample_annotation", ann_token)
    category = ann["category_name"]
    det_name = _NUSCENES_DETECTION_NAMES.get(category)
    if det_name is None:
        return None

    from pyquaternion import Quaternion

    # Box centre (global) → sensor frame
    centre_global = np.array(ann["translation"] + [1.0], dtype=np.float64)
    global_to_sensor = np.linalg.inv(ego_pose)
    centre_sensor = (global_to_sensor @ centre_global)[:3]

    w, l, h = ann["size"]  # NuScenes wlh order

    # Compute 8 corners in sensor frame
    q_global = Quaternion(ann["rotation"])
    # Transform rotation: sensor_R = global_to_sensor_R @ global_R
    R_global = q_global.rotation_matrix
    R_sensor = global_to_sensor[:3, :3] @ R_global
    half = np.array([l / 2, w / 2, h / 2])
    # Standard box corners in local frame
    signs = np.array([
        [ 1,  1,  1], [ 1,  1, -1], [ 1, -1,  1], [ 1, -1, -1],
        [-1,  1,  1], [-1,  1, -1], [-1, -1,  1], [-1, -1, -1],
    ], dtype=np.float64)
    corners_local = signs * half  # (8, 3)
    corners_sensor = (R_sensor @ corners_local.T).T + centre_sensor  # (8, 3)

    # Heading angle (yaw) in sensor frame — approximate via rotation matrix
    fwd_global = R_global[:, 0]  # forward direction in global
    fwd_sensor = global_to_sensor[:3, :3] @ fwd_global
    fwd_sensor[2] = 0.0  # project onto ground plane before computing yaw
    rotation_y = float(np.arctan2(fwd_sensor[1], fwd_sensor[0]))

    return ObjectLabel(
        type=det_name,
        truncated=None,
        occluded=None,
        alpha=None,
        bbox_2d=None,
        height=float(h),
        width=float(w),
        length=float(l),
        x=float(centre_sensor[0]),
        y=float(centre_sensor[1]),
        z=float(centre_sensor[2]),
        rotation_y=rotation_y,
        corners_velo=corners_sensor.astype(np.float32),
    )


class NuScenesDataset:
    """NuScenes dataset loader — yields Frame objects in scene-temporal order.

    Parameters
    ----------
    root
        Path to the NuScenes dataset root (contains ``samples/``, ``sweeps/``,
        ``v1.0-mini/`` or ``v1.0-trainval/`` etc.).
    version
        Dataset version string, e.g. ``"v1.0-mini"`` or ``"v1.0-trainval"``.
    split
        Scene split to load. For mini: ``"mini_train"`` or ``"mini_val"``.
        For trainval: ``"train"`` or ``"val"``.  Pass ``None`` to load all scenes.
    scene_names
        Optional explicit list of scene names (overrides split).
    lidar_channel
        Sensor channel name for LiDAR data.
    keyframes_only
        When True, yield only annotated keyframes (2 Hz).  When False (default),
        yield every LiDAR sweep at the native 10 Hz cadence, which is required
        for temporal defenses that need densely-sampled history.
    frame_ids
        Optional allowlist of frame IDs to yield.  Frames are returned in
        dataset-natural order filtered to this set.  No LiDAR is loaded during
        filtering — only the token strings are compared.
    verbose
        Whether the NuScenes devkit should print loading progress.
    """

    def __init__(
        self,
        root: str | pathlib.Path,
        version: str = "v1.0-mini",
        split: str | None = "mini_val",
        scene_names: list[str] | None = None,
        lidar_channel: str = "LIDAR_TOP",
        keyframes_only: bool = False,
        frame_ids: list[str] | None = None,
        verbose: bool = False,
    ) -> None:
        try:
            from nuscenes.nuscenes import NuScenes
        except ImportError as e:
            raise ImportError(
                "nuscenes-devkit is required for NuScenesDataset. "
                "Add 'nuscenes-devkit' to pixi.toml and run 'pixi install'."
            ) from e

        self.root = pathlib.Path(root)
        self.version = version
        self.lidar_channel = lidar_channel
        self.keyframes_only = keyframes_only

        self._nusc = NuScenes(version=version, dataroot=str(self.root), verbose=verbose)

        # Resolve which scenes to iterate
        if scene_names is not None:
            self._scene_names = set(scene_names)
        elif split is not None:
            from nuscenes.utils.splits import create_splits_scenes
            all_splits = create_splits_scenes()
            if split not in all_splits:
                raise ValueError(
                    f"Unknown split '{split}'. Available: {list(all_splits)}"
                )
            self._scene_names = set(all_splits[split])
        else:
            self._scene_names = None  # all scenes

        # Build flat ordered list of (scene_token, sample_data_token) pairs
        self._entries: list[tuple[str, str]] = self._build_entries()
        if frame_ids is not None:
            wanted = set(frame_ids)
            self._entries = [(st, sdt) for st, sdt in self._entries if sdt[:16] in wanted]
        logger.info(
            "NuScenesDataset: %d frames from %s (%s)",
            len(self._entries), version, split or "all scenes",
        )

    def _build_entries(self) -> list[tuple[str, str]]:
        entries: list[tuple[str, str]] = []
        for scene in self._nusc.scene:
            if self._scene_names is not None and scene["name"] not in self._scene_names:
                continue
            scene_token = scene["token"]
            # Walk sample_data chain for the LiDAR channel
            first_sample = self._nusc.get("sample", scene["first_sample_token"])
            sd_token = first_sample["data"][self.lidar_channel]
            # Rewind to the first sweep in the scene (may precede first keyframe)
            sd = self._nusc.get("sample_data", sd_token)
            while sd["prev"] != "":
                sd = self._nusc.get("sample_data", sd["prev"])
            # Walk forward
            while True:
                if not self.keyframes_only or sd["is_key_frame"]:
                    entries.append((scene_token, sd["token"]))
                if sd["next"] == "":
                    break
                sd = self._nusc.get("sample_data", sd["next"])
        return entries

    # ------------------------------------------------------------------
    # Sequence protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._entries)

    def __getitem__(self, idx: int) -> Frame:
        scene_token, sd_token = self._entries[idx]
        return self._load_frame(scene_token, sd_token)

    def __iter__(self) -> Iterator[Frame]:
        for scene_token, sd_token in self._entries:
            yield self._load_frame(scene_token, sd_token)

    def scene_lengths(self) -> dict[str, int]:
        """Return mapping from sequence_id (scene token) to frame count.

        Reads from the pre-built _entries list — no lidar is loaded.
        """
        counts: dict[str, int] = {}
        for scene_token, _ in self._entries:
            counts[scene_token] = counts.get(scene_token, 0) + 1
        return counts

    # ------------------------------------------------------------------
    # Internal loading
    # ------------------------------------------------------------------

    def _load_frame(self, scene_token: str, sd_token: str) -> Frame:
        sd = self._nusc.get("sample_data", sd_token)
        ego_pose, sensor_to_ego = _sensor_to_global(self._nusc, sd)
        lidar = _load_lidar(self.root, sd)
        timestamp = sd["timestamp"] / 1e6  # microseconds → seconds

        # Derive a deterministic frame_id from the token (truncated for readability)
        frame_id = sd_token[:16]

        # Labels only at keyframes; empty otherwise
        labels: list[ObjectLabel] = []
        if sd["is_key_frame"]:
            sample = self._nusc.get("sample", sd["sample_token"])
            for ann_token in sample["anns"]:
                label = _annotation_to_label(self._nusc, ann_token, ego_pose)
                if label is not None:
                    labels.append(label)

        return Frame(
            frame_id=frame_id,
            sequence_id=scene_token,
            timestamp=timestamp,
            lidar=lidar,
            image=None,
            labels=labels,
            kitti_calib=None,  # NuScenes has no KITTI-style calibration matrices
            nuscenes_ego_pose=ego_pose,
            nuscenes_sensor_to_ego=sensor_to_ego,
        )
