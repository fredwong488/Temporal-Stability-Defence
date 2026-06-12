"""LidarSwapAttack — replace a frame's LiDAR with a precomputed adversarial sweep.

The nuScenes dataset root is expected to contain:
    sweeps/LIDAR_TOP_ATTACK/LIDAR_TOP_ATTACK_{CLASS}/<basename>.pcd.bin

where each file has the same basename as the original sweep and is stored in the
standard nuScenes binary format: (N, 5) float32 — x, y, z, intensity, ring index.
"""

import pathlib

import numpy as np

from ..base import BaseAttack
from ..types import Frame

_ATTACK_SUBDIR = "sweeps/LIDAR_TOP_ATTACK"


class LidarSwapAttack(BaseAttack):
    """Replace a frame's LiDAR with the precomputed attack sweep for the given class.

    Parameters
    ----------
    attack_class:
        Object class variant to use, e.g. ``"car"``, ``"ped"``, ``"cyl"``.
        The value is upper-cased to form the directory name
        ``LIDAR_TOP_ATTACK_{CLASS}`` (so ``"car"`` → ``LIDAR_TOP_ATTACK_CAR``).
    debug:
        If True, print the resolved attack path on each apply call.
    """

    def __init__(self, attack_class: str, debug: bool = False) -> None:
        self._class = attack_class.upper()
        self._debug = debug

    @property
    def modality(self) -> str:
        return "lidar"

    def apply(self, frame: Frame) -> Frame:
        if frame.nuscenes_lidar_path is None:
            raise ValueError(
                f"LidarSwapAttack requires a nuScenes frame with a source lidar path; "
                f"frame '{frame.frame_id}' has none. "
                "Ensure the dataset is NuScenesDataset and the pipeline is up to date."
            )

        orig = pathlib.Path(frame.nuscenes_lidar_path)
        # orig is <root>/{samples,sweeps}/LIDAR_TOP/<basename> — two levels deep.
        root = orig.parents[2]
        attack_path = (
            root / _ATTACK_SUBDIR / f"LIDAR_TOP_ATTACK_{self._class}" / orig.name
        )

        if self._debug:
            print(f"[LidarSwapAttack] {orig.name} -> {attack_path}")

        if not attack_path.exists():
            raise FileNotFoundError(
                f"LidarSwapAttack: no attack sweep found for class '{self._class}'.\n"
                f"  Expected: {attack_path}"
            )

        # Same loading convention as _load_lidar in datasets/nuscenes.py:
        # nuScenes .pcd.bin is (N, 5) float32; drop the ring-index column.
        pts = np.fromfile(attack_path, dtype=np.float32).reshape(-1, 5)[:, :4]

        return frame.with_lidar(
            pts,
            is_attacked=True,
            attacked_modalities=frozenset({"lidar"}),
            attack_metadata={
                "attack": "LidarSwap",
                "attack_class": self._class,
                "source": str(attack_path),
                "n_points": int(len(pts)),
            },
        )
