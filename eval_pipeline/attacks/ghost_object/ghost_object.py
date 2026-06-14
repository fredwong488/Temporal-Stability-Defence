"""
attacks/ghost_object.py
-----------------------
3D-TC2 Ghost Object Attack — injects a pre-recorded ghost point-cloud pattern
into every frame to create a phantom object at the same position as recorded.

The ghost cloud is a (N, 4) float32 array [x, y, z, intensity] in the original
sensor frame coordinates.  It is produced by running:

    pixi run python tools/visualise_ghost_attack.py --attack-type <car|cyl|ped>

which writes ``ghost_cloud_<attack_type>.npy`` alongside the HTML viewer.
"""

from __future__ import annotations

import pathlib

import numpy as np
from scipy.spatial import cKDTree

from ...base import BaseAttack
from ...types import Frame
from ...utils.spoofing_noise import SpoofingNoiseModel


class GhostObjectAttack(BaseAttack):
    """Inject a pre-recorded ghost point-cloud pattern into LiDAR frames.

    The attack loads ghost points from *ghost_cloud_path* and appends them to
    every frame's lidar, creating a phantom object at the recorded sensor-frame
    position.  An optional :class:`SpoofingNoiseModel` perturbs the injected
    points to model realistic spoofing errors.

    Parameters
    ----------
    ghost_cloud_path
        Path to a ``.npy`` file containing a (N, 4) float32 array with columns
        x, y, z, intensity in sensor frame coordinates (as produced by
        ``tools/visualise_ghost_attack.py``).
    noise_model
        Optional spoofing noise model applied to the ghost points each frame.
        ``begin_frame()`` is called once per frame so δinter is re-sampled.
    """

    def __init__(
        self,
        ghost_cloud_path: str | pathlib.Path = "eval_pipeline/attacks/ghost_object/traces/",
        ghost_cloud_file: str = "ghost_cloud_car.npy",
        noise_model: SpoofingNoiseModel | None = None,
        occlusion_angular_tol: float = 0.015,
        debug: bool = False,
    ) -> None:
        path = pathlib.Path(ghost_cloud_path) / ghost_cloud_file
        if not path.exists():
            raise FileNotFoundError(
                f"Ghost cloud file not found: {path}\n"
                "Generate it with: pixi run python tools/visualise_ghost_attack.py"
            )
        raw = np.load(str(path)).astype(np.float32)
        if raw.ndim != 2 or raw.shape[1] != 4:
            raise ValueError(
                f"Expected (N, 4) array in {path}, got shape {raw.shape}"
            )
        self._ghost_pts = raw
        self._cloud_path = str(path)
        self.noise_model = noise_model
        self.occlusion_angular_tol = occlusion_angular_tol
        self.debug = debug

    @property
    def modality(self) -> str:
        return "lidar"

    @property
    def attack_types(self) -> frozenset[str]:
        return frozenset({"GHOST_OBJECT"})

    @property
    def name(self) -> str:
        return "GhostObjectAttack"

    def apply(self, frame: Frame) -> Frame:
        """Return a new Frame with the ghost object injected and occluded points removed."""
        if self.noise_model is not None:
            self.noise_model.begin_frame()

        ghost = self._ghost_pts.copy()
        if self.noise_model is not None:
            ghost = self.noise_model.apply(ghost)

        # Per-beam occlusion: remove any real point whose beam direction (az, el)
        # is within occlusion_angular_tol of a ghost point's beam direction.
        # This is tighter than a rectangular bounding box — ground returns at azimuths
        # or elevations not covered by an actual ghost beam are left intact.
        gx, gy, gz = ghost[:, 0], ghost[:, 1], ghost[:, 2]
        g_az = np.arctan2(gy, gx)
        g_el = np.arctan2(gz, np.sqrt(gx ** 2 + gy ** 2))

        lx, ly, lz = frame.lidar[:, 0], frame.lidar[:, 1], frame.lidar[:, 2]
        l_az = np.arctan2(ly, lx)
        l_el = np.arctan2(lz, np.sqrt(lx ** 2 + ly ** 2))

        tree = cKDTree(np.stack([g_az, g_el], axis=1))
        dists, _ = tree.query(np.stack([l_az, l_el], axis=1), k=1)
        occluded = dists < self.occlusion_angular_tol

        new_lidar = np.concatenate([frame.lidar[~occluded], ghost], axis=0)
        metadata: dict = {
            "attack": "GhostObject",
            "ghost_cloud_path": self._cloud_path,
            "n_injected": len(ghost),
            "n_occluded": int(occluded.sum()),
            "injected_centroid": ghost[:, :3].mean(axis=0).tolist(),
        }
        if self.debug:
            metadata["injected_xyz"] = ghost[:, :3].tolist()

        return frame.with_lidar(
            new_lidar,
            is_attacked=True,
            attacked_modalities=frozenset({"lidar"}),
            attack_metadata=metadata,
        )
