"""
utils/spoofing_noise.py
-----------------------
Spoofing noise model for LiDAR point injection attacks.

Models the three error sources from:

    Sato et al. (2024). LiDAR Spoofing Meets the New-Gen: Capability Improvements, Broken Assumptions, and New Attack Strategies

The injection model (Eq. 1):

    PI(xij) = xij + (δrand_ij + δinner_ij + δinter) · g(xij)

where g(xij) = xij / ||xij||₂ is the unit radial direction (LiDAR at origin).

Error sources
-------------
δinner_ij ~ N(0, inner_frame_std)
    Per-point inaccuracy in signal bursting within a single frame (≈10 cm).

δinter ~ N(0, inter_frame_std)
    Per-frame scalar; same value applied to every injected point in the frame.
    Models inaccuracy in triggering timing across frames (≈35 cm).
    Must call begin_frame() once per attacked frame to re-sample.

δrand_ij ~ N(0, rand_std) or U(−rand_std·√3, rand_std·√3)
    Per-point timing-randomization error. Distribution and std are
    LiDAR-specific (Table V). "none" for first-gen LiDARs (VLP-16, VLP-32c).

injection_success_rate (R)
    Fraction of injected points that survive pulse fingerprinting / sync
    (Table IV). Points are uniformly downsampled before noise is applied.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class _LidarProfile:
    rand_distribution: Literal["none", "gaussian", "uniform"]
    rand_std: float               # metres
    injection_success_rate: float  # R ∈ (0, 1]
    inner_frame_std: float = 0.10  # metres
    inter_frame_std: float = 0.35  # metres


# Presets from Tables IV and V (Tu et al. 2023).
# rand_std values are the measured Std σ in metres.
# For uniform distributions, half-width = rand_std * √3 ≈ Max Δ in the table.
_PRESETS: dict[str, _LidarProfile] = {
    "worst_case":            _LidarProfile("none",     0.0,  1),
    "worst_case_high_error": _LidarProfile("none",     0.0,  1,    inner_frame_std=0.30, inter_frame_std=0.70),
    "vlp16":                 _LidarProfile("none",     0.0,  0.985),
    "vlp32c":                _LidarProfile("none",     0.0,  0.829),
    "os1_32":                _LidarProfile("uniform",  0.333, 0.438),
    "helios":                _LidarProfile("gaussian", 0.015, 0.194),
    "horizon":               _LidarProfile("uniform",  0.260, 0.799),
    "l515":                  _LidarProfile("gaussian", 0.075, 0.001),
    "xt32":                  _LidarProfile("none",      0.0, 0.021),
}


class SpoofingNoiseModel:
    """Point-injection noise model for LiDAR spoofing attacks.

    Parameters
    ----------
    rand_distribution
        Timing-randomization error distribution. Use "none" for first-gen
        LiDARs (VLP-16, VLP-32c) which have no timing randomization.
    rand_std
        Standard deviation of δrand_ij in metres. Ignored when
        rand_distribution is "none".
    inner_frame_std
        Standard deviation of per-point inner-frame error δinner_ij in metres.
        Default 0.10 m (10 cm) from paper measurements.
    inter_frame_std
        Standard deviation of per-frame inter-frame error δinter in metres.
        Default 0.35 m (35 cm) from paper measurements.
    injection_success_rate
        Fraction R ∈ (0, 1] of injected points that survive.  Models pulse
        fingerprinting / synchronisation success rate (Table IV).
    seed
        RNG seed for reproducibility.
    """

    def __init__(
        self,
        rand_distribution: Literal["none", "gaussian", "uniform"] = "none",
        rand_std: float = 0.0,
        inner_frame_std: float = 0.10,
        inter_frame_std: float = 0.35,
        injection_success_rate: float = 1.0,
        seed: int | None = None,
    ) -> None:
        self.rand_distribution = rand_distribution
        self.rand_std = rand_std
        self.inner_frame_std = inner_frame_std
        self.inter_frame_std = inter_frame_std
        self.injection_success_rate = injection_success_rate
        self._rng = np.random.default_rng(seed)
        self._inter_delta: float = 0.0

    @classmethod
    def from_preset(
        cls,
        lidar: str,
        seed: int | None = None,
        **overrides,
    ) -> SpoofingNoiseModel:
        """Construct from a named LiDAR preset.

        Available presets: vlp16, vlp32c, os1_32, helios, horizon, l515, xt32.

        Parameters
        ----------
        lidar
            Preset name (case-sensitive).
        seed
            RNG seed.
        **overrides
            Any constructor keyword argument to override the preset value.
        """
        if lidar not in _PRESETS:
            raise ValueError(
                f"Unknown LiDAR preset '{lidar}'. Available: {sorted(_PRESETS)}"
            )
        p = _PRESETS[lidar]
        params: dict = dict(
            rand_distribution=p.rand_distribution,
            rand_std=p.rand_std,
            injection_success_rate=p.injection_success_rate,
            inner_frame_std=p.inner_frame_std,
            inter_frame_std=p.inter_frame_std,
            seed=seed,
        )
        params.update(overrides)
        return cls(**params)

    # ------------------------------------------------------------------
    # Stateful frame boundary
    # ------------------------------------------------------------------

    def begin_frame(self) -> None:
        """Sample δinter for the upcoming frame.

        Must be called once per attacked frame, before any apply() calls for
        that frame.  This ensures all objects attacked within the same frame
        share the same inter-frame error, matching the physical model.
        """
        self._inter_delta = float(self._rng.normal(0.0, self.inter_frame_std))

    # ------------------------------------------------------------------
    # Core application
    # ------------------------------------------------------------------

    def apply(self, pts: np.ndarray) -> np.ndarray:
        """Apply injection noise to a batch of injected points.

        Parameters
        ----------
        pts
            (N, 4) float32 array [x, y, z, intensity] of candidate injected
            points prior to noise.

        Returns
        -------
        (N', 4) float32 array after downsampling by injection_success_rate
        and adding δrand + δinner + δinter along the radial direction.
        """
        if len(pts) == 0:
            return pts

        pts = self._downsample(pts)
        if len(pts) == 0:
            return pts

        xyz = pts[:, :3]
        n = len(xyz)

        # Unit radial direction: g(xij) = xij / ||xij||₂
        norms = np.linalg.norm(xyz, axis=1, keepdims=True)
        norms = np.where(norms == 0.0, 1.0, norms)
        g = xyz / norms                                     # (N, 3)

        delta_inner = self._rng.normal(0.0, self.inner_frame_std, size=(n, 1))
        delta_rand = self._sample_rand(n)                   # (N, 1)

        total = delta_rand + delta_inner + self._inter_delta  # (N, 1) broadcast
        new_xyz = xyz + total * g                           # (N, 3)

        return np.hstack([new_xyz, pts[:, 3:4]]).astype(np.float32)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _downsample(self, pts: np.ndarray) -> np.ndarray:
        if self.injection_success_rate >= 1.0:
            return pts
        n_keep = max(1, round(len(pts) * self.injection_success_rate))
        idx = self._rng.choice(len(pts), size=n_keep, replace=False)
        return pts[idx]

    def _sample_rand(self, n: int) -> np.ndarray:
        if self.rand_distribution == "none" or self.rand_std == 0.0:
            return np.zeros((n, 1))
        if self.rand_distribution == "gaussian":
            return self._rng.normal(0.0, self.rand_std, size=(n, 1))
        # uniform: U(−std·√3, +std·√3) gives std σ and Max Δ = std·√3
        half = self.rand_std * np.sqrt(3.0)
        return self._rng.uniform(-half, half, size=(n, 1))
