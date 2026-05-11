"""
defenses/carlo/defense.py
--------------------------
CARLO — oCclusion-Aware hieRarchy anomaLy detectiOn.

Reference
---------
Sun, J., Cao, Y., Chen, Q. A., & Mao, Z. M. (2020).
Towards Robust LiDAR-based Perception in Autonomous Driving: General
Black-box Adversarial Sensor Attack and Countermeasures.
arXiv:2006.16974v1 [cs.CR], §7.

Algorithm (Algorithm 1, Appendix C)
-------------------------------------
For each detected bounding box B:
  1. Build the ray-frustum F_B: the set of LiDAR rays that hit B or pass through it.
  2. Compute the LPD ratio g_B (Equation 6).
  3. If g_B < a′ − ε  → box is valid (skip FSD).
     If g_B > b′ + ε  → box is adversarial (skip FSD).
     Otherwise        → uncertain; escalate to FSD.
  4. In FSD: voxelise B at 0.25³ m³, mark free cells via 3D Bresenham traversal,
     compute f_B (Equation 5).  Adversarial if f_B ≥ (a + b) / 2.
Frame verdict: adversarial if ANY box is classified adversarial.
"""

from __future__ import annotations

import numpy as np

from ...base import BaseDefense
from ...types import DetectionResult, Frame, FrameHistory, Prediction
from .free_space import compute_fsd
from .lpd import compute_lpd


class CARLODefense(BaseDefense):
    """Detect LiDAR-spoofing attacks by verifying inter- and intra-occlusion
    patterns in each detector-predicted bounding box.

    CARLO operates entirely in the velodyne sensor frame with the sensor at the
    origin, consistent with the KITTI coordinate convention.

    # Assumption 1 (carlo_replication_spec.md §13): All geometry is computed in
    # the velodyne frame.  The sensor origin is (0, 0, 0) per §7.1.1 of the paper.

    Parameters
    ----------
    a : float
        FSD lower bound on f′ (spoofed vehicle distribution).
        Together with *b*, the decision threshold is (a + b) / 2.
    b : float
        FSD upper bound on f (valid vehicle distribution).
    a_prime : float
        LPD lower bound on g′ (spoofed).  Boxes with g < a′ − ε are valid.
    b_prime : float
        LPD upper bound on g (valid).  Boxes with g > b′ + ε are adversarial.
    epsilon : float
        LPD slack ε that widens both thresholds to account for detector noise
        (§7.1.2).  Boxes with g ∈ [a′ − ε, b′ + ε] are escalated to FSD.
    voxel_size : float
        Edge length (m) of each cubic FSD cell.  Paper specifies 0.25 m (§7.1.1).
    score_threshold : float
        Only inspect predictions with score ≥ this value.  Useful for ignoring
        very-low-confidence detections that the detector itself is uncertain about.

    # Assumption 3 (carlo_replication_spec.md §13): The exact values of a, b,
    # a′, b′, ε are NOT specified in the paper.  They must be calibrated on the
    # KITTI training set + 600 adversarial attack traces (§7.1.1, §9.2).
    # The defaults below are approximate starting points based on the paper's
    # Figures 10–11 descriptions; they are NOT authoritative and MUST be tuned.
    # TODO: run scripts/calibrate_carlo.py (to be written) to fit these values.
    """

    def __init__(
        self,
        a: float = 0.50,
        b: float = 0.30,
        a_prime: float = 0.10,
        b_prime: float = 0.05,
        epsilon: float = 0.05,
        voxel_size: float = 0.25,
        score_threshold: float = 0.0,
    ) -> None:
        self.a = float(a)
        self.b = float(b)
        self.a_prime = float(a_prime)
        self.b_prime = float(b_prime)
        self.epsilon = float(epsilon)
        self.voxel_size = float(voxel_size)
        self.score_threshold = float(score_threshold)

    @property
    def temporal_window(self) -> int:
        return 1  # stateless — operates on the current frame only

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, frame: Frame, history: FrameHistory) -> DetectionResult:
        """Run Algorithm 1 over all predictions in *frame* and return a frame-level verdict.

        A frame is flagged as attacked if at least one predicted bounding box
        is classified as adversarial by CARLO.

        # Assumption 4 (carlo_replication_spec.md §13): The LiDAR ray set L is
        # derived from the returned points in frame.lidar.  Each point represents
        # one laser ray that fired and returned.  Rays that fired but returned no
        # point (e.g. beyond max range) are absent from the computation.
        """
        pts_xyz = frame.lidar[:, :3]  # (N, 3) velodyne frame, sensor at origin

        preds = [
            p for p in frame.predictions if p.score >= self.score_threshold
        ]

        if not preds:
            return DetectionResult(
                is_attack_detected=False,
                confidence=0.0,
                metadata={"n_boxes": 0, "reason": "no_predictions"},
            )

        per_box: list[dict] = []
        for pred in preds:
            box_result = self._classify_box(pred, pts_xyz)
            per_box.append(box_result)

        attack_detected = any(r["verdict"] == "adversarial" for r in per_box)
        confidence = max(r["confidence"] for r in per_box)

        return DetectionResult(
            is_attack_detected=attack_detected,
            confidence=float(confidence),
            metadata={"n_boxes": len(preds), "per_box": per_box},
        )

    # ------------------------------------------------------------------
    # Per-box classification (Algorithm 1)
    # ------------------------------------------------------------------

    def _classify_box(self, pred: Prediction, pts_xyz: np.ndarray) -> dict:
        """Classify one bounding box as 'valid' or 'adversarial' via LPD → FSD.

        Returns a JSON-serialisable dict with the verdict, the stage that
        decided it, and diagnostic ratio values.
        """
        # --- Stage 1: LPD (Equation 6) ---
        g, lpd_details = compute_lpd(pred, pts_xyz)

        # Extract the frustum mask before stripping internal keys.
        frustum_mask: np.ndarray = lpd_details.pop("_frustum_mask")

        lpd_lo = self.a_prime - self.epsilon
        lpd_hi = self.b_prime + self.epsilon

        if g < lpd_lo:
            # Algorithm 1 line 9-10: definitely valid
            return {
                "verdict": "valid",
                "stage": "lpd",
                "g": g,
                "confidence": float(lpd_lo - g),
                **lpd_details,
            }

        if g > lpd_hi:
            # Algorithm 1 line 11-12: definitely adversarial
            return {
                "verdict": "adversarial",
                "stage": "lpd",
                "g": g,
                "confidence": float(g - lpd_hi),
                **lpd_details,
            }

        # --- Stage 2: FSD fallback (Equation 5) ---
        # Algorithm 1 lines 13-21: uncertain — escalate to free-space check.
        frustum_pts = pts_xyz[frustum_mask]
        f, fsd_details = compute_fsd(pred, frustum_pts, self.voxel_size)

        fsd_threshold = (self.a + self.b) / 2.0
        if f >= fsd_threshold:
            verdict = "adversarial"
        else:
            verdict = "valid"

        return {
            "verdict": verdict,
            "stage": "fsd",
            "g": g,
            "f": f,
            "fsd_threshold": fsd_threshold,
            "confidence": float(abs(f - fsd_threshold)),
            **lpd_details,
            **fsd_details,
        }
