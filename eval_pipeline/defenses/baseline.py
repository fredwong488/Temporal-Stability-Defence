"""
defenses/baseline.py
--------------------
Random-chance baseline defense.

Chooses uniformly at random between attack detected and not detected.
Used as a sanity-check floor when evaluating other defenses.
"""

from __future__ import annotations

import numpy as np

from ..base import BaseDefense
from ..types import DetectionResult, Frame, FrameHistory


class BaselineDefense(BaseDefense):
    """Random-coin-flip detector used as an evaluation baseline.

    Parameters
    ----------
    seed
        Seed for the internal RNG. Defaults to 1 for reproducibility.
    """

    def __init__(self, seed: int = 1) -> None:
        self._rng = np.random.default_rng(seed)

    @property
    def temporal_window(self) -> int:
        return 1

    def detect(self, frame: Frame, history: FrameHistory) -> DetectionResult:
        detected = bool(self._rng.integers(0, 2))
        return DetectionResult(
            is_attack_detected=detected,
            confidence=0.5,
            metadata={"random_decision": detected},
        )
