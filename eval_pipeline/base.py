"""
base.py
-------
Abstract base classes for attacks, detectors, and defenses.
All concrete implementations must subclass these.
"""

from __future__ import annotations

import abc
from collections import deque
from collections.abc import Iterable

import numpy as np

from .types import DetectionResult, Frame, FrameHistory, Prediction


class BaseAttack(abc.ABC):
    """Interface for adversarial perturbations applied to sensor data.

    Implementations must be stateless with respect to individual frames
    (any randomness should use a seeded RNG stored on the instance).
    """

    @property
    @abc.abstractmethod
    def modality(self) -> str:
        """Sensor modality affected: 'lidar', 'camera', or 'fusion'."""
        ...

    @abc.abstractmethod
    def apply(self, frame: Frame) -> Frame:
        """Return a NEW attacked Frame. Must not mutate the input frame."""
        ...

    @property
    def attack_types(self) -> frozenset[str]:
        """LLM AttackType enum values this attack may produce (e.g. "GHOST_OBJECT").

        Used by llm_attack_type_accuracy to check whether the LLM correctly
        identified the attack category.  Return an empty frozenset if not
        applicable.
        """
        return frozenset()

    @property
    def name(self) -> str:
        return self.__class__.__name__


class BaseDetector(abc.ABC):
    """Interface for 3D object detectors."""

    @abc.abstractmethod
    def predict(
        self, frame: Frame, history: Iterable[Frame] | None = None
    ) -> list[Prediction]:
        """Run inference on a single frame and return predicted bounding boxes.

        ``history`` holds preceding frames (oldest-first) and is used by
        multi-sweep detectors to accumulate past sweeps into the current frame.
        Single-sweep detectors ignore it.
        """
        ...

    def predict_batch(self, frames: list[Frame]) -> list[list[Prediction]]:
        """Run inference on multiple frames.

        Override for GPU-batched backends; default falls back to per-frame predict().
        """
        return [self.predict(f) for f in frames]

    @property
    def num_sweeps(self) -> int:
        """Number of LiDAR sweeps (including current) the detector consumes.

        1 = single-sweep. Multi-sweep detectors override this so the pipeline
        sizes its frame history to supply enough past sweeps.
        """
        return 1

    @property
    def name(self) -> str:
        return self.__class__.__name__


class BaseDefense(abc.ABC):
    """Interface for attack-detection defenses.

    Defenses are framed as binary classifiers: given a (possibly attacked)
    frame and recent history, determine whether an attack is present.

    For stateless defenses set temporal_window = 1 and ignore history.
    For temporal defenses set temporal_window > 1; the pipeline guarantees
    the history deque will contain at most temporal_window - 1 prior frames.
    """

    @property
    def temporal_window(self) -> int:
        """Number of frames (including current) the defense may inspect.
        1 = stateless (current frame only).
        """
        return 1

    @property
    def async_detect(self) -> bool:
        """Whether the pipeline should run detect() calls concurrently.

        Override to True for defenses whose detect() blocks on an external
        API (e.g. LLM backends) so frames are processed in parallel.
        """
        return False

    @abc.abstractmethod
    def detect(self, frame: Frame, history: FrameHistory) -> DetectionResult:
        """Determine whether the frame has been attacked.

        Parameters
        ----------
        frame   : the current (possibly attacked) frame
        history : FrameHistory containing two deques, each with up to
                  (temporal_window - 1) preceding frames (oldest-first):
                    history.clean    — pre-attack frames as yielded by dataset
                    history.dirty — post-attack frames the vehicle
        """
        ...

    def reset(self) -> None:
        """Called by the pipeline at scene boundaries to clear per-scene state.

        Override in stateful defenses (e.g. to evict internal caches).
        The default is a no-op.
        """

    @property
    def name(self) -> str:
        return self.__class__.__name__
