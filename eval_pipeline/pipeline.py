"""
pipeline.py
-----------
EvalPipeline — orchestrates attack, detection, and defense over a dataset.
"""

from __future__ import annotations

import logging
from collections import deque

from tqdm import tqdm

from .base import BaseAttack, BaseDefense, BaseDetector
from .types import EvalResults, Frame, FrameResult, Prediction

logger = logging.getLogger(__name__)


class EvalPipeline:
    """Run adversarial evaluation over an iterable of Frame objects.

    Each frame passes through up to three optional stages:

    1. **Attack** — applies adversarial perturbations, returning a new Frame.
    2. **Detection** — runs a 3D object detector on both the clean and the
       attacked frame (clean predictions are cached by default).
    3. **Defense** — runs the attack detector on the (possibly attacked) frame
       together with a rolling history buffer.

    Parameters
    ----------
    dataset
        Any iterable of :class:`Frame` objects (e.g. :class:`KittiObjectDataset`).
    attack
        Optional attack to apply.  If None, frames are passed through unchanged
        and ``is_attacked`` will remain False throughout.
    detector
        Optional 3D object detector.  If None, prediction lists are empty.
    defense
        Optional attack-detection defense.  If None, ``defense_result`` is None
        on every FrameResult.
    cache_clean_preds
        Cache clean detector predictions keyed by ``frame_id``.  Saves compute
        when the same dataset is used across multiple experiments.
    """

    def __init__(
        self,
        dataset,                               # Iterable[Frame]
        attack: BaseAttack | None = None,
        detector: BaseDetector | None = None,
        defense: BaseDefense | None = None,
        cache_clean_preds: bool = True,
        desc: str = "Frames",
    ) -> None:
        self.dataset = dataset
        self.attack = attack
        self.detector = detector
        self.defense = defense
        self.cache_clean_preds = cache_clean_preds
        self.desc = desc
        self._clean_pred_cache: dict[str, list[Prediction]] = {}

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self) -> EvalResults:
        """Execute the pipeline over all frames and return aggregated results."""
        max_window = self.defense.temporal_window if self.defense else 1
        history: deque[Frame] = deque(maxlen=max_window)

        frame_results: list[FrameResult] = []
        n = 0

        for frame in tqdm(self.dataset, desc=self.desc, unit="frame"):
            n += 1
            logger.debug("Processing frame %s", frame.frame_id)

            # Stage 1: Clean detection (cached)
            clean_preds = self._get_clean_preds(frame)

            # Stage 2: Attack
            attacked_frame: Frame | None = None
            attacked_preds: list[Prediction] | None = None

            if self.attack is not None:
                attacked_frame = self.attack.apply(frame)
                if self.detector is not None:
                    attacked_preds = self.detector.predict(attacked_frame)

            # Stage 3: Defense — operates on what the vehicle actually received
            current_frame = attacked_frame if attacked_frame is not None else frame
            defense_result = None
            if self.defense is not None:
                defense_result = self.defense.detect(current_frame, history)

            # Update rolling history with the frame the vehicle received
            history.append(current_frame)

            frame_results.append(FrameResult(
                frame_id=frame.frame_id,
                labels=frame.labels,
                is_attacked=current_frame.is_attacked,
                clean_predictions=clean_preds,
                attacked_predictions=attacked_preds,
                defense_result=defense_result,
            ))

        logger.info("Pipeline complete: %d frames processed.", n)
        return EvalResults(frame_results=frame_results)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_clean_preds(self, frame: Frame) -> list[Prediction]:
        if self.detector is None:
            return []
        if self.cache_clean_preds and frame.frame_id in self._clean_pred_cache:
            return self._clean_pred_cache[frame.frame_id]
        preds = self.detector.predict(frame)
        if self.cache_clean_preds:
            self._clean_pred_cache[frame.frame_id] = preds
        return preds
