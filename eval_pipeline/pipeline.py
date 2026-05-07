"""
pipeline.py
-----------
EvalPipeline — orchestrates attack, detection, and defense over a dataset.
"""

from __future__ import annotations

import dataclasses
import logging
import pathlib
import pickle
from collections import deque

import numpy as np
from tqdm import tqdm

from .base import BaseAttack, BaseDefense, BaseDetector
from .types import EvalResults, Frame, FrameCacheEntry, FrameHistory, FrameResult, Prediction

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
    use_cached_attacks
        Only meaningful when a precomputed cache is loaded.  If False (default),
        the attack is re-applied live for each cache-flagged frame and the
        detector is re-run on the new lidar; the fresh result is not written back
        to the cache.  If True, cached ``attacked_predictions`` and
        ``attack_metadata`` are used directly and the attack is not re-applied —
        guarantees consistency between predictions and metadata.
    """

    def __init__(
        self,
        dataset,                               # Iterable[Frame]
        attack: BaseAttack | None = None,
        detector: BaseDetector | None = None,
        defense: BaseDefense | None = None,
        cache_clean_preds: bool = True,
        attack_fraction: float = 1.0,
        attack_fraction_seed: int = 0,
        desc: str = "Frames",
        precomputed_cache_path: str | None = None,
        use_cached_attacks: bool = False,
    ) -> None:
        self.dataset = dataset
        self.attack = attack
        self.detector = detector
        self.defense = defense
        self.cache_clean_preds = cache_clean_preds
        self.attack_fraction = attack_fraction
        self._attack_rng = np.random.default_rng(attack_fraction_seed)
        self.use_cached_attacks = use_cached_attacks
        self.desc = desc
        self._clean_pred_cache: dict[str, list[Prediction]] = {}

        # Precomputed cache: load if the file exists; save after run() if it doesn't.
        self._precomputed_cache: dict[str, FrameCacheEntry] | None = None
        self._precomputed_save_path: str | None = None
        if precomputed_cache_path is not None:
            p = pathlib.Path(precomputed_cache_path)
            if p.exists():
                with open(p, "rb") as f:
                    self._precomputed_cache = pickle.load(f)
                logger.info(
                    "Loaded precomputed cache: %d frame entries from %s",
                    len(self._precomputed_cache), p,
                )
            else:
                self._precomputed_save_path = str(p)
                logger.info("Will save precomputed cache to: %s", p)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self) -> EvalResults:
        """Execute the pipeline over all frames and return aggregated results."""
        max_window = self.defense.temporal_window if self.defense else 1
        # Two parallel histories: clean (pre-attack) and dirty (post-attack).
        # Sized to temporal_window - 1 so the defense sees prior frames only.
        clean_history: deque[Frame] = deque(maxlen=max(0, max_window - 1))
        dirty_history: deque[Frame] = deque(maxlen=max(0, max_window - 1))
        last_sequence_id: str | None = None

        frame_results: list[FrameResult] = []
        accumulator: dict[str, FrameCacheEntry] = {}  # built when saving cache
        n = 0
        live_run_frames = []

        for frame in tqdm(self.dataset, desc=self.desc, unit="frame"):
            n += 1
            logger.debug("Processing frame %s", frame.frame_id)

            # Reset history at scene boundaries so temporal defenses never read
            # across a discontinuity between unrelated scenes.
            if frame.sequence_id != last_sequence_id:
                clean_history.clear()
                dirty_history.clear()
                last_sequence_id = frame.sequence_id

            if self._precomputed_cache is not None:
                # -------------------------------------------------------
                # Replay mode: use cached clean predictions and attack decisions.
                # -------------------------------------------------------
                entry = self._precomputed_cache.get(frame.frame_id)
                if entry is None:
                    live_run_frames.append(frame.frame_id)
                    clean_preds, attacked_frame, attacked_preds = self._run_live(frame)
                else:
                    clean_preds = entry.clean_predictions
                    if entry.is_attacked and self.attack is not None:
                        if self.use_cached_attacks:
                            # Use cached predictions + metadata; don't re-run the
                            # attack so predictions and metadata are consistent.
                            attacked_preds = entry.attacked_predictions
                            attacked_frame = dataclasses.replace(
                                frame,
                                is_attacked=True,
                                attacked_modalities=frozenset({"lidar"}),
                                attack_metadata=entry.attack_metadata,
                            )
                        else:
                            # Re-run the attack live for a fresh lidar and fresh
                            # predictions; the new result is not saved to cache.
                            attacked_frame = self.attack.apply(frame)
                            attacked_preds = (
                                self.detector.predict(attacked_frame)
                                if self.detector is not None else None
                            )
                    else:
                        attacked_frame = None
                        attacked_preds = None
            else:
                # -------------------------------------------------------
                # Live mode: run attack + detector, accumulate cache entry.
                # -------------------------------------------------------
                clean_preds, attacked_frame, attacked_preds = self._run_live(frame)

                if self._precomputed_save_path is not None:
                    accumulator[frame.frame_id] = FrameCacheEntry(
                        clean_predictions=clean_preds,
                        attacked_predictions=attacked_preds,
                        is_attacked=attacked_frame is not None,
                        attack_metadata=(
                            dict(attacked_frame.attack_metadata)
                            if attacked_frame is not None else {}
                        ),
                    )

            # Stage 3: Defense — operates on what the vehicle actually received
            current_frame = attacked_frame if attacked_frame is not None else frame
            current_preds = attacked_preds if attacked_preds is not None else clean_preds
            current_frame = current_frame.with_predictions(current_preds)

            defense_result = None
            if self.defense is not None:
                defense_result = self.defense.detect(
                    current_frame,
                    FrameHistory(clean=clean_history, dirty=dirty_history),
                )

            # Update both histories after the defense has been called.
            clean_history.append(frame)          # pre-attack, as yielded by dataset
            dirty_history.append(current_frame)  # post-attack, what the vehicle received

            frame_results.append(FrameResult(
                frame_id=frame.frame_id,
                labels=frame.labels,
                is_attacked=current_frame.is_attacked,
                clean_predictions=clean_preds,
                attacked_predictions=attacked_preds,
                defense_result=defense_result,
            ))

        logger.info("Pipeline complete: %d frames processed. %d frames processed live as they were not found in precomputed cache", n, len(live_run_frames))

        if self._precomputed_save_path is not None and accumulator:
            self._save_cache(accumulator)

        return EvalResults(frame_results=frame_results)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_live(
        self, frame: Frame
    ) -> tuple[list[Prediction], Frame | None, list[Prediction] | None]:
        """Run attack + detection for one frame without consulting the cache.

        Returns (clean_preds, attacked_frame_or_None, attacked_preds_or_None).
        """
        clean_preds = self._get_clean_preds(frame)

        attacked_frame: Frame | None = None
        attacked_preds: list[Prediction] | None = None

        if self.attack is not None and self._attack_rng.random() < self.attack_fraction:
            attacked_frame = self.attack.apply(frame)
            if self.detector is not None:
                attacked_preds = self.detector.predict(attacked_frame)

        return clean_preds, attacked_frame, attacked_preds

    def _get_clean_preds(self, frame: Frame) -> list[Prediction]:
        if self.detector is None:
            return []
        if self.cache_clean_preds and frame.frame_id in self._clean_pred_cache:
            return self._clean_pred_cache[frame.frame_id]
        preds = self.detector.predict(frame)
        if self.cache_clean_preds:
            self._clean_pred_cache[frame.frame_id] = preds
        return preds

    def _save_cache(self, accumulator: dict[str, FrameCacheEntry]) -> None:
        path = pathlib.Path(self._precomputed_save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(accumulator, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(
            "Saved precomputed cache: %d frame entries → %s", len(accumulator), path
        )
