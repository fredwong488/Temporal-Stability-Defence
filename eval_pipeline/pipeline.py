"""
pipeline.py
-----------
EvalPipeline — orchestrates attack, detection, and defense over a dataset.
"""

from __future__ import annotations

import dataclasses
import dbm
import itertools
import logging
import os
import pathlib
import pickle
import shelve
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor

import numpy as np
from tqdm import tqdm

from .base import BaseAttack, BaseDefense, BaseDetector
from .types import EvalResults, Frame, FrameCacheEntry, FrameHistory, FrameResult, ObjectLabel, Prediction


def _prediction_to_label(pred: Prediction) -> ObjectLabel:
    """Convert a detector Prediction to an ObjectLabel for use as attack input."""
    return ObjectLabel(
        type=pred.type,
        truncated=None,
        occluded=None,
        alpha=None,
        bbox_2d=None,
        height=pred.height,
        width=pred.width,
        length=pred.length,
        x=pred.x,
        y=pred.y,
        z=pred.z,
        rotation_y=pred.rotation_y,
        corners_velo=pred.corners_velo,
    )

logger = logging.getLogger(__name__)


def _cache_exists(path: str) -> bool:
    """Return True if a shelve database already exists at *path*."""
    return dbm.whichdb(path) is not None


# Maps dataset class name → attack granularity.  Unknown datasets fall back to "frame".
_DATASET_GRANULARITY: dict[str, str] = {
    "NuScenesDataset": "scene",
    "KittiObjectDataset": "frame",
}


@dataclasses.dataclass
class _SceneAttackPlan:
    """Per-scene attack decision, computed once before iteration begins."""
    attack: bool
    prefix: int          # randomized frames to leave unattacked at scene start
    scene_length: int
    attack_start_frame_id: str | None = None  # set when the first attacked frame is seen


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
    use_predicted_labels
        When True, the attack receives clean detector predictions converted to
        labels rather than the ground-truth ``frame.labels``.  Use this for
        datasets where not every frame is annotated (e.g. NuScenes at 10 Hz,
        where only 2 Hz keyframes carry ground-truth labels) so that the attack
        fires on every frame rather than only on annotated ones.  Requires a
        detector to be configured.
    pred_label_score_threshold
        Minimum detection score for a prediction to be included as an attack
        label when ``use_predicted_labels`` is True.  Default 0.5.
    min_unattacked_frames
        For scene-granularity datasets (e.g. NuScenes): minimum number of frames
        left unattacked at the start of each attacked scene.  The actual prefix
        is randomised uniformly in [min_unattacked_frames,
        scene_length - min_attacked_frames].  Ignored in frame-granularity mode.
    min_attacked_frames
        For scene-granularity datasets: minimum number of frames that must be
        attacked in a chosen scene.  Scenes where
        scene_length < min_unattacked_frames + min_attacked_frames revert to
        fully unattacked.  Ignored in frame-granularity mode.
    max_frames
        Stop after processing this many frames across all scenes.  Useful for
        quick inspection runs.  If None (default), all frames are processed.
    skip_unattacked_frames_per_scene
        Skip this many unattacked frames at the start of each scene before
        beginning to process any unattacked frame.  Skipped frames are
        completely bypassed (no attack, no defense, no history update).
        Default 0.  Only effective in scene-granularity mode.
    skip_attacked_frames_per_scene
        Skip this many attacked frames at the start of each scene's attack
        phase before beginning to process any attacked frame.  Default 0.
        Only effective in scene-granularity mode.
    max_unattacked_frames_per_scene
        After ``skip_unattacked_frames_per_scene``, process at most this many
        unattacked frames per scene; remaining unattacked frames are skipped
        entirely.  If None (default), all unattacked frames are processed.
        Only effective in scene-granularity mode.
    max_attacked_frames_per_scene
        After ``skip_attacked_frames_per_scene``, process at most this many
        attacked frames per scene; remaining attacked frames are skipped
        entirely.  If None (default), all attacked frames are processed.
        Only effective in scene-granularity mode.
    checkpoint_path
        Path stem for checkpoint sidecars written at each scene boundary.
        Two files are created: ``<stem>.ckpt.pkl`` (atomic metadata) and
        ``<stem>.frames.pkl`` (append-only FrameResult stream).  If the
        ``.ckpt.pkl`` sidecar already exists when the pipeline starts, the
        run resumes from the last completed scene.  Only supported for
        scene-granularity datasets with synchronous defenses.  Pass None
        (default) to disable checkpointing.
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
        read_only_cache: bool = True,
        use_cached_attacks: bool = False,
        use_predicted_labels: bool = False,
        pred_label_score_threshold: float = 0.5,
        min_unattacked_frames: int = 0,
        min_attacked_frames: int = 1,
        max_frames: int | None = None,
        skip_unattacked_frames_per_scene: int = 0,
        skip_attacked_frames_per_scene: int = 0,
        max_unattacked_frames_per_scene: int | None = None,
        max_attacked_frames_per_scene: int | None = None,
        checkpoint_path: str | None = None,
    ) -> None:
        self.dataset = dataset
        self.attack = attack
        self.detector = detector
        self.defense = defense
        self.cache_clean_preds = cache_clean_preds
        self.attack_fraction = attack_fraction
        self._attack_rng = np.random.default_rng(attack_fraction_seed)
        self.use_cached_attacks = use_cached_attacks
        self.use_predicted_labels = use_predicted_labels
        self.pred_label_score_threshold = pred_label_score_threshold
        self.min_unattacked_frames = min_unattacked_frames
        self.min_attacked_frames = min_attacked_frames
        self.max_frames = max_frames
        self.skip_unattacked_frames_per_scene = skip_unattacked_frames_per_scene
        self.skip_attacked_frames_per_scene = skip_attacked_frames_per_scene
        self.max_unattacked_frames_per_scene = max_unattacked_frames_per_scene
        self.max_attacked_frames_per_scene = max_attacked_frames_per_scene
        self.desc = desc
        self._clean_pred_cache: dict[str, list[Prediction]] = {}

        from .attacks.ora import ORAAttack
        self._is_ora_attack: bool = isinstance(attack, ORAAttack)

        self._granularity: str = _DATASET_GRANULARITY.get(type(dataset).__name__, "frame")
        self._scene_plan: dict[str, _SceneAttackPlan] | None = None

        # Checkpoint support
        self._checkpoint_path: str | None = checkpoint_path
        self._ckpt_stream: object = None  # open file handle for the .frames.pkl stream

        # Precomputed cache: opened lazily via shelve so frames are read/written
        # one at a time rather than loading the entire dict into memory.
        # read_only_cache=True  → flag='r': reads only, safe for concurrent Optuna trials.
        # read_only_cache=False → flag='c': reads + writes; generate fresh or resume partial.
        self._cache: shelve.Shelf | None = None
        self._cache_writable: bool = False
        if precomputed_cache_path is not None:
            cache_exists = _cache_exists(precomputed_cache_path)
            if cache_exists and read_only_cache:
                self._cache = shelve.open(precomputed_cache_path, flag='r')
                self._cache_writable = False
                logger.info(
                    "Opened precomputed cache read-only: %s (~%d entries)",
                    precomputed_cache_path, len(self._cache),
                )
            else:
                pathlib.Path(precomputed_cache_path).parent.mkdir(parents=True, exist_ok=True)
                self._cache = shelve.open(precomputed_cache_path, flag='c')
                self._cache_writable = True
                if cache_exists:
                    logger.info(
                        "Opened precomputed cache for resume/append: %s (~%d existing entries)",
                        precomputed_cache_path, len(self._cache),
                    )
                else:
                    logger.info("Will generate precomputed cache: %s", precomputed_cache_path)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self) -> EvalResults:
        """Execute the pipeline over all frames and return aggregated results."""
        # Validate checkpoint support before doing any work.
        _checkpointing = self._checkpoint_path is not None
        if _checkpointing:
            if self._granularity != "scene":
                raise NotImplementedError(
                    "Checkpointing is only supported for scene-granularity datasets "
                    "(e.g. NuScenes). Pass checkpoint_path=None for KITTI / frame-mode."
                )
            if self.defense is not None and self.defense.async_detect:
                raise NotImplementedError(
                    "Checkpointing is not supported with async defenses (async_detect=True). "
                    "Pass checkpoint_path=None or use a synchronous defense."
                )

        # History must satisfy both the defense's temporal window and the
        # detector's sweep-accumulation depth (multi-sweep NuScenes models need
        # ~9 past sweeps even with no temporal defense).
        defense_window = self.defense.temporal_window if self.defense else 1
        detector_sweeps = getattr(self.detector, "num_sweeps", 1) if self.detector else 1
        max_window = max(defense_window, detector_sweeps)
        # Two parallel histories: clean (pre-attack) and dirty (post-attack).
        # Sized to max_window - 1 so each prediction sees prior frames only.
        clean_history: deque[Frame] = deque(maxlen=max(0, max_window - 1))
        dirty_history: deque[Frame] = deque(maxlen=max(0, max_window - 1))
        last_sequence_id: str | None = None

        # Restore from checkpoint if one exists.
        resume_start_index = 0
        if _checkpointing:
            resume_start_index, restored = self._load_checkpoint()
        else:
            restored = []

        frame_results: list[FrameResult] = restored
        # These track frames in the current scene that haven't been checkpointed yet.
        _scene_pending: list[FrameResult] = []

        defense_futures: list[Future | None] = []  # parallel to frame_results; None = sync
        executor: ThreadPoolExecutor | None = (
            ThreadPoolExecutor()
            if self.defense is not None and self.defense.async_detect
            else None
        )
        n = 0
        live_run_frames = []
        live_attack_rerun_no_lidar = 0  # cache hit but lidar missing (old cache format)
        live_attack_rerun_not_use_cache = 0  # use_cached_attacks=False
        live_attack_rerun_not_cached = 0  # cache hit but entry.is_attacked=False, attack required
        frame_index_in_scene = 0
        n_unattacked_in_scene = 0
        n_attacked_in_scene = 0

        _scene_skip_active = any([
            self.skip_unattacked_frames_per_scene,
            self.max_unattacked_frames_per_scene is not None,
            self.skip_attacked_frames_per_scene,
            self.max_attacked_frames_per_scene is not None,
        ])
        if _scene_skip_active and self._granularity != "scene":
            logger.warning(
                "skip/max per-scene frame params are only effective in scene-granularity "
                "mode (e.g. NuScenes); they will be ignored for this dataset."
            )

        # Pre-compute scene plans (scene mode) or just scene lengths (frame mode).
        if self._granularity == "scene":
            self._scene_plan = self._plan_scene_attacks()
            scene_lengths = {sid: p.scene_length for sid, p in self._scene_plan.items()}
        elif hasattr(self.dataset, "scene_lengths"):
            scene_lengths = self.dataset.scene_lengths()
        else:
            scene_lengths = {}

        frame_iter = self._make_frame_iter(resume_start_index)
        if self.max_frames is not None:
            remaining = self.max_frames - resume_start_index
            frame_iter = itertools.islice(frame_iter, max(0, remaining))
        _dataset_len = len(self.dataset) if hasattr(self.dataset, "__len__") else None
        total = (
            (self.max_frames if self.max_frames is not None else _dataset_len)
        )
        if total is not None:
            total = max(0, total - resume_start_index)
        absolute_index = resume_start_index
        for frame in tqdm(frame_iter, desc=self.desc, unit="frame", total=total):
            n += 1
            logger.debug("Processing frame %s", frame.frame_id)

            # Reset history at scene boundaries so temporal defenses never read
            # across a discontinuity between unrelated scenes.
            if frame.sequence_id != last_sequence_id:
                # Write a checkpoint at every scene boundary (after the first).
                # At this point all FrameResults in _scene_pending belong to the
                # just-finished scene and the history deques are about to be cleared,
                # so no lidar or defense state needs to be persisted.
                if _checkpointing and absolute_index > resume_start_index and _scene_pending:
                    self._write_checkpoint(
                        next_frame_index=absolute_index,
                        new_records=_scene_pending,
                        frame_count=len(frame_results),
                    )
                    _scene_pending = []

                clean_history.clear()
                dirty_history.clear()
                if self.defense is not None:
                    self.defense.reset()
                frame_index_in_scene = 0
                n_unattacked_in_scene = 0
                n_attacked_in_scene = 0
                last_sequence_id = frame.sequence_id
            else:
                frame_index_in_scene += 1

            absolute_index += 1

            # Determine per-frame attack decision.
            # Scene mode: use precomputed plan (bool).
            # Frame mode: None — _run_live uses the per-frame RNG.
            do_attack: bool | None
            if self._granularity == "scene":
                plan_entry = self._scene_plan[frame.sequence_id]
                do_attack = plan_entry.attack and frame_index_in_scene >= plan_entry.prefix
            else:
                do_attack = None

            # Per-scene skip/max logic (scene-granularity only; do_attack is bool here).
            if _scene_skip_active and do_attack is not None:
                if do_attack:
                    n_attacked_in_scene += 1
                    _in_skip = n_attacked_in_scene <= self.skip_attacked_frames_per_scene
                    _past_max = (
                        self.max_attacked_frames_per_scene is not None
                        and n_attacked_in_scene
                        > self.skip_attacked_frames_per_scene + self.max_attacked_frames_per_scene
                    )
                else:
                    n_unattacked_in_scene += 1
                    _in_skip = n_unattacked_in_scene <= self.skip_unattacked_frames_per_scene
                    _past_max = (
                        self.max_unattacked_frames_per_scene is not None
                        and n_unattacked_in_scene
                        > self.skip_unattacked_frames_per_scene + self.max_unattacked_frames_per_scene
                    )
                if _in_skip or _past_max:
                    # History-only pass: maintain temporal continuity without
                    # running the defense.  Use cached attacked lidar if cheap
                    # to obtain; never run the attack live.
                    _history_frame = frame
                    if self._cache is not None:
                        _entry = self._cache.get(frame.frame_id)
                        if _entry is not None:
                            _attack_this_frame = do_attack if do_attack is not None else _entry.is_attacked
                            if (_attack_this_frame and
                                self.attack is not None and
                                self.use_cached_attacks and
                                _entry.is_attacked and
                                _entry.attacked_lidar is not None):
                                    _history_frame = dataclasses.replace(
                                        frame,
                                        lidar=_entry.attacked_lidar,
                                        is_attacked=True,
                                        attacked_modalities=frozenset({"lidar"}),
                                        attack_metadata=_entry.attack_metadata,
                                    )
                        else:
                            logger.warning(f"Precomputed cache not found for frame_id {frame.frame_id}. Scene skip active, dirty history will be clean frame.")
                    clean_history.append(frame)
                    dirty_history.append(_history_frame)
                    continue

            if self._cache is not None:
                # -------------------------------------------------------
                # Cache mode: hit → replay; miss → run live (+ write back
                # if writable, enabling generate/resume).
                # -------------------------------------------------------
                entry = self._cache.get(frame.frame_id)
                if entry is None:
                    live_run_frames.append(frame.frame_id)
                    clean_preds, attacked_frame, attacked_preds = self._run_live(
                        frame, clean_history, dirty_history, should_attack=do_attack
                    )
                    if self._cache_writable:
                        self._cache[frame.frame_id] = FrameCacheEntry(
                            clean_predictions=clean_preds,
                            attacked_predictions=attacked_preds,
                            is_attacked=attacked_frame is not None,
                            attack_metadata=(
                                dict(attacked_frame.attack_metadata)
                                if attacked_frame is not None else {}
                            ),
                            attacked_lidar=(
                                attacked_frame.lidar if attacked_frame is not None else None
                            ),
                        )
                        self._cache.sync()
                else:
                    attacked_frame: Frame | None = None
                    attacked_preds: list[Prediction] | None = None
                    clean_preds = entry.clean_predictions
                    # In scene mode the plan overrides the cached decision.
                    # In frame mode fall back to the cached decision.
                    attack_this_frame = do_attack if do_attack is not None else entry.is_attacked
                    if attack_this_frame and self.attack is not None:
                        if self.use_cached_attacks and entry.is_attacked:
                            if entry.attacked_lidar is not None:
                                # Use cached predictions, metadata, and lidar so that
                                # the defense receives the exact same data as the
                                # original run without re-running the attack.
                                attacked_preds = entry.attacked_predictions
                                attacked_frame = dataclasses.replace(
                                    frame,
                                    lidar=entry.attacked_lidar,
                                    is_attacked=True,
                                    attacked_modalities=frozenset({"lidar"}),
                                    attack_metadata=entry.attack_metadata,
                                )
                            else:
                                # Cache pre-dates lidar storage — run the attack live.
                                attacked_frame = self.attack.apply(
                                    self._get_attack_frame(frame, clean_preds)
                                )
                                if self.detector is not None:
                                    attacked_preds = self.detector.predict(attacked_frame, dirty_history)
                                live_attack_rerun_no_lidar += 1
                        else:
                            # Re-run the attack live for a fresh lidar and fresh
                            # predictions; the new result is not saved to cache.
                            attacked_frame = self.attack.apply(
                                self._get_attack_frame(frame, clean_preds)
                            )
                            if self.detector is not None:
                                attacked_preds = self.detector.predict(attacked_frame, dirty_history)
                            if not self.use_cached_attacks:
                                live_attack_rerun_not_use_cache += 1
                            else:
                                # use_cached_attacks=True but cached entry was not attacked
                                live_attack_rerun_not_cached += 1
            else:
                # -------------------------------------------------------
                # No cache configured: run everything live.
                # -------------------------------------------------------
                clean_preds, attacked_frame, attacked_preds = self._run_live(
                    frame, clean_history, dirty_history, should_attack=do_attack
                )

            # Stage 3: Defense — operates on what the vehicle actually received
            current_frame = attacked_frame if attacked_frame is not None else frame
            current_preds = attacked_preds if attacked_preds is not None else clean_preds
            current_frame = current_frame.with_predictions(current_preds)

            defense_result = None
            if self.defense is not None:
                if executor is not None:
                    # Snapshot history before submitting: the deques will be
                    # mutated by subsequent frames before the future resolves.
                    history_snapshot = FrameHistory(
                        clean=deque(clean_history),
                        dirty=deque(dirty_history),
                    )
                    defense_futures.append(
                        executor.submit(self.defense.detect, current_frame, history_snapshot)
                    )
                else:
                    defense_result = self.defense.detect(
                        current_frame,
                        FrameHistory(clean=clean_history, dirty=dirty_history),
                    )
                    defense_futures.append(None)
            else:
                defense_futures.append(None)

            # Update both histories after the defense has been called.
            clean_history.append(frame)          # pre-attack, as yielded by dataset
            dirty_history.append(current_frame)  # post-attack, what the vehicle received

            # Compute scene-position metadata for FrameResult.
            sl = scene_lengths.get(frame.sequence_id, 0)
            if self._granularity == "scene":
                plan_entry = self._scene_plan[frame.sequence_id]
                attack_start_index = plan_entry.prefix if plan_entry.attack else None
            else:
                attack_start_index = None

            _is_attacked = current_frame.is_attacked
            if self._is_ora_attack and _is_attacked:
                _removed_per_obj = current_frame.attack_metadata.get("removed_per_obj", [])
                _total_removed = sum(o.get("n_removed", 0) for o in _removed_per_obj)
                if _total_removed == 0:
                    _is_attacked = False

            fr = FrameResult(
                frame_id=frame.frame_id,
                labels=frame.labels,
                is_attacked=_is_attacked,
                attack_metadata=current_frame.attack_metadata,
                clean_predictions=clean_preds,
                attacked_predictions=attacked_preds,
                defense_result=defense_result,
                sequence_id=frame.sequence_id,
                frame_index_in_scene=frame_index_in_scene,
                scene_length=sl,
                attack_start_index=attack_start_index,
                attack_start_frame_id=None,  # filled in below
            )
            frame_results.append(fr)
            if _checkpointing:
                _scene_pending.append(fr)

        # Write final checkpoint for the last scene (no subsequent boundary triggers it).
        if _checkpointing and _scene_pending:
            self._write_checkpoint(
                next_frame_index=absolute_index,
                new_records=_scene_pending,
                frame_count=len(frame_results),
            )

        # Resolve async defense futures and attach results back to frame_results.
        if executor is not None:
            executor.shutdown(wait=True)
            for i, fut in enumerate(defense_futures):
                if fut is not None:
                    try:
                        frame_results[i].defense_result = fut.result()
                    except Exception:
                        logger.exception(
                            "Async defense failed for frame %s", frame_results[i].frame_id
                        )

        if self._cache is None:
            logger.info("Pipeline complete: %d frames processed live (no precomputed cache).", n)
        else:
            parts = [
                f"{n} frames processed",
                f"{len(live_run_frames)} cache misses ran fully live "
                f"({'written to cache' if self._cache_writable else 'discarded'})",
            ]
            if live_attack_rerun_no_lidar:
                parts.append(
                    f"{live_attack_rerun_no_lidar} cache hits re-ran attack live "
                    f"(cached lidar missing — old cache format)"
                )
            if live_attack_rerun_not_use_cache:
                parts.append(
                    f"{live_attack_rerun_not_use_cache} cache hits re-ran attack live "
                    f"(use_cached_attacks=False)"
                )
            if live_attack_rerun_not_cached:
                parts.append(
                    f"{live_attack_rerun_not_cached} cache hits re-ran attack live "
                    f"(cached entry was clean — scene plan required attack)"
                )
            logger.info("Pipeline complete: %s.", "; ".join(parts))

        self._close_cache()

        # Fill in attack_start_frame_id for all frames in attacked scenes.
        # Find the frame_id of the frame at the attack-onset index in each scene.
        if self._granularity == "scene":
            scene_onset_frame_id: dict[str, str] = {}
            for fr in frame_results:
                if (
                    fr.attack_start_index is not None
                    and fr.frame_index_in_scene == fr.attack_start_index
                ):
                    scene_onset_frame_id[fr.sequence_id] = fr.frame_id
            for fr in frame_results:
                if fr.attack_start_index is not None and fr.sequence_id in scene_onset_frame_id:
                    fr.attack_start_frame_id = scene_onset_frame_id[fr.sequence_id]

        return EvalResults(
            frame_results=frame_results,
            attack_types=self.attack.attack_types if self.attack is not None else frozenset(),
        )

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def _ckpt_meta_path(self) -> pathlib.Path:
        return pathlib.Path(self._checkpoint_path + ".ckpt.pkl")

    def _ckpt_stream_path(self) -> pathlib.Path:
        return pathlib.Path(self._checkpoint_path + ".frames.pkl")

    def _load_checkpoint(self) -> tuple[int, list]:
        """Load checkpoint meta and restore frame_results from the stream.

        Returns (resume_start_index, frame_results).  If no checkpoint exists,
        returns (0, []).
        """
        meta_path = self._ckpt_meta_path()
        stream_path = self._ckpt_stream_path()
        if not meta_path.exists():
            return 0, []
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        next_frame_index: int = meta["next_frame_index"]
        frame_count: int = meta["frame_count"]
        frames_offset: int = meta["frames_offset"]
        frame_results: list = []
        if stream_path.exists() and frame_count > 0:
            with open(stream_path, "rb") as f:
                for _ in range(frame_count):
                    try:
                        frame_results.append(pickle.load(f))
                    except (EOFError, pickle.UnpicklingError):
                        break
            if len(frame_results) != frame_count:
                logger.warning(
                    "Checkpoint stream truncated: expected %d records, got %d. "
                    "Resume index (%d) may be ahead of restored frame_results — "
                    "rerun from scratch if results look incomplete.",
                    frame_count, len(frame_results), next_frame_index,
                )
        # Truncate the stream to the last durably-written position so that a
        # subsequent checkpoint write appends cleanly.
        if stream_path.exists():
            with open(stream_path, "r+b") as f:
                f.truncate(frames_offset)
        logger.info(
            "Resuming from checkpoint: %d frames restored, next dataset index %d",
            len(frame_results), next_frame_index,
        )
        return next_frame_index, frame_results

    def _write_checkpoint(
        self,
        next_frame_index: int,
        new_records: list,
        frame_count: int,
    ) -> None:
        """Append new_records to the stream and atomically update the meta."""
        stream_path = self._ckpt_stream_path()
        meta_path = self._ckpt_meta_path()
        stream_path.parent.mkdir(parents=True, exist_ok=True)
        # Append new records to the stream.
        with open(stream_path, "ab") as f:
            for rec in new_records:
                pickle.dump(rec, f)
            f.flush()
            os.fsync(f.fileno())
            offset = f.tell()
        # Atomically replace the meta file.
        tmp = meta_path.with_suffix(".tmp")
        with open(tmp, "wb") as f:
            pickle.dump(
                {
                    "next_frame_index": next_frame_index,
                    "frame_count": frame_count,
                    "frames_offset": offset,
                },
                f,
            )
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, meta_path)
        logger.debug(
            "Checkpoint written: next_index=%d  total_frames=%d  offset=%d",
            next_frame_index, frame_count, offset,
        )

    def cleanup_checkpoint(self) -> None:
        """Delete checkpoint sidecars after a successful run."""
        if self._checkpoint_path is None:
            return
        for path in (self._ckpt_meta_path(), self._ckpt_stream_path()):
            try:
                path.unlink()
            except FileNotFoundError:
                pass

    def _make_frame_iter(self, start_index: int):
        """Return an iterator over dataset frames starting at start_index."""
        if start_index == 0:
            return iter(self.dataset)
        if hasattr(self.dataset, "__getitem__") and hasattr(self.dataset, "__len__"):
            end = len(self.dataset)
            return (self.dataset[i] for i in range(start_index, end))
        return itertools.islice(iter(self.dataset), start_index, None)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _plan_scene_attacks(self) -> dict[str, _SceneAttackPlan]:
        """Pre-compute per-scene attack decisions and prefix lengths."""
        lengths = self.dataset.scene_lengths()
        plan: dict[str, _SceneAttackPlan] = {}
        for seq_id, scene_length in lengths.items():
            if self._attack_rng.random() < self.attack_fraction:
                if scene_length >= self.min_unattacked_frames + self.min_attacked_frames:
                    prefix = int(self._attack_rng.integers(
                        self.min_unattacked_frames,
                        scene_length - self.min_attacked_frames + 1,
                    ))
                    plan[seq_id] = _SceneAttackPlan(
                        attack=True, prefix=prefix, scene_length=scene_length,
                    )
                else:
                    plan[seq_id] = _SceneAttackPlan(
                        attack=False, prefix=0, scene_length=scene_length,
                    )
            else:
                plan[seq_id] = _SceneAttackPlan(
                    attack=False, prefix=0, scene_length=scene_length,
                )
        return plan

    def _get_attack_frame(self, frame: Frame, preds: list[Prediction]) -> Frame:
        """Return frame for attack. When self.use_predicted_labels is True,
        labels are replaced by filtered clean predictions.

        Predictions below pred_label_score_threshold are dropped before substitution.
        """
        if not self.use_predicted_labels:
            return frame
        labels = [
            _prediction_to_label(p)
            for p in preds
            if p.score >= self.pred_label_score_threshold
        ]
        return dataclasses.replace(frame, labels=labels)

    def _run_live(
        self,
        frame: Frame,
        clean_history: deque[Frame],
        dirty_history: deque[Frame],
        *,
        should_attack: bool | None = None,
    ) -> tuple[list[Prediction], Frame | None, list[Prediction] | None]:
        """Run attack + detection for one frame without consulting the cache.

        Returns (clean_preds, attacked_frame_or_None, attacked_preds_or_None).

        should_attack=None  → use per-frame RNG (frame-granularity mode).
        should_attack=bool  → use provided decision (scene-granularity mode).

        clean_history / dirty_history supply past sweeps for multi-sweep
        detectors: clean for the clean prediction, dirty for the attacked one.
        """
        clean_preds = self._get_clean_preds(frame, clean_history)

        attacked_frame: Frame | None = None
        attacked_preds: list[Prediction] | None = None

        will_attack = (
            should_attack
            if should_attack is not None
            else self._attack_rng.random() < self.attack_fraction
        )
        if self.attack is not None and will_attack:
            attacked_frame = self.attack.apply(
                self._get_attack_frame(frame, clean_preds)
            )
            if self.detector is not None:
                attacked_preds = self.detector.predict(attacked_frame, dirty_history)

        return clean_preds, attacked_frame, attacked_preds

    def _get_clean_preds(
        self, frame: Frame, clean_history: deque[Frame]
    ) -> list[Prediction]:
        if self.detector is None:
            return []
        if self.cache_clean_preds and frame.frame_id in self._clean_pred_cache:
            return self._clean_pred_cache[frame.frame_id]
        preds = self.detector.predict(frame, clean_history)
        if self.cache_clean_preds:
            self._clean_pred_cache[frame.frame_id] = preds
        return preds

    def _close_cache(self) -> None:
        """Flush and close the shelve cache if one is open."""
        if self._cache is not None:
            self._cache.close()
            self._cache = None
