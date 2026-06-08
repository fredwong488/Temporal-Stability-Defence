"""
eval_pipeline/defenses/llm/defense.py
---------------------------------------
Multi-modal LLM-based adversarial attack defense.

Sends three rendered views (BEV LiDAR, isometric LiDAR, camera) to a
vision-language model and maps the structured response to a DetectionResult.

Supported backends: 'gemini' (Gemini 3 Flash via google-genai)
"""

from __future__ import annotations

import hashlib
import pathlib
import threading
import time
from collections import deque
from typing import Literal
import numpy as np
import dataclasses


class _RateLimiter:
    """Allow at most max_calls requests in any rolling 60-second window.

    Thread-safe: multiple worker threads share one instance and block here
    until capacity is available.
    """

    def __init__(self, max_calls_per_minute: int) -> None:
        self._max = max_calls_per_minute
        self._window: deque[float] = deque()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                while self._window and now - self._window[0] >= 60.0:
                    self._window.popleft()
                if len(self._window) < self._max:
                    self._window.append(now)
                    return
                wait_until = self._window[0] + 60.0
            time.sleep(max(wait_until - time.monotonic(), 0.05))

from eval_pipeline.base import BaseDefense
from eval_pipeline.types import DetectionResult, Frame, FrameHistory

from .cache import LLMCache, make_cache_key
from .schema import ConfidenceLevel, LLMVerdict, Verdict

_CONFIDENCE_MAP = {
    ConfidenceLevel.LOW: 0.33,
    ConfidenceLevel.MEDIUM: 0.66,
    ConfidenceLevel.HIGH: 1.0,
}

_DEFAULT_MODELS = {
    "gemini": "gemini-3.1-flash-lite",
}


class LLMDefense(BaseDefense):
    """Attack detection via a multimodal LLM judging three sensor views."""

    def __init__(
        self,
        backend: Literal["gemini"] = "gemini",
        model: str | None = None,
        prompt_path: str = "eval_pipeline/defenses/llm/llm_prompt.md",
        cache_dir: str = "cache/llm_defense",
        force_refresh: bool = False,
        attack_threshold: Literal["any", "high_conf"] = "any",
        render_dpi: int = 150,
        thinking_effort: int = 1,
        roi_forward: float = 50.0,
        roi_side: float = 20.0,
        roi_rear: float = 50.0,
        ego_front: float = 2.0,
        ego_rear: float = 2.0,
        ego_side: float = 1.4,
        api_key_env: str | None = None,
        requests_per_minute: int | None = 200,
    ) -> None:
        self._backend_name = backend
        self._model = model or _DEFAULT_MODELS[backend]
        self._force_refresh = force_refresh
        self._attack_threshold = attack_threshold
        self._render_dpi = render_dpi
        self._thinking_effort = thinking_effort
        self._roi_min = (-roi_rear, -roi_side)
        self._roi_max = (roi_forward, roi_side)
        self._ego_front = ego_front
        self._ego_rear = ego_rear
        self._ego_side = ego_side

        prompt_file = pathlib.Path(prompt_path)
        if not prompt_file.exists():
            raise FileNotFoundError(f"LLM prompt file not found: {prompt_file.resolve()}")
        self._prompt = prompt_file.read_text()
        self._prompt_hash = hashlib.sha1(self._prompt.encode()).hexdigest()[:12]

        self._cache = LLMCache(cache_dir, backend, self._model)

        kwargs = {}
        if api_key_env is not None:
            kwargs["api_key_env"] = api_key_env

        self._rate_limiter = (
            _RateLimiter(requests_per_minute) if requests_per_minute is not None else None
        )

        if backend == "gemini":
            from .backends.gemini import GeminiBackend
            self._backend = GeminiBackend(model=self._model, **kwargs)
        else:
            raise ValueError(f"Unknown LLM backend: {backend!r}. Choose only 'gemini' supported for now.")

    @property
    def async_detect(self) -> bool:
        return True

    @property
    def name(self) -> str:
        return f"LLMDefense({self._backend_name}/{self._model})"

    def detect(self, frame: Frame, history: FrameHistory) -> DetectionResult:
        from eval_pipeline.visualisation.render_views import render_three_views
        t0 = time.perf_counter()

        predictions = frame.predictions or []

        cache_key = make_cache_key(
            frame_id=frame.frame_id,
            sequence_id=frame.sequence_id,
            predictions=predictions,
            is_attacked=frame.is_attacked,
            attack_metadata=frame.attack_metadata,
            backend=self._backend_name,
            model=self._model,
            prompt_hash=self._prompt_hash,
        )

        cache_hit = False
        token_info: dict | None = None
        raw: dict | None = None
        if not self._force_refresh:
            raw = self._cache.load(cache_key)
            if raw is not None:
                cache_hit = True

        t_query: float | None = None
        if raw is None:
            if self._rate_limiter is not None:
                self._rate_limiter.acquire()

            render_frame = frame
            if frame.lidar is not None and len(frame.lidar):
                from eval_pipeline.defenses._multiframe_common import remove_ego_box
                filtered = remove_ego_box(
                    frame.lidar, self._ego_front, self._ego_rear, self._ego_side,
                )
                render_frame = dataclasses.replace(frame, lidar=filtered)

            images = render_three_views(
                render_frame, predictions,
                roi_min=self._roi_min,
                roi_max=self._roi_max,
                dpi=self._render_dpi,
            )

            t_query = time.perf_counter()
            raw, token_info = self._backend.query(images, self._prompt, self._thinking_effort)
            self._cache.save(cache_key, raw)

        t_end = time.perf_counter()
        total_elapsed_s = t_end - t0
        query_elapsed_s = t_end - t_query if t_query is not None else 0.0

        return self._parse(raw, cache_hit, total_elapsed_s, query_elapsed_s, token_info)

    def _parse(self, raw: dict, cache_hit: bool, total_elapsed_s: float, query_elapsed_s: float, token_info: dict | None = None) -> DetectionResult:
        try:
            verdict_obj = LLMVerdict.model_validate(raw)
        except Exception:
            metadata: dict = {"error": "Failed to parse LLM response", "raw_response": raw, "cache_hit": cache_hit, "total_elapsed_s": total_elapsed_s, "query_elapsed_s": query_elapsed_s}
            if token_info is not None:
                metadata["token_usage"] = token_info
            return DetectionResult(
                is_attack_detected=False,
                confidence=0.0,
                metadata=metadata,
            )

        verdict = verdict_obj.verdict
        is_attack = verdict == Verdict.ATTACK_SUSPECTED

        if is_attack and self._attack_threshold == "high_conf":
            # Require at least MEDIUM confidence on the primary suspected attack
            if verdict_obj.suspected_attacks:
                top_conf = verdict_obj.suspected_attacks[0].confidence
                if top_conf == ConfidenceLevel.LOW:
                    is_attack = False
            else:
                is_attack = False

        # Aggregate confidence: max over all suspected attacks, 0 if benign
        if verdict_obj.suspected_attacks:
            confidence = max(
                _CONFIDENCE_MAP[a.confidence] for a in verdict_obj.suspected_attacks
            )
        else:
            confidence = 0.0

        metadata: dict = {
            "verdict": verdict.value,
            "backend": self._backend_name,
            "model": self._model,
            "cache_hit": cache_hit,
            "total_elapsed_s": total_elapsed_s,
            "query_elapsed_s": query_elapsed_s,
            "raw_response": raw,
        }
        if token_info is not None:
            metadata["token_usage"] = token_info

        if verdict_obj.suspected_attacks:
            metadata["suspected_attacks"] = [
                {
                    "attack_type": a.attack_type.value,
                    "confidence": a.confidence.value,
                    "affected_region": a.affected_region.model_dump(),
                    "evidence": a.evidence,
                    "alternatives_ruled_out": a.alternatives_ruled_out,
                }
                for a in verdict_obj.suspected_attacks
            ]

        return DetectionResult(
            is_attack_detected=is_attack,
            confidence=confidence,
            metadata=metadata,
        )
