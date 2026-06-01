"""
eval_pipeline/defenses/llm/cache.py
-------------------------------------
On-disk JSON cache for LLM-defense responses.

Layout: <cache_dir>/<backend>/<model>/<frame_id>__<short_hash>.json
"""

from __future__ import annotations

import datetime
import hashlib
import json
import pathlib
from typing import Any


def _hash_key(key: dict) -> str:
    raw = json.dumps(key, sort_keys=True, default=str)
    return hashlib.sha1(raw.encode()).hexdigest()[:12]


class LLMCache:
    def __init__(self, cache_dir: str | pathlib.Path, backend: str, model: str) -> None:
        self._dir = pathlib.Path(cache_dir) / backend / model
        self._dir.mkdir(parents=True, exist_ok=True)

    def _path(self, frame_id: str, key_hash: str) -> pathlib.Path:
        return self._dir / f"{frame_id}__{key_hash}.json"

    def load(self, key: dict) -> dict | None:
        h = _hash_key(key)
        path = self._path(key["frame_id"], h)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text())["response"]
        except Exception:
            return None

    def save(self, key: dict, response: dict) -> None:
        h = _hash_key(key)
        path = self._path(key["frame_id"], h)
        payload = {
            "key": key,
            "response": response,
            "ts": datetime.datetime.utcnow().isoformat(),
        }
        path.write_text(json.dumps(payload, indent=2, default=str))


def make_cache_key(
    frame_id: str,
    sequence_id: str,
    predictions: list[Any],
    is_attacked: bool,
    attack_metadata: dict,
    backend: str,
    model: str,
    prompt_hash: str,
) -> dict:
    def _pred_sig(p: Any) -> tuple:
        return (
            getattr(p, "type", ""),
            round(float(getattr(p, "score", 0)), 4),
            round(float(getattr(p, "x", 0)), 3),
            round(float(getattr(p, "y", 0)), 3),
            round(float(getattr(p, "z", 0)), 3),
            round(float(getattr(p, "height", 0)), 3),
            round(float(getattr(p, "width", 0)), 3),
            round(float(getattr(p, "length", 0)), 3),
            round(float(getattr(p, "rotation_y", 0)), 4),
        )

    return {
        "frame_id": frame_id,
        "sequence_id": sequence_id,
        "predictions": sorted([_pred_sig(p) for p in predictions]),
        "is_attacked": is_attacked,
        "attack_metadata": attack_metadata,
        "backend": backend,
        "model": model,
        "prompt_hash": prompt_hash,
    }
