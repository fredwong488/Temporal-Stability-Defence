"""
eval_pipeline/defenses/llm/schema.py
--------------------------------------
Pydantic models for the LLM-defense structured response.

These serve dual purpose:
- Gemini: passed as response_schema to enforce JSON structure.
- Qwen: serialised via .model_json_schema() for response_format=json_schema.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel


class Verdict(str, Enum):
    BENIGN = "BENIGN"
    ATTACK_SUSPECTED = "ATTACK_SUSPECTED"
    UNCERTAIN = "UNCERTAIN"


class AttackType(str, Enum):
    OBJECT_HIDING = "OBJECT_HIDING"
    GHOST_OBJECT = "GHOST_OBJECT"
    OBJECT_TRANSLATION = "OBJECT_TRANSLATION"
    UNCERTAIN = "UNCERTAIN"


class ConfidenceLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class AffectedRegion(BaseModel):
    camera: str | None = None
    bev: str | None = None
    isometric: str | None = None


class SuspectedAttack(BaseModel):
    attack_type: AttackType
    confidence: ConfidenceLevel
    affected_region: AffectedRegion
    evidence: list[str]
    alternatives_ruled_out: list[str]


class LLMVerdict(BaseModel):
    verdict: Verdict
    suspected_attacks: list[SuspectedAttack] = []
