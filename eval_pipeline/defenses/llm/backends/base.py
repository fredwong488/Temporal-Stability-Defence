"""Abstract base for LLM vision backends."""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from eval_pipeline.defenses.llm.schema import LLMVerdict


class LLMBackend(abc.ABC):
    @abc.abstractmethod
    def query(
        self,
        images: dict[str, bytes],
        prompt: str,
    ) -> dict:
        """Send images + prompt to the LLM; return the parsed response dict.

        Parameters
        ----------
        images : dict with keys 'bev', 'isometric', 'camera' — PNG bytes.
        prompt : the system + user prompt string.
        """
        ...
