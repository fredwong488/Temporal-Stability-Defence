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
    ) -> tuple[dict, dict | None]:
        """Send images + prompt to the LLM; return (response_dict, token_info).

        Parameters
        ----------
        images : dict with keys 'bev', 'isometric', 'camera' — PNG bytes.
        prompt : the system + user prompt string.

        Returns
        -------
        response_dict : parsed JSON response from the model.
        token_info : dict with 'input_tokens', 'output_tokens', 'total_tokens',
                     or None if the backend does not expose usage data.
        """
        ...
