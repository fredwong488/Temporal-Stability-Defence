"""Gemini backend using the google-genai SDK."""

from __future__ import annotations

import json
import logging
import os

from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("google_genai").setLevel(logging.WARNING)

load_dotenv()

from .base import LLMBackend

_DEFAULT_MODEL = "gemini-3.1-flash-lite"


class GeminiBackend(LLMBackend):
    def __init__(self, model: str | None = None, api_key_env: str = "GOOGLE_API_KEY") -> None:
        try:
            from google import genai
            from google.genai import types as genai_types
        except ImportError as exc:
            raise ImportError(
                "google-genai is required for the Gemini backend. "
                "Add it to pixi.toml pypi-dependencies."
            ) from exc

        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise EnvironmentError(
                f"Gemini API key not found. Set the {api_key_env} environment variable."
            )

        self._client = genai.Client(api_key=api_key)
        self._types = genai_types
        self.model = model or _DEFAULT_MODEL

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=16))
    def query(self, images: dict[str, bytes], prompt: str) -> dict:
        from google.genai import types as genai_types
        from eval_pipeline.defenses.llm.schema import LLMVerdict

        image_parts = [
            genai_types.Part.from_bytes(data=images["bev"], mime_type="image/png"),
            genai_types.Part.from_bytes(data=images["isometric"], mime_type="image/png"),
            genai_types.Part.from_bytes(data=images["camera"], mime_type="image/png"),
        ]

        response = self._client.models.generate_content(
            model=self.model,
            contents=[prompt, *image_parts],
            config=genai_types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=LLMVerdict,
                media_resolution=genai_types.MediaResolution.MEDIA_RESOLUTION_MEDIUM,
                thinking_config=genai_types.ThinkingConfig(thinking_budget=200),
            ),
        )
        usage = response.usage_metadata
        token_info: dict | None = None
        if usage is not None:
            token_info = {
                "input_tokens": usage.prompt_token_count,
                "output_tokens": usage.candidates_token_count,
                "total_tokens": usage.total_token_count,
            }
        return json.loads(response.text), token_info
