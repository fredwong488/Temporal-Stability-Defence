"""Qwen 3-VL backend via DashScope's OpenAI-compatible endpoint."""

from __future__ import annotations

import base64
import json
import os

from tenacity import retry, stop_after_attempt, wait_exponential

from .base import LLMBackend

_DEFAULT_MODEL = "qwen3.5-flash"
_BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"


class QwenBackend(LLMBackend):
    def __init__(self, model: str | None = None, api_key_env: str = "DASHSCOPE_API_KEY") -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "openai is required for the Qwen backend. "
                "Add it to pixi.toml pypi-dependencies."
            ) from exc

        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise EnvironmentError(
                f"DashScope API key not found. Set the {api_key_env} environment variable."
            )

        from openai import OpenAI
        self._client = OpenAI(api_key=api_key, base_url=_BASE_URL)
        self.model = model or _DEFAULT_MODEL

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=16))
    def query(self, images: dict[str, bytes], prompt: str) -> dict:
        from eval_pipeline.defenses.llm.schema import LLMVerdict

        def _b64(data: bytes) -> str:
            return base64.b64encode(data).decode()

        content = [{"type": "text", "text": prompt}]
        for key in ("bev", "isometric", "camera"):
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{_b64(images[key])}"},
            })

        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": content}],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "llm_verdict",
                    "schema": LLMVerdict.model_json_schema(),
                },
            },
            extra_body={"enable_thinking": False},
        )
        return json.loads(resp.choices[0].message.content)
