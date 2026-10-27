"""Gemini text generation (answers, query routing, reranking, vision) with model fallback.

Embeddings are NOT done here: they run locally through app/embedder.py (bge-small, no API quota).
"""

from __future__ import annotations

import time

from google import genai
from google.genai import errors, types

from app.config import Settings


class Gemini:
    def __init__(self, settings: Settings):
        if not settings.gemini_api_key:
            raise RuntimeError("GEMINI_API_KEY is not set. Copy .env.example to .env and add your key.")
        self.s = settings
        self.client = genai.Client(api_key=settings.gemini_api_key)

    # ---- generation -------------------------------------------------------------------------
    def generate(
        self,
        prompt: str,
        *,
        model: str,
        system: str | None = None,
        json_output: bool = False,
        temperature: float = 0.1,
    ) -> str:
        """Call `model`; on 429/503 retry, then fall back to the router model (a cheaper, separate quota)."""
        config = types.GenerateContentConfig(
            temperature=temperature,
            system_instruction=system,
            response_mime_type="application/json" if json_output else None,
        )
        candidates = [model] if model == self.s.router_model else [model, self.s.router_model]
        last_error: Exception | None = None
        for candidate in candidates:
            for attempt in range(3):
                try:
                    res = self.client.models.generate_content(model=candidate, contents=prompt, config=config)
                    return (res.text or "").strip()
                except errors.APIError as e:
                    last_error = e
                    if e.code in (429, 500, 503):
                        time.sleep(2 * (attempt + 1))
                        continue
                    raise
        raise RuntimeError(f"all models failed: {last_error}")
