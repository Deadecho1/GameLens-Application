from __future__ import annotations

import base64
import json
import os
import re
from io import BytesIO

from openai import OpenAI
from PIL import Image

from .model import BossDetectionResult

_GOOGLE_AI_STUDIO_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
_DEFAULT_MODEL = "models/gemma-4-26b-a4b-it"
_PROMPT = """You are analyzing a gameplay screenshot.
Determine whether this frame shows a boss fight or regular gameplay.

A boss fight frame typically contains:
- A large health bar for an enemy (often at the top or bottom of the screen)
- A named enemy with a prominent title plate
- Dramatic UI elements or arena indicators

Respond with a JSON object in exactly this format:
{"class": "boss", "confidence": 0.95}

The "class" field must be either "boss" or "regular_gameplay".
The "confidence" field must be a float between 0.0 and 1.0.
"""


class GemmaBossClassifier:
    """Classifies gameplay frames as boss fight or regular gameplay using Gemma 4 via Google AI Studio."""

    def __init__(self, model: str = _DEFAULT_MODEL) -> None:
        api_key = os.environ.get("GOOGLE_AI_STUDIO_API_KEY")
        self._client = OpenAI(api_key=api_key, base_url=_GOOGLE_AI_STUDIO_BASE_URL)
        self._model = model

    def classify_frame(self, image: Image.Image) -> BossDetectionResult:
        buf = BytesIO()
        image.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        data_url = f"data:image/png;base64,{b64}"

        resp = self._client.chat.completions.create(
            model=self._model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Classify this frame."},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                },
            ],
        )

        content = resp.choices[0].message.content
        if isinstance(content, str):
            content = re.sub(r"^<thought>.*?</thought>", "", content, flags=re.DOTALL).strip()
        parsed = json.loads(content) if isinstance(content, str) else content
        class_name = str(parsed.get("class", "regular_gameplay"))
        confidence = float(parsed.get("confidence", 0.0))
        return BossDetectionResult(class_name=class_name, confidence=confidence)
