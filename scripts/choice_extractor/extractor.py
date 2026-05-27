from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass
from typing import Optional

from openai import OpenAI

from app_core.logging import get_logger
from .models import ExtractionResult

logger = get_logger(__name__)

_DEFAULT_MODEL = "gpt-5.5"

_DEFAULT_PROMPT = """
        You are analyzing a roguelike game screenshot.
        Extract the item/upgrade titles shown on a choice or reward screen and identify the currently selected (hovered) one.

        Step 1 — Find the choice cards: locate all item/upgrade cards and read their titles.

        Step 2 — For each card, count how many of these selection indicators it has. All indicators are equal.
        Select the card with the most indicators (even one is enough if no other card has any):

        - Decorative elements at or near the card: corner markers, L-shaped brackets, arrows or triangles
          pointing inward from any side, side indicators, glows, highlights, or any other ornament. These can
          appear directly on the border or anywhere close enough to clearly belong to that card.
        - The card appears noticeably darker than expected for its rarity color (a dark overlay on top of the
          natural rarity shade).
        - A visible mouse cursor on the card.
        - Short lines or dashes flanking the card's sides.

        Step 3 — In your reasoning, for EACH card:
        - Describe what you see on the card itself and in the area immediately surrounding it.
        - List every selection indicator you observe (or "none").

        Step 4 — Select the card with the highest indicator count. If tied, pick the one whose indicators
        are clearest. Do not confuse rarity color with selection: a card that looks different only because
        of its rarity (gray, green, blue, yellow, purple) has zero selection indicators.

        If no card shows any hover indicator after careful inspection, return an empty string for selected_choice.
        If the screen is not a choice or reward screen, return an empty list and empty string.

        Respond with a JSON object in exactly this format:
        {
          "choices": ["title1", "title2", "title3"],
          "reasoning": "describe each card and its indicators here",
          "selected_choice": "title1"
        }
        """


@dataclass(frozen=True)
class ChoiceExtractorConfig:
    api_key: str = ""
    model: str = _DEFAULT_MODEL
    timeout_seconds: float = 30.0


class ChoiceExtractor:
    def __init__(
        self,
        api_key: str = "",
        config: ChoiceExtractorConfig | None = None,
    ) -> None:
        cfg = config or ChoiceExtractorConfig(api_key=api_key)
        key = cfg.api_key or os.getenv("OPENAI_API_KEY", "")
        self._client = OpenAI(api_key=key, timeout=cfg.timeout_seconds)
        self._model = cfg.model

    def reset_session(self) -> None:
        pass

    def extract_frame(
        self,
        image_bytes: bytes,
        prompt: Optional[str] = None,
        model: Optional[str] = None,
    ) -> ExtractionResult:
        b64 = base64.b64encode(image_bytes).decode("ascii")
        data_url = f"data:image/png;base64,{b64}"
        used_model = model or self._model
        used_prompt = prompt or _DEFAULT_PROMPT

        try:
            resp = self._client.chat.completions.create(
                model=used_model,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": used_prompt},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Return all choice titles and the selected choice.",
                            },
                            {"type": "image_url", "image_url": {"url": data_url}},
                        ],
                    },
                ],
            )
            content = resp.choices[0].message.content
            data = json.loads(content) if isinstance(content, str) else content
            return ExtractionResult(
                choices=data.get("choices", []),
                selected_choice=data.get("selected_choice"),
            )
        except Exception as e:
            logger.error("OpenAI extraction failed: %s", e)
            raise
