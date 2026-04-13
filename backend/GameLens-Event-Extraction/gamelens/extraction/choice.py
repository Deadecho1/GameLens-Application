import base64
import gc
import json
import os

# Assuming this exists in your project
from fastapi import APIRouter, File, HTTPException, UploadFile
from openai import OpenAI
from pydantic import BaseModel

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
DEFAULT_PROMPT_CHOICE = """
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
        """

# Initialize the router and OpenAI client
router = APIRouter(prefix="/api/v1/choice", tags=["choice"])
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else OpenAI()


class ExtractionResponse(BaseModel):
    choices: list[str]
    reasoning: str
    selected_choice: str


@router.post("/extract-choices", response_model=ExtractionResponse)
async def extract_choices(
    file: UploadFile = File(...),
    prompt: str = DEFAULT_PROMPT_CHOICE,
    model: str = "gpt-5.4",
):
    """
    Analyzes a roguelike game screenshot to extract choice titles
    and identify the currently selected one.
    """
    # Validate the uploaded file is an image
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    try:
        image_bytes = await file.read()
    except Exception:
        raise HTTPException(
            status_code=400, detail="Could not read the uploaded image."
        )

    b64_encoded = base64.b64encode(image_bytes).decode("ascii")
    data_url = f"data:{file.content_type};base64,{b64_encoded}"

    schema = {
        "name": "choice_extraction",
        "schema": {
            "type": "object",
            "properties": {
                "choices": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Choice titles, in on-screen order, or empty if not a choice screen.",
                },
                "reasoning": {
                    "type": "string",
                    "description": "Describe each card's background shade and any hover indicators (corner triangles, side arrows, cursor), then state which card is the visual outlier.",
                },
                "selected_choice": {
                    "type": "string",
                    "description": "The currently selected choice title, or empty string if none.",
                },
            },
            "required": ["choices", "reasoning", "selected_choice"],
            "additionalProperties": False,
        },
    }

    try:
        resp = client.chat.completions.create(
            model=model,
            temperature=0,
            reasoning_effort="none",
            prompt_cache_key="gamelens-choice-extraction",
            prompt_cache_retention="24h",
            response_format={"type": "json_schema", "json_schema": schema},
            messages=[
                {
                    "role": "system",
                    "content": (prompt),
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Return all choice titles and the selected choice.",
                        },
                        {"type": "image_url", "image_url": {"url": data_url, "detail": "original"}},
                    ],
                },
            ],
        )

        content = resp.choices[0].message.content
        result = json.loads(content) if isinstance(content, str) else content
        return result

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to process image with OpenAI: {str(e)}"
        )
    finally:
        # Explicitly free large objects — base64 image data and OpenAI response
        # can be several MB each; with many sequential requests this accumulates fast
        del image_bytes, b64_encoded, data_url
        gc.collect()
