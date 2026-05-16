import base64
import gc
import json
import os

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

        Respond with a JSON object in exactly this format:
        {
          "choices": ["title1", "title2", "title3"],
          "reasoning": "describe each card and its indicators here",
          "selected_choice": "title1"
        }
        """

router = APIRouter(prefix="/api/v1/choice", tags=["choice"])
client = OpenAI(api_key=OPENAI_API_KEY)


class ExtractionResponse(BaseModel):
    choices: list[str]
    reasoning: str
    selected_choice: str


@router.post("/extract-choices", response_model=ExtractionResponse)
async def extract_choices(
    file: UploadFile = File(...),
    prompt: str = DEFAULT_PROMPT_CHOICE,
    model: str = "gpt-5.5",
):
    """
    Analyzes a roguelike game screenshot to extract choice titles
    and identify the currently selected one.
    """
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

    try:
        resp = client.chat.completions.create(
            model=model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": prompt,
                },
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
        result = json.loads(content) if isinstance(content, str) else content
        return result

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to process image with OpenAI: {str(e)}"
        )
    finally:
        del image_bytes, b64_encoded, data_url
        gc.collect()
