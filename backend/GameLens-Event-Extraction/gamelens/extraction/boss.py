import base64
import gc
import json
import os

from fastapi import APIRouter, File, HTTPException, UploadFile
from openai import OpenAI
from pydantic import BaseModel

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

DEFAULT_PROMPT_BOSS = """
You are analyzing a roguelike game screenshot during a boss fight.
Identify the names of all bosses or elite enemies the player is currently fighting.

There may be multiple simultaneous bosses on screen at once (e.g. two enemies with the same name,
or two different named enemies). List every distinct boss name that appears in the UI.

Look for:
- Health bars or name plates at the top or bottom of the screen with boss names
- Text overlays that label enemies
- Any UI element that identifies enemies by name

Return all boss names exactly as they appear in the game's UI.
If no boss name is identifiable, return an empty list.
If this is not a boss fight screen, return an empty list.

Respond with a JSON object in exactly this format:
{
  "boss_names": ["Boss Name"],
  "reasoning": "describe what UI elements you found the boss names in"
}
"""

router = APIRouter(prefix="/api/v1/boss", tags=["boss"])
client = OpenAI(api_key=OPENAI_API_KEY)


class BossNameResponse(BaseModel):
    boss_names: list[str]
    reasoning: str


@router.post("/extract-name", response_model=BossNameResponse)
async def extract_boss_name(
    file: UploadFile = File(...),
    prompt: str = DEFAULT_PROMPT_BOSS,
    model: str = "gpt-5.5",
):
    """
    Analyzes a roguelike game screenshot to extract the names of all bosses being fought.
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    try:
        image_bytes = await file.read()
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read the uploaded image.")

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
                            "text": "Return all boss names visible in this screenshot.",
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
