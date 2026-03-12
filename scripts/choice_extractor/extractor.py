import base64
import imghdr
import json
from app_core.settings import OPENAI_API_KEY

from openai import OpenAI


class ChoiceExtractor:
    def __init__(self, model: str = "gpt-5.1"):
        self.model = model
        if OPENAI_API_KEY:
            self.client = OpenAI(api_key=OPENAI_API_KEY)
        else:
            self.client = OpenAI()

    def _data_url_from_bytes(self, image_bytes: bytes) -> str:
        img_type = imghdr.what(None, h=image_bytes)
        if img_type is None:
            img_type = "png"
        mime = f"image/{img_type}"
        b64 = base64.b64encode(image_bytes).decode("ascii")
        return f"data:{mime};base64,{b64}"

    def extract_frame(self, image_bytes: bytes) -> dict:
        schema = {
            "name": "choice_extraction",
            "schema": {
                "type": "object",
                "properties": {
                    "choices": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Choice titles, in on-screen order.",
                    },
                    "selected_choice": {
                        "type": "string",
                        "description": "The currently selected choice title, or empty string if none.",
                    },
                },
                "required": ["choices", "selected_choice"],
                "additionalProperties": False,
            },
        }

        data_url = self._data_url_from_bytes(image_bytes)

        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            response_format={"type": "json_schema", "json_schema": schema},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are analyzing a roguelike game screenshot."
                        "Extract the item/upgrade titles shown and identify the currently selected one."
                    ),
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
        return json.loads(content) if isinstance(content, str) else content
