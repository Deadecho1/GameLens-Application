from typing import Any

from scripts.choice_extractor.extractor import ChoiceExtractor


class ChoiceExtractionService:
    def __init__(self, choice_extractor: ChoiceExtractor):
        self.choice_extractor = choice_extractor

    # Returns a dict with keys "options" and "selected_choice"
    def extract_choice(self, frame_bytes: bytes) -> dict:
        result = self.choice_extractor.extract_frame(frame_bytes)

        if isinstance(result, dict):
            options = result.get("choices")
            selected_choice = result.get("selected_choice")
        else:
            options = getattr(result, "choices", None)
            selected_choice = getattr(result, "selected_choice", None)

        return {
            "options": options,
            "selected_choice": selected_choice,
        }