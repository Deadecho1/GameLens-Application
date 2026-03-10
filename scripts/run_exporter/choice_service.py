from typing import Any

class ChoiceExtractorProtocol:
    def process(self, frame: bytes) -> Any:
        raise NotImplementedError


class ChoiceExtractionService:
    def __init__(self, choice_extractor: ChoiceExtractorProtocol):
        self.choice_extractor = choice_extractor

    def extract_choice(self, frame_bytes: bytes) -> dict:
        result = self.choice_extractor.process(frame_bytes)

        if isinstance(result, dict):
            options = result.get("options")
            selected_choice = result.get("selected_choice")
        else:
            options = getattr(result, "options", None)
            selected_choice = getattr(result, "selected_choice", None)

        return {
            "options": options,
            "selected_choice": selected_choice,
        }