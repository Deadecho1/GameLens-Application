from __future__ import annotations

from typing import Optional, Protocol

from .models import ExtractionResult


class ChoiceClient(Protocol):
    """Sends an image to the classifier backend and returns structured results."""

    def extract_frame(
        self,
        image_bytes: bytes,
        prompt: Optional[str] = None,
        model: Optional[str] = None,
    ) -> ExtractionResult:
        ...
