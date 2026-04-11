from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import requests

from app_core.logging import get_logger
from .models import ExtractionResult

logger = get_logger(__name__)

_MAX_RETRIES = 3
_RETRY_DELAY = 2.0  # seconds between retries


@dataclass(frozen=True)
class ChoiceExtractorConfig:
    base_url: str = "http://localhost:7761"
    timeout_seconds: float = 15.0
    endpoint_path: str = "/api/v1/choice/extract-choices"


class ChoiceExtractor:
    def __init__(
        self,
        base_url: str = "http://localhost:7761",
        config: ChoiceExtractorConfig | None = None,
    ) -> None:
        cfg = config or ChoiceExtractorConfig(base_url=base_url)
        self._endpoint = cfg.base_url.rstrip("/") + cfg.endpoint_path
        self._timeout = cfg.timeout_seconds
        self._session = requests.Session()

    def reset_session(self) -> None:
        """Close and recreate the HTTP session to release connection pool resources."""
        self._session.close()
        self._session = requests.Session()

    def extract_frame(
        self,
        image_bytes: bytes,
        prompt: Optional[str] = None,
        model: Optional[str] = None,
    ) -> ExtractionResult:
        """Send image bytes to the FastAPI backend and return extracted choices.

        Retries up to _MAX_RETRIES times on timeout, resetting the session
        between attempts to avoid holding stale connections.
        """
        params = {}
        if prompt:
            params["prompt"] = prompt
        if model:
            params["model"] = model

        last_exc: Exception | None = None
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                files = {"file": ("screenshot.png", image_bytes, "image/png")}
                response = self._session.post(
                    self._endpoint, files=files, params=params, timeout=self._timeout
                )
                response.raise_for_status()
                data = response.json()
                return ExtractionResult(
                    choices=data.get("choices", []),
                    selected_choice=data.get("selected_choice"),
                )
            except requests.exceptions.Timeout as e:
                last_exc = e
                logger.warning(
                    "Extraction API timed out (attempt %d/%d), retrying in %.0fs...",
                    attempt, _MAX_RETRIES, _RETRY_DELAY,
                )
                self.reset_session()
                time.sleep(_RETRY_DELAY)
            except requests.exceptions.RequestException as e:
                logger.error("Failed to communicate with the extraction API: %s", e)
                raise

        logger.error("Extraction API timed out after %d attempts", _MAX_RETRIES)
        raise last_exc
