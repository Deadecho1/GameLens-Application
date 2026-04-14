from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import requests

from app_core.logging import get_logger

from .models import BossNameResult

logger = get_logger(__name__)

_MAX_RETRIES = 3
_RETRY_DELAY = 2.0


@dataclass(frozen=True)
class BossExtractorConfig:
    base_url: str = "http://localhost:7761"
    timeout_seconds: float = 15.0
    endpoint_path: str = "/api/v1/boss/extract-name"


class BossNameExtractor:
    def __init__(
        self,
        base_url: str = "http://localhost:7761",
        config: BossExtractorConfig | None = None,
    ) -> None:
        cfg = config or BossExtractorConfig(base_url=base_url)
        self._endpoint = cfg.base_url.rstrip("/") + cfg.endpoint_path
        self._timeout = cfg.timeout_seconds
        self._session = requests.Session()

    def reset_session(self) -> None:
        self._session.close()
        self._session = requests.Session()

    def extract_name(
        self,
        image_bytes: bytes,
        model: Optional[str] = None,
    ) -> BossNameResult:
        """Send a boss fight frame to the FastAPI backend and return the extracted boss name."""
        params = {}
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
                return BossNameResult(boss_names=data.get("boss_names", []))
            except requests.exceptions.Timeout as e:
                last_exc = e
                logger.warning(
                    "Boss name API timed out (attempt %d/%d), retrying in %.0fs...",
                    attempt, _MAX_RETRIES, _RETRY_DELAY,
                )
                self.reset_session()
                time.sleep(_RETRY_DELAY)
            except requests.exceptions.RequestException as e:
                logger.error("Failed to communicate with the boss name API: %s", e)
                raise

        logger.error("Boss name API timed out after %d attempts", _MAX_RETRIES)
        raise last_exc
