"""Unit tests for ChoiceExtractor using a fake HTTP session."""
from __future__ import annotations

from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

from scripts.choice_extractor.extractor import ChoiceExtractor, ChoiceExtractorConfig
from scripts.choice_extractor.models import ExtractionResult


def _make_extractor(base_url: str = "http://localhost:7761") -> ChoiceExtractor:
    cfg = ChoiceExtractorConfig(base_url=base_url, timeout_seconds=5.0)
    return ChoiceExtractor(config=cfg)


class TestChoiceExtractorConfig:
    def test_default_values(self):
        cfg = ChoiceExtractorConfig()
        assert cfg.base_url == "http://localhost:7761"
        assert cfg.timeout_seconds == 10.0
        assert cfg.endpoint_path == "/api/v1/choice/extract-choices"

    def test_custom_base_url(self):
        cfg = ChoiceExtractorConfig(base_url="http://myserver:8080")
        assert cfg.base_url == "http://myserver:8080"


class TestChoiceExtractorInit:
    def test_endpoint_constructed_correctly(self):
        cfg = ChoiceExtractorConfig(base_url="http://localhost:7761")
        extractor = ChoiceExtractor(config=cfg)
        assert extractor._endpoint == "http://localhost:7761/api/v1/choice/extract-choices"

    def test_trailing_slash_stripped(self):
        cfg = ChoiceExtractorConfig(base_url="http://localhost:7761/")
        extractor = ChoiceExtractor(config=cfg)
        assert not extractor._endpoint.startswith("http://localhost:7761//")


class TestExtractFrame:
    def _mock_response(self, choices, selected_choice):
        mock = MagicMock()
        mock.raise_for_status.return_value = None
        mock.json.return_value = {"choices": choices, "selected_choice": selected_choice}
        return mock

    def test_returns_extraction_result(self):
        extractor = _make_extractor()
        mock_resp = self._mock_response(["Item A", "Item B"], "Item A")
        with patch("scripts.choice_extractor.extractor.requests.post", return_value=mock_resp):
            result = extractor.extract_frame(b"fake_image")

        assert isinstance(result, ExtractionResult)
        assert result.choices == ["Item A", "Item B"]
        assert result.selected_choice == "Item A"

    def test_passes_timeout_to_requests(self):
        cfg = ChoiceExtractorConfig(timeout_seconds=7.5)
        extractor = ChoiceExtractor(config=cfg)
        mock_resp = self._mock_response([], None)
        with patch("scripts.choice_extractor.extractor.requests.post", return_value=mock_resp) as mock_post:
            extractor.extract_frame(b"img")

        _, kwargs = mock_post.call_args
        assert kwargs.get("timeout") == pytest.approx(7.5)

    def test_optional_prompt_and_model_passed_as_params(self):
        extractor = _make_extractor()
        mock_resp = self._mock_response([], None)
        with patch("scripts.choice_extractor.extractor.requests.post", return_value=mock_resp) as mock_post:
            extractor.extract_frame(b"img", prompt="custom", model="gpt-4")

        _, kwargs = mock_post.call_args
        assert kwargs["params"]["prompt"] == "custom"
        assert kwargs["params"]["model"] == "gpt-4"

    def test_no_prompt_means_no_params(self):
        extractor = _make_extractor()
        mock_resp = self._mock_response([], None)
        with patch("scripts.choice_extractor.extractor.requests.post", return_value=mock_resp) as mock_post:
            extractor.extract_frame(b"img")

        _, kwargs = mock_post.call_args
        assert kwargs.get("params") == {}

    def test_request_exception_re_raised(self):
        import requests
        extractor = _make_extractor()
        with patch(
            "scripts.choice_extractor.extractor.requests.post",
            side_effect=requests.exceptions.ConnectionError("refused"),
        ):
            with pytest.raises(requests.exceptions.ConnectionError):
                extractor.extract_frame(b"img")
