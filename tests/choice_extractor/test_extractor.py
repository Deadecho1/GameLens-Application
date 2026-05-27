"""Unit tests for ChoiceExtractor using a mocked OpenAI client."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from scripts.choice_extractor.extractor import ChoiceExtractor, ChoiceExtractorConfig
from scripts.choice_extractor.models import ExtractionResult


def _make_extractor(api_key: str = "test-key") -> ChoiceExtractor:
    cfg = ChoiceExtractorConfig(api_key=api_key, timeout_seconds=5.0)
    return ChoiceExtractor(config=cfg)


def _mock_openai_response(choices: list, selected_choice: str | None) -> MagicMock:
    content = json.dumps({"choices": choices, "selected_choice": selected_choice})
    message = MagicMock()
    message.content = content
    choice = MagicMock()
    choice.message = message
    resp = MagicMock()
    resp.choices = [choice]
    return resp


class TestChoiceExtractorConfig:
    def test_default_values(self):
        cfg = ChoiceExtractorConfig()
        assert cfg.api_key == ""
        assert cfg.timeout_seconds == 30.0

    def test_custom_api_key(self):
        cfg = ChoiceExtractorConfig(api_key="sk-abc")
        assert cfg.api_key == "sk-abc"


class TestChoiceExtractorInit:
    def test_uses_configured_model(self):
        cfg = ChoiceExtractorConfig(api_key="k", model="gpt-4o")
        extractor = ChoiceExtractor(config=cfg)
        assert extractor._model == "gpt-4o"

    def test_default_model_set(self):
        extractor = _make_extractor()
        assert extractor._model  # non-empty


class TestExtractFrame:
    def test_returns_extraction_result(self):
        extractor = _make_extractor()
        mock_resp = _mock_openai_response(["Item A", "Item B"], "Item A")
        with patch.object(extractor._client.chat.completions, "create", return_value=mock_resp):
            result = extractor.extract_frame(b"fake_image")

        assert isinstance(result, ExtractionResult)
        assert result.choices == ["Item A", "Item B"]
        assert result.selected_choice == "Item A"

    def test_empty_choices_returned(self):
        extractor = _make_extractor()
        mock_resp = _mock_openai_response([], None)
        with patch.object(extractor._client.chat.completions, "create", return_value=mock_resp):
            result = extractor.extract_frame(b"img")

        assert result.choices == []
        assert result.selected_choice is None

    def test_custom_model_forwarded(self):
        extractor = _make_extractor()
        mock_resp = _mock_openai_response([], None)
        with patch.object(
            extractor._client.chat.completions, "create", return_value=mock_resp
        ) as mock_create:
            extractor.extract_frame(b"img", model="gpt-4o")

        call_kwargs = mock_create.call_args[1]
        assert call_kwargs["model"] == "gpt-4o"

    def test_custom_prompt_forwarded(self):
        extractor = _make_extractor()
        mock_resp = _mock_openai_response([], None)
        with patch.object(
            extractor._client.chat.completions, "create", return_value=mock_resp
        ) as mock_create:
            extractor.extract_frame(b"img", prompt="custom prompt")

        messages = mock_create.call_args[1]["messages"]
        system_msg = next(m for m in messages if m["role"] == "system")
        assert system_msg["content"] == "custom prompt"

    def test_openai_exception_re_raised(self):
        extractor = _make_extractor()
        with patch.object(
            extractor._client.chat.completions,
            "create",
            side_effect=RuntimeError("network error"),
        ):
            with pytest.raises(RuntimeError, match="network error"):
                extractor.extract_frame(b"img")
