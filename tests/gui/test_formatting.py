"""Unit tests for shared formatting utilities."""
from __future__ import annotations

import pytest

from app_core.formatting import format_seconds


class TestFormatSeconds:
    def test_zero(self):
        assert format_seconds(0) == "0s"

    def test_seconds_only(self):
        assert format_seconds(45) == "45s"

    def test_minutes_and_seconds(self):
        assert format_seconds(90) == "1m 30s"

    def test_hours_minutes_seconds(self):
        assert format_seconds(3661) == "1h 1m 1s"

    def test_exactly_one_minute(self):
        assert format_seconds(60) == "1m 0s"

    def test_exactly_one_hour(self):
        assert format_seconds(3600) == "1h 0m 0s"

    def test_float_truncated(self):
        # 90.9 → 90 seconds total
        assert format_seconds(90.9) == "1m 30s"
