"""Unit tests for Vibe Extractor."""

import pytest

from app.ingestion.analyzers.vibe_extractor import VibeExtractor


class TestVibeExtractor:
    def setup_method(self):
        self.extractor = VibeExtractor()

    def test_default_result(self):
        result = self.extractor._default_result()
        assert "dimensions" in result
        assert result["dimensions"]["formality"] == 50
        assert result["target_audience"] == "General"

    def test_validate_result_clamps_values(self):
        raw = {
            "dimensions": {"formality": 150, "craftsmanship": -10},
            "tags": ["a"] * 20,
            "signature_items": [],
            "target_audience": "Expert",
            "summary": "test",
        }
        result = self.extractor._validate_result(raw)
        assert result["dimensions"]["formality"] == 100
        assert result["dimensions"]["craftsmanship"] == 0
        assert len(result["tags"]) <= 15
