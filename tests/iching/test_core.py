"""
Tests for core hexagram generation logic.
"""

import random

import pytest

from modules.iching.core.hexagram import (
    analyze_line,
    generate_ns_string,
    group_string,
    prepare_hexagram,
)


class TestGenerateNsString:
    """Test generate_ns_string function."""

    def test_generate_ns_string_default_length(self):
        """Test generating string with default length."""
        result = generate_ns_string()
        assert len(result) == 18  # HEXAGRAM_STRING_LENGTH
        assert all(c in ["N", "S"] for c in result)

    def test_generate_ns_string_custom_length(self):
        """Test generating string with custom length."""
        result = generate_ns_string(10)
        assert len(result) == 10
        assert all(c in ["N", "S"] for c in result)

    def test_generate_ns_string_randomness(self):
        """Test that generated strings are random (not all same)."""
        random.seed(42)
        strings = [generate_ns_string(20) for _ in range(5)]
        # With seeded random, we can verify strings are not all identical
        assert len(set(strings)) > 1


class TestGroupString:
    """Test group_string function."""

    def test_group_string_default_size(self):
        """Test grouping with default group size."""
        string = "NNSNSSNNSSNNSNSSNN"
        result = group_string(string)
        # String length 18, group size 3 = 6 groups
        assert len(result) == 6
        assert all(len(group) == 3 for group in result)
        # Verify first few groups
        assert result[0] == "NNS"
        assert result[1] == "NSS"
        assert result[2] == "NNS"

    def test_group_string_custom_size(self):
        """Test grouping with custom group size."""
        string = "NNSNSSNNSS"
        result = group_string(string, group_size=2)
        assert len(result) == 5
        assert all(len(group) == 2 for group in result)
        assert result == ["NN", "SN", "SS", "NN", "SS"]

    def test_group_string_empty(self):
        """Test grouping empty string."""
        result = group_string("")
        assert result == []


class TestAnalyzeLine:
    """Test analyze_line function."""

    def test_analyze_line_solid_patterns(self):
        """Test solid line patterns."""
        # Solid patterns: NNS, SNN, NSN
        assert analyze_line("NNS") == (True, False)
        assert analyze_line("SNN") == (True, False)
        assert analyze_line("NSN") == (True, False)

    def test_analyze_line_broken_patterns(self):
        """Test broken line patterns."""
        # Broken patterns: NSS, SSN, SNS
        assert analyze_line("NSS") == (False, False)
        assert analyze_line("SSN") == (False, False)
        assert analyze_line("SNS") == (False, False)

    def test_analyze_line_red_solid(self):
        """Test red solid line (SSS)."""
        assert analyze_line("SSS") == (True, True)

    def test_analyze_line_red_broken(self):
        """Test red broken line (NNN)."""
        assert analyze_line("NNN") == (False, True)


class TestPrepareHexagram:
    """Test prepare_hexagram function."""

    @pytest.fixture
    def mock_create_image(self, monkeypatch):
        """Mock create_hexagram_image to avoid file I/O in tests."""

        def mock_image(*args, **kwargs):
            return "mock_path.png"

        monkeypatch.setattr("modules.iching.core.hexagram.create_hexagram_image", mock_image)

    def test_prepare_hexagram_structure(self, mock_create_image):
        """Test that prepare_hexagram returns correct structure."""
        result = prepare_hexagram()

        assert len(result) == 6
        assert all(isinstance(item, dict) for item in result)
        assert all("is_solid" in item and "is_red" in item for item in result)
        assert all(isinstance(item["is_solid"], bool) for item in result)
        assert all(isinstance(item["is_red"], bool) for item in result)
