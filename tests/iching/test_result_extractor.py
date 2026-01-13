import json
from unittest.mock import Mock, patch

import pytest

from modules.iching.core.data_models import HaoInfo, IChingResult
from modules.iching.core.result_extractor import IChingResultExtractor

"""
Tests for I Ching result extractor.
"""


@pytest.fixture
def mock_gemini_analyzer():
    """Create a mock GeminiChartAnalyzer."""
    with patch("modules.iching.core.result_extractor.GeminiChartAnalyzer") as mock:
        analyzer_instance = Mock()
        mock.return_value = analyzer_instance
        yield analyzer_instance


class TestIChingResultExtractor:
    """Test IChingResultExtractor class."""

    def test_init(self, mock_gemini_analyzer):
        """Test IChingResultExtractor initialization."""
        extractor = IChingResultExtractor(api_key="test-key")
        assert extractor.analyzer is not None

    def test_extract_from_image_success(self, mock_gemini_analyzer, temp_image_path):
        """Test successful extraction from image."""
        # Mock response with valid JSON
        mock_response = json.dumps(
            {
                "nhat_than": "Sửu-Thổ",
                "nguyet_lenh": "Tý-Thủy",
                "que_trai": [
                    {"hao": 1, "luc_than": "Thê Tài", "can_chi": "Tý-Thủy", "is_dong": False, "phuc_than": None},
                    {"hao": 2, "luc_than": "Quan Quỷ", "can_chi": "Dần-Mộc", "is_dong": True, "phuc_than": "Giáp Tý"},
                    {"hao": 3, "luc_than": "Huynh Đệ", "can_chi": "Thìn-Thổ", "is_dong": False, "phuc_than": None},
                    {"hao": 4, "luc_than": "Phụ Mẫu", "can_chi": "Ngọ-Hỏa", "is_dong": False, "phuc_than": None},
                    {"hao": 5, "luc_than": "Tử Tôn", "can_chi": "Thân-Kim", "is_dong": False, "phuc_than": None},
                    {"hao": 6, "luc_than": "Huynh Đệ", "can_chi": "Tuất-Thổ", "is_dong": False, "phuc_than": None},
                ],
                "que_phai": [
                    {"hao": 1, "luc_than": "Thê Tài", "can_chi": "Tý-Thủy", "luc_thu": "Thanh Long", "is_dong": False},
                    {"hao": 2, "luc_than": "Quan Quỷ", "can_chi": "Dần-Mộc", "luc_thu": "Chu Tước", "is_dong": False},
                    {"hao": 3, "luc_than": "Huynh Đệ", "can_chi": "Thìn-Thổ", "luc_thu": "Câu Trần", "is_dong": False},
                    {"hao": 4, "luc_than": "Phụ Mẫu", "can_chi": "Ngọ-Hỏa", "luc_thu": "Đằng Xà", "is_dong": False},
                    {"hao": 5, "luc_than": "Tử Tôn", "can_chi": "Thân-Kim", "luc_thu": "Bạch Hổ", "is_dong": False},
                    {"hao": 6, "luc_than": "Huynh Đệ", "can_chi": "Tuất-Thổ", "luc_thu": "Huyền Vũ", "is_dong": False},
                ],
                "the_vi_tri": 4,
                "ung_vi_tri": 1,
                "tk_vi_tri": [6],
            }
        )

        mock_gemini_analyzer.analyze_chart.return_value = mock_response

        extractor = IChingResultExtractor()
        result = extractor.extract_from_image(temp_image_path)

        assert result is not None
        assert isinstance(result, IChingResult)
        assert result.nhat_than == "Sửu-Thổ"
        assert result.nguyet_lenh == "Tý-Thủy"
        assert len(result.que_trai) == 6
        assert len(result.que_phai) == 6
        # Check that hao_dong is synced
        assert result.que_phai[1].is_dong is True  # Should be synced from que_trai

    def test_extract_from_image_invalid_json(self, mock_gemini_analyzer, temp_image_path):
        """Test extraction with invalid JSON response."""
        mock_gemini_analyzer.analyze_chart.return_value = "Invalid JSON response"

        extractor = IChingResultExtractor()
        result = extractor.extract_from_image(temp_image_path)

        assert result is None

    def test_extract_json_from_response(self):
        """Test extracting JSON from response text."""
        extractor = IChingResultExtractor()

        # Test with markdown code block
        response = '```json\n{"test": "value"}\n```'
        json_text = extractor._extract_json_from_response(response)
        assert json_text == '{"test": "value"}'

        # Test with plain JSON
        response = '{"test": "value"}'
        json_text = extractor._extract_json_from_response(response)
        assert json_text == '{"test": "value"}'

        # Test with no JSON
        response = "No JSON here"
        json_text = extractor._extract_json_from_response(response)
        assert json_text is None

    def test_sync_hao_dong(self, sample_hao_info_list):
        """Test syncing hao_dong between left and right hexagrams."""
        # Create fresh HaoInfo lists to ensure no is_dong is set
        que_trai = [
            HaoInfo(hao=h.hao, luc_than=h.luc_than, can_chi=h.can_chi, is_dong=False) for h in sample_hao_info_list
        ]
        que_phai = [
            HaoInfo(hao=h.hao, luc_than=h.luc_than, can_chi=h.can_chi, is_dong=False) for h in sample_hao_info_list
        ]

        # Create HaoInfo with is_dong=True for que_trai[1] and que_trai[4]
        que_trai_with_dong = [
            HaoInfo(hao=h.hao, luc_than=h.luc_than, can_chi=h.can_chi, is_dong=True) if i in (1, 4) else h
            for i, h in enumerate(que_trai)
        ]

        result = IChingResult(
            nhat_than="Sửu-Thổ",
            nguyet_lenh="Tý-Thủy",
            que_trai=que_trai_with_dong,
            que_phai=que_phai,
            the_vi_tri=4,
            ung_vi_tri=1,
        )

        extractor = IChingResultExtractor()

        # Initially que_phai should not have hao_dong
        assert result.que_phai[1].is_dong is False
        assert result.que_phai[4].is_dong is False

        # Force synchronization to verify _sync_hao_dong correctly updates hao_dong state
        extractor._sync_hao_dong(result)

        # Now que_phai should have matching hao_dong
        assert result.que_phai[1].is_dong is True
        assert result.que_phai[4].is_dong is True
        # Others should be False
        assert result.que_phai[0].is_dong is False
        assert result.que_phai[2].is_dong is False

    def test_extract_from_image_empty_que_trai(self, mock_gemini_analyzer, temp_image_path):
        """Test extraction when que_trai is empty."""
        mock_response = json.dumps(
            {
                "nhat_than": "Sửu-Thổ",
                "nguyet_lenh": "Tý-Thủy",
                "que_trai": [],
                "que_phai": [
                    {"hao": 1, "luc_than": "Thê Tài", "can_chi": "Tý-Thủy", "luc_thu": "Thanh Long", "is_dong": False},
                    {"hao": 2, "luc_than": "Quan Quỷ", "can_chi": "Dần-Mộc", "luc_thu": "Chu Tước", "is_dong": False},
                    {"hao": 3, "luc_than": "Huynh Đệ", "can_chi": "Thìn-Thổ", "luc_thu": "Câu Trần", "is_dong": False},
                    {"hao": 4, "luc_than": "Phụ Mẫu", "can_chi": "Ngọ-Hỏa", "luc_thu": "Đằng Xà", "is_dong": False},
                    {"hao": 5, "luc_than": "Tử Tôn", "can_chi": "Thân-Kim", "luc_thu": "Bạch Hổ", "is_dong": False},
                    {"hao": 6, "luc_than": "Huynh Đệ", "can_chi": "Tuất-Thổ", "luc_thu": "Huyền Vũ", "is_dong": False},
                ],
                "the_vi_tri": 4,
                "ung_vi_tri": 1,
                "tk_vi_tri": [6],
            }
        )

        mock_gemini_analyzer.analyze_chart.return_value = mock_response

        extractor = IChingResultExtractor()
        result = extractor.extract_from_image(temp_image_path)

        # Should return a valid IChingResult with empty que_trai
        assert result is not None
        assert isinstance(result, IChingResult)
        assert len(result.que_trai) == 0
        assert len(result.que_phai) == 6

    def test_extract_from_image_empty_que_phai(self, mock_gemini_analyzer, temp_image_path):
        """Test extraction when que_phai is empty."""
        mock_response = json.dumps(
            {
                "nhat_than": "Sửu-Thổ",
                "nguyet_lenh": "Tý-Thủy",
                "que_trai": [
                    {"hao": 1, "luc_than": "Thê Tài", "can_chi": "Tý-Thủy", "is_dong": False, "phuc_than": None},
                    {"hao": 2, "luc_than": "Quan Quỷ", "can_chi": "Dần-Mộc", "is_dong": True, "phuc_than": "Giáp Tý"},
                    {"hao": 3, "luc_than": "Huynh Đệ", "can_chi": "Thìn-Thổ", "is_dong": False, "phuc_than": None},
                    {"hao": 4, "luc_than": "Phụ Mẫu", "can_chi": "Ngọ-Hỏa", "is_dong": False, "phuc_than": None},
                    {"hao": 5, "luc_than": "Tử Tôn", "can_chi": "Thân-Kim", "is_dong": False, "phuc_than": None},
                    {"hao": 6, "luc_than": "Huynh Đệ", "can_chi": "Tuất-Thổ", "is_dong": False, "phuc_than": None},
                ],
                "que_phai": [],
                "the_vi_tri": 4,
                "ung_vi_tri": 1,
                "tk_vi_tri": [6],
            }
        )

        mock_gemini_analyzer.analyze_chart.return_value = mock_response

        extractor = IChingResultExtractor()
        result = extractor.extract_from_image(temp_image_path)

        # Should return a valid IChingResult with empty que_phai
        assert result is not None
        assert isinstance(result, IChingResult)
        assert len(result.que_trai) == 6
        assert len(result.que_phai) == 0

    def test_extract_from_image_both_empty_que(self, mock_gemini_analyzer, temp_image_path):
        """Test extraction when both que_trai and que_phai are empty."""
        mock_response = json.dumps(
            {
                "nhat_than": "Sửu-Thổ",
                "nguyet_lenh": "Tý-Thủy",
                "que_trai": [],
                "que_phai": [],
                "the_vi_tri": 4,
                "ung_vi_tri": 1,
                "tk_vi_tri": [6],
            }
        )

        mock_gemini_analyzer.analyze_chart.return_value = mock_response

        extractor = IChingResultExtractor()
        result = extractor.extract_from_image(temp_image_path)

        # Should return a valid IChingResult with both lists empty
        assert result is not None
        assert isinstance(result, IChingResult)
        assert len(result.que_trai) == 0
        assert len(result.que_phai) == 0

    def test_extract_from_image_mismatched_lengths(self, mock_gemini_analyzer, temp_image_path):
        """Test extraction with mismatched list lengths."""
        mock_response = json.dumps(
            {
                "nhat_than": "Sửu-Thổ",
                "nguyet_lenh": "Tý-Thủy",
                "que_trai": [
                    {"hao": 1, "luc_than": "Thê Tài", "can_chi": "Tý-Thủy", "is_dong": False, "phuc_than": None},
                    {"hao": 2, "luc_than": "Quan Quỷ", "can_chi": "Dần-Mộc", "is_dong": True, "phuc_than": "Giáp Tý"},
                    {"hao": 3, "luc_than": "Huynh Đệ", "can_chi": "Thìn-Thổ", "is_dong": False, "phuc_than": None},
                    {"hao": 4, "luc_than": "Phụ Mẫu", "can_chi": "Ngọ-Hỏa", "is_dong": False, "phuc_than": None},
                ],
                "que_phai": [
                    {"hao": 1, "luc_than": "Thê Tài", "can_chi": "Tý-Thủy", "luc_thu": "Thanh Long", "is_dong": False},
                    {"hao": 2, "luc_than": "Quan Quỷ", "can_chi": "Dần-Mộc", "luc_thu": "Chu Tước", "is_dong": False},
                    {"hao": 3, "luc_than": "Huynh Đệ", "can_chi": "Thìn-Thổ", "luc_thu": "Câu Trần", "is_dong": False},
                    {"hao": 4, "luc_than": "Phụ Mẫu", "can_chi": "Ngọ-Hỏa", "luc_thu": "Đằng Xà", "is_dong": False},
                    {"hao": 5, "luc_than": "Tử Tôn", "can_chi": "Thân-Kim", "luc_thu": "Bạch Hổ", "is_dong": False},
                    {"hao": 6, "luc_than": "Huynh Đệ", "can_chi": "Tuất-Thổ", "luc_thu": "Huyền Vũ", "is_dong": False},
                ],
                "the_vi_tri": 4,
                "ung_vi_tri": 1,
                "tk_vi_tri": [6],
            }
        )

        mock_gemini_analyzer.analyze_chart.return_value = mock_response

        extractor = IChingResultExtractor()
        result = extractor.extract_from_image(temp_image_path)

        # Should handle mismatched lengths gracefully (returns result with different lengths)
        assert result is not None
        assert isinstance(result, IChingResult)
        assert len(result.que_trai) == 4
        assert len(result.que_phai) == 6

    def test_extract_from_image_nonexistent_file(self, mock_gemini_analyzer):
        """Test extraction with non-existent image path."""
        # Mock analyze_chart to raise FileNotFoundError
        mock_gemini_analyzer.analyze_chart.side_effect = FileNotFoundError("File not found")

        extractor = IChingResultExtractor()
        result = extractor.extract_from_image("nonexistent_image.png")

        # Should return None when file doesn't exist
        assert result is None

    def test_extract_from_image_missing_nhat_than(self, mock_gemini_analyzer, temp_image_path):
        """Test extraction with missing nhat_than field."""
        mock_response = json.dumps(
            {
                "nguyet_lenh": "Tý-Thủy",
                "que_trai": [
                    {"hao": 1, "luc_than": "Thê Tài", "can_chi": "Tý-Thủy", "is_dong": False, "phuc_than": None},
                    {"hao": 2, "luc_than": "Quan Quỷ", "can_chi": "Dần-Mộc", "is_dong": True, "phuc_than": "Giáp Tý"},
                    {"hao": 3, "luc_than": "Huynh Đệ", "can_chi": "Thìn-Thổ", "is_dong": False, "phuc_than": None},
                    {"hao": 4, "luc_than": "Phụ Mẫu", "can_chi": "Ngọ-Hỏa", "is_dong": False, "phuc_than": None},
                    {"hao": 5, "luc_than": "Tử Tôn", "can_chi": "Thân-Kim", "is_dong": False, "phuc_than": None},
                    {"hao": 6, "luc_than": "Huynh Đệ", "can_chi": "Tuất-Thổ", "is_dong": False, "phuc_than": None},
                ],
                "que_phai": [
                    {"hao": 1, "luc_than": "Thê Tài", "can_chi": "Tý-Thủy", "luc_thu": "Thanh Long", "is_dong": False},
                    {"hao": 2, "luc_than": "Quan Quỷ", "can_chi": "Dần-Mộc", "luc_thu": "Chu Tước", "is_dong": False},
                    {"hao": 3, "luc_than": "Huynh Đệ", "can_chi": "Thìn-Thổ", "luc_thu": "Câu Trần", "is_dong": False},
                    {"hao": 4, "luc_than": "Phụ Mẫu", "can_chi": "Ngọ-Hỏa", "luc_thu": "Đằng Xà", "is_dong": False},
                    {"hao": 5, "luc_than": "Tử Tôn", "can_chi": "Thân-Kim", "luc_thu": "Bạch Hổ", "is_dong": False},
                    {"hao": 6, "luc_than": "Huynh Đệ", "can_chi": "Tuất-Thổ", "luc_thu": "Huyền Vũ", "is_dong": False},
                ],
                "the_vi_tri": 4,
                "ung_vi_tri": 1,
                "tk_vi_tri": [6],
            }
        )

        mock_gemini_analyzer.analyze_chart.return_value = mock_response

        extractor = IChingResultExtractor()
        result = extractor.extract_from_image(temp_image_path)

        # Should return None when required field is missing
        assert result is None

    def test_extract_from_image_missing_nguyet_lenh(self, mock_gemini_analyzer, temp_image_path):
        """Test extraction with missing nguyet_lenh field."""
        mock_response = json.dumps(
            {
                "nhat_than": "Sửu-Thổ",
                "que_trai": [
                    {"hao": 1, "luc_than": "Thê Tài", "can_chi": "Tý-Thủy", "is_dong": False, "phuc_than": None},
                    {"hao": 2, "luc_than": "Quan Quỷ", "can_chi": "Dần-Mộc", "is_dong": True, "phuc_than": "Giáp Tý"},
                    {"hao": 3, "luc_than": "Huynh Đệ", "can_chi": "Thìn-Thổ", "is_dong": False, "phuc_than": None},
                    {"hao": 4, "luc_than": "Phụ Mẫu", "can_chi": "Ngọ-Hỏa", "is_dong": False, "phuc_than": None},
                    {"hao": 5, "luc_than": "Tử Tôn", "can_chi": "Thân-Kim", "is_dong": False, "phuc_than": None},
                    {"hao": 6, "luc_than": "Huynh Đệ", "can_chi": "Tuất-Thổ", "is_dong": False, "phuc_than": None},
                ],
                "que_phai": [
                    {"hao": 1, "luc_than": "Thê Tài", "can_chi": "Tý-Thủy", "luc_thu": "Thanh Long", "is_dong": False},
                    {"hao": 2, "luc_than": "Quan Quỷ", "can_chi": "Dần-Mộc", "luc_thu": "Chu Tước", "is_dong": False},
                    {"hao": 3, "luc_than": "Huynh Đệ", "can_chi": "Thìn-Thổ", "luc_thu": "Câu Trần", "is_dong": False},
                    {"hao": 4, "luc_than": "Phụ Mẫu", "can_chi": "Ngọ-Hỏa", "luc_thu": "Đằng Xà", "is_dong": False},
                    {"hao": 5, "luc_than": "Tử Tôn", "can_chi": "Thân-Kim", "luc_thu": "Bạch Hổ", "is_dong": False},
                    {"hao": 6, "luc_than": "Huynh Đệ", "can_chi": "Tuất-Thổ", "luc_thu": "Huyền Vũ", "is_dong": False},
                ],
                "the_vi_tri": 4,
                "ung_vi_tri": 1,
                "tk_vi_tri": [6],
            }
        )

        mock_gemini_analyzer.analyze_chart.return_value = mock_response

        extractor = IChingResultExtractor()
        result = extractor.extract_from_image(temp_image_path)

        # Should return None when required field is missing
        assert result is None

    def test_extract_from_image_missing_que_trai(self, mock_gemini_analyzer, temp_image_path):
        """Test extraction with missing que_trai field."""
        mock_response = json.dumps(
            {
                "nhat_than": "Sửu-Thổ",
                "nguyet_lenh": "Tý-Thủy",
                "que_phai": [
                    {"hao": 1, "luc_than": "Thê Tài", "can_chi": "Tý-Thủy", "luc_thu": "Thanh Long", "is_dong": False},
                    {"hao": 2, "luc_than": "Quan Quỷ", "can_chi": "Dần-Mộc", "luc_thu": "Chu Tước", "is_dong": False},
                    {"hao": 3, "luc_than": "Huynh Đệ", "can_chi": "Thìn-Thổ", "luc_thu": "Câu Trần", "is_dong": False},
                    {"hao": 4, "luc_than": "Phụ Mẫu", "can_chi": "Ngọ-Hỏa", "luc_thu": "Đằng Xà", "is_dong": False},
                    {"hao": 5, "luc_than": "Tử Tôn", "can_chi": "Thân-Kim", "luc_thu": "Bạch Hổ", "is_dong": False},
                    {"hao": 6, "luc_than": "Huynh Đệ", "can_chi": "Tuất-Thổ", "luc_thu": "Huyền Vũ", "is_dong": False},
                ],
                "the_vi_tri": 4,
                "ung_vi_tri": 1,
                "tk_vi_tri": [6],
            }
        )

        mock_gemini_analyzer.analyze_chart.return_value = mock_response

        extractor = IChingResultExtractor()
        result = extractor.extract_from_image(temp_image_path)

        # Should return a valid result with empty que_trai (defaults to empty list)
        assert result is not None
        assert isinstance(result, IChingResult)
        assert len(result.que_trai) == 0
        assert len(result.que_phai) == 6

    def test_extract_from_image_missing_que_phai(self, mock_gemini_analyzer, temp_image_path):
        """Test extraction with missing que_phai field."""
        mock_response = json.dumps(
            {
                "nhat_than": "Sửu-Thổ",
                "nguyet_lenh": "Tý-Thủy",
                "que_trai": [
                    {"hao": 1, "luc_than": "Thê Tài", "can_chi": "Tý-Thủy", "is_dong": False, "phuc_than": None},
                    {"hao": 2, "luc_than": "Quan Quỷ", "can_chi": "Dần-Mộc", "is_dong": True, "phuc_than": "Giáp Tý"},
                    {"hao": 3, "luc_than": "Huynh Đệ", "can_chi": "Thìn-Thổ", "is_dong": False, "phuc_than": None},
                    {"hao": 4, "luc_than": "Phụ Mẫu", "can_chi": "Ngọ-Hỏa", "is_dong": False, "phuc_than": None},
                    {"hao": 5, "luc_than": "Tử Tôn", "can_chi": "Thân-Kim", "is_dong": False, "phuc_than": None},
                    {"hao": 6, "luc_than": "Huynh Đệ", "can_chi": "Tuất-Thổ", "is_dong": False, "phuc_than": None},
                ],
                "the_vi_tri": 4,
                "ung_vi_tri": 1,
                "tk_vi_tri": [6],
            }
        )

        mock_gemini_analyzer.analyze_chart.return_value = mock_response

        extractor = IChingResultExtractor()
        result = extractor.extract_from_image(temp_image_path)

        # Should return a valid result with empty que_phai (defaults to empty list)
        assert result is not None
        assert isinstance(result, IChingResult)
        assert len(result.que_trai) == 6
        assert len(result.que_phai) == 0

    def test_extract_from_image_empty_nhat_than(self, mock_gemini_analyzer, temp_image_path):
        """Test extraction with empty nhat_than field."""
        mock_response = json.dumps(
            {
                "nhat_than": "",
                "nguyet_lenh": "Tý-Thủy",
                "que_trai": [
                    {"hao": 1, "luc_than": "Thê Tài", "can_chi": "Tý-Thủy", "is_dong": False, "phuc_than": None},
                    {"hao": 2, "luc_than": "Quan Quỷ", "can_chi": "Dần-Mộc", "is_dong": True, "phuc_than": "Giáp Tý"},
                    {"hao": 3, "luc_than": "Huynh Đệ", "can_chi": "Thìn-Thổ", "is_dong": False, "phuc_than": None},
                    {"hao": 4, "luc_than": "Phụ Mẫu", "can_chi": "Ngọ-Hỏa", "is_dong": False, "phuc_than": None},
                    {"hao": 5, "luc_than": "Tử Tôn", "can_chi": "Thân-Kim", "is_dong": False, "phuc_than": None},
                    {"hao": 6, "luc_than": "Huynh Đệ", "can_chi": "Tuất-Thổ", "is_dong": False, "phuc_than": None},
                ],
                "que_phai": [
                    {"hao": 1, "luc_than": "Thê Tài", "can_chi": "Tý-Thủy", "luc_thu": "Thanh Long", "is_dong": False},
                    {"hao": 2, "luc_than": "Quan Quỷ", "can_chi": "Dần-Mộc", "luc_thu": "Chu Tước", "is_dong": False},
                    {"hao": 3, "luc_than": "Huynh Đệ", "can_chi": "Thìn-Thổ", "luc_thu": "Câu Trần", "is_dong": False},
                    {"hao": 4, "luc_than": "Phụ Mẫu", "can_chi": "Ngọ-Hỏa", "luc_thu": "Đằng Xà", "is_dong": False},
                    {"hao": 5, "luc_than": "Tử Tôn", "can_chi": "Thân-Kim", "luc_thu": "Bạch Hổ", "is_dong": False},
                    {"hao": 6, "luc_than": "Huynh Đệ", "can_chi": "Tuất-Thổ", "luc_thu": "Huyền Vũ", "is_dong": False},
                ],
                "the_vi_tri": 4,
                "ung_vi_tri": 1,
                "tk_vi_tri": [6],
            }
        )

        mock_gemini_analyzer.analyze_chart.return_value = mock_response

        extractor = IChingResultExtractor()
        result = extractor.extract_from_image(temp_image_path)

        # Should return None when required field is empty
        assert result is None

    def test_extract_from_image_empty_nguyet_lenh(self, mock_gemini_analyzer, temp_image_path):
        """Test extraction with empty nguyet_lenh field."""
        mock_response = json.dumps(
            {
                "nhat_than": "Sửu-Thổ",
                "nguyet_lenh": "",
                "que_trai": [
                    {"hao": 1, "luc_than": "Thê Tài", "can_chi": "Tý-Thủy", "is_dong": False, "phuc_than": None},
                    {"hao": 2, "luc_than": "Quan Quỷ", "can_chi": "Dần-Mộc", "is_dong": True, "phuc_than": "Giáp Tý"},
                    {"hao": 3, "luc_than": "Huynh Đệ", "can_chi": "Thìn-Thổ", "is_dong": False, "phuc_than": None},
                    {"hao": 4, "luc_than": "Phụ Mẫu", "can_chi": "Ngọ-Hỏa", "is_dong": False, "phuc_than": None},
                    {"hao": 5, "luc_than": "Tử Tôn", "can_chi": "Thân-Kim", "is_dong": False, "phuc_than": None},
                    {"hao": 6, "luc_than": "Huynh Đệ", "can_chi": "Tuất-Thổ", "is_dong": False, "phuc_than": None},
                ],
                "que_phai": [
                    {"hao": 1, "luc_than": "Thê Tài", "can_chi": "Tý-Thủy", "luc_thu": "Thanh Long", "is_dong": False},
                    {"hao": 2, "luc_than": "Quan Quỷ", "can_chi": "Dần-Mộc", "luc_thu": "Chu Tước", "is_dong": False},
                    {"hao": 3, "luc_than": "Huynh Đệ", "can_chi": "Thìn-Thổ", "luc_thu": "Câu Trần", "is_dong": False},
                    {"hao": 4, "luc_than": "Phụ Mẫu", "can_chi": "Ngọ-Hỏa", "luc_thu": "Đằng Xà", "is_dong": False},
                    {"hao": 5, "luc_than": "Tử Tôn", "can_chi": "Thân-Kim", "luc_thu": "Bạch Hổ", "is_dong": False},
                    {"hao": 6, "luc_than": "Huynh Đệ", "can_chi": "Tuất-Thổ", "luc_thu": "Huyền Vũ", "is_dong": False},
                ],
                "the_vi_tri": 4,
                "ung_vi_tri": 1,
                "tk_vi_tri": [6],
            }
        )

        mock_gemini_analyzer.analyze_chart.return_value = mock_response

        extractor = IChingResultExtractor()
        result = extractor.extract_from_image(temp_image_path)

        # Should return None when required field is empty
        assert result is None
