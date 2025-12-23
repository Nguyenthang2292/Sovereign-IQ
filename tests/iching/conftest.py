"""
Shared fixtures for I Ching tests.
"""

import pytest
from pathlib import Path

from modules.iching.core.data_models import HaoInfo, IChingResult


@pytest.fixture
def sample_hao_info_list():
    """Create a sample list of HaoInfo for testing."""
    return [
        HaoInfo(hao=1, luc_than="Thê Tài", can_chi="Tý-Thủy"),
        HaoInfo(hao=2, luc_than="Quan Quỷ", can_chi="Dần-Mộc"),
        HaoInfo(hao=3, luc_than="Huynh Đệ", can_chi="Thìn-Thổ"),
        HaoInfo(hao=4, luc_than="Phụ Mẫu", can_chi="Ngọ-Hỏa"),
        HaoInfo(hao=5, luc_than="Tử Tôn", can_chi="Thân-Kim"),
        HaoInfo(hao=6, luc_than="Huynh Đệ", can_chi="Tuất-Thổ"),
    ]


@pytest.fixture
def sample_iching_result(sample_hao_info_list):
    """Create a sample IChingResult for testing."""
    return IChingResult(
        nhat_than="Sửu-Thổ",
        nguyet_lenh="Tý-Thủy",
        que_trai=sample_hao_info_list.copy(),
        que_phai=sample_hao_info_list.copy(),
        the_vi_tri=4,
        ung_vi_tri=1,
        tk_vi_tri=[6]
    )


@pytest.fixture
def temp_image_path(tmp_path):
    """Create a temporary image file for testing."""
    image_path = tmp_path / "test_image.png"
    # Create a minimal valid PNG file
    png_header = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xdb\x00\x00\x00\x00IEND\xaeB`\x82'
    image_path.write_bytes(png_header)
    return str(image_path)

