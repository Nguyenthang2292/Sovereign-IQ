"""
Tests for I Ching image generator.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch

from modules.iching.core.image_generator import create_hexagram_image


class TestCreateHexagramImage:
    """Test create_hexagram_image function."""
    
    def test_create_hexagram_image_empty_list(self):
        """Test creating image with empty grouped_string."""
        with pytest.raises(ValueError, match="grouped_string không được rỗng"):
            create_hexagram_image([])
    
    def test_create_hexagram_image_insufficient_groups(self):
        """Test creating image with insufficient groups."""
        with pytest.raises(ValueError, match="Cần ít nhất 6 nhóm"):
            create_hexagram_image(["NNS", "NSS", "NNS"])  # Only 3 groups
    
    @patch('modules.iching.core.image_generator.ImageDraw.ImageDraw.text')
    @patch('modules.iching.core.image_generator.get_font')
    def test_create_hexagram_image_success(self, mock_get_font, mock_text, tmp_path, monkeypatch):
        """Test successful image creation."""
        # Setup mock font (not used since we patch draw.text)
        mock_font = Mock()
        mock_get_font.return_value = mock_font
        
        # Monkeypatch IMAGES_DIR to use tmp_path
        images_dir = tmp_path / "images"
        # Ensure images_dir is absolute for comparison
        images_dir = images_dir.resolve()
        monkeypatch.setattr('modules.iching.core.image_generator.IMAGES_DIR', images_dir)
        
        # Test with valid grouped_string
        grouped_string = ["NNS", "NSS", "NNS", "SNN", "SSN", "NSS"]
        result_path = create_hexagram_image(grouped_string, filename="test_hexagram.png")
        
        # Verify image was created
        # create_hexagram_image returns str, not tuple
        expected_path = images_dir / "test_hexagram.png"
        # Normalize paths for comparison (handle Windows path separators)
        result_path_obj = Path(result_path).resolve()
        expected_path_obj = expected_path.resolve()
        assert result_path_obj == expected_path_obj, f"Path mismatch: {result_path_obj} != {expected_path_obj}"
        assert expected_path.exists()
        assert expected_path.stat().st_size > 0
    
    @patch('modules.iching.core.image_generator.ImageDraw.ImageDraw.text')
    @patch('modules.iching.core.image_generator.get_font')
    def test_create_hexagram_image_all_patterns(self, mock_get_font, mock_text, tmp_path, monkeypatch):
        """Test creating image with all different patterns."""
        # Setup mock font (not used since we patch draw.text)
        mock_font = Mock()
        mock_get_font.return_value = mock_font
        
        # Monkeypatch IMAGES_DIR to use tmp_path
        images_dir = tmp_path / "images"
        # Ensure images_dir is absolute for comparison
        images_dir = images_dir.resolve()
        monkeypatch.setattr('modules.iching.core.image_generator.IMAGES_DIR', images_dir)
        
        # Test all patterns: solid, broken, red solid, red broken
        grouped_string = ["NNS", "NSS", "SSS", "NNN", "SNN", "SNS"]
        result_path = create_hexagram_image(grouped_string, filename="all_patterns.png")
        
        expected_path = images_dir / "all_patterns.png"
        # Normalize paths for comparison
        result_path_obj = Path(result_path).resolve()
        expected_path_obj = expected_path.resolve()
        assert result_path_obj == expected_path_obj, f"Path mismatch: {result_path_obj} != {expected_path_obj}"
        assert expected_path.exists()
        assert expected_path.stat().st_size > 0
    
    @patch('modules.iching.core.image_generator.get_font')
    def test_create_hexagram_image_invalid_pattern(self, mock_get_font, tmp_path, monkeypatch):
        """Test creating image with invalid pattern (should raise ValueError)."""
        # Setup mock font (though this test will raise ValueError before using font)
        mock_font = Mock()
        mock_get_font.return_value = mock_font
        
        # Monkeypatch IMAGES_DIR to use tmp_path
        images_dir = tmp_path / "images"
        monkeypatch.setattr('modules.iching.core.image_generator.IMAGES_DIR', images_dir)
        
        # Test with invalid pattern - should raise ValueError due to validation
        grouped_string = ["XXX", "YYY", "ZZZ", "AAA", "BBB", "CCC"]
        with pytest.raises(ValueError, match="chứa ký tự không hợp lệ"):
            create_hexagram_image(grouped_string, filename="invalid_pattern.png")
    
    @patch('modules.iching.core.image_generator.ImageDraw.ImageDraw.text')
    @patch('modules.iching.core.image_generator.get_font')
    def test_create_hexagram_image_default_filename(self, mock_get_font, mock_text, tmp_path, monkeypatch):
        """Test creating image with default filename."""
        # Setup mock font (not used since we patch draw.text)
        mock_font = Mock()
        mock_get_font.return_value = mock_font
        
        # Monkeypatch IMAGES_DIR to use tmp_path
        images_dir = tmp_path / "images"
        # Ensure images_dir is absolute for comparison
        images_dir = images_dir.resolve()
        monkeypatch.setattr('modules.iching.core.image_generator.IMAGES_DIR', images_dir)
        
        grouped_string = ["NNS", "NSS", "NNS", "SNN", "SSN", "NSS"]
        result_path = create_hexagram_image(grouped_string)  # No filename specified
        
        # Should use default filename "hexagram.png"
        expected_path = images_dir / "hexagram.png"
        # Normalize paths for comparison
        result_path_obj = Path(result_path).resolve()
        expected_path_obj = expected_path.resolve()
        assert result_path_obj == expected_path_obj, f"Path mismatch: {result_path_obj} != {expected_path_obj}"
        assert expected_path.exists()
        assert expected_path.stat().st_size > 0

