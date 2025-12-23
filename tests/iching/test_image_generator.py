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
    
    @patch('modules.iching.core.image_generator.get_font')
    def test_create_hexagram_image_success(self, mock_get_font, tmp_path, monkeypatch):
        """Test successful image creation."""
        # Setup mock font
        mock_font = Mock()
        mock_get_font.return_value = mock_font
        
        # Monkeypatch IMAGES_DIR to use tmp_path
        images_dir = tmp_path / "images"
        monkeypatch.setattr('modules.iching.core.image_generator.IMAGES_DIR', images_dir)
        
        # Test with valid grouped_string
        grouped_string = ["NNS", "NSS", "NNS", "SNN", "SSN", "NSS"]
        result_path = create_hexagram_image(grouped_string, filename="test_hexagram.png")
        
        # Verify image was created
        expected_path = images_dir / "test_hexagram.png"
        assert Path(result_path) == expected_path
        assert expected_path.exists()
        assert expected_path.stat().st_size > 0
    
    @patch('modules.iching.core.image_generator.get_font')
    def test_create_hexagram_image_all_patterns(self, mock_get_font, tmp_path, monkeypatch):
        """Test creating image with all different patterns."""
        mock_font = Mock()
        mock_get_font.return_value = mock_font
        
        # Monkeypatch IMAGES_DIR to use tmp_path
        images_dir = tmp_path / "images"
        monkeypatch.setattr('modules.iching.core.image_generator.IMAGES_DIR', images_dir)
        
        # Test all patterns: solid, broken, red solid, red broken
        grouped_string = ["NNS", "NSS", "SSS", "NNN", "SNN", "SNS"]
        result_path = create_hexagram_image(grouped_string, filename="all_patterns.png")
        
        expected_path = images_dir / "all_patterns.png"
        assert Path(result_path) == expected_path
        assert expected_path.exists()
        assert expected_path.stat().st_size > 0
    
    @patch('modules.iching.core.image_generator.get_font')
    def test_create_hexagram_image_invalid_pattern(self, mock_get_font, tmp_path, monkeypatch):
        """Test creating image with invalid pattern (should use gray)."""
        mock_font = Mock()
        mock_get_font.return_value = mock_font
        
        # Monkeypatch IMAGES_DIR to use tmp_path
        images_dir = tmp_path / "images"
        monkeypatch.setattr('modules.iching.core.image_generator.IMAGES_DIR', images_dir)
        
        # Test with invalid pattern (should still create image but with gray)
        grouped_string = ["XXX", "YYY", "ZZZ", "AAA", "BBB", "CCC"]
        result_path = create_hexagram_image(grouped_string, filename="invalid_pattern.png")
        
        # Should still create the image
        expected_path = images_dir / "invalid_pattern.png"
        assert Path(result_path) == expected_path
        assert expected_path.exists()
        assert expected_path.stat().st_size > 0
    
    @patch('modules.iching.core.image_generator.get_font')
    def test_create_hexagram_image_default_filename(self, mock_get_font, tmp_path, monkeypatch):
        """Test creating image with default filename."""
        mock_font = Mock()
        mock_get_font.return_value = mock_font
        
        # Monkeypatch IMAGES_DIR to use tmp_path
        images_dir = tmp_path / "images"
        monkeypatch.setattr('modules.iching.core.image_generator.IMAGES_DIR', images_dir)
        
        grouped_string = ["NNS", "NSS", "NNS", "SNN", "SSN", "NSS"]
        result_path = create_hexagram_image(grouped_string)  # No filename specified
        
        # Should use default filename "hexagram.png"
        expected_path = images_dir / "hexagram.png"
        assert Path(result_path) == expected_path
        assert expected_path.exists()
        assert expected_path.stat().st_size > 0

